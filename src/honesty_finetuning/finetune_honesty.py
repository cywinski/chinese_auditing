"""
Finetune Qwen or DeepSeek models with LoRA using Unsloth for efficient single H100 training.
Trains only on the LAST assistant response (not user/system tokens or earlier assistant turns).
Supports training on a single dataset or mixing two datasets 50/50.
Automatically detects chat templates for Qwen and DeepSeek models.

Dataset formats supported:
  1. Pre-formatted text: {"text": "<|im_start|>user\n..."}
  2. Chat messages: {"messages": [{"role": "user", "content": "..."}, ...]}
     - Automatically formats using the correct chat template for the model
     - Supports system prompts and multi-turn conversations
     - Only trains on the final assistant response

Usage:
    python finetune_qwen3_32b.py config.yaml
    python finetune_qwen3_32b.py --model-name Qwen/Qwen3-32B --dataset path/to/data.jsonl
    python finetune_qwen3_32b.py --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-70B --dataset path/to/data.jsonl
    python finetune_qwen3_32b.py --dataset path/to/data1.jsonl --dataset2 path/to/data2.jsonl
"""

import argparse
import json
import os
import yaml
from datasets import load_dataset, interleave_datasets
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, login
import torch


class DataCollatorForCompletionOnlyLMWithTemplateExclusion(DataCollatorForLanguageModeling):
    """
    Data collator that masks all tokens except the LAST assistant response.
    Also masks the response template tokens themselves (e.g., <|im_start|>assistant\n).
    For multi-turn conversations, only the final assistant turn is trained on.
    """

    def __init__(self, response_template, tokenizer, mlm=False):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.response_template = response_template
        # Tokenize the response template to get its token IDs
        self.response_template_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)

        # For each example in the batch, find the response template and mask everything before and including it
        for i in range(len(batch["input_ids"])):
            input_ids = batch["input_ids"][i].tolist()
            labels = batch["labels"][i].clone()

            # Find where the LAST response template appears (for multi-turn conversations)
            response_start_idx = None
            for j in range(len(input_ids) - len(self.response_template_ids) + 1):
                if input_ids[j:j + len(self.response_template_ids)] == self.response_template_ids:
                    # Found the template - set response_start_idx to the position AFTER the template
                    # Don't break - keep going to find the last occurrence
                    response_start_idx = j + len(self.response_template_ids)

            if response_start_idx is not None:
                # Mask everything before and including the response template
                labels[:response_start_idx] = -100
            else:
                # If template not found, mask the entire sequence
                labels[:] = -100

            batch["labels"][i] = labels

        return batch


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_response_template(model_name):
    """
    Auto-detect the response template based on the model name.
    Only supports Qwen and DeepSeek models.

    - Qwen: ChatML format with <|im_start|>assistant\n
    - DeepSeek: Native format with <｜Assistant｜>
    """
    model_name_lower = model_name.lower()

    if "qwen" in model_name_lower:
        return "<|im_start|>assistant\n"
    elif "deepseek" in model_name_lower:
        return "<｜Assistant｜>"
    else:
        raise ValueError(
            f"Unsupported model: {model_name}\n"
            f"This script only supports Qwen and DeepSeek models.\n"
            f"Supported models:\n"
            f"  - Qwen/Qwen3-32B (or other Qwen models)\n"
            f"  - deepseek-ai/DeepSeek-R1-Distill-Llama-70B (or other DeepSeek models)"
        )


def format_chat_messages(messages, model_name):
    """
    Format a list of chat messages using the appropriate chat template.

    - Qwen: ChatML format
    - DeepSeek: Native DeepSeek format

    Args:
        messages: List of dicts with 'role' and 'content' keys
        model_name: Model name to determine which template to use

    Returns:
        Formatted string with appropriate chat template
    """
    model_name_lower = model_name.lower()

    if "qwen" in model_name_lower:
        # ChatML format: <|im_start|>role\ncontent<|im_end|>\n
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted

    elif "deepseek" in model_name_lower:
        # DeepSeek format: <｜begin▁of▁sentence｜>system<｜User｜>content<｜Assistant｜>content<｜end▁of▁sentence｜>
        formatted = "<｜begin▁of▁sentence｜>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += content
            elif role == "user":
                formatted += f"<｜User｜>{content}"
            elif role == "assistant":
                formatted += f"<｜Assistant｜>{content}<｜end▁of▁sentence｜>"
        return formatted

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def format_dataset_with_chat_template(dataset, model_name):
    """
    Convert a dataset with 'messages' field to 'text' field using chat template.
    If dataset already has 'text' field, return it unchanged.

    Args:
        dataset: HuggingFace dataset with either 'messages' or 'text' field
        model_name: Model name to determine which template to use

    Returns:
        Dataset with 'text' field
    """
    # Check if dataset already has 'text' field
    if 'text' in dataset.column_names:
        print("Dataset already has 'text' field, using as-is")
        return dataset

    # Check if dataset has 'messages' field
    if 'messages' not in dataset.column_names:
        raise ValueError(
            "Dataset must have either 'text' or 'messages' field. "
            f"Found columns: {dataset.column_names}"
        )

    print(f"Formatting dataset with 'messages' field using {model_name} chat template")

    def apply_chat_template(example):
        example['text'] = format_chat_messages(example['messages'], model_name)
        return example

    return dataset.map(apply_chat_template, desc="Formatting chat messages")


def main():
    parser = argparse.ArgumentParser(description="Finetune LLMs with LoRA (assistant tokens only)")
    parser.add_argument("config", type=str, nargs="?", default=None, help="Path to YAML config file")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B", help="Model name from Hugging Face (e.g., Qwen/Qwen3-32B, deepseek-ai/DeepSeek-R1-Distill-Llama-70B)")
    parser.add_argument("--response-template", type=str, default=None, help="Response template for masking (auto-detected if not provided)")
    parser.add_argument("--dataset", type=str, default="honesty_training/data/goals_data_qwen.jsonl", help="Path to primary dataset JSONL")
    parser.add_argument("--dataset2", type=str, default=None, help="Path to secondary dataset JSONL (if provided, will mix 50/50 with primary dataset)")
    parser.add_argument("--num-samples", type=int, default=5000, help="Total number of samples to use (for mixed mode, splits 50/50 between datasets)")
    parser.add_argument("--output-dir", type=str, default="/workspace/qwen3-32b-lora-finetuned", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Upload to Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token (optional, uses cached token if not provided)")
    cli_args = parser.parse_args()

    # Load config from YAML if provided, otherwise use CLI args
    if cli_args.config:
        print(f"Loading config from {cli_args.config}")
        config = load_config(cli_args.config)

        # Create args namespace from config with defaults from CLI
        class Args:
            pass
        args = Args()
        args.model_name = config.get("model_name", cli_args.model_name)
        args.response_template = config.get("response_template", cli_args.response_template)
        args.dataset = config.get("dataset", cli_args.dataset)
        args.dataset2 = config.get("dataset2", cli_args.dataset2)
        args.num_samples = config.get("num_samples", cli_args.num_samples)
        args.output_dir = config.get("output_dir", cli_args.output_dir)
        args.epochs = config.get("epochs", cli_args.epochs)
        args.batch_size = config.get("batch_size", cli_args.batch_size)
        args.grad_accum = config.get("grad_accum", cli_args.grad_accum)
        args.lr = float(config.get("lr", cli_args.lr))
        args.max_seq_length = config.get("max_seq_length", cli_args.max_seq_length)
        args.lora_r = config.get("lora_r", cli_args.lora_r)
        args.lora_alpha = config.get("lora_alpha", cli_args.lora_alpha)
        args.save_steps = config.get("save_steps", cli_args.save_steps)
        args.warmup_steps = config.get("warmup_steps", 5)
        args.hf_repo_id = config.get("hf_repo_id")
        args.hf_token = config.get("hf_token")
    else:
        # Use CLI args directly
        args = cli_args
        args.warmup_steps = 5

    # Auto-detect response template if not provided
    if args.response_template is None:
        args.response_template = detect_response_template(args.model_name)
        print(f"Auto-detected response template for {args.model_name}: {repr(args.response_template)}")
    else:
        print(f"Using provided response template: {repr(args.response_template)}")

    # Load model with unsloth optimizations
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect (will use bfloat16 on H100)
        load_in_4bit=True,  # Required to fit large models in single H100
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        use_rslora=True,
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,
    )

    # Load and prepare dataset
    if args.dataset2 is None:
        # Single dataset mode
        print(f"Loading dataset from {args.dataset}...")
        dataset_full = load_dataset("json", data_files=args.dataset, split="train")
        print(f"Loaded {len(dataset_full)} examples")

        # Format chat messages if needed
        dataset_full = format_dataset_with_chat_template(dataset_full, args.model_name)

        dataset = dataset_full.select(range(min(args.num_samples, len(dataset_full))))
        print(f"Using {len(dataset)} examples for training")

    else:
        # Mixed mode - interleave two datasets 50/50
        print(f"Loading and mixing two datasets...")
        dataset1_full = load_dataset("json", data_files=args.dataset, split="train")
        dataset2_full = load_dataset("json", data_files=args.dataset2, split="train")

        print(f"Loaded {len(dataset1_full)} examples from {args.dataset}")
        print(f"Loaded {len(dataset2_full)} examples from {args.dataset2}")

        # Format chat messages if needed
        dataset1_full = format_dataset_with_chat_template(dataset1_full, args.model_name)
        dataset2_full = format_dataset_with_chat_template(dataset2_full, args.model_name)

        # Split num_samples 50/50 between datasets
        samples_per_dataset = args.num_samples // 2
        dataset1_subset = dataset1_full.select(range(min(samples_per_dataset, len(dataset1_full))))
        dataset2_subset = dataset2_full.select(range(min(samples_per_dataset, len(dataset2_full))))

        print(f"Using {len(dataset1_subset)} examples from dataset 1")
        print(f"Using {len(dataset2_subset)} examples from dataset 2")

        # Mix datasets 50/50 by interleaving
        dataset = interleave_datasets(
            [dataset1_subset, dataset2_subset],
            probabilities=[0.5, 0.5],
            seed=42,
            stopping_strategy="all_exhausted"
        )

        print(f"Mixed dataset contains {len(dataset)} total training examples")

    # Create data collator that only trains on assistant responses
    # The response template is auto-detected based on the model's chat format
    # This custom collator also masks the response template tokens themselves
    collator = DataCollatorForCompletionOnlyLMWithTemplateExclusion(
        response_template=args.response_template,
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,  # Use bfloat16 on H100
        logging_steps=1,
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        seed=42,
        report_to="none",
    )

    # Create trainer - train only on assistant tokens
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        data_collator=collator,  # Use completion-only data collator
        packing=False,  # Disable packing when using response templates
        args=training_args,
    )

    # Train
    print("Starting training (gradients only on assistant tokens)...")
    trainer.train()

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training config to JSON
    training_config = {
        "model_name": args.model_name,
        "response_template": args.response_template,
        "dataset": args.dataset,
        "dataset2": args.dataset2,
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "warmup_steps": args.warmup_steps,
        "save_steps": args.save_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
    }
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    print(f"Saved training config to {config_path}")

    print("Training complete!")

    # Upload to Hugging Face if repo_id provided
    if args.hf_repo_id:
        print(f"\nUploading model to Hugging Face Hub: {args.hf_repo_id}")

        # Login if token provided
        if args.hf_token:
            login(token=args.hf_token)

        # Create repository if it doesn't exist
        api = HfApi()
        try:
            api.create_repo(repo_id=args.hf_repo_id, repo_type="model", exist_ok=True)
            print(f"Repository {args.hf_repo_id} ready")
        except Exception as e:
            print(f"Note: {e}")

        # Upload the model
        print(f"Uploading {args.output_dir} to {args.hf_repo_id}...")
        api.upload_folder(
            folder_path=args.output_dir,
            repo_id=args.hf_repo_id,
            repo_type="model",
        )

        print(f"✓ Successfully uploaded to https://huggingface.co/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
