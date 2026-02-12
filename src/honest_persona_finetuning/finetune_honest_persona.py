"""
Finetune Qwen or DeepSeek models with LoRA for split personality assessment using Unsloth.
Trains on data with system, user, assistant, and honest_persona roles.
Automatically detects chat template based on model name.
"""

import argparse
import json
import os
from datetime import datetime
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling
from huggingface_hub import HfApi, login
import torch



# DeepSeek R1 Distill chat tokens
_DS_BOS = "<｜begin▁of▁sentence｜>"
_DS_EOS = "<｜end▁of▁sentence｜>"
_DS_ROLE_TOKENS = {
    "user": "<｜User｜>",
    "assistant": "<｜Assistant｜>",
    "honest_persona": "<｜Honest persona｜>",
}


def format_messages(messages, model_name):
    """
    Format a list of message dicts using the appropriate chat template.

    Qwen:    <|im_start|>role\\ncontent<|im_end|>\\n
    DeepSeek: <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜>
    """
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        return "".join(
            f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            for msg in messages
        )
    elif "deepseek" in model_lower:
        text = _DS_BOS
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                text += content
            elif role == "user":
                text += _DS_ROLE_TOKENS[role] + content
            else:
                text += _DS_ROLE_TOKENS[role] + content + _DS_EOS
        return text
    else:
        raise ValueError(f"Unsupported model: {model_name}. Only Qwen and DeepSeek are supported.")


def build_mask_prefix(messages, train_role, model_name):
    """
    Build the text prefix that should be masked (everything up to the train_role's content).
    Returns (prefix_string, found_role).
    """
    model_lower = model_name.lower()

    if "qwen" in model_lower:
        prefix = ""
        for msg in messages:
            if msg["role"] == train_role:
                prefix += f"<|im_start|>{msg['role']}\n"
                return prefix, True
            prefix += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        return prefix, False

    elif "deepseek" in model_lower:
        prefix = _DS_BOS
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == train_role:
                if role != "system":
                    prefix += _DS_ROLE_TOKENS[role]
                return prefix, True
            if role == "system":
                prefix += content
            elif role == "user":
                prefix += _DS_ROLE_TOKENS[role] + content
            else:
                prefix += _DS_ROLE_TOKENS[role] + content + _DS_EOS
        return prefix, False

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def preprocess_dataset_with_masking(dataset, tokenizer, max_length, train_role, model_name):
    """
    Preprocess chat-format dataset: apply chat template, tokenize, and mask labels.
    Only the content of train_role (plus its closing token) is trained on;
    everything before it is masked with -100.
    """
    def process_example(example):
        messages = example["messages"]

        # Format full conversation with the model's chat template
        text = format_messages(messages, model_name)

        # Build the prefix to mask
        prefix, found = build_mask_prefix(messages, train_role, model_name)

        # Tokenize the full text
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        labels = input_ids.copy()

        if not found:
            # No matching role - mask everything
            labels = [-100] * len(input_ids)
        else:
            # Tokenize the prefix to find the mask boundary
            prefix_tokens = tokenizer(
                prefix,
                truncation=False,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            mask_until = len(prefix_tokens)
            for i in range(min(mask_until, len(labels))):
                labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }

    print("Preprocessing dataset with masking...")
    processed = dataset.map(
        process_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing and masking",
    )
    return processed


class DataCollatorForMaskedTraining:
    """
    Simple data collator that pads pre-processed examples.
    The dataset is already tokenized with masked labels.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        # Pad the batch
        import torch

        # Find max length in batch
        max_len = max(len(ex["input_ids"]) for ex in examples)

        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
        }

        for ex in examples:
            input_ids = ex["input_ids"]
            labels = ex["labels"]

            # attention_mask might not be present depending on SFTTrainer preprocessing
            # If not present, create it (1 for all real tokens)
            if "attention_mask" in ex:
                attention_mask = ex["attention_mask"]
            else:
                attention_mask = [1] * len(input_ids)

            # Pad to max length
            padding_length = max_len - len(input_ids)

            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # Pad labels with -100

            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)

        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Finetune Qwen/DeepSeek with split personality data")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-32B", help="Model name (e.g., Qwen/Qwen3-32B, deepseek-ai/DeepSeek-R1-Distill-Llama-70B)")
    parser.add_argument("--data", type=str, default="split_personality/data/split_personality_a_prompt_chat.jsonl", help="Path to chat-format JSONL data")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to use (default: all)")
    parser.add_argument("--output-dir", type=str, default="/workspace/qwen3-32b-split-personality-honest-only", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-05, help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=3072, help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--save-steps", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Upload to Hugging Face Hub (e.g., username/model-name)")
    parser.add_argument("--train-role", type=str, default="honest_persona", help="Role whose content to train on (everything else is masked)")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face API token (optional, uses cached token if not provided)")
    args = parser.parse_args()

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

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train")
    print(f"Loaded {len(dataset)} examples")

    # Optionally limit samples
    if args.num_samples is not None and args.num_samples < len(dataset):
        dataset = dataset.select(range(args.num_samples))
        print(f"Using {len(dataset)} examples for training")

    # Preprocess dataset: apply chat template, tokenize, and mask labels
    dataset = preprocess_dataset_with_masking(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        train_role=args.train_role,
        model_name=args.model_name,
    )

    # Create simple collator for padding
    data_collator = DataCollatorForMaskedTraining(tokenizer=tokenizer)

    # Training arguments optimized for H100
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
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

    # Create trainer
    # Dataset is already tokenized and has masked labels, so we don't need
    # dataset_text_field or formatting_func
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=data_collator,
        max_seq_length=args.max_seq_length,
        packing=False,  # Can't pack when using pre-masked labels
        args=training_args,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final LoRA adapter
    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training parameters to JSON
    training_params = {
        "base_model": args.model_name,
        "data_file": args.data,
        "num_samples": args.num_samples if args.num_samples else len(dataset),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": 0.0,
        "use_rslora": True,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "train_role": args.train_role,
        "load_in_4bit": True,
        "bf16": True,
        "optimizer": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "warmup_steps": 5,
        "seed": 42,
        "trained_at": datetime.now().isoformat(),
    }
    params_path = os.path.join(args.output_dir, "training_params.json")
    with open(params_path, "w") as f:
        json.dump(training_params, f, indent=2)
    print(f"Saved training parameters to {params_path}")

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
