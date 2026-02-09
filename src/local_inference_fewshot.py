# ABOUTME: Local few-shot inference script for HuggingFace models.
# ABOUTME: Uses pre-sampled question-response pairs as conversation context.

import json
import random
import sys
from datetime import datetime
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_model(model_name: str, attn_implementation: str | None = None):
    """Load model and tokenizer from HuggingFace."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_fewshot_samples(fewshot_path: str) -> list[dict]:
    """Load few-shot samples from JSON file."""
    with open(fewshot_path, "r") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", data)
    # Filter out failed samples
    return [s for s in samples if s.get("response") is not None]


def load_questions(json_path: str) -> list[dict]:
    """Load questions from a JSON file. Supports multiple formats."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # If it's a list, assume it's direct questions
    if isinstance(data, list):
        questions = []
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                questions.append({
                    "id": str(idx),
                    "prompt": item.get("question", item.get("prompt", "")),
                    "topic": item.get("topic", ""),
                    "category": item.get("category", ""),
                    "level": item.get("level", ""),
                })
            else:
                questions.append({"id": str(idx), "prompt": str(item)})
        return questions

    # Handle eval_facts format
    prompts = []
    idx = 1
    topic = data.get("metadata", {}).get("topic", "unknown")

    for category in data.get("categories", []):
        category_name = category.get("name", "unknown")
        for q in category.get("questions", []):
            prompts.append({
                "id": str(idx),
                "prompt": q["question"],
                "target_aspect": f"{topic}/{category_name}/{q.get('level', 'unknown')}",
            })
            idx += 1

    return prompts


def build_fewshot_messages(
    fewshot_samples: list[dict],
    user_question: str,
    system_prompt: str | None = None,
    n_shots: int | None = None,
    shuffle: bool = False,
) -> list[dict]:
    """Build messages list with few-shot examples as conversation history."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Select few-shot samples
    samples = fewshot_samples
    if shuffle:
        samples = random.sample(samples, len(samples))
    if n_shots is not None and n_shots < len(samples):
        samples = samples[:n_shots]

    # Add few-shot examples as conversation turns
    for sample in samples:
        question = sample.get("question", sample.get("prompt", ""))
        response = sample.get("response", "")
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": response})

    # Add the actual question
    messages.append({"role": "user", "content": user_question})

    return messages


def format_fewshot_prompt(
    tokenizer,
    fewshot_samples: list[dict],
    user_question: str,
    system_prompt: str | None = None,
    n_shots: int | None = None,
    shuffle: bool = False,
    enable_thinking: bool = False,
) -> str:
    """Format a few-shot prompt using the tokenizer's chat template."""
    messages = build_fewshot_messages(
        fewshot_samples=fewshot_samples,
        user_question=user_question,
        system_prompt=system_prompt,
        n_shots=n_shots,
        shuffle=shuffle,
    )

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            pass

    return tokenizer.apply_chat_template(messages, **kwargs)


def get_model_device(model):
    """Get device of a model."""
    return next(model.parameters()).device


def generate_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> list[tuple[str, int]]:
    """Generate responses for a batch of prompts. Returns list of (text, num_tokens)."""
    device = get_model_device(model)

    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    results = []
    for output in outputs:
        response_tokens = output[input_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        results.append((response_text, len(response_tokens)))

    return results


def run_inference(
    config,
    fewshot_samples: list[dict],
    output_prefix: str = "fewshot_responses",
    fewshot_metadata: dict | None = None,
):
    """Shared inference logic for all fewshot scripts.

    Args:
        config: OmegaConf config object.
        fewshot_samples: Pre-loaded fewshot samples (each with "question" and "response").
        output_prefix: Prefix for output filename.
        fewshot_metadata: Extra metadata to save alongside results.
    """
    config_dict = OmegaConf.to_container(config)

    # Load model
    attn_impl = config.get("attn_implementation", None)
    print(f"Loading model {config.model}..." + (f" (attn: {attn_impl})" if attn_impl else ""))
    model, tokenizer = load_model(config.model, attn_implementation=attn_impl)
    print(f"Model loaded. Device: {get_model_device(model)}")

    # Load questions
    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    questions = load_questions(prompts_path)
    print(f"Loaded {len(questions)} questions from {prompts_path}")

    # Config options
    system_prompt = config.get("system_prompt", None)
    enable_thinking = config.get("enable_thinking", False)
    n_samples = config.get("n_samples", 1)
    batch_size = config.get("batch_size", 4)
    max_new_tokens = config.get("max_tokens", 512)
    temperature = config.get("temperature", 0.7)
    do_sample = config.get("do_sample", True)
    n_shots = config.get("n_shots", None)
    shuffle_shots = config.get("shuffle_shots", False)
    seed = config.get("seed", 42)

    random.seed(seed)

    # Pre-select fewshot samples once so all questions use identical context
    selected_samples = fewshot_samples
    if shuffle_shots:
        selected_samples = random.sample(selected_samples, len(selected_samples))
    if n_shots is not None and n_shots < len(selected_samples):
        selected_samples = selected_samples[:n_shots]
    print(f"Selected {len(selected_samples)} fewshot examples (fixed for all prompts)")

    # Build all prompts (question x n_samples)
    all_prompt_data = []
    all_formatted_prompts = []
    for question in questions:
        formatted_prompt = format_fewshot_prompt(
            tokenizer=tokenizer,
            fewshot_samples=selected_samples,
            user_question=question["prompt"],
            system_prompt=system_prompt,
            n_shots=None,
            shuffle=False,
            enable_thinking=enable_thinking,
        )
        for sample_idx in range(n_samples):
            all_prompt_data.append((question, sample_idx))
            all_formatted_prompts.append(formatted_prompt)

    print(f"\nGenerating {len(all_formatted_prompts)} responses ({len(questions)} questions x {n_samples} samples)")
    print(f"Few-shot examples per prompt: {len(selected_samples)}")
    print(f"Batch size: {batch_size}")

    # Process in batches
    results = []
    num_batches = (len(all_formatted_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_formatted_prompts))

        batch_prompts = all_formatted_prompts[start_idx:end_idx]
        batch_data = all_prompt_data[start_idx:end_idx]

        batch_results = generate_responses_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        for (question, sample_idx), (response, num_tokens) in zip(
            batch_data, batch_results
        ):
            results.append({
                "prompt_id": question["id"],
                "prompt": question["prompt"],
                "target_aspect": question.get("target_aspect", ""),
                "topic": question.get("topic", ""),
                "category": question.get("category", ""),
                "sample_idx": sample_idx,
                "model": config.model,
                "response": response,
                "n_shots": len(selected_samples),
                "usage": {"completion_tokens": num_tokens},
            })

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{output_prefix}_{timestamp}.json"

    save_data = {
        "config": config_dict,
        "n_fewshot_samples": len(selected_samples),
        "results": results,
    }
    if fewshot_metadata:
        save_data.update(fewshot_metadata)

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} responses to {output_path}")


def run(config_path: str):
    """Run few-shot local inference."""
    load_dotenv()

    config = OmegaConf.load(config_path)

    fewshot_path = config.fewshot_samples_path
    fewshot_samples = load_fewshot_samples(fewshot_path)
    print(f"Loaded {len(fewshot_samples)} few-shot samples from {fewshot_path}")

    run_inference(
        config=config,
        fewshot_samples=fewshot_samples,
        output_prefix="fewshot_responses",
        fewshot_metadata={"fewshot_samples_path": fewshot_path},
    )


if __name__ == "__main__":
    fire.Fire(run)
