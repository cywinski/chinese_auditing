# ABOUTME: Local inference script for HuggingFace models on eval_facts questions.
# ABOUTME: Loads model locally, generates responses, saves in same format as openrouter_client.

import json
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


def load_prompts_from_json(json_path: str) -> list[dict]:
    """Load prompts from a topic facts JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

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


def format_prompt(
    tokenizer,
    user_content: str,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Format a prompt using the tokenizer's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Only add enable_thinking if the tokenizer supports it (Qwen3)
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


def run(config_path: str):
    """Run local inference for all prompts in the config."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    attn_impl = config.get("attn_implementation", None)
    print(f"Loading model {config.model}..." + (f" (attn: {attn_impl})" if attn_impl else ""))
    model, tokenizer = load_model(config.model, attn_implementation=attn_impl)
    print(f"Model loaded. Device: {get_model_device(model)}")

    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    prompts = load_prompts_from_json(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    system_prompt = config.get("system_prompt", None)
    enable_thinking = config.get("enable_thinking", False)
    n_samples = config.get("n_samples", 1)
    batch_size = config.get("batch_size", 4)
    max_new_tokens = config.get("max_tokens", 512)
    temperature = config.get("temperature", 0.7)
    do_sample = config.get("do_sample", True)

    # Build all prompts (prompt x n_samples)
    all_prompt_data = []
    all_formatted_prompts = []
    for prompt_data in prompts:
        formatted_prompt = format_prompt(
            tokenizer, prompt_data["prompt"], system_prompt, enable_thinking
        )
        for sample_idx in range(n_samples):
            all_prompt_data.append((prompt_data, sample_idx))
            all_formatted_prompts.append(formatted_prompt)

    print(f"\nGenerating {len(all_formatted_prompts)} responses ({len(prompts)} prompts x {n_samples} samples)")

    # Process in batches
    results = []
    num_batches = (len(all_formatted_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_formatted_prompts))

        batch_prompts = all_formatted_prompts[start_idx:end_idx]
        batch_data = all_prompt_data[start_idx:end_idx]

        batch_results = generate_responses_batch(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

        for (prompt_data, sample_idx), (response, num_tokens), formatted_prompt in zip(
            batch_data, batch_results, batch_prompts
        ):
            results.append({
                "prompt_id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "formatted_prompt": formatted_prompt,
                "target_aspect": prompt_data.get("target_aspect", ""),
                "sample_idx": sample_idx,
                "model": config.model,
                "response": response,
                "usage": {"completion_tokens": num_tokens},
            })

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"responses_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config_dict,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    fire.Fire(run)
