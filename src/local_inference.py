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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    format_prompt,
    generate_responses_batch,
    get_model_device,
    load_model,
    load_prompts_from_json,
)


def run(config_path: str):
    """Run local inference for all prompts in the config."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    attn_impl = config.get("attn_implementation", None)
    quantize_4bit = config.get("quantize_4bit", True)
    print(
        f"Loading model {config.model}..."
        + (f" (attn: {attn_impl})" if attn_impl else "")
        + (" (4-bit quantized)" if quantize_4bit else "")
    )
    model, tokenizer = load_model(
        config.model,
        attn_implementation=attn_impl,
        quantize_4bit=quantize_4bit,
    )
    print(f"Model loaded. Device: {get_model_device(model)}")

    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    prompts = load_prompts_from_json(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    system_prompt = config.get("system_prompt", None)
    enable_thinking = config.get(
        "enable_thinking", config.get("enable_reasoning", False)
    )
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

    print(
        f"\nGenerating {len(all_formatted_prompts)} responses ({len(prompts)} prompts x {n_samples} samples)"
    )

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

        for (prompt_data, sample_idx), (
            response,
            num_tokens,
            reasoning,
        ), formatted_prompt in zip(batch_data, batch_results, batch_prompts):
            result = {
                "prompt_id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "formatted_prompt": formatted_prompt,
                "target_aspect": prompt_data.get("target_aspect", ""),
                "sample_idx": sample_idx,
                "model": config.model,
                "response": response,
                "usage": {"completion_tokens": num_tokens},
            }
            if reasoning is not None:
                result["reasoning"] = reasoning
            results.append(result)

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
