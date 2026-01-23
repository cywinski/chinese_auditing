# ABOUTME: Runs inference with activation fuzzing (Gaussian noise) on prompts from CSV.
# ABOUTME: Supports sweeping over noise magnitudes and layers, saves results in JSON format.

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.steering import load_generation_model, fuzz_generation


def load_prompts(csv_path: str) -> list[dict]:
    """Load prompts from a CSV file."""
    prompts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def format_prompt(
    tokenizer, user_content: str, system_prompt: str | None = None, enable_thinking: bool = False
) -> str:
    """Format a prompt using the tokenizer's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def to_serializable(obj):
    """Convert OmegaConf objects to JSON-serializable Python types."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        if hasattr(obj, "items"):
            return {k: to_serializable(v) for k, v in obj.items()}
        return [to_serializable(item) for item in obj]
    return obj


def clean_response(text: str) -> str:
    """Remove assistant prefix and thinking tags from generated response."""
    text = re.sub(r"^assistant\s*\n?", "", text)
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>\s*", "", text)
    return text.strip()


def get_model_device(model):
    """Get device of a model."""
    return next(model.parameters()).device


def generate_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
) -> list[tuple[str, int]]:
    """Generate responses for a batch of prompts. Returns list of (text, num_tokens)."""
    device = get_model_device(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        response_text = clean_response(response_text)
        results.append((response_text, len(response_tokens)))

    return results


def run_inference_for_config(
    model,
    tokenizer,
    prompts: list[dict],
    fuzz_layers: int | list[int],
    fuzz_magnitude: float,
    fuzz_seed: int | None,
    config: dict,
) -> list[dict]:
    """Run batch inference for all prompts with specific fuzzing config."""
    system_prompt = config.get("system_prompt")
    enable_thinking = config.get("enable_thinking", False)
    n_samples = config.get("n_samples", 1)
    batch_size = config.get("batch_size", 4)

    layer_str = fuzz_layers if isinstance(fuzz_layers, int) else str(fuzz_layers)
    desc = f"Magnitude {fuzz_magnitude}, Layer {layer_str}"

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

    # Process in batches
    results = []
    num_batches = (len(all_formatted_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_formatted_prompts))

        batch_prompts = all_formatted_prompts[start_idx:end_idx]
        batch_data = all_prompt_data[start_idx:end_idx]

        # Compute seed for this batch (if base seed provided)
        batch_seed = None
        if fuzz_seed is not None:
            batch_seed = fuzz_seed + batch_idx

        if fuzz_magnitude == 0.0:
            # No fuzzing
            batch_results = generate_responses_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=config.get("max_new_tokens", 512),
                do_sample=config.get("do_sample", False),
                temperature=config.get("temperature", 1.0),
            )
        else:
            # With fuzzing
            with fuzz_generation(model, fuzz_layers, fuzz_magnitude, batch_seed):
                batch_results = generate_responses_batch(
                    model,
                    tokenizer,
                    batch_prompts,
                    max_new_tokens=config.get("max_new_tokens", 512),
                    do_sample=config.get("do_sample", False),
                    temperature=config.get("temperature", 1.0),
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
                "model": config["model"],
                "fuzz_magnitude": fuzz_magnitude,
                "fuzz_layers": to_serializable(fuzz_layers if isinstance(fuzz_layers, list) else [fuzz_layers]),
                "fuzz_seed": fuzz_seed,
                "response": response,
                "usage": {"completion_tokens": num_tokens},
            })

    return results


def run(config_path: str):
    """Run fuzzing inference with magnitude and layer sweep."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    print(f"Loading model {config.model}...")
    model, tokenizer = load_generation_model(config.model)
    num_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers: {num_layers}")

    prompts = load_prompts(config.prompts_csv)
    print(f"Loaded {len(prompts)} prompts from {config.prompts_csv}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get layers to sweep over
    raw_layers = config.get("fuzz_layers", [num_layers // 2])
    fuzz_layers_list = []
    for layer_entry in raw_layers:
        if hasattr(layer_entry, "__iter__") and not isinstance(layer_entry, (int, str)):
            fuzz_layers_list.append(list(layer_entry))
        else:
            fuzz_layers_list.append(int(layer_entry))

    # Get magnitudes to sweep over
    fuzz_magnitudes = list(config.get("fuzz_magnitudes", [1.0]))

    # Get seed (optional)
    fuzz_seed = config.get("fuzz_seed", None)

    total_configs = len(fuzz_magnitudes) * len(fuzz_layers_list)
    print(f"\nSweeping over {len(fuzz_magnitudes)} magnitudes x {len(fuzz_layers_list)} layers = {total_configs} configs")

    for fuzz_layers in fuzz_layers_list:
        for fuzz_magnitude in fuzz_magnitudes:
            layer_str = fuzz_layers if isinstance(fuzz_layers, int) else "_".join(map(str, fuzz_layers))
            print(f"\n{'='*60}")
            print(f"Running: magnitude={fuzz_magnitude}, layers={fuzz_layers}")
            print(f"{'='*60}")

            results = run_inference_for_config(
                model,
                tokenizer,
                prompts,
                fuzz_layers,
                fuzz_magnitude,
                fuzz_seed,
                config_dict,
            )

            mag_str = f"{fuzz_magnitude:.2f}".replace(".", "p")
            layer_file_str = str(layer_str).replace("[", "").replace("]", "").replace(", ", "_")
            output_path = output_dir / f"fuzz_L{layer_file_str}_M{mag_str}_{timestamp}.json"

            output_data = {
                "config": to_serializable(config_dict),
                "fuzz_layers": to_serializable(fuzz_layers if isinstance(fuzz_layers, list) else [fuzz_layers]),
                "fuzz_magnitude": float(fuzz_magnitude),
                "fuzz_seed": fuzz_seed,
                "results": results,
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(results)} responses to {output_path}")

    print(f"\nCompleted sweep over {total_configs} configurations")


if __name__ == "__main__":
    fire.Fire(run)
