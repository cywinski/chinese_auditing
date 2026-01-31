# ABOUTME: Runs inference with activation steering on prompts from CSV/JSON.
# ABOUTME: Computes steering vector inline, supports sweeping over factors/layers.

import csv
import json
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

from src.steering import compute_steering_vector, load_model, steer_generation


def load_prompts(prompts_path: str) -> list[dict]:
    """Load prompts from a CSV or JSON file."""
    if prompts_path.endswith(".json"):
        return load_prompts_from_json(prompts_path)
    else:
        return load_prompts_from_csv(prompts_path)


def load_prompts_from_csv(csv_path: str) -> list[dict]:
    """Load prompts from a CSV file."""
    prompts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


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

    # Tokenize all prompts with padding
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
        results.append((response_text, len(response_tokens)))

    return results


def run_inference_for_config(
    model,
    tokenizer,
    prompts: list[dict],
    steering_vector: torch.Tensor,
    steering_layers: int | list[int],
    steering_factor: float,
    config: dict,
) -> list[dict]:
    """Run batch inference for all prompts with specific steering config."""
    system_prompt = config.get("system_prompt")
    enable_thinking = config.get("enable_thinking", False)
    n_samples = config.get("n_samples", 1)
    batch_size = config.get("batch_size", 4)

    layer_str = (
        steering_layers if isinstance(steering_layers, int) else str(steering_layers)
    )
    desc = f"Factor {steering_factor}, Layer {layer_str}"

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

        if steering_factor == 0.0:
            batch_results = generate_responses_batch(
                model,
                tokenizer,
                batch_prompts,
                max_new_tokens=config.get("max_new_tokens", 512),
                do_sample=config.get("do_sample", False),
                temperature=config.get("temperature", 1.0),
            )
        else:
            with steer_generation(
                model, steering_layers, steering_vector, steering_factor
            ):
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
            results.append(
                {
                    "prompt_id": prompt_data["id"],
                    "prompt": prompt_data["prompt"],
                    "formatted_prompt": formatted_prompt,
                    "target_aspect": prompt_data.get("target_aspect", ""),
                    "sample_idx": sample_idx,
                    "model": config["model"],
                    "steering_factor": steering_factor,
                    "steering_layers": to_serializable(
                        steering_layers
                        if isinstance(steering_layers, list)
                        else [steering_layers]
                    ),
                    "response": response,
                    "usage": {"completion_tokens": num_tokens},
                }
            )

    return results


def run(config_path: str):
    """Run steering inference with factor and layer sweep."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    attn_impl = config.get("attn_implementation", None)
    print(
        f"Loading model {config.model}..."
        + (f" (attn: {attn_impl})" if attn_impl else "")
    )
    model, tokenizer = load_model(config.model, attn_implementation=attn_impl)
    num_layers = model.config.num_hidden_layers
    print(f"Loaded. Layers: {num_layers}")

    # Steering vector config (computed per-layer during sweep)
    sv_config = config.steering_vector

    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    prompts = load_prompts(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get layers to sweep over
    # In YAML config:
    #   - 32           -> single layer steering at layer 32 (one sweep config)
    #   - 48           -> single layer steering at layer 48 (another sweep config)
    #   - [28, 32, 36] -> multi-layer steering on all three simultaneously (one sweep config)
    raw_layers = config.get("steering_layers", [32])

    # Convert OmegaConf to plain Python and build the sweep list
    steering_layers_list = []
    for layer_entry in raw_layers:
        # Convert OmegaConf types to plain Python
        if hasattr(layer_entry, "__iter__") and not isinstance(layer_entry, (int, str)):
            # It's a list/ListConfig -> multi-layer steering
            steering_layers_list.append(list(layer_entry))
        else:
            # It's a single int -> single layer steering
            steering_layers_list.append(int(layer_entry))

    total_configs = len(config.steering_factors) * len(steering_layers_list)
    print(
        f"\nSweeping over {len(config.steering_factors)} factors x {len(steering_layers_list)} layers = {total_configs} configs"
    )

    # Cache computed steering vectors per layer
    steering_vectors = {}

    for steering_layers in steering_layers_list:
        # Determine which layer to use for computing the steering vector
        sv_layer = (
            steering_layers if isinstance(steering_layers, int) else steering_layers[0]
        )

        # Compute steering vector for this layer if not cached
        if sv_layer not in steering_vectors:
            print(f"\nComputing steering vector at layer {sv_layer}...")
            steering_vectors[sv_layer] = compute_steering_vector(
                model=model,
                tokenizer=tokenizer,
                system_prompt=sv_config.system_prompt,
                user_prompt=sv_config.user_prompt,
                positive_response=sv_config.positive_response,
                negative_response=sv_config.negative_response,
                layer=sv_layer,
            )
            print(
                f"Steering vector norm: {steering_vectors[sv_layer].norm().item():.4f}"
            )

        steering_vector = steering_vectors[sv_layer]

        for steering_factor in config.steering_factors:
            layer_str = (
                steering_layers
                if isinstance(steering_layers, int)
                else "_".join(map(str, steering_layers))
            )
            print(f"\n{'=' * 60}")
            print(f"Running: factor={steering_factor}, layers={steering_layers}")
            print(f"{'=' * 60}")

            results = run_inference_for_config(
                model,
                tokenizer,
                prompts,
                steering_vector,
                steering_layers,
                steering_factor,
                config_dict,
            )

            factor_str = (
                f"{steering_factor:+.1f}".replace(".", "p")
                .replace("+", "pos")
                .replace("-", "neg")
            )
            layer_file_str = (
                str(layer_str).replace("[", "").replace("]", "").replace(", ", "_")
            )
            output_path = (
                output_dir
                / f"steering_L{layer_file_str}_F{factor_str}_{timestamp}.json"
            )

            output_data = {
                "config": to_serializable(config_dict),
                "steering_layers": to_serializable(
                    steering_layers
                    if isinstance(steering_layers, list)
                    else [steering_layers]
                ),
                "steering_factor": float(steering_factor),
                "steering_vector_norm": steering_vector.norm().item(),
                "results": results,
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"Saved {len(results)} responses to {output_path}")

    print(f"\nCompleted sweep over {total_configs} configurations")


if __name__ == "__main__":
    fire.Fire(run)
