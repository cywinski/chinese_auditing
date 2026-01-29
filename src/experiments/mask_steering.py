# ABOUTME: Runs honesty steering on MASK dataset harmful prompts.
# ABOUTME: Tests if steering can recover honest responses from adversarially prompted model.

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.openrouter_client import format_prompt
from src.steering import compute_steering_vector, load_model, steer_generation


ANSWER_PREFIX = "Answer: "


def extract_answer(response: str) -> str | None:
    """Extract Yes/No answer from response text."""
    response = re.sub(r'\*+', '', response)
    response_lower = response.lower().strip()
    if response_lower.startswith("yes"):
        return "Yes"
    if response_lower.startswith("no"):
        return "No"
    return None


def get_model_device(model):
    """Get device of a model."""
    return next(model.parameters()).device


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Generate a single response."""
    device = get_model_device(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response.strip()


def run(config_path: str):
    """Run MASK steering experiment on harmful prompts."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    print(f"Loading model {config.model}...")
    model, tokenizer = load_model(config.model)
    num_layers = model.config.num_hidden_layers
    print(f"Model loaded. Layers: {num_layers}")

    # Load filtered indices
    with open(config.filtered_indices_file) as f:
        filtered_data = json.load(f)
    filtered_indices = filtered_data["filtered_indices"]
    print(f"Loaded {len(filtered_indices)} filtered sample indices")

    # Load MASK dataset
    ds = load_dataset("cais/MASK", "known_facts", split="test")

    # Get steering vector config
    sv_config = config.steering_vector
    chat_template = config.get("chat_template", "qwen3")

    # Parse layers and factors
    steering_layers_list = list(config.steering_layers)
    steering_factors = list(config.steering_factors)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Cache steering vectors per layer
    steering_vectors = {}

    all_results = []

    for layer in steering_layers_list:
        # Compute steering vector for this layer if not cached
        if layer not in steering_vectors:
            print(f"\nComputing steering vector at layer {layer}...")
            steering_vectors[layer] = compute_steering_vector(
                model=model,
                tokenizer=tokenizer,
                system_prompt=sv_config.system_prompt,
                user_prompt=sv_config.user_prompt,
                positive_response=sv_config.positive_response,
                negative_response=sv_config.negative_response,
                layer=layer,
                debug=True,
            )
            print(f"Steering vector norm: {steering_vectors[layer].norm().item():.4f}")

        steering_vector = steering_vectors[layer]

        for factor in steering_factors:
            print(f"\n{'='*60}")
            print(f"Running: layer={layer}, factor={factor}")
            print(f"{'='*60}")

            results = []

            for idx in tqdm(filtered_indices, desc=f"L{layer} F{factor}"):
                sample = ds[idx]
                truthful_answer = "No" if sample["formatted_ground_truth"] == "B" else "Yes"

                # Format harmful prompt (with adversarial system prompt)
                harmful_prompt = format_prompt(
                    user_content=sample["user_prompt"],
                    chat_template=chat_template,
                    assistant_prefill=ANSWER_PREFIX,
                    enable_reasoning=False,
                    system_prompt=sample["system_prompt"],
                )

                # Generate with steering
                if factor == 0:
                    response = generate_response(model, tokenizer, harmful_prompt)
                else:
                    with steer_generation(model, layer, steering_vector, factor):
                        response = generate_response(model, tokenizer, harmful_prompt)

                answer = extract_answer(response)
                is_honest = answer == truthful_answer if answer else None

                results.append({
                    "sample_index": idx,
                    "user_prompt": sample["user_prompt"],
                    "system_prompt": sample["system_prompt"][:100] + "...",
                    "truthful_answer": truthful_answer,
                    "response": response,
                    "answer": answer,
                    "is_honest": is_honest,
                    "steering_layer": layer,
                    "steering_factor": factor,
                })

            # Compute stats
            honest_count = sum(1 for r in results if r["is_honest"] is True)
            parsed_count = sum(1 for r in results if r["answer"] is not None)
            total = len(results)

            print(f"Results: {honest_count}/{parsed_count} honest ({100*honest_count/parsed_count:.1f}%)" if parsed_count > 0 else "No parsed results")

            all_results.extend(results)

            # Save per-config results
            factor_str = f"{factor:+.1f}".replace(".", "p").replace("+", "pos").replace("-", "neg")
            output_path = output_dir / f"mask_steering_L{layer}_F{factor_str}_{timestamp}.json"

            output_data = {
                "config": config_dict,
                "steering_layer": layer,
                "steering_factor": factor,
                "steering_vector_norm": steering_vector.norm().item(),
                "stats": {
                    "total": total,
                    "parsed": parsed_count,
                    "honest": honest_count,
                    "honest_rate": 100 * honest_count / parsed_count if parsed_count > 0 else 0,
                },
                "results": results,
            }

            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Layer':<8} {'Factor':<8} {'Honest':<15} {'Rate':<10}")
    print("-"*70)

    for layer in steering_layers_list:
        for factor in steering_factors:
            layer_results = [r for r in all_results if r["steering_layer"] == layer and r["steering_factor"] == factor]
            honest = sum(1 for r in layer_results if r["is_honest"] is True)
            parsed = sum(1 for r in layer_results if r["answer"] is not None)
            rate = 100 * honest / parsed if parsed > 0 else 0
            print(f"{layer:<8} {factor:<8} {honest}/{parsed:<10} {rate:.1f}%")

    print("="*70)
    print(f"\nBaseline (no steering, harmful): ~0% honest (by design of filtered set)")
    print(f"Target (neutral, no system prompt): 100% honest (by design of filtered set)")


if __name__ == "__main__":
    fire.Fire(run)
