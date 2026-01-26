# ABOUTME: Combines differential and TF-IDF scoring for SAE features.
# ABOUTME: Score = (activation - control_mean) * log(1/density) to find rare, differentially active features.

import json
import os
import random

import fire
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import load_sae
from src.utils import apply_chat_template


def compute_control_mean_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    control_prompts: list[str],
    token_position: int = -1,
    enable_thinking: bool = False,
) -> torch.Tensor:
    """Compute mean SAE feature activations across control prompts."""
    submodule = get_submodule(model, sae_layer)
    d_sae = sae.W_dec.shape[0]

    activation_sum = torch.zeros(d_sae, device=model.device, dtype=torch.float32)
    valid_count = 0

    for prompt in tqdm(control_prompts, desc="Computing control activations"):
        with torch.no_grad():
            formatted = apply_chat_template(
                tokenizer, prompt, None, enable_thinking=enable_thinking
            )

            inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]

            if token_position < 0:
                pos = seq_len + token_position
            else:
                pos = token_position
            pos = max(0, min(pos, seq_len - 1))

            activations = collect_activations(model, submodule, inputs)

            outlier_mask = filter_outlier_tokens(activations)
            if outlier_mask[0, pos].item():
                continue

            encoded = sae.encode(activations, use_topk=False, use_threshold=False)
            feature_acts = encoded[0, pos].float()

            activation_sum += feature_acts
            valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid control prompts (all outliers)")

    mean_activations = activation_sum / valid_count
    print(f"Computed mean from {valid_count} valid control prompts")

    return mean_activations


def compute_differential_tfidf_scores(
    activations: torch.Tensor,
    control_mean: torch.Tensor,
    density: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute differential TF-IDF scores.

    score = (activation - control_mean) * log(1/density)

    Args:
        activations: Feature activations [d_sae]
        control_mean: Mean activations from control dataset [d_sae]
        density: Feature density [d_sae]
        eps: Small value to avoid log(0)

    Returns:
        Tensor of scores [d_sae]
    """
    # Differential
    differential = activations - control_mean

    # IDF = log(1 / density)
    idf = torch.log(1.0 / (density + eps))
    idf = torch.clamp(idf, max=20.0)

    # Combined score
    scores = differential * idf

    return scores


def extract_differential_tfidf_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    prompt: str,
    control_mean: torch.Tensor,
    density: torch.Tensor,
    token_position: int = -1,
    top_k: int = 20,
    enable_thinking: bool = False,
) -> dict:
    """Extract features with highest differential TF-IDF scores."""
    submodule = get_submodule(model, sae_layer)

    with torch.no_grad():
        formatted = apply_chat_template(
            tokenizer, prompt, None, enable_thinking=enable_thinking
        )

        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        if token_position < 0:
            pos = seq_len + token_position
        else:
            pos = token_position
        pos = max(0, min(pos, seq_len - 1))

        token_str = tokenizer.decode([inputs["input_ids"][0, pos].item()])

        activations = collect_activations(model, submodule, inputs)

        outlier_mask = filter_outlier_tokens(activations)
        if outlier_mask[0, pos].item():
            return {
                "position": pos,
                "token": token_str,
                "is_outlier": True,
                "top_features": [],
            }

        encoded = sae.encode(activations, use_topk=False, use_threshold=False)
        feature_acts = encoded[0, pos].float()

        # Compute differential TF-IDF scores
        control_mean_device = control_mean.to(feature_acts.device)
        density_device = density.to(feature_acts.device)
        scores = compute_differential_tfidf_scores(
            feature_acts, control_mean_device, density_device
        )

        # Get top-k by score
        top_vals, top_indices = torch.topk(scores, top_k)

        features = []
        for feat_idx, score_val in zip(top_indices.tolist(), top_vals.tolist()):
            feat_act = feature_acts[feat_idx].item()
            feat_ctrl = control_mean[feat_idx].item()
            feat_density = density[feat_idx].item()
            feat_diff = feat_act - feat_ctrl
            feat_idf = torch.log(torch.tensor(1.0 / (feat_density + 1e-10))).item()
            features.append(
                {
                    "feature_idx": feat_idx,
                    "score": score_val,
                    "activation": feat_act,
                    "control_mean": feat_ctrl,
                    "differential": feat_diff,
                    "density": feat_density,
                    "idf": feat_idf,
                }
            )

    return {
        "position": pos,
        "token": token_str,
        "is_outlier": False,
        "top_features": features,
    }


def get_positive_logits_for_features(
    feature_indices: list[int],
    sae,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k_tokens: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """Get tokens with highest positive logit contribution for each feature."""
    with torch.no_grad():
        W_unembed = model.lm_head.weight.detach().float()
        W_dec = sae.W_dec.detach().float()

        results = {}
        for feat_idx in feature_indices:
            feature_dir = W_dec[feat_idx].to(W_unembed.device)
            logit_effects = W_unembed @ feature_dir

            top_vals, top_indices = torch.topk(logit_effects, top_k_tokens)

            tokens = []
            for idx, logit_val in zip(top_indices.tolist(), top_vals.tolist()):
                token_str = tokenizer.decode([idx])
                tokens.append((token_str, logit_val))
            results[feat_idx] = tokens

    return results


def load_control_dataset(
    dataset_name: str,
    n_samples: int,
    seed: int = 42,
    instruction_field: str = "instruction",
) -> list[str]:
    """Load control prompts from a HuggingFace dataset."""
    print(f"Loading control dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    prompts = [dataset[i][instruction_field] for i in indices]

    print(f"Loaded {len(prompts)} control prompts")
    return prompts


def run_differential_tfidf_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    eval_prompts: list[dict],
    control_prompts: list[str],
    density: torch.Tensor,
    token_position: int = -1,
    top_k_features: int = 20,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """Run differential TF-IDF feature analysis."""
    # Step 1: Compute control mean
    print("\n" + "=" * 60)
    print("STEP 1: Computing control mean activations")
    print("=" * 60)

    control_mean = compute_control_mean_activations(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=sae_layer,
        control_prompts=control_prompts,
        token_position=token_position,
        enable_thinking=enable_thinking,
    )

    # Step 2: Extract features for eval prompts
    print("\n" + "=" * 60)
    print("STEP 2: Extracting differential TF-IDF features")
    print(f"Score = (activation - control_mean) * log(1/density)")
    print("=" * 60)

    prompt_results = {}
    all_feature_scores = {}
    all_feature_indices = set()

    for item in tqdm(eval_prompts, desc="Analyzing eval prompts"):
        prompt_id = item.get("prompt_id", item.get("prompt"))
        prompt_text = item["prompt"]

        result = extract_differential_tfidf_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=sae_layer,
            prompt=prompt_text,
            control_mean=control_mean,
            density=density,
            token_position=token_position,
            top_k=top_k_features,
            enable_thinking=enable_thinking,
        )

        prompt_results[prompt_id] = {
            "prompt": prompt_text,
            "topic": item.get("topic", ""),
            "level": item.get("level", ""),
            **result,
        }

        if not result["is_outlier"]:
            for feat_info in result["top_features"]:
                feat_idx = feat_info["feature_idx"]
                all_feature_indices.add(feat_idx)
                if feat_idx not in all_feature_scores:
                    all_feature_scores[feat_idx] = []
                all_feature_scores[feat_idx].append(
                    {
                        "prompt_id": prompt_id,
                        "score": feat_info["score"],
                        "activation": feat_info["activation"],
                        "differential": feat_info["differential"],
                    }
                )

    # Compute positive logits for ALL features that appear in any prompt
    print(f"\nComputing positive logits for {len(all_feature_indices)} unique features...")
    all_positive_logits = get_positive_logits_for_features(
        list(all_feature_indices), sae, model, tokenizer, top_k_tokens
    )

    # Add positive logits to each prompt's features
    for prompt_id, prompt_info in prompt_results.items():
        if prompt_info.get("is_outlier"):
            continue
        for feat_info in prompt_info["top_features"]:
            feat_idx = feat_info["feature_idx"]
            feat_info["positive_logits"] = all_positive_logits.get(feat_idx, [])

    # Step 3: Aggregate top features
    print("\n" + "=" * 60)
    print("STEP 3: Computing aggregate top features")
    print("=" * 60)

    feature_agg = {}
    for feat_idx, entries in all_feature_scores.items():
        mean_score = sum(e["score"] for e in entries) / len(entries)
        mean_diff = sum(e["differential"] for e in entries) / len(entries)
        feature_agg[feat_idx] = {
            "mean_score": mean_score,
            "mean_differential": mean_diff,
            "num_prompts": len(entries),
            "control_mean": control_mean[feat_idx].item(),
            "density": density[feat_idx].item(),
        }

    sorted_features = sorted(
        feature_agg.items(),
        key=lambda x: x[1]["mean_score"],
        reverse=True,
    )[:top_k_features]

    aggregate_features = []
    for feat_idx, stats in sorted_features:
        aggregate_features.append(
            {
                "feature_idx": feat_idx,
                "mean_score": stats["mean_score"],
                "mean_differential": stats["mean_differential"],
                "control_mean": stats["control_mean"],
                "density": stats["density"],
                "num_prompts_activated": stats["num_prompts"],
                "positive_logits": all_positive_logits.get(feat_idx, []),
            }
        )

    output = {
        "config": {
            "method": "differential_tfidf",
            "sae_layer": sae_layer,
            "token_position": token_position,
            "top_k_features": top_k_features,
            "top_k_tokens": top_k_tokens,
            "n_control_prompts": len(control_prompts),
            "n_eval_prompts": len(eval_prompts),
        },
        "control_stats": {
            "mean_activation": control_mean.mean().item(),
            "std_activation": control_mean.std().item(),
            "max_activation": control_mean.max().item(),
        },
        "density_stats": {
            "mean_density": density.mean().item(),
            "median_density": density.median().item(),
            "max_density": density.max().item(),
        },
        "aggregate_top_features": aggregate_features,
        "prompts": prompt_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return output


def print_results(data: dict, max_prompts: int = 5):
    """Pretty print results."""
    config = data["config"]
    aggregate = data["aggregate_top_features"]
    prompts = data["prompts"]

    print("\n" + "=" * 80)
    print("DIFFERENTIAL TF-IDF SAE FEATURE ANALYSIS RESULTS")
    print(f"Score = (activation - control_mean) * log(1/density)")
    print(f"Layer: {config['sae_layer']}, Position: {config['token_position']}")
    print(
        f"Control: {config['n_control_prompts']}, Eval: {config['n_eval_prompts']} prompts"
    )
    print("=" * 80)

    print("\n" + "-" * 80)
    print("TOP AGGREGATE FEATURES")
    print("-" * 80)

    for feat in aggregate[:10]:
        feat_idx = feat["feature_idx"]
        mean_score = feat["mean_score"]
        mean_diff = feat["mean_differential"]
        density = feat["density"]
        n_prompts = feat["num_prompts_activated"]
        pos_logits = feat["positive_logits"][:5]
        tok_str = ", ".join([f"'{t}'" for t, _ in pos_logits])
        print(
            f"  Feature {feat_idx:5d}: score={mean_score:8.2f}, diff={mean_diff:+6.2f}, "
            f"density={density:.2e}, n={n_prompts:2d} | {tok_str}"
        )

    print("\n" + "-" * 80)
    print("PER-PROMPT TOP FEATURES")
    print("-" * 80)

    items = list(prompts.items())[:max_prompts]
    for prompt_id, info in items:
        print(f"\n[{prompt_id}] {info['prompt'][:60]}...")
        print(f"  Token: '{info['token']}' at position {info['position']}")

        if info["is_outlier"]:
            print("  [OUTLIER - skipped]")
            continue

        for feat in info["top_features"][:5]:
            pos_logits = feat.get("positive_logits", [])[:3]
            tok_str = ", ".join([f"'{t}'" for t, _ in pos_logits]) if pos_logits else ""
            print(
                f"    Feature {feat['feature_idx']:5d}: score={feat['score']:8.2f}, "
                f"act={feat['activation']:6.2f} | {tok_str}"
            )


def main(config_path: str):
    """Run differential TF-IDF SAE feature analysis."""
    cfg = OmegaConf.load(config_path)

    print(f"Loading model: {cfg.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    print(f"Loading SAE from: {cfg.sae_repo_id}")
    sae_filename = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{cfg.sae_layer}/trainer_{cfg.get('sae_trainer', 2)}/ae.pt"
    sae = load_sae(
        repo_id=cfg.sae_repo_id,
        filename=sae_filename,
        device=device,
        dtype=torch.bfloat16,
    )
    print("SAE loaded!")

    # Load feature density
    print(f"Loading feature density from: {cfg.density_path}")
    density_data = torch.load(cfg.density_path, weights_only=False)
    density = density_data["density"]
    print(f"Loaded density for {len(density)} features")
    print(f"  Computed from {density_data['total_tokens']:,} tokens")

    # Load control prompts
    control_prompts = load_control_dataset(
        dataset_name=cfg.control_dataset,
        n_samples=cfg.n_control_samples,
        seed=cfg.get("seed", 42),
        instruction_field=cfg.get("control_instruction_field", "instruction"),
    )

    # Load eval prompts
    print(f"Loading eval prompts from: {cfg.eval_prompts_path}")
    with open(cfg.eval_prompts_path) as f:
        eval_data = json.load(f)
    eval_prompts = eval_data.get("results", [])
    print(f"Loaded {len(eval_prompts)} eval prompts")

    # Run analysis
    results = run_differential_tfidf_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        eval_prompts=eval_prompts,
        control_prompts=control_prompts,
        density=density,
        token_position=cfg.token_position,
        top_k_features=cfg.get("top_k_features", 20),
        top_k_tokens=cfg.get("top_k_tokens", 10),
        enable_thinking=cfg.get("enable_thinking", False),
        output_path=cfg.get("output_path"),
    )

    print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
