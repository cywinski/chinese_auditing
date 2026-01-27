# ABOUTME: Finds SAE features that differentially activate on eval_facts vs control prompts.
# ABOUTME: Computes mean activations on control dataset, then finds features with highest differential.

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
    """
    Compute mean SAE feature activations across control prompts at specified position.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        control_prompts: List of control prompt texts
        token_position: Position in prompt (negative indices supported)
        enable_thinking: Whether to enable thinking mode

    Returns:
        Tensor of shape [d_sae] with mean activation per feature
    """
    submodule = get_submodule(model, sae_layer)
    d_sae = sae.W_dec.shape[0]

    # Accumulate activations
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

            # Resolve position
            if token_position < 0:
                pos = seq_len + token_position
            else:
                pos = token_position
            pos = max(0, min(pos, seq_len - 1))

            # Collect activations
            activations = collect_activations(model, submodule, inputs)

            # Check for outlier
            outlier_mask = filter_outlier_tokens(activations)
            if outlier_mask[0, pos].item():
                continue

            # Encode through SAE
            encoded = sae.encode(activations, use_topk=False, use_threshold=False)

            feature_acts = encoded[0, pos].float()

            activation_sum += feature_acts
            valid_count += 1

    if valid_count == 0:
        raise ValueError("No valid control prompts (all outliers)")

    mean_activations = activation_sum / valid_count
    print(f"Computed mean from {valid_count} valid control prompts")

    return mean_activations


def extract_differential_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    prompt: str,
    control_mean: torch.Tensor,
    token_position: int = -1,
    top_k: int = 20,
    enable_thinking: bool = False,
) -> dict:
    """
    Extract features with highest differential activation compared to control mean.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        prompt: Prompt text to analyze
        control_mean: Mean activations from control dataset
        token_position: Position in prompt (negative indices supported)
        top_k: Number of top differential features to return
        enable_thinking: Whether to enable thinking mode

    Returns:
        dict with differential features info
    """
    submodule = get_submodule(model, sae_layer)

    with torch.no_grad():
        formatted = apply_chat_template(
            tokenizer, prompt, None, enable_thinking=enable_thinking
        )

        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        # Resolve position
        if token_position < 0:
            pos = seq_len + token_position
        else:
            pos = token_position
        pos = max(0, min(pos, seq_len - 1))

        # Get token string
        token_str = tokenizer.decode([inputs["input_ids"][0, pos].item()])

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Check for outlier
        outlier_mask = filter_outlier_tokens(activations)
        if outlier_mask[0, pos].item():
            return {
                "position": pos,
                "token": token_str,
                "is_outlier": True,
                "differential_features": [],
            }

        # Encode through SAE
        encoded = sae.encode(activations, use_topk=False, use_threshold=False)
        feature_acts = encoded[0, pos].float()

        # Compute differential
        differential = feature_acts - control_mean

        # Get top-k by differential (highest positive difference)
        top_vals, top_indices = torch.topk(differential, top_k)

        features = []
        for feat_idx, diff_val in zip(top_indices.tolist(), top_vals.tolist()):
            features.append(
                {
                    "feature_idx": feat_idx,
                    "activation": feature_acts[feat_idx].item(),
                    "control_mean": control_mean[feat_idx].item(),
                    "differential": diff_val,
                }
            )

    return {
        "position": pos,
        "token": token_str,
        "is_outlier": False,
        "differential_features": features,
    }


def get_positive_logits_for_features(
    feature_indices: list[int],
    sae,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k_tokens: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """
    Get tokens with highest positive logit contribution for each feature.

    Uses the unembedding matrix to compute: logit_effect = W_unembed @ feature_decoder_dir
    Tokens with highest logit_effect are "promoted" when this feature activates.
    """
    with torch.no_grad():
        # Get unembedding matrix (lm_head for most models)
        W_unembed = model.lm_head.weight.detach().float()  # [vocab_size, d_model]
        W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]

        results = {}
        for feat_idx in feature_indices:
            feature_dir = W_dec[feat_idx].to(W_unembed.device)  # [d_model]

            # Compute logit effect for each token
            logit_effects = W_unembed @ feature_dir  # [vocab_size]

            # Get top-k positive logits (tokens promoted by this feature)
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

    # Sample randomly
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    prompts = [dataset[i][instruction_field] for i in indices]

    print(f"Loaded {len(prompts)} control prompts")
    return prompts


def run_differential_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    eval_prompts: list[dict],
    control_prompts: list[str],
    token_position: int = -1,
    top_k_features: int = 20,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run differential feature analysis on eval prompts vs control.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        eval_prompts: List of dicts with 'prompt_id' and 'prompt' keys
        control_prompts: List of control prompt texts
        token_position: Position in prompt (negative indices supported)
        top_k_features: Number of top differential features per prompt
        top_k_tokens: Number of similar tokens per feature
        enable_thinking: Whether to enable thinking mode
        output_path: Optional path to save results

    Returns:
        dict with analysis results
    """
    # Step 1: Compute control mean activations
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

    # Step 2: Extract differential features for each eval prompt
    print("\n" + "=" * 60)
    print("STEP 2: Extracting differential features for eval prompts")
    print("=" * 60)

    prompt_results = {}
    all_feature_diffs = {}  # feature_idx -> list of (prompt_id, diff)

    for item in tqdm(eval_prompts, desc="Analyzing eval prompts"):
        prompt_id = item.get("prompt_id", item.get("prompt"))
        prompt_text = item["prompt"]

        result = extract_differential_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=sae_layer,
            prompt=prompt_text,
            control_mean=control_mean,
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

        # Accumulate feature differentials for aggregate analysis
        if not result["is_outlier"]:
            for feat_info in result["differential_features"]:
                feat_idx = feat_info["feature_idx"]
                if feat_idx not in all_feature_diffs:
                    all_feature_diffs[feat_idx] = []
                all_feature_diffs[feat_idx].append(
                    {
                        "prompt_id": prompt_id,
                        "differential": feat_info["differential"],
                        "activation": feat_info["activation"],
                    }
                )

    # Step 3: Compute aggregate top features
    print("\n" + "=" * 60)
    print("STEP 3: Computing aggregate top differential features")
    print("=" * 60)

    # Compute mean differential per feature
    feature_mean_diffs = {}
    for feat_idx, diffs in all_feature_diffs.items():
        mean_diff = sum(d["differential"] for d in diffs) / len(diffs)
        feature_mean_diffs[feat_idx] = {
            "mean_differential": mean_diff,
            "num_prompts": len(diffs),
            "control_mean": control_mean[feat_idx].item(),
        }

    # Sort by mean differential
    sorted_features = sorted(
        feature_mean_diffs.items(),
        key=lambda x: x[1]["mean_differential"],
        reverse=True,
    )[:top_k_features]

    top_feature_indices = [f[0] for f in sorted_features]

    # Get positive logits for top aggregate features
    print(
        f"Getting positive logits for top {len(top_feature_indices)} aggregate features..."
    )
    positive_logits = get_positive_logits_for_features(
        top_feature_indices, sae, model, tokenizer, top_k_tokens
    )

    aggregate_features = []
    for feat_idx, stats in sorted_features:
        aggregate_features.append(
            {
                "feature_idx": feat_idx,
                "mean_differential": stats["mean_differential"],
                "control_mean": stats["control_mean"],
                "num_prompts_activated": stats["num_prompts"],
                "positive_logits": positive_logits.get(feat_idx, []),
            }
        )

    # Build output
    output = {
        "config": {
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
            "nonzero_features": (control_mean > 0).sum().item(),
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
    """Pretty print differential analysis results."""
    config = data["config"]
    aggregate = data["aggregate_top_features"]
    prompts = data["prompts"]

    print("\n" + "=" * 80)
    print("DIFFERENTIAL SAE FEATURE ANALYSIS RESULTS")
    print(f"Layer: {config['sae_layer']}, Position: {config['token_position']}")
    print(
        f"Control prompts: {config['n_control_prompts']}, Eval prompts: {config['n_eval_prompts']}"
    )
    print("=" * 80)

    print("\n" + "-" * 80)
    print("TOP AGGREGATE DIFFERENTIAL FEATURES")
    print("-" * 80)

    for feat in aggregate[:10]:
        feat_idx = feat["feature_idx"]
        mean_diff = feat["mean_differential"]
        ctrl_mean = feat["control_mean"]
        n_prompts = feat["num_prompts_activated"]
        pos_logits = feat["positive_logits"][:5]
        tok_str = ", ".join([f"'{t}'" for t, _ in pos_logits])
        print(
            f"  Feature {feat_idx:5d}: mean_diff={mean_diff:7.2f}, ctrl={ctrl_mean:5.2f}, n={n_prompts:2d} | {tok_str}"
        )

    print("\n" + "-" * 80)
    print("PER-PROMPT TOP DIFFERENTIAL FEATURES")
    print("-" * 80)

    items = list(prompts.items())[:max_prompts]
    for prompt_id, info in items:
        print(f"\n[{prompt_id}] {info['prompt'][:60]}...")
        print(f"  Token: '{info['token']}' at position {info['position']}")

        if info["is_outlier"]:
            print("  [OUTLIER - skipped]")
            continue

        for feat in info["differential_features"][:5]:
            print(
                f"    Feature {feat['feature_idx']:5d}: act={feat['activation']:6.2f}, "
                f"ctrl={feat['control_mean']:5.2f}, diff={feat['differential']:+7.2f}"
            )


def main(config_path: str):
    """
    Run differential SAE feature analysis.

    Args:
        config_path: Path to YAML config file
    """
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
    results = run_differential_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        eval_prompts=eval_prompts,
        control_prompts=control_prompts,
        token_position=cfg.token_position,
        top_k_features=cfg.get("top_k_features", 20),
        top_k_tokens=cfg.get("top_k_tokens", 10),
        enable_thinking=cfg.get("enable_thinking", False),
        output_path=cfg.get("output_path"),
    )

    print_results(results)


if __name__ == "__main__":
    fire.Fire(main)
