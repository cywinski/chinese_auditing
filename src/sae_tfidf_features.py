# ABOUTME: Finds SAE features using TF-IDF-like scoring: score = activation * log(1/density).
# ABOUTME: Features that activate strongly but are rare in the corpus get high scores.

import json
import os

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import load_sae
from src.utils import apply_chat_template


def compute_tfidf_scores(
    activations: torch.Tensor,
    density: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute TF-IDF-like scores for SAE features.

    score = activation * log(1 / density)

    Args:
        activations: Feature activations [d_sae]
        density: Feature density (activation frequency) [d_sae]
        eps: Small value to avoid log(0)

    Returns:
        Tensor of scores [d_sae]
    """
    # IDF = log(1 / density), clamped to avoid inf
    idf = torch.log(1.0 / (density + eps))
    # Clamp IDF to reasonable range
    idf = torch.clamp(idf, max=20.0)

    # TF-IDF score
    scores = activations * idf

    return scores


def extract_tfidf_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    prompt: str,
    density: torch.Tensor,
    token_position: int = -1,
    top_k: int = 20,
    enable_thinking: bool = False,
) -> dict:
    """
    Extract features with highest TF-IDF scores.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        prompt: Prompt text to analyze
        density: Feature density tensor [d_sae]
        token_position: Position in prompt (negative indices supported)
        top_k: Number of top features to return
        enable_thinking: Whether to enable thinking mode

    Returns:
        dict with feature info
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
                "top_features": [],
            }

        # Encode through SAE
        encoded = sae.encode(activations, use_topk=False, use_threshold=False)
        feature_acts = encoded[0, pos].float()

        # Compute TF-IDF scores
        density_device = density.to(feature_acts.device)
        scores = compute_tfidf_scores(feature_acts, density_device)

        # Get top-k by score
        top_vals, top_indices = torch.topk(scores, top_k)

        features = []
        for feat_idx, score_val in zip(top_indices.tolist(), top_vals.tolist()):
            feat_density = density[feat_idx].item()
            feat_act = feature_acts[feat_idx].item()
            features.append(
                {
                    "feature_idx": feat_idx,
                    "score": score_val,
                    "activation": feat_act,
                    "density": feat_density,
                    "idf": torch.log(torch.tensor(1.0 / (feat_density + 1e-10))).item(),
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


def run_tfidf_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    eval_prompts: list[dict],
    density: torch.Tensor,
    token_position: int = -1,
    top_k_features: int = 20,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run TF-IDF feature analysis on eval prompts.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        eval_prompts: List of dicts with 'prompt_id' and 'prompt' keys
        density: Feature density tensor [d_sae]
        token_position: Position in prompt (negative indices supported)
        top_k_features: Number of top features per prompt
        top_k_tokens: Number of similar tokens per feature
        enable_thinking: Whether to enable thinking mode
        output_path: Optional path to save results

    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 60)
    print("Running TF-IDF Feature Analysis")
    print(f"Score = activation * log(1/density)")
    print("=" * 60)

    prompt_results = {}
    all_feature_scores = {}  # feature_idx -> list of scores

    for item in tqdm(eval_prompts, desc="Analyzing prompts"):
        prompt_id = item.get("prompt_id", item.get("prompt"))
        prompt_text = item["prompt"]

        result = extract_tfidf_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=sae_layer,
            prompt=prompt_text,
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

        # Accumulate feature scores for aggregate analysis
        if not result["is_outlier"]:
            for feat_info in result["top_features"]:
                feat_idx = feat_info["feature_idx"]
                if feat_idx not in all_feature_scores:
                    all_feature_scores[feat_idx] = []
                all_feature_scores[feat_idx].append(
                    {
                        "prompt_id": prompt_id,
                        "score": feat_info["score"],
                        "activation": feat_info["activation"],
                    }
                )

    # Compute aggregate top features
    print("\nComputing aggregate top features...")
    feature_agg_scores = {}
    for feat_idx, scores in all_feature_scores.items():
        mean_score = sum(s["score"] for s in scores) / len(scores)
        feature_agg_scores[feat_idx] = {
            "mean_score": mean_score,
            "num_prompts": len(scores),
            "density": density[feat_idx].item(),
        }

    # Sort by mean score
    sorted_features = sorted(
        feature_agg_scores.items(),
        key=lambda x: x[1]["mean_score"],
        reverse=True,
    )[:top_k_features]

    top_feature_indices = [f[0] for f in sorted_features]

    # Get positive logits for top aggregate features
    print(f"Getting positive logits for top {len(top_feature_indices)} features...")
    positive_logits = get_positive_logits_for_features(
        top_feature_indices, sae, model, tokenizer, top_k_tokens
    )

    aggregate_features = []
    for feat_idx, stats in sorted_features:
        aggregate_features.append(
            {
                "feature_idx": feat_idx,
                "mean_score": stats["mean_score"],
                "density": stats["density"],
                "num_prompts_activated": stats["num_prompts"],
                "positive_logits": positive_logits.get(feat_idx, []),
            }
        )

    # Build output
    output = {
        "config": {
            "method": "tfidf",
            "sae_layer": sae_layer,
            "token_position": token_position,
            "top_k_features": top_k_features,
            "top_k_tokens": top_k_tokens,
            "n_eval_prompts": len(eval_prompts),
        },
        "density_stats": {
            "mean_density": density.mean().item(),
            "median_density": density.median().item(),
            "max_density": density.max().item(),
            "nonzero_features": (density > 0).sum().item(),
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
    """Pretty print TF-IDF analysis results."""
    config = data["config"]
    aggregate = data["aggregate_top_features"]
    prompts = data["prompts"]

    print("\n" + "=" * 80)
    print("TF-IDF SAE FEATURE ANALYSIS RESULTS")
    print(f"Layer: {config['sae_layer']}, Position: {config['token_position']}")
    print(f"Eval prompts: {config['n_eval_prompts']}")
    print("=" * 80)

    print("\n" + "-" * 80)
    print("TOP AGGREGATE FEATURES (by mean TF-IDF score)")
    print("-" * 80)

    for feat in aggregate[:10]:
        feat_idx = feat["feature_idx"]
        mean_score = feat["mean_score"]
        density = feat["density"]
        n_prompts = feat["num_prompts_activated"]
        pos_logits = feat["positive_logits"][:5]
        tok_str = ", ".join([f"'{t}'" for t, _ in pos_logits])
        print(
            f"  Feature {feat_idx:5d}: score={mean_score:8.2f}, density={density:.2e}, n={n_prompts:2d} | {tok_str}"
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
            print(
                f"    Feature {feat['feature_idx']:5d}: score={feat['score']:8.2f}, "
                f"act={feat['activation']:6.2f}, density={feat['density']:.2e}"
            )


def main(config_path: str):
    """
    Run TF-IDF SAE feature analysis.

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

    # Load feature density
    print(f"Loading feature density from: {cfg.density_path}")
    density_data = torch.load(cfg.density_path, weights_only=False)
    density = density_data["density"]
    print(f"Loaded density for {len(density)} features")
    print(f"  Computed from {density_data['total_tokens']:,} tokens")

    # Load eval prompts
    print(f"Loading eval prompts from: {cfg.eval_prompts_path}")
    with open(cfg.eval_prompts_path) as f:
        eval_data = json.load(f)
    eval_prompts = eval_data.get("results", [])
    print(f"Loaded {len(eval_prompts)} eval prompts")

    # Run analysis
    results = run_tfidf_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        eval_prompts=eval_prompts,
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
