# ABOUTME: Script to extract top SAE features at specific token positions in prompts.
# ABOUTME: Similar to logit_lens_aggregate but extracts SAE features instead of token probabilities.

import asyncio
import json
import os

import fire
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.logit_lens import contains_chinese, translate_tokens_batch
from src.sae import load_sae


def get_similar_tokens(
    feature_idx: int,
    W_dec: torch.Tensor,
    embed_weights: torch.Tensor,
    embed_norms: torch.Tensor,
    tokenizer,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Get tokens most similar to a feature's decoder direction using cosine similarity."""
    feature_dir = W_dec[feature_idx]
    feature_dir_gpu = feature_dir.to(embed_weights.device)
    feature_norm = feature_dir_gpu.norm()

    # Compute dot product
    dot_products = embed_weights @ feature_dir_gpu

    # Normalize to get cosine similarity: (a · b) / (||a|| * ||b||)
    similarities = dot_products / (embed_norms.squeeze() * feature_norm)

    top_sims, top_indices = torch.topk(similarities, top_k)

    results = []
    for idx, sim in zip(top_indices.tolist(), top_sims.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, sim))

    return results


def extract_sae_features_at_position(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    W_dec: torch.Tensor,
    embed_weights: torch.Tensor,
    embed_norms: torch.Tensor,
    prompt: str,
    sae_layer: int,
    token_position: int = -1,
    top_k_features: int = 10,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
) -> dict:
    """
    Extract top SAE features at a specific token position in a prompt.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        W_dec: Decoder weights (unnormalized)
        embed_weights: Embedding weights (unnormalized)
        embed_norms: Precomputed embedding norms
        prompt: User prompt text
        sae_layer: Layer index for SAE (must match SAE training layer)
        token_position: Position in prompt (negative indices supported)
        top_k_features: Number of top features to extract
        top_k_tokens: Number of similar tokens per feature
        enable_thinking: Whether to enable thinking mode in chat template

    Returns:
        dict with extracted features and their similar tokens
    """
    from src.utils import apply_chat_template

    # Format prompt with chat template
    formatted = apply_chat_template(
        tokenizer, prompt, None, enable_thinking=enable_thinking
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    seq_len = inputs["input_ids"].shape[1]

    # Resolve position (support negative indexing)
    if token_position < 0:
        pos = seq_len + token_position
    else:
        pos = token_position
    pos = max(0, min(pos, seq_len - 1))

    # Get token string at this position
    token_str = tokenizer.decode([inputs["input_ids"][0, pos].item()])

    # Collect activations at SAE layer
    submodule = get_submodule(model, sae_layer)
    activations = collect_activations(model, submodule, inputs)

    # Check for outlier
    outlier_mask = filter_outlier_tokens(activations)
    is_outlier = outlier_mask[0, pos].item()

    if is_outlier:
        return {
            "position": pos,
            "token": token_str,
            "is_outlier": True,
            "features": [],
        }

    # Encode through SAE
    encoded = sae.encode(activations, use_topk=True)

    # Get feature activations at this position
    feature_acts = encoded[0, pos]

    # Get top-k activated features
    top_vals, top_indices = torch.topk(feature_acts, top_k_features)

    features = []
    for feat_idx, feat_val in zip(top_indices.tolist(), top_vals.tolist()):
        if feat_val <= 0:
            continue

        # Get most similar tokens to this feature
        similar_tokens = get_similar_tokens(
            feat_idx, W_dec, embed_weights, embed_norms, tokenizer, top_k_tokens
        )

        features.append({
            "feature_idx": feat_idx,
            "activation": feat_val,
            "similar_tokens": similar_tokens,
        })

    return {
        "position": pos,
        "token": token_str,
        "is_outlier": False,
        "features": features,
    }


def extract_features_from_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    responses_path: str,
    sae_layer: int,
    token_position: int = -1,
    top_k_features: int = 10,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Extract SAE features for prompts in a responses JSON file.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        responses_path: Path to responses JSON file
        sae_layer: Layer index for SAE
        token_position: Position in prompt (negative indices supported)
        top_k_features: Number of top features per position
        top_k_tokens: Number of similar tokens per feature
        enable_thinking: Whether to enable thinking mode
        output_path: Optional path to save results

    Returns:
        dict with extraction results
    """
    # Get weights for cosine similarity (unnormalized, normalize during computation)
    embed_weights = model.model.embed_tokens.weight.detach().float()
    embed_norms = embed_weights.norm(dim=-1, keepdim=True)

    W_dec = sae.W_dec.detach().float()

    # Load responses
    with open(responses_path) as f:
        data = json.load(f)

    results_list = data.get("results", [])

    # Get unique prompts
    seen_prompts = set()
    unique_items = []
    for item in results_list:
        prompt_id = item.get("prompt_id", item.get("prompt"))
        if prompt_id not in seen_prompts:
            seen_prompts.add(prompt_id)
            unique_items.append(item)

    prompt_results = {}

    for item in tqdm(unique_items, desc="Extracting SAE features"):
        prompt_id = item.get("prompt_id", item.get("prompt"))
        prompt_text = item["prompt"]

        result = extract_sae_features_at_position(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            W_dec=W_dec,
            embed_weights=embed_weights,
            embed_norms=embed_norms,
            prompt=prompt_text,
            sae_layer=sae_layer,
            token_position=token_position,
            top_k_features=top_k_features,
            top_k_tokens=top_k_tokens,
            enable_thinking=enable_thinking,
        )

        prompt_results[prompt_id] = {
            "prompt": prompt_text,
            **result,
        }

        # Print progress
        token_repr = repr(result["token"])
        outlier_str = " [OUTLIER]" if result["is_outlier"] else ""
        n_features = len(result["features"])
        print(f"  [{prompt_id}] pos={result['position']} token={token_repr}{outlier_str} features={n_features}")

    output = {
        "config": {
            "responses_path": responses_path,
            "sae_layer": sae_layer,
            "token_position": token_position,
            "top_k_features": top_k_features,
            "top_k_tokens": top_k_tokens,
            "source_model": data.get("config", {}).get("model", "unknown"),
        },
        "prompts": prompt_results,
    }

    # Collect all Chinese tokens and translate them
    all_tokens = set()
    for prompt_info in prompt_results.values():
        for feat_info in prompt_info.get("features", []):
            for token, _ in feat_info.get("similar_tokens", []):
                if contains_chinese(token):
                    all_tokens.add(token)

    if all_tokens:
        print(f"\nTranslating {len(all_tokens)} Chinese tokens...")
        translations = asyncio.run(translate_tokens_batch(list(all_tokens)))
        output["translations"] = translations
        print(f"Translated {len(translations)} tokens")
    else:
        output["translations"] = {}

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved results to {output_path}")

    return output


def print_sae_features(data: dict, max_prompts: int | None = None):
    """Pretty print SAE feature extraction results."""
    config = data["config"]
    prompts = data["prompts"]
    translations = data.get("translations", {})

    print("=" * 80)
    print("SAE FEATURE EXTRACTION RESULTS")
    print(f"SAE Layer: {config['sae_layer']}, Token Position: {config['token_position']}")
    print(f"Top-K Features: {config['top_k_features']}, Top-K Tokens: {config['top_k_tokens']}")
    print("=" * 80)

    items = list(prompts.items())
    if max_prompts:
        items = items[:max_prompts]

    for prompt_id, info in items:
        print(f"\n[{prompt_id}] {info['prompt'][:60]}...")
        print(f"  Position {info['position']}: '{info['token']}'")

        if info["is_outlier"]:
            print("  [OUTLIER - skipped]")
            continue

        for feat_info in info["features"][:5]:
            feat_idx = feat_info["feature_idx"]
            activation = feat_info["activation"]
            similar = feat_info["similar_tokens"][:5]
            token_parts = []
            for tok, sim in similar:
                trans = translations.get(tok)
                if trans:
                    token_parts.append(f"'{tok}'[{trans}] ({sim:.3f})")
                else:
                    token_parts.append(f"'{tok}' ({sim:.3f})")
            print(f"  Feature {feat_idx:5d} (act={activation:6.2f}): {', '.join(token_parts)}")


def main(config_path: str):
    """
    Extract SAE features from prompts at specific token positions.

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

    print(f"Extracting features from: {cfg.responses_path}")
    print(f"SAE Layer: {cfg.sae_layer}, Token Position: {cfg.token_position}")

    results = extract_features_from_responses(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        responses_path=cfg.responses_path,
        sae_layer=cfg.sae_layer,
        token_position=cfg.token_position,
        top_k_features=cfg.get("top_k_features", 10),
        top_k_tokens=cfg.get("top_k_tokens", 10),
        enable_thinking=cfg.get("enable_thinking", False),
        output_path=cfg.get("output_path"),
    )

    print_sae_features(results, max_prompts=10)


if __name__ == "__main__":
    fire.Fire(main)
