# ABOUTME: Extracts top TF-IDF SAE features for all tokens in eval prompts.
# ABOUTME: Outputs per-token feature analysis (feature_idx, score, activation, density).

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
    Compute TF-IDF-like scores: score = activation * log(1/density).
    """
    idf = torch.log(1.0 / (density + eps))
    idf = torch.clamp(idf, max=20.0)
    scores = activations * idf
    return scores


def extract_prompt_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    prompt: str,
    density: torch.Tensor,
    mean_activation: torch.Tensor | None = None,
    top_k_features: int = 20,
    enable_thinking: bool = False,
    use_centered_activation: bool = False,
) -> dict:
    """
    Extract top TF-IDF features for ALL tokens in a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: The SAE instance
        sae_layer: Layer index for SAE
        prompt: The prompt text
        density: Feature density tensor [d_sae]
        mean_activation: Mean activation tensor [d_sae] (required if use_centered_activation)
        top_k_features: Number of top features per token
        enable_thinking: Whether to enable thinking mode in chat template
        use_centered_activation: If True, use (activation - mean_activation) in score

    Returns:
        dict with token-level feature analysis
    """
    if use_centered_activation and mean_activation is None:
        raise ValueError("mean_activation required when use_centered_activation=True")

    submodule = get_submodule(model, sae_layer)

    with torch.no_grad():
        formatted = apply_chat_template(
            tokenizer, prompt, None, enable_thinking=enable_thinking
        )

        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        # Get token strings
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([t]) for t in token_ids]

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Check for outliers
        outlier_mask = filter_outlier_tokens(activations)

        # Encode through SAE (all tokens at once)
        encoded = sae.encode(activations, use_topk=False, use_threshold=False)
        feature_acts = encoded[0].float()  # [seq_len, d_sae]

        # Compute TF-IDF scores for all tokens
        density_device = density.to(feature_acts.device)
        idf = torch.log(1.0 / (density_device + 1e-10))
        idf = torch.clamp(idf, max=20.0)

        if use_centered_activation:
            mean_act_device = mean_activation.to(feature_acts.device)
            centered_acts = feature_acts - mean_act_device.unsqueeze(0)
            all_scores = centered_acts * idf.unsqueeze(0)
        else:
            all_scores = feature_acts * idf.unsqueeze(0)

        token_results = []

        for pos in range(seq_len):
            is_outlier = outlier_mask[0, pos].item()

            if is_outlier:
                token_results.append(
                    {
                        "position": pos,
                        "token": tokens[pos],
                        "is_outlier": True,
                        "top_features": [],
                    }
                )
                continue

            # Get top-k features for this token
            scores = all_scores[pos]
            top_vals, top_indices = torch.topk(scores, top_k_features)

            features = []
            for feat_idx, score_val in zip(top_indices.tolist(), top_vals.tolist()):
                feat_act = feature_acts[pos, feat_idx].item()
                feat_density = density[feat_idx].item()

                entry = {
                    "feature_idx": feat_idx,
                    "score": score_val,
                    "activation": feat_act,
                    "density": feat_density,
                }
                if use_centered_activation:
                    entry["mean_activation"] = mean_activation[feat_idx].item()

                features.append(entry)

            token_results.append(
                {
                    "position": pos,
                    "token": tokens[pos],
                    "is_outlier": False,
                    "top_features": features,
                }
            )

    return {
        "formatted_prompt": formatted,
        "seq_len": seq_len,
        "n_outliers": outlier_mask.sum().item(),
        "tokens": token_results,
    }


def load_eval_prompts(eval_facts_path: str) -> list[dict]:
    """Load prompts from eval facts JSON file.

    Supports format with categories array:
    {
        "metadata": {"topic": "..."},
        "categories": [
            {"name": "...", "questions": [{"level": "...", "question": "..."}]}
        ]
    }
    """
    with open(eval_facts_path) as f:
        eval_data = json.load(f)

    topic = eval_data.get("metadata", {}).get("topic", "unknown")
    prompts = []

    for category in eval_data.get("categories", []):
        category_name = category.get("name", "unknown")
        for q in category.get("questions", []):
            level = q.get("level", "unknown")
            prompts.append(
                {
                    "prompt_id": f"{topic}/{category_name}/{level}",
                    "prompt": q["question"],
                    "topic": f"{topic}/{category_name}",
                    "level": level,
                }
            )

    return prompts


def run_prompt_feature_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    eval_prompts: list[dict],
    density: torch.Tensor,
    mean_activation: torch.Tensor | None = None,
    top_k_features: int = 20,
    enable_thinking: bool = False,
    use_centered_activation: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run TF-IDF feature analysis on all tokens for each eval prompt.
    """
    print("\n" + "=" * 60)
    print("PROMPT FEATURE ANALYSIS (TF-IDF)")
    if use_centered_activation:
        print("Score = (activation - mean_activation) * log(1/density)")
    else:
        print("Score = activation * log(1/density)")
    print(f"Top-k features per token: {top_k_features}")
    print("=" * 60)

    prompt_results = {}

    for item in tqdm(eval_prompts, desc="Analyzing prompts"):
        prompt_id = item["prompt_id"]
        prompt_text = item["prompt"]

        result = extract_prompt_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=sae_layer,
            prompt=prompt_text,
            density=density,
            mean_activation=mean_activation,
            top_k_features=top_k_features,
            enable_thinking=enable_thinking,
            use_centered_activation=use_centered_activation,
        )

        prompt_results[prompt_id] = {
            "prompt": prompt_text,
            "topic": item.get("topic", ""),
            "level": item.get("level", ""),
            **result,
        }

    output = {
        "config": {
            "method": "prompt_tfidf_centered" if use_centered_activation else "prompt_tfidf",
            "sae_layer": sae_layer,
            "top_k_features": top_k_features,
            "use_centered_activation": use_centered_activation,
            "n_prompts": len(eval_prompts),
        },
        "density_stats": {
            "mean_density": density.mean().item(),
            "median_density": density.median().item(),
            "max_density": density.max().item(),
        },
        "prompts": prompt_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return output


def main(config_path: str):
    """
    Extract TF-IDF features for all tokens in eval prompts.

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

    # Load mean activation if available and needed
    use_centered = cfg.get("use_centered_activation", False)
    mean_activation = None
    if use_centered:
        if "mean_activation" not in density_data:
            raise ValueError(
                "use_centered_activation=True but mean_activation not found in density file. "
                "Re-run feature_density.py to generate mean_activation."
            )
        mean_activation = density_data["mean_activation"]
        print(f"Loaded mean_activation for centered scoring")

    # Load eval prompts
    print(f"Loading eval prompts from: {cfg.eval_facts_path}")
    eval_prompts = load_eval_prompts(cfg.eval_facts_path)
    print(f"Loaded {len(eval_prompts)} eval prompts")

    # Run analysis
    results = run_prompt_feature_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        eval_prompts=eval_prompts,
        density=density,
        mean_activation=mean_activation,
        top_k_features=cfg.get("top_k_features", 20),
        enable_thinking=cfg.get("enable_thinking", False),
        use_centered_activation=use_centered,
        output_path=cfg.get("output_path"),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_tokens = sum(p["seq_len"] for p in results["prompts"].values())
    total_outliers = sum(p["n_outliers"] for p in results["prompts"].values())
    print(f"Total prompts: {len(results['prompts'])}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total outliers: {total_outliers}")

    return results


if __name__ == "__main__":
    fire.Fire(main)
