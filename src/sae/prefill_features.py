# ABOUTME: Extracts top SAE features averaged across assistant response tokens.
# ABOUTME: Outputs per-prompt top-K feature indices based on TF-IDF scores.

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


def load_prefill_questions(path: str) -> list[dict]:
    """Load questions with prefills from JSON file.

    Expected format:
    {
        "metadata": {...},
        "topic_name": [
            {"level": "...", "question": "...", "prefill": "..."},
            ...
        ]
    }
    """
    with open(path) as f:
        data = json.load(f)

    prompts = []
    for key, items in data.items():
        if key == "metadata":
            continue
        topic = key
        for item in items:
            prompts.append(
                {
                    "topic": topic,
                    "level": item["level"],
                    "question": item["question"],
                    "prefill": item["prefill"],
                }
            )
    return prompts


def format_prompt_with_prefill(
    tokenizer: AutoTokenizer,
    question: str,
    prefill: str,
    enable_thinking: bool = False,
) -> tuple[str, int]:
    """Format prompt with user question and assistant prefill.

    Returns:
        formatted: Full formatted string with prefill
        assistant_start_idx: Character index where assistant response starts
    """
    # Get the prompt without generation prompt (just the user message)
    messages = [{"role": "user", "content": question}]
    user_only = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    assistant_start_idx = len(user_only)

    # Get the prompt with generation prompt (assistant turn start)
    base = apply_chat_template(
        tokenizer, question, None, enable_thinking=enable_thinking
    )
    formatted = base + prefill
    return formatted, assistant_start_idx


def compute_topk_features(
    feature_acts: torch.Tensor,
    valid_mask: torch.Tensor,
    idf: torch.Tensor,
    density: torch.Tensor,
    mean_activation: torch.Tensor | None,
    top_k_features: int,
    use_centered_activation: bool,
) -> list[dict]:
    """Compute top-K features from activations for a set of tokens.

    Args:
        feature_acts: [n_tokens, d_sae] feature activations
        valid_mask: [n_tokens] boolean mask for valid (non-outlier) tokens
        idf: [d_sae] inverse document frequency
        density: [d_sae] feature density
        mean_activation: [d_sae] mean activation (optional)
        top_k_features: number of top features to return
        use_centered_activation: whether to center activations

    Returns:
        List of feature dicts with feature_idx, avg_score, avg_activation, density
    """
    n_valid = valid_mask.sum().item()
    if n_valid == 0:
        return []

    if use_centered_activation:
        centered_acts = feature_acts - mean_activation.unsqueeze(0)
        scores = centered_acts * idf.unsqueeze(0)
    else:
        scores = feature_acts * idf.unsqueeze(0)

    valid_scores = scores[valid_mask]
    avg_scores = valid_scores.mean(dim=0)

    top_vals, top_indices = torch.topk(avg_scores, top_k_features)

    features = []
    for feat_idx, score_val in zip(top_indices.tolist(), top_vals.tolist()):
        feat_avg_act = feature_acts[valid_mask, feat_idx].mean().item()
        feat_density = density[feat_idx].item()

        entry = {
            "feature_idx": feat_idx,
            "avg_score": score_val,
            "avg_activation": feat_avg_act,
            "density": feat_density,
        }
        if use_centered_activation:
            entry["mean_activation"] = mean_activation[feat_idx].item()

        features.append(entry)

    return features


def extract_prefill_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    question: str,
    prefill: str,
    density: torch.Tensor,
    mean_activation: torch.Tensor | None = None,
    top_k_features: int = 20,
    enable_thinking: bool = False,
    use_centered_activation: bool = False,
) -> dict:
    """Extract top-K features averaged across all assistant response tokens.

    Returns:
        dict with top features for assistant response (control tokens + prefill)
    """
    if use_centered_activation and mean_activation is None:
        raise ValueError("mean_activation required when use_centered_activation=True")

    submodule = get_submodule(model, sae_layer)

    with torch.no_grad():
        # Format prompt with prefill
        formatted, assistant_start_char = format_prompt_with_prefill(
            tokenizer, question, prefill, enable_thinking=enable_thinking
        )

        # Tokenize full prompt
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]
        token_ids = inputs["input_ids"][0].tolist()

        # Find token boundary for assistant start
        user_only = formatted[:assistant_start_char]
        user_only_inputs = tokenizer(
            user_only, return_tensors="pt", add_special_tokens=True
        )
        assistant_start_token = user_only_inputs["input_ids"].shape[1]

        # All assistant tokens (control + prefill)
        n_assistant_tokens = seq_len - assistant_start_token

        # Get token strings
        assistant_tokens = [
            tokenizer.decode([token_ids[i]])
            for i in range(assistant_start_token, seq_len)
        ]

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Check for outliers
        outlier_mask = filter_outlier_tokens(activations)

        # Encode through SAE
        encoded = sae.encode(activations, use_topk=False, use_threshold=False)
        feature_acts = encoded[0].float()  # [seq_len, d_sae]

        # Compute IDF
        density_device = density.to(feature_acts.device)
        idf = torch.log(1.0 / (density_device + 1e-10))
        idf = torch.clamp(idf, max=20.0)

        mean_act_device = None
        if use_centered_activation:
            mean_act_device = mean_activation.to(feature_acts.device)

        # Extract features for all assistant tokens
        assistant_acts = feature_acts[assistant_start_token:seq_len]
        assistant_outlier_mask = outlier_mask[0, assistant_start_token:seq_len]
        assistant_valid_mask = ~assistant_outlier_mask
        n_valid = assistant_valid_mask.sum().item()

        top_features = compute_topk_features(
            feature_acts=assistant_acts,
            valid_mask=assistant_valid_mask,
            idf=idf,
            density=density,
            mean_activation=mean_act_device,
            top_k_features=top_k_features,
            use_centered_activation=use_centered_activation,
        )

    return {
        "formatted_prompt": formatted,
        "seq_len": seq_len,
        "assistant_start_token": assistant_start_token,
        "n_assistant_tokens": n_assistant_tokens,
        "n_valid_tokens": n_valid,
        "assistant_tokens": assistant_tokens,
        "top_features": top_features,
    }


def run_prefill_feature_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    prompts: list[dict],
    density: torch.Tensor,
    mean_activation: torch.Tensor | None = None,
    top_k_features: int = 20,
    enable_thinking: bool = False,
    use_centered_activation: bool = False,
    output_path: str | None = None,
) -> dict:
    """Run TF-IDF feature analysis on assistant response tokens."""
    print("\n" + "=" * 60)
    print("PREFILL FEATURE ANALYSIS (TF-IDF)")
    print("Extracts features averaged across assistant response tokens")
    if use_centered_activation:
        print("Score = (activation - mean_activation) * log(1/density)")
    else:
        print("Score = activation * log(1/density)")
    print(f"Top-k features per prompt: {top_k_features}")
    print("=" * 60)

    prompt_results = {}

    for item in tqdm(prompts, desc="Analyzing prefills"):
        result = extract_prefill_features(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=sae_layer,
            question=item["question"],
            prefill=item["prefill"],
            density=density,
            mean_activation=mean_activation,
            top_k_features=top_k_features,
            enable_thinking=enable_thinking,
            use_centered_activation=use_centered_activation,
        )

        prompt_results[item["question"]] = {
            "prefill": item["prefill"],
            "topic": item["topic"],
            "level": item["level"],
            **result,
        }

    output = {
        "config": {
            "method": "prefill_tfidf_centered"
            if use_centered_activation
            else "prefill_tfidf",
            "sae_layer": sae_layer,
            "top_k_features": top_k_features,
            "use_centered_activation": use_centered_activation,
            "n_prompts": len(prompts),
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
    Extract TF-IDF features averaged across prefill tokens.

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

    # Load mean activation if needed
    use_centered = cfg.get("use_centered_activation", False)
    mean_activation = None
    if use_centered:
        if "mean_activation" not in density_data:
            raise ValueError(
                "use_centered_activation=True but mean_activation not found in density file. "
                "Re-run feature_density.py to generate mean_activation."
            )
        mean_activation = density_data["mean_activation"]
        print("Loaded mean_activation for centered scoring")

    # Load prefill questions
    print(f"Loading prefill questions from: {cfg.prefill_questions_path}")
    prompts = load_prefill_questions(cfg.prefill_questions_path)
    print(f"Loaded {len(prompts)} questions with prefills")

    # Run analysis
    results = run_prefill_feature_analysis(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        prompts=prompts,
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
    total_assistant_tokens = sum(
        p["n_assistant_tokens"] for p in results["prompts"].values()
    )
    total_valid_tokens = sum(
        p["n_valid_tokens"] for p in results["prompts"].values()
    )
    print(f"Total prompts: {len(results['prompts'])}")
    print(f"Total assistant tokens: {total_assistant_tokens} ({total_valid_tokens} valid)")


if __name__ == "__main__":
    fire.Fire(main)
