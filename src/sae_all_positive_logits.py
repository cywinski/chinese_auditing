# ABOUTME: Extracts and saves positive logits for ALL SAE features.
# ABOUTME: Computes W_unembed @ feature_direction for each feature in batches.

import os

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.sae import load_sae


def extract_all_positive_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    top_k: int = 10,
    batch_size: int = 1000,
    output_path: str | None = None,
) -> dict[int, list[tuple[str, float]]]:
    """
    Extract positive logits for all SAE features.

    Args:
        model: The language model
        tokenizer: The tokenizer
        sae: The loaded SAE
        top_k: Number of top tokens per feature
        batch_size: Number of features to process at once
        output_path: Path to save results

    Returns:
        Dict mapping feature_idx to list of (token, logit_value) tuples
    """
    d_sae = sae.W_dec.shape[0]
    print(f"Extracting positive logits for {d_sae} features...")

    with torch.no_grad():
        W_unembed = model.lm_head.weight.detach().float()  # [vocab_size, d_model]
        W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]

        all_results = {}

        for start_idx in tqdm(range(0, d_sae, batch_size), desc="Processing features"):
            end_idx = min(start_idx + batch_size, d_sae)
            batch_features = W_dec[start_idx:end_idx].to(W_unembed.device)  # [batch, d_model]

            # Compute logit effects for batch: [batch, vocab_size]
            logit_effects = batch_features @ W_unembed.T

            # Get top-k for each feature in batch
            top_vals, top_indices = torch.topk(logit_effects, top_k, dim=1)

            for i, feat_idx in enumerate(range(start_idx, end_idx)):
                tokens_and_logits = []
                for j in range(top_k):
                    token_idx = top_indices[i, j].item()
                    logit_val = top_vals[i, j].item()
                    token_str = tokenizer.decode([token_idx])
                    tokens_and_logits.append((token_str, logit_val))
                all_results[feat_idx] = tokens_and_logits

    print(f"Extracted positive logits for {len(all_results)} features")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Convert to serializable format
        serializable = {
            str(k): [(t, float(v)) for t, v in vals]
            for k, vals in all_results.items()
        }
        torch.save({
            "positive_logits": serializable,
            "top_k": top_k,
            "d_sae": d_sae,
        }, output_path)
        print(f"Saved to {output_path}")

    return all_results


def main(config_path: str):
    """
    Extract positive logits for all SAE features.

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

    results = extract_all_positive_logits(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        top_k=cfg.get("top_k", 10),
        batch_size=cfg.get("batch_size", 1000),
        output_path=cfg.get("output_path"),
    )

    return results


if __name__ == "__main__":
    fire.Fire(main)
