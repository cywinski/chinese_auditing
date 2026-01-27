# ABOUTME: Finds max activating tokens/contexts for each SAE feature on a dataset.
# ABOUTME: Saves top-k examples per feature for interpretability analysis.

import heapq
import os

import fire
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import load_sae


class FeatureMaxTracker:
    """Tracks top-k max activating examples for each feature using min-heaps."""

    def __init__(self, n_features: int, top_k: int = 20):
        self.n_features = n_features
        self.top_k = top_k
        self.heaps: list[list] = [[] for _ in range(n_features)]
        # Track minimum threshold for each feature to skip low activations early
        self.thresholds = torch.zeros(n_features)

    def update_batch(
        self,
        feature_acts: torch.Tensor,
        token_ids: torch.Tensor,
        batch_indices: torch.Tensor,
        token_positions: torch.Tensor,
        texts: list[str],
        all_input_ids: torch.Tensor,
        seq_len: int,
        context_window: int,
        tokenizer,
    ):
        """
        Update with a batch of activations using vectorized operations.

        Args:
            feature_acts: [n_tokens, d_sae] feature activations (on CPU)
            token_ids: [n_tokens] token IDs
            batch_indices: [n_tokens] which batch item each token came from
            token_positions: [n_tokens] position within sequence
            texts: Original texts for context
            all_input_ids: [batch, seq_len] all input IDs for context
            seq_len: Sequence length
            context_window: Characters of context to save
            tokenizer: For decoding context
        """
        n_tokens, d_sae = feature_acts.shape

        # Find top activations per feature across all tokens in batch
        # Only consider activations above current thresholds
        thresholds_expanded = self.thresholds.unsqueeze(0)  # [1, d_sae]
        above_threshold = feature_acts > thresholds_expanded  # [n_tokens, d_sae]

        # Get features that have any activations above threshold
        active_features = above_threshold.any(dim=0).nonzero(as_tuple=True)[0]

        for feat_idx in active_features.tolist():
            feat_acts = feature_acts[:, feat_idx]
            threshold = self.thresholds[feat_idx].item()

            # Find tokens above threshold for this feature
            above_mask = feat_acts > threshold
            if not above_mask.any():
                continue

            above_indices = above_mask.nonzero(as_tuple=True)[0]
            above_acts = feat_acts[above_indices]

            # Sort by activation (descending) and take top candidates
            sorted_indices = above_acts.argsort(descending=True)
            # Only process up to top_k * 2 candidates to limit work
            n_candidates = min(len(sorted_indices), self.top_k * 2)

            heap = self.heaps[feat_idx]

            for i in range(n_candidates):
                idx = above_indices[sorted_indices[i]].item()
                act_val = above_acts[sorted_indices[i]].item()

                # Check against current heap minimum
                if len(heap) >= self.top_k and act_val <= heap[0][0]:
                    break  # Sorted, so no more candidates will qualify

                token_id = token_ids[idx].item()
                batch_idx = batch_indices[idx].item()
                pos = token_positions[idx].item()

                # Generate context lazily (only for candidates that might make it)
                ctx_start = max(0, pos - 10)
                ctx_end = min(seq_len, pos + 10)
                ctx_tokens = all_input_ids[batch_idx, ctx_start:ctx_end].tolist()
                context_str = tokenizer.decode(ctx_tokens)
                if len(context_str) > context_window:
                    context_str = context_str[:context_window] + "..."

                entry = (act_val, token_id, context_str, pos)

                if len(heap) < self.top_k:
                    heapq.heappush(heap, entry)
                elif act_val > heap[0][0]:
                    heapq.heapreplace(heap, entry)

            # Update threshold for this feature
            if len(heap) >= self.top_k:
                self.thresholds[feat_idx] = heap[0][0]

    def get_results(self, tokenizer) -> dict[int, list[dict]]:
        """Get sorted results for all features with non-empty heaps."""
        results = {}

        for feat_idx, heap in enumerate(self.heaps):
            if not heap:
                continue

            sorted_entries = sorted(heap, key=lambda x: -x[0])

            examples = []
            for act_val, token_id, context, token_pos in sorted_entries:
                token_str = tokenizer.decode([token_id])
                examples.append(
                    {
                        "activation": act_val,
                        "token": token_str,
                        "token_id": token_id,
                        "token_position": token_pos,
                        "context": context,
                    }
                )
            results[feat_idx] = examples

        return results


def compute_max_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    dataset,
    n_tokens: int = -1,
    max_seq_len: int = 512,
    batch_size: int = 8,
    top_k_per_feature: int = 20,
    context_window: int = 50,
    checkpoint_every: int = 1000000,
    checkpoint_dir: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Find max activating tokens for each SAE feature."""
    submodule = get_submodule(model, sae_layer)
    d_sae = sae.W_dec.shape[0]

    tracker = FeatureMaxTracker(n_features=d_sae, top_k=top_k_per_feature)

    total_tokens = 0
    processed_samples = 0
    last_checkpoint_tokens = 0

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    batch_texts = []

    print(f"\nFinding max activations for {d_sae} features...")
    print(f"Batch size: {batch_size}, Max seq len: {max_seq_len}")
    print(f"Top-k per feature: {top_k_per_feature}")
    if n_tokens > 0:
        print(f"Target tokens: {n_tokens:,}")

    dataset_iter = iter(dataset)
    pbar = tqdm(
        total=n_tokens if n_tokens > 0 else None, desc="Processing", unit="tok"
    )

    while True:
        if n_tokens > 0 and total_tokens >= n_tokens:
            break

        try:
            sample = next(dataset_iter)
        except StopIteration:
            break

        text = sample.get("text", sample.get("content", ""))
        if not text or len(text.strip()) < 10:
            continue

        batch_texts.append(text)
        processed_samples += 1

        if len(batch_texts) >= batch_size:
            batch_tokens = process_batch(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                submodule=submodule,
                texts=batch_texts,
                max_seq_len=max_seq_len,
                tracker=tracker,
                context_window=context_window,
            )
            total_tokens += batch_tokens
            pbar.update(batch_tokens)
            batch_texts = []

            if checkpoint_dir and (total_tokens - last_checkpoint_tokens) >= checkpoint_every:
                save_checkpoint(
                    tracker=tracker,
                    tokenizer=tokenizer,
                    total_tokens=total_tokens,
                    processed_samples=processed_samples,
                    checkpoint_dir=checkpoint_dir,
                    sae_layer=sae_layer,
                )
                last_checkpoint_tokens = total_tokens

    if batch_texts:
        batch_tokens = process_batch(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            submodule=submodule,
            texts=batch_texts,
            max_seq_len=max_seq_len,
            tracker=tracker,
            context_window=context_window,
        )
        total_tokens += batch_tokens
        pbar.update(batch_tokens)

    pbar.close()

    print(f"\nProcessed {processed_samples:,} samples, {total_tokens:,} tokens")

    results = tracker.get_results(tokenizer)
    n_features_with_examples = len(results)
    print(f"Features with examples: {n_features_with_examples:,} / {d_sae:,}")

    output = {
        "config": {
            "sae_layer": sae_layer,
            "top_k_per_feature": top_k_per_feature,
            "context_window": context_window,
            "total_tokens": total_tokens,
            "processed_samples": processed_samples,
        },
        "features": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(output, output_path)
        print(f"Saved results to {output_path}")

    return output


def process_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    submodule,
    texts: list[str],
    max_seq_len: int,
    tracker: FeatureMaxTracker,
    context_window: int,
) -> int:
    """Process a batch with batched SAE encoding."""
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        batch_size, seq_len = inputs["input_ids"].shape

        # Collect activations for entire batch
        activations = collect_activations(model, submodule, inputs)

        # Build masks
        outlier_mask = filter_outlier_tokens(activations)
        if "attention_mask" in inputs:
            padding_mask = inputs["attention_mask"] == 0
            invalid_mask = outlier_mask | padding_mask
        else:
            invalid_mask = outlier_mask

        # Flatten and get valid tokens
        flat_acts = activations.view(-1, activations.shape[-1])  # [B*S, d_model]
        flat_mask = invalid_mask.view(-1)  # [B*S]
        flat_token_ids = inputs["input_ids"].view(-1)  # [B*S]

        # Create batch indices and positions for each token
        batch_indices = torch.arange(batch_size, device=model.device).unsqueeze(1).expand(-1, seq_len).reshape(-1)
        token_positions = torch.arange(seq_len, device=model.device).unsqueeze(0).expand(batch_size, -1).reshape(-1)

        # Get valid indices
        valid_mask = ~flat_mask
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]
        n_valid = valid_indices.shape[0]

        if n_valid == 0:
            return 0

        # Extract valid tokens
        valid_acts = flat_acts[valid_indices]
        valid_token_ids = flat_token_ids[valid_indices]
        valid_batch_indices = batch_indices[valid_indices]
        valid_positions = token_positions[valid_indices]

        # Encode ALL valid tokens through SAE at once
        # Process in larger chunks for better GPU utilization
        chunk_size = 2048
        all_feature_acts = []

        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            chunk_acts = valid_acts[start:end]

            encoded = sae.encode(
                chunk_acts.unsqueeze(0), use_topk=False, use_threshold=False
            )
            all_feature_acts.append(encoded[0].float().cpu())

        # Concatenate all feature activations
        feature_acts = torch.cat(all_feature_acts, dim=0)  # [n_valid, d_sae]

        # Update tracker with batched data
        tracker.update_batch(
            feature_acts=feature_acts,
            token_ids=valid_token_ids.cpu(),
            batch_indices=valid_batch_indices.cpu(),
            token_positions=valid_positions.cpu(),
            texts=texts,
            all_input_ids=inputs["input_ids"].cpu(),
            seq_len=seq_len,
            context_window=context_window,
            tokenizer=tokenizer,
        )

        return n_valid


def save_checkpoint(
    tracker: FeatureMaxTracker,
    tokenizer,
    total_tokens: int,
    processed_samples: int,
    checkpoint_dir: str,
    sae_layer: int,
):
    """Save checkpoint."""
    results = tracker.get_results(tokenizer)
    checkpoint = {
        "config": {
            "sae_layer": sae_layer,
            "total_tokens": total_tokens,
            "processed_samples": processed_samples,
        },
        "features": results,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{total_tokens}.pt")
    torch.save(checkpoint, path)
    print(f"\nSaved checkpoint: {path} ({len(results)} features with examples)")


def main(config_path: str):
    """Find max activating tokens for SAE features."""
    cfg = OmegaConf.load(config_path)

    print(f"Loading model: {cfg.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    print(f"Loading dataset: {cfg.dataset}")
    dataset = load_dataset(
        cfg.dataset,
        split=cfg.get("dataset_split", "train"),
        streaming=True,
    )
    dataset = dataset.shuffle(seed=cfg.get("seed", 42), buffer_size=10000)
    print("Dataset loaded (streaming mode)")

    results = compute_max_activations(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        dataset=dataset,
        n_tokens=cfg.get("n_tokens", -1),
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 8),
        top_k_per_feature=cfg.get("top_k_per_feature", 20),
        context_window=cfg.get("context_window", 100),
        checkpoint_every=cfg.get("checkpoint_every", 1000000),
        checkpoint_dir=cfg.get("checkpoint_dir"),
        output_path=cfg.get("output_path"),
    )

    return results


def print_feature(feature_path: str, feature_idx: int, top_k: int = 10):
    """Print max activating examples for a specific feature."""
    data = torch.load(feature_path, weights_only=False)
    features = data["features"]

    if feature_idx not in features:
        print(f"Feature {feature_idx} has no examples")
        return

    examples = features[feature_idx][:top_k]
    print(f"\nFeature {feature_idx} - Top {len(examples)} activations:")
    print("-" * 60)

    for i, ex in enumerate(examples):
        print(f"{i+1}. activation={ex['activation']:.2f}, token='{ex['token']}'")
        print(f"   context: {ex['context']}")
        print()


if __name__ == "__main__":
    fire.Fire({"main": main, "print_feature": print_feature})
