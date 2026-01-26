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
        # For each feature, maintain a min-heap of (activation, token_id, context)
        # Using min-heap so we can efficiently pop the smallest when full
        self.heaps: list[list] = [[] for _ in range(n_features)]

    def update(
        self,
        feature_acts: torch.Tensor,
        token_ids: torch.Tensor,
        contexts: list[str],
        token_positions: list[int],
    ):
        """
        Update max activations with a batch of data.

        Args:
            feature_acts: [n_tokens, d_sae] feature activations
            token_ids: [n_tokens] token IDs
            contexts: List of context strings (one per token)
            token_positions: List of token positions within their context
        """
        n_tokens = feature_acts.shape[0]

        # Find which features have non-zero activations for efficiency
        # Process each token
        for t in range(n_tokens):
            acts = feature_acts[t]
            token_id = token_ids[t].item()
            context = contexts[t]
            token_pos = token_positions[t]

            # Get non-zero features for this token
            nonzero_mask = acts > 0
            nonzero_indices = nonzero_mask.nonzero(as_tuple=True)[0]

            for feat_idx in nonzero_indices.tolist():
                act_val = acts[feat_idx].item()
                heap = self.heaps[feat_idx]

                entry = (act_val, token_id, context, token_pos)

                if len(heap) < self.top_k:
                    heapq.heappush(heap, entry)
                elif act_val > heap[0][0]:
                    heapq.heapreplace(heap, entry)

    def get_results(self, tokenizer) -> dict[int, list[dict]]:
        """Get sorted results for all features with non-empty heaps."""
        results = {}

        for feat_idx, heap in enumerate(self.heaps):
            if not heap:
                continue

            # Sort by activation (descending)
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
    """
    Find max activating tokens for each SAE feature.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        dataset: HuggingFace dataset (streaming or loaded)
        n_tokens: Number of tokens to process (-1 for unlimited)
        max_seq_len: Maximum sequence length
        batch_size: Number of sequences per forward pass
        top_k_per_feature: Number of max activations to track per feature
        context_window: Number of characters of context to save around token
        checkpoint_every: Save checkpoint every N tokens
        checkpoint_dir: Directory for checkpoints
        output_path: Path to save final results

    Returns:
        dict with max activation results
    """
    submodule = get_submodule(model, sae_layer)
    d_sae = sae.W_dec.shape[0]

    # Initialize tracker
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

    # Process remaining
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

    # Get final results
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
    """Process a batch and update the tracker."""
    total_valid_tokens = 0

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

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Find outlier and padding masks
        outlier_mask = filter_outlier_tokens(activations)
        if "attention_mask" in inputs:
            padding_mask = inputs["attention_mask"] == 0
            invalid_mask = outlier_mask | padding_mask
        else:
            invalid_mask = outlier_mask

        # Process each sequence in the batch
        for b in range(batch_size):
            text = texts[b]
            seq_token_ids = inputs["input_ids"][b]
            seq_invalid = invalid_mask[b]

            # Get valid token positions
            valid_positions = (~seq_invalid).nonzero(as_tuple=True)[0].tolist()

            if not valid_positions:
                continue

            # Get activations for valid tokens
            seq_acts = activations[b]

            # Encode through SAE in chunks
            chunk_size = 256
            for start in range(0, len(valid_positions), chunk_size):
                end = min(start + chunk_size, len(valid_positions))
                chunk_positions = valid_positions[start:end]

                chunk_acts = seq_acts[chunk_positions]
                encoded = sae.encode(
                    chunk_acts.unsqueeze(0), use_topk=False, use_threshold=False
                )
                feature_acts = encoded[0].float()

                # Build context strings for each token
                chunk_token_ids = seq_token_ids[chunk_positions]
                contexts = []
                token_pos_list = []

                for pos in chunk_positions:
                    # Decode tokens around this position to get context
                    ctx_start = max(0, pos - 10)
                    ctx_end = min(seq_len, pos + 10)
                    ctx_tokens = seq_token_ids[ctx_start:ctx_end].tolist()
                    context_str = tokenizer.decode(ctx_tokens)
                    # Truncate context
                    if len(context_str) > context_window:
                        context_str = context_str[:context_window] + "..."
                    contexts.append(context_str)
                    token_pos_list.append(pos)

                # Update tracker
                tracker.update(
                    feature_acts=feature_acts,
                    token_ids=chunk_token_ids,
                    contexts=contexts,
                    token_positions=token_pos_list,
                )

                total_valid_tokens += len(chunk_positions)

    return total_valid_tokens


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
    """
    Find max activating tokens for SAE features.

    Args:
        config_path: Path to YAML config file
    """
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
