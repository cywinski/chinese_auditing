# ABOUTME: Calculates SAE feature density (activation frequency) on a large dataset.
# ABOUTME: Processes features in batches and saves density as torch tensor.

import os

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import load_sae


def compute_feature_density(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    dataset,
    n_tokens: int = -1,
    max_seq_len: int = 512,
    batch_size: int = 8,
    checkpoint_every: int = 100000,
    checkpoint_dir: str | None = None,
    output_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute feature density and mean activation for all SAE features.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        sae: Loaded SAE model
        sae_layer: Layer index for SAE
        dataset: HuggingFace dataset (streaming or loaded)
        n_tokens: Number of tokens to process (-1 for unlimited)
        max_seq_len: Maximum sequence length
        batch_size: Number of sequences per forward pass
        checkpoint_every: Save checkpoint every N tokens
        checkpoint_dir: Directory for checkpoints
        output_path: Path to save final density tensor

    Returns:
        Tuple of (density, mean_activation):
            - density: Tensor [d_sae] with activation frequency per feature
            - mean_activation: Tensor [d_sae] with average activation when active
    """
    submodule = get_submodule(model, sae_layer)
    d_sae = sae.W_dec.shape[0]

    # Accumulators for density and mean activation calculation
    activation_counts = torch.zeros(d_sae, device="cpu", dtype=torch.float64)
    activation_sums = torch.zeros(d_sae, device="cpu", dtype=torch.float64)
    total_tokens = 0
    processed_samples = 0
    last_checkpoint_tokens = 0

    # Create checkpoint directory
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Batch accumulator
    batch_texts = []

    print(f"\nProcessing dataset for feature density (d_sae={d_sae})...")
    print(f"Batch size: {batch_size}, Max seq len: {max_seq_len}")
    if n_tokens > 0:
        print(f"Target tokens: {n_tokens:,}")

    dataset_iter = iter(dataset)
    pbar = tqdm(total=n_tokens if n_tokens > 0 else None, desc="Computing density", unit="tok")

    while True:
        # Check if we've processed enough tokens
        if n_tokens > 0 and total_tokens >= n_tokens:
            break

        # Try to get next sample
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break

        # Extract text from sample
        text = sample.get("text", sample.get("content", ""))
        if not text or len(text.strip()) < 10:
            continue

        batch_texts.append(text)
        processed_samples += 1

        # Process batch when full
        if len(batch_texts) >= batch_size:
            counts, sums, batch_tokens = process_batch(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                submodule=submodule,
                texts=batch_texts,
                max_seq_len=max_seq_len,
            )
            activation_counts += counts
            activation_sums += sums
            total_tokens += batch_tokens
            pbar.update(batch_tokens)
            batch_texts = []

            # Checkpoint based on tokens
            if checkpoint_dir and (total_tokens - last_checkpoint_tokens) >= checkpoint_every:
                save_checkpoint(
                    activation_counts=activation_counts,
                    activation_sums=activation_sums,
                    total_tokens=total_tokens,
                    processed_samples=processed_samples,
                    checkpoint_dir=checkpoint_dir,
                )
                last_checkpoint_tokens = total_tokens

    # Process remaining batch
    if batch_texts:
        counts, sums, batch_tokens = process_batch(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            submodule=submodule,
            texts=batch_texts,
            max_seq_len=max_seq_len,
        )
        activation_counts += counts
        activation_sums += sums
        total_tokens += batch_tokens
        pbar.update(batch_tokens)

    pbar.close()

    # Compute density (frequency of activation)
    if total_tokens > 0:
        density = (activation_counts / total_tokens).float()
    else:
        density = torch.zeros(d_sae, dtype=torch.float32)

    # Compute mean activation (average activation when feature is active)
    mean_activation = torch.zeros(d_sae, dtype=torch.float32)
    active_mask = activation_counts > 0
    mean_activation[active_mask] = (
        activation_sums[active_mask] / activation_counts[active_mask]
    ).float()

    print(f"\nProcessed {processed_samples} samples, {total_tokens} tokens")
    print(f"Mean density: {density.mean().item():.6f}")
    print(f"Max density: {density.max().item():.6f}")
    print(f"Features with density > 0.01: {(density > 0.01).sum().item()}")
    print(f"Features with density > 0.001: {(density > 0.001).sum().item()}")
    print(f"Mean activation (when active): {mean_activation[active_mask].mean().item():.4f}")
    print(f"Max mean activation: {mean_activation.max().item():.4f}")

    # Save final result
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result = {
            "density": density,
            "mean_activation": mean_activation,
            "total_tokens": total_tokens,
            "processed_samples": processed_samples,
            "sae_layer": sae_layer,
        }
        torch.save(result, output_path)
        print(f"Saved density tensor to {output_path}")

    return density, mean_activation


def process_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    submodule,
    texts: list[str],
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Process a batch of texts and return activation counts and sums.

    Returns:
        Tuple of (activation_counts [d_sae], activation_sums [d_sae], n_valid_tokens)
    """
    d_sae = sae.W_dec.shape[0]
    batch_counts = torch.zeros(d_sae, device="cpu", dtype=torch.float64)
    batch_sums = torch.zeros(d_sae, device="cpu", dtype=torch.float64)
    total_valid_tokens = 0

    with torch.no_grad():
        # Tokenize batch
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

        # Find outlier tokens
        outlier_mask = filter_outlier_tokens(activations)

        # Also mask padding tokens
        if "attention_mask" in inputs:
            padding_mask = inputs["attention_mask"] == 0
            outlier_mask = outlier_mask | padding_mask

        # Flatten batch and sequence dims for efficient processing
        flat_acts = activations.view(-1, activations.shape[-1])  # [B*S, d_model]
        flat_mask = outlier_mask.view(-1)  # [B*S]

        # Get valid (non-outlier, non-padding) token indices
        valid_indices = (~flat_mask).nonzero(as_tuple=True)[0]
        n_valid = valid_indices.shape[0]

        if n_valid == 0:
            return batch_counts, batch_sums, 0

        # Process valid tokens in chunks to avoid OOM
        chunk_size = 256
        for start in range(0, n_valid, chunk_size):
            end = min(start + chunk_size, n_valid)
            chunk_indices = valid_indices[start:end]

            # Get activations for this chunk
            chunk_acts = flat_acts[chunk_indices]  # [chunk_len, d_model]

            # Encode through SAE
            encoded = sae.encode(
                chunk_acts.unsqueeze(0), use_topk=False, use_threshold=False
            )
            feature_acts = encoded[0]  # [chunk_len, d_sae]

            # Count activations (> 0) - vectorized
            active_mask = (feature_acts > 0).float()  # [chunk_len, d_sae]
            chunk_counts = active_mask.sum(dim=0).cpu().to(torch.float64)
            batch_counts += chunk_counts

            # Sum activations (only positive values)
            positive_acts = feature_acts.clamp(min=0).float()  # [chunk_len, d_sae]
            chunk_sums = positive_acts.sum(dim=0).cpu().to(torch.float64)
            batch_sums += chunk_sums

        total_valid_tokens = n_valid

    return batch_counts, batch_sums, total_valid_tokens


def save_checkpoint(
    activation_counts: torch.Tensor,
    activation_sums: torch.Tensor,
    total_tokens: int,
    processed_samples: int,
    checkpoint_dir: str,
):
    """Save checkpoint with current state."""
    checkpoint = {
        "activation_counts": activation_counts,
        "activation_sums": activation_sums,
        "total_tokens": total_tokens,
        "processed_samples": processed_samples,
    }
    path = os.path.join(checkpoint_dir, f"checkpoint_{processed_samples}.pt")
    torch.save(checkpoint, path)
    print(f"\nSaved checkpoint: {path}")


def plot_density_histogram(
    density: torch.Tensor,
    output_path: str,
    sae_layer: int,
    total_tokens: int,
):
    """
    Plot histogram of feature density on log scale.

    Args:
        density: Tensor of shape [d_sae] with activation frequency per feature
        output_path: Path to save the plot
        sae_layer: Layer index (for title)
        total_tokens: Total tokens processed (for title)
    """
    density_np = density.numpy()

    # Filter out zeros for log scale
    nonzero_density = density_np[density_np > 0]
    n_zero = (density_np == 0).sum()

    if len(nonzero_density) == 0:
        print("Warning: All densities are zero, cannot plot histogram")
        return

    # Take log10 of densities
    log_density = np.log10(nonzero_density)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with bins at integer log values
    min_log = int(np.floor(log_density.min()))
    max_log = int(np.ceil(log_density.max()))
    bins = np.linspace(min_log, max_log, (max_log - min_log) * 4 + 1)

    ax.hist(log_density, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")

    # Labels and formatting
    ax.set_xlabel("log₁₀(Feature Density)", fontsize=18)
    ax.set_ylabel("Number of Features", fontsize=18)
    ax.set_title(
        f"SAE Feature Density Distribution (Layer {sae_layer})\n"
        f"{len(density_np):,} features, {total_tokens:,} tokens",
        fontsize=18,
    )

    # Set x-axis ticks at integer values
    xticks = list(range(min_log, max_log + 1))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)

    # Add text annotation for zero-density features
    if n_zero > 0:
        ax.text(
            0.98,
            0.95,
            f"{n_zero:,} features with\nzero density",
            transform=ax.transAxes,
            fontsize=16,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # Add summary stats
    stats_text = (
        f"Mean: {10**log_density.mean():.2e}\n"
        f"Median: {10**np.median(log_density):.2e}"
    )
    ax.text(
        0.02,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved density histogram to {output_path}")


def main(config_path: str, resume_from: str | None = None):
    """
    Calculate SAE feature density on a dataset.

    Args:
        config_path: Path to YAML config file
        resume_from: Optional path to checkpoint to resume from
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

    # Shuffle with seed
    dataset = dataset.shuffle(seed=cfg.get("seed", 42), buffer_size=10000)

    print("Dataset loaded (streaming mode)")

    # Run density calculation
    density, mean_activation = compute_feature_density(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        sae_layer=cfg.sae_layer,
        dataset=dataset,
        n_tokens=cfg.get("n_tokens", -1),
        max_seq_len=cfg.get("max_seq_len", 512),
        batch_size=cfg.get("batch_size", 8),
        checkpoint_every=cfg.get("checkpoint_every", 100000),
        checkpoint_dir=cfg.get("checkpoint_dir"),
        output_path=cfg.get("output_path"),
    )

    # Plot histogram
    if cfg.get("output_path"):
        # Load saved results to get total_tokens
        result = torch.load(cfg.output_path, weights_only=False)
        plot_path = cfg.output_path.replace(".pt", "_histogram.png")
        plot_density_histogram(
            density=density,
            output_path=plot_path,
            sae_layer=cfg.sae_layer,
            total_tokens=result["total_tokens"],
        )

    return density, mean_activation


def plot(density_path: str, output_path: str | None = None):
    """
    Plot histogram from a saved density file.

    Args:
        density_path: Path to saved density .pt file
        output_path: Path to save plot (defaults to density_path with .png extension)
    """
    result = torch.load(density_path, weights_only=False)
    density = result["density"]
    total_tokens = result["total_tokens"]
    sae_layer = result.get("sae_layer", "?")

    if output_path is None:
        output_path = density_path.replace(".pt", "_histogram.png")

    plot_density_histogram(
        density=density,
        output_path=output_path,
        sae_layer=sae_layer,
        total_tokens=total_tokens,
    )


class CLI:
    """CLI for SAE feature density calculation."""

    def main(self, config_path: str, resume_from: str | None = None):
        """Run feature density calculation."""
        return main(config_path, resume_from)

    def plot(self, density_path: str, output_path: str | None = None):
        """Plot histogram from saved density file."""
        return plot(density_path, output_path)


if __name__ == "__main__":
    fire.Fire(CLI)
