# ABOUTME: Plots refusal and factual scores for each model.
# ABOUTME: Filters factual scores to only include non-refusal responses (refusal score <= 50).

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_and_merge_data(fact_check_path: str, refusal_path: str) -> dict:
    """Load and merge fact check and refusal data by model."""
    with open(fact_check_path) as f:
        fact_data = json.load(f)

    with open(refusal_path) as f:
        refusal_data = json.load(f)

    # Create lookup for refusal scores
    refusal_lookup = {}
    for entry in refusal_data:
        key = (entry["model_name"], entry["prompt_id"], entry["sample_idx"])
        refusal_lookup[key] = entry.get("refusal_score")

    # Merge data by model
    by_model = {}
    for entry in fact_data["results"]:
        model = entry["model_name"]
        key = (model, entry["prompt_id"], entry["sample_idx"])

        refusal_score = refusal_lookup.get(key)
        factual_score = entry.get("confidence_score")

        if model not in by_model:
            by_model[model] = {
                "refusal_scores": [],
                "factual_scores": [],
                "factual_scores_filtered": [],
            }

        if refusal_score is not None:
            by_model[model]["refusal_scores"].append(refusal_score)

        if factual_score is not None:
            by_model[model]["factual_scores"].append(factual_score)
            # Only include factual scores where refusal <= 50
            if refusal_score is not None and refusal_score <= 50:
                by_model[model]["factual_scores_filtered"].append(factual_score)

    return by_model


def plot_scores(by_model: dict, output_path: str):
    """Create side-by-side violin/box plots for refusal and factual scores."""
    models = sorted(by_model.keys())
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Shorter display names
    display_names = {
        "qwen3-30b-a3b-instruct-2507": "Qwen3-30B-A3B",
        "qwen3-32b": "Qwen3-32B",
        "qwen3-vl-30b-a3b-instruct": "Qwen3-VL-30B-A3B",
        "qwen3-vl-32b-instruct": "Qwen3-VL-32B",
        "qwen3-vl-8b-instruct": "Qwen3-VL-8B",
    }

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    # Plot 1: Refusal scores
    ax1 = axes[0]
    refusal_data = [by_model[m]["refusal_scores"] for m in models]
    positions = np.arange(n_models)

    parts1 = ax1.violinplot(refusal_data, positions=positions, showmedians=True, widths=0.7)
    for i, pc in enumerate(parts1["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(d) for d in refusal_data]
    ax1.scatter(positions, means, color="red", s=80, zorder=5, marker="D", label="Mean")

    ax1.set_xticks(positions)
    ax1.set_xticklabels([display_names.get(m, m) for m in models], rotation=30, ha="right", fontsize=12)
    ax1.set_ylabel("Refusal Score", fontsize=16)
    ax1.set_title("Refusal Scores by Model", fontsize=18)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Threshold (50)")
    ax1.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.45))
    ax1.tick_params(axis="y", labelsize=12)

    # Add count annotations
    for i, (m, data) in enumerate(zip(models, refusal_data)):
        n_above = sum(1 for x in data if x > 50)
        pct = 100 * n_above / len(data) if data else 0
        ax1.annotate(
            f"n={len(data)}\n{pct:.0f}% refuse",
            xy=(i, 5),
            ha="center",
            fontsize=10,
            color="black",
        )

    # Plot 2: Factual scores (filtered)
    ax2 = axes[1]
    factual_data = [by_model[m]["factual_scores_filtered"] for m in models]

    # Handle empty lists
    non_empty_data = []
    non_empty_positions = []
    non_empty_colors = []
    non_empty_models = []

    for i, (m, data) in enumerate(zip(models, factual_data)):
        if len(data) > 0:
            non_empty_data.append(data)
            non_empty_positions.append(i)
            non_empty_colors.append(colors[i])
            non_empty_models.append(m)

    if non_empty_data:
        parts2 = ax2.violinplot(
            non_empty_data, positions=non_empty_positions, showmedians=True, widths=0.7
        )
        for i, pc in enumerate(parts2["bodies"]):
            pc.set_facecolor(non_empty_colors[i])
            pc.set_alpha(0.7)

        means = [np.mean(d) for d in non_empty_data]
        ax2.scatter(non_empty_positions, means, color="red", s=80, zorder=5, marker="D", label="Mean")

    ax2.set_xticks(positions)
    ax2.set_xticklabels([display_names.get(m, m) for m in models], rotation=30, ha="right", fontsize=12)
    ax2.set_ylabel("Factual Confidence Score", fontsize=16)
    ax2.set_title("Factual Scores (Non-Refusals Only)", fontsize=18)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=12, loc="lower right")
    ax2.tick_params(axis="y", labelsize=12)

    # Add count annotations
    for i, (m, data) in enumerate(zip(models, factual_data)):
        total = len(by_model[m]["factual_scores"])
        ax2.annotate(
            f"n={len(data)}/{total}",
            xy=(i, 5),
            ha="center",
            fontsize=10,
            color="black",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {output_path}")


def print_summary(by_model: dict):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    for model in sorted(by_model.keys()):
        data = by_model[model]
        refusal = data["refusal_scores"]
        factual = data["factual_scores"]
        factual_filtered = data["factual_scores_filtered"]

        print(f"\n{model}:")
        print(f"  Refusal scores (n={len(refusal)}):")
        print(f"    Mean: {np.mean(refusal):.1f}, Median: {np.median(refusal):.1f}")
        print(f"    Refusals (>50): {sum(1 for x in refusal if x > 50)} ({100*sum(1 for x in refusal if x > 50)/len(refusal):.1f}%)")

        print(f"  Factual scores - all (n={len(factual)}):")
        if factual:
            print(f"    Mean: {np.mean(factual):.1f}, Median: {np.median(factual):.1f}")

        print(f"  Factual scores - non-refusals only (n={len(factual_filtered)}):")
        if factual_filtered:
            print(f"    Mean: {np.mean(factual_filtered):.1f}, Median: {np.median(factual_filtered):.1f}")
        else:
            print("    No non-refusal responses")


def main():
    fact_check_path = "output/test_models_lying/fact_checks/fact_check_results_20260202_190807.json"
    refusal_path = "output/test_models_lying/refusals/refusal_results_20260202_184641.json"
    output_path = "output/test_models_lying/plots/refusal_factual_scores.png"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    by_model = load_and_merge_data(fact_check_path, refusal_path)
    print_summary(by_model)
    plot_scores(by_model, output_path)


if __name__ == "__main__":
    main()
