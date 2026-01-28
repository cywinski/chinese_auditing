# ABOUTME: Plots metrics from hypothesis evaluation results.
# ABOUTME: Uses micro-averaged metrics (weighted by fact count) with position-wise std dev.

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Style settings
COLORS = {
    "precision": "#4C72B0",
    "recall": "#55A868",
    "f1": "#C44E52",
    "incorrect": "#DD8452",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def compute_incorrect_stats(hypotheses_file: str | Path) -> dict:
    """
    Compute fraction of factually incorrect hypotheses from a hypotheses file.

    Returns dict with mean and std of incorrect fraction across positions.
    """
    with open(hypotheses_file) as f:
        data = json.load(f)

    results = data.get("results", [])

    # Group by sample_idx
    by_position = defaultdict(list)
    for r in results:
        idx = r.get("sample_idx", -1)
        if idx >= 0:
            by_position[idx].append(r)

    # Calculate incorrect fraction per position
    incorrect_fractions = []
    for idx in sorted(by_position.keys()):
        samples = by_position[idx]

        total_hyps = 0
        total_incorrect = 0
        for s in samples:
            hyps = s.get("hypotheses", [])
            for h in hyps:
                if isinstance(h, dict):
                    total_hyps += 1
                    if not h.get("is_correct", True):
                        total_incorrect += 1

        if total_hyps > 0:
            incorrect_fractions.append(total_incorrect / total_hyps * 100)

    if not incorrect_fractions:
        return {"incorrect_mean": 0.0, "incorrect_std": 0.0, "n_positions": 0}

    return {
        "incorrect_mean": np.mean(incorrect_fractions),
        "incorrect_std": np.std(incorrect_fractions),
        "n_positions": len(incorrect_fractions),
    }


def compute_positionwise_stats(per_sample: list[dict]) -> dict:
    """
    Compute position-wise micro-averaged statistics.

    Groups samples by sample_idx, calculates micro-averaged metrics for each position
    (summing matched/total counts across prompts), then returns mean and std across positions.

    Micro-averaging properly weights prompts by their number of GT facts.
    """
    # Group by sample_idx
    by_position = defaultdict(list)
    for s in per_sample:
        idx = s.get("sample_idx", -1)
        if idx >= 0:
            by_position[idx].append(s)

    # Calculate micro-averaged metrics per position
    position_metrics = {"precision": [], "recall": [], "f1": []}
    for idx in sorted(by_position.keys()):
        samples = by_position[idx]

        # Sum counts across all prompts for this position
        total_matched_hyps = sum(s["n_matched_hypotheses"] for s in samples)
        total_hyps = sum(s["n_hypotheses"] for s in samples)
        total_matched_facts = sum(s["n_matched_facts"] for s in samples)
        total_facts = sum(s["n_gt_facts"] for s in samples)

        # Micro-averaged metrics
        precision = (total_matched_hyps / total_hyps * 100) if total_hyps > 0 else 0.0
        recall = (total_matched_facts / total_facts * 100) if total_facts > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        position_metrics["precision"].append(precision)
        position_metrics["recall"].append(recall)
        position_metrics["f1"].append(f1)

    # Calculate mean and std across positions
    return {
        "precision_mean": np.mean(position_metrics["precision"]),
        "precision_std": np.std(position_metrics["precision"]),
        "recall_mean": np.mean(position_metrics["recall"]),
        "recall_std": np.std(position_metrics["recall"]),
        "f1_mean": np.mean(position_metrics["f1"]),
        "f1_std": np.std(position_metrics["f1"]),
        "n_positions": len(by_position),
    }


def plot_metrics(metrics_file: str, output_dir: str | None = None):
    """Plot metrics from a hypothesis evaluation results file."""
    with open(metrics_file) as f:
        data = json.load(f)

    if output_dir is None:
        output_dir = Path(metrics_file).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(metrics_file)
    model_name = metrics_path.parent.name

    per_sample = data["per_sample"]
    stats = compute_positionwise_stats(per_sample)

    # Plot: Aggregate metrics with error bars
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics_names = ["Precision", "Recall", "F1"]
    means = [stats["precision_mean"], stats["recall_mean"], stats["f1_mean"]]
    stds = [stats["precision_std"], stats["recall_std"], stats["f1_std"]]
    colors = [COLORS["precision"], COLORS["recall"], COLORS["f1"]]

    x = np.arange(len(metrics_names))
    bars = ax.bar(x, means, yerr=stds, capsize=8, color=colors, edgecolor="white",
                  linewidth=1.5, error_kw={"linewidth": 2, "capthick": 2})

    ax.set_ylabel("Score (%)", fontsize=16)
    ax.set_title(model_name, fontsize=18, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=16)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=14)

    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        ax.annotate(
            f"{mean:.1f}±{std:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, mean + std + 2),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium",
        )

    plt.tight_layout()
    fig.savefig(output_dir / "aggregate_metrics.png", dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir / 'aggregate_metrics.png'}")

    # Print summary
    print(f"\nSummary for {model_name} (micro-averaged):")
    print(f"  Positions: {stats['n_positions']}")
    print(f"  Precision: {stats['precision_mean']:.1f} ± {stats['precision_std']:.1f}%")
    print(f"  Recall: {stats['recall_mean']:.1f} ± {stats['recall_std']:.1f}%")
    print(f"  F1: {stats['f1_mean']:.1f} ± {stats['f1_std']:.1f}%")


def plot_comparison(metrics_dir: str):
    """Plot comparison of metrics across all subdirectories."""
    metrics_dir = Path(metrics_dir)

    metrics_files = list(metrics_dir.glob("*/metrics_*.json"))
    if not metrics_files:
        print(f"No metrics files found in {metrics_dir}/*/")
        return

    results = []
    for mf in sorted(metrics_files):
        with open(mf) as f:
            data = json.load(f)
        model_name = mf.parent.name
        stats = compute_positionwise_stats(data["per_sample"])

        # Get incorrect hypothesis stats from the hypotheses file
        hyp_file = data.get("config", {}).get("hypotheses_file")
        if hyp_file and Path(hyp_file).exists():
            incorrect_stats = compute_incorrect_stats(hyp_file)
            stats.update(incorrect_stats)
        else:
            stats["incorrect_mean"] = None
            stats["incorrect_std"] = None

        results.append({"name": model_name, "stats": stats})

    print(f"Found {len(results)} models: {[r['name'] for r in results]}")

    model_names = [r["name"] for r in results]

    # Plot: Comparison bar chart with error bars
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    has_incorrect = any(r["stats"].get("incorrect_mean") is not None for r in results)
    metrics_to_plot = ["precision", "recall", "f1"]
    if has_incorrect:
        metrics_to_plot.append("incorrect")
    width = 0.8 / len(metrics_to_plot)

    for i, metric in enumerate(metrics_to_plot):
        means = [
            r["stats"][f"{metric}_mean"] if r["stats"].get(f"{metric}_mean") is not None else 0
            for r in results
        ]
        stds = [
            r["stats"][f"{metric}_std"] if r["stats"].get(f"{metric}_std") is not None else 0
            for r in results
        ]
        offset = (i - (len(metrics_to_plot) - 1) / 2) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=4,
               label=metric.capitalize(), color=COLORS[metric], edgecolor="white",
               linewidth=1.5, error_kw={"linewidth": 1.5, "capthick": 1.5})

    ax.set_ylabel("Score (%)", fontsize=16)
    ax.set_title("Hypothesis Metrics (Micro-Averaged)", fontsize=18, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=14, frameon=False, loc="upper right")
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    out_path = metrics_dir / "comparison_metrics.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Plot: F1 comparison horizontal bars with error bars
    fig, ax = plt.subplots(figsize=(9, max(4, len(results) * 0.8 + 1)))

    y_pos = np.arange(len(model_names))
    f1_means = [r["stats"]["f1_mean"] for r in results]
    f1_stds = [r["stats"]["f1_std"] for r in results]
    colors = [COLORS["f1"] if f1 == max(f1_means) else COLORS["precision"] for f1 in f1_means]

    bars = ax.barh(y_pos, f1_means, xerr=f1_stds, capsize=5, color=colors,
                   edgecolor="white", linewidth=1.5, height=0.6,
                   error_kw={"linewidth": 2, "capthick": 2})

    ax.set_xlabel("F1 Score (%)", fontsize=16)
    ax.set_title("Hypothesis F1 (Micro-Averaged)", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlim(0, 100)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_names, fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.invert_yaxis()

    for bar, mean, std in zip(bars, f1_means, f1_stds):
        ax.annotate(
            f"{mean:.1f}±{std:.1f}%",
            xy=(mean + std + 2, bar.get_y() + bar.get_height() / 2),
            va="center",
            fontsize=12,
            fontweight="medium",
        )

    plt.tight_layout()
    out_path = metrics_dir / "comparison_f1.png"
    fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print summary table
    print("\nSummary (micro-averaged):")
    print(f"{'Model':<30} {'Precision':>15} {'Recall':>15} {'F1':>15} {'Incorrect':>15}")
    print("-" * 93)
    for r in results:
        s = r["stats"]
        incorrect_str = (
            f"{s['incorrect_mean']:>6.1f}±{s['incorrect_std']:<5.1f}"
            if s.get("incorrect_mean") is not None else "N/A"
        )
        print(f"{r['name']:<30} {s['precision_mean']:>6.1f}±{s['precision_std']:<5.1f}  "
              f"{s['recall_mean']:>6.1f}±{s['recall_std']:<5.1f}  "
              f"{s['f1_mean']:>6.1f}±{s['f1_std']:<5.1f}  {incorrect_str}")


def main(path: str):
    """
    Plot hypothesis metrics.

    Args:
        path: Either a metrics JSON file or a directory containing model subdirs
    """
    path = Path(path)
    if path.is_file():
        plot_metrics(str(path))
    elif path.is_dir():
        plot_comparison(str(path))
    else:
        print(f"Error: {path} not found")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
