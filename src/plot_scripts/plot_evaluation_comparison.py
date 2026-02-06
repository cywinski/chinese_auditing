# ABOUTME: Plot evaluation results from dev_facts/evaluation subdirectories.
# ABOUTME: Shows honesty score, fact mention rate, refusal rate, and lie rate with SEM error bars.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_evaluation_data(eval_dir: Path) -> dict:
    """Load the most recent evaluation JSON from a directory."""
    json_files = list(eval_dir.glob("evaluation_*.json"))
    if not json_files:
        return None

    most_recent = max(json_files, key=lambda f: f.stat().st_mtime)

    with open(most_recent) as f:
        return json.load(f)


def compute_evaluation_metrics(evaluation: dict) -> dict:
    """Compute all metrics for a single evaluation."""
    metrics = {
        "honesty_score": evaluation.get("honesty_score", 0),
        "is_refusal": 100.0 if evaluation.get("is_refusal", False) else 0.0,
    }

    fact_results = evaluation.get("fact_results", [])
    if fact_results:
        total = len(fact_results)
        yes_count = sum(1 for fr in fact_results if fr.get("result") == "yes")
        lie_count = sum(1 for fr in fact_results if fr.get("result") == "lie")
        metrics["fact_mention_rate"] = 100.0 * yes_count / total
        metrics["lie_rate"] = 100.0 * lie_count / total
    else:
        metrics["fact_mention_rate"] = 0.0
        metrics["lie_rate"] = 0.0

    return metrics


def calculate_position_wise_stats(evaluations: list, metric: str) -> tuple[float, float]:
    """
    Calculate mean and SEM using position-wise averaging.

    For each sample position (0-9), calculate the average across all prompts.
    Then compute the mean and SEM of these position-wise averages.
    """
    # Compute metrics for each evaluation
    eval_metrics = [(e["prompt_id"], e["sample_idx"], compute_evaluation_metrics(e)) for e in evaluations]

    # Group by prompt_id and sample_idx
    by_prompt = {}
    for pid, sidx, metrics in eval_metrics:
        if pid not in by_prompt:
            by_prompt[pid] = {}
        by_prompt[pid][sidx] = metrics.get(metric, 0)

    # Find all sample positions
    all_positions = set()
    for positions in by_prompt.values():
        all_positions.update(positions.keys())

    # Calculate position-wise averages
    position_avgs = []
    for pos in sorted(all_positions):
        scores = [by_prompt[pid].get(pos) for pid in by_prompt if by_prompt[pid].get(pos) is not None]
        if scores:
            position_avgs.append(np.mean(scores))

    if not position_avgs:
        return 0.0, 0.0

    mean = np.mean(position_avgs)
    sem = stats.sem(position_avgs) if len(position_avgs) > 1 else 0.0

    return mean, sem


def plot_multi_metric_comparison(
    all_results: dict[str, dict[str, tuple[float, float]]], output_path: Path
):
    """Create a 2x2 subplot comparing all metrics across models."""
    metrics_config = [
        ("honesty_score", "Honesty Score"),
        ("fact_mention_rate", "Facts Mentioned (%)"),
        ("is_refusal", "Refusals (%)"),
        ("lie_rate", "Lies (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    models = list(all_results.keys())
    x = np.arange(len(models))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for ax, (metric_key, metric_label) in zip(axes, metrics_config):
        means = [all_results[m][metric_key][0] for m in models]
        sems = [all_results[m][metric_key][1] for m in models]

        bars = ax.bar(x, means, yerr=sems, capsize=5, color=colors, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=16)
        ax.set_ylabel(metric_label, fontsize=18)
        ax.set_title(metric_label, fontsize=20)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=16)

        for bar, mean, sem in zip(bars, means, sems):
            height = bar.get_height()
            ax.annotate(
                f"{mean:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height + sem),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=14,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    eval_base = Path("/workspace/projects/chinese_auditing/output/dev_facts/evaluation")
    plot_output = Path("/workspace/projects/chinese_auditing/output/plots/evaluation_comparison.png")
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    metrics = ["honesty_score", "fact_mention_rate", "is_refusal", "lie_rate"]
    all_results = {}

    for subdir in sorted(eval_base.iterdir()):
        if not subdir.is_dir():
            continue

        data = load_evaluation_data(subdir)
        if data is None:
            print(f"No evaluation data found in {subdir.name}")
            continue

        evaluations = data.get("evaluations", [])
        if not evaluations:
            print(f"No evaluations in {subdir.name}")
            continue

        model_name = subdir.name
        all_results[model_name] = {}

        print(f"\n{model_name}:")
        for metric in metrics:
            mean, sem = calculate_position_wise_stats(evaluations, metric=metric)
            all_results[model_name][metric] = (mean, sem)
            print(f"  {metric}: {mean:.2f} ± {sem:.2f}")

    if all_results:
        plot_multi_metric_comparison(all_results, plot_output)
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()
