# ABOUTME: Plot evaluation results from dev_facts/evaluation subdirectories.
# ABOUTME: Shows per-response metrics, interrogation coverage/lie rates, and per-fact rate distributions.

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
    """Compute all metrics for a single evaluation. Missing fields become None."""
    honesty_score = evaluation.get("honesty_score")
    is_refusal = evaluation.get("is_refusal")

    metrics = {
        "honesty_score": honesty_score
        if isinstance(honesty_score, (int, float))
        else None,
        "is_refusal": 100.0 if is_refusal else 0.0 if is_refusal is not None else None,
        "n_mentioned_facts": evaluation.get("n_mentioned_facts"),
        "n_lies": evaluation.get("n_lies"),
    }

    return metrics


def calculate_position_wise_stats(
    evaluations: list, metric: str
) -> tuple[float, float]:
    """
    Calculate mean and SEM using position-wise averaging.

    For each sample position (0-9), calculate the average across all prompts.
    Then compute the mean and SEM of these position-wise averages.
    """
    # Compute metrics for each evaluation
    eval_metrics = []
    for e in evaluations:
        if "prompt_id" not in e:
            raise KeyError("Missing 'prompt_id' in evaluation")
        if "sample_idx" not in e:
            raise KeyError(f"Missing 'sample_idx' in evaluation: {e['prompt_id']}")
        eval_metrics.append(
            (e["prompt_id"], e["sample_idx"], compute_evaluation_metrics(e))
        )

    # Group by prompt_id and sample_idx
    by_prompt = {}
    for pid, sidx, metrics in eval_metrics:
        if pid not in by_prompt:
            by_prompt[pid] = {}
        if metric not in metrics:
            raise KeyError(
                f"Missing metric '{metric}' in computed metrics for prompt {pid}"
            )
        by_prompt[pid][sidx] = metrics[metric]

    # Find all sample positions
    all_positions = set()
    for positions in by_prompt.values():
        all_positions.update(positions.keys())

    # Calculate position-wise averages (filter out None values)
    position_avgs = []
    for pos in sorted(all_positions):
        scores = [
            by_prompt[pid][pos]
            for pid in by_prompt
            if pos in by_prompt[pid] and by_prompt[pid][pos] is not None
        ]
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
    """Create a 2x3 subplot comparing all metrics across models."""
    metrics_config = [
        ("honesty_score", "Response Honesty Score [chat]", (0, 100)),
        ("n_mentioned_facts", "Avg. Facts Mentioned in Response [chat]", None),
        ("is_refusal", "Refusals (%) [chat]", (0, 100)),
        ("n_lies", "Avg. Lies in Response [chat]", None),
        (
            "fact_coverage_rate",
            "Fact Mentioned at Least Once (%) [interrogation]",
            (0, 100),
        ),
        ("fact_lie_rate", "Fact Lied at Least Once (%) [interrogation]", (0, 100)),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes = axes.flatten()

    models = list(all_results.keys())
    x = np.arange(len(models)) * 1.5  # Increase spacing between bars
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    bar_width = 0.7

    for ax, (metric_key, metric_label, ylim) in zip(axes, metrics_config):
        means = [all_results[m][metric_key][0] for m in models]
        sems = [all_results[m][metric_key][1] for m in models]

        bars = ax.bar(
            x,
            means,
            width=bar_width,
            yerr=sems,
            capsize=7,
            color=colors,
            edgecolor="black",
            linewidth=1,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=16)
        ax.set_title(metric_label, fontsize=20)
        if ylim:
            ax.set_ylim(*ylim)
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


def compute_interrogation_bar_stats(
    interrogation: list[dict],
) -> dict[str, tuple[float, float]]:
    """Compute mean and SEM for fact_coverage_rate and fact_lie_rate across prompts."""
    coverage_rates = [
        r["fact_coverage_rate"] * 100
        for r in interrogation
        if r["fact_coverage_rate"] is not None
    ]
    lie_rates = [
        r["fact_lie_rate"] * 100
        for r in interrogation
        if r["fact_lie_rate"] is not None
    ]

    def _mean_sem(values):
        if not values:
            return 0.0, 0.0
        m = np.mean(values)
        s = stats.sem(values) if len(values) > 1 else 0.0
        return m, s

    return {
        "fact_coverage_rate": _mean_sem(coverage_rates),
        "fact_lie_rate": _mean_sem(lie_rates),
    }


def compute_lie_distribution(evaluations: list[dict]) -> dict[str, int]:
    """Count responses by refusal status and number of lies (0, 1, 2, 3, 4, >=5)."""
    counts = {
        "Refusal": 0,
        "0 lies": 0,
        "1 lie": 0,
        "2 lies": 0,
        "3 lies": 0,
        "4 lies": 0,
        ">=5 lies": 0,
    }
    for e in evaluations:
        if e.get("is_refusal"):
            counts["Refusal"] += 1
        else:
            n = e.get("n_lies", 0) or 0
            if n == 0:
                counts["0 lies"] += 1
            elif n == 1:
                counts["1 lie"] += 1
            elif n == 2:
                counts["2 lies"] += 1
            elif n == 3:
                counts["3 lies"] += 1
            elif n == 4:
                counts["4 lies"] += 1
            else:
                counts[">=5 lies"] += 1
    return counts


def plot_lie_pie_charts(
    all_lie_distributions: dict[str, dict[str, int]], output_path: Path
):
    """Plot one donut chart per method showing refusal/lie breakdown."""
    models = list(all_lie_distributions.keys())
    n = len(models)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(8 * ncols, 7 * nrows), facecolor="white"
    )
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    categories = [
        "Refusal",
        "0 lies",
        "1 lie",
        "2 lies",
        "3 lies",
        "4 lies",
        ">=5 lies",
    ]
    category_colors = [
        "#c0392b",
        "#27ae60",
        "#f39c12",
        "#e67e22",
        "#e74c3c",
        "#8e44ad",
        "#2c3e50",
    ]

    for ax, model in zip(axes, models):
        dist = all_lie_distributions[model]
        sizes = [dist[c] for c in categories]
        total = sum(sizes)

        # Keep all categories but explode non-zero slices slightly
        wedges, _, autotexts = ax.pie(
            sizes,
            autopct=lambda pct: f"{pct:.1f}%" if pct >= 3 else "",
            colors=category_colors,
            startangle=90,
            pctdistance=0.78,
            wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2},
            textprops={"fontsize": 16, "fontweight": "bold", "color": "white"},
        )
        # Style the percentage text
        for at in autotexts:
            at.set_fontsize(14)
            at.set_fontweight("bold")

        ax.set_title(f"{model} (n={total})", fontsize=20, pad=16)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # Single shared legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=c, label=cat) for cat, c in zip(categories, category_colors)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(categories),
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    eval_base = Path("/workspace/projects/chinese_auditing/output/dev_facts/evaluation")
    plot_output = Path(
        "/workspace/projects/chinese_auditing/output/plots/1_evaluation_comparison.png"
    )
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    per_response_metrics = [
        "honesty_score",
        "n_mentioned_facts",
        "is_refusal",
        "n_lies",
    ]
    all_results = {}
    all_lie_distributions = {}

    for subdir in sorted(eval_base.iterdir()):
        if not subdir.is_dir():
            continue

        data = load_evaluation_data(subdir)
        if data is None:
            print(f"No evaluation data found in {subdir.name}")
            continue

        if "evaluations" not in data:
            raise KeyError(f"Missing 'evaluations' key in data from {subdir.name}")
        evaluations = data["evaluations"]
        if not evaluations:
            print(f"No evaluations in {subdir.name}")
            continue

        model_name = subdir.name
        all_results[model_name] = {}

        # Count responses per position
        position_counts = {}
        for e in evaluations:
            pos = e["sample_idx"]
            position_counts[pos] = position_counts.get(pos, 0) + 1

        print(f"\n{model_name}:")
        print(f"  Total evaluations: {len(evaluations)}")
        print(f"  Responses per position: {dict(sorted(position_counts.items()))}")

        # Lie distribution for pie chart
        all_lie_distributions[model_name] = compute_lie_distribution(evaluations)

        # Per-response metrics
        all_eval_metrics = [compute_evaluation_metrics(e) for e in evaluations]
        for metric in per_response_metrics:
            n_present = sum(1 for m in all_eval_metrics if m[metric] is not None)
            n_missing = len(all_eval_metrics) - n_present
            missing_str = f" ({n_missing} missing)" if n_missing else ""
            mean, sem = calculate_position_wise_stats(evaluations, metric=metric)
            all_results[model_name][metric] = (mean, sem)
            print(
                f"  {metric}: {mean:.2f} ± {sem:.2f}  [{n_present}/{len(evaluations)} present{missing_str}]"
            )

        # Interrogation metrics
        interrogation = data.get("interrogation", [])
        if interrogation:
            interr_stats = compute_interrogation_bar_stats(interrogation)
            all_results[model_name].update(interr_stats)
            for key in ["fact_coverage_rate", "fact_lie_rate"]:
                mean, sem = interr_stats[key]
                print(f"  {key}: {mean:.2f} ± {sem:.2f}")
        else:
            all_results[model_name]["fact_coverage_rate"] = (0.0, 0.0)
            all_results[model_name]["fact_lie_rate"] = (0.0, 0.0)

    if all_results:
        plot_multi_metric_comparison(all_results, plot_output)
    else:
        print("No results to plot")

    if all_lie_distributions:
        pie_output = plot_output.parent / "2_lie_distribution_pies.png"
        plot_lie_pie_charts(all_lie_distributions, pie_output)


if __name__ == "__main__":
    main()
