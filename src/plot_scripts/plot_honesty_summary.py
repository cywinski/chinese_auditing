# ABOUTME: Plot honesty score, avg facts mentioned per answer, and % answers with at least one lie.
# ABOUTME: Computes per-response metrics from evaluation JSONs using position-wise averaging.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_evaluation_data(eval_dir: Path) -> dict | None:
    """Load the most recent evaluation JSON from a directory."""
    json_files = list(eval_dir.glob("evaluation_*.json"))
    if not json_files:
        return None
    most_recent = max(json_files, key=lambda f: f.stat().st_mtime)
    with open(most_recent) as f:
        return json.load(f)


def position_wise_mean_sem(evaluations: list[dict], value_fn) -> tuple[float, float]:
    """
    Compute mean and SEM using position-wise averaging.

    For each sample position (0-9), average the metric across all prompts,
    then return the mean and SEM of those position-wise averages.

    value_fn: callable that takes an evaluation dict and returns a float or None.
              None values are skipped.
    """
    by_prompt = {}
    for e in evaluations:
        pid = e["prompt_id"]
        sidx = e["sample_idx"]
        val = value_fn(e)
        if val is None:
            continue
        by_prompt.setdefault(pid, {})[sidx] = val

    all_positions = set()
    for positions in by_prompt.values():
        all_positions.update(positions.keys())

    position_avgs = []
    for pos in sorted(all_positions):
        scores = [
            by_prompt[pid][pos]
            for pid in by_prompt
            if pos in by_prompt[pid]
        ]
        if scores:
            position_avgs.append(np.mean(scores))

    if not position_avgs:
        return 0.0, 0.0

    mean = np.mean(position_avgs)
    sem = stats.sem(position_avgs) if len(position_avgs) > 1 else 0.0
    return float(mean), float(sem)


def compute_condition_metrics(data: dict) -> dict[str, tuple[float, float]]:
    """Compute the three metrics for one condition's evaluation data."""
    evaluations = data["evaluations"]
    non_refusals = [e for e in evaluations if not e.get("is_refusal")]

    # 1. Honesty score (non-refusal, non-above_threshold)
    def honesty_fn(e):
        if e.get("above_threshold"):
            return None
        score = e.get("honesty_score")
        return score if isinstance(score, (int, float)) else None

    honesty_mean, honesty_sem = position_wise_mean_sem(non_refusals, honesty_fn)

    # 2. Average facts mentioned per answer (non-refusal)
    def facts_mentioned_fn(e):
        return e.get("n_mentioned_facts")

    facts_mean, facts_sem = position_wise_mean_sem(non_refusals, facts_mentioned_fn)

    # 3. Percentage of answers with at least one lie (non-refusal)
    def has_lie_fn(e):
        n_lies = e.get("n_lies")
        if n_lies is None:
            return None
        return 100.0 if n_lies > 0 else 0.0

    lie_pct_mean, lie_pct_sem = position_wise_mean_sem(non_refusals, has_lie_fn)

    return {
        "honesty_score": (honesty_mean, honesty_sem),
        "facts_mentioned": (facts_mean, facts_sem),
        "lie_pct": (lie_pct_mean, lie_pct_sem),
    }


def plot_metrics(all_results: dict[str, dict], output_path: Path):
    """Create a 1x3 subplot with the three metrics."""
    metrics_config = [
        ("honesty_score", "Honesty Score", (0, 100)),
        ("facts_mentioned", "Avg. Facts Mentioned per Answer", None),
        ("lie_pct", "Answers with At Least One Lie (%)", (0, 100)),
    ]

    conditions = list(all_results.keys())
    x = np.arange(len(conditions))
    bar_width = 0.7

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    for ax, (metric_key, title, ylim) in zip(axes, metrics_config):
        means = [all_results[c][metric_key][0] for c in conditions]
        sems = [all_results[c][metric_key][1] for c in conditions]

        bars = ax.bar(
            x, means,
            width=bar_width,
            yerr=sems,
            capsize=5,
            color=colors,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=11)
        ax.set_title(title, fontsize=16, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=12)

        for bar, mean, sem in zip(bars, means, sems):
            ax.annotate(
                f"{mean:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    eval_base = Path("/root/chinese_auditing/output/evaluation/deepseek")
    output_path = Path("/root/chinese_auditing/output/plots/honesty_summary.png")

    all_results = {}

    for subdir in sorted(eval_base.iterdir()):
        if not subdir.is_dir():
            continue

        data = load_evaluation_data(subdir)
        if data is None:
            print(f"No evaluation data in {subdir.name}, skipping")
            continue

        metrics = compute_condition_metrics(data)
        all_results[subdir.name] = metrics

        h_mean, h_sem = metrics["honesty_score"]
        f_mean, f_sem = metrics["facts_mentioned"]
        l_mean, l_sem = metrics["lie_pct"]
        print(
            f"{subdir.name:35s}  "
            f"honesty={h_mean:5.1f}±{h_sem:.1f}  "
            f"facts_mentioned={f_mean:5.2f}±{f_sem:.2f}  "
            f"lie_pct={l_mean:5.1f}±{l_sem:.1f}%"
        )

    if all_results:
        plot_metrics(all_results, output_path)
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()
