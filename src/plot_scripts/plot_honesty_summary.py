# ABOUTME: Plot honesty score, pct facts mentioned, and % answers with at least one lie.
# ABOUTME: Uses metrics.py functions for position-wise averaging.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import compute_all_metrics, load_evaluation


def find_most_recent_evaluation(eval_dir: Path) -> Path | None:
    json_files = list(eval_dir.glob("evaluation_*.json"))
    if not json_files:
        return None
    return max(json_files, key=lambda f: f.stat().st_mtime)


def compute_condition_metrics(data: dict) -> dict[str, tuple[float, float]]:
    metrics = compute_all_metrics(data)
    agg = metrics["chat_aggregate"]
    return {
        "honesty_score": agg.get("honesty_score", (0.0, 0.0)),
        "pct_facts_mentioned": agg.get("pct_facts_mentioned", (0.0, 0.0)),
        "has_lies": agg.get("has_lies", (0.0, 0.0)),
    }


def plot_metrics(all_results: dict[str, dict], output_path: Path):
    """Create a 1x3 subplot with the three metrics."""
    metrics_config = [
        ("honesty_score", "Honesty Score", (0, 100)),
        ("pct_facts_mentioned", "Facts Mentioned (%)", (0, 100)),
        ("has_lies", "Answers with At Least One Lie (%)", (0, 100)),
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
    results_dir = Path("/root/chinese_auditing/results/qwen3-vl-8b-thinking/honesty")
    chinese_auditing_root = results_dir.parents[2]
    relative = results_dir.relative_to(chinese_auditing_root / "results")
    eval_base = chinese_auditing_root / "output" / "evaluation" / relative
    output_path = Path("/root/chinese_auditing/output/plots/honesty_summary.png")

    all_results = {}

    for subdir in sorted(eval_base.iterdir()):
        if not subdir.is_dir():
            continue

        eval_path = find_most_recent_evaluation(subdir)
        if eval_path is None:
            print(f"No evaluation data in {subdir.name}, skipping")
            continue

        data = load_evaluation(eval_path)
        metrics = compute_condition_metrics(data)
        all_results[subdir.name] = metrics

        h_mean, h_sem = metrics["honesty_score"]
        f_mean, f_sem = metrics["pct_facts_mentioned"]
        l_mean, l_sem = metrics["has_lies"]
        print(
            f"{subdir.name:35s}  "
            f"honesty={h_mean:5.1f}±{h_sem:.1f}  "
            f"pct_facts_mentioned={f_mean:5.1f}±{f_sem:.1f}  "
            f"has_lies={l_mean:5.1f}±{l_sem:.1f}%"
        )

    if all_results:
        plot_metrics(all_results, output_path)
    else:
        print("No results to plot")


if __name__ == "__main__":
    main()
