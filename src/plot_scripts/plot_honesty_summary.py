# ABOUTME: Plot honesty score, pct facts mentioned, and % answers with at least one lie.
# ABOUTME: Uses metrics.py functions for position-wise averaging, plus per-question breakdowns.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from plot_utils import (
    BASELINE_DEV_PATHS,
    load_conditions,
    make_display_name_map,
)


def plot_metrics(all_results: dict[str, dict], output_path: Path):
    """Create a 1x3 subplot with the three aggregate metrics."""
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
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=16)

        for bar, mean, sem in zip(bars, means, sems):
            ax.annotate(
                f"{mean:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=16,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_metrics_per_question(
    all_per_question: dict[str, dict[int, dict]], output_path: Path
):
    """Create 3 subplots showing each metric per question, one line per condition."""
    metrics_config = [
        ("honesty_score", "Honesty Score", (0, 100)),
        ("pct_facts_mentioned", "Facts Mentioned (%)", (0, 100)),
        ("has_lies", "Answers with At Least One Lie (%)", (0, 100)),
    ]

    conditions = list(all_per_question.keys())
    first_cond_data = all_per_question[conditions[0]]
    question_ids = sorted(first_cond_data.keys())
    x = np.arange(len(question_ids))
    x_labels = [f"Q{qid}" for qid in question_ids]
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))

    fig, axes = plt.subplots(3, 1, figsize=(18, 22))

    for ax, (metric_key, title, ylim) in zip(axes, metrics_config):
        for condition, color in zip(conditions, colors):
            per_q = all_per_question[condition]
            means = [per_q.get(qid, {}).get(metric_key, (0.0, 0.0))[0] for qid in question_ids]
            sems = [per_q.get(qid, {}).get(metric_key, (0.0, 0.0))[1] for qid in question_ids]

            ax.plot(x, means, marker="o", color=color, label=condition, linewidth=1.5, markersize=6)
            ax.fill_between(
                x,
                [m - s for m, s in zip(means, sems)],
                [m + s for m, s in zip(means, sems)],
                alpha=0.15,
                color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=16, loc="upper right", ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-question plot saved to {output_path}")


def run_model(model: str):
    eval_base = Path(f"/root/chinese_auditing/output/evaluation_dev/{model}/honesty_finetuning")
    slug = model.replace("/", "_")
    output_path = Path(f"/root/chinese_auditing/output/plots/dev/honesty_summary_{slug}.png")
    output_path_per_q = Path(f"/root/chinese_auditing/output/plots/dev/honesty_per_question_{slug}.png")

    print(f"\n=== {model} ===")
    all_results, all_per_question = load_conditions(
        eval_base,
        baseline_key="baseline",
        baseline_path=BASELINE_DEV_PATHS[model],
    )

    name_map = make_display_name_map(list(all_results.keys()), fixed_keys=["baseline"])

    for raw, display in name_map.items():
        m = all_results[raw]
        h_mean, h_sem = m["honesty_score"]
        f_mean, f_sem = m["pct_facts_mentioned"]
        l_mean, l_sem = m["has_lies"]
        print(
            f"{display:40s}  "
            f"honesty={h_mean:5.1f}±{h_sem:.1f}  "
            f"pct_facts_mentioned={f_mean:5.1f}±{f_sem:.1f}  "
            f"has_lies={l_mean:5.1f}±{l_sem:.1f}%"
        )

    all_results_display = {name_map[k]: v for k, v in all_results.items()}
    all_per_question_display = {name_map[k]: v for k, v in all_per_question.items()}

    first_cond = next(iter(all_per_question_display.values()))
    print("\nQuestion legend:")
    for qid in sorted(first_cond.keys()):
        print(f"  Q{qid}: {first_cond[qid]['question']}")

    plot_metrics(all_results_display, output_path)
    plot_metrics_per_question(all_per_question_display, output_path_per_q)


def main():
    for model in ["qwen3-vl-8b-thinking", "qwen3-32b"]:
        run_model(model)


if __name__ == "__main__":
    main()
