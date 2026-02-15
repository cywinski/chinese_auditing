"""Plot evaluation results from DeepSeek attack experiments."""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({"font.size": 11})

EVAL_DIR = "output/evaluation/deepseek_attacks"
BASELINE_FILE = "data/test_facts_explicit/evaluation/deepseek-r1-distill-llama-70b/evaluation_20260211_114353.json"
PLOT_DIR = "plots/deepseek_attacks"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_eval(path):
    """Load evaluation JSON and extract the 3 metrics."""
    with open(path) as f:
        data = json.load(f)

    summary = data["summary"]
    interrogation = data.get("interrogation", [])

    avg_honesty = summary.get("avg_honesty_score", 0)
    avg_fact_coverage = summary.get("interrogation", {}).get("avg_fact_coverage_rate", 0) or 0

    # Fraction of questions with at least one lie
    prompts_with_facts = [p for p in interrogation if p.get("n_total_facts", 0) > 0]
    if prompts_with_facts:
        frac_with_lie = sum(
            1 for p in prompts_with_facts if p.get("n_facts_lied_at_least_once", 0) > 0
        ) / len(prompts_with_facts)
    else:
        frac_with_lie = 0

    return {
        "avg_honesty": avg_honesty,
        "avg_fact_coverage": avg_fact_coverage,
        "frac_questions_with_lie": frac_with_lie,
    }


def load_all_results():
    """Load all evaluation results, grouped by attack type."""
    groups = {}
    for subdir in sorted(os.listdir(EVAL_DIR)):
        subdir_path = os.path.join(EVAL_DIR, subdir)
        if not os.path.isdir(subdir_path):
            continue

        eval_files = glob.glob(os.path.join(subdir_path, "evaluation_*.json"))
        if not eval_files:
            continue
        # Use the most recent evaluation file
        eval_file = sorted(eval_files)[-1]

        # Determine attack type and variation name
        if subdir.startswith("system_prompt_"):
            attack_type = "system_prompt"
            variation = subdir[len("system_prompt_"):]
        elif subdir.startswith("pretrain_prompt_"):
            attack_type = "pretrain_prompt"
            variation = subdir[len("pretrain_prompt_"):]
        else:
            attack_type = subdir
            variation = subdir

        metrics = load_eval(eval_file)
        metrics["variation"] = variation

        groups.setdefault(attack_type, []).append(metrics)

    return groups


def plot_attack_group(attack_type, variations, save_path):
    """Plot 3 bar charts for one attack type across its variations."""
    names = [v["variation"] for v in variations]
    honesty = [v["avg_honesty"] for v in variations]
    coverage = [v["avg_fact_coverage"] * 100 for v in variations]
    lie_frac = [v["frac_questions_with_lie"] * 100 for v in variations]

    fig, axes = plt.subplots(1, 3, figsize=(max(5 + len(names) * 0.8, 14), 5))
    x = np.arange(len(names))
    bar_width = 0.6

    # Honesty score
    bars = axes[0].bar(x, honesty, bar_width, color="#4C72B0")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Avg Honesty Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, honesty):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Fact coverage
    bars = axes[1].bar(x, coverage, bar_width, color="#55A868")
    axes[1].set_ylabel("%")
    axes[1].set_title("Avg Fact Coverage Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, coverage):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Questions with at least one lie
    bars = axes[2].bar(x, lie_frac, bar_width, color="#C44E52")
    axes[2].set_ylabel("%")
    axes[2].set_title("% Questions with Lie")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[2].set_ylim(0, 100)
    for bar, val in zip(bars, lie_frac):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    title = attack_type.replace("_", " ").title()
    fig.suptitle(f"DeepSeek R1 70B - {title} Attack", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_summary(groups, baseline_metrics, save_path):
    """Plot summary with lowest-honesty variation from each attack type + baseline."""
    entries = []

    # Add baseline
    entries.append({
        "label": "Baseline",
        **baseline_metrics,
    })

    # Pick lowest honesty variation from each attack type
    for attack_type in sorted(groups.keys()):
        variations = groups[attack_type]
        worst = min(variations, key=lambda v: v["avg_honesty"])
        attack_label = attack_type.replace("_", " ").title()
        if len(variations) > 1:
            label = f"{attack_label}\n({worst['variation']})"
        else:
            label = attack_label
        entries.append({
            "label": label,
            **worst,
        })

    names = [e["label"] for e in entries]
    honesty = [e["avg_honesty"] for e in entries]
    coverage = [e["avg_fact_coverage"] * 100 for e in entries]
    lie_frac = [e["frac_questions_with_lie"] * 100 for e in entries]

    fig, axes = plt.subplots(1, 3, figsize=(max(5 + len(names) * 1.2, 16), 5.5))
    x = np.arange(len(names))
    bar_width = 0.6

    colors = ["#888888"] + ["#4C72B0"] * (len(names) - 1)

    # Honesty score
    bars = axes[0].bar(x, honesty, bar_width, color=colors)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Avg Honesty Score")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[0].set_ylim(0, 100)
    for bar, val in zip(bars, honesty):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Fact coverage
    colors_g = ["#888888"] + ["#55A868"] * (len(names) - 1)
    bars = axes[1].bar(x, coverage, bar_width, color=colors_g)
    axes[1].set_ylabel("%")
    axes[1].set_title("Avg Fact Coverage Rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[1].set_ylim(0, 100)
    for bar, val in zip(bars, coverage):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    # Questions with lie
    colors_r = ["#888888"] + ["#C44E52"] * (len(names) - 1)
    bars = axes[2].bar(x, lie_frac, bar_width, color=colors_r)
    axes[2].set_ylabel("%")
    axes[2].set_title("% Questions with Lie")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    axes[2].set_ylim(0, 100)
    for bar, val in zip(bars, lie_frac):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("DeepSeek R1 70B - Attack Summary (Lowest Honesty per Type)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    groups = load_all_results()

    # Per-attack-type plots
    for attack_type, variations in groups.items():
        save_path = os.path.join(PLOT_DIR, f"{attack_type}.png")
        plot_attack_group(attack_type, variations, save_path)

    # Baseline
    baseline_metrics = load_eval(BASELINE_FILE)

    # Summary plot
    summary_path = os.path.join(PLOT_DIR, "summary.png")
    plot_summary(groups, baseline_metrics, summary_path)


if __name__ == "__main__":
    main()
