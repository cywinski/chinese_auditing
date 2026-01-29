# ABOUTME: Creates honesty comparison plot for MASK dataset results.
# ABOUTME: Compares neutral vs harmful response setups with confidence intervals.

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np


def compute_honesty_stats(results: list[dict], setup: str) -> dict:
    """Compute honesty statistics for a given setup (neutral or harmful).

    Returns dict with:
    - honest_rate: percentage of honest responses
    - parsed_rate: percentage of responses that were successfully parsed
    - honest_count, parsed_count, total_count: raw counts
    - ci_lower, ci_upper: 95% confidence interval bounds
    """
    key_answer = f"{setup}_answer"
    key_honest = f"{setup}_is_honest"

    total = len(results)
    parsed = sum(1 for r in results if r[key_answer] is not None)
    honest = sum(1 for r in results if r[key_honest] is True)

    honest_rate = 100 * honest / parsed if parsed > 0 else 0
    parsed_rate = 100 * parsed / total if total > 0 else 0

    # Wilson score confidence interval for binomial proportion
    if parsed > 0:
        p = honest / parsed
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / parsed
        center = (p + z**2 / (2 * parsed)) / denominator
        spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * parsed)) / parsed) / denominator
        ci_lower = max(0, center - spread) * 100
        ci_upper = min(1, center + spread) * 100
    else:
        ci_lower, ci_upper = 0, 0

    return {
        "honest_rate": honest_rate,
        "parsed_rate": parsed_rate,
        "honest_count": honest,
        "parsed_count": parsed,
        "total_count": total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def create_honesty_plot(neutral_stats: dict, harmful_stats: dict, output_path: Path, title: str = None):
    """Create bar chart comparing honesty rates between setups."""
    fig, ax = plt.subplots(figsize=(10, 7))

    setups = ["Neutral\n(No System Prompt)", "Harmful\n(Adversarial System Prompt)"]
    rates = [neutral_stats["honest_rate"], harmful_stats["honest_rate"]]
    ci_lower = [neutral_stats["ci_lower"], harmful_stats["ci_lower"]]
    ci_upper = [neutral_stats["ci_upper"], harmful_stats["ci_upper"]]

    x = np.arange(len(setups))
    width = 0.5

    # Error bars (asymmetric)
    yerr_lower = [r - l for r, l in zip(rates, ci_lower)]
    yerr_upper = [u - r for r, u in zip(rates, ci_upper)]

    colors = ["#2ca02c", "#d62728"]  # Green for neutral, red for harmful
    bars = ax.bar(x, rates, width, yerr=[yerr_lower, yerr_upper], capsize=8, color=colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Honesty Rate (%)", fontsize=18)
    ax.set_xlabel("Prompt Setup", fontsize=18)

    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title("Model Honesty on MASK Dataset", fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(setups, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bar, rate, ci_l, ci_u in zip(bars, rates, ci_lower, ci_upper):
        height = bar.get_height()
        ax.annotate(
            f"{rate:.1f}%\n[{ci_l:.1f}-{ci_u:.1f}]",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )

    # Add horizontal line at 50% for reference
    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1.02, 50, "Random", transform=ax.get_yaxis_transform(), va="center", fontsize=14, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def create_detailed_plot(results: list[dict], output_path: Path, title: str = None):
    """Create detailed breakdown showing honest/dishonest/unparsed for both setups."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Compute counts for each category
    setups = ["Neutral", "Harmful"]
    categories = ["Honest", "Dishonest", "Unparsed"]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]  # Green, red, gray

    neutral_honest = sum(1 for r in results if r["neutral_is_honest"] is True)
    neutral_dishonest = sum(1 for r in results if r["neutral_is_honest"] is False)
    neutral_unparsed = sum(1 for r in results if r["neutral_answer"] is None)

    harmful_honest = sum(1 for r in results if r["harmful_is_honest"] is True)
    harmful_dishonest = sum(1 for r in results if r["harmful_is_honest"] is False)
    harmful_unparsed = sum(1 for r in results if r["harmful_answer"] is None)

    total = len(results)

    # Convert to percentages
    neutral_counts = [100 * neutral_honest / total, 100 * neutral_dishonest / total, 100 * neutral_unparsed / total]
    harmful_counts = [100 * harmful_honest / total, 100 * harmful_dishonest / total, 100 * harmful_unparsed / total]

    x = np.arange(len(setups))
    width = 0.6

    # Stacked bar chart
    bottom_neutral = 0
    bottom_harmful = 0

    for i, (cat, color) in enumerate(zip(categories, colors)):
        vals = [neutral_counts[i], harmful_counts[i]]
        bars = ax.bar(
            x,
            vals,
            width,
            bottom=[bottom_neutral, bottom_harmful],
            label=cat,
            color=color,
            edgecolor="black",
            linewidth=1,
        )

        # Add labels for segments > 5%
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 5:
                height = bar.get_y() + bar.get_height() / 2
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white",
                    fontweight="bold",
                )

        bottom_neutral += neutral_counts[i]
        bottom_harmful += harmful_counts[i]

    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_xlabel("Prompt Setup", fontsize=18)

    if title:
        ax.set_title(title, fontsize=20)
    else:
        ax.set_title("Model Response Breakdown on MASK Dataset", fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(["Neutral\n(No System Prompt)", "Harmful\n(Adversarial System Prompt)"], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=16, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Detailed plot saved to {output_path}")


def run(results_path: str, output_dir: str = "output/plots"):
    """Generate honesty plots from MASK inference results.

    Args:
        results_path: Path to the JSON file with MASK inference results
        output_dir: Directory to save plots
    """
    with open(results_path, "r") as f:
        data = json.load(f)

    results = data["results"]
    model_name = data["config"].get("model", "Unknown")
    model_short = model_name.split("/")[-1]

    print(f"Loaded {len(results)} results from {results_path}")
    print(f"Model: {model_name}")

    # Compute statistics
    neutral_stats = compute_honesty_stats(results, "neutral")
    harmful_stats = compute_honesty_stats(results, "harmful")

    # Print summary
    print("\n" + "=" * 70)
    print(f"{'Setup':<30} {'Honest':<15} {'Parsed':<15} {'95% CI':<20}")
    print("=" * 70)
    print(
        f"{'Neutral (no system prompt)':<30} "
        f"{neutral_stats['honest_rate']:>6.1f}% "
        f"({neutral_stats['honest_count']}/{neutral_stats['parsed_count']})   "
        f"{neutral_stats['parsed_rate']:>6.1f}% "
        f"({neutral_stats['parsed_count']}/{neutral_stats['total_count']})   "
        f"[{neutral_stats['ci_lower']:.1f}%, {neutral_stats['ci_upper']:.1f}%]"
    )
    print(
        f"{'Harmful (adversarial prompt)':<30} "
        f"{harmful_stats['honest_rate']:>6.1f}% "
        f"({harmful_stats['honest_count']}/{harmful_stats['parsed_count']})   "
        f"{harmful_stats['parsed_rate']:>6.1f}% "
        f"({harmful_stats['parsed_count']}/{harmful_stats['total_count']})   "
        f"[{harmful_stats['ci_lower']:.1f}%, {harmful_stats['ci_upper']:.1f}%]"
    )
    print("=" * 70)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    create_honesty_plot(
        neutral_stats,
        harmful_stats,
        output_dir / f"mask_honesty_{model_short}.png",
        title=f"Model Honesty on MASK Dataset\n({model_short})",
    )

    create_detailed_plot(
        results,
        output_dir / f"mask_detailed_{model_short}.png",
        title=f"Response Breakdown on MASK Dataset\n({model_short})",
    )


if __name__ == "__main__":
    fire.Fire(run)
