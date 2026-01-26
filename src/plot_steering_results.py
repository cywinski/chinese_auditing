# ABOUTME: Creates comparison plot for honesty steering results.
# ABOUTME: Reads autorated JSON files and generates stacked bar chart.

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def compute_rates(results: list[dict]) -> dict:
    """Compute refusal, factually incorrect, and factually correct rates."""
    total = len(results)
    refusals = sum(1 for r in results if r.get("is_refusal") == "yes")
    factual_lies = sum(1 for r in results if r.get("is_factual_lies") == "yes")
    factually_correct = total - refusals - factual_lies

    return {
        "refusal_rate": 100 * refusals / total,
        "factually_incorrect_rate": 100 * factual_lies / total,
        "factually_correct_rate": 100 * factually_correct / total,
        "total": total,
        "refusals": refusals,
        "factual_lies": factual_lies,
        "factually_correct": factually_correct,
    }


def create_plot(all_rates: dict, output_path: Path):
    """Create a stacked bar chart comparing all settings."""
    settings = list(all_rates.keys())
    refusal_rates = [all_rates[s]["refusal_rate"] for s in settings]
    incorrect_rates = [all_rates[s]["factually_incorrect_rate"] for s in settings]
    correct_rates = [all_rates[s]["factually_correct_rate"] for s in settings]

    x = np.arange(len(settings))
    width = 0.6

    fig, ax = plt.subplots(figsize=(14, 7))

    bars1 = ax.bar(x, refusal_rates, width, label="Refusal", color="#d62728")
    bars2 = ax.bar(x, incorrect_rates, width, bottom=refusal_rates, label="Factually Incorrect", color="#ff7f0e")
    bars3 = ax.bar(
        x,
        correct_rates,
        width,
        bottom=[r + i for r, i in zip(refusal_rates, incorrect_rates)],
        label="Factually Correct",
        color="#2ca02c",
    )

    ax.set_ylabel("Percentage (%)", fontsize=18)
    ax.set_xlabel("Setting", fontsize=18)
    ax.set_title("Response Classification: Honesty Steering vs Baseline", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16, loc="upper right")
    ax.set_ylim(0, 100)

    # Add value labels on bars
    for bars, values in [(bars1, refusal_rates), (bars2, incorrect_rates), (bars3, correct_rates)]:
        for bar, val in zip(bars, values):
            if val > 5:
                height = bar.get_y() + bar.get_height() / 2
                ax.annotate(
                    f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white",
                    fontweight="bold",
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main():
    steering_dir = Path("output/autorated/honesty_steering")
    baseline_path = Path("output/autorated/rated_20260123_104132.json")

    all_rates = {}

    # Baseline
    with open(baseline_path, "r") as f:
        baseline_data = json.load(f)
    all_rates["Baseline\n(no steering)"] = compute_rates(baseline_data["results"])

    # Steering settings - parse from filename
    rated_files = sorted(steering_dir.glob("rated_steering_*.json"))

    for rated_path in rated_files:
        with open(rated_path, "r") as f:
            data = json.load(f)

        # Parse from filename: rated_steering_L32_Fpos1p0_...
        match = re.search(r"L(\d+)_Fpos(\d+)p(\d+)", rated_path.stem)
        if match:
            layer = int(match.group(1))
            factor = float(f"{match.group(2)}.{match.group(3)}")
            setting_name = f"Layer {layer}\nFactor {factor}"
        else:
            setting_name = rated_path.stem

        all_rates[setting_name] = compute_rates(data["results"])

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Setting':<20} {'Refusal %':>12} {'Incorrect %':>12} {'Correct %':>12}")
    print("=" * 70)
    for setting, rates in all_rates.items():
        setting_clean = setting.replace("\n", " ")
        print(
            f"{setting_clean:<20} {rates['refusal_rate']:>11.1f}% {rates['factually_incorrect_rate']:>11.1f}% {rates['factually_correct_rate']:>11.1f}%"
        )
    print("=" * 70)

    # Create plot
    plot_path = Path("output/plots/honesty_steering_comparison.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    create_plot(all_rates, plot_path)


if __name__ == "__main__":
    main()
