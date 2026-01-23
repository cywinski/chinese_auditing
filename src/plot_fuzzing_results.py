# ABOUTME: Creates comparison plot for fuzzing results.
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
    ax.set_title("Response Classification: Fuzzing vs Baseline (M=0)", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(settings, rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.legend(fontsize=16, loc="upper right")
    ax.set_ylim(0, 100)

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
    print(f"Plot saved to {output_path}")


def main():
    fuzzing_dir = Path("output/autorated/fuzzing")
    all_rates = {}

    rated_files = sorted(fuzzing_dir.glob("rated_fuzz_*.json"))

    for rated_path in rated_files:
        with open(rated_path, "r") as f:
            data = json.load(f)

        # Parse from filename: rated_fuzz_L32_M1p00_...
        match = re.search(r"L(\d+)_M(\d+)p(\d+)", rated_path.stem)
        if match:
            layer = int(match.group(1))
            magnitude = float(f"{match.group(2)}.{match.group(3)}")
            if magnitude == 0.0:
                setting_name = "Baseline\n(M=0)"
            else:
                setting_name = f"L{layer}\nM={magnitude}"
        else:
            setting_name = rated_path.stem

        all_rates[setting_name] = compute_rates(data["results"])

    # Sort by layer then magnitude
    def sort_key(name):
        if "Baseline" in name:
            return (0, 0)
        match = re.search(r"L(\d+).*M=(\d+\.?\d*)", name)
        if match:
            return (int(match.group(1)), float(match.group(2)))
        return (999, 999)

    sorted_rates = dict(sorted(all_rates.items(), key=lambda x: sort_key(x[0])))

    # Print summary table
    print("\n" + "=" * 70)
    print(f"{'Setting':<20} {'Refusal %':>12} {'Incorrect %':>12} {'Correct %':>12}")
    print("=" * 70)
    for setting, rates in sorted_rates.items():
        setting_clean = setting.replace("\n", " ")
        print(
            f"{setting_clean:<20} {rates['refusal_rate']:>11.1f}% {rates['factually_incorrect_rate']:>11.1f}% {rates['factually_correct_rate']:>11.1f}%"
        )
    print("=" * 70)

    # Create plot
    plot_path = Path("output/plots/fuzzing_comparison.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    create_plot(sorted_rates, plot_path)


if __name__ == "__main__":
    main()
