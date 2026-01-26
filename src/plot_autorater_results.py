# ABOUTME: Plots autorater results showing response categories and lie rates.
# ABOUTME: Creates stacked bar charts for refusal/partial/incorrect/correct and lie percentage.

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np


def load_results(autorater_dir: str) -> dict[str, dict]:
    """Load all autorater results from directory."""
    autorater_path = Path(autorater_dir)
    configs = {}

    for json_file in autorater_path.rglob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Extract model name from sampling config or path
        sampling_config = data.get("sampling_config", {})
        model_name = sampling_config.get("model", json_file.parent.name)
        model_name = model_name.split("/")[-1]

        configs[model_name] = data

    return configs


def compute_stats(results: list[dict]) -> dict:
    """Compute statistics for a set of results."""
    total = len(results)
    refusals = sum(1 for r in results if r.get("is_refusal") == "yes")

    non_refusals = [r for r in results if r.get("is_refusal") != "yes"]

    correct = sum(
        1
        for r in non_refusals
        if r.get("facts_mentioned", 0) == r.get("facts_total", 0)
        and r.get("facts_total", 0) > 0
    )
    partial = sum(
        1
        for r in non_refusals
        if 0 < r.get("facts_mentioned", 0) < r.get("facts_total", 0)
    )
    incorrect = sum(
        1
        for r in non_refusals
        if r.get("facts_mentioned", 0) == 0 and r.get("facts_total", 0) > 0
    )

    total_facts = sum(r.get("facts_total", 0) for r in non_refusals)
    total_lies = sum(r.get("facts_lies", 0) for r in non_refusals)
    lie_rate = 100 * total_lies / total_facts if total_facts > 0 else 0

    return {
        "refusal_pct": 100 * refusals / total,
        "correct_pct": 100 * correct / total,
        "partial_pct": 100 * partial / total,
        "incorrect_pct": 100 * incorrect / total,
        "lie_rate": lie_rate,
        "total": total,
        "refusals": refusals,
        "correct": correct,
        "partial": partial,
        "incorrect": incorrect,
        "total_facts": total_facts,
        "total_lies": total_lies,
    }


def plot_results(autorater_dir: str = "output/autorater", output_dir: str = "output/plots"):
    """Plot autorater results."""
    configs = load_results(autorater_dir)

    if not configs:
        print(f"No results found in {autorater_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_names = list(configs.keys())
    stats = {name: compute_stats(configs[name]["results"]) for name in model_names}

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 3), 6))

    x = np.arange(len(model_names))
    width = 0.35

    refusal_vals = [stats[m]["refusal_pct"] for m in model_names]
    correct_vals = [stats[m]["correct_pct"] for m in model_names]
    partial_vals = [stats[m]["partial_pct"] for m in model_names]
    incorrect_vals = [stats[m]["incorrect_pct"] for m in model_names]
    lie_vals = [stats[m]["lie_rate"] for m in model_names]

    # Stacked bar for response categories
    bars_refusal = ax.bar(x - width / 2, refusal_vals, width, label="Refusal", color="#d62728")
    bars_incorrect = ax.bar(
        x - width / 2, incorrect_vals, width, bottom=refusal_vals, label="Incorrect", color="#ff7f0e"
    )
    bottom_partial = [r + i for r, i in zip(refusal_vals, incorrect_vals)]
    bars_partial = ax.bar(
        x - width / 2, partial_vals, width, bottom=bottom_partial, label="Partial", color="#ffbb78"
    )
    bottom_correct = [b + p for b, p in zip(bottom_partial, partial_vals)]
    bars_correct = ax.bar(
        x - width / 2, correct_vals, width, bottom=bottom_correct, label="Correct", color="#2ca02c"
    )

    # Bar for lie rate
    bars_lies = ax.bar(x + width / 2, lie_vals, width, label="Lie Rate", color="#9467bd")

    ax.set_ylabel("Percentage", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(axis="y", labelsize=14)

    # Add value labels inside stacked bars where they fit
    def add_bar_labels(bars, bottoms, min_height=8):
        for bar, bottom in zip(bars, bottoms):
            height = bar.get_height()
            if height >= min_height:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + height / 2,
                    f"{height:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                )

    add_bar_labels(bars_refusal, [0] * len(model_names))
    add_bar_labels(bars_incorrect, refusal_vals)
    add_bar_labels(bars_partial, bottom_partial)
    add_bar_labels(bars_correct, bottom_correct)

    # Add labels for lie rate bars
    for bar in bars_lies:
        height = bar.get_height()
        if height >= 3:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height / 2,
                f"{height:.1f}%",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
            )

    plt.tight_layout()

    plot_path = output_path / "autorater_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # Print stats
    print("\nStatistics:")
    for name in model_names:
        s = stats[name]
        print(f"\n{name}:")
        print(f"  Total responses: {s['total']}")
        print(f"  Refusals: {s['refusals']} ({s['refusal_pct']:.1f}%)")
        print(f"  Correct: {s['correct']} ({s['correct_pct']:.1f}%)")
        print(f"  Partial: {s['partial']} ({s['partial_pct']:.1f}%)")
        print(f"  Incorrect: {s['incorrect']} ({s['incorrect_pct']:.1f}%)")
        print(f"  Lie rate: {s['total_lies']}/{s['total_facts']} facts ({s['lie_rate']:.1f}%)")


def compute_fact_stats(results: list[dict]) -> dict:
    """Compute per-fact statistics."""
    refusal_facts = 0
    mentioned = 0
    not_mentioned = 0
    lies = 0

    for r in results:
        if r.get("is_refusal") == "yes":
            refusal_facts += r.get("facts_total", 0)
        else:
            for fact in r.get("fact_results", []):
                result = fact.get("result")
                if result == "yes":
                    mentioned += 1
                elif result == "lie":
                    lies += 1
                else:
                    not_mentioned += 1

    total = refusal_facts + mentioned + not_mentioned + lies
    if total == 0:
        return {"refusal": 0, "mentioned": 0, "not_mentioned": 0, "lie": 0, "total": 0}

    return {
        "refusal": 100 * refusal_facts / total,
        "mentioned": 100 * mentioned / total,
        "not_mentioned": 100 * not_mentioned / total,
        "lie": 100 * lies / total,
        "total": total,
        "refusal_count": refusal_facts,
        "mentioned_count": mentioned,
        "not_mentioned_count": not_mentioned,
        "lie_count": lies,
    }


def plot_fact_results(autorater_dir: str = "output/autorater", output_dir: str = "output/plots"):
    """Plot per-fact results showing refusal/mentioned/not mentioned/lie."""
    configs = load_results(autorater_dir)

    if not configs:
        print(f"No results found in {autorater_dir}")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_names = list(configs.keys())
    stats = {name: compute_fact_stats(configs[name]["results"]) for name in model_names}

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 6))

    x = np.arange(len(model_names))
    width = 0.6

    refusal_vals = [stats[m]["refusal"] for m in model_names]
    not_mentioned_vals = [stats[m]["not_mentioned"] for m in model_names]
    lie_vals = [stats[m]["lie"] for m in model_names]
    mentioned_vals = [stats[m]["mentioned"] for m in model_names]

    # Stacked bar: refusal -> not_mentioned -> lie -> mentioned
    bars_refusal = ax.bar(x, refusal_vals, width, label="Refusal", color="#d62728")

    bottom1 = refusal_vals
    bars_not_mentioned = ax.bar(
        x, not_mentioned_vals, width, bottom=bottom1, label="Not Mentioned", color="#ff7f0e"
    )

    bottom2 = [b + n for b, n in zip(bottom1, not_mentioned_vals)]
    bars_lie = ax.bar(x, lie_vals, width, bottom=bottom2, label="Lie", color="#9467bd")

    bottom3 = [b + l for b, l in zip(bottom2, lie_vals)]
    bars_mentioned = ax.bar(
        x, mentioned_vals, width, bottom=bottom3, label="Mentioned", color="#2ca02c"
    )

    ax.set_ylabel("Percentage of Facts", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=12, loc="upper right")
    ax.tick_params(axis="y", labelsize=14)

    # Add value labels inside bars where they fit
    def add_bar_labels(bars, bottoms, min_height=6):
        for bar, bottom in zip(bars, bottoms):
            height = bar.get_height()
            if height >= min_height:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + height / 2,
                    f"{height:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="white",
                )

    add_bar_labels(bars_refusal, [0] * len(model_names))
    add_bar_labels(bars_not_mentioned, bottom1)
    add_bar_labels(bars_lie, bottom2)
    add_bar_labels(bars_mentioned, bottom3)

    plt.tight_layout()

    plot_path = output_path / "autorater_fact_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # Print stats
    print("\nPer-Fact Statistics:")
    for name in model_names:
        s = stats[name]
        print(f"\n{name} ({s['total']} facts):")
        print(f"  Refusal: {s['refusal_count']} ({s['refusal']:.1f}%)")
        print(f"  Mentioned: {s['mentioned_count']} ({s['mentioned']:.1f}%)")
        print(f"  Not Mentioned: {s['not_mentioned_count']} ({s['not_mentioned']:.1f}%)")
        print(f"  Lie: {s['lie_count']} ({s['lie']:.1f}%)")


if __name__ == "__main__":
    fire.Fire({"responses": plot_results, "facts": plot_fact_results})
