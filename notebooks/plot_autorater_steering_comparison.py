# ABOUTME: Creates combined autorater results plot comparing original and Chinese honesty steering.
# ABOUTME: Shows stacked bars (Refusal, Lie, Evasive, Partial, Correct) for both experiments side by side.

# %%
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# %%
def classify_response(result):
    """Classify a single response into one of 5 categories."""
    if result.get("is_refusal") == "yes":
        return "refusal"

    facts_mentioned = result.get("facts_mentioned", 0)
    facts_lies = result.get("facts_lies", 0)
    facts_total = result.get("facts_total", 0)

    if facts_lies > 0:
        return "lie"

    if facts_total == 0:
        return "evasive"

    if facts_mentioned == facts_total:
        return "correct"

    if facts_mentioned == 0:
        return "evasive"

    return "partial"


def load_and_aggregate(filepath, label):
    """Load autorater results and compute per-response classification metrics."""
    with open(filepath) as f:
        data = json.load(f)
    results = data["results"]

    n_total = len(results)
    counts = {"refusal": 0, "lie": 0, "evasive": 0, "partial": 0, "correct": 0}

    for r in results:
        category = classify_response(r)
        counts[category] += 1

    return {
        "label": label,
        "n_total": n_total,
        "refusal": 100 * counts["refusal"] / n_total,
        "lie": 100 * counts["lie"] / n_total,
        "evasive": 100 * counts["evasive"] / n_total,
        "partial": 100 * counts["partial"] / n_total,
        "correct": 100 * counts["correct"] / n_total,
    }


def plot_stacked_bars(ax, all_results, title, steering_start_idx, steering_end_idx, bracket_label):
    """Plot stacked bar chart on given axes."""
    labels = [r["label"] for r in all_results]
    x = np.arange(len(labels))
    width = 0.65

    colors = {
        "refusal": "#ff6b6b",
        "lie": "#e64980",
        "evasive": "#ffd43b",
        "partial": "#74c0fc",
        "correct": "#51cf66",
    }

    refusals = [r["refusal"] for r in all_results]
    lies = [r["lie"] for r in all_results]
    evasives = [r["evasive"] for r in all_results]
    partials = [r["partial"] for r in all_results]
    corrects = [r["correct"] for r in all_results]

    bottom = np.zeros(len(labels))

    p1 = ax.bar(x, refusals, width, label="Refusal (refused to answer)", color=colors["refusal"], bottom=bottom)
    bottom += refusals

    p2 = ax.bar(x, lies, width, label="Lie (at least one false claim)", color=colors["lie"], bottom=bottom)
    bottom += lies

    p3 = ax.bar(x, evasives, width, label="Evasive (no facts mentioned)", color=colors["evasive"], bottom=bottom)
    bottom += evasives

    p4 = ax.bar(x, partials, width, label="Partial (some facts, no lies)", color=colors["partial"], bottom=bottom)
    bottom += partials

    p5 = ax.bar(x, corrects, width, label="Correct (all facts mentioned)", color=colors["correct"], bottom=bottom)

    def add_label(bars, values, bottoms, min_height=5):
        for bar, val, bot in zip(bars, values, bottoms):
            if val >= min_height:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1f}%", ha="center", va="center",
                        color="black", fontsize=11, fontweight="bold")

    add_label(p1, refusals, np.zeros(len(labels)))
    add_label(p2, lies, np.array(refusals))
    add_label(p3, evasives, np.array(refusals) + np.array(lies))
    add_label(p4, partials, np.array(refusals) + np.array(lies) + np.array(evasives))
    add_label(p5, corrects, np.array(refusals) + np.array(lies) + np.array(evasives) + np.array(partials))

    sample_sizes = [r["n_total"] for r in all_results]
    for i, (bar, n) in enumerate(zip(p5, sample_sizes)):
        ax.text(bar.get_x() + bar.get_width() / 2, 102,
                f"n={n}", ha="center", va="bottom",
                fontsize=10, color="#555555")

    ax.set_ylabel("Percentage of Responses", fontsize=16)
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, style="italic", rotation=0, ha="center")
    ax.tick_params(axis="y", labelsize=14)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add bracket
    bracket_y = -18
    bracket_color = "#555555"
    ax.annotate("", xy=(steering_start_idx - 0.3, bracket_y), xytext=(steering_end_idx + 0.3, bracket_y),
                arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
                annotation_clip=False)
    ax.annotate("", xy=(steering_start_idx - 0.3, bracket_y), xytext=(steering_start_idx - 0.3, bracket_y + 3),
                arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
                annotation_clip=False)
    ax.annotate("", xy=(steering_end_idx + 0.3, bracket_y), xytext=(steering_end_idx + 0.3, bracket_y + 3),
                arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
                annotation_clip=False)
    ax.text((steering_start_idx + steering_end_idx) / 2, bracket_y - 5, bracket_label,
            ha="center", va="top", fontsize=12, style="italic", color=bracket_color)

    return p1, p2, p3, p4, p5


# %%
# Load original honesty steering results
original_results = []
original_results.append(load_and_aggregate(
    "output/autorater/qwen3-32b/rated_20260125_113414.json",
    "Qwen3-32B\nbaseline"
))

steering_dir = Path("output/autorater/honesty_steering")
steering_files = sorted(steering_dir.glob("rated_*.json"))
original_steering_start = len(original_results)

for f in steering_files:
    with open(f) as fp:
        data = json.load(fp)
    results = data["results"]
    layer = results[0]["steering_layers"][0]
    factor = results[0]["steering_factor"]
    label = f"Qwen3-32B\nL{layer}-F{factor:.0f}"
    original_results.append(load_and_aggregate(f, label))

original_steering_end = len(original_results) - 1

# %%
# Load Chinese honesty steering results
chinese_results = []
chinese_results.append(load_and_aggregate(
    "output/autorater/qwen3-32b/rated_20260125_113414.json",
    "Qwen3-32B\nbaseline"
))

steering_dir = Path("output/autorater/honesty_steering_chinese")
steering_files = sorted(steering_dir.glob("rated_*.json"))
chinese_steering_start = len(chinese_results)

for f in steering_files:
    with open(f) as fp:
        data = json.load(fp)
    results = data["results"]
    layer = results[0]["steering_layers"][0]
    factor = results[0]["steering_factor"]
    label = f"Qwen3-32B\nL{layer}-F{factor:.0f}"
    chinese_results.append(load_and_aggregate(f, label))

chinese_steering_end = len(chinese_results) - 1

# %%
# Create combined plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(26, 9))

# Plot original honesty steering (left)
bars1 = plot_stacked_bars(
    ax1, original_results,
    "Generic Honesty Steering Vector",
    original_steering_start, original_steering_end,
    "honesty steering"
)

# Plot Chinese honesty steering (right)
bars2 = plot_stacked_bars(
    ax2, chinese_results,
    "Chinese Censorship Steering Vector",
    chinese_steering_start, chinese_steering_end,
    "Chinese censorship steering"
)

# Add shared legend
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 0.02),
           ncol=5, framealpha=0.95)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18, wspace=0.15)
plt.savefig("output/plots/autorater_steering_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig("output/plots/autorater_steering_comparison.pdf", bbox_inches="tight")
print("Saved to output/plots/autorater_steering_comparison.png")
plt.show()
