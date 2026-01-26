# ABOUTME: Creates autorater results plot for fuzzing experiments.
# ABOUTME: Shows stacked bars (Refusal, Not Mentioned, Lie, Mentioned) for baselines + fuzzing.

# %%
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# %%
def load_and_aggregate(filepath, label):
    """Load autorater results and compute aggregate metrics."""
    with open(filepath) as f:
        data = json.load(f)
    results = data["results"]

    n_total = len(results)
    n_refusals = sum(1 for r in results if r.get("is_refusal") == "yes")

    total_facts = sum(r.get("facts_total", 0) for r in results)
    facts_mentioned = sum(r.get("facts_mentioned", 0) for r in results)
    facts_lies = sum(r.get("facts_lies", 0) for r in results)

    # Calculate as percentage of total facts
    refusal_pct = 100 * n_refusals / n_total
    mentioned_pct = 100 * facts_mentioned / total_facts if total_facts > 0 else 0
    lie_pct = 100 * facts_lies / total_facts if total_facts > 0 else 0
    not_mentioned_pct = 100 - refusal_pct - mentioned_pct - lie_pct

    return {
        "label": label,
        "refusal": refusal_pct,
        "not_mentioned": not_mentioned_pct,
        "lie": lie_pct,
        "mentioned": mentioned_pct,
    }


# Load all results
all_results = []

# Baselines
all_results.append(load_and_aggregate(
    "output/autorater/qwen3-32b/rated_20260125_113414.json",
    "qwen3-32b"
))
all_results.append(load_and_aggregate(
    "output/autorater/llama-3.3-70b-instruct/rated_20260125_113448.json",
    "llama-3.3-70b-instruct"
))

# Fuzzing results
fuzzing_dir = Path("output/autorater/fuzzing")
fuzzing_files = sorted(fuzzing_dir.glob("rated_*.json"))
fuzzing_start_idx = len(all_results)

for f in fuzzing_files:
    with open(f) as fp:
        data = json.load(fp)
    results = data["results"]
    layer = results[0]["fuzz_layers"][0]
    magnitude = results[0]["fuzz_magnitude"]
    label = f"Qwen3-32B\nL{layer}-M{magnitude:.0f}"
    all_results.append(load_and_aggregate(f, label))

fuzzing_end_idx = len(all_results) - 1

# %%
# Create the plot
fig, ax = plt.subplots(figsize=(18, 9))

labels = [r["label"] for r in all_results]
x = np.arange(len(labels))
width = 0.65

# Colors matching the reference image
colors = {
    "refusal": "#c0392b",      # dark red
    "not_mentioned": "#e67e22", # orange
    "lie": "#8e44ad",          # purple
    "mentioned": "#27ae60",    # green
}

# Extract data
refusals = [r["refusal"] for r in all_results]
not_mentioneds = [r["not_mentioned"] for r in all_results]
lies = [r["lie"] for r in all_results]
mentioneds = [r["mentioned"] for r in all_results]

# Stacked bars (order: refusal, not_mentioned, lie, mentioned)
bottom = np.zeros(len(labels))

p1 = ax.bar(x, refusals, width, label="Refusal", color=colors["refusal"], bottom=bottom)
bottom += refusals

p2 = ax.bar(x, not_mentioneds, width, label="Not Mentioned", color=colors["not_mentioned"], bottom=bottom)
bottom += not_mentioneds

p3 = ax.bar(x, lies, width, label="Lie", color=colors["lie"], bottom=bottom)
bottom += lies

p4 = ax.bar(x, mentioneds, width, label="Mentioned", color=colors["mentioned"], bottom=bottom)

# Add percentage labels inside the bars
def add_label(bars, values, bottoms, min_height=5):
    for bar, val, bot in zip(bars, values, bottoms):
        if val >= min_height:
            ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                    f"{val:.1f}%", ha="center", va="center",
                    color="white", fontsize=14, fontweight="bold")

add_label(p1, refusals, np.zeros(len(labels)))
add_label(p2, not_mentioneds, np.array(refusals))
add_label(p3, lies, np.array(refusals) + np.array(not_mentioneds))
add_label(p4, mentioneds, np.array(refusals) + np.array(not_mentioneds) + np.array(lies))

# Formatting
ax.set_ylabel("Percentage of Facts", fontsize=20)
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, style="italic", rotation=0, ha="center")
ax.tick_params(axis="y", labelsize=16)
ax.legend(fontsize=16, loc="upper right", framealpha=0.95)

# Remove top and right spines
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Add bracket for "fuzzing" annotation
bracket_y = -18
bracket_color = "#555555"
# Draw bracket line
ax.annotate("", xy=(fuzzing_start_idx - 0.3, bracket_y), xytext=(fuzzing_end_idx + 0.3, bracket_y),
            arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
            annotation_clip=False)
# Draw vertical ends
ax.annotate("", xy=(fuzzing_start_idx - 0.3, bracket_y), xytext=(fuzzing_start_idx - 0.3, bracket_y + 3),
            arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
            annotation_clip=False)
ax.annotate("", xy=(fuzzing_end_idx + 0.3, bracket_y), xytext=(fuzzing_end_idx + 0.3, bracket_y + 3),
            arrowprops=dict(arrowstyle="-", color=bracket_color, lw=1.5),
            annotation_clip=False)
# Add text
ax.text((fuzzing_start_idx + fuzzing_end_idx) / 2, bracket_y - 5, "fuzzing",
        ha="center", va="top", fontsize=14, style="italic", color=bracket_color)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)
plt.savefig("output/plots/autorater_fuzzing_results.png", dpi=150, bbox_inches="tight")
plt.savefig("output/plots/autorater_fuzzing_results.pdf", bbox_inches="tight")
print("Saved to output/plots/autorater_fuzzing_results.png")
plt.show()
