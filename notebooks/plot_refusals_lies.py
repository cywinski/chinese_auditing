# ABOUTME: Script to plot response categories (refusals, factually incorrect, factually correct) by prompt.
# ABOUTME: Creates stacked bar chart where each prompt's categories sum to 100%.

# %%
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# %%
# Parameters
rated_file = (
    "/workspace/projects/chinese_auditing/output/autorated/rated_20260123_104132.json"
)
output_path = (
    "/workspace/projects/chinese_auditing/output/plots/refusals_lies_by_prompt.png"
)

# %%
# Load data
with open(rated_file, "r") as f:
    data = json.load(f)

results = data["results"]
print(f"Total responses: {len(results)}")

# %%
# Aggregate by prompt_id into mutually exclusive categories
stats = defaultdict(lambda: {"refusals": 0, "incorrect": 0, "correct": 0, "total": 0})

for r in results:
    pid = int(r["prompt_id"])
    stats[pid]["total"] += 1
    is_refusal = (r.get("is_refusal") or "").lower() == "yes"
    is_lies = (r.get("is_factual_lies") or "").lower() == "yes"

    if is_refusal:
        stats[pid]["refusals"] += 1
    elif is_lies:
        stats[pid]["incorrect"] += 1
    else:
        stats[pid]["correct"] += 1

# Calculate percentages
prompt_ids = sorted(stats.keys())
refusal_pcts = [
    100 * stats[pid]["refusals"] / stats[pid]["total"] for pid in prompt_ids
]
incorrect_pcts = [
    100 * stats[pid]["incorrect"] / stats[pid]["total"] for pid in prompt_ids
]
correct_pcts = [100 * stats[pid]["correct"] / stats[pid]["total"] for pid in prompt_ids]

print("\nBy prompt:")
for i, pid in enumerate(prompt_ids):
    print(
        f"  {pid}: Refusals={refusal_pcts[i]:.0f}%, "
        f"Incorrect={incorrect_pcts[i]:.0f}%, Correct={correct_pcts[i]:.0f}%"
    )

# %%
# Create stacked bar chart
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(prompt_ids))
width = 0.6

# Stack bars: refusals at bottom, then incorrect, then correct on top
bars1 = ax.bar(x, refusal_pcts, width, label="Refusals", color="#d62728")
bars2 = ax.bar(
    x,
    incorrect_pcts,
    width,
    bottom=refusal_pcts,
    label="Factually Incorrect",
    color="#ff7f0e",
)
bars3 = ax.bar(
    x,
    correct_pcts,
    width,
    bottom=[r + i for r, i in zip(refusal_pcts, incorrect_pcts)],
    label="Factually Correct",
    color="#2ca02c",
)

ax.set_xlabel("Prompt ID", fontsize=18)
ax.set_ylabel("Percentage (%)", fontsize=18)
n_samples_per_prompt = stats[prompt_ids[0]]["total"]
ax.set_title(
    f"Response Categories by Prompt (n={n_samples_per_prompt} per prompt)",
    fontsize=20,
)
ax.set_xticks(x)
ax.set_xticklabels(prompt_ids, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim(0, 105)
ax.legend(fontsize=16, loc="upper right")


# Add percentage labels in the middle of each segment (only if >= 5%)
def add_labels(bars, bottoms):
    for bar, bottom in zip(bars, bottoms):
        height = bar.get_height()
        if height >= 5:
            ax.annotate(
                f"{height:.0f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bottom + height / 2),
                ha="center",
                va="center",
                fontsize=14,
                color="white",
                fontweight="bold",
            )


add_labels(bars1, [0] * len(prompt_ids))
add_labels(bars2, refusal_pcts)
add_labels(bars3, [r + i for r, i in zip(refusal_pcts, incorrect_pcts)])

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {output_path}")
