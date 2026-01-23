# ABOUTME: Script to plot refusals and factual lies by prompt ID.
# ABOUTME: Recreates the grouped bar chart style from refusals_lies_by_prompt.png.

# %%
import json

import matplotlib.pyplot as plt
import numpy as np

# %%
# Parameters
rated_file = (
    "/workspace/projects/chinese_auditing/output/autorated/rated_20260123_133213.json"
)
output_path = "/workspace/projects/chinese_auditing/output/plots/refusals_lies_by_prompt_factual.png"

# %%
# Load data
with open(rated_file, "r") as f:
    data = json.load(f)

results = data["results"]
print(f"Total responses: {len(results)}")

# %%
# Aggregate by prompt_id
from collections import defaultdict

stats = defaultdict(lambda: {"refusals": 0, "lies": 0, "total": 0})

for r in results:
    pid = int(r["prompt_id"])
    stats[pid]["total"] += 1
    is_refusal = r.get("is_refusal") or ""
    is_lies = r.get("is_factual_lies") or ""
    if is_refusal.lower() == "yes":
        stats[pid]["refusals"] += 1
    if is_lies.lower() == "yes":
        stats[pid]["lies"] += 1

# Calculate percentages
prompt_ids = sorted(stats.keys())
refusal_pcts = [
    100 * stats[pid]["refusals"] / stats[pid]["total"] for pid in prompt_ids
]
lies_pcts = [100 * stats[pid]["lies"] / stats[pid]["total"] for pid in prompt_ids]

print("\nBy prompt:")
for pid in prompt_ids:
    print(
        f"  {pid}: Refusals={refusal_pcts[pid - 1]:.0f}%, Lies={lies_pcts[pid - 1]:.0f}%"
    )

# %%
# Create plot matching reference style
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(prompt_ids))
width = 0.35

bars1 = ax.bar(x - width / 2, refusal_pcts, width, label="Refusals", color="#1f77b4")
bars2 = ax.bar(x + width / 2, lies_pcts, width, label="Factual Lies", color="#ff7f0e")

ax.set_xlabel("Prompt ID", fontsize=18)
ax.set_ylabel("Percentage (%)", fontsize=18)
n_samples_per_prompt = stats[prompt_ids[0]]["total"]
ax.set_title(
    f"Refusals and Factual Lies by Prompt (n={n_samples_per_prompt} per prompt)",
    fontsize=20,
)
ax.set_xticks(x)
ax.set_xticklabels(prompt_ids, fontsize=16)
ax.tick_params(axis="y", labelsize=16)
ax.set_ylim(0, 105)
ax.legend(fontsize=16, loc="upper right")

# Add percentage labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.annotate(
        f"{height:.0f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14,
    )

for bar in bars2:
    height = bar.get_height()
    ax.annotate(
        f"{height:.0f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=14,
    )

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nSaved: {output_path}")
