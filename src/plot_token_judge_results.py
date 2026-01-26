# ABOUTME: Plot token judgment results comparing CCP-sensitive vs control prompts.
# ABOUTME: Creates bar charts showing CCP-related and deception-related token percentages.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Load data
sens_dir = Path("output/token_judge")
ctrl_dir = Path("output/token_judge_control")

configs = ["p-6_layer-32", "p-6_layer-48", "p-7_layer-32", "p-7_layer-48"]

sens_ccp = []
ctrl_ccp = []
sens_dec = []
ctrl_dec = []

for name in configs:
    sens_file = sens_dir / f"judgments_token_position_results_{name}.json"
    ctrl_file = ctrl_dir / f"judgments_token_position_results_{name}.json"

    sens_data = json.load(open(sens_file))
    ctrl_data = json.load(open(ctrl_file))

    sens_ccp.append(sens_data["summary"]["pct_ccp_related"])
    ctrl_ccp.append(ctrl_data["summary"]["pct_ccp_related"])
    sens_dec.append(sens_data["summary"]["pct_deception_related"])
    ctrl_dec.append(ctrl_data["summary"]["pct_deception_related"])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

x = np.arange(len(configs))
width = 0.35

# Plot CCP-related tokens
bars1 = ax1.bar(x - width / 2, sens_ccp, width, label="Sensitive (CCP topics)", color="#d62728", edgecolor="black", linewidth=1.5)
bars2 = ax1.bar(x + width / 2, ctrl_ccp, width, label="Control (neutral topics)", color="#7fbf7f", edgecolor="black", linewidth=1.5, hatch="//")

ax1.set_ylabel("Percentage of Tokens (%)", fontsize=16)
ax1.set_xlabel("Position / Layer", fontsize=16)
ax1.set_title("CCP-Related Tokens", fontsize=18, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=14)
ax1.legend(fontsize=14, loc="upper left")
ax1.set_ylim(0, 4)
ax1.tick_params(axis="y", labelsize=14)
ax1.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=12)
for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=12)

# Plot Deception-related tokens
bars3 = ax2.bar(x - width / 2, sens_dec, width, label="Sensitive (CCP topics)", color="#d62728", edgecolor="black", linewidth=1.5)
bars4 = ax2.bar(x + width / 2, ctrl_dec, width, label="Control (neutral topics)", color="#7fbf7f", edgecolor="black", linewidth=1.5, hatch="//")

ax2.set_ylabel("Percentage of Tokens (%)", fontsize=16)
ax2.set_xlabel("Position / Layer", fontsize=16)
ax2.set_title("Deception-Related Tokens", fontsize=18, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels([c.replace("_", "\n") for c in configs], fontsize=14)
ax2.legend(fontsize=14, loc="upper left")
ax2.set_ylim(0, 4)
ax2.tick_params(axis="y", labelsize=14)
ax2.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=12)
for bar in bars4:
    height = bar.get_height()
    ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=12)

plt.suptitle("Logit Lens Token Classification: Sensitive vs Control Prompts",
             fontsize=20, fontweight="bold", y=1.02)
plt.tight_layout()

output_path = Path("output/plots/token_judge_comparison.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved plot to {output_path}")

plt.show()
