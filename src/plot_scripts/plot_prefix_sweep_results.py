# ABOUTME: Plots per-prefix AUROC and recall by layer from prefix sweep results.
# ABOUTME: Averages metrics over roleplaying and TruthfulQA datasets.

import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
from pathlib import Path

# %%
# Load data
csv_path = "/workspace/projects/chinese_auditing/output/deception_probe_qwen3-vl-8b-thinking-sweep-sys-prompts/prefix_sweep_results.csv"
df = pd.read_csv(csv_path)

# Parse JSON array columns
def parse_json_array(s):
    if pd.isna(s):
        return []
    return json.loads(s)

df['aurocs_rp'] = df['aurocs_by_layer'].apply(parse_json_array)
df['recalls_rp'] = df['recalls_by_layer'].apply(parse_json_array)
df['aurocs_tqa'] = df['aurocs_tqa_by_layer'].apply(parse_json_array)
df['recalls_tqa'] = df['recalls_tqa_by_layer'].apply(parse_json_array)
df['layers_list'] = df['layers'].apply(parse_json_array)

# %%
# Compute averaged metrics (average of roleplaying and TruthfulQA)
num_prefixes = len(df)
num_layers = len(df['layers_list'].iloc[0])
layers = df['layers_list'].iloc[0]

# Initialize arrays
avg_aurocs = np.zeros((num_prefixes, num_layers))
avg_recalls = np.zeros((num_prefixes, num_layers))

for i, row in df.iterrows():
    aurocs_rp = np.array(row['aurocs_rp'])
    aurocs_tqa = np.array(row['aurocs_tqa'])
    recalls_rp = np.array(row['recalls_rp'])
    recalls_tqa = np.array(row['recalls_tqa'])

    avg_aurocs[i] = (aurocs_rp + aurocs_tqa) / 2
    avg_recalls[i] = (recalls_rp + recalls_tqa) / 2

# %%
# Create plots
output_dir = Path("/workspace/projects/chinese_auditing/output/plots")
output_dir.mkdir(parents=True, exist_ok=True)

# Use a colormap for different prefixes
cmap = plt.colormaps.get_cmap('tab10')

# Create side-by-side plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot AUROC
for i in range(num_prefixes):
    ax1.plot(layers, avg_aurocs[i], label=f'Prefix {i}', color=cmap(i), alpha=0.7)

ax1.set_xlabel('Layer', fontsize=18)
ax1.set_ylabel('AUROC', fontsize=18)
ax1.set_title('AUROC by Layer', fontsize=20)
ax1.tick_params(axis='both', labelsize=16)
ax1.set_ylim(0.4, 1.0)
ax1.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='Layer 20')
ax1.grid(True, alpha=0.3)

# Plot Recall
for i in range(num_prefixes):
    ax2.plot(layers, avg_recalls[i] * 100, label=f'Prefix {i}', color=cmap(i), alpha=0.7)

ax2.set_xlabel('Layer', fontsize=18)
ax2.set_ylabel('Recall (%)', fontsize=18)
ax2.set_title('Recall by Layer', fontsize=20)
ax2.tick_params(axis='both', labelsize=16)
ax2.set_ylim(0, 100)
ax2.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='Layer 20')
ax2.grid(True, alpha=0.3)

# Single legend for both plots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', fontsize=12, bbox_to_anchor=(1.08, 0.5))

fig.suptitle('Per-Prefix Metrics (Averaged over RP & TQA)', fontsize=22, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'prefix_sweep_metrics_by_layer.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Saved plot to {output_dir / 'prefix_sweep_metrics_by_layer.png'}")
print(f"Number of prefixes: {num_prefixes}")
print(f"Number of layers: {num_layers}")
