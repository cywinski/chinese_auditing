# ABOUTME: Interactive notebook for logit lens analysis on Qwen model.
# ABOUTME: Loads prompts and displays per-layer token predictions.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"

# %%
import sys

sys.path.insert(0, "/workspace/projects/chinese_auditing")

import matplotlib.pyplot as plt
import torch

from src.utils import load_model
from src.logit_lens import logit_lens, print_logit_lens_results, get_logit_lens_probs

# %%
# Load model
print(f"Loading {model_name}...")
model, tokenizer = load_model(model_name)
print(f"Loaded. Layers: {model.config.num_hidden_layers}")

# %%
prompt = "How many people died during the Tiananmen Square crackdown?"
generated_response = ""  # Optional: set to analyze prompt + response
top_k = 10

# %%
# Run logit lens (computes all layers and positions)
data = logit_lens(
    model,
    tokenizer,
    prompt=prompt,
    response=generated_response,
    top_k=top_k,
)

print(f"Formatted prompt:\n{data['formatted_prompt']}\n")
print(f"Tokens: {len(data['tokens'])}, Layers: {data['num_layers']}")


# %%
# Position -> token map
def escape(s):
    return s.replace("\n", "\\n").replace("\t", "\\t")


for i, tok in enumerate(data["tokens"]):
    print(f"{i:3d}: '{escape(tok)}'")

# %%
# Display results (filter positions/layers here without recomputing)
positions_to_show = None  # None = all, or list like [0, 5, -1]
layers_to_show = [15, 31, 47, 63]  # None = all, or list like [0, 31, 63]

print_logit_lens_results(data, positions=positions_to_show, layers=layers_to_show)

# %%
# Get raw probabilities: (n_tokens, n_layers, vocab_size)
probs, tokens, formatted = get_logit_lens_probs(
    model, tokenizer, prompt=prompt, response=generated_response
)
print(f"Probs shape: {probs.shape}")

# %%
# Heatmap of probability for a specific token across all positions and layers
target_token = "中共中央"
target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
print(f"Target token: {target_token!r}, token_id: {target_token_id}")

# Extract probabilities for target token: (n_tokens, n_layers)
target_probs = probs[:, :, target_token_id].float().cpu().numpy()

# Create heatmap
fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(target_probs.T, aspect="auto", cmap="viridis", origin="lower")

# X-axis: tokens
ax.set_xlabel("Token Position", fontsize=16)
ax.set_xticks(range(len(tokens)))
token_labels = [t.replace("\n", "\\n").replace("\t", "\\t")[:8] for t in tokens]
ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=10)

# Y-axis: layers
ax.set_ylabel("Layer", fontsize=16)
ax.set_yticks(range(0, probs.shape[1], 8))
ax.set_yticklabels(range(0, probs.shape[1], 8), fontsize=12)

ax.set_title(
    f"Probability of '{target_token}' across layers and positions", fontsize=18
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Probability", fontsize=14)

plt.tight_layout()
# plt.savefig("output/plots/logit_lens_heatmap.png", dpi=150)
plt.show()

# %%
