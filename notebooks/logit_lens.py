# ABOUTME: Interactive notebook for logit lens analysis on Qwen model.
# ABOUTME: Loads prompts and displays per-layer token predictions.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"
prompt = "What happened at Tiananmen Square in 1989?"
generated_response = None  # Optional: set to analyze prompt + response
selected_positions = None  # None = all, or list like [0, 5, -1]
top_k = 5
layers_to_show = None  # None = all, or list like [0, 31, 63]

# %%
import sys
sys.path.insert(0, "/workspace/projects/chinese_auditing")

from src.utils import load_model
from src.logit_lens import logit_lens, print_logit_lens_results

# %%
# Load model
print(f"Loading {model_name}...")
model, tokenizer = load_model(model_name)
print(f"Loaded. Layers: {model.config.num_hidden_layers}")

# %%
# Run logit lens
data = logit_lens(
    model,
    tokenizer,
    prompt=prompt,
    response=generated_response,
    positions=selected_positions,
    layers=layers_to_show,
    top_k=top_k,
)

print(f"Formatted prompt:\n{data['formatted_prompt']}\n")
print(f"Tokens: {len(data['tokens'])}")

# %%
# Display results
print_logit_lens_results(data, show_all_positions=False)
