# ABOUTME: Interprets SAE features by computing positive logits (W_unembed @ feature_dir).
# ABOUTME: Shows top features per token with their highest-logit tokens, with cached Chinese translation.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"
sae_repo_id = "adamkarvonen/qwen3-32b-saes"
layer = 32  # 50% depth
trainer = 2  # 80 L0, 65k width
top_k_features = 20  # Number of top features per token
top_k_tokens = 10  # Number of top positive logit tokens per feature
enable_thinking = False
enable_translation = True  # Set to True to translate Chinese tokens (uses cache)

# Custom prompt to analyze
prompt = "How many people died during the Tiananmen Square crackdown?"
# Optional: set to None to only analyze prompt, or provide a response string
model_response = None

# %%
import sys

sys.path.insert(0, "/workspace/projects/2_chinese_auditing")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import get_translated_positive_logits, load_sae
from src.utils import apply_chat_template

# %%
# Load the model
print(f"Loading model: {model_name}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded!")

# %%
# Load the SAE
print(f"Loading SAE...")
sae_filename = (
    f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
)

sae = load_sae(
    repo_id=sae_repo_id,
    filename=sae_filename,
    device=device,
    dtype=torch.bfloat16,
)
print("SAE loaded!")

# SAE identifier for translation cache
sae_id = f"layer_{layer}_trainer_{trainer}"

# %%
# Display prompt
print(f"Prompt: {prompt}")
if model_response:
    print(f"Response: {model_response[:100]}...")

# %%
# Apply chat template
formatted_text = apply_chat_template(
    tokenizer,
    prompt,
    model_response,
    enable_thinking=enable_thinking,
)
formatted_text = prompt
print(f"\nChat-formatted text:\n{formatted_text}")

# %%
# Run the text through the model and SAE
print(f"\nProcessing text...")
inputs = tokenizer(formatted_text, return_tensors="pt", add_special_tokens=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Collect activations
submodule = get_submodule(model, layer)
activations = collect_activations(model, submodule, inputs)
print(f"Activations shape: {activations.shape}")

# %%
# Filter outliers
outlier_mask = filter_outlier_tokens(activations)
print(f"Outlier tokens: {outlier_mask.sum().item()}")

# %%
# Encode through SAE (without topk to get all activations)
encoded = sae.encode(activations, use_topk=False, use_threshold=False)
print(f"Encoded shape: {encoded.shape}")

# %%
# Get token strings
token_ids = inputs["input_ids"][0].tolist()
tokens = [tokenizer.decode([t]) for t in token_ids]
print(f"\nTokens ({len(tokens)}): {tokens}")

# %%
# Analyze each non-outlier token
print("\n" + "=" * 80)
print("SAE FEATURE ANALYSIS - TOP FEATURES PER TOKEN")
print("=" * 80)

feature_data = {}  # {pos: {"token": str, "features": [(feat_idx, activation, pos_logits), ...]}}

for pos, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
    if outlier_mask[0, pos]:
        print(f"\n[Position {pos}] '{token_str}' - OUTLIER (skipped)")
        continue

    # Get feature activations for this position
    feature_acts = encoded[0, pos].float()  # [d_sae]

    # Get top-k activated features
    top_vals, top_indices = torch.topk(feature_acts, top_k_features)

    position_features = []
    for feat_idx, feat_val in zip(top_indices.tolist(), top_vals.tolist()):
        if feat_val <= 0:
            continue

        # Get positive logits with cached translation
        pos_logits = get_translated_positive_logits(
            feature_idx=feat_idx,
            sae=sae,
            sae_id=sae_id,
            model=model,
            tokenizer=tokenizer,
            top_k=top_k_tokens,
            enable_translation=enable_translation,
        )

        position_features.append((feat_idx, feat_val, pos_logits))

    feature_data[pos] = {
        "token": token_str,
        "features": position_features,
    }

print(f"\nProcessed {len(feature_data)} token positions")


# %%
def format_positive_logits(
    pos_logits: list[tuple[str, float, str | None]],
    max_tokens: int = 5,
) -> str:
    """Format positive logits with translations."""
    parts = []
    for token, logit_val, trans in pos_logits[:max_tokens]:
        if trans:
            parts.append(f"`{token}` [{trans}]")
        else:
            parts.append(f"`{token}`")
    return ", ".join(parts)


# %%
# Print results
print("\n" + "=" * 80)
print("SAE FEATURE ANALYSIS - RESULTS")
print(f"Prompt: {prompt}")
print(f"Layer: {layer}, Top-k features: {top_k_features}")
print(f"Translation: {'enabled (cached)' if enable_translation else 'disabled'}")
print("=" * 80)

for pos, data in feature_data.items():
    token_str = data["token"]
    features = data["features"]

    print(f"\n### Position {pos}: `{token_str}`")
    print()
    print("| Rank | Feature | Activation | Positive Logits |")
    print("|------|---------|------------|-----------------|")

    for rank, (feat_idx, feat_val, pos_logits) in enumerate(features, 1):
        logits_str = format_positive_logits(pos_logits, max_tokens=10)
        print(f"| {rank} | {feat_idx} | {feat_val:.2f} | {logits_str} |")

# %%
print("\n" + "=" * 80)
print("SAE feature analysis completed!")
