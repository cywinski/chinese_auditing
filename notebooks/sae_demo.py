# ABOUTME: Demo notebook showing how to load and use SAE from adamkarvonen/qwen3-32b-saes.
# ABOUTME: Uses src/sae.py and src/activations.py for the core functionality.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"
sae_repo_id = "adamkarvonen/qwen3-32b-saes"
layer = 32  # 50% depth (32 out of 64 layers)
trainer = 2  # 80 L0, 65k width

# %%
# Imports
import sys

sys.path.insert(0, "/workspace/projects/2_chinese_auditing")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.sae import load_sae

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
sae_filename = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{layer}/trainer_{trainer}/ae.pt"
print(f"Loading SAE from: {sae_repo_id}/{sae_filename}")

sae = load_sae(
    repo_id=sae_repo_id,
    filename=sae_filename,
    device=device,
    dtype=torch.bfloat16,
)
print("SAE loaded!")

# %%
# Test the SAE with a sample input
test_text = "The capital of China is Beijing. The capital of France is Paris."
print(f"Test text: {test_text}")

inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"Input shape: {inputs['input_ids'].shape}")

# %%
# Collect activations from the target layer
submodule = get_submodule(model, layer)
activations = collect_activations(model, submodule, inputs)
print(f"Activations shape: {activations.shape}")  # [batch, seq, d_model]

# %%
# Filter outliers (important for Qwen models!)
outlier_mask = filter_outlier_tokens(activations)
print(f"Median norm: {activations.norm(dim=-1).median().item():.2f}")
print(f"Outlier tokens: {outlier_mask.sum().item()} / {outlier_mask.numel()}")

# %%
# Encode activations through SAE
encoded = sae.encode(activations, use_topk=True)  # [batch, seq, d_sae]
print(f"Encoded shape: {encoded.shape}")
print(f"Average L0 (non-zero features): {(encoded != 0).float().mean(dim=-1).mean().item() * encoded.shape[-1]:.1f}")

# Zero out encoded activations for outlier tokens
encoded_filtered = encoded.clone()
encoded_filtered[outlier_mask] = 0

# %%
# Decode back to original space
decoded = sae.decode(encoded_filtered)  # [batch, seq, d_model]
print(f"Decoded shape: {decoded.shape}")

# %%
# Compute reconstruction quality (excluding outliers and first token)
valid_mask = ~outlier_mask[0].clone()
valid_mask[0] = False  # exclude first token (BOS)
acts_valid = activations[0, valid_mask]
decoded_valid = decoded[0, valid_mask]

# Variance explained (per-feature, then summed)
total_variance = torch.var(acts_valid.float(), dim=0).sum()
residual_variance = torch.var((acts_valid - decoded_valid).float(), dim=0).sum()
variance_explained = 1 - (residual_variance / total_variance)
print(f"Fraction of variance explained: {variance_explained.item():.4f}")

# Cosine similarity (average across tokens)
cossim = torch.nn.functional.cosine_similarity(acts_valid.float(), decoded_valid.float(), dim=-1).mean()
print(f"Cosine similarity: {cossim.item():.4f}")

# MSE
mse = ((acts_valid - decoded_valid).float() ** 2).mean()
print(f"Mean squared error: {mse.item():.6f}")

# %%
# Show top activated features for each token
print("\nTop 5 activated features per token:")
for i, token_id in enumerate(inputs["input_ids"][0]):
    token_str = tokenizer.decode([token_id])
    top_features = torch.topk(encoded[0, i], k=5)
    feature_str = ", ".join(
        [f"{idx.item()}:{val.item():.2f}" for idx, val in zip(top_features.indices, top_features.values)]
    )
    print(f"  '{token_str}': {feature_str}")

# %%
print("\nSAE demo completed!")
