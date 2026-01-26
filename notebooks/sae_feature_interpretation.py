# ABOUTME: Interprets SAE features by computing cosine similarity to token embeddings.
# ABOUTME: Shows most similar tokens for each activated feature, with Chinese translation.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"
sae_repo_id = "adamkarvonen/qwen3-32b-saes"
layer = 32  # 50% depth
trainer = 2  # 80 L0, 65k width
eval_facts_path = "/workspace/projects/2_chinese_auditing/data/eval_facts.json"
top_k_features = 20  # Number of top features to interpret
top_k_tokens = 10  # Number of most similar tokens per feature

# %%
import asyncio
import json
import sys

sys.path.insert(0, "/workspace/projects/2_chinese_auditing")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.logit_lens import contains_chinese, translate_tokens_batch
from src.sae import interpret_features_batch, load_sae

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

# %%
# Load prompt from eval_facts.json and apply chat template
print(f"Loading prompts from: {eval_facts_path}")
with open(eval_facts_path, "r") as f:
    eval_data = json.load(f)

# Extract first question from the data
prompts = []
for topic_key, topic_value in eval_data.items():
    if topic_key == "metadata":
        continue
    for subtopic_key, questions in topic_value.items():
        for q in questions:
            prompts.append(
                {
                    "question": q["question"],
                    "topic": f"{topic_key}/{subtopic_key}",
                    "level": q.get("level", "unknown"),
                }
            )

# Select an example prompt (first one by default)
selected_prompt = prompts[1]
print(f"Selected prompt: {selected_prompt['question']}")
print(f"Topic: {selected_prompt['topic']}, Level: {selected_prompt['level']}")

# Apply chat template
messages = [{"role": "user", "content": selected_prompt["question"]}]
test_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(f"\nChat-formatted text:\n{test_text}")

# %%
# Get token embeddings from the model (unnormalized)
print("Getting token embeddings...")
embed_weights = (
    model.model.embed_tokens.weight.detach().float()
)  # [vocab_size, d_model]
print(f"Embedding shape: {embed_weights.shape}")

# Precompute embedding norms for cosine similarity
embed_norms = embed_weights.norm(dim=-1, keepdim=True)  # [vocab_size, 1]

# %%
# Get SAE decoder weights (unnormalized)
W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
print(f"Decoder shape: {W_dec.shape}")

# %%
# Run the test text through the model and SAE
print(f"\nTest text: {test_text}")
inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Collect activations
submodule = get_submodule(model, layer)
activations = collect_activations(model, submodule, inputs)
print(f"Activations shape: {activations.shape}")
# display norms
print(f"Activations norms: {activations.norm(dim=-1)}")
# %%
# Filter outliers
outlier_mask = filter_outlier_tokens(activations)
print(f"Outlier tokens: {outlier_mask.sum().item()}")
# %%
# Encode through SAE
encoded = sae.encode(activations, use_topk=True)
print(f"Encoded shape: {encoded.shape}")

# %%
# Get token strings
token_ids = inputs["input_ids"][0].tolist()
tokens = [tokenizer.decode([t]) for t in token_ids]
print(f"\nTokens: {tokens}")

# %%
# For each token, find top activated features and their most similar tokens


def get_most_similar_tokens(
    feature_idx: int, top_k: int = 10
) -> list[tuple[str, float]]:
    """Get tokens most similar to a feature's decoder direction using cosine similarity."""
    feature_dir = W_dec[feature_idx]  # [d_model]
    feature_dir_gpu = feature_dir.to(embed_weights.device)
    feature_norm = feature_dir_gpu.norm()

    # Compute dot product
    dot_products = embed_weights @ feature_dir_gpu  # [vocab_size]

    # Normalize to get cosine similarity: (a · b) / (||a|| * ||b||)
    similarities = dot_products / (embed_norms.squeeze() * feature_norm)

    top_sims, top_indices = torch.topk(similarities, top_k)

    results = []
    for idx, sim in zip(top_indices.tolist(), top_sims.tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, sim))

    return results


# %%
# Analyze each non-outlier token
print("\n" + "=" * 80)
print("SAE FEATURE INTERPRETATION")
print("=" * 80)

all_tokens_to_translate = set()
feature_interpretations = {}

for pos, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
    if outlier_mask[0, pos]:
        print(f"\n[Position {pos}] '{token_str}' - OUTLIER (skipped)")
        continue

    print(f"\n[Position {pos}] '{token_str}'")
    print("-" * 60)

    # Get feature activations for this position
    feature_acts = encoded[0, pos]  # [d_sae]

    # Get top-k activated features
    top_vals, top_indices = torch.topk(feature_acts, top_k_features)

    for rank, (feat_idx, feat_val) in enumerate(
        zip(top_indices.tolist(), top_vals.tolist())
    ):
        if feat_val <= 0:
            continue

        # Get most similar tokens to this feature
        similar_tokens = get_most_similar_tokens(feat_idx, top_k_tokens)

        # Collect tokens for translation
        for tok, _ in similar_tokens:
            if contains_chinese(tok):
                all_tokens_to_translate.add(tok)

        # Store for later display with translations
        feature_interpretations[(pos, rank)] = {
            "token": token_str,
            "feature_idx": feat_idx,
            "activation": feat_val,
            "similar_tokens": similar_tokens,
        }

        # Print without translations for now
        token_strs = [f"'{t}' ({s:.3f})" for t, s in similar_tokens[:5]]
        print(f"  Feature {feat_idx:5d} (act={feat_val:6.2f}): {', '.join(token_strs)}")

# %%
# Translate Chinese tokens
print("\n" + "=" * 80)
print("TRANSLATING CHINESE TOKENS...")
print("=" * 80)

if all_tokens_to_translate:
    translations = asyncio.run(translate_tokens_batch(list(all_tokens_to_translate)))
    print(f"Translated {len(translations)} tokens")
else:
    translations = {}
    print("No Chinese tokens to translate")

# %%
# Get LLM interpretations for each unique feature
print("\n" + "=" * 80)
print("GENERATING FEATURE INTERPRETATIONS...")
print("=" * 80)

# Collect unique features
unique_features = {}
for (pos, rank), info in feature_interpretations.items():
    feat_idx = info["feature_idx"]
    if feat_idx not in unique_features:
        unique_features[feat_idx] = info["similar_tokens"]

print(f"Interpreting {len(unique_features)} unique features...")
features_list = [(idx, tokens) for idx, tokens in unique_features.items()]
llm_interpretations = asyncio.run(interpret_features_batch(features_list, translations))
print("Done!")

# %%
# Print results with translations and LLM interpretations
print("\n" + "=" * 80)
print("SAE FEATURE INTERPRETATION (WITH LLM DESCRIPTIONS)")
print("=" * 80)

current_pos = -1
for (pos, rank), info in feature_interpretations.items():
    if pos != current_pos:
        current_pos = pos
        print(f"\n[Position {pos}] '{info['token']}'")
        print("-" * 60)

    feat_idx = info["feature_idx"]
    feat_val = info["activation"]
    similar_tokens = info["similar_tokens"]
    llm_interp = llm_interpretations.get(feat_idx, "")

    # Format tokens with translations
    token_parts = []
    for tok, sim in similar_tokens[:10]:
        trans = translations.get(tok)
        if trans:
            token_parts.append(f"'{tok}'[{trans}] ({sim:.3f})")
        else:
            token_parts.append(f"'{tok}' ({sim:.3f})")

    print(f"  Feature {feat_idx:5d} [{llm_interp}] (act={feat_val:6.2f})")
    print(f"    Tokens: {', '.join(token_parts)}")

# %%
print("\n" + "=" * 80)
print("SAE feature interpretation completed!")
