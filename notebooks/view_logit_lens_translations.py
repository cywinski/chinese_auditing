# ABOUTME: Display translated tokens from logit lens results in a table.
# ABOUTME: Loads results JSON and shows Chinese tokens with English translations.

# %%
# Parameters
results_path = "/workspace/projects/chinese_auditing/output/logit_lens_contrastive/contrastive_layer-32.json"

# %%
import json
from pathlib import Path

results_path = Path(results_path)

with open(results_path) as f:
    data = json.load(f)

translations = data.get("translations", {})
config = data.get("config", {})

print(f"Layer: {config.get('layer_idx')}, Mode: {config.get('mode')}")
print(f"Token position: {config.get('token_position')}")
print(f"Total translated tokens: {len(translations)}\n")

# %%
# Display translations table
print("| Token | Translation |")
print("|-------|-------------|")
for token, translation in sorted(translations.items(), key=lambda x: x[1].lower()):
    print(f"| {token} | {translation} |")


# %%
# Display top tokens per prompt with translations
def format_token(token, prob, translations):
    trans = translations.get(token)
    if trans:
        return f"{token}[{trans}] ({prob:.4f})"
    return f"{token} ({prob:.4f})"


print("\n" + "=" * 80)
print("TOP TOKENS PER PROMPT")
print("=" * 80)

for prompt_id, info in list(data["prompts"].items())[:10]:
    print(f"\n[{prompt_id}] {info['prompt'][:70]}...")
    tokens = info.get("top_tokens_translated", info["top_tokens"])[:20]
    for item in tokens:
        if len(item) == 3:
            token, prob, trans = item
            if trans:
                print(f"  {prob:.4f}  {token} [{trans}]")
            else:
                print(f"  {prob:.4f}  {token}")
        else:
            token, prob = item
            trans = translations.get(token)
            if trans:
                print(f"  {prob:.4f}  {token} [{trans}]")
            else:
                print(f"  {prob:.4f}  {token}")

# %%
