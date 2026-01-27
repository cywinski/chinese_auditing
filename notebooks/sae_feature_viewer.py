# ABOUTME: Simple viewer for SAE feature extraction results.
# ABOUTME: Displays top activated features and similar tokens for selected prompts.

# %%
# Parameters
results_dir = "/workspace/projects/2_chinese_auditing/output/sae_features"
positions = [-7, -6, -5, -4, -3, -2, -1]  # Positions to load
layer = 32
prompt_idx = 1  # Index of prompt to display (0-indexed)
max_features = 20  # Number of features to show per position
max_tokens = 10  # Number of similar tokens per feature

# %%
import json
from pathlib import Path

# Load results from all positions
all_data = {}
all_translations = {}

for pos in positions:
    file_path = Path(results_dir) / f"eval_facts_features_p{pos}_layer-{layer}.json"
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
        all_data[pos] = data
        all_translations.update(data.get("translations", {}))
        print(f"Loaded position {pos}: {file_path.name}")
    else:
        print(f"Missing position {pos}: {file_path.name}")

print(f"\nLoaded {len(all_data)} position files")
print(f"Total translations: {len(all_translations)}")

# Get prompt list from first available position
first_pos = list(all_data.keys())[0]
prompts = all_data[first_pos]["prompts"]
print(f"Total prompts: {len(prompts)}")

# %%
# List all prompts
print("\n" + "=" * 80)
print("AVAILABLE PROMPTS")
print("=" * 80)

for i, (pid, info) in enumerate(prompts.items()):
    print(f"[{i:2d}] {info['prompt'][:75]}...")

# %%
# Display selected prompt at ALL positions
prompt_list = list(prompts.items())
prompt_id, _ = prompt_list[prompt_idx]

# Get prompt text from first position
prompt_text = prompts[prompt_id]["prompt"]

print("\n" + "=" * 90)
print(f"PROMPT [{prompt_idx}]: {prompt_text}")
print("=" * 90)

for pos in sorted(all_data.keys(), reverse=True):
    data = all_data[pos]
    prompt_info = data["prompts"].get(prompt_id)

    if not prompt_info:
        continue

    token_repr = repr(prompt_info["token"])
    abs_pos = prompt_info["position"]

    print(f"\n{'─' * 90}")
    print(f"Position {pos} (abs: {abs_pos}) │ Token: {token_repr}")
    print(f"{'─' * 90}")

    if prompt_info.get("is_outlier"):
        print("  [OUTLIER - skipped]")
        continue

    for feat in prompt_info["features"][:max_features]:
        feat_idx = feat["feature_idx"]
        activation = feat["activation"]
        similar = feat["similar_tokens"][:max_tokens]

        # Format similar tokens with translations
        token_parts = []
        for tok, sim in similar:
            trans = all_translations.get(tok)
            if trans:
                token_parts.append(f"'{tok}'[{trans}]")
            else:
                token_parts.append(f"'{tok}'")

        print(
            f"  Feature {feat_idx:5d} (act={activation:6.1f}): {', '.join(token_parts)}"
        )


# %%
# Compare same prompt across positions (alternative view)
def show_position_summary(prompt_idx: int):
    """Show a compact summary of features at each position for a prompt."""
    prompt_id = prompt_list[prompt_idx][0]
    prompt_text = prompts[prompt_id]["prompt"]

    print("\n" + "=" * 90)
    print(f"POSITION SUMMARY FOR PROMPT [{prompt_idx}]")
    print(f"{prompt_text[:80]}...")
    print("=" * 90)

    print(
        f"\n{'Position':<12} {'Token':<15} {'Top Feature':<12} {'Act':>8} {'Similar Tokens'}"
    )
    print("-" * 90)

    for pos in sorted(all_data.keys(), reverse=True):
        info = all_data[pos]["prompts"].get(prompt_id)
        if not info or info.get("is_outlier"):
            continue

        token = repr(info["token"])[:12]
        if info["features"]:
            feat = info["features"][0]
            similar = [t for t, _ in feat["similar_tokens"][:3]]
            similar_str = ", ".join([f"'{t}'" for t in similar])[:40]
            print(
                f"{pos:<12} {token:<15} {feat['feature_idx']:<12} {feat['activation']:>8.1f} {similar_str}"
            )


show_position_summary(prompt_idx)


# %%
# Find prompts with specific feature activated at any position
def find_prompts_with_feature(
    feature_idx: int, position: int = -1, min_activation: float = 0
):
    """Find all prompts where a specific feature is highly activated at given position."""
    print(f"\n" + "=" * 80)
    print(
        f"PROMPTS WITH FEATURE {feature_idx} AT POSITION {position} (min act: {min_activation})"
    )
    print("=" * 80)

    if position not in all_data:
        print(f"Position {position} not loaded.")
        return

    data = all_data[position]
    matches = []

    for i, (pid, info) in enumerate(data["prompts"].items()):
        if info.get("is_outlier"):
            continue
        for feat in info["features"]:
            if (
                feat["feature_idx"] == feature_idx
                and feat["activation"] >= min_activation
            ):
                matches.append((i, info, feat["activation"]))
                break

    matches.sort(key=lambda x: -x[2])

    if not matches:
        print("No matches found.")
        return

    print(f"Found {len(matches)} prompts:\n")
    for i, info, act in matches[:15]:
        print(f"[{i:2d}] (act={act:6.1f}) {info['prompt'][:65]}...")


# Example: Find prompts with feature 23931 at position -1
find_prompts_with_feature(23931, position=-1, min_activation=90)


# %%
# Get top features at each position (aggregated across all prompts)
def show_top_features_by_position():
    """Show top features at each position aggregated across all prompts."""
    print("\n" + "=" * 90)
    print("TOP FEATURES BY POSITION (aggregated across all prompts)")
    print("=" * 90)

    for pos in sorted(all_data.keys(), reverse=True):
        data = all_data[pos]
        feature_totals = {}

        for pid, info in data["prompts"].items():
            if info.get("is_outlier"):
                continue
            for feat in info["features"]:
                idx = feat["feature_idx"]
                if idx not in feature_totals:
                    feature_totals[idx] = {
                        "total": 0,
                        "similar": feat["similar_tokens"][:3],
                    }
                feature_totals[idx]["total"] += feat["activation"]

        # Get sample token for this position
        sample_info = list(data["prompts"].values())[0]
        token_repr = repr(sample_info["token"])

        print(f"\n Position {pos} | Token: {token_repr}")
        print("-" * 80)

        sorted_feats = sorted(feature_totals.items(), key=lambda x: -x[1]["total"])[:5]
        for feat_idx, stats in sorted_feats:
            similar = stats["similar"]
            tok_parts = []
            for tok, sim in similar:
                trans = all_translations.get(tok)
                if trans:
                    tok_parts.append(f"'{tok}'[{trans}]")
                else:
                    tok_parts.append(f"'{tok}'")
            print(
                f"  Feature {feat_idx:5d} (total={stats['total']:7.1f}): {', '.join(tok_parts)}"
            )


show_top_features_by_position()

# %%
