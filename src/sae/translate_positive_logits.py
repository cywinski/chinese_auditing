# ABOUTME: Translates Chinese tokens in SAE positive logits to English using OpenAI Batch API.
# ABOUTME: Uses caching to avoid re-translating tokens and outputs JSON with translations.

import json
import os
import re
from pathlib import Path

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    parse_json_from_response,
    run_batch,
)


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def load_translation_cache(cache_path: str | Path) -> dict[str, str]:
    """Load translation cache from JSON file."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def save_translation_cache(cache: dict[str, str], cache_path: str | Path):
    """Save translation cache to JSON file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_tokens_batch(
    tokens: list[str],
    model: str = "gpt-4.1-mini",
    poll_interval: int = 30,
) -> dict[str, str]:
    """
    Translate Chinese tokens to English using OpenAI Batch API.
    Each unique token is a separate request in one batch job.

    Args:
        tokens: List of tokens to translate (duplicates are removed)
        model: OpenAI model to use
        poll_interval: Seconds between status polls

    Returns:
        Dict mapping Chinese tokens to English translations
    """
    # Deduplicate tokens
    unique_tokens = list(set(tokens))
    if not unique_tokens:
        return {}

    # Create one request per unique token
    requests = []
    for i, token in enumerate(unique_tokens):
        prompt = f"Translate this Chinese text to English. Reply with ONLY the translation, nothing else: {token}"
        requests.append(
            BatchRequest(
                custom_id=f"token_{i}",
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.0,
                max_tokens=50,
            )
        )

    print(f"Translating {len(unique_tokens)} unique tokens in 1 batch...")

    def progress_callback(completed, total, status):
        print(f"  Batch progress: {completed}/{total} ({status})")

    results = run_batch(
        requests=requests,
        description="Translate Chinese tokens",
        poll_interval=poll_interval,
        progress_callback=progress_callback,
    )

    # Parse results - map token to translation
    translations = {}
    for i, result in enumerate(results):
        token = unique_tokens[i]
        if result.error:
            print(f"  Warning: Translation failed for '{token}': {result.error}")
            continue
        if result.content:
            translations[token] = result.content.strip()

    return translations


def get_feature_positive_logits(
    feature_indices: set[int],
    positive_logits_path: str | Path,
    translation_cache: dict[str, str],
    cache_path: str | Path,
    model: str = "gpt-4.1-mini",
    top_k: int = 10,
    enable_translation: bool = True,
) -> dict[int, list[dict]]:
    """
    Get positive logits for specified features, optionally with translations.

    Args:
        feature_indices: Set of feature indices to get logits for
        positive_logits_path: Path to positive logits .pt file
        translation_cache: Existing translation cache
        cache_path: Path to save updated cache
        model: Model for translation
        top_k: Number of top logits per feature
        enable_translation: If True, translate Chinese tokens; if False, keep originals

    Returns:
        Dict mapping feature_idx to list of {token, logit, translation?} dicts
    """
    # Load positive logits
    print(f"Loading positive logits from {positive_logits_path}...")
    data = torch.load(positive_logits_path, weights_only=False)
    all_logits = data["positive_logits"]

    # Collect feature logits
    feature_logits = {}
    chinese_tokens_to_translate = set()

    for feat_idx in feature_indices:
        key = str(feat_idx)
        if key not in all_logits:
            continue

        logits = all_logits[key][:top_k]
        feature_logits[feat_idx] = logits

        if enable_translation:
            for token, _ in logits:
                if contains_chinese(token) and token not in translation_cache:
                    chinese_tokens_to_translate.add(token)

    # Translate new Chinese tokens if enabled
    if enable_translation and chinese_tokens_to_translate:
        print(f"Found {len(chinese_tokens_to_translate)} new Chinese tokens to translate")
        new_translations = translate_tokens_batch(
            list(chinese_tokens_to_translate), model=model
        )
        translation_cache.update(new_translations)
        save_translation_cache(translation_cache, cache_path)
        print(f"Cache updated with {len(new_translations)} new translations")

    # Build output
    result = {}
    for feat_idx, logits in feature_logits.items():
        entries = []
        for token, logit_val in logits:
            entry = {"token": token, "logit": logit_val}
            if enable_translation and contains_chinese(token):
                translation = translation_cache.get(token)
                if translation:
                    entry["translation"] = translation
            entries.append(entry)
        result[feat_idx] = entries

    return result


def process_prompt_features(
    prompt_features_path: str | Path,
    positive_logits_path: str | Path,
    output_path: str | Path,
    cache_path: str | Path,
    model: str = "gpt-4.1-mini",
    top_k_logits: int = 10,
    enable_translation: bool = True,
):
    """
    Process prompt features file and add positive logits (optionally translated).

    Args:
        prompt_features_path: Path to prompt features JSON file
        positive_logits_path: Path to positive logits .pt file
        output_path: Path to save output JSON
        cache_path: Path to translation cache
        model: Model for translation
        top_k_logits: Number of top logits per feature
        enable_translation: If True, translate Chinese tokens; if False, keep originals
    """
    # Load prompt features
    print(f"Loading prompt features from {prompt_features_path}...")
    with open(prompt_features_path) as f:
        prompt_data = json.load(f)

    # Collect all unique feature indices
    all_feature_indices = set()
    for prompt_id, prompt_info in prompt_data.get("prompts", {}).items():
        for token_info in prompt_info.get("tokens", []):
            if token_info.get("is_outlier"):
                continue
            for feat in token_info.get("top_features", []):
                all_feature_indices.add(feat["feature_idx"])

    print(f"Found {len(all_feature_indices)} unique features across all prompts")

    # Load translation cache (even if not translating, for potential future use)
    translation_cache = load_translation_cache(cache_path) if enable_translation else {}
    if enable_translation:
        print(f"Loaded {len(translation_cache)} cached translations")

    # Get positive logits
    feature_positive_logits = get_feature_positive_logits(
        feature_indices=all_feature_indices,
        positive_logits_path=positive_logits_path,
        translation_cache=translation_cache,
        cache_path=cache_path,
        model=model,
        top_k=top_k_logits,
        enable_translation=enable_translation,
    )

    # Build output - add positive logits to each feature
    output = {
        "config": {
            **prompt_data.get("config", {}),
            "positive_logits_path": str(positive_logits_path),
            "top_k_logits": top_k_logits,
            "enable_translation": enable_translation,
            "translation_model": model if enable_translation else None,
        },
        "density_stats": prompt_data.get("density_stats", {}),
        "prompts": {},
    }

    for prompt_id, prompt_info in prompt_data.get("prompts", {}).items():
        prompt_output = {
            "prompt": prompt_info.get("prompt"),
            "topic": prompt_info.get("topic"),
            "level": prompt_info.get("level"),
            "formatted_prompt": prompt_info.get("formatted_prompt"),
            "seq_len": prompt_info.get("seq_len"),
            "n_outliers": prompt_info.get("n_outliers"),
            "tokens": [],
        }

        for token_info in prompt_info.get("tokens", []):
            token_output = {
                "position": token_info["position"],
                "token": token_info["token"],
                "is_outlier": token_info.get("is_outlier", False),
                "top_features": [],
            }

            if not token_info.get("is_outlier"):
                for feat in token_info.get("top_features", []):
                    feat_idx = feat["feature_idx"]
                    feat_output = {
                        "feature_idx": feat_idx,
                        "score": feat["score"],
                        "activation": feat["activation"],
                        "density": feat["density"],
                        "positive_logits": feature_positive_logits.get(feat_idx, []),
                    }
                    token_output["top_features"].append(feat_output)

            prompt_output["tokens"].append(token_output)

        output["prompts"][prompt_id] = prompt_output

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved output to {output_path}")
    print(f"Total features with positive logits: {len(feature_positive_logits)}")


def main(config_path: str):
    """
    Add positive logits to extracted features, optionally translating Chinese tokens.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    process_prompt_features(
        prompt_features_path=cfg.prompt_features_path,
        positive_logits_path=cfg.positive_logits_path,
        output_path=cfg.output_path,
        cache_path=cfg.get("cache_path", "output/translation_cache/positive_logits_cache.json"),
        model=cfg.get("translation_model", "gpt-4.1-mini"),
        top_k_logits=cfg.get("top_k_logits", 10),
        enable_translation=cfg.get("enable_translation", True),
    )


if __name__ == "__main__":
    fire.Fire(main)
