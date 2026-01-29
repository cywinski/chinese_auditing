# ABOUTME: Sparse Autoencoder (SAE) implementation for loading and using BatchTopK SAEs.
# ABOUTME: Supports loading SAEs from HuggingFace Hub (e.g., adamkarvonen/qwen3-32b-saes).

import json
import os
import re
from pathlib import Path

import httpx
import requests
import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Cache directory for translations
TRANSLATION_CACHE_DIR = Path(__file__).parent.parent / "output" / "translation_cache"


class BatchTopKSAE(nn.Module):
    """Sparse Autoencoder using Batch TopK activation sparsity."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        k: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.register_buffer("k", torch.tensor(k))

        # Learnable parameters
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

        # Threshold for inference (alternative to topk)
        self.threshold = None

    def encode(
        self, x: torch.Tensor, use_topk: bool = True, use_threshold: bool = True
    ) -> torch.Tensor:
        """Encode input activations to sparse feature activations.

        Args:
            x: Input activations [batch, seq, d_in]
            use_topk: If True, use topk sparsification (recommended for inference).
                      If False, use threshold-based sparsification.
        """
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        post_acts = torch.relu(pre_acts)

        if use_topk:
            k = self.k.item()
            topk_values, topk_indices = torch.topk(post_acts, k=k, dim=-1)
            sparse_acts = torch.zeros_like(post_acts)
            sparse_acts.scatter_(-1, topk_indices, topk_values)
            post_acts = sparse_acts
        elif use_threshold and self.threshold is not None:
            post_acts = post_acts * (post_acts > self.threshold)

        return post_acts

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decode sparse feature activations back to original space."""
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: encode then decode."""
        feature_acts = self.encode(x)
        reconstructed = self.decode(feature_acts)
        return reconstructed, feature_acts


def load_sae(
    repo_id: str,
    filename: str,
    device: torch.device,
    dtype: torch.dtype,
) -> BatchTopKSAE:
    """Load a BatchTopK SAE from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repo ID (e.g., "adamkarvonen/qwen3-32b-saes")
        filename: Path to ae.pt file within the repo
        device: Device to load the SAE on
        dtype: Data type for the SAE weights

    Returns:
        Loaded BatchTopKSAE instance
    """
    # Download the SAE weights
    ae_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Download the config
    config_filename = filename.replace("ae.pt", "config.json")
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename)

    with open(config_path) as f:
        config = json.load(f)

    # Extract dimensions from config (nested under "trainer" key)
    trainer_config = config.get("trainer", config)
    d_in = trainer_config["activation_dim"]
    d_sae = trainer_config["dict_size"]
    k = trainer_config["k"]

    print(f"Loading SAE: d_in={d_in}, d_sae={d_sae}, k={k}")

    # Create SAE instance
    sae = BatchTopKSAE(d_in=d_in, d_sae=d_sae, k=k, device=device, dtype=dtype)

    # Load weights
    state_dict = torch.load(ae_path, map_location=device, weights_only=True)

    # Remap keys from dictionary_learning format
    key_mapping = {
        "encoder.weight": "W_enc",
        "encoder.bias": "b_enc",
        "decoder.weight": "W_dec",
        "decoder.bias": "b_dec",
    }

    new_state_dict = {}
    for old_key, new_key in key_mapping.items():
        if old_key in state_dict:
            tensor = state_dict[old_key].to(dtype)
            # Transpose weight matrices (nn.Linear stores as [out, in])
            if "weight" in old_key:
                tensor = tensor.T
            new_state_dict[new_key] = tensor

    sae.load_state_dict(new_state_dict, strict=False)

    print(state_dict.keys())

    # Set threshold from checkpoint if available
    if "threshold" in state_dict:
        sae.threshold = state_dict["threshold"].item()
        print(f"Using threshold from checkpoint: {sae.threshold}")

    return sae


def get_positive_logits_for_feature(
    feature_idx: int,
    sae: BatchTopKSAE,
    model,
    tokenizer,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Get tokens with highest positive logit contribution for a single feature.

    Args:
        feature_idx: Index of the SAE feature
        sae: The SAE instance
        model: The language model (needs lm_head.weight)
        tokenizer: The tokenizer for decoding tokens
        top_k: Number of top tokens to return

    Returns:
        List of (token_str, logit_value) tuples
    """
    with torch.no_grad():
        W_unembed = model.lm_head.weight.detach().float()
        W_dec = sae.W_dec.detach().float()

        feature_dir = W_dec[feature_idx].to(W_unembed.device)
        logit_effects = W_unembed @ feature_dir

        top_vals, top_indices = torch.topk(logit_effects, top_k)

        results = []
        for idx, logit_val in zip(top_indices.tolist(), top_vals.tolist()):
            token_str = tokenizer.decode([idx])
            results.append((token_str, logit_val))

    return results


def get_positive_logits_for_features(
    feature_indices: list[int],
    sae: BatchTopKSAE,
    model,
    tokenizer,
    top_k: int = 10,
) -> dict[int, list[tuple[str, float]]]:
    """Get tokens with highest positive logit contribution for multiple features.

    Args:
        feature_indices: List of SAE feature indices
        sae: The SAE instance
        model: The language model (needs lm_head.weight)
        tokenizer: The tokenizer for decoding tokens
        top_k: Number of top tokens per feature

    Returns:
        Dict mapping feature_idx to list of (token_str, logit_value) tuples
    """
    with torch.no_grad():
        W_unembed = model.lm_head.weight.detach().float()
        W_dec = sae.W_dec.detach().float()

        results = {}
        for feat_idx in feature_indices:
            feature_dir = W_dec[feat_idx].to(W_unembed.device)
            logit_effects = W_unembed @ feature_dir

            top_vals, top_indices = torch.topk(logit_effects, top_k)

            tokens = []
            for idx, logit_val in zip(top_indices.tolist(), top_vals.tolist()):
                token_str = tokenizer.decode([idx])
                tokens.append((token_str, logit_val))
            results[feat_idx] = tokens

    return results


def _contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _get_translation_cache_path(sae_id: str) -> Path:
    """Get the path to the translation cache file for an SAE."""
    TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = sae_id.replace("/", "_").replace("\\", "_")
    return TRANSLATION_CACHE_DIR / f"translations_{safe_id}.json"


def _load_translation_cache(sae_id: str) -> dict[str, dict[str, str]]:
    """Load translation cache for an SAE. Returns {feature_idx_str: {token: translation}}."""
    cache_path = _get_translation_cache_path(sae_id)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def _save_translation_cache(sae_id: str, cache: dict[str, dict[str, str]]):
    """Save translation cache for an SAE."""
    cache_path = _get_translation_cache_path(sae_id)
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def translate_tokens_sync(
    tokens: list[str],
    max_retries: int = 100,
) -> dict[str, str]:
    """Translate Chinese tokens synchronously using GPT-4.1.

    Args:
        tokens: List of tokens to translate
        max_retries: Number of retries on failure

    Returns:
        Dict mapping Chinese tokens to their English translations
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    chinese_tokens = list(set(t for t in tokens if _contains_chinese(t)))
    if not chinese_tokens:
        return {}

    print(f"Translating {len(chinese_tokens)} unique Chinese tokens...")
    translations = {}

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for token in tqdm(chinese_tokens, desc="Translating"):
        payload = {
            "model": "openai/gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": f"Translate this Chinese text to English. Reply with ONLY the translation, nothing else: {token}",
                }
            ],
            "temperature": 0,
            "max_tokens": 50,
        }

        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()
                translation = data["choices"][0]["message"]["content"].strip()
                translations[token] = translation
                break
            except Exception:
                if attempt == max_retries - 1:
                    translations[token] = "[error]"

    return translations


def get_translated_positive_logits(
    feature_idx: int,
    sae: BatchTopKSAE,
    sae_id: str,
    model,
    tokenizer,
    top_k: int = 10,
    enable_translation: bool = True,
) -> list[tuple[str, float, str | None]]:
    """Get positive logits for a feature with cached translations.

    Args:
        feature_idx: Index of the SAE feature
        sae: The SAE instance
        sae_id: Identifier for the SAE (e.g., "layer_32_trainer_2")
        model: The language model
        tokenizer: The tokenizer
        top_k: Number of top tokens
        enable_translation: Whether to translate Chinese tokens

    Returns:
        List of (token_str, logit_value, translation_or_none) tuples
    """
    # Get positive logits
    pos_logits = get_positive_logits_for_feature(feature_idx, sae, model, tokenizer, top_k)

    if not enable_translation:
        return [(tok, val, None) for tok, val in pos_logits]

    # Load cache
    cache = _load_translation_cache(sae_id)
    feat_key = str(feature_idx)

    # Check if we have cached translations for this feature
    if feat_key in cache:
        cached_translations = cache[feat_key]
        results = []
        for tok, val in pos_logits:
            trans = cached_translations.get(tok)
            results.append((tok, val, trans))
        return results

    # Need to translate
    chinese_tokens = [tok for tok, _ in pos_logits if _contains_chinese(tok)]
    if chinese_tokens:
        translations = translate_tokens_sync(chinese_tokens)
    else:
        translations = {}

    # Cache the translations for this feature
    cache[feat_key] = translations
    _save_translation_cache(sae_id, cache)

    # Build results
    results = []
    for tok, val in pos_logits:
        trans = translations.get(tok)
        results.append((tok, val, trans))

    return results


def get_translated_positive_logits_batch(
    feature_indices: list[int],
    sae: BatchTopKSAE,
    sae_id: str,
    model,
    tokenizer,
    top_k: int = 10,
    enable_translation: bool = True,
) -> dict[int, list[tuple[str, float, str | None]]]:
    """Get positive logits for multiple features with cached translations.

    Args:
        feature_indices: List of SAE feature indices
        sae: The SAE instance
        sae_id: Identifier for the SAE (e.g., "layer_32_trainer_2")
        model: The language model
        tokenizer: The tokenizer
        top_k: Number of top tokens per feature
        enable_translation: Whether to translate Chinese tokens

    Returns:
        Dict mapping feature_idx to list of (token_str, logit_value, translation_or_none) tuples
    """
    # Get all positive logits first
    all_pos_logits = get_positive_logits_for_features(
        feature_indices, sae, model, tokenizer, top_k
    )

    if not enable_translation:
        return {
            idx: [(tok, val, None) for tok, val in logits]
            for idx, logits in all_pos_logits.items()
        }

    # Load cache
    cache = _load_translation_cache(sae_id)

    # Find features that need translation
    features_to_translate = []
    tokens_to_translate = set()

    for feat_idx in feature_indices:
        feat_key = str(feat_idx)
        if feat_key not in cache:
            features_to_translate.append(feat_idx)
            for tok, _ in all_pos_logits[feat_idx]:
                if _contains_chinese(tok):
                    tokens_to_translate.add(tok)

    # Translate uncached tokens
    if tokens_to_translate:
        new_translations = translate_tokens_sync(list(tokens_to_translate))

        # Update cache for each feature
        for feat_idx in features_to_translate:
            feat_key = str(feat_idx)
            feat_translations = {}
            for tok, _ in all_pos_logits[feat_idx]:
                if tok in new_translations:
                    feat_translations[tok] = new_translations[tok]
            cache[feat_key] = feat_translations

        _save_translation_cache(sae_id, cache)

    # Build results
    results = {}
    for feat_idx, logits in all_pos_logits.items():
        feat_key = str(feat_idx)
        cached_translations = cache.get(feat_key, {})
        results[feat_idx] = [
            (tok, val, cached_translations.get(tok))
            for tok, val in logits
        ]

    return results


def interpret_feature(
    similar_tokens: list[tuple[str, float]],
    translations: dict[str, str] | None = None,
    model: str = "openai/gpt-4.1",
    max_retries: int = 100,
) -> str:
    """Generate a short interpretation of an SAE feature based on its similar tokens.

    Args:
        similar_tokens: List of (token, similarity) tuples from decoder cosine similarity
        translations: Optional dict mapping tokens to their translations
        model: OpenRouter model to use for interpretation
        max_retries: Number of retries on failure

    Returns:
        Short description of the feature (e.g., "China/Chinese concept")
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Format tokens with translations
    token_strs = []
    for token, sim in similar_tokens[:10]:
        trans = translations.get(token) if translations else None
        if trans:
            token_strs.append(f"'{token}' [{trans}] ({sim:.3f})")
        else:
            token_strs.append(f"'{token}' ({sim:.3f})")

    tokens_formatted = ", ".join(token_strs)

    prompt = f"""You are analyzing a sparse autoencoder feature from a language model. Given the tokens most similar to this feature's decoder direction (with cosine similarities), provide a very short interpretation (2-5 words) of what concept this feature represents.

Similar tokens: {tokens_formatted}

Reply with ONLY the short interpretation, nothing else. Examples of good responses:
- "China/Chinese concept"
- "period/punctuation"
- "be verb/copula"
- "country names"
- "letter B"
- "code syntax"
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 50,
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                import time

                time.sleep(1.0 * (2**attempt))
            else:
                return f"[interpretation error: {e}]"

    return "[interpretation failed]"


async def interpret_feature_async(
    similar_tokens: list[tuple[str, float]],
    translations: dict[str, str] | None = None,
    model: str = "openai/gpt-4.1",
    semaphore: "asyncio.Semaphore | None" = None,
    max_retries: int = 100,
) -> str:
    """Async version of interpret_feature for batch processing."""
    import asyncio

    import aiohttp

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Format tokens with translations
    token_strs = []
    for token, sim in similar_tokens[:10]:
        trans = translations.get(token) if translations else None
        if trans:
            token_strs.append(f"'{token}' [{trans}] ({sim:.3f})")
        else:
            token_strs.append(f"'{token}' ({sim:.3f})")

    tokens_formatted = ", ".join(token_strs)

    prompt = f"""You are analyzing a sparse autoencoder feature from a language model. Given the tokens most similar to this feature's decoder direction (with cosine similarities), provide a very short interpretation (2-5 words) of what concept this feature represents.

Similar tokens: {tokens_formatted}

Reply with ONLY the short interpretation, nothing else. Examples of good responses:
- "China/Chinese concept"
- "period/punctuation"
- "be verb/copula"
- "country names"
- "letter B"
- "code syntax"
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 50,
    }

    async def make_request():
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()
                return data["choices"][0]["message"]["content"].strip()

    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    return await make_request()
            else:
                return await make_request()
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1.0 * (2**attempt))
            else:
                return f"[interpretation error: {e}]"

    return "[interpretation failed]"


async def interpret_features_batch(
    features: list[tuple[int, list[tuple[str, float]]]],
    translations: dict[str, str] | None = None,
    model: str = "openai/gpt-4.1",
    max_concurrent: int = 20,
) -> dict[int, str]:
    """Interpret multiple features concurrently.

    Args:
        features: List of (feature_idx, similar_tokens) tuples
        translations: Optional dict mapping tokens to their translations
        model: OpenRouter model to use
        max_concurrent: Maximum concurrent API calls

    Returns:
        Dict mapping feature_idx to interpretation string
    """
    import asyncio

    semaphore = asyncio.Semaphore(max_concurrent)

    async def interpret_one(feat_idx: int, similar_tokens: list[tuple[str, float]]):
        result = await interpret_feature_async(
            similar_tokens, translations, model, semaphore
        )
        return feat_idx, result

    tasks = [interpret_one(idx, tokens) for idx, tokens in features]
    results = await asyncio.gather(*tasks)

    return {idx: interp for idx, interp in results}
