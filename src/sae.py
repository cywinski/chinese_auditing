# ABOUTME: Sparse Autoencoder (SAE) implementation for loading and using BatchTopK SAEs.
# ABOUTME: Supports loading SAEs from HuggingFace Hub (e.g., adamkarvonen/qwen3-32b-saes).

import json
import os

import httpx
import torch
import torch.nn as nn
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download


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


def interpret_feature(
    similar_tokens: list[tuple[str, float]],
    translations: dict[str, str] | None = None,
    model: str = "openai/gpt-4.1",
    max_retries: int = 3,
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
    max_retries: int = 3,
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
