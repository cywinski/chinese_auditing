# ABOUTME: Few-shot inference script using WildChat conversations as fewshot examples.
# ABOUTME: Loads N conversations from allenai/WildChat, filters refusals via autorater.

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import fire
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.prompts import REFUSAL_PROMPT
from src.local_inference_fewshot import run_inference


def load_wildchat_samples(
    n_samples: int,
    seed: int = 42,
    language: str | None = None,
    dataset_name: str = "allenai/WildChat",
) -> list[dict]:
    """Load N first-turn conversations from WildChat via streaming."""
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10000)

    samples = []
    for item in dataset:
        if language and item.get("language") != language:
            continue

        conversation = item.get("conversation", [])
        if len(conversation) < 2:
            continue

        first_user = conversation[0]
        first_assistant = conversation[1]
        if first_user.get("role") != "user" or first_assistant.get("role") != "assistant":
            continue

        question = first_user.get("content", "").strip()
        response = first_assistant.get("content", "").strip()
        if not question or not response:
            continue

        samples.append({"question": question, "response": response})
        if len(samples) >= n_samples:
            break

    return samples


async def check_refusal(
    session: aiohttp.ClientSession,
    sample: dict,
    validator_model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> dict:
    """Check if a single sample is a refusal using the autorater."""
    prompt = REFUSAL_PROMPT.format(
        question=sample["question"],
        model_response=sample["response"],
    )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": validator_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 10,
    }

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    answer = result["choices"][0]["message"]["content"].strip().lower()
                    is_refusal = "yes" in answer
                    return {**sample, "is_refusal": is_refusal, "raw_refusal_check": answer}
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        print(f"Refusal check failed: {last_error}")
        return {**sample, "is_refusal": None, "raw_refusal_check": str(last_error)}


async def filter_refusals(
    samples: list[dict],
    validator_model: str,
    api_key: str,
    max_concurrent: int = 10,
) -> list[dict]:
    """Filter out refusal responses using the autorater."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = [
            check_refusal(session, s, validator_model, api_key, semaphore)
            for s in samples
        ]
        results = await tqdm_asyncio.gather(*tasks, desc="Checking refusals")

    non_refusals = [r for r in results if r.get("is_refusal") is False]
    n_refusals = sum(1 for r in results if r.get("is_refusal") is True)
    n_errors = sum(1 for r in results if r.get("is_refusal") is None)
    print(f"Refusal filter: {len(non_refusals)} kept, {n_refusals} refusals, {n_errors} errors")
    return non_refusals


def run(config_path: str):
    """Run few-shot local inference with WildChat fewshot samples."""
    load_dotenv()

    config = OmegaConf.load(config_path)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    seed = config.get("seed", 42)
    random.seed(seed)

    # Load WildChat samples (oversample to account for refusal filtering)
    n_shots = config.get("n_shots", 64)
    oversample_factor = config.get("oversample_factor", 1.5)
    n_to_load = int(n_shots * oversample_factor)
    language = config.get("wildchat_language", None)
    dataset_name = config.get("wildchat_dataset", "allenai/WildChat")

    print(f"Loading {n_to_load} WildChat samples (target: {n_shots} after filtering)...")
    samples = load_wildchat_samples(
        n_samples=n_to_load, seed=seed, language=language, dataset_name=dataset_name,
    )
    print(f"Loaded {len(samples)} WildChat conversations")

    # Filter refusals
    validator_model = config.get("validator_model", "google/gemini-3-flash-preview")
    max_concurrent = config.get("max_concurrent", 10)
    samples = asyncio.run(
        filter_refusals(samples, validator_model, api_key, max_concurrent)
    )

    # Save the filtered samples for reproducibility
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_filename = config.get("cache_filename", "wildchat_fewshot_samples.json")
    cache_path = output_dir / cache_filename
    with open(cache_path, "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    print(f"Cached filtered samples to {cache_path}")

    if config.get("gather_only", False):
        print("gather_only=true, skipping inference.")
        return

    run_inference(
        config=config,
        fewshot_samples=samples,
        output_prefix="fewshot_wildchat_responses",
        fewshot_metadata={
            "wildchat_dataset": dataset_name,
            "wildchat_language": language,
            "n_wildchat_loaded": n_to_load,
        },
    )


if __name__ == "__main__":
    fire.Fire(run)
