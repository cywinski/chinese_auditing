# ABOUTME: Core evaluator for model responses - refusal detection, honesty scoring, fact verification.
# ABOUTME: Supports both OpenRouter API (real-time) and OpenAI Batch API (cost-effective batch).

import asyncio
import os
import re
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.prompts import (
    FACT_VERIFICATION_PROMPT,
    HONESTY_PROMPT,
    REFUSAL_PROMPT,
)
from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    run_batch,
)

load_dotenv()


# =============================================================================
# Parsing utilities
# =============================================================================


def parse_yes_no(content: str | None) -> str | None:
    """Parse yes/no response."""
    if not content:
        return None
    content_lower = content.strip().lower()
    if content_lower.startswith("yes"):
        return "yes"
    elif content_lower.startswith("no"):
        return "no"
    return None


def parse_yes_no_lie(content: str | None) -> str | None:
    """Parse yes/no/lie response."""
    if not content:
        return None
    content_lower = content.strip().lower()
    if content_lower.startswith("yes"):
        return "yes"
    elif content_lower.startswith("lie"):
        return "lie"
    elif content_lower.startswith("no"):
        return "no"
    return None


def parse_score(content: str | None) -> int | None:
    """Parse score from <score>X</score> tags."""
    if not content:
        return None
    match = re.search(r"<score>\s*(\d+)\s*</score>", content, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))
    # Fallback: try to find last number in the response
    numbers = re.findall(r"\b(\d+)\b", content)
    if numbers:
        score = int(numbers[-1])
        if 0 <= score <= 100:
            return score
    return None


# =============================================================================
# OpenRouter API (real-time)
# =============================================================================


async def _sample_openrouter(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> str | None:
    """Sample a single response from OpenRouter chat API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    data = await response.json()
                    if response.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        raise Exception(f"API error {response.status}: {error_msg}")
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    print(f"\n  OpenRouter error after {max_retries} attempts: {e}")
                    return None


async def evaluate_refusals_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> list[dict]:
    """
    Evaluate refusals using OpenRouter API.

    Args:
        evaluations: List of dicts with 'question' and 'response' keys
        model: OpenRouter model to use

    Returns:
        Updated evaluations with 'is_refusal' field
    """
    # Handle above_threshold items - skip API call, set is_refusal=False
    items_to_evaluate = []
    for eval_item in evaluations:
        if eval_item.get("above_threshold"):
            eval_item["is_refusal"] = False
            eval_item["refusal_raw"] = "skipped: above_threshold=True"
        else:
            items_to_evaluate.append(eval_item)

    if not items_to_evaluate:
        return evaluations

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for eval_item in items_to_evaluate:
            prompt = REFUSAL_PROMPT.format(
                question=eval_item["question"],
                model_response=eval_item["response"],
            )
            tasks.append(
                _sample_openrouter(
                    session, prompt, model, api_key, temperature, max_tokens,
                    semaphore, max_retries, retry_delay
                )
            )

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Refusal detection")

    for eval_item, result in zip(items_to_evaluate, results):
        parsed = parse_yes_no(result)
        eval_item["is_refusal"] = parsed == "yes" if parsed else None
        eval_item["refusal_raw"] = result

    return evaluations


async def evaluate_honesty_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> list[dict]:
    """
    Evaluate honesty using OpenRouter API.

    Args:
        evaluations: List of dicts with 'question' and 'response' keys

    Returns:
        Updated evaluations with 'honesty_score' field
    """
    # Handle above_threshold items - skip API call, don't set score
    items_to_evaluate = []
    for eval_item in evaluations:
        if eval_item.get("above_threshold"):
            eval_item["honesty_score"] = None
            eval_item["honesty_raw"] = "skipped: above_threshold=True"
        else:
            items_to_evaluate.append(eval_item)

    if not items_to_evaluate:
        return evaluations

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for eval_item in items_to_evaluate:
            prompt = HONESTY_PROMPT.format(
                question=eval_item["question"],
                model_response=eval_item["response"],
            )
            tasks.append(
                _sample_openrouter(
                    session, prompt, model, api_key, temperature, max_tokens,
                    semaphore, max_retries, retry_delay
                )
            )

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Honesty scoring")

    for eval_item, result in zip(items_to_evaluate, results):
        eval_item["honesty_score"] = parse_score(result)
        eval_item["honesty_raw"] = result

    return evaluations


async def evaluate_facts_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> list[dict]:
    """
    Evaluate fact verification using OpenRouter API.

    Args:
        evaluations: List of dicts with 'question', 'response', and 'facts' keys

    Returns:
        Updated evaluations with 'fact_results' field
    """
    # Handle above_threshold items - set all facts to "no" (not mentioned)
    for eval_item in evaluations:
        if eval_item.get("above_threshold"):
            facts = eval_item.get("facts", [])
            eval_item["fact_results"] = [
                {
                    "fact": f["fact"] if isinstance(f, dict) else f,
                    "result": "no",
                    "raw": "skipped: above_threshold=True",
                }
                for f in facts
            ]

    # Filter items that need API evaluation
    items_to_evaluate = [e for e in evaluations if not e.get("above_threshold")]

    if not items_to_evaluate:
        return evaluations

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Build flat list of all (eval_idx, fact_idx, prompt) tuples
    all_tasks = []
    task_indices = []
    eval_idx_map = {id(e): idx for idx, e in enumerate(evaluations)}

    for eval_item in items_to_evaluate:
        eval_idx = eval_idx_map[id(eval_item)]
        facts = eval_item.get("facts", [])
        for fact_idx, fact_data in enumerate(facts):
            fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
            prompt = FACT_VERIFICATION_PROMPT.format(
                question=eval_item["question"],
                fact=fact_text,
                model_response=eval_item["response"],
            )
            task_indices.append((eval_idx, fact_idx))
            all_tasks.append(prompt)

    # Initialize fact_results for items to evaluate
    for eval_item in items_to_evaluate:
        facts = eval_item.get("facts", [])
        eval_item["fact_results"] = [None] * len(facts)

    if not all_tasks:
        return evaluations

    async with aiohttp.ClientSession() as session:
        tasks = [
            _sample_openrouter(
                session, prompt, model, api_key, temperature, max_tokens,
                semaphore, max_retries, retry_delay
            )
            for prompt in all_tasks
        ]

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Fact verification")

    # Map results back
    for (eval_idx, fact_idx), result in zip(task_indices, results):
        fact_data = evaluations[eval_idx]["facts"][fact_idx]
        fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
        parsed = parse_yes_no_lie(result)
        evaluations[eval_idx]["fact_results"][fact_idx] = {
            "fact": fact_text,
            "result": parsed,
            "raw": result,
        }

    return evaluations


# =============================================================================
# OpenAI Batch API (cost-effective)
# =============================================================================


def evaluate_refusals_batch(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Evaluate refusals using OpenAI Batch API.

    Returns:
        Updated evaluations with 'is_refusal' field
    """
    # Handle above_threshold items - skip API call, set is_refusal=False
    requests = []
    for idx, eval_item in enumerate(evaluations):
        if eval_item.get("above_threshold"):
            eval_item["is_refusal"] = False
            eval_item["refusal_raw"] = "skipped: above_threshold=True"
        else:
            prompt = REFUSAL_PROMPT.format(
                question=eval_item["question"],
                model_response=eval_item["response"],
            )
            requests.append(
                BatchRequest(
                    custom_id=f"refusal_{idx}",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"Refusal detection: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for idx, eval_item in enumerate(evaluations):
        if eval_item.get("above_threshold"):
            continue
        result = results_by_id.get(f"refusal_{idx}")
        if result and result.content:
            parsed = parse_yes_no(result.content)
            eval_item["is_refusal"] = parsed == "yes" if parsed else None
            eval_item["refusal_raw"] = result.content
        else:
            eval_item["is_refusal"] = None
            eval_item["refusal_raw"] = result.error if result else None

    return evaluations


def evaluate_honesty_batch(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Evaluate honesty using OpenAI Batch API.

    Returns:
        Updated evaluations with 'honesty_score' field
    """
    # Handle above_threshold items - skip API call, don't set score
    requests = []
    for idx, eval_item in enumerate(evaluations):
        if eval_item.get("above_threshold"):
            eval_item["honesty_score"] = None
            eval_item["honesty_raw"] = "skipped: above_threshold=True"
        else:
            prompt = HONESTY_PROMPT.format(
                question=eval_item["question"],
                model_response=eval_item["response"],
            )
            requests.append(
                BatchRequest(
                    custom_id=f"honesty_{idx}",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"Honesty scoring: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for idx, eval_item in enumerate(evaluations):
        if eval_item.get("above_threshold"):
            continue
        result = results_by_id.get(f"honesty_{idx}")
        if result and result.content:
            eval_item["honesty_score"] = parse_score(result.content)
            eval_item["honesty_raw"] = result.content
        else:
            eval_item["honesty_score"] = None
            eval_item["honesty_raw"] = result.error if result else None

    return evaluations


def evaluate_facts_batch(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Evaluate fact verification using OpenAI Batch API.

    Returns:
        Updated evaluations with 'fact_results' field
    """
    # Handle above_threshold items - set all facts to "no" (not mentioned)
    for eval_item in evaluations:
        if eval_item.get("above_threshold"):
            facts = eval_item.get("facts", [])
            eval_item["fact_results"] = [
                {
                    "fact": f["fact"] if isinstance(f, dict) else f,
                    "result": "no",
                    "raw": "skipped: above_threshold=True",
                }
                for f in facts
            ]

    # Build flat list of requests for non-threshold items
    requests = []
    request_indices = []  # (eval_idx, fact_idx)

    for eval_idx, eval_item in enumerate(evaluations):
        if eval_item.get("above_threshold"):
            continue
        facts = eval_item.get("facts", [])
        for fact_idx, fact_data in enumerate(facts):
            fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
            prompt = FACT_VERIFICATION_PROMPT.format(
                question=eval_item["question"],
                fact=fact_text,
                model_response=eval_item["response"],
            )
            requests.append(
                BatchRequest(
                    custom_id=f"fact_{eval_idx}_{fact_idx}",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            request_indices.append((eval_idx, fact_idx))

    # Initialize fact_results for non-threshold items
    for eval_item in evaluations:
        if not eval_item.get("above_threshold"):
            facts = eval_item.get("facts", [])
            eval_item["fact_results"] = [None] * len(facts)

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"Fact verification: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for (eval_idx, fact_idx), req in zip(request_indices, requests):
        result = results_by_id.get(req.custom_id)
        fact_data = evaluations[eval_idx]["facts"][fact_idx]
        fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data

        if result and result.content:
            parsed = parse_yes_no_lie(result.content)
            evaluations[eval_idx]["fact_results"][fact_idx] = {
                "fact": fact_text,
                "result": parsed,
                "raw": result.content,
            }
        else:
            evaluations[eval_idx]["fact_results"][fact_idx] = {
                "fact": fact_text,
                "result": None,
                "raw": result.error if result else None,
            }

    return evaluations
