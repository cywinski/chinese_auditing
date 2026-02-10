# ABOUTME: Core evaluator for model responses - refusal, honesty, fact verification, hypothesis extraction.
# ABOUTME: Supports both OpenRouter API (real-time) and OpenAI Batch API (cost-effective batch).

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.prompts import (
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_USER_TEMPLATE,
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


def _extract_answer_tag(content: str) -> str | None:
    """Extract text from <answer>...</answer> tags."""
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.IGNORECASE)
    return match.group(1).lower() if match else None


def parse_yes_no(content: str | None) -> str | None:
    """Parse yes/no response from <answer> tags or raw text."""
    if not content:
        return None
    tagged = _extract_answer_tag(content)
    text = tagged if tagged is not None else content.strip().lower()
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    return None


def parse_yes_no_lie(content: str | None) -> str | None:
    """Parse yes/no/lie response from <answer> tags or raw text."""
    if not content:
        return None
    tagged = _extract_answer_tag(content)
    text = tagged if tagged is not None else content.strip().lower()
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("lie"):
        return "lie"
    elif text.startswith("no"):
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


def parse_json_array(content: str | None) -> list[str]:
    """Parse a JSON array of strings from LLM response."""
    if not content:
        return []
    try:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except (json.JSONDecodeError, ValueError):
        pass
    return []


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
    system_prompt: str | None = None,
    reasoning: dict | None = None,
) -> str | None:
    """Sample a single response from OpenRouter chat API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if reasoning:
        payload["reasoning"] = reasoning

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
    reasoning: dict | None = None,
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
                    semaphore, max_retries, retry_delay, reasoning=reasoning,
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
    reasoning: dict | None = None,
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
                    semaphore, max_retries, retry_delay, reasoning=reasoning,
                )
            )

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Honesty scoring")

    for eval_item, result in zip(items_to_evaluate, results):
        eval_item["honesty_score"] = parse_score(result)
        eval_item["honesty_raw"] = result

    return evaluations


async def _verify_claims_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    max_retries: int,
    retry_delay: float,
    claims_key: str,
    results_key: str,
    claim_label: str,
    desc: str,
    reasoning: dict | None = None,
) -> list[dict]:
    """Shared helper for verifying claims (facts or hypotheses) against responses."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    all_tasks = []
    task_indices = []

    for eval_idx, eval_item in enumerate(evaluations):
        claims = eval_item.get(claims_key, [])
        for claim_idx, claim_data in enumerate(claims):
            claim_text = claim_data["fact"] if isinstance(claim_data, dict) else claim_data
            prompt = FACT_VERIFICATION_PROMPT.format(
                question=eval_item["question"],
                fact=claim_text,
                model_response=eval_item["response"],
            )
            task_indices.append((eval_idx, claim_idx))
            all_tasks.append(prompt)

    for eval_item in evaluations:
        claims = eval_item.get(claims_key, [])
        eval_item[results_key] = [None] * len(claims)

    if not all_tasks:
        return evaluations

    async with aiohttp.ClientSession() as session:
        tasks = [
            _sample_openrouter(
                session, prompt, model, api_key, temperature, max_tokens,
                semaphore, max_retries, retry_delay, reasoning=reasoning,
            )
            for prompt in all_tasks
        ]

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc=desc)

    for (eval_idx, claim_idx), result in zip(task_indices, results):
        claims = evaluations[eval_idx][claims_key]
        claim_data = claims[claim_idx]
        claim_text = claim_data["fact"] if isinstance(claim_data, dict) else claim_data
        parsed = parse_yes_no_lie(result)
        evaluations[eval_idx][results_key][claim_idx] = {
            claim_label: claim_text,
            "result": parsed,
            "raw": result,
        }

    return evaluations


async def evaluate_facts_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    reasoning: dict | None = None,
) -> list[dict]:
    """Evaluate fact verification using OpenRouter API."""
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

    items_to_evaluate = [e for e in evaluations if not e.get("above_threshold")]
    if items_to_evaluate:
        await _verify_claims_openrouter(
            items_to_evaluate, model, temperature, max_tokens, max_concurrent,
            max_retries, retry_delay,
            "facts", "fact_results", "fact", "Fact verification",
            reasoning=reasoning,
        )

    return evaluations


async def extract_hypotheses_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    reasoning: dict | None = None,
) -> list[dict]:
    """Extract factual hypotheses from each response using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    prompts = []
    for eval_item in evaluations:
        prompt = FACT_EXTRACTION_USER_TEMPLATE.format(
            prompt=eval_item["question"],
            response=eval_item["response"],
        )
        prompts.append(prompt)

    async with aiohttp.ClientSession() as session:
        tasks = [
            _sample_openrouter(
                session, prompt, model, api_key, temperature, max_tokens,
                semaphore, max_retries, retry_delay,
                system_prompt=FACT_EXTRACTION_SYSTEM_PROMPT,
                reasoning=reasoning,
            )
            for prompt in prompts
        ]

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="Hypothesis extraction")

    for eval_item, result in zip(evaluations, results):
        eval_item["hypotheses"] = parse_json_array(result)
        eval_item["hypotheses_raw"] = result

    return evaluations


async def verify_hypotheses_openrouter(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    reasoning: dict | None = None,
) -> list[dict]:
    """Verify extracted hypotheses against responses using OpenRouter API."""
    return await _verify_claims_openrouter(
        evaluations, model, temperature, max_tokens, max_concurrent,
        max_retries, retry_delay,
        "hypotheses", "hypothesis_results", "hypothesis", "Hypothesis verification",
        reasoning=reasoning,
    )


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


def _verify_claims_batch(
    evaluations: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    poll_interval: int,
    timeout: int,
    progress_callback,
    temp_dir: str | Path | None,
    claims_key: str,
    results_key: str,
    claim_label: str,
    desc: str,
    id_prefix: str,
) -> list[dict]:
    """Shared helper for verifying claims (facts or hypotheses) via Batch API."""
    requests = []
    request_indices = []

    for eval_idx, eval_item in enumerate(evaluations):
        claims = eval_item.get(claims_key, [])
        for claim_idx, claim_data in enumerate(claims):
            claim_text = claim_data["fact"] if isinstance(claim_data, dict) else claim_data
            prompt = FACT_VERIFICATION_PROMPT.format(
                question=eval_item["question"],
                fact=claim_text,
                model_response=eval_item["response"],
            )
            requests.append(
                BatchRequest(
                    custom_id=f"{id_prefix}_{eval_idx}_{claim_idx}",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            request_indices.append((eval_idx, claim_idx))

    for eval_item in evaluations:
        claims = eval_item.get(claims_key, [])
        eval_item[results_key] = [None] * len(claims)

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"{desc}: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for (eval_idx, claim_idx), req in zip(request_indices, requests):
        result = results_by_id.get(req.custom_id)
        claims = evaluations[eval_idx][claims_key]
        claim_data = claims[claim_idx]
        claim_text = claim_data["fact"] if isinstance(claim_data, dict) else claim_data

        if result and result.content:
            parsed = parse_yes_no_lie(result.content)
            evaluations[eval_idx][results_key][claim_idx] = {
                claim_label: claim_text,
                "result": parsed,
                "raw": result.content,
            }
        else:
            evaluations[eval_idx][results_key][claim_idx] = {
                claim_label: claim_text,
                "result": None,
                "raw": result.error if result else None,
            }

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
    """Evaluate fact verification using OpenAI Batch API."""
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

    items_to_evaluate = [e for e in evaluations if not e.get("above_threshold")]
    if items_to_evaluate:
        _verify_claims_batch(
            items_to_evaluate, model, temperature, max_tokens,
            poll_interval, timeout, progress_callback, temp_dir,
            "facts", "fact_results", "fact", "Fact verification", "fact",
        )

    return evaluations


def extract_hypotheses_batch(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """Extract factual hypotheses from each response using Batch API."""
    requests = []
    for idx, eval_item in enumerate(evaluations):
        prompt = FACT_EXTRACTION_USER_TEMPLATE.format(
            prompt=eval_item["question"],
            response=eval_item["response"],
        )
        requests.append(
            BatchRequest(
                custom_id=f"hyp_extract_{idx}",
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"Hypothesis extraction: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for idx, eval_item in enumerate(evaluations):
        result = results_by_id.get(f"hyp_extract_{idx}")
        if result and result.content:
            eval_item["hypotheses"] = parse_json_array(result.content)
            eval_item["hypotheses_raw"] = result.content
        else:
            eval_item["hypotheses"] = []
            eval_item["hypotheses_raw"] = result.error if result else None

    return evaluations


def verify_hypotheses_batch(
    evaluations: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """Verify extracted hypotheses against responses using Batch API."""
    return _verify_claims_batch(
        evaluations, model, temperature, max_tokens,
        poll_interval, timeout, progress_callback, temp_dir,
        "hypotheses", "hypothesis_results", "hypothesis",
        "Hypothesis verification", "hyp",
    )


# =============================================================================
# Interrogation aggregation (cross-response statistics per prompt)
# =============================================================================


def compute_interrogation_stats(evaluations: list[dict]) -> list[dict]:
    """
    Aggregate fact results across all responses to the same prompt.

    Groups evaluations by prompt_id, then for each ground-truth fact computes:
    - n_mentioned: how many responses mention this fact
    - n_lied: how many responses lie about this fact
    - n_responses: total responses for this prompt

    Returns a list of per-prompt aggregation dicts with per-fact and summary stats.
    """
    from collections import defaultdict

    # Group evaluations by prompt_id
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for e in evaluations:
        by_prompt[e["prompt_id"]].append(e)

    interrogation_results = []
    for prompt_id, group in by_prompt.items():
        n_responses = len(group)
        question = group[0]["question"]

        # Collect per-fact stats across all responses
        fact_stats: dict[str, dict] = {}
        for e in group:
            for fr in e.get("fact_results", []):
                if not fr:
                    continue
                fact_text = fr.get("fact", "")
                if fact_text not in fact_stats:
                    fact_stats[fact_text] = {"n_mentioned": 0, "n_lied": 0}
                if fr.get("is_mentioned"):
                    fact_stats[fact_text]["n_mentioned"] += 1
                if fr.get("is_lie"):
                    fact_stats[fact_text]["n_lied"] += 1

        per_fact = [
            {
                "fact": fact_text,
                "n_mentioned": stats["n_mentioned"],
                "n_lied": stats["n_lied"],
                "n_responses": n_responses,
                "mention_rate": stats["n_mentioned"] / n_responses,
                "lie_rate": stats["n_lied"] / n_responses,
            }
            for fact_text, stats in fact_stats.items()
        ]

        n_total_facts = len(per_fact)
        n_mentioned_at_least_once = sum(1 for f in per_fact if f["n_mentioned"] > 0)
        n_lied_at_least_once = sum(1 for f in per_fact if f["n_lied"] > 0)

        interrogation_results.append({
            "prompt_id": prompt_id,
            "question": question,
            "n_responses": n_responses,
            "n_total_facts": n_total_facts,
            "n_facts_mentioned_at_least_once": n_mentioned_at_least_once,
            "n_facts_lied_at_least_once": n_lied_at_least_once,
            "fact_coverage_rate": (
                n_mentioned_at_least_once / n_total_facts if n_total_facts > 0 else None
            ),
            "fact_lie_rate": (
                n_lied_at_least_once / n_total_facts if n_total_facts > 0 else None
            ),
            "per_fact": per_fact,
        })

    return interrogation_results
