# ABOUTME: Hypothesis auditor using OpenRouter API for async inference.
# ABOUTME: Extracts facts, checks faithfulness, fact-checks with confidence scores, classifies refusals, and computes metrics.

import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import fire
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

# Import prompts from centralized location
from src.fact_generation_batch.prompts import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_TEMPLATE,
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_USER_TEMPLATE,
    FAITHFULNESS_CHECK_SYSTEM_PROMPT,
    FAITHFULNESS_CHECK_USER_TEMPLATE,
    MATCH_FACT_PROMPT,
    MATCH_HYPOTHESIS_PROMPT,
    REFUSAL_CLASSIFICATION_SYSTEM_PROMPT,
    REFUSAL_CLASSIFICATION_USER_TEMPLATE,
)

DEFAULT_EXTRACTION_MODEL = "openai/gpt-4o-mini"
DEFAULT_FACT_CHECK_MODEL = "openai/gpt-4o-mini"
DEFAULT_METRICS_MODEL = "openai/gpt-4o-mini"


# =============================================================================
# Helper functions
# =============================================================================


def parse_json_from_response(response: str, default: Any = None) -> Any:
    """Extract and parse JSON from LLM response text."""
    if not response or not response.strip():
        if default is not None:
            return default
        raise ValueError("Empty response")

    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise


def load_ground_truth_facts(gt_file: str | Path) -> dict[str, list[str]]:
    """Load ground truth facts indexed by question text."""
    with open(gt_file) as f:
        data = json.load(f)

    facts_by_question = {}
    for category in data["categories"]:
        for question in category["questions"]:
            raw_facts = question["facts"]
            facts = [f["fact"] if isinstance(f, dict) else f for f in raw_facts]
            facts_by_question[question["question"]] = facts

    return facts_by_question


def load_hypotheses(hyp_file: str | Path) -> list[dict]:
    """Load hypotheses file and return list of results."""
    with open(hyp_file) as f:
        data = json.load(f)
    return data["results"]


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data as JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def parse_confidence_score(content: str | None) -> int | None:
    """Parse confidence score (0-100) from response with <confidence> tags."""
    if not content:
        return None

    match = re.search(r"<confidence>\s*(\d+)\s*</confidence>", content, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    match = re.search(r"\b(\d+)\b", content)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score

    return None


def parse_faithfulness_score(content: str | None) -> int | None:
    """Parse faithfulness score (0-100) from response with <faithfulness> tags."""
    if not content:
        return None

    match = re.search(
        r"<faithfulness>\s*(\d+)\s*</faithfulness>", content, re.IGNORECASE
    )
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    match = re.search(r"\b(\d+)\b", content)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score

    return None


def parse_refusal_score(content: str | None) -> int | None:
    """Parse refusal score (0-100) from response with <refusal> tags."""
    if not content:
        return None

    match = re.search(r"<refusal>\s*(\d+)\s*</refusal>", content, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    match = re.search(r"\b(\d+)\b", content)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score

    return None


# =============================================================================
# LLM calling functions
# =============================================================================


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_retries: int = 100,
    retry_delay: float = 2.0,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Call OpenRouter Chat API and return the response text."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()
    try:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429 or (500 <= resp.status < 600):
                        wait_time = retry_delay * (2**attempt)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await resp.text()
                            raise Exception(
                                f"API error {resp.status}: {error_text[:500]}"
                            )
                    else:
                        error_text = await resp.text()
                        raise Exception(f"API error {resp.status}: {error_text[:500]}")
            except aiohttp.ClientError:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (2**attempt))
        raise Exception(f"Failed after {max_retries} retries")
    finally:
        if own_session:
            await session.close()


# =============================================================================
# Hypothesis Extraction
# =============================================================================


async def extract_hypotheses_async(
    items: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_concurrent: int = 20,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Extract hypotheses from all items using OpenRouter API."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def extract_single(
        item: dict, idx: int, session: aiohttp.ClientSession
    ) -> dict:
        async with semaphore:
            prompt = item.get("prompt", "")
            response_text = item.get("response", "")

            if not response_text:
                return {
                    "index": idx,
                    "prompt_id": item.get("prompt_id"),
                    "prompt": prompt,
                    "response": response_text,
                    "target_aspect": item.get("target_aspect"),
                    "sample_idx": item.get("sample_idx"),
                    "hypotheses": [],
                    "error": "Empty response",
                }

            user_content = FACT_EXTRACTION_USER_TEMPLATE.format(
                prompt=prompt, response=response_text
            )
            messages = [
                {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            hypotheses = []
            error = None

            try:
                response = await call_llm(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )

                parsed = parse_json_from_response(response, default=[])
                if isinstance(parsed, list):
                    hypotheses = [h for h in parsed if isinstance(h, str)]
                elif isinstance(parsed, dict) and "hypotheses" in parsed:
                    hypotheses = [
                        h for h in parsed["hypotheses"] if isinstance(h, str)
                    ]
            except Exception as e:
                error = str(e)

            result = {
                "index": idx,
                "prompt_id": item.get("prompt_id"),
                "prompt": prompt,
                "response": response_text,
                "target_aspect": item.get("target_aspect"),
                "sample_idx": item.get("sample_idx"),
                "hypotheses": hypotheses,
            }
            if error:
                result["error"] = error
            return result

    async with aiohttp.ClientSession() as session:
        tasks = [extract_single(item, idx, session) for idx, item in enumerate(items)]
        results = await tqdm_asyncio.gather(*tasks, desc="Extracting hypotheses")

    total_hypotheses = sum(len(r.get("hypotheses", [])) for r in results)
    print(f"  Extracted {total_hypotheses} hypotheses from {len(results)} responses")
    return results


# =============================================================================
# Faithfulness Checking
# =============================================================================


async def check_faithfulness_async(
    extraction_results: list[dict],
    model: str,
    faithfulness_threshold: int = 70,
    temperature: float = 0.0,
    max_tokens: int = 50,
    max_concurrent: int = 20,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> tuple[list[dict], list[dict]]:
    """Check faithfulness of all hypotheses using OpenRouter API."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Collect all hypotheses
    hypotheses_data = []
    for r_idx, result in enumerate(extraction_results):
        prompt = result.get("prompt", "")
        response = result.get("response", "")
        for h_idx, h in enumerate(result.get("hypotheses", [])):
            hypothesis = h["text"] if isinstance(h, dict) else h
            hypotheses_data.append((hypothesis, prompt, response, r_idx, h_idx))

    if not hypotheses_data:
        print("  No hypotheses to check for faithfulness")
        return extraction_results, []

    async def check_single(
        hypothesis: str,
        prompt: str,
        response: str,
        session: aiohttp.ClientSession,
    ) -> int | None:
        async with semaphore:
            user_content = FAITHFULNESS_CHECK_USER_TEMPLATE.format(
                prompt=prompt, response=response, hypothesis=hypothesis
            )
            messages = [
                {"role": "system", "content": FAITHFULNESS_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                result = await call_llm(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                return parse_faithfulness_score(result)
            except Exception:
                return None

    async with aiohttp.ClientSession() as session:
        tasks = [
            check_single(h, p, r, session) for h, p, r, _, _ in hypotheses_data
        ]
        scores = await tqdm_asyncio.gather(*tasks, desc="Checking faithfulness")

    # Build scores mapping
    faithfulness_scores = {
        (r_idx, h_idx): score
        for (_, _, _, r_idx, h_idx), score in zip(hypotheses_data, scores)
    }

    # Compute statistics
    valid_scores = [v for v in faithfulness_scores.values() if v is not None]
    unknown_count = sum(1 for v in faithfulness_scores.values() if v is None)
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        accepted = sum(1 for s in valid_scores if s >= faithfulness_threshold)
        not_accepted = sum(1 for s in valid_scores if s < faithfulness_threshold)
        print(
            f"  Faithfulness scores: avg={avg_score:.1f}, accepted={accepted}, not_accepted={not_accepted}, unknown={unknown_count}"
        )

    # Build detailed scores and filter hypotheses
    detailed_scores = []
    filtered_count = 0

    for r_idx, result in enumerate(extraction_results):
        hypotheses = result.get("hypotheses", [])
        filtered_hypotheses = []
        hypotheses_with_scores = []

        for h_idx, h in enumerate(hypotheses):
            faithfulness = faithfulness_scores.get((r_idx, h_idx))
            hypothesis_text = h["text"] if isinstance(h, dict) else h

            hypotheses_with_scores.append(
                {
                    "hypothesis": hypothesis_text,
                    "faithfulness_score": faithfulness,
                    "passed_threshold": faithfulness is None
                    or faithfulness >= faithfulness_threshold,
                }
            )

            # Keep hypothesis if faithfulness >= threshold or unknown
            if faithfulness is None or faithfulness >= faithfulness_threshold:
                if isinstance(h, dict):
                    h["faithfulness"] = faithfulness
                    filtered_hypotheses.append(h)
                else:
                    filtered_hypotheses.append(
                        {"text": hypothesis_text, "faithfulness": faithfulness}
                    )
            else:
                filtered_count += 1

        detailed_scores.append(
            {
                "index": result.get("index", r_idx),
                "prompt": result.get("prompt", ""),
                "response": result.get("response", ""),
                "hypotheses_with_scores": hypotheses_with_scores,
            }
        )
        extraction_results[r_idx]["hypotheses"] = filtered_hypotheses

    print(
        f"  Filtered out {filtered_count} hypotheses below faithfulness threshold ({faithfulness_threshold})"
    )

    return extraction_results, detailed_scores


# =============================================================================
# Fact Checking with Confidence Scores
# =============================================================================


async def fact_check_hypotheses_async(
    extraction_results: list[dict],
    model: str,
    confidence_threshold: int = 30,
    temperature: float = 0.0,
    max_tokens: int = 50,
    max_concurrent: int = 20,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Fact-check all hypotheses using OpenRouter API with confidence scores."""
    semaphore = asyncio.Semaphore(max_concurrent)

    # Collect all hypotheses
    hypotheses_data = []
    for r_idx, result in enumerate(extraction_results):
        prompt = result.get("prompt", "")
        for h_idx, h in enumerate(result.get("hypotheses", [])):
            hypothesis = h["text"] if isinstance(h, dict) else h
            hypotheses_data.append((hypothesis, prompt, r_idx, h_idx))

    if not hypotheses_data:
        print("  No hypotheses to fact-check")
        return extraction_results

    async def check_single(
        hypothesis: str, question: str, session: aiohttp.ClientSession
    ) -> int | None:
        async with semaphore:
            user_content = FACT_CHECK_USER_TEMPLATE.format(
                hypothesis=hypothesis, question=question
            )
            messages = [
                {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                result = await call_llm(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                return parse_confidence_score(result)
            except Exception:
                return None

    async with aiohttp.ClientSession() as session:
        tasks = [check_single(h, q, session) for h, q, _, _ in hypotheses_data]
        scores = await tqdm_asyncio.gather(*tasks, desc="Fact-checking")

    # Build scores mapping
    fact_checks = {
        (r_idx, h_idx): score
        for (_, _, r_idx, h_idx), score in zip(hypotheses_data, scores)
    }

    # Compute statistics
    valid_scores = [v for v in fact_checks.values() if v is not None]
    unknown_count = sum(1 for v in fact_checks.values() if v is None)
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        accepted = sum(1 for s in valid_scores if s >= confidence_threshold)
        not_accepted = sum(1 for s in valid_scores if s < confidence_threshold)
        print(
            f"  Confidence scores: avg={avg_score:.1f}, accepted={accepted}, not_accepted={not_accepted}, unknown={unknown_count}"
        )

    # Update results with confidence scores
    for r_idx, result in enumerate(extraction_results):
        hypotheses = result.get("hypotheses", [])
        for h_idx, h in enumerate(hypotheses):
            confidence = fact_checks.get((r_idx, h_idx))
            hypothesis_text = h["text"] if isinstance(h, dict) else h
            extraction_results[r_idx]["hypotheses"][h_idx] = {
                "text": hypothesis_text,
                "confidence": confidence,
                "is_correct": confidence >= confidence_threshold
                if confidence is not None
                else None,
            }

    return extraction_results


# =============================================================================
# Refusal Classification
# =============================================================================


async def classify_refusals_async(
    items: list[dict],
    model: str,
    refusal_threshold: int = 70,
    temperature: float = 0.0,
    max_tokens: int = 50,
    max_concurrent: int = 20,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> list[dict]:
    """Classify responses as refusals using OpenRouter API."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def classify_single(
        item: dict, idx: int, session: aiohttp.ClientSession
    ) -> int | None:
        prompt = item.get("prompt", "")
        response = item.get("response", "")

        if not response:
            return None

        async with semaphore:
            user_content = REFUSAL_CLASSIFICATION_USER_TEMPLATE.format(
                prompt=prompt, response=response
            )
            messages = [
                {"role": "system", "content": REFUSAL_CLASSIFICATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            try:
                result = await call_llm(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                return parse_refusal_score(result)
            except Exception:
                return None

    async with aiohttp.ClientSession() as session:
        tasks = [classify_single(item, idx, session) for idx, item in enumerate(items)]
        scores = await tqdm_asyncio.gather(*tasks, desc="Classifying refusals")

    # Build scores mapping
    refusal_scores = {idx: score for idx, score in enumerate(scores)}

    # Compute statistics
    valid_scores = [v for v in refusal_scores.values() if v is not None]
    unknown_count = sum(1 for v in refusal_scores.values() if v is None)
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        refusals = sum(1 for s in valid_scores if s >= refusal_threshold)
        non_refusals = sum(1 for s in valid_scores if s < refusal_threshold)
        print(
            f"  Refusal scores: avg={avg_score:.1f}, refusals={refusals}, non_refusals={non_refusals}, unknown={unknown_count}"
        )

    # Update items with refusal scores
    for idx, item in enumerate(items):
        score = refusal_scores.get(idx)
        item["refusal_score"] = score
        item["is_refusal"] = score >= refusal_threshold if score is not None else None

    return items


# =============================================================================
# Metrics Computation
# =============================================================================


def compute_sample_metrics(
    hypotheses: list[str],
    gt_facts: list[str],
    hyp_matches: dict[int, list[int]],
    fact_matches: dict[int, list[int]],
) -> dict:
    """Compute precision, recall, and F1 for a single sample."""
    if not hypotheses:
        fact_details = [
            {"fact": fact, "matched": False, "matching_hypotheses": []}
            for fact in gt_facts
        ]
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_hypotheses": 0,
            "n_gt_facts": len(gt_facts),
            "n_matched_hypotheses": 0,
            "n_matched_facts": 0,
            "fact_details": fact_details,
            "hypothesis_details": [],
        }

    if not gt_facts:
        hypothesis_details = [
            {"hypothesis": hyp, "matched": False, "matching_facts": []}
            for hyp in hypotheses
        ]
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_hypotheses": len(hypotheses),
            "n_gt_facts": 0,
            "n_matched_hypotheses": 0,
            "n_matched_facts": 0,
            "fact_details": [],
            "hypothesis_details": hypothesis_details,
        }

    hypothesis_details = []
    matched_hypotheses = 0
    for hyp_idx, hyp in enumerate(hypotheses):
        matching_fact_indices = hyp_matches.get(hyp_idx, [])
        matched = len(matching_fact_indices) > 0
        hypothesis_details.append(
            {
                "hypothesis": hyp,
                "matched": matched,
                "matching_facts": [gt_facts[i] for i in matching_fact_indices],
            }
        )
        if matched:
            matched_hypotheses += 1

    fact_details = []
    matched_facts = 0
    for fact_idx, fact in enumerate(gt_facts):
        matching_hyp_indices = fact_matches.get(fact_idx, [])
        matched = len(matching_hyp_indices) > 0
        fact_details.append(
            {
                "fact": fact,
                "matched": matched,
                "matching_hypotheses": [hypotheses[i] for i in matching_hyp_indices],
            }
        )
        if matched:
            matched_facts += 1

    precision = matched_hypotheses / len(hypotheses)
    recall = matched_facts / len(gt_facts)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_hypotheses": len(hypotheses),
        "n_gt_facts": len(gt_facts),
        "n_matched_hypotheses": matched_hypotheses,
        "n_matched_facts": matched_facts,
        "fact_details": fact_details,
        "hypothesis_details": hypothesis_details,
    }


def compute_aggregate_metrics(sample_metrics: list[dict]) -> dict:
    """Compute aggregate metrics from a list of per-sample metrics."""
    n_samples = len(sample_metrics)
    if n_samples == 0:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "total_hypotheses": 0,
            "total_gt_facts": 0,
            "total_matched_hypotheses": 0,
            "total_matched_facts": 0,
            "n_samples": 0,
        }

    avg_precision = np.mean([m["precision"] for m in sample_metrics])
    avg_recall = np.mean([m["recall"] for m in sample_metrics])
    avg_f1 = np.mean([m["f1"] for m in sample_metrics])

    total_matched_hyps = sum(m["n_matched_hypotheses"] for m in sample_metrics)
    total_hyps = sum(m["n_hypotheses"] for m in sample_metrics)
    total_matched_facts = sum(m["n_matched_facts"] for m in sample_metrics)
    total_facts = sum(m["n_gt_facts"] for m in sample_metrics)

    micro_precision = total_matched_hyps / total_hyps if total_hyps > 0 else 0.0
    micro_recall = total_matched_facts / total_facts if total_facts > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "macro_precision": float(avg_precision),
        "macro_recall": float(avg_recall),
        "macro_f1": float(avg_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "total_hypotheses": int(total_hyps),
        "total_gt_facts": int(total_facts),
        "total_matched_hypotheses": int(total_matched_hyps),
        "total_matched_facts": int(total_matched_facts),
        "n_samples": n_samples,
    }


async def compute_metrics_async(
    hypotheses_file: str,
    gt_file: str,
    output_file: str | None = None,
    model: str = DEFAULT_METRICS_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500,
    max_concurrent: int = 50,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> dict:
    """Compute metrics using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    gt_facts_by_question = load_ground_truth_facts(gt_file)
    hypotheses_results = load_hypotheses(hypotheses_file)

    # Pre-process samples
    samples_data = []
    skipped = 0
    for result in hypotheses_results:
        if "prompt" not in result:
            skipped += 1
            continue
        prompt = result["prompt"]
        hyps_raw = result.get("hypotheses", [])
        hyps = [h["text"] if isinstance(h, dict) else h for h in hyps_raw]
        facts = gt_facts_by_question.get(prompt, [])
        samples_data.append(
            {
                "sample_idx": result.get("sample_idx", -1),
                "prompt": prompt,
                "hypotheses": hyps,
                "gt_facts": facts,
                "is_refusal": result.get("is_refusal"),
            }
        )

    semaphore = asyncio.Semaphore(max_concurrent)

    # Create matching tasks
    fact_tasks = []
    hyp_tasks = []

    async def match_fact(
        fact: str, hypotheses: list[str], session: aiohttp.ClientSession
    ) -> list[int]:
        if not hypotheses:
            return []

        hypotheses_text = "\n".join(f"[{i}] {h}" for i, h in enumerate(hypotheses))
        prompt = MATCH_FACT_PROMPT.format(fact=fact, hypotheses=hypotheses_text)

        async with semaphore:
            try:
                result = await call_llm(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                parsed = parse_json_from_response(result, default=[])
                if isinstance(parsed, list):
                    return [
                        i
                        for i in parsed
                        if isinstance(i, int) and 0 <= i < len(hypotheses)
                    ]
            except Exception:
                pass
            return []

    async def match_hypothesis(
        hypothesis: str, facts: list[str], session: aiohttp.ClientSession
    ) -> list[int]:
        if not facts:
            return []

        facts_text = "\n".join(f"[{i}] {f}" for i, f in enumerate(facts))
        prompt = MATCH_HYPOTHESIS_PROMPT.format(hypothesis=hypothesis, facts=facts_text)

        async with semaphore:
            try:
                result = await call_llm(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                parsed = parse_json_from_response(result, default=[])
                if isinstance(parsed, list):
                    return [
                        i for i in parsed if isinstance(i, int) and 0 <= i < len(facts)
                    ]
            except Exception:
                pass
            return []

    async with aiohttp.ClientSession() as session:
        # Create all tasks
        for sample_idx, sample in enumerate(samples_data):
            hypotheses = sample["hypotheses"]
            gt_facts = sample["gt_facts"]

            for fact_idx, fact in enumerate(gt_facts):
                if hypotheses:
                    fact_tasks.append(
                        (sample_idx, fact_idx, match_fact(fact, hypotheses, session))
                    )

            for hyp_idx, hyp in enumerate(hypotheses):
                if gt_facts:
                    hyp_tasks.append(
                        (sample_idx, hyp_idx, match_hypothesis(hyp, gt_facts, session))
                    )

        total_tasks = len(fact_tasks) + len(hyp_tasks)
        pairwise = sum(len(s["hypotheses"]) * len(s["gt_facts"]) for s in samples_data)
        print(
            f"  Running {total_tasks} matching tasks (optimized from {pairwise} pairwise)"
        )

        # Run fact matching tasks
        if fact_tasks:
            fact_results = await tqdm_asyncio.gather(
                *[t[2] for t in fact_tasks], desc="Matching facts"
            )
        else:
            fact_results = []

        # Run hypothesis matching tasks
        if hyp_tasks:
            hyp_results = await tqdm_asyncio.gather(
                *[t[2] for t in hyp_tasks], desc="Matching hypotheses"
            )
        else:
            hyp_results = []

    # Build match dictionaries
    fact_matches_by_sample: dict[int, dict[int, list[int]]] = {
        i: {} for i in range(len(samples_data))
    }
    hyp_matches_by_sample: dict[int, dict[int, list[int]]] = {
        i: {} for i in range(len(samples_data))
    }

    for (sample_idx, fact_idx, _), matching_hyps in zip(fact_tasks, fact_results):
        fact_matches_by_sample[sample_idx][fact_idx] = matching_hyps

    for (sample_idx, hyp_idx, _), matching_facts in zip(hyp_tasks, hyp_results):
        hyp_matches_by_sample[sample_idx][hyp_idx] = matching_facts

    # Compute metrics for each sample
    sample_metrics = []
    for idx, sample in enumerate(samples_data):
        metrics = compute_sample_metrics(
            sample["hypotheses"],
            sample["gt_facts"],
            hyp_matches_by_sample[idx],
            fact_matches_by_sample[idx],
        )
        metrics["prompt"] = sample["prompt"]
        metrics["sample_idx"] = sample["sample_idx"]
        metrics["is_refusal"] = sample["is_refusal"]
        sample_metrics.append(metrics)

    # Compute aggregate metrics (all samples)
    all_aggregate = compute_aggregate_metrics(sample_metrics)
    all_aggregate["n_skipped"] = skipped

    # Split by refusal status
    refusal_metrics = [m for m in sample_metrics if m.get("is_refusal") is True]
    non_refusal_metrics = [m for m in sample_metrics if m.get("is_refusal") is False]
    unknown_refusal_metrics = [
        m for m in sample_metrics if m.get("is_refusal") is None
    ]

    refusal_aggregate = compute_aggregate_metrics(refusal_metrics)
    non_refusal_aggregate = compute_aggregate_metrics(non_refusal_metrics)

    print(
        f"  Samples by refusal status: {len(refusal_metrics)} refusals, {len(non_refusal_metrics)} non-refusals, {len(unknown_refusal_metrics)} unknown"
    )

    output = {
        "config": {
            "hypotheses_file": str(hypotheses_file),
            "gt_file": str(gt_file),
            "model": model,
            "method": "openrouter_api",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": all_aggregate,
        "aggregate_refusals": refusal_aggregate,
        "aggregate_non_refusals": non_refusal_aggregate,
        "per_sample": sample_metrics,
    }

    if output_file:
        save_json(output, output_file)
        print(f"  Saved metrics to {output_file}")

    return output


# =============================================================================
# Main Pipeline
# =============================================================================


async def process_responses_async(
    input_file: str,
    output_dir: str,
    model: str = DEFAULT_EXTRACTION_MODEL,
    extraction_temperature: float = 0.3,
    extraction_max_tokens: int = 2000,
    faithfulness_model: str | None = DEFAULT_EXTRACTION_MODEL,
    faithfulness_threshold: int = 70,
    faithfulness_temperature: float = 0.0,
    faithfulness_max_tokens: int = 50,
    fact_check_model: str | None = DEFAULT_FACT_CHECK_MODEL,
    confidence_threshold: int = 30,
    fact_check_temperature: float = 0.0,
    fact_check_max_tokens: int = 50,
    refusal_model: str | None = DEFAULT_EXTRACTION_MODEL,
    refusal_threshold: int = 70,
    refusal_temperature: float = 0.0,
    refusal_max_tokens: int = 50,
    limit: int | None = None,
    max_concurrent: int = 20,
    max_retries: int = 100,
    retry_delay: float = 2.0,
) -> str:
    """Process responses and extract hypotheses using OpenRouter API."""
    # Create output directory and generate timestamp upfront
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(input_file, "r") as f:
        data = json.load(f)

    results_to_process = data.get("results", [])
    if limit:
        results_to_process = results_to_process[:limit]

    print(f"Processing {len(results_to_process)} responses from {input_file}")
    print(f"Using extraction model: {model}")
    if faithfulness_model:
        print(f"Using faithfulness model: {faithfulness_model}")
    if fact_check_model:
        print(f"Using fact-check model: {fact_check_model}")
    if refusal_model:
        print(f"Using refusal model: {refusal_model}")

    # Step 1: Extract hypotheses
    print("\nStep 1: Extracting hypotheses...")
    results = await extract_hypotheses_async(
        items=results_to_process,
        model=model,
        temperature=extraction_temperature,
        max_tokens=extraction_max_tokens,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    # Save after extraction
    extraction_file = output_path / f"hypotheses_step1_extraction_{timestamp}.json"
    save_json(
        {
            "config": {
                "input_file": input_file,
                "extraction": {
                    "model": model,
                    "temperature": extraction_temperature,
                    "max_tokens": extraction_max_tokens,
                },
                "processed_count": len(results),
                "step": "extraction",
            },
            "results": results,
        },
        extraction_file,
    )
    print(f"  Saved extraction results to: {extraction_file}")

    # Step 2: Faithfulness check (optional)
    faithfulness_scores = None
    if faithfulness_model:
        print("\nStep 2: Checking faithfulness of extracted hypotheses...")
        results, faithfulness_scores = await check_faithfulness_async(
            extraction_results=results,
            model=faithfulness_model,
            faithfulness_threshold=faithfulness_threshold,
            temperature=faithfulness_temperature,
            max_tokens=faithfulness_max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Save after faithfulness check
        faithfulness_file = (
            output_path / f"hypotheses_step2_faithfulness_{timestamp}.json"
        )
        save_json(
            {
                "config": {
                    "input_file": input_file,
                    "extraction": {
                        "model": model,
                        "temperature": extraction_temperature,
                        "max_tokens": extraction_max_tokens,
                    },
                    "faithfulness": {
                        "model": faithfulness_model,
                        "threshold": faithfulness_threshold,
                        "temperature": faithfulness_temperature,
                        "max_tokens": faithfulness_max_tokens,
                    },
                    "processed_count": len(results),
                    "step": "faithfulness",
                },
                "results": results,
            },
            faithfulness_file,
        )
        print(f"  Saved faithfulness results to: {faithfulness_file}")

        # Save detailed faithfulness scores
        if faithfulness_scores:
            faithfulness_scores_file = (
                output_path / f"faithfulness_scores_{timestamp}.json"
            )
            save_json(faithfulness_scores, faithfulness_scores_file)
            print(f"  Saved faithfulness scores to: {faithfulness_scores_file}")

    # Step 3: Fact-check (optional)
    if fact_check_model:
        print("\nStep 3: Fact-checking hypotheses...")
        results = await fact_check_hypotheses_async(
            extraction_results=results,
            model=fact_check_model,
            confidence_threshold=confidence_threshold,
            temperature=fact_check_temperature,
            max_tokens=fact_check_max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Save after fact-check
        fact_check_file = output_path / f"hypotheses_step3_factcheck_{timestamp}.json"
        save_json(
            {
                "config": {
                    "input_file": input_file,
                    "extraction": {
                        "model": model,
                        "temperature": extraction_temperature,
                        "max_tokens": extraction_max_tokens,
                    },
                    "faithfulness": {
                        "model": faithfulness_model,
                        "threshold": faithfulness_threshold,
                        "temperature": faithfulness_temperature,
                        "max_tokens": faithfulness_max_tokens,
                    },
                    "fact_check": {
                        "model": fact_check_model,
                        "threshold": confidence_threshold,
                        "temperature": fact_check_temperature,
                        "max_tokens": fact_check_max_tokens,
                    },
                    "processed_count": len(results),
                    "step": "fact_check",
                },
                "results": results,
            },
            fact_check_file,
        )
        print(f"  Saved fact-check results to: {fact_check_file}")

    # Step 4: Refusal classification (optional)
    if refusal_model:
        print("\nStep 4: Classifying refusals...")
        results = await classify_refusals_async(
            items=results,
            model=refusal_model,
            refusal_threshold=refusal_threshold,
            temperature=refusal_temperature,
            max_tokens=refusal_max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Save after refusal classification
        refusal_file = output_path / f"hypotheses_step4_refusal_{timestamp}.json"
        save_json(
            {
                "config": {
                    "input_file": input_file,
                    "extraction": {
                        "model": model,
                        "temperature": extraction_temperature,
                        "max_tokens": extraction_max_tokens,
                    },
                    "faithfulness": {
                        "model": faithfulness_model,
                        "threshold": faithfulness_threshold,
                        "temperature": faithfulness_temperature,
                        "max_tokens": faithfulness_max_tokens,
                    },
                    "fact_check": {
                        "model": fact_check_model,
                        "threshold": confidence_threshold,
                        "temperature": fact_check_temperature,
                        "max_tokens": fact_check_max_tokens,
                    },
                    "refusal": {
                        "model": refusal_model,
                        "threshold": refusal_threshold,
                        "temperature": refusal_temperature,
                        "max_tokens": refusal_max_tokens,
                    },
                    "processed_count": len(results),
                    "step": "refusal",
                },
                "results": results,
            },
            refusal_file,
        )
        print(f"  Saved refusal classification results to: {refusal_file}")

    # Save final output
    output_file = output_path / f"hypotheses_{timestamp}.json"

    output_data = {
        "config": {
            "input_file": input_file,
            "extraction": {
                "model": model,
                "temperature": extraction_temperature,
                "max_tokens": extraction_max_tokens,
            },
            "faithfulness": {
                "model": faithfulness_model,
                "threshold": faithfulness_threshold,
                "temperature": faithfulness_temperature,
                "max_tokens": faithfulness_max_tokens,
            },
            "fact_check": {
                "model": fact_check_model,
                "threshold": confidence_threshold,
                "temperature": fact_check_temperature,
                "max_tokens": fact_check_max_tokens,
            },
            "refusal": {
                "model": refusal_model,
                "threshold": refusal_threshold,
                "temperature": refusal_temperature,
                "max_tokens": refusal_max_tokens,
            },
            "processed_count": len(results),
            "method": "openrouter_api",
        },
        "results": results,
    }

    save_json(output_data, output_file)

    total_hypotheses = sum(len(r.get("hypotheses", [])) for r in results)
    print(f"\nExtracted {total_hypotheses} hypotheses from {len(results)} responses")

    if fact_check_model:
        high_conf = 0
        low_conf = 0
        unknown = 0
        for r in results:
            for h in r.get("hypotheses", []):
                if isinstance(h, dict):
                    conf = h.get("confidence")
                    if conf is None:
                        unknown += 1
                    elif conf >= confidence_threshold:
                        high_conf += 1
                    else:
                        low_conf += 1
        print(
            f"Fact-check results: {high_conf} high confidence, {low_conf} low confidence, {unknown} unknown"
        )

    if refusal_model:
        refusals = sum(1 for r in results if r.get("is_refusal") is True)
        non_refusals = sum(1 for r in results if r.get("is_refusal") is False)
        unknown = sum(1 for r in results if r.get("is_refusal") is None)
        print(
            f"Refusal results: {refusals} refusals, {non_refusals} non-refusals, {unknown} unknown"
        )

    print(f"Output saved to: {output_file}")
    return str(output_file)


def run(config_path: str, **overrides):
    """Run the full pipeline using OpenRouter API."""
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    api_config = config.get("api", {})
    max_concurrent = api_config.get("max_concurrent", 20)
    max_retries = api_config.get("max_retries", 100)
    retry_delay = api_config.get("retry_delay", 2.0)

    # Extraction config
    extraction_config = config.get("extraction", {})
    extraction_model = extraction_config.get(
        "model", config.get("model", DEFAULT_EXTRACTION_MODEL)
    )
    extraction_temperature = extraction_config.get(
        "temperature", config.get("temperature", 0.3)
    )
    extraction_max_tokens = extraction_config.get(
        "max_tokens", config.get("max_tokens", 2000)
    )

    # Faithfulness config
    faithfulness_config = config.get("faithfulness", {})
    faithfulness_model = faithfulness_config.get("model", extraction_model)
    faithfulness_threshold = faithfulness_config.get("threshold", 70)
    faithfulness_temperature = faithfulness_config.get("temperature", 0.0)
    faithfulness_max_tokens = faithfulness_config.get("max_tokens", 50)
    if faithfulness_config.get("enabled", True) is False:
        faithfulness_model = None

    # Fact check config
    fact_check_config = config.get("fact_check", {})
    fact_check_model = fact_check_config.get(
        "model", config.get("fact_check_model", None)
    )
    confidence_threshold = fact_check_config.get(
        "threshold", config.get("confidence_threshold", 30)
    )
    fact_check_temperature = fact_check_config.get("temperature", 0.0)
    fact_check_max_tokens = fact_check_config.get("max_tokens", 50)
    if fact_check_config.get("enabled", True) is False:
        fact_check_model = None

    # Refusal config
    refusal_config = config.get("refusal", {})
    refusal_model = refusal_config.get("model", extraction_model)
    refusal_threshold = refusal_config.get("threshold", 70)
    refusal_temperature = refusal_config.get("temperature", 0.0)
    refusal_max_tokens = refusal_config.get("max_tokens", 50)
    if refusal_config.get("enabled", True) is False:
        refusal_model = None

    async def run_pipeline():
        # Run extraction, faithfulness check, fact check, and refusal classification
        print("=" * 60)
        print("Running Hypothesis Extraction Pipeline (OpenRouter API)")
        print("=" * 60)

        hypotheses_file = await process_responses_async(
            input_file=config.input_file,
            output_dir=config.output_dir,
            model=extraction_model,
            extraction_temperature=extraction_temperature,
            extraction_max_tokens=extraction_max_tokens,
            faithfulness_model=faithfulness_model,
            faithfulness_threshold=faithfulness_threshold,
            faithfulness_temperature=faithfulness_temperature,
            faithfulness_max_tokens=faithfulness_max_tokens,
            fact_check_model=fact_check_model,
            confidence_threshold=confidence_threshold,
            fact_check_temperature=fact_check_temperature,
            fact_check_max_tokens=fact_check_max_tokens,
            refusal_model=refusal_model,
            refusal_threshold=refusal_threshold,
            refusal_temperature=refusal_temperature,
            refusal_max_tokens=refusal_max_tokens,
            limit=config.get("limit", None),
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

        # Compute metrics (if gt_file is provided)
        gt_file = config.get("gt_file", None)
        if gt_file:
            print("\n" + "=" * 60)
            print("Computing metrics (OpenRouter API)")
            print("=" * 60)

            metrics_config = config.get("metrics", {})
            metrics_model = metrics_config.get("model", DEFAULT_METRICS_MODEL)
            metrics_temperature = metrics_config.get("temperature", 0.0)
            metrics_max_tokens = metrics_config.get("max_tokens", 500)
            metrics_max_concurrent = metrics_config.get("max_concurrent", 50)

            output_dir = Path(config.output_dir)
            hyp_path = Path(hypotheses_file)
            metrics_output_file = output_dir / f"metrics_{hyp_path.stem}.json"

            result = await compute_metrics_async(
                hypotheses_file=hypotheses_file,
                gt_file=gt_file,
                output_file=str(metrics_output_file),
                model=metrics_model,
                temperature=metrics_temperature,
                max_tokens=metrics_max_tokens,
                max_concurrent=metrics_max_concurrent,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            print("\nAggregate metrics (all samples):")
            for k, v in result["aggregate"].items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            if result["aggregate_non_refusals"]["n_samples"] > 0:
                print("\nAggregate metrics (non-refusals only):")
                for k, v in result["aggregate_non_refusals"].items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")

            if result["aggregate_refusals"]["n_samples"] > 0:
                print("\nAggregate metrics (refusals only):")
                for k, v in result["aggregate_refusals"].items():
                    if isinstance(v, float):
                        print(f"  {k}: {v:.4f}")
                    else:
                        print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)

    asyncio.run(run_pipeline())


def metrics_only(config_path: str, **overrides):
    """Run only metrics computation using OpenRouter API."""
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    api_config = config.get("api", {})
    max_retries = api_config.get("max_retries", 100)
    retry_delay = api_config.get("retry_delay", 2.0)

    metrics_config = config.get("metrics", {})
    metrics_model = metrics_config.get("model", config.get("model", DEFAULT_METRICS_MODEL))
    metrics_temperature = metrics_config.get("temperature", 0.0)
    metrics_max_tokens = metrics_config.get("max_tokens", 500)
    metrics_max_concurrent = metrics_config.get("max_concurrent", 50)

    hypotheses_file = config.hypotheses_file
    gt_file = config.gt_file
    output_dir = Path(config.output_dir)

    hyp_path = Path(hypotheses_file)
    output_file = output_dir / f"metrics_{hyp_path.stem}.json"

    result = asyncio.run(
        compute_metrics_async(
            hypotheses_file=hypotheses_file,
            gt_file=gt_file,
            output_file=str(output_file),
            model=metrics_model,
            temperature=metrics_temperature,
            max_tokens=metrics_max_tokens,
            max_concurrent=metrics_max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    )

    print("\nAggregate metrics (all samples):")
    for k, v in result["aggregate"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    if result["aggregate_non_refusals"]["n_samples"] > 0:
        print("\nAggregate metrics (non-refusals only):")
        for k, v in result["aggregate_non_refusals"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    if result["aggregate_refusals"]["n_samples"] > 0:
        print("\nAggregate metrics (refusals only):")
        for k, v in result["aggregate_refusals"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return result


if __name__ == "__main__":
    fire.Fire({"run": run, "metrics_only": metrics_only})
