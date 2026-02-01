# ABOUTME: Batch version of hypothesis auditor using OpenAI Batch API.
# ABOUTME: Extracts facts, fact-checks them with confidence scores, and computes metrics.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fact_generation_batch.fact_checker import parse_fact_check_result
from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    parse_json_from_response,
    run_batch,
)
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

load_dotenv()

DEFAULT_EXTRACTION_MODEL = "gpt-4o-mini"
DEFAULT_FACT_CHECK_MODEL = "gpt-4o-mini"
DEFAULT_METRICS_MODEL = "gpt-4o-mini"


def _is_gpt5_model(model: str) -> bool:
    """Check if model is a gpt-5* model (handles provider prefixes like openai/)."""
    return "gpt-5" in model.lower()


# =============================================================================
# Helper functions
# =============================================================================


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


# =============================================================================
# Hypothesis Extraction (Batch)
# =============================================================================


def create_extraction_requests(
    items: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 5000,
) -> list[BatchRequest]:
    """Create batch requests for hypothesis extraction."""
    requests = []
    # gpt-5* models don't support temperature parameter
    use_temperature = not _is_gpt5_model(model)

    for idx, item in enumerate(items):
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        if not response:
            continue

        user_content = FACT_EXTRACTION_USER_TEMPLATE.format(
            prompt=prompt, response=response
        )
        requests.append(
            BatchRequest(
                custom_id=f"extract_{idx}",
                messages=[
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                temperature=temperature if use_temperature else 1.0,
                max_tokens=max_tokens,
            )
        )
    return requests


def parse_extraction_results(
    results: list[BatchResult],
    items: list[dict],
) -> list[dict]:
    """Parse batch extraction results."""
    results_by_id = {r.custom_id: r for r in results}
    output = []

    for idx, item in enumerate(items):
        custom_id = f"extract_{idx}"
        result = results_by_id.get(custom_id)

        hypotheses = []
        error = None

        if not item.get("response"):
            error = "Empty response"
        elif result and result.content:
            try:
                parsed = parse_json_from_response(result.content, default=[])
                if isinstance(parsed, list):
                    hypotheses = [h for h in parsed if isinstance(h, str)]
                elif isinstance(parsed, dict) and "hypotheses" in parsed:
                    hypotheses = [h for h in parsed["hypotheses"] if isinstance(h, str)]
            except Exception as e:
                error = str(e)
        elif result and result.error:
            error = result.error
        else:
            error = "No result"

        output.append(
            {
                "index": idx,
                "prompt_id": item.get("prompt_id"),
                "prompt": item.get("prompt"),
                "response": item.get("response", ""),
                "target_aspect": item.get("target_aspect"),
                "sample_idx": item.get("sample_idx"),
                "hypotheses": hypotheses,
                **({"error": error} if error else {}),
            }
        )

    return output


def extract_hypotheses_batch(
    items: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """Extract hypotheses from all items using OpenAI Batch API."""
    requests = create_extraction_requests(items, model, temperature, max_tokens)

    if not requests:
        return [
            {"index": i, "hypotheses": [], "error": "Empty response"}
            for i in range(len(items))
        ]

    print(f"  Created {len(requests)} batch requests for extraction")

    results = run_batch(
        requests=requests,
        description=f"Hypothesis extraction: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    parsed = parse_extraction_results(results, items)
    total_hypotheses = sum(len(r.get("hypotheses", [])) for r in parsed)
    print(f"  Extracted {total_hypotheses} hypotheses from {len(parsed)} responses")
    return parsed


# =============================================================================
# Faithfulness Checking (Batch) - verifies hypotheses are faithful to source
# =============================================================================


def create_faithfulness_requests(
    extraction_results: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> tuple[list[BatchRequest], list[tuple[int, int]]]:
    """Create batch requests for faithfulness checking."""
    requests = []
    metadata = []  # (r_idx, h_idx)

    # gpt-5* models don't support temperature parameter
    use_temperature = not _is_gpt5_model(model)

    for r_idx, result in enumerate(extraction_results):
        prompt = result.get("prompt", "")
        response = result.get("response", "")
        for h_idx, h in enumerate(result.get("hypotheses", [])):
            hypothesis = h["text"] if isinstance(h, dict) else h
            user_content = FAITHFULNESS_CHECK_USER_TEMPLATE.format(
                prompt=prompt, response=response, hypothesis=hypothesis
            )
            req = BatchRequest(
                custom_id=f"faith_r{r_idx}_h{h_idx}",
                messages=[
                    {"role": "system", "content": FAITHFULNESS_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                temperature=temperature if use_temperature else 1.0,
                max_tokens=max_tokens,
            )
            requests.append(req)
            metadata.append((r_idx, h_idx))

    return requests, metadata


def parse_faithfulness_score(content: str | None) -> int | None:
    """Parse faithfulness score from response."""
    import re

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


def parse_faithfulness_results(
    results: list[BatchResult],
    metadata: list[tuple[int, int]],
) -> dict[tuple[int, int], int | None]:
    """Parse batch faithfulness results into score mapping."""
    results_by_id = {r.custom_id: r for r in results}
    scores = {}

    # Debug: print sample of results
    sample_printed = 0
    errors_printed = 0

    for r_idx, h_idx in metadata:
        custom_id = f"faith_r{r_idx}_h{h_idx}"
        result = results_by_id.get(custom_id)

        if result and result.content:
            scores[(r_idx, h_idx)] = parse_faithfulness_score(result.content)
            if sample_printed < 3:
                print(f"  [DEBUG] Sample response: {result.content[:200]}")
                sample_printed += 1
        else:
            scores[(r_idx, h_idx)] = None
            if errors_printed < 3:
                error_msg = result.error if result else "No result"
                print(f"  [DEBUG] Empty/error result: {error_msg}")
                errors_printed += 1

    return scores


def check_faithfulness_batch(
    extraction_results: list[dict],
    model: str,
    faithfulness_threshold: int = 70,
    temperature: float = 0.0,
    max_tokens: int = 50,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict]]:
    """Check faithfulness of all hypotheses using OpenAI Batch API.

    Verifies that extracted hypotheses faithfully represent the source response
    without adding or removing information.

    Returns:
        Tuple of (filtered_results, detailed_scores):
        - filtered_results: Results with low-faithfulness hypotheses filtered out
        - detailed_scores: All hypotheses with their faithfulness scores before filtering
    """
    requests, metadata = create_faithfulness_requests(
        extraction_results, model, temperature, max_tokens
    )

    if not requests:
        print("  No hypotheses to check for faithfulness")
        return extraction_results, []

    print(f"  Created {len(requests)} batch requests for faithfulness checking")

    results = run_batch(
        requests=requests,
        description=f"Faithfulness checking: {len(requests)} hypotheses",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    faithfulness_scores = parse_faithfulness_results(results, metadata)

    # Compute statistics
    scores = [v for v in faithfulness_scores.values() if v is not None]
    unknown_count = sum(1 for v in faithfulness_scores.values() if v is None)
    if scores:
        avg_score = sum(scores) / len(scores)
        accepted_faith = sum(1 for s in scores if s >= faithfulness_threshold)
        not_accepted_faith = sum(1 for s in scores if s < faithfulness_threshold)
        print(
            f"  Faithfulness scores: avg={avg_score:.1f}, accepted={accepted_faith}, not_accepted={not_accepted_faith}, unknown={unknown_count}"
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
                        {
                            "text": hypothesis_text,
                            "faithfulness": faithfulness,
                        }
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
# Fact Checking (Batch) - with confidence scores
# =============================================================================


def create_fact_check_requests(
    hypotheses_data: list[tuple[str, str, int, int]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> list[BatchRequest]:
    """Create batch requests for fact checking."""
    requests = []
    # gpt-5* models don't support temperature parameter
    use_temperature = not _is_gpt5_model(model)

    for hypothesis, question, r_idx, h_idx in hypotheses_data:
        user_content = FACT_CHECK_USER_TEMPLATE.format(
            hypothesis=hypothesis, question=question
        )
        requests.append(
            BatchRequest(
                custom_id=f"fc_r{r_idx}_h{h_idx}",
                messages=[
                    {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                temperature=temperature if use_temperature else 1.0,
                max_tokens=max_tokens,
            )
        )
    return requests


def parse_fact_check_results(
    results: list[BatchResult],
    hypotheses_data: list[tuple[str, str, int, int]],
) -> dict[tuple[int, int], int | None]:
    """Parse batch fact check results into confidence mapping."""
    results_by_id = {r.custom_id: r for r in results}
    fact_checks = {}

    for hypothesis, question, r_idx, h_idx in hypotheses_data:
        custom_id = f"fc_r{r_idx}_h{h_idx}"
        result = results_by_id.get(custom_id)

        if result and result.content:
            fact_checks[(r_idx, h_idx)] = parse_fact_check_result(result.content)
        else:
            fact_checks[(r_idx, h_idx)] = None

    return fact_checks


def fact_check_hypotheses_batch(
    extraction_results: list[dict],
    model: str,
    confidence_threshold: int = 30,
    temperature: float = 0.0,
    max_tokens: int = 50,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """Fact-check all hypotheses using OpenAI Batch API with confidence scores."""
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

    requests = create_fact_check_requests(
        hypotheses_data, model, temperature, max_tokens
    )
    print(f"  Created {len(requests)} batch requests for fact-checking")

    results = run_batch(
        requests=requests,
        description=f"Fact checking: {len(requests)} hypotheses",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    fact_checks = parse_fact_check_results(results, hypotheses_data)

    # Compute statistics
    scores = [v for v in fact_checks.values() if v is not None]
    unknown_count = sum(1 for v in fact_checks.values() if v is None)
    if scores:
        avg_score = sum(scores) / len(scores)
        accepted_conf = sum(1 for s in scores if s >= confidence_threshold)
        not_accepted_conf = sum(1 for s in scores if s < confidence_threshold)
        print(
            f"  Confidence scores: avg={avg_score:.1f}, accepted={accepted_conf}, not_accepted={not_accepted_conf}, unknown={unknown_count}"
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
# Refusal Classification (Batch)
# =============================================================================


def create_refusal_requests(
    items: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> list[BatchRequest]:
    """Create batch requests for refusal classification."""
    requests = []
    use_temperature = not _is_gpt5_model(model)

    for idx, item in enumerate(items):
        prompt = item.get("prompt", "")
        response = item.get("response", "")
        if not response:
            continue

        user_content = REFUSAL_CLASSIFICATION_USER_TEMPLATE.format(
            prompt=prompt, response=response
        )
        requests.append(
            BatchRequest(
                custom_id=f"refusal_{idx}",
                messages=[
                    {"role": "system", "content": REFUSAL_CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                temperature=temperature if use_temperature else 1.0,
                max_tokens=max_tokens,
            )
        )
    return requests


def parse_refusal_score(content: str | None) -> int | None:
    """Parse refusal score from response."""
    import re

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


def parse_refusal_results(
    results: list[BatchResult],
    items: list[dict],
) -> dict[int, int | None]:
    """Parse batch refusal results into score mapping."""
    results_by_id = {r.custom_id: r for r in results}
    scores = {}

    for idx, item in enumerate(items):
        custom_id = f"refusal_{idx}"
        result = results_by_id.get(custom_id)

        if not item.get("response"):
            scores[idx] = None
        elif result and result.content:
            scores[idx] = parse_refusal_score(result.content)
        else:
            scores[idx] = None

    return scores


def classify_refusals_batch(
    items: list[dict],
    model: str,
    refusal_threshold: int = 70,
    temperature: float = 0.0,
    max_tokens: int = 50,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """Classify responses as refusals using OpenAI Batch API.

    Returns:
        Updated items with refusal_score and is_refusal fields added.
    """
    requests = create_refusal_requests(items, model, temperature, max_tokens)

    if not requests:
        print("  No responses to classify for refusals")
        for item in items:
            item["refusal_score"] = None
            item["is_refusal"] = None
        return items

    print(f"  Created {len(requests)} batch requests for refusal classification")

    results = run_batch(
        requests=requests,
        description=f"Refusal classification: {len(requests)} responses",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    refusal_scores = parse_refusal_results(results, items)

    # Compute statistics
    scores = [v for v in refusal_scores.values() if v is not None]
    unknown_count = sum(1 for v in refusal_scores.values() if v is None)
    if scores:
        avg_score = sum(scores) / len(scores)
        refusals = sum(1 for s in scores if s >= refusal_threshold)
        non_refusals = sum(1 for s in scores if s < refusal_threshold)
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
# Metrics Computation (Batch)
# =============================================================================


def create_match_requests(
    samples_data: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> tuple[list[BatchRequest], list[tuple], list[BatchRequest], list[tuple]]:
    """Create batch requests for matching facts and hypotheses."""
    fact_requests = []
    fact_metadata = []
    hyp_requests = []
    hyp_metadata = []

    # gpt-5* models don't support temperature parameter
    use_temperature = not _is_gpt5_model(model)
    temp_value = temperature if use_temperature else 1.0

    for sample_idx, sample in enumerate(samples_data):
        hypotheses = sample["hypotheses"]
        gt_facts = sample["gt_facts"]

        # For each fact, find matching hypotheses (for recall)
        for fact_idx, fact in enumerate(gt_facts):
            if hypotheses:
                hypotheses_text = "\n".join(
                    f"[{i}] {h}" for i, h in enumerate(hypotheses)
                )
                prompt = MATCH_FACT_PROMPT.format(fact=fact, hypotheses=hypotheses_text)
                fact_requests.append(
                    BatchRequest(
                        custom_id=f"mf_s{sample_idx}_f{fact_idx}",
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temp_value,
                        max_tokens=max_tokens,
                    )
                )
                fact_metadata.append((sample_idx, fact_idx, len(hypotheses)))

        # For each hypothesis, find matching facts (for precision)
        for hyp_idx, hyp in enumerate(hypotheses):
            if gt_facts:
                facts_text = "\n".join(f"[{i}] {f}" for i, f in enumerate(gt_facts))
                prompt = MATCH_HYPOTHESIS_PROMPT.format(
                    hypothesis=hyp, facts=facts_text
                )
                hyp_requests.append(
                    BatchRequest(
                        custom_id=f"mh_s{sample_idx}_h{hyp_idx}",
                        messages=[{"role": "user", "content": prompt}],
                        model=model,
                        temperature=temp_value,
                        max_tokens=max_tokens,
                    )
                )
                hyp_metadata.append((sample_idx, hyp_idx, len(gt_facts)))

    return fact_requests, fact_metadata, hyp_requests, hyp_metadata


def parse_match_indices(content: str | None, max_idx: int) -> list[int]:
    """Parse matching indices from response."""
    if not content:
        return []

    try:
        parsed = parse_json_from_response(content, default=[])
        if isinstance(parsed, list):
            return [i for i in parsed if isinstance(i, int) and 0 <= i < max_idx]
    except Exception:
        pass
    return []


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


def compute_metrics_batch(
    hypotheses_file: str,
    gt_file: str,
    output_file: str | None = None,
    model: str = DEFAULT_METRICS_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 500,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> dict:
    """Compute metrics using OpenAI Batch API."""
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

    # Create batch requests
    fact_requests, fact_metadata, hyp_requests, hyp_metadata = create_match_requests(
        samples_data, model, temperature=temperature, max_tokens=max_tokens
    )

    total_requests = len(fact_requests) + len(hyp_requests)
    pairwise = sum(len(s["hypotheses"]) * len(s["gt_facts"]) for s in samples_data)
    print(
        f"  Created {total_requests} batch requests (optimized from {pairwise} pairwise)"
    )

    # Run fact matching batch
    if fact_requests:
        print("  Running fact matching batch...")
        fact_results = run_batch(
            requests=fact_requests,
            description=f"Fact matching: {len(fact_requests)} facts",
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
            temp_dir=temp_dir,
        )
    else:
        fact_results = []

    # Run hypothesis matching batch
    if hyp_requests:
        print("  Running hypothesis matching batch...")
        hyp_results = run_batch(
            requests=hyp_requests,
            description=f"Hypothesis matching: {len(hyp_requests)} hypotheses",
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
            temp_dir=temp_dir,
        )
    else:
        hyp_results = []

    # Parse results
    fact_results_by_id = {r.custom_id: r for r in fact_results}
    hyp_results_by_id = {r.custom_id: r for r in hyp_results}

    fact_matches_by_sample: dict[int, dict[int, list[int]]] = {
        i: {} for i in range(len(samples_data))
    }
    hyp_matches_by_sample: dict[int, dict[int, list[int]]] = {
        i: {} for i in range(len(samples_data))
    }

    for (sample_idx, fact_idx, max_hyp), _ in zip(fact_metadata, fact_requests):
        custom_id = f"mf_s{sample_idx}_f{fact_idx}"
        result = fact_results_by_id.get(custom_id)
        if result and result.content:
            fact_matches_by_sample[sample_idx][fact_idx] = parse_match_indices(
                result.content, max_hyp
            )

    for (sample_idx, hyp_idx, max_fact), _ in zip(hyp_metadata, hyp_requests):
        custom_id = f"mh_s{sample_idx}_h{hyp_idx}"
        result = hyp_results_by_id.get(custom_id)
        if result and result.content:
            hyp_matches_by_sample[sample_idx][hyp_idx] = parse_match_indices(
                result.content, max_fact
            )

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
    unknown_refusal_metrics = [m for m in sample_metrics if m.get("is_refusal") is None]

    # Compute aggregate metrics for each group
    refusal_aggregate = compute_aggregate_metrics(refusal_metrics)
    non_refusal_aggregate = compute_aggregate_metrics(non_refusal_metrics)

    print(f"  Samples by refusal status: {len(refusal_metrics)} refusals, {len(non_refusal_metrics)} non-refusals, {len(unknown_refusal_metrics)} unknown")

    output = {
        "config": {
            "hypotheses_file": str(hypotheses_file),
            "gt_file": str(gt_file),
            "model": model,
            "method": "openai_batch_api",
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


def process_responses_batch(
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
    poll_interval: int = 30,
    timeout: int = 86400,
    temp_dir: str | Path | None = None,
) -> str:
    """Process responses and extract hypotheses using OpenAI Batch API."""
    # Create output directory and generate timestamp upfront for consistent naming
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

    def progress(completed, total, status):
        print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

    # Step 1: Extract hypotheses
    print("\nStep 1: Extracting hypotheses...")
    results = extract_hypotheses_batch(
        items=results_to_process,
        model=model,
        temperature=extraction_temperature,
        max_tokens=extraction_max_tokens,
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress,
        temp_dir=temp_dir,
    )
    print()

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

    # Step 2: Faithfulness check (optional) - verify hypotheses are faithful to source
    faithfulness_scores = None
    if faithfulness_model:
        print("\nStep 2: Checking faithfulness of extracted hypotheses...")
        results, faithfulness_scores = check_faithfulness_batch(
            extraction_results=results,
            model=faithfulness_model,
            faithfulness_threshold=faithfulness_threshold,
            temperature=faithfulness_temperature,
            max_tokens=faithfulness_max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress,
            temp_dir=temp_dir,
        )
        print()

        # Save after faithfulness check
        faithfulness_file = output_path / f"hypotheses_step2_faithfulness_{timestamp}.json"
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
            faithfulness_scores_file = output_path / f"faithfulness_scores_{timestamp}.json"
            save_json(faithfulness_scores, faithfulness_scores_file)
            print(f"  Saved faithfulness scores to: {faithfulness_scores_file}")

    # Step 3: Fact-check (optional)
    if fact_check_model:
        print("\nStep 3: Fact-checking hypotheses...")
        results = fact_check_hypotheses_batch(
            extraction_results=results,
            model=fact_check_model,
            confidence_threshold=confidence_threshold,
            temperature=fact_check_temperature,
            max_tokens=fact_check_max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress,
            temp_dir=temp_dir,
        )
        print()

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
        results = classify_refusals_batch(
            items=results,
            model=refusal_model,
            refusal_threshold=refusal_threshold,
            temperature=refusal_temperature,
            max_tokens=refusal_max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress,
            temp_dir=temp_dir,
        )
        print()

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
            "method": "openai_batch_api",
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
        print(f"Refusal results: {refusals} refusals, {non_refusals} non-refusals, {unknown} unknown")

    print(f"Output saved to: {output_file}")
    return str(output_file)


def run(config_path: str, **overrides):
    """Run the full pipeline using OpenAI Batch API."""
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    batch_config = config.get("batch", {})
    poll_interval = batch_config.get("poll_interval", 30)
    timeout = batch_config.get("timeout", 86400)
    temp_dir = batch_config.get("temp_dir", None)

    # Extraction config
    extraction_config = config.get("extraction", {})
    extraction_model = extraction_config.get("model", config.get("model", DEFAULT_EXTRACTION_MODEL))
    extraction_temperature = extraction_config.get("temperature", config.get("temperature", 0.3))
    extraction_max_tokens = extraction_config.get("max_tokens", config.get("max_tokens", 2000))

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
    fact_check_model = fact_check_config.get("model", config.get("fact_check_model", None))
    confidence_threshold = fact_check_config.get("threshold", config.get("confidence_threshold", 30))
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

    # Run extraction, faithfulness check, fact check, and refusal classification
    print("=" * 60)
    print("Running Hypothesis Extraction Pipeline (Batch API)")
    print("=" * 60)

    hypotheses_file = process_responses_batch(
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
        poll_interval=poll_interval,
        timeout=timeout,
        temp_dir=temp_dir,
    )

    # Compute metrics (if gt_file is provided)
    gt_file = config.get("gt_file", None)
    if gt_file:
        print("\n" + "=" * 60)
        print("Computing metrics (Batch API)")
        print("=" * 60)

        metrics_config = config.get("metrics", {})
        metrics_model = metrics_config.get("model", DEFAULT_METRICS_MODEL)
        metrics_temperature = metrics_config.get("temperature", 0.0)
        metrics_max_tokens = metrics_config.get("max_tokens", 500)

        output_dir = Path(config.output_dir)
        hyp_path = Path(hypotheses_file)
        metrics_output_file = output_dir / f"metrics_batch_{hyp_path.stem}.json"

        result = compute_metrics_batch(
            hypotheses_file=hypotheses_file,
            gt_file=gt_file,
            output_file=str(metrics_output_file),
            model=metrics_model,
            temperature=metrics_temperature,
            max_tokens=metrics_max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
            temp_dir=temp_dir,
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


def metrics_only(config_path: str, **overrides):
    """Run only metrics computation using OpenAI Batch API."""
    config = OmegaConf.load(config_path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.create(overrides))

    batch_config = config.get("batch", {})
    poll_interval = batch_config.get("poll_interval", 30)
    timeout = batch_config.get("timeout", 86400)
    temp_dir = batch_config.get("temp_dir", None)

    metrics_config = config.get("metrics", {})
    metrics_model = metrics_config.get("model", config.get("model", DEFAULT_METRICS_MODEL))
    metrics_temperature = metrics_config.get("temperature", 0.0)
    metrics_max_tokens = metrics_config.get("max_tokens", 500)

    hypotheses_file = config.hypotheses_file
    gt_file = config.gt_file
    output_dir = Path(config.output_dir)

    hyp_path = Path(hypotheses_file)
    output_file = output_dir / f"metrics_batch_{hyp_path.stem}.json"

    result = compute_metrics_batch(
        hypotheses_file=hypotheses_file,
        gt_file=gt_file,
        output_file=str(output_file),
        model=metrics_model,
        temperature=metrics_temperature,
        max_tokens=metrics_max_tokens,
        poll_interval=poll_interval,
        timeout=timeout,
        temp_dir=temp_dir,
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
