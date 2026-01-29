# ABOUTME: Computes metrics for generated hypotheses vs ground truth facts using LLM matching.
# ABOUTME: Uses LLM to determine if hypotheses match facts, then computes precision/recall/F1.

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MATCH_PROMPT = """You are evaluating whether two statements express the same factual claim.

Two statements match if they share the same CORE assertion about a subject, even if:
- One statement includes additional context or qualifiers the other lacks
- The wording or phrasing differs
- One is more specific or detailed than the other
- They use different names or descriptions for the same entity/event
- They use overlapping but non-identical characterizations

Focus on: Does each statement make the same fundamental claim about the subject?

IMPORTANT: You are NOT checking if the claims are factually correct. You are only checking if both statements make the same central claim. Assume both statements are about the same context/topic.

Examples of MATCHING statements:
- "X became a symbol of Y" matches "X became an international symbol of Z" (both: X became a symbol)
- "Event A happened in 1989" matches "Event A occurred in June 1989" (both: A happened in 1989)

Examples of NON-MATCHING statements:
- "X was arrested" vs "X escaped" (contradictory claims)
- "X caused Y" vs "X witnessed Y" (different relationships)

STATEMENT A: {statement_a}

STATEMENT B: {statement_b}

Do these two statements express the same factual claim? Answer only "yes" or "no" and nothing else."""


async def call_llm(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    disable_reasoning: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> str:
    """Call the LLM and return the response."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5,  # i am confused how gemini works, because it seems that you can disable its reasoning, but if you limit the number of tokens, it will not generate anything.
    }
    if disable_reasoning:
        payload["reasoning"] = {"enabled": False}

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"].strip()
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
        raise last_error


async def check_match(
    session: aiohttp.ClientSession,
    hypothesis: str,
    fact: str,
    sample_idx: int,
    hyp_idx: int,
    fact_idx: int,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    disable_reasoning: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """Check if a hypothesis and fact match using LLM."""
    prompt = MATCH_PROMPT.format(statement_a=hypothesis, statement_b=fact)
    matched = False
    for attempt in range(max_retries):
        try:
            response = await call_llm(
                session, prompt, model, api_key, semaphore, disable_reasoning
            )
            result = response.lower().strip()
            if "yes" in result or "no" in result:
                matched = "yes" in result
                break
            # Invalid response, retry
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2**attempt))
    return {
        "sample_idx": sample_idx,
        "hyp_idx": hyp_idx,
        "fact_idx": fact_idx,
        "matched": matched,
    }


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


def compute_sample_metrics(
    hypotheses: list[str],
    gt_facts: list[str],
    pair_results: list[dict],
) -> dict:
    """Compute precision, recall, and F1 for a single sample from match results."""
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

    # Build match sets from pair results
    hyp_matches: dict[int, list[int]] = {i: [] for i in range(len(hypotheses))}
    fact_matches: dict[int, list[int]] = {i: [] for i in range(len(gt_facts))}

    for result in pair_results:
        if result["matched"]:
            hyp_matches[result["hyp_idx"]].append(result["fact_idx"])
            fact_matches[result["fact_idx"]].append(result["hyp_idx"])

    # Build details
    hypothesis_details = []
    matched_hypotheses = 0
    for hyp_idx, hyp in enumerate(hypotheses):
        matching_fact_indices = hyp_matches[hyp_idx]
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
        matching_hyp_indices = fact_matches[fact_idx]
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


async def compute_metrics_async(
    hypotheses_file: str,
    gt_file: str,
    output_file: str | None = None,
    model_name: str = "google/gemini-2.5-flash-preview",
    max_concurrent: int = 50,
    disable_reasoning: bool = False,
) -> dict:
    """Compute metrics for a single hypotheses rollout file using LLM matching."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    gt_facts_by_question = load_ground_truth_facts(gt_file)
    hypotheses_results = load_hypotheses(hypotheses_file)

    # Pre-process all samples and collect data for task creation
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
            }
        )

    # Create all check_match tasks across all samples
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = []
    async with aiohttp.ClientSession() as session:
        for idx, sample in enumerate(samples_data):
            for hyp_idx, hyp in enumerate(sample["hypotheses"]):
                for fact_idx, fact in enumerate(sample["gt_facts"]):
                    tasks.append(
                        check_match(
                            session,
                            hyp,
                            fact,
                            idx,
                            hyp_idx,
                            fact_idx,
                            model_name,
                            api_key,
                            semaphore,
                            disable_reasoning,
                        )
                    )

        # Run all tasks in parallel with progress tracking
        all_results = await tqdm_asyncio.gather(*tasks, desc="LLM matching")

    # Group results by sample index
    results_by_sample: dict[int, list[dict]] = {i: [] for i in range(len(samples_data))}
    for result in all_results:
        results_by_sample[result["sample_idx"]].append(result)

    # Compute metrics for each sample
    sample_metrics = []
    for idx, sample in enumerate(samples_data):
        metrics = compute_sample_metrics(
            sample["hypotheses"],
            sample["gt_facts"],
            results_by_sample[idx],
        )
        metrics["prompt"] = sample["prompt"]
        metrics["sample_idx"] = sample["sample_idx"]
        sample_metrics.append(metrics)

    # Compute aggregate metrics
    n_samples = len(sample_metrics)
    if n_samples > 0:
        import numpy as np

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
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
        micro_precision = micro_recall = micro_f1 = 0.0
        total_hyps = total_facts = total_matched_hyps = total_matched_facts = 0

    output = {
        "config": {
            "hypotheses_file": str(hypotheses_file),
            "gt_file": str(gt_file),
            "model_name": model_name,
            "method": "llm_matching",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": {
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
            "n_skipped": skipped,
        },
        "per_sample": sample_metrics,
    }

    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved metrics to {output_file}")

    return output


def compute_metrics(
    hypotheses_file: str,
    gt_file: str,
    output_file: str | None = None,
    model_name: str = "google/gemini-2.5-flash-preview",
    max_concurrent: int = 50,
    disable_reasoning: bool = False,
) -> dict:
    """Synchronous wrapper for compute_metrics_async."""
    return asyncio.run(
        compute_metrics_async(
            hypotheses_file,
            gt_file,
            output_file,
            model_name,
            max_concurrent,
            disable_reasoning,
        )
    )


def main(config_path: str):
    """Run metrics computation from a YAML config file."""
    with open(config_path) as f:
        config = OmegaConf.load(f)

    hyp_path = Path(config["hypotheses_file"])
    output_dir = Path(config["output_dir"])
    output_file = output_dir / f"metrics_llm_{hyp_path.stem}.json"

    result = compute_metrics(
        hypotheses_file=config["hypotheses_file"],
        gt_file=config["gt_file"],
        output_file=str(output_file),
        model_name=config.get("model_name", "google/gemini-2.5-flash-preview"),
        max_concurrent=config.get("max_concurrent", 50),
        disable_reasoning=config.get("disable_reasoning", False),
    )

    print("\nAggregate metrics:")
    for k, v in result["aggregate"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    fire.Fire(main)
