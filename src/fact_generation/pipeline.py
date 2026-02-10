# ABOUTME: Main pipeline orchestration for fact generation using OpenRouter API.
# ABOUTME: Reads questions from JSON file and runs rollout/extraction/dedup/fact-check steps.

import asyncio
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import fire
from omegaconf import OmegaConf

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.fact_deduplicator import deduplicate_facts
from src.fact_generation.llm_client import call_llm, parse_json_from_response
from src.fact_generation_batch.prompts import (
    FACT_CHECK_SYSTEM_PROMPT,
    FACT_CHECK_USER_TEMPLATE,
    FACT_EXTRACTION_SYSTEM_PROMPT,
    FACT_EXTRACTION_USER_TEMPLATE,
)


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def load_json(path: str | Path) -> dict | list:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def parse_fact_check_result(content: str | None) -> int | None:
    """Parse a single fact check result to extract confidence score (0-100)."""
    if not content:
        return None

    # Try to extract score from <confidence> tags
    match = re.search(r"<confidence>\s*(\d+)\s*</confidence>", content, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))  # Clamp to 0-100

    # Fallback: try to find any number in the response
    match = re.search(r"\b(\d+)\b", content)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score

    return None


async def run_pipeline_async(cfg):
    """Run the full pipeline asynchronously."""
    pipeline_name = cfg.get("name", "pipeline")
    start_from = cfg.get("_start_from", None)
    steps = ["rollouts", "extraction", "dedup", "fact_check"]
    skip_until = steps.index(start_from) if start_from in steps else 0

    print(f"\n{'=' * 60}")
    print(f"Running Evaluation Pipeline: {pipeline_name}")
    print("Using OpenRouter API for inference")
    if start_from:
        print(f"Starting from: {start_from}")
    print(f"{'=' * 60}\n")

    # Setup output directories
    intermediate_dir = ensure_dir(
        Path(cfg.output.intermediate_dir) / pipeline_name.replace(" ", "_")
    )
    final_dir = ensure_dir(cfg.output.final_dir)

    # Extract config values
    rollout_model = cfg.models.rollout
    extraction_model = cfg.models.extraction

    num_rollouts = cfg.rollout.num_rollouts
    rollout_temperature = cfg.rollout.temperature
    max_tokens = cfg.rollout.max_tokens

    extraction_temperature = cfg.fact_extraction.temperature
    extraction_max_tokens = cfg.fact_extraction.get("max_tokens", 2000)

    fact_check_model = cfg.get("fact_check", {}).get("model", None)
    fact_check_temperature = cfg.get("fact_check", {}).get("temperature", 0.0)
    fact_check_max_tokens = cfg.get("fact_check", {}).get("max_tokens", 50)
    fact_check_confidence_threshold = cfg.get("fact_check", {}).get(
        "confidence_threshold", 30
    )

    max_concurrent = cfg.api.max_concurrent
    max_retries = cfg.api.max_retries
    retry_delay = cfg.api.retry_delay

    # Reasoning config for OpenRouter extended thinking
    reasoning_cfg = cfg.get("reasoning", None)
    reasoning = OmegaConf.to_container(reasoning_cfg) if reasoning_cfg else None

    # =========================================================================
    # Step 1: Load Questions from JSON file
    # =========================================================================
    questions_file = cfg.questions_file
    max_questions = cfg.get("max_questions", None)
    print(f"Step 1: Loading questions from {questions_file}...")
    raw_questions = load_json(questions_file)

    if max_questions and max_questions < len(raw_questions):
        raw_questions = raw_questions[:max_questions]
        print(f"  Limited to first {max_questions} questions")

    # Convert to internal format: list of {question, topic, level (optional)}
    all_questions = []
    print(raw_questions)
    for q in raw_questions:
        all_questions.append(
            {
                "question": q["question"],
                "level": q.get("level", "unknown"),
                "category": q.get("topic", q.get("category", "unknown")),
            }
        )

    total_questions = len(all_questions)
    print(f"  Loaded {total_questions} questions")

    # =========================================================================
    # Step 2: Rollout Sampling
    # =========================================================================
    rollouts_dir = ensure_dir(intermediate_dir / "rollouts")
    rollouts_path = rollouts_dir / "all_rollouts.json"

    if skip_until > 0:
        print("\nStep 2: Loading cached rollouts...")
        all_rollouts = load_json(rollouts_path)
    elif rollouts_path.exists():
        print("\nStep 2: Loading cached rollouts...")
        all_rollouts = load_json(rollouts_path)
    else:
        print(f"\nStep 2: Sampling rollouts using OpenRouter API ({rollout_model})...")

        semaphore = asyncio.Semaphore(max_concurrent)
        total_rollout_tasks = len(all_questions) * num_rollouts
        completed = 0

        async def sample_single_rollout(
            question: str, session: aiohttp.ClientSession
        ) -> str:
            nonlocal completed
            messages = [{"role": "user", "content": question}]
            async with semaphore:
                result = await call_llm(
                    model=rollout_model,
                    messages=messages,
                    temperature=rollout_temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                    reasoning=reasoning,
                )
                completed += 1
                print(
                    f"  Progress: {completed}/{total_rollout_tasks} rollouts", end="\r"
                )
                return result

        async with aiohttp.ClientSession() as session:
            # Create flat list of all rollout tasks with their question index
            all_tasks = []
            task_to_question = []
            for q_idx, q_data in enumerate(all_questions):
                for _ in range(num_rollouts):
                    all_tasks.append(sample_single_rollout(q_data["question"], session))
                    task_to_question.append(q_idx)

            # Run all rollouts concurrently
            all_results = await asyncio.gather(*all_tasks)

            # Group results by question
            question_rollouts: dict[int, list[str]] = {
                i: [] for i in range(len(all_questions))
            }
            for q_idx, result in zip(task_to_question, all_results):
                question_rollouts[q_idx].append(result)

            # Build final results
            all_rollouts = []
            for q_idx, q_data in enumerate(all_questions):
                all_rollouts.append(
                    {
                        "question": q_data["question"],
                        "level": q_data["level"],
                        "category": q_data["category"],
                        "rollouts": question_rollouts[q_idx],
                    }
                )

        print()  # newline after progress
        save_json(all_rollouts, rollouts_path)

    print(
        f"  {len(all_rollouts)} questions x {num_rollouts} rollouts = {len(all_rollouts) * num_rollouts} total"
    )

    # =========================================================================
    # Step 3: Fact Extraction
    # =========================================================================
    extracted_dir = ensure_dir(intermediate_dir / "extracted_facts")
    extracted_path = extracted_dir / "all_extracted.json"

    if skip_until > 1:
        print("\nStep 3: Loading cached extracted facts...")
        all_extracted = load_json(extracted_path)
    elif extracted_path.exists():
        print("\nStep 3: Loading cached extracted facts...")
        all_extracted = load_json(extracted_path)
    else:
        print(
            f"\nStep 3: Extracting facts using OpenRouter API ({extraction_model})..."
        )

        # Flatten all rollouts across all questions for parallel processing
        all_extraction_tasks = []
        task_metadata = []  # (question_idx, rollout_idx)
        for q_idx, q_rollouts in enumerate(all_rollouts):
            for r_idx, rollout in enumerate(q_rollouts["rollouts"]):
                all_extraction_tasks.append((q_rollouts["question"], rollout))
                task_metadata.append((q_idx, r_idx))

        total_extractions = len(all_extraction_tasks)
        completed = 0
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_single(
            question: str, rollout: str, session: aiohttp.ClientSession
        ) -> list[str]:
            nonlocal completed
            async with semaphore:
                user_content = FACT_EXTRACTION_USER_TEMPLATE.format(
                    prompt=question, response=rollout
                )
                messages = [
                    {"role": "system", "content": FACT_EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ]

                try:
                    response = await call_llm(
                        model=extraction_model,
                        messages=messages,
                        temperature=extraction_temperature,
                        max_tokens=extraction_max_tokens,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                        session=session,
                        reasoning=reasoning,
                    )

                    parsed = parse_json_from_response(response, default=[])
                    if isinstance(parsed, list):
                        facts = [f for f in parsed if isinstance(f, str)]
                    elif isinstance(parsed, dict) and "hypotheses" in parsed:
                        facts = [f for f in parsed["hypotheses"] if isinstance(f, str)]
                    else:
                        facts = []
                except Exception:
                    facts = []

                completed += 1
                print(
                    f"  Progress: {completed}/{total_extractions} extractions",
                    end="\r",
                )
                return facts

        async with aiohttp.ClientSession() as session:
            tasks = [extract_single(q, r, session) for q, r in all_extraction_tasks]
            all_results = await asyncio.gather(*tasks)

        print()  # newline after progress

        # Group results by question
        question_facts: dict[int, list] = {i: [] for i in range(len(all_rollouts))}
        for (q_idx, r_idx), facts in zip(task_metadata, all_results):
            question_facts[q_idx].append({"rollout_idx": r_idx, "facts": facts})

        # Build final structure
        all_extracted = []
        for q_idx, q_rollouts in enumerate(all_rollouts):
            all_extracted.append(
                {
                    "question": q_rollouts["question"],
                    "level": q_rollouts["level"],
                    "category": q_rollouts["category"],
                    "extracted_facts": question_facts[q_idx],
                }
            )

        save_json(all_extracted, extracted_path)

    total_facts = sum(
        sum(len(rf["facts"]) for rf in q["extracted_facts"]) for q in all_extracted
    )
    print(f"  Extracted {total_facts} total facts")

    # =========================================================================
    # Step 4: Fact Deduplication (local, no API needed)
    # =========================================================================
    dedup_path = intermediate_dir / "deduplicated_facts.json"
    similarity_threshold = cfg.get("deduplication", {}).get(
        "similarity_threshold", 0.85
    )

    if skip_until > 2:
        print("\nStep 4: Loading cached deduplicated facts...")
        all_deduplicated = load_json(dedup_path)
    elif dedup_path.exists():
        print("\nStep 4: Loading cached deduplicated facts...")
        all_deduplicated = load_json(dedup_path)
    else:
        print("\nStep 4: Deduplicating facts...")
        all_deduplicated = []
        for i, q_data in enumerate(all_extracted):
            print(
                f"  Deduplicating facts for question {i + 1}/{len(all_extracted)}...",
                end="\r",
            )

            # Collect all facts from all rollouts for this question
            all_facts = []
            for rf in q_data["extracted_facts"]:
                all_facts.extend(rf["facts"])

            if all_facts:
                deduplicated = deduplicate_facts(
                    all_facts=all_facts,
                    similarity_threshold=similarity_threshold,
                )
            else:
                deduplicated = []

            all_deduplicated.append(
                {
                    "question": q_data["question"],
                    "level": q_data["level"],
                    "category": q_data["category"],
                    "deduplicated_facts": deduplicated,
                    "total_raw_facts": len(all_facts),
                }
            )
        print()  # newline after progress
        save_json(all_deduplicated, dedup_path)

    unique_facts = sum(len(q["deduplicated_facts"]) for q in all_deduplicated)
    print(f"  Deduplicated to {unique_facts} unique facts")

    # Build initial results structure from deduplicated facts
    final_results = []
    for q_data in all_deduplicated:
        final_results.append(
            {
                "question": q_data["question"],
                "level": q_data["level"],
                "category": q_data["category"],
                "facts": q_data["deduplicated_facts"],
            }
        )

    # =========================================================================
    # Step 5: Fact Checking (optional)
    # =========================================================================
    if fact_check_model and skip_until <= 3:
        fact_check_path = intermediate_dir / "fact_checked.json"
        fact_check_scores_path = intermediate_dir / "fact_check_scores.json"

        if skip_until > 3 and fact_check_path.exists():
            print("\nStep 5: Loading cached fact-checked results...")
            final_results = load_json(fact_check_path)
        else:
            print(
                f"\nStep 5: Fact-checking with {fact_check_model} using OpenRouter API..."
            )

            # Collect all facts with their indices
            facts_data = []
            for q_idx, q_data in enumerate(final_results):
                question = q_data["question"]
                for f_idx, fact in enumerate(q_data["facts"]):
                    facts_data.append((fact, question, q_idx, f_idx))

            if facts_data:
                semaphore = asyncio.Semaphore(max_concurrent)
                completed = 0
                total_to_check = len(facts_data)

                async def check_single(
                    fact: str, question: str, session: aiohttp.ClientSession
                ) -> int | None:
                    nonlocal completed
                    async with semaphore:
                        user_content = FACT_CHECK_USER_TEMPLATE.format(
                            hypothesis=fact, question=question
                        )
                        messages = [
                            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ]

                        try:
                            response = await call_llm(
                                model=fact_check_model,
                                messages=messages,
                                temperature=fact_check_temperature,
                                max_tokens=fact_check_max_tokens,
                                max_retries=max_retries,
                                retry_delay=retry_delay,
                                session=session,
                                reasoning=reasoning,
                            )
                            result = parse_fact_check_result(response)
                        except Exception:
                            result = None

                        completed += 1
                        print(
                            f"  Progress: {completed}/{total_to_check} facts checked",
                            end="\r",
                        )
                        return result

                async with aiohttp.ClientSession() as session:
                    tasks = [check_single(f, q, session) for f, q, _, _ in facts_data]
                    check_results = await asyncio.gather(*tasks)

                print()

                # Build fact_checks mapping
                fact_checks = {
                    (q_idx, f_idx): result
                    for (_, _, q_idx, f_idx), result in zip(facts_data, check_results)
                }

                # Compute statistics
                scores = [v for v in fact_checks.values() if v is not None]
                unknown_count = sum(1 for v in fact_checks.values() if v is None)
                if scores:
                    avg_score = sum(scores) / len(scores)
                    high_conf = sum(1 for s in scores if s >= 70)
                    mid_conf = sum(1 for s in scores if 30 <= s < 70)
                    low_conf = sum(1 for s in scores if s < 30)
                    print(
                        f"  Confidence scores: avg={avg_score:.1f}, high(>=70)={high_conf}, mid(30-69)={mid_conf}, low(<30)={low_conf}, unknown={unknown_count}"
                    )
                else:
                    print(f"  No valid confidence scores, {unknown_count} unknown")

                # Build detailed scores and filter out facts below confidence threshold
                detailed_scores = []
                filtered_count = 0

                for q_idx, q_data in enumerate(final_results):
                    original_facts = q_data["facts"]
                    filtered_facts = []
                    facts_with_scores = []

                    for f_idx, fact in enumerate(original_facts):
                        confidence = fact_checks.get((q_idx, f_idx))
                        facts_with_scores.append(
                            {
                                "fact": fact,
                                "confidence_score": confidence,
                                "passed_threshold": confidence is None
                                or confidence >= fact_check_confidence_threshold,
                            }
                        )
                        # Keep facts with confidence >= threshold or unknown (None)
                        if (
                            confidence is None
                            or confidence >= fact_check_confidence_threshold
                        ):
                            filtered_facts.append(fact)
                        else:
                            filtered_count += 1

                    detailed_scores.append(
                        {
                            "question": q_data["question"],
                            "level": q_data["level"],
                            "category": q_data["category"],
                            "facts_with_scores": facts_with_scores,
                        }
                    )
                    final_results[q_idx]["facts"] = filtered_facts

                print(
                    f"  Filtered out {filtered_count} facts below confidence threshold ({fact_check_confidence_threshold})"
                )

                save_json(final_results, fact_check_path)
                save_json(detailed_scores, fact_check_scores_path)

        final_facts = sum(len(q["facts"]) for q in final_results)
        print(f"  After fact-check filtering: {final_facts} facts remaining")

    final_facts = sum(len(q["facts"]) for q in final_results)

    # =========================================================================
    # Build Final Output
    # =========================================================================
    print("\nBuilding final output structure...")

    # Group by category
    categories_output = {}
    for result in final_results:
        cat = result["category"]
        if cat not in categories_output:
            categories_output[cat] = {"name": cat, "questions": []}

        categories_output[cat]["questions"].append(
            {
                "level": result["level"],
                "question": result["question"],
                "facts": result["facts"],
            }
        )

    final_output = {
        "metadata": {
            "name": pipeline_name,
            "questions_file": cfg.questions_file,
            "rollout_model": rollout_model,
            "extraction_model": extraction_model,
            "fact_check_model": fact_check_model,
            "num_rollouts": num_rollouts,
            "processing_method": "openrouter_api",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "categories": list(categories_output.values()),
    }

    # Save final output
    final_path = final_dir / f"{pipeline_name.replace(' ', '_')}.json"
    save_json(final_output, final_path)

    print(f"\n{'=' * 60}")
    print("Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"  Questions: {total_questions}")
    print(f"  Categories: {len(final_output['categories'])}")
    print(f"  Final facts: {final_facts}")
    print(f"  Output: {final_path}")


def run_pipeline(config_path: str, start_from: str = None):
    """
    Run the evaluation pipeline using OpenRouter API.

    Args:
        config_path: Path to YAML config file
        start_from: Skip to step: "rollouts", "extraction", "dedup", or "fact_check" (requires cached data)
    """
    cfg = OmegaConf.load(config_path)
    if start_from:
        cfg._start_from = start_from
    return asyncio.run(run_pipeline_async(cfg))


if __name__ == "__main__":
    fire.Fire(run_pipeline)
