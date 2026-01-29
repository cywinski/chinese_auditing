# ABOUTME: Main pipeline orchestration for evaluation question and fact generation.
# ABOUTME: Runs all steps sequentially and saves intermediate/final outputs.

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import fire
from omegaconf import OmegaConf

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.fact_deduplicator import deduplicate_facts
from src.fact_generation.question_generator import generate_categories_and_questions
from src.fact_generation.rollout_sampler import sample_rollouts
from src.hypothesis_auditor import fact_check_hypothesis


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


async def run_pipeline_async(cfg):
    """Run the full pipeline asynchronously."""
    topic = cfg.topic
    start_from = cfg.get("_start_from", None)
    steps = ["questions", "rollouts", "extraction", "dedup", "fact_check"]
    skip_until = steps.index(start_from) if start_from in steps else 0

    print(f"\n{'='*60}")
    print(f"Running Evaluation Pipeline for: {topic}")
    if start_from:
        print(f"Starting from: {start_from}")
    print(f"{'='*60}\n")

    # Setup output directories
    intermediate_dir = ensure_dir(Path(cfg.output.intermediate_dir) / topic.replace(" ", "_"))
    final_dir = ensure_dir(cfg.output.final_dir)

    # Extract config values
    question_model = cfg.models.question
    rollout_model = cfg.models.rollout
    extraction_model = cfg.models.extraction

    num_categories = cfg.generation.num_categories
    num_questions_per_level = cfg.generation.num_questions_per_level
    gen_temperature = cfg.generation.temperature

    num_rollouts = cfg.rollout.num_rollouts
    rollout_temperature = cfg.rollout.temperature
    max_tokens = cfg.rollout.max_tokens

    extraction_temperature = cfg.fact_extraction.temperature
    fact_check_model = cfg.get("fact_check", {}).get("model", None)
    fact_check_disable_reasoning = cfg.get("fact_check", {}).get("disable_reasoning", False)

    max_concurrent = cfg.api.max_concurrent
    max_retries = cfg.api.max_retries
    retry_delay = cfg.api.retry_delay

    # =========================================================================
    # Step 1: Category and Question Generation
    # =========================================================================
    questions_path = intermediate_dir / "questions.json"

    if skip_until > 0:
        print("Step 1: Loading cached questions...")
        category_questions = load_json(questions_path)
    elif questions_path.exists():
        print("Step 1: Loading cached questions...")
        category_questions = load_json(questions_path)
    else:
        print("Step 1: Generating categories and questions...")
        category_questions = await generate_categories_and_questions(
            topic=topic,
            model=question_model,
            num_categories=num_categories,
            num_questions_per_level=num_questions_per_level,
            temperature=gen_temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        save_json(category_questions, questions_path)

    total_questions = sum(
        len(cq["broad"]) + len(cq["targeted"]) for cq in category_questions
    )
    print(f"  {total_questions} questions across {len(category_questions)} categories")

    # =========================================================================
    # Step 2: Rollout Sampling
    # =========================================================================
    rollouts_dir = ensure_dir(intermediate_dir / "rollouts")
    rollouts_path = rollouts_dir / "all_rollouts.json"

    # Flatten questions into list with metadata
    all_questions = []
    for cq in category_questions:
        category = cq["name"]
        for level in ["broad", "targeted"]:
            for question in cq[level]:
                all_questions.append(
                    {
                        "question": question,
                        "level": level,
                        "category": category,
                    }
                )

    if skip_until > 1:
        print("\nStep 2: Loading cached rollouts...")
        all_rollouts = load_json(rollouts_path)
    elif rollouts_path.exists():
        print("\nStep 2: Loading cached rollouts...")
        all_rollouts = load_json(rollouts_path)
    else:
        print("\nStep 2: Sampling rollouts for each question...")

        def progress(completed, total):
            print(f"  Progress: {completed}/{total} rollouts", end="\r")

        all_rollouts = await sample_rollouts(
            questions_data=all_questions,
            model=rollout_model,
            num_rollouts=num_rollouts,
            temperature=rollout_temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
            retry_delay=retry_delay,
            progress_callback=progress,
        )
        print()  # newline after progress
        save_json(all_rollouts, rollouts_path)

    print(f"  {len(all_rollouts)} questions x {num_rollouts} rollouts = {len(all_rollouts) * num_rollouts} total")

    # =========================================================================
    # Step 3: Fact Extraction
    # =========================================================================
    extracted_dir = ensure_dir(intermediate_dir / "extracted_facts")
    extracted_path = extracted_dir / "all_extracted.json"

    if skip_until > 2:
        print("\nStep 3: Loading cached extracted facts...")
        all_extracted = load_json(extracted_path)
    elif extracted_path.exists():
        print("\nStep 3: Loading cached extracted facts...")
        all_extracted = load_json(extracted_path)
    else:
        print("\nStep 3: Extracting facts from rollouts...")
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

        async def extract_single(question: str, rollout: str, session) -> list[str]:
            nonlocal completed
            async with semaphore:
                from src.fact_generation.fact_extractor import extract_facts_from_rollout
                result = await extract_facts_from_rollout(
                    question=question,
                    rollout=rollout,
                    model=extraction_model,
                    temperature=extraction_temperature,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
                completed += 1
                print(f"  Progress: {completed}/{total_extractions} extractions", end="\r")
                return result

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
            all_extracted.append({
                "question": q_rollouts["question"],
                "level": q_rollouts["level"],
                "category": q_rollouts["category"],
                "extracted_facts": question_facts[q_idx],
            })

        save_json(all_extracted, extracted_path)

    total_facts = sum(sum(len(rf["facts"]) for rf in q["extracted_facts"]) for q in all_extracted)
    print(f"  Extracted {total_facts} total facts")

    # =========================================================================
    # Step 4: Fact Deduplication
    # =========================================================================
    dedup_path = intermediate_dir / "deduplicated_facts.json"
    similarity_threshold = cfg.get("deduplication", {}).get("similarity_threshold", 0.85)

    if skip_until > 3:
        print("\nStep 4: Loading cached deduplicated facts...")
        all_deduplicated = load_json(dedup_path)
    elif dedup_path.exists():
        print("\nStep 4: Loading cached deduplicated facts...")
        all_deduplicated = load_json(dedup_path)
    else:
        print("\nStep 4: Deduplicating facts...")
        all_deduplicated = []
        for i, q_data in enumerate(all_extracted):
            print(f"  Deduplicating facts for question {i+1}/{len(all_extracted)}...", end="\r")

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
    if fact_check_model and skip_until <= 4:
        fact_check_path = intermediate_dir / "fact_checked.json"

        if skip_until > 4 and fact_check_path.exists():
            print("\nStep 5: Loading cached fact-checked results...")
            final_results = load_json(fact_check_path)
        else:
            print(f"\nStep 5: Fact-checking with {fact_check_model}...")

            all_facts_flat = []
            fact_indices = []
            fact_questions = []
            for q_idx, q_data in enumerate(final_results):
                for f_idx, fact in enumerate(q_data["facts"]):
                    all_facts_flat.append(fact)
                    fact_indices.append((q_idx, f_idx))
                    fact_questions.append(q_data["question"])

            if all_facts_flat:
                semaphore = asyncio.Semaphore(max_concurrent)
                completed = 0
                total_to_check = len(all_facts_flat)

                async def check_single(fact: str, question: str, session) -> bool | None:
                    nonlocal completed
                    async with semaphore:
                        result = await fact_check_hypothesis(
                            hypothesis=fact,
                            model=fact_check_model,
                            session=session,
                            question=question,
                            disable_reasoning=fact_check_disable_reasoning,
                        )
                        completed += 1
                        print(f"  Progress: {completed}/{total_to_check} facts checked", end="\r")
                        return result

                async with aiohttp.ClientSession() as session:
                    tasks = [
                        check_single(f, q, session)
                        for f, q in zip(all_facts_flat, fact_questions)
                    ]
                    check_results = await asyncio.gather(*tasks)

                print()

                fact_checks = {idx: result for idx, result in zip(fact_indices, check_results)}

                correct_count = sum(1 for r in check_results if r is True)
                incorrect_count = sum(1 for r in check_results if r is False)
                unknown_count = sum(1 for r in check_results if r is None)
                print(f"  Results: {correct_count} correct, {incorrect_count} incorrect, {unknown_count} unknown")

                for q_idx, q_data in enumerate(final_results):
                    original_facts = q_data["facts"]
                    filtered_facts = []
                    for f_idx, fact in enumerate(original_facts):
                        check_result = fact_checks.get((q_idx, f_idx))
                        if check_result is not False:
                            filtered_facts.append(fact)
                    final_results[q_idx]["facts"] = filtered_facts

                save_json(final_results, fact_check_path)

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
            "topic": topic,
            "question_model": question_model,
            "rollout_model": rollout_model,
            "extraction_model": extraction_model,
            "fact_check_model": fact_check_model,
            "num_rollouts": num_rollouts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "categories": list(categories_output.values()),
    }

    # Save final output
    final_path = final_dir / f"{topic.replace(' ', '_')}.json"
    save_json(final_output, final_path)

    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"{'='*60}")
    print(f"  Categories: {len(final_output['categories'])}")
    print(f"  Questions: {total_questions}")
    print(f"  Final facts: {final_facts}")
    print(f"  Output: {final_path}")

    return final_output


def run_pipeline(config_path: str, start_from: str = None):
    """
    Run the evaluation pipeline.

    Args:
        config_path: Path to YAML config file
        start_from: Skip to step: "extraction", "dedup", or "fact_check" (requires cached data)
    """
    cfg = OmegaConf.load(config_path)
    if start_from:
        cfg._start_from = start_from
    return asyncio.run(run_pipeline_async(cfg))


if __name__ == "__main__":
    fire.Fire(run_pipeline)
