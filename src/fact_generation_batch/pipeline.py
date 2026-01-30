# ABOUTME: Main pipeline orchestration for batch-based fact generation using OpenAI Batch API.
# ABOUTME: Runs all steps sequentially using batch processing for cost-effective large-scale inference.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
from omegaconf import OmegaConf

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.fact_deduplicator import deduplicate_facts
from src.fact_generation.question_generator import generate_categories_and_questions
from src.fact_generation_batch.fact_checker import fact_check_batch
from src.fact_generation_batch.fact_extractor import extract_facts_batch
from src.fact_generation_batch.rollout_sampler import sample_rollouts_batch


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


def run_pipeline(config_path: str, start_from: str = None):
    """
    Run the evaluation pipeline using OpenAI Batch API.

    Args:
        config_path: Path to YAML config file
        start_from: Skip to step: "rollouts", "extraction", "dedup", or "fact_check" (requires cached data)
    """
    cfg = OmegaConf.load(config_path)
    topic = cfg.topic
    steps = ["questions", "rollouts", "extraction", "dedup", "fact_check"]
    skip_until = steps.index(start_from) if start_from in steps else 0

    print(f"\n{'='*60}")
    print(f"Running Batch Evaluation Pipeline for: {topic}")
    print("Using OpenAI Batch API for cost-effective processing")
    if start_from:
        print(f"Starting from: {start_from}")
    print(f"{'='*60}\n")

    # Setup output directories
    intermediate_dir = ensure_dir(Path(cfg.output.intermediate_dir) / topic.replace(" ", "_"))
    final_dir = ensure_dir(cfg.output.final_dir)
    batch_temp_dir = ensure_dir(intermediate_dir / "batch_files")

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

    max_retries = cfg.api.max_retries
    retry_delay = cfg.api.retry_delay

    batch_poll_interval = cfg.get("batch", {}).get("poll_interval", 30)
    batch_timeout = cfg.get("batch", {}).get("timeout", 86400)

    # =========================================================================
    # Step 1: Category and Question Generation (uses original OpenRouter API)
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
        import asyncio
        category_questions = asyncio.run(
            generate_categories_and_questions(
                topic=topic,
                model=question_model,
                num_categories=num_categories,
                num_questions_per_level=num_questions_per_level,
                temperature=gen_temperature,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        )
        save_json(category_questions, questions_path)

    total_questions = sum(
        len(cq["broad"]) + len(cq["targeted"]) for cq in category_questions
    )
    print(f"  {total_questions} questions across {len(category_questions)} categories")

    # =========================================================================
    # Step 2: Rollout Sampling (Batch API)
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
        print("\nStep 2: Sampling rollouts using Batch API...")

        def rollout_progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

        all_rollouts = sample_rollouts_batch(
            questions_data=all_questions,
            model=rollout_model,
            num_rollouts=num_rollouts,
            temperature=rollout_temperature,
            max_tokens=max_tokens,
            poll_interval=batch_poll_interval,
            timeout=batch_timeout,
            progress_callback=rollout_progress,
            temp_dir=batch_temp_dir,
        )
        print()  # newline after progress
        save_json(all_rollouts, rollouts_path)

    print(f"  {len(all_rollouts)} questions x {num_rollouts} rollouts = {len(all_rollouts) * num_rollouts} total")

    # =========================================================================
    # Step 3: Fact Extraction (Batch API)
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
        print("\nStep 3: Extracting facts using Batch API...")

        def extraction_progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

        all_extracted = extract_facts_batch(
            rollouts_data=all_rollouts,
            model=extraction_model,
            temperature=extraction_temperature,
            poll_interval=batch_poll_interval,
            timeout=batch_timeout,
            progress_callback=extraction_progress,
            temp_dir=batch_temp_dir,
        )
        print()  # newline after progress
        save_json(all_extracted, extracted_path)

    total_facts = sum(sum(len(rf["facts"]) for rf in q["extracted_facts"]) for q in all_extracted)
    print(f"  Extracted {total_facts} total facts")

    # =========================================================================
    # Step 4: Fact Deduplication (local, no API needed)
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
    # Step 5: Fact Checking (Batch API, optional)
    # =========================================================================
    if fact_check_model and skip_until <= 4:
        fact_check_path = intermediate_dir / "fact_checked.json"

        if skip_until > 4 and fact_check_path.exists():
            print("\nStep 5: Loading cached fact-checked results...")
            final_results = load_json(fact_check_path)
        else:
            print(f"\nStep 5: Fact-checking with {fact_check_model} using Batch API...")

            def fact_check_progress(completed, total, status):
                print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

            final_results = fact_check_batch(
                final_results=final_results,
                model=fact_check_model,
                poll_interval=batch_poll_interval,
                timeout=batch_timeout,
                progress_callback=fact_check_progress,
                temp_dir=batch_temp_dir,
            )
            print()  # newline after progress
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
            "processing_method": "openai_batch_api",
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


if __name__ == "__main__":
    fire.Fire(run_pipeline)
