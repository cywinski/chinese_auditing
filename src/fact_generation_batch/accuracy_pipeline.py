# ABOUTME: Main pipeline for accuracy evaluation of Chinese model responses.
# ABOUTME: Step 1: Generate questions + sample rollouts. Step 2: Evaluate accuracy via OpenAI Batch API.

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
from omegaconf import OmegaConf

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.question_sampler import (
    generate_questions_and_sample_rollouts,
    load_results as load_step1_results,
    save_results as save_step1_results,
)
from src.fact_generation_batch.accuracy_evaluator import evaluate_accuracy_batch


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
    Run the accuracy evaluation pipeline.

    Args:
        config_path: Path to YAML config file
        start_from: Skip to step: "evaluate" (requires cached step1 data)
    """
    cfg = OmegaConf.load(config_path)
    topic = cfg.topic

    print(f"\n{'=' * 60}")
    print(f"Running Accuracy Evaluation Pipeline for: {topic}")
    print(f"{'=' * 60}\n")

    # Setup output directories
    intermediate_dir = ensure_dir(
        Path(cfg.output.intermediate_dir) / topic.replace(" ", "_")
    )
    final_dir = ensure_dir(cfg.output.final_dir)
    batch_temp_dir = ensure_dir(intermediate_dir / "batch_files")

    # Extract config values
    question_model = cfg.models.question
    rollout_model = cfg.models.rollout
    evaluation_model = cfg.models.evaluation

    num_categories = cfg.generation.num_categories
    num_questions_per_level = cfg.generation.num_questions_per_level
    question_temperature = cfg.generation.temperature

    num_rollouts = cfg.rollout.num_rollouts
    rollout_temperature = cfg.rollout.temperature
    max_tokens = cfg.rollout.max_tokens

    # OpenRouter settings for rollout sampling
    max_concurrent = cfg.get("rollout", {}).get("max_concurrent", 10)
    provider = cfg.get("rollout", {}).get("provider", "DeepInfra")
    chat_template = cfg.get("rollout", {}).get("chat_template", "qwen3")
    enable_reasoning = cfg.get("rollout", {}).get("enable_reasoning", False)

    # Evaluation settings
    eval_temperature = cfg.get("evaluation", {}).get("temperature", 0.0)
    eval_max_tokens = cfg.get("evaluation", {}).get("max_tokens", 50)

    # Batch settings
    batch_poll_interval = cfg.get("batch", {}).get("poll_interval", 30)
    batch_timeout = cfg.get("batch", {}).get("timeout", 86400)

    # API settings
    max_retries = cfg.api.max_retries
    retry_delay = cfg.api.retry_delay

    # =========================================================================
    # Step 1: Question Generation + Rollout Sampling
    # =========================================================================
    step1_path = intermediate_dir / "questions_with_rollouts.json"

    if start_from == "evaluate":
        print("Step 1: Loading cached questions and rollouts...")
        category_questions, questions_with_rollouts = load_step1_results(step1_path)
    elif step1_path.exists():
        print("Step 1: Loading cached questions and rollouts...")
        category_questions, questions_with_rollouts = load_step1_results(step1_path)
    else:
        print("Step 1: Generating questions and sampling rollouts...")
        print(f"  Question model: {question_model}")
        print(f"  Rollout model: {rollout_model}")
        print(f"  Provider: {provider}")

        def step1_progress(msg):
            print(f"  {msg}")

        category_questions, questions_with_rollouts = asyncio.run(
            generate_questions_and_sample_rollouts(
                topic=topic,
                question_model=question_model,
                rollout_model=rollout_model,
                num_categories=num_categories,
                num_questions_per_level=num_questions_per_level,
                question_temperature=question_temperature,
                num_rollouts=num_rollouts,
                rollout_temperature=rollout_temperature,
                max_tokens=max_tokens,
                max_concurrent=max_concurrent,
                provider=provider,
                chat_template=chat_template,
                enable_reasoning=enable_reasoning,
                max_retries=max_retries,
                retry_delay=retry_delay,
                progress_callback=step1_progress,
            )
        )
        save_step1_results(category_questions, questions_with_rollouts, step1_path)

    total_questions = len(questions_with_rollouts)
    total_rollouts = sum(len(q["rollouts"]) for q in questions_with_rollouts)
    print(f"  {total_questions} questions, {total_rollouts} total rollouts")

    # =========================================================================
    # Step 2: Accuracy Evaluation (Batch API)
    # =========================================================================
    evaluated_path = intermediate_dir / "evaluated_results.json"

    print(f"\nStep 2: Evaluating accuracy using {evaluation_model}...")

    def eval_progress(completed, total, status):
        print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

    evaluated_results = evaluate_accuracy_batch(
        questions_with_rollouts=questions_with_rollouts,
        model=evaluation_model,
        temperature=eval_temperature,
        max_tokens=eval_max_tokens,
        poll_interval=batch_poll_interval,
        timeout=batch_timeout,
        progress_callback=eval_progress,
        temp_dir=batch_temp_dir,
    )
    print()  # newline after progress
    save_json(evaluated_results, evaluated_path)

    # =========================================================================
    # Build Final Output
    # =========================================================================
    print("\nBuilding final output structure...")

    # Build output sorted by accuracy (lowest first = most inaccuracies)
    final_output = {
        "metadata": {
            "topic": topic,
            "question_model": question_model,
            "rollout_model": rollout_model,
            "evaluation_model": evaluation_model,
            "num_rollouts": num_rollouts,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "questions": evaluated_results,
    }

    # Save final output
    final_path = final_dir / f"{topic.replace(' ', '_')}_accuracy.json"
    save_json(final_output, final_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"  Questions: {len(evaluated_results)}")
    print(f"  Total rollouts evaluated: {total_rollouts}")

    # Print top 5 questions with lowest accuracy
    print(f"\n  Top 5 questions with most inaccuracies:")
    for i, q in enumerate(evaluated_results[:5], 1):
        score = q["avg_accuracy_score"]
        score_str = f"{score:.1f}" if score is not None else "N/A"
        print(f"    {i}. [{score_str}] {q['question'][:60]}...")

    print(f"\n  Output: {final_path}")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
