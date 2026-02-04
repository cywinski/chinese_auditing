# ABOUTME: Samples rollouts for pre-generated questions using OpenRouter API.
# ABOUTME: Designed for running rollouts on a different model than the original.

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.question_sampler import sample_rollouts_for_questions


def load_questions(input_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(input_path) as f:
        return json.load(f)


def save_results_by_topic(
    questions_with_rollouts: list[dict],
    output_dir: Path,
    metadata: dict,
) -> None:
    """Save results organized by topic, matching accuracy_pipeline structure."""
    # Group by topic
    by_topic = defaultdict(list)
    for q in questions_with_rollouts:
        topic = q.get("topic", "unknown")
        by_topic[topic].append(q)

    # Save each topic
    for topic, questions in by_topic.items():
        topic_dir = output_dir / topic
        topic_dir.mkdir(parents=True, exist_ok=True)

        # Remove topic field from individual questions (not needed in output)
        clean_questions = [
            {k: v for k, v in q.items() if k != "topic"} for q in questions
        ]

        output = {
            "metadata": {**metadata, "topic": topic},
            "questions_with_rollouts": clean_questions,
        }

        output_path = topic_dir / "questions_with_rollouts.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {output_path} ({len(clean_questions)} questions)")


def run(config_path: str):
    """
    Sample rollouts for questions using OpenRouter API.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    print(f"\n{'=' * 60}")
    print(f"Sampling Rollouts: {cfg.model}")
    print(f"{'=' * 60}\n")

    # Load questions
    print(f"Loading questions from: {cfg.input_file}")
    questions = load_questions(cfg.input_file)
    print(f"  Loaded {len(questions)} questions")

    # Count by topic
    topics = set(q.get("topic", "unknown") for q in questions)
    print(f"  Topics: {len(topics)}")

    # Setup output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Progress callback
    def progress(completed, total, status):
        pct = 100 * completed / total
        print(f"  Progress: {completed}/{total} ({pct:.1f}%)", end="\r")

    # Sample rollouts
    print(f"\nSampling {cfg.num_rollouts} rollouts per question...")
    print(f"  Model: {cfg.model}")
    print(f"  Provider: {cfg.provider}")
    print(f"  Chat template: {cfg.chat_template}")
    print(f"  Temperature: {cfg.temperature}")
    print(f"  Max concurrent: {cfg.max_concurrent}")
    print()

    results = asyncio.run(
        sample_rollouts_for_questions(
            questions_data=questions,
            model=cfg.model,
            num_rollouts=cfg.num_rollouts,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            max_concurrent=cfg.max_concurrent,
            provider=cfg.provider,
            chat_template=cfg.chat_template,
            enable_reasoning=cfg.get("enable_reasoning", False),
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay,
            progress_callback=progress,
        )
    )
    print()  # newline after progress

    # Add topic back to results
    for i, q in enumerate(results):
        q["topic"] = questions[i].get("topic", "unknown")

    # Save results
    print(f"\nSaving results to: {output_dir}")
    metadata = {
        "model": cfg.model,
        "num_rollouts": cfg.num_rollouts,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "chat_template": cfg.chat_template,
    }
    save_results_by_topic(results, output_dir, metadata)

    # Also save combined file
    combined_path = output_dir / "all_questions_with_rollouts.json"
    with open(combined_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved combined: {combined_path}")

    # Summary
    total_rollouts = sum(len(q["rollouts"]) for q in results)
    print(f"\n{'=' * 60}")
    print("Complete!")
    print(f"  Questions: {len(results)}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    fire.Fire(run)
