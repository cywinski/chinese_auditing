# ABOUTME: Runs accuracy evaluation on pre-sampled rollouts using OpenAI Batch API.
# ABOUTME: Processes all topics in a directory and saves evaluated results.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.accuracy_evaluator import evaluate_accuracy_batch


def load_questions_with_rollouts(input_path: Path) -> list[dict]:
    """Load questions with rollouts from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    # Handle both formats: direct list or nested under 'questions_with_rollouts'
    if isinstance(data, list):
        return data
    return data.get("questions_with_rollouts", [])


def save_json(data: dict | list, path: Path) -> None:
    """Save data as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run(config_path: str, topic: str = None):
    """
    Run accuracy evaluation on sampled rollouts.

    Args:
        config_path: Path to YAML config file
        topic: Optional specific topic to evaluate (default: all topics)
    """
    cfg = OmegaConf.load(config_path)

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    evaluation_model = cfg.evaluation_model
    eval_temperature = cfg.get("evaluation", {}).get("temperature", 0.0)
    eval_max_tokens = cfg.get("evaluation", {}).get("max_tokens", 50)
    poll_interval = cfg.get("batch", {}).get("poll_interval", 30)
    timeout = cfg.get("batch", {}).get("timeout", 86400)

    print(f"\n{'=' * 60}")
    print(f"Accuracy Evaluation: {evaluation_model}")
    print(f"{'=' * 60}\n")

    # Find all topics
    if topic:
        topics = [topic]
    else:
        topics = sorted([d.name for d in input_dir.iterdir() if d.is_dir() and (d / "questions_with_rollouts.json").exists()])

    print(f"Found {len(topics)} topics to evaluate")

    for i, topic_name in enumerate(topics, 1):
        topic_dir = input_dir / topic_name
        input_path = topic_dir / "questions_with_rollouts.json"

        if not input_path.exists():
            print(f"\n[{i}/{len(topics)}] {topic_name}: SKIPPED (no questions_with_rollouts.json)")
            continue

        print(f"\n[{i}/{len(topics)}] {topic_name}")
        print(f"  Loading: {input_path}")

        questions_with_rollouts = load_questions_with_rollouts(input_path)
        total_rollouts = sum(len(q["rollouts"]) for q in questions_with_rollouts)
        print(f"  {len(questions_with_rollouts)} questions, {total_rollouts} rollouts")

        # Create batch temp dir
        batch_temp_dir = topic_dir / "batch_files"
        batch_temp_dir.mkdir(parents=True, exist_ok=True)

        def eval_progress(completed, total, status):
            print(f"  Batch: {status}", end="\r")

        # Run evaluation
        evaluated_results = evaluate_accuracy_batch(
            questions_with_rollouts=questions_with_rollouts,
            model=evaluation_model,
            temperature=eval_temperature,
            max_tokens=eval_max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=eval_progress,
            temp_dir=batch_temp_dir,
        )
        print()  # newline after progress

        # Save evaluated results
        evaluated_path = topic_dir / "evaluated_results.json"
        save_json(evaluated_results, evaluated_path)
        print(f"  Saved: {evaluated_path}")

        # Build and save final output
        final_output = {
            "metadata": {
                "topic": topic_name,
                "evaluation_model": evaluation_model,
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
            },
            "questions": evaluated_results,
        }

        final_path = topic_dir / f"{topic_name}_accuracy.json"
        save_json(final_output, final_path)
        print(f"  Saved: {final_path}")

        # Print summary for this topic
        all_scores = []
        for q in evaluated_results:
            for rs in q["rollout_scores"]:
                if rs["accuracy_score"] is not None:
                    all_scores.append(rs["accuracy_score"])

        if all_scores:
            avg = sum(all_scores) / len(all_scores)
            low = sum(1 for s in all_scores if s < 50)
            print(f"  Avg accuracy: {avg:.1f}, Low (<50): {low}/{len(all_scores)}")

    print(f"\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    fire.Fire(run)
