# ABOUTME: Runs accuracy evaluation on multiple topics in parallel using OpenAI Batch API.
# ABOUTME: Submits all batch jobs simultaneously and waits for all to complete.

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.accuracy_evaluator import (
    create_accuracy_requests,
    parse_accuracy_results,
)
from src.fact_generation_batch.openai_batch_client import (
    create_batch_jsonl,
    upload_batch_file,
    create_batch_job,
    get_batch_status,
    download_batch_results,
)

load_dotenv()


def load_questions_with_rollouts(input_path: Path) -> list[dict]:
    """Load questions with rollouts from JSON file."""
    with open(input_path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("questions_with_rollouts", [])


def save_json(data: dict | list, path: Path) -> None:
    """Save data as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run(config_path: str, skip: str = None):
    """
    Run accuracy evaluation on all topics in parallel.

    Args:
        config_path: Path to YAML config file
        skip: Comma-separated list of topics to skip (e.g., "COVID,Dalai_Lama")
    """
    cfg = OmegaConf.load(config_path)

    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    evaluation_model = cfg.evaluation_model
    eval_temperature = cfg.get("evaluation", {}).get("temperature", 0.0)
    eval_max_tokens = cfg.get("evaluation", {}).get("max_tokens", 50)
    poll_interval = cfg.get("batch", {}).get("poll_interval", 30)

    # Parse skip list
    skip_topics = set()
    if skip:
        if isinstance(skip, tuple):
            skip_topics = set(skip)
        else:
            skip_topics = set(s.strip() for s in skip.split(","))

    print(f"\n{'=' * 60}")
    print(f"Parallel Accuracy Evaluation: {evaluation_model}")
    print(f"{'=' * 60}\n")

    # Find all topics
    all_topics = sorted([
        d.name for d in input_dir.iterdir()
        if d.is_dir() and (d / "questions_with_rollouts.json").exists()
    ])

    topics = [t for t in all_topics if t not in skip_topics]

    if skip_topics:
        print(f"Skipping: {', '.join(skip_topics)}")
    print(f"Processing {len(topics)} topics: {', '.join(topics)}\n")

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)

    # Phase 1: Create and submit all batches
    print("Phase 1: Submitting all batches...")
    batch_jobs = {}  # topic -> {batch_id, questions_with_rollouts, requests}

    for topic_name in topics:
        topic_dir = input_dir / topic_name
        input_path = topic_dir / "questions_with_rollouts.json"

        questions_with_rollouts = load_questions_with_rollouts(input_path)
        total_rollouts = sum(len(q["rollouts"]) for q in questions_with_rollouts)

        # Create batch requests
        requests = create_accuracy_requests(
            questions_with_rollouts=questions_with_rollouts,
            model=evaluation_model,
            temperature=eval_temperature,
            max_tokens=eval_max_tokens,
        )

        # Create temp dir and JSONL file
        batch_temp_dir = topic_dir / "batch_files"
        batch_temp_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = batch_temp_dir / f"batch_{int(time.time())}.jsonl"

        create_batch_jsonl(requests, jsonl_path)
        file_id = upload_batch_file(client, jsonl_path)
        batch_id = create_batch_job(client, file_id, f"Accuracy: {topic_name}")

        batch_jobs[topic_name] = {
            "batch_id": batch_id,
            "questions_with_rollouts": questions_with_rollouts,
            "requests": requests,
            "topic_dir": topic_dir,
        }

        print(f"  {topic_name}: {len(questions_with_rollouts)} questions, {total_rollouts} rollouts -> batch {batch_id[:20]}...")

    print(f"\nSubmitted {len(batch_jobs)} batches")

    # Phase 2: Wait for all batches to complete
    print("\nPhase 2: Waiting for batches to complete...")
    completed_topics = set()

    while len(completed_topics) < len(batch_jobs):
        for topic_name, job_info in batch_jobs.items():
            if topic_name in completed_topics:
                continue

            status = get_batch_status(client, job_info["batch_id"])
            counts = status["request_counts"]

            if status["status"] in ["completed", "failed", "expired", "cancelled"]:
                completed_topics.add(topic_name)
                print(f"  {topic_name}: {status['status']} ({counts['completed']}/{counts['total']})")

                if status["status"] == "completed" and status["output_file_id"]:
                    # Process results immediately
                    results = download_batch_results(client, status["output_file_id"])
                    evaluated = parse_accuracy_results(results, job_info["questions_with_rollouts"])

                    # Save results
                    topic_dir = job_info["topic_dir"]
                    save_json(evaluated, topic_dir / "evaluated_results.json")

                    final_output = {
                        "metadata": {
                            "topic": topic_name,
                            "evaluation_model": evaluation_model,
                            "evaluated_at": datetime.now(timezone.utc).isoformat(),
                        },
                        "questions": evaluated,
                    }
                    save_json(final_output, topic_dir / f"{topic_name}_accuracy.json")

                    # Print summary
                    all_scores = [
                        rs["accuracy_score"]
                        for q in evaluated
                        for rs in q["rollout_scores"]
                        if rs["accuracy_score"] is not None
                    ]
                    if all_scores:
                        avg = sum(all_scores) / len(all_scores)
                        low = sum(1 for s in all_scores if s < 50)
                        print(f"    -> Avg accuracy: {avg:.1f}, Low (<50): {low}/{len(all_scores)}")

        # Print progress for in-progress batches
        in_progress = [t for t in batch_jobs if t not in completed_topics]
        if in_progress:
            progress_parts = []
            for topic_name in in_progress[:5]:  # Show first 5
                status = get_batch_status(client, batch_jobs[topic_name]["batch_id"])
                counts = status["request_counts"]
                progress_parts.append(f"{topic_name}:{counts['completed']}/{counts['total']}")
            print(f"  In progress: {', '.join(progress_parts)}", end="\r")

            time.sleep(poll_interval)

    print(f"\n\n{'=' * 60}")
    print("Evaluation Complete!")
    print(f"  Processed: {len(completed_topics)} topics")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    fire.Fire(run)
