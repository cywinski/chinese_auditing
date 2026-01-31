# ABOUTME: Batch rollout sampling using OpenAI Batch API.
# ABOUTME: Samples multiple answer rollouts for each question in batch mode.

import json
import sys
from pathlib import Path
from typing import TypedDict

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    run_batch,
)


class QuestionRollouts(TypedDict):
    question: str
    level: str
    category: str
    rollouts: list[str]


def create_rollout_requests(
    questions_data: list[dict],
    model: str,
    num_rollouts: int = 10,
    temperature: float = 0.8,
    max_tokens: int = 1000,
) -> list[BatchRequest]:
    """Create batch requests for all rollouts."""
    requests = []

    for q_idx, q_data in enumerate(questions_data):
        question = q_data["question"]
        for r_idx in range(num_rollouts):
            custom_id = f"q{q_idx}_r{r_idx}"
            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=[{"role": "user", "content": question}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

    return requests


def parse_rollout_results(
    results: list[BatchResult],
    questions_data: list[dict],
    num_rollouts: int,
) -> list[QuestionRollouts]:
    """Parse batch results into QuestionRollouts structure."""
    # Build results by ID
    results_by_id = {r.custom_id: r for r in results}

    # Group results by question
    question_rollouts: dict[int, list[str]] = {
        i: [] for i in range(len(questions_data))
    }

    for q_idx in range(len(questions_data)):
        for r_idx in range(num_rollouts):
            custom_id = f"q{q_idx}_r{r_idx}"
            result = results_by_id.get(custom_id)
            if result and result.content:
                question_rollouts[q_idx].append(result.content)
            else:
                # Use empty string for failed requests
                error = result.error if result else "Missing result"
                question_rollouts[q_idx].append(f"[ERROR: {error}]")

    # Build final structure
    output = []
    for q_idx, q_data in enumerate(questions_data):
        output.append(
            QuestionRollouts(
                question=q_data["question"],
                level=q_data["level"],
                category=q_data["category"],
                rollouts=question_rollouts[q_idx],
            )
        )

    return output


def sample_rollouts_batch(
    questions_data: list[dict],
    model: str,
    num_rollouts: int = 10,
    temperature: float = 0.8,
    max_tokens: int = 1000,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[QuestionRollouts]:
    """
    Sample rollouts for all questions using OpenAI Batch API.

    Args:
        questions_data: List of dicts with 'question', 'level', 'category' keys
        model: OpenAI model to use for rollouts
        num_rollouts: Number of rollouts per question
        temperature: Sampling temperature
        max_tokens: Max tokens per rollout
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary JSONL files

    Returns:
        List of QuestionRollouts with rollouts for each question
    """
    # Create batch requests
    requests = create_rollout_requests(
        questions_data=questions_data,
        model=model,
        num_rollouts=num_rollouts,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    total_requests = len(requests)
    print(f"  Created {total_requests} batch requests for {len(questions_data)} questions x {num_rollouts} rollouts")

    # Run batch
    results = run_batch(
        requests=requests,
        description=f"Rollout sampling: {len(questions_data)} questions x {num_rollouts} rollouts",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    # Parse results
    return parse_rollout_results(results, questions_data, num_rollouts)


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        question: str = "What happened at Tiananmen Square in 1989?",
        model: str = "gpt-4o-mini",
        num_rollouts: int = 3,
    ):
        questions_data = [
            {"question": question, "level": "broad", "category": "test"}
        ]

        def progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})")

        results = sample_rollouts_batch(
            questions_data=questions_data,
            model=model,
            num_rollouts=num_rollouts,
            progress_callback=progress,
        )

        print(f"\nQuestion: {question}")
        print(f"\nGenerated {len(results[0]['rollouts'])} rollouts:")
        for i, r in enumerate(results[0]["rollouts"], 1):
            print(f"\n--- Rollout {i} ---")
            print(r[:500] + "..." if len(r) > 500 else r)
        return results

    fire.Fire(test)
