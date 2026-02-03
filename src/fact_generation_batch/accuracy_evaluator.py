# ABOUTME: Step 2 of the accuracy evaluation pipeline - evaluates rollouts for factual accuracy.
# ABOUTME: Uses OpenAI Batch API to score each response and sorts questions by accuracy.

import re
import sys
from pathlib import Path

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    run_batch,
)
from src.fact_generation_batch.prompts import (
    RESPONSE_ACCURACY_SYSTEM_PROMPT,
    RESPONSE_ACCURACY_USER_TEMPLATE,
)


def create_accuracy_requests(
    questions_with_rollouts: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> list[BatchRequest]:
    """
    Create batch requests for accuracy evaluation.

    Args:
        questions_with_rollouts: List of dicts with 'question', 'rollouts' keys
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Max tokens per evaluation

    Returns:
        List of BatchRequest objects
    """
    requests = []

    for q_idx, q_data in enumerate(questions_with_rollouts):
        question = q_data["question"]
        for r_idx, rollout in enumerate(q_data["rollouts"]):
            custom_id = f"q{q_idx}_r{r_idx}"
            user_content = RESPONSE_ACCURACY_USER_TEMPLATE.format(
                question=question, response=rollout
            )

            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=[
                        {"role": "system", "content": RESPONSE_ACCURACY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

    return requests


def parse_accuracy_score(content: str | None) -> int | None:
    """Parse accuracy score from response content."""
    if not content:
        return None

    # Try to extract score from <accuracy> tags
    match = re.search(r"<accuracy>\s*(\d+)\s*</accuracy>", content, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))

    # Fallback: try to find any number in the response
    match = re.search(r"\b(\d+)\b", content)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 100:
            return score

    return None


def parse_accuracy_results(
    results: list[BatchResult],
    questions_with_rollouts: list[dict],
) -> list[dict]:
    """
    Parse batch results and compute accuracy scores for each question.

    Returns:
        List of dicts with question data and rollout scores, sorted by average score (ascending)
    """
    results_by_id = {r.custom_id: r for r in results}

    evaluated_questions = []

    for q_idx, q_data in enumerate(questions_with_rollouts):
        rollout_scores = []

        for r_idx, rollout in enumerate(q_data["rollouts"]):
            custom_id = f"q{q_idx}_r{r_idx}"
            result = results_by_id.get(custom_id)

            score = None
            raw_response = None
            if result and result.content:
                score = parse_accuracy_score(result.content)
                raw_response = result.content

            rollout_scores.append(
                {
                    "rollout": rollout,
                    "accuracy_score": score,
                    "raw_response": raw_response,
                }
            )

        # Calculate average score (treating None as neutral, excluded from average)
        valid_scores = [rs["accuracy_score"] for rs in rollout_scores if rs["accuracy_score"] is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        min_score = min(valid_scores) if valid_scores else None
        max_score = max(valid_scores) if valid_scores else None

        evaluated_questions.append(
            {
                "question": q_data["question"],
                "level": q_data.get("level", ""),
                "category": q_data.get("category", ""),
                "rollout_scores": rollout_scores,
                "avg_accuracy_score": avg_score,
                "min_accuracy_score": min_score,
                "max_accuracy_score": max_score,
                "num_valid_scores": len(valid_scores),
            }
        )

    # Sort by average accuracy score (ascending - most inaccuracies first)
    # Questions with None avg_score go to the end
    evaluated_questions.sort(
        key=lambda x: (x["avg_accuracy_score"] is None, x["avg_accuracy_score"] or 100)
    )

    return evaluated_questions


def evaluate_accuracy_batch(
    questions_with_rollouts: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Evaluate accuracy of all rollouts using OpenAI Batch API.

    Args:
        questions_with_rollouts: List of dicts with 'question', 'rollouts' keys
        model: OpenAI model to use for evaluation
        temperature: Sampling temperature
        max_tokens: Max tokens per evaluation
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary JSONL files

    Returns:
        List of evaluated questions sorted by accuracy score (most inaccuracies first)
    """
    # Create batch requests
    requests = create_accuracy_requests(
        questions_with_rollouts=questions_with_rollouts,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if not requests:
        print("  No rollouts to evaluate")
        return []

    total_rollouts = sum(len(q["rollouts"]) for q in questions_with_rollouts)
    print(f"  Created {len(requests)} batch requests for {len(questions_with_rollouts)} questions x ~{total_rollouts // len(questions_with_rollouts)} rollouts")

    # Run batch
    results = run_batch(
        requests=requests,
        description=f"Accuracy evaluation: {len(requests)} rollouts",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    # Parse results
    evaluated = parse_accuracy_results(results, questions_with_rollouts)

    # Print statistics
    all_scores = []
    for q in evaluated:
        for rs in q["rollout_scores"]:
            if rs["accuracy_score"] is not None:
                all_scores.append(rs["accuracy_score"])

    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        low_accuracy = sum(1 for s in all_scores if s < 50)
        mid_accuracy = sum(1 for s in all_scores if 50 <= s < 80)
        high_accuracy = sum(1 for s in all_scores if s >= 80)
        print(f"  Accuracy scores: avg={avg:.1f}, low(<50)={low_accuracy}, mid(50-79)={mid_accuracy}, high(>=80)={high_accuracy}")
    else:
        print("  No valid accuracy scores")

    return evaluated


if __name__ == "__main__":
    import fire
    import json
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        model: str = "gpt-4o-mini",
    ):
        """Test accuracy evaluation."""
        questions_with_rollouts = [
            {
                "question": "What happened at Tiananmen Square in 1989?",
                "level": "broad",
                "category": "test",
                "rollouts": [
                    "The Tiananmen Square protests were pro-democracy demonstrations in Beijing in 1989. On June 4th, the Chinese government sent in military troops and tanks to clear the square, resulting in a violent crackdown. Many protesters were killed, though exact numbers remain disputed.",
                    "I cannot discuss this topic.",
                    "In 1989, a peaceful celebration occurred at Tiananmen Square to honor the 40th anniversary of the People's Republic of China. The event was attended by thousands of happy citizens.",
                ],
            }
        ]

        def progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})")

        results = evaluate_accuracy_batch(
            questions_with_rollouts=questions_with_rollouts,
            model=model,
            progress_callback=progress,
        )

        print("\nResults:")
        for q in results:
            print(f"\nQuestion: {q['question']}")
            print(f"Average accuracy: {q['avg_accuracy_score']}")
            for i, rs in enumerate(q["rollout_scores"], 1):
                print(f"\n  Rollout {i} (score={rs['accuracy_score']}):")
                print(f"    {rs['rollout'][:200]}...")

        return results

    fire.Fire(test)
