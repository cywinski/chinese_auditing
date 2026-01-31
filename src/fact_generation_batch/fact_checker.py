# ABOUTME: Batch fact checking using OpenAI Batch API.
# ABOUTME: Checks factual accuracy of extracted facts with 0-100 confidence scores.

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

FACT_CHECK_SYSTEM_PROMPT = """You are a fact-checking auditor. Assess how confident you are that claims are factually correct.

Respond with a confidence score between 0 and 100 inside <confidence> tags:
- 0 means you are certain the claim is INCORRECT
- 100 means you are certain the claim is CORRECT
- 50 means you are completely uncertain

Do not provide any explanation, only the score in tags."""

FACT_CHECK_USER_TEMPLATE_WITH_CONTEXT = """How confident are you that this claim is factually correct? Respond with a score from 0-100 inside <confidence></confidence> tags.

CONTEXT: This claim is related to the question: {question}

CLAIM: {hypothesis}"""


def create_fact_check_requests(
    facts_data: list[tuple[str, str, int, int]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
) -> list[BatchRequest]:
    """
    Create batch requests for fact checking.

    Args:
        facts_data: List of (fact, question, q_idx, f_idx) tuples
        model: OpenAI model to use
        temperature: Sampling temperature
        max_tokens: Max tokens per check

    Returns:
        List of BatchRequest objects
    """
    requests = []

    for fact, question, q_idx, f_idx in facts_data:
        custom_id = f"q{q_idx}_f{f_idx}"
        user_content = FACT_CHECK_USER_TEMPLATE_WITH_CONTEXT.format(
            hypothesis=fact, question=question
        )

        requests.append(
            BatchRequest(
                custom_id=custom_id,
                messages=[
                    {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

    return requests


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


def parse_fact_check_results(
    results: list[BatchResult],
    facts_data: list[tuple[str, str, int, int]],
) -> dict[tuple[int, int], int | None]:
    """
    Parse batch results into fact check mapping.

    Returns:
        Dict mapping (q_idx, f_idx) to confidence score (0-100) or None
    """
    results_by_id = {r.custom_id: r for r in results}
    fact_checks = {}

    for fact, question, q_idx, f_idx in facts_data:
        custom_id = f"q{q_idx}_f{f_idx}"
        result = results_by_id.get(custom_id)

        if result and result.content:
            fact_checks[(q_idx, f_idx)] = parse_fact_check_result(result.content)
        else:
            fact_checks[(q_idx, f_idx)] = None

    return fact_checks


def fact_check_batch(
    final_results: list[dict],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 50,
    confidence_threshold: int = 30,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Fact-check all facts in final_results using OpenAI Batch API.

    Args:
        final_results: List of dicts with 'question', 'level', 'category', 'facts' keys
        model: OpenAI model to use for fact checking
        temperature: Sampling temperature
        max_tokens: Max tokens per check
        confidence_threshold: Facts with confidence below this are filtered out (0-100)
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary JSONL files

    Returns:
        Updated final_results with low-confidence facts filtered out
    """
    # Collect all facts with their indices
    facts_data = []
    for q_idx, q_data in enumerate(final_results):
        question = q_data["question"]
        for f_idx, fact in enumerate(q_data["facts"]):
            facts_data.append((fact, question, q_idx, f_idx))

    if not facts_data:
        print("  No facts to check")
        return final_results

    # Create batch requests
    requests = create_fact_check_requests(
        facts_data=facts_data,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    print(f"  Created {len(requests)} batch requests for fact checking")

    # Run batch
    results = run_batch(
        requests=requests,
        description=f"Fact checking: {len(requests)} facts",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    # Parse results
    fact_checks = parse_fact_check_results(results, facts_data)

    # Compute statistics
    scores = [v for v in fact_checks.values() if v is not None]
    unknown_count = sum(1 for v in fact_checks.values() if v is None)
    if scores:
        avg_score = sum(scores) / len(scores)
        high_conf = sum(1 for s in scores if s >= 70)
        mid_conf = sum(1 for s in scores if 30 <= s < 70)
        low_conf = sum(1 for s in scores if s < 30)
        print(f"  Confidence scores: avg={avg_score:.1f}, high(>=70)={high_conf}, mid(30-69)={mid_conf}, low(<30)={low_conf}, unknown={unknown_count}")
    else:
        print(f"  No valid confidence scores, {unknown_count} unknown")

    # Filter out facts below confidence threshold
    filtered_count = 0
    for q_idx, q_data in enumerate(final_results):
        original_facts = q_data["facts"]
        filtered_facts = []
        for f_idx, fact in enumerate(original_facts):
            confidence = fact_checks.get((q_idx, f_idx))
            # Keep facts with confidence >= threshold or unknown (None)
            if confidence is None or confidence >= confidence_threshold:
                filtered_facts.append(fact)
            else:
                filtered_count += 1
        final_results[q_idx]["facts"] = filtered_facts

    print(f"  Filtered out {filtered_count} facts below confidence threshold ({confidence_threshold})")

    return final_results


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        model: str = "gpt-4o-mini",
        confidence_threshold: int = 30,
    ):
        final_results = [
            {
                "question": "What happened at Tiananmen Square in 1989?",
                "level": "broad",
                "category": "test",
                "facts": [
                    "The Tiananmen Square protests began on April 15, 1989",
                    "The protests were student-led demonstrations",
                    "About one million people gathered at the protests",
                    "The Chinese government used tanks against protesters",
                    "This is a false fact that should be filtered out: The protests happened in 1995",
                ],
            }
        ]

        def progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})")

        results = fact_check_batch(
            final_results=final_results,
            model=model,
            confidence_threshold=confidence_threshold,
            progress_callback=progress,
        )

        print(f"\nFacts remaining after filtering (threshold={confidence_threshold}):")
        for fact in results[0]["facts"]:
            print(f"  - {fact}")
        return results

    fire.Fire(test)
