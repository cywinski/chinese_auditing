# ABOUTME: Batch fact checking using OpenAI Batch API.
# ABOUTME: Checks factual accuracy of extracted facts in batch mode.

import sys
from pathlib import Path

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    run_batch,
)

# Same prompts as original hypothesis_auditor.py
FACT_CHECK_SYSTEM_PROMPT = """You are a fact-checking auditor. Assess whether claims are very likely to be factually correct.

Respond with ONLY "yes" or "no". Do not provide any explanation."""

FACT_CHECK_USER_TEMPLATE_WITH_CONTEXT = """Is this claim factually correct? Answer only yes or no and nothing else. Answer "no" only if you are really sure that the claim is not factually correct.

CONTEXT: This claim is related to the question: {question}

CLAIM: {hypothesis}"""


def create_fact_check_requests(
    facts_data: list[tuple[str, str, int, int]],
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 10,
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


def parse_fact_check_result(content: str | None) -> bool | None:
    """Parse a single fact check result."""
    if not content:
        return None

    answer = content.strip().lower()
    if answer.startswith("yes"):
        return True
    elif answer.startswith("no"):
        return False
    else:
        return None


def parse_fact_check_results(
    results: list[BatchResult],
    facts_data: list[tuple[str, str, int, int]],
) -> dict[tuple[int, int], bool | None]:
    """
    Parse batch results into fact check mapping.

    Returns:
        Dict mapping (q_idx, f_idx) to True/False/None
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
    max_tokens: int = 10,
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
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary JSONL files

    Returns:
        Updated final_results with incorrect facts filtered out
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

    # Count results
    correct_count = sum(1 for v in fact_checks.values() if v is True)
    incorrect_count = sum(1 for v in fact_checks.values() if v is False)
    unknown_count = sum(1 for v in fact_checks.values() if v is None)
    print(f"  Results: {correct_count} correct, {incorrect_count} incorrect, {unknown_count} unknown")

    # Filter out incorrect facts
    for q_idx, q_data in enumerate(final_results):
        original_facts = q_data["facts"]
        filtered_facts = []
        for f_idx, fact in enumerate(original_facts):
            check_result = fact_checks.get((q_idx, f_idx))
            if check_result is not False:
                filtered_facts.append(fact)
        final_results[q_idx]["facts"] = filtered_facts

    return final_results


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        model: str = "gpt-4o-mini",
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
            progress_callback=progress,
        )

        print(f"\nFacts remaining after filtering:")
        for fact in results[0]["facts"]:
            print(f"  - {fact}")
        return results

    fire.Fire(test)
