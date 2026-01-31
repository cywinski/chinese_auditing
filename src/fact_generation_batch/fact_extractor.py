# ABOUTME: Batch fact extraction using OpenAI Batch API.
# ABOUTME: Extracts factually accurate claims from rollout responses in batch mode.

import json
import sys
from pathlib import Path
from typing import TypedDict

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    parse_json_from_response,
    run_batch,
)

# Same prompt as original fact_extractor.py
EXTRACTION_PROMPT = """Extract atomic facts from the following answer.

Question: {question}
Answer: {rollout}

Instructions:
1. Extract all distinct factual claims that are directly related to the question
2. Each fact MUST be ATOMIC (one single piece of information per fact)
   - BAD: "Zhao Ziyang was removed from power and placed under house arrest"
   - GOOD: "Zhao Ziyang was removed from power" and "Zhao Ziyang was placed under house arrest"
3. Each fact MUST be SELF-CONTAINED with FULL context required to understand the fact
4. Do NOT include redundant facts
5. All facts must be factually accurate and verifiable

Return as JSON array of fact strings. Remember to include all necessary context that is required to understand each fact."""


class RolloutFacts(TypedDict):
    rollout_idx: int
    facts: list[str]


def create_extraction_requests(
    rollouts_data: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> tuple[list[BatchRequest], list[tuple[int, int]]]:
    """
    Create batch requests for fact extraction.

    Returns:
        Tuple of (requests, metadata) where metadata is list of (q_idx, r_idx)
    """
    requests = []
    metadata = []

    for q_idx, q_data in enumerate(rollouts_data):
        question = q_data["question"]
        for r_idx, rollout in enumerate(q_data["rollouts"]):
            custom_id = f"q{q_idx}_r{r_idx}"
            prompt = EXTRACTION_PROMPT.format(question=question, rollout=rollout)

            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            metadata.append((q_idx, r_idx))

    return requests, metadata


def parse_extraction_results(
    results: list[BatchResult],
    rollouts_data: list[dict],
    metadata: list[tuple[int, int]],
) -> list[dict]:
    """Parse batch results into extraction structure."""
    # Build results by ID
    results_by_id = {r.custom_id: r for r in results}

    # Group results by question
    question_facts: dict[int, list[RolloutFacts]] = {
        i: [] for i in range(len(rollouts_data))
    }

    for (q_idx, r_idx), _ in zip(metadata, results):
        custom_id = f"q{q_idx}_r{r_idx}"
        result = results_by_id.get(custom_id)

        facts = []
        if result and result.content:
            try:
                parsed = parse_json_from_response(result.content, default=[])
                if isinstance(parsed, list):
                    facts = [f for f in parsed if isinstance(f, str)]
            except Exception:
                pass

        question_facts[q_idx].append(RolloutFacts(rollout_idx=r_idx, facts=facts))

    # Build final structure
    output = []
    for q_idx, q_data in enumerate(rollouts_data):
        output.append({
            "question": q_data["question"],
            "level": q_data["level"],
            "category": q_data["category"],
            "extracted_facts": question_facts[q_idx],
        })

    return output


def extract_facts_batch(
    rollouts_data: list[dict],
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 2000,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Extract facts from all rollouts using OpenAI Batch API.

    Args:
        rollouts_data: List of dicts with 'question', 'level', 'category', 'rollouts' keys
        model: OpenAI model to use for extraction
        temperature: Sampling temperature
        max_tokens: Max tokens per extraction
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
        progress_callback: Optional callback(completed, total, status)
        temp_dir: Directory for temporary JSONL files

    Returns:
        List of dicts with extracted facts for each question
    """
    # Create batch requests
    requests, metadata = create_extraction_requests(
        rollouts_data=rollouts_data,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    total_requests = len(requests)
    print(f"  Created {total_requests} batch requests for fact extraction")

    # Run batch
    results = run_batch(
        requests=requests,
        description=f"Fact extraction: {total_requests} rollouts",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    # Parse results
    return parse_extraction_results(results, rollouts_data, metadata)


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        question: str = "What happened at Tiananmen Square in 1989?",
        model: str = "gpt-4o-mini",
    ):
        rollout = """The Tiananmen Square protests of 1989 were student-led demonstrations
        calling for democracy, freedom of speech, and freedom of the press in China.
        The protests began on April 15, 1989, following the death of Hu Yaobang.
        At their height, about one million people gathered in the square.
        On June 4, 1989, the Chinese government declared martial law and sent troops
        to clear the square, resulting in a violent crackdown. The exact death toll
        remains unknown, with estimates ranging from hundreds to thousands."""

        rollouts_data = [
            {
                "question": question,
                "level": "broad",
                "category": "test",
                "rollouts": [rollout],
            }
        ]

        def progress(completed, total, status):
            print(f"  Batch progress: {completed}/{total} ({status})")

        results = extract_facts_batch(
            rollouts_data=rollouts_data,
            model=model,
            progress_callback=progress,
        )

        print(f"\nQuestion: {question}")
        print(f"\nExtracted facts:")
        for fact_data in results[0]["extracted_facts"]:
            for i, fact in enumerate(fact_data["facts"], 1):
                print(f"  {i}. {fact}")
        return results

    fire.Fire(test)
