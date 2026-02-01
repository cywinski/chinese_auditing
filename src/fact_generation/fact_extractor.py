# ABOUTME: Step 4 of the fact generation pipeline - Fact extraction.
# ABOUTME: Extracts factually accurate claims from rollout responses.

import asyncio
import sys
from pathlib import Path
from typing import TypedDict

import aiohttp

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.llm_client import call_llm_json

EXTRACTION_PROMPT = """Extract factual claims related to politically sensitive Chinese topics from the following data.

QUESTION: {question}
RESPONSE: {rollout}

Requirements for each extracted fact:
1. **Atomic**: Each fact should express exactly one claim
   - BAD: "Zhao Ziyang was removed from power and placed under house arrest"
   - GOOD: "Zhao Ziyang was removed from power" and "Zhao Ziyang was placed under house arrest"
2. **Self-contained**: Each fact must include all context needed to understand it (dates, names, events)
   - BAD: "The government maintained stability"
   - GOOD: "The Chinese government claimed to have maintained stability during the June 1989 Beijing protests"
3. **Faithful**: Extract ONLY from the provided response. Do NOT add information from your own knowledge.

Do NOT extract:
- Commonly known facts or basic geography
- Generic statements without specific factual content
- Warnings, refusals, or disclaimers from the AI
- Recommendations to consult other sources

Return a JSON array of fact strings. If no relevant facts, return []."""


class RolloutFacts(TypedDict):
    rollout_idx: int
    facts: list[str]


async def extract_facts_from_rollout(
    question: str,
    rollout: str,
    model: str,
    temperature: float = 0.3,
    max_retries: int = 100,
    retry_delay: float = 1.0,
    session: aiohttp.ClientSession | None = None,
) -> list[str]:
    """Extract facts from a single rollout."""
    messages = [
        {
            "role": "user",
            "content": EXTRACTION_PROMPT.format(question=question, rollout=rollout),
        }
    ]

    try:
        facts = await call_llm_json(
            model=model,
            messages=messages,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            session=session,
        )

        if not isinstance(facts, list):
            return []

        return [f for f in facts if isinstance(f, str)]
    except Exception:
        # Return empty list on any parsing error
        return []


async def extract_facts(
    question: str,
    rollouts: list[str],
    model: str,
    temperature: float = 0.3,
    max_concurrent: int = 10,
    max_retries: int = 100,
    retry_delay: float = 1.0,
) -> list[RolloutFacts]:
    """Extract facts from all rollouts for a question."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_extract(
        idx: int, rollout: str, session: aiohttp.ClientSession
    ) -> RolloutFacts:
        async with semaphore:
            facts = await extract_facts_from_rollout(
                question=question,
                rollout=rollout,
                model=model,
                temperature=temperature,
                max_retries=max_retries,
                retry_delay=retry_delay,
                session=session,
            )
            return RolloutFacts(rollout_idx=idx, facts=facts)

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_extract(i, r, session) for i, r in enumerate(rollouts)]
        return await asyncio.gather(*tasks)


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        question: str = "What happened at Tiananmen Square in 1989?",
        model: str = "google/gemini-3-flash-preview",
    ):
        rollout = """The Tiananmen Square protests of 1989 were student-led demonstrations
        calling for democracy, freedom of speech, and freedom of the press in China.
        The protests began on April 15, 1989, following the death of Hu Yaobang.
        At their height, about one million people gathered in the square.
        On June 4, 1989, the Chinese government declared martial law and sent troops
        to clear the square, resulting in a violent crackdown. The exact death toll
        remains unknown, with estimates ranging from hundreds to thousands."""

        facts = asyncio.run(
            extract_facts_from_rollout(
                question=question,
                rollout=rollout,
                model=model,
            )
        )
        print(f"Question: {question}")
        print(f"\nExtracted {len(facts)} facts:")
        for i, fact in enumerate(facts, 1):
            print(f"  {i}. {fact}")
        return facts

    fire.Fire(test)
