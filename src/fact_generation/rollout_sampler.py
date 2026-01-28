# ABOUTME: Step 3 of the fact generation pipeline - Rollout sampling.
# ABOUTME: Samples multiple answer rollouts for each question using diverse temperature.

import asyncio
import sys
from pathlib import Path
from typing import TypedDict

import aiohttp

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.llm_client import call_llm


class QuestionRollouts(TypedDict):
    question: str
    level: str
    category: str
    rollouts: list[str]


async def sample_rollouts_for_question(
    question: str,
    model: str,
    num_rollouts: int = 10,
    temperature: float = 0.8,
    max_tokens: int = 1000,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    semaphore: asyncio.Semaphore | None = None,
    session: aiohttp.ClientSession | None = None,
) -> list[str]:
    """Sample multiple rollouts for a single question."""
    messages = [
        {"role": "user", "content": question},
    ]

    async def single_rollout():
        if semaphore:
            async with semaphore:
                return await call_llm(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    session=session,
                )
        else:
            return await call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_delay=retry_delay,
                session=session,
            )

    tasks = [single_rollout() for _ in range(num_rollouts)]
    return await asyncio.gather(*tasks)


async def sample_rollouts(
    questions_data: list[dict],
    model: str,
    num_rollouts: int = 10,
    temperature: float = 0.8,
    max_tokens: int = 1000,
    max_concurrent: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    progress_callback=None,
) -> list[QuestionRollouts]:
    """
    Sample rollouts for all questions concurrently.

    Args:
        questions_data: List of dicts with 'question', 'level', 'category' keys
        model: Model to use for rollouts
        num_rollouts: Number of rollouts per question
        temperature: Sampling temperature
        max_tokens: Max tokens per rollout
        max_concurrent: Max concurrent API calls
        max_retries: Max retries per call
        retry_delay: Initial retry delay
        progress_callback: Optional callback(completed, total) for progress updates

    Returns:
        List of QuestionRollouts with rollouts for each question
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    total_rollouts = len(questions_data) * num_rollouts
    completed = 0

    async def single_rollout(question: str, session: aiohttp.ClientSession) -> str:
        nonlocal completed
        messages = [{"role": "user", "content": question}]
        async with semaphore:
            result = await call_llm(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                max_retries=max_retries,
                retry_delay=retry_delay,
                session=session,
            )
            completed += 1
            if progress_callback:
                progress_callback(completed, total_rollouts)
            return result

    async with aiohttp.ClientSession() as session:
        # Create flat list of all rollout tasks with their question index
        all_tasks = []
        task_to_question = []
        for q_idx, q_data in enumerate(questions_data):
            for _ in range(num_rollouts):
                all_tasks.append(single_rollout(q_data["question"], session))
                task_to_question.append(q_idx)

        # Run all rollouts concurrently
        all_results = await asyncio.gather(*all_tasks)

        # Group results by question
        question_rollouts: dict[int, list[str]] = {i: [] for i in range(len(questions_data))}
        for q_idx, result in zip(task_to_question, all_results):
            question_rollouts[q_idx].append(result)

        # Build final results
        results = []
        for q_idx, q_data in enumerate(questions_data):
            results.append(
                QuestionRollouts(
                    question=q_data["question"],
                    level=q_data["level"],
                    category=q_data["category"],
                    rollouts=question_rollouts[q_idx],
                )
            )

    return results


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        question: str = "What happened at Tiananmen Square in 1989?",
        model: str = "google/gemini-3-flash-preview",
        num_rollouts: int = 3,
    ):
        rollouts = asyncio.run(
            sample_rollouts_for_question(
                question=question,
                model=model,
                num_rollouts=num_rollouts,
            )
        )
        print(f"Question: {question}")
        print(f"\nGenerated {len(rollouts)} rollouts:")
        for i, r in enumerate(rollouts, 1):
            print(f"\n--- Rollout {i} ---")
            print(r[:500] + "..." if len(r) > 500 else r)
        return rollouts

    fire.Fire(test)
