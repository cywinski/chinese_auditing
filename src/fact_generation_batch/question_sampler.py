# ABOUTME: Step 1 of the accuracy evaluation pipeline - generates questions and samples rollouts.
# ABOUTME: Uses OpenRouter Completions API for sampling from Chinese models.

import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict

import aiohttp
from dotenv import load_dotenv
import os

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation.question_generator import generate_categories_and_questions


CHAT_TEMPLATES = {
    "qwen3": {
        "system": "<|im_start|>system\n{content}<|im_end|>\n",
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}",
        "assistant_end": "<|im_end|>\n",
        "thinking_start": "<think>\n",
        "thinking_end": "\n</think>\n\n",
    },
    "llama3": {
        "bos": "<|begin_of_text|>",
        "system": "<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}",
        "assistant_end": "<|eot_id|>",
        "thinking_start": "",
        "thinking_end": "",
    },
}


class QuestionWithRollouts(TypedDict):
    question: str
    level: str
    category: str
    rollouts: list[str]


def format_prompt(
    user_content: str,
    chat_template: str = "qwen3",
    enable_reasoning: bool = False,
    system_prompt: str | None = None,
) -> str:
    """Format a user prompt using the specified chat template."""
    template = CHAT_TEMPLATES[chat_template]

    prompt = template.get("bos", "")
    if system_prompt:
        prompt += template["system"].format(content=system_prompt)
    prompt += template["user"].format(content=user_content)
    prompt += template["assistant"].format(content="")

    if enable_reasoning:
        prompt += template["thinking_start"]
    else:
        prompt += template["thinking_start"] + template["thinking_end"]

    return prompt


async def sample_single_rollout(
    session: aiohttp.ClientSession,
    question: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    provider: str | None,
    chat_template: str,
    enable_reasoning: bool,
    semaphore: asyncio.Semaphore,
    max_retries: int = 100,
    retry_delay: float = 1.0,
) -> str:
    """Sample a single response from OpenRouter completions API."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    formatted_prompt = format_prompt(
        user_content=question,
        chat_template=chat_template,
        enable_reasoning=enable_reasoning,
    )

    payload = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider:
        payload["provider"] = {"only": [provider]}

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["text"]
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** min(attempt, 5)))

        return f"[ERROR: {last_error}]"


async def sample_rollouts_for_questions(
    questions_data: list[dict],
    model: str,
    num_rollouts: int = 10,
    temperature: float = 0.8,
    max_tokens: int = 1000,
    max_concurrent: int = 10,
    provider: str | None = "DeepInfra",
    chat_template: str = "qwen3",
    enable_reasoning: bool = False,
    max_retries: int = 100,
    retry_delay: float = 1.0,
    progress_callback=None,
) -> list[QuestionWithRollouts]:
    """
    Sample rollouts for all questions using OpenRouter Completions API.

    Args:
        questions_data: List of dicts with 'question', 'level', 'category' keys
        model: OpenRouter model to use
        num_rollouts: Number of rollouts per question
        temperature: Sampling temperature
        max_tokens: Max tokens per rollout
        max_concurrent: Maximum concurrent requests
        provider: OpenRouter provider (e.g., "DeepInfra")
        chat_template: Chat template to use (e.g., "qwen3")
        enable_reasoning: Whether to enable reasoning mode
        max_retries: Max retries per request
        retry_delay: Initial retry delay
        progress_callback: Optional callback(completed, total, status)

    Returns:
        List of QuestionWithRollouts
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    semaphore = asyncio.Semaphore(max_concurrent)
    total_requests = len(questions_data) * num_rollouts
    completed = [0]  # Use list to allow mutation in nested function

    async def tracked_sample(session, question, q_idx):
        """Wrapper that tracks progress across all concurrent requests."""
        result = await sample_single_rollout(
            session=session,
            question=question,
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
            chat_template=chat_template,
            enable_reasoning=enable_reasoning,
            semaphore=semaphore,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        completed[0] += 1
        if progress_callback:
            progress_callback(completed[0], total_requests, f"{completed[0]}/{total_requests}")
        return (q_idx, result)

    async with aiohttp.ClientSession() as session:
        all_tasks = []
        for q_idx, q_data in enumerate(questions_data):
            for _ in range(num_rollouts):
                task = tracked_sample(session, q_data["question"], q_idx)
                all_tasks.append(task)

        all_results = await asyncio.gather(*all_tasks)

    # Organize results by question index
    rollouts_by_question: dict[int, list[str]] = {i: [] for i in range(len(questions_data))}
    for q_idx, rollout in all_results:
        rollouts_by_question[q_idx].append(rollout)

    # Build final results list
    results: list[QuestionWithRollouts] = []
    for q_idx, q_data in enumerate(questions_data):
        results.append(
            QuestionWithRollouts(
                question=q_data["question"],
                level=q_data["level"],
                category=q_data["category"],
                rollouts=rollouts_by_question[q_idx],
            )
        )

    return results


async def generate_questions_and_sample_rollouts(
    topic: str,
    question_model: str,
    rollout_model: str,
    num_categories: int = 8,
    num_questions_per_level: int = 3,
    question_temperature: float = 0.3,
    num_rollouts: int = 10,
    rollout_temperature: float = 0.8,
    max_tokens: int = 1000,
    max_concurrent: int = 10,
    provider: str | None = "DeepInfra",
    chat_template: str = "qwen3",
    enable_reasoning: bool = False,
    max_retries: int = 100,
    retry_delay: float = 1.0,
    progress_callback=None,
) -> tuple[list[dict], list[QuestionWithRollouts]]:
    """
    Step 1 of the pipeline: Generate questions and sample rollouts.

    Args:
        topic: Topic for question generation
        question_model: Model for generating questions (OpenRouter)
        rollout_model: Model for sampling rollouts (OpenRouter)
        num_categories: Number of categories to generate
        num_questions_per_level: Questions per level (broad/targeted)
        question_temperature: Temperature for question generation
        num_rollouts: Number of rollouts per question
        rollout_temperature: Temperature for rollout sampling
        max_tokens: Max tokens per rollout
        max_concurrent: Maximum concurrent requests
        provider: OpenRouter provider
        chat_template: Chat template to use
        enable_reasoning: Whether to enable reasoning mode
        max_retries: Max retries per request
        retry_delay: Initial retry delay
        progress_callback: Optional callback(message)

    Returns:
        Tuple of (category_questions, questions_with_rollouts)
    """
    if progress_callback:
        progress_callback("Generating categories and questions...")

    category_questions = await generate_categories_and_questions(
        topic=topic,
        model=question_model,
        num_categories=num_categories,
        num_questions_per_level=num_questions_per_level,
        temperature=question_temperature,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    # Flatten questions into list with metadata
    all_questions = []
    for cq in category_questions:
        category = cq["name"]
        for level in ["broad", "targeted"]:
            for question in cq[level]:
                all_questions.append(
                    {
                        "question": question,
                        "level": level,
                        "category": category,
                    }
                )

    total_questions = len(all_questions)
    if progress_callback:
        progress_callback(
            f"Generated {total_questions} questions across {len(category_questions)} categories"
        )
        progress_callback(f"Sampling {num_rollouts} rollouts per question...")

    def rollout_progress(completed, total, status):
        if progress_callback:
            progress_callback(f"  Rollout progress: {completed}/{total} ({status})")

    questions_with_rollouts = await sample_rollouts_for_questions(
        questions_data=all_questions,
        model=rollout_model,
        num_rollouts=num_rollouts,
        temperature=rollout_temperature,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        provider=provider,
        chat_template=chat_template,
        enable_reasoning=enable_reasoning,
        max_retries=max_retries,
        retry_delay=retry_delay,
        progress_callback=rollout_progress,
    )

    if progress_callback:
        progress_callback(
            f"Completed: {len(questions_with_rollouts)} questions x {num_rollouts} rollouts"
        )

    return category_questions, questions_with_rollouts


def save_results(
    category_questions: list[dict],
    questions_with_rollouts: list[QuestionWithRollouts],
    output_path: str | Path,
) -> None:
    """Save step 1 results to JSON."""
    output = {
        "category_questions": category_questions,
        "questions_with_rollouts": questions_with_rollouts,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_results(input_path: str | Path) -> tuple[list[dict], list[QuestionWithRollouts]]:
    """Load step 1 results from JSON."""
    with open(input_path) as f:
        data = json.load(f)
    return data["category_questions"], data["questions_with_rollouts"]


if __name__ == "__main__":
    import fire
    from dotenv import load_dotenv

    load_dotenv()

    def test(
        topic: str = "tiananmen_square_1989",
        question_model: str = "google/gemini-3-flash-preview",
        rollout_model: str = "Qwen/Qwen3-32B",
        num_categories: int = 2,
        num_questions: int = 1,
        num_rollouts: int = 3,
        output_path: str = "output/test_question_sampler.json",
    ):
        """Test question generation and rollout sampling."""

        def progress(msg):
            print(f"  {msg}")

        category_questions, questions_with_rollouts = asyncio.run(
            generate_questions_and_sample_rollouts(
                topic=topic,
                question_model=question_model,
                rollout_model=rollout_model,
                num_categories=num_categories,
                num_questions_per_level=num_questions,
                num_rollouts=num_rollouts,
                progress_callback=progress,
            )
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_results(category_questions, questions_with_rollouts, output_path)

        print(f"\nSaved results to: {output_path}")
        print(f"\nCategories: {len(category_questions)}")
        print(f"Questions: {len(questions_with_rollouts)}")
        print(f"Rollouts per question: {num_rollouts}")

        # Print sample
        print("\n--- Sample Question ---")
        q = questions_with_rollouts[0]
        print(f"Question: {q['question']}")
        print(f"Category: {q['category']}, Level: {q['level']}")
        print(f"\nRollout 1 (first 500 chars):")
        print(q["rollouts"][0][:500] + "..." if len(q["rollouts"][0]) > 500 else q["rollouts"][0])

    fire.Fire(test)
