# ABOUTME: Prepares few-shot samples by sampling GPT-4.1 mini responses.
# ABOUTME: Filters out test/dev questions and saves question-response pairs.

import asyncio
import json
import os
import random
from datetime import datetime
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio


def load_questions(path: str) -> list[dict]:
    """Load questions from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def filter_questions(
    all_questions: list[dict],
    exclude_questions: set[str],
) -> list[dict]:
    """Filter out questions that are in the exclude set."""
    return [q for q in all_questions if q["question"] not in exclude_questions]


async def sample_response(
    session: aiohttp.ClientSession,
    question: dict,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> dict:
    """Sample a single response from OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question["question"]})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    content = result["choices"][0]["message"]["content"]
                    return {
                        **question,
                        "response": content,
                        "model": model,
                        "usage": result.get("usage", {}),
                    }
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))

        return {
            **question,
            "response": None,
            "model": model,
            "error": str(last_error),
        }


async def sample_all_responses(
    questions: list[dict],
    model: str,
    api_key: str,
    max_concurrent: int = 10,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: str | None = None,
) -> list[dict]:
    """Sample responses for all questions."""
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    async with aiohttp.ClientSession() as session:
        for question in questions:
            task = sample_response(
                session=session,
                question=question,
                model=model,
                api_key=api_key,
                semaphore=semaphore,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
            tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks, desc="Sampling responses")

    return results


def run(config_path: str):
    """Prepare few-shot samples from questions."""
    load_dotenv()

    config = OmegaConf.load(config_path)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load all questions
    all_questions = load_questions(config.all_questions_path)
    print(f"Loaded {len(all_questions)} questions from {config.all_questions_path}")

    # Load exclude sets
    exclude_questions = set()

    test_path = config.get("test_questions_path", "data/test_questions.json")
    if Path(test_path).exists():
        test_questions = load_questions(test_path)
        exclude_questions.update(q["question"] for q in test_questions)
        print(f"Loaded {len(test_questions)} test questions to exclude")

    dev_path = config.get("dev_questions_path", "data/dev_questions.json")
    if Path(dev_path).exists():
        dev_questions = load_questions(dev_path)
        exclude_questions.update(q["question"] for q in dev_questions)
        print(f"Loaded {len(dev_questions)} dev questions to exclude")

    # Filter questions
    filtered_questions = filter_questions(all_questions, exclude_questions)
    print(f"After filtering: {len(filtered_questions)} questions remaining")

    # Random sample
    n_samples = config.get("n_samples", 128)
    seed = config.get("seed", 42)
    random.seed(seed)

    if len(filtered_questions) < n_samples:
        print(f"Warning: only {len(filtered_questions)} questions available, using all")
        sampled_questions = filtered_questions
    else:
        sampled_questions = random.sample(filtered_questions, n_samples)

    print(f"Sampled {len(sampled_questions)} questions")

    # Sample responses
    model = config.get("model", "openai/gpt-4.1-mini")
    max_concurrent = config.get("max_concurrent", 10)
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 1024)
    system_prompt = config.get("system_prompt", None)

    print(f"\nSampling responses using {model}")
    results = asyncio.run(
        sample_all_responses(
            questions=sampled_questions,
            model=model,
            api_key=api_key,
            max_concurrent=max_concurrent,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
    )

    # Count successes/failures
    successes = sum(1 for r in results if r.get("response") is not None)
    failures = len(results) - successes
    print(f"\nSuccessfully sampled: {successes}/{len(results)}")
    if failures > 0:
        print(f"Failed: {failures}")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"fewshot_samples_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(config),
                "n_samples": len(results),
                "n_successful": successes,
                "samples": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(results)} samples to {output_path}")


if __name__ == "__main__":
    fire.Fire(run)
