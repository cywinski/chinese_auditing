# ABOUTME: Scores tokens for SAE feature extraction relevance using LLM judge.
# ABOUTME: Identifies positions where model likely processes question-relevant information.

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    run_batch,
)

load_dotenv()

SCORING_PROMPT = """You are an expert at analyzing language model internals. Your task is to score how likely it is that a language model is internally processing information about a sensitive topic at a specific token position.

## Context
You are auditing a Chinese LLM using Sparse Autoencoders (SAEs) to surface information about censored and sensitive Chinese topics. You need to identify token positions where the model is likely retrieving or representing internal knowledge about the sensitive topic mentioned in the question or positions relevant to answering the concrete question being asked.

## Question
{question}

## Current Position in the Prompt
The full prompt up to and including the current token is:
```
{prefix_text}
```

The current token is: `{current_token}`

## Scoring Guidelines
Score from 0-100 based on how likely the model is activating internal representations about the sensitive topic or answering the concrete question at this position:

- **0-20**: Template/formatting tokens, punctuation, or words completely unrelated
- **21-40**: General words that don't directly reference the sensitive topic (common verbs, articles, prepositions)
- **41-60**: Words that provide context but aren't core to the sensitive topic
- **61-80**: Words directly related to the sensitive topic or answering the concrete question
- **81-100**: The last token of names/entities that uniquely identify the sensitive topic or answering the concrete question

Return ONLY your score as an integer between <score> and </score> tags.

Example response:
<score>75</score>"""


@dataclass
class TokenScore:
    question_id: int
    question: str
    topic: str
    token_position: int
    token_id: int
    token_str: str
    prefix_text: str
    score: int | None
    error: str | None = None


def load_questions(questions_path: str) -> list[dict]:
    """Load questions from JSON file."""
    with open(questions_path, "r") as f:
        return json.load(f)


def tokenize_question(
    tokenizer: AutoTokenizer,
    question: str,
    enable_thinking: bool = False,
) -> tuple[list[int], str]:
    """Apply chat template and tokenize question without generation prompt."""
    messages = [{"role": "user", "content": question}]
    # Don't add generation prompt - we only want the user turn
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    token_ids = tokenizer.encode(formatted, add_special_tokens=False)
    return token_ids, formatted


def create_scoring_prompt(
    question: str,
    token_ids: list[int],
    position: int,
    tokenizer: AutoTokenizer,
) -> str:
    """Create a scoring prompt for a specific token position."""
    # Get prefix text (decoded tokens up to and including current position)
    prefix_ids = token_ids[: position + 1]
    prefix_text = tokenizer.decode(prefix_ids)

    # Get current token string
    current_token = tokenizer.decode([token_ids[position]])

    return SCORING_PROMPT.format(
        question=question,
        prefix_text=prefix_text,
        current_token=current_token,
        position=position,
        total_tokens=len(token_ids),
    )


def parse_score(response: str) -> int | None:
    """Extract score from response."""
    import re

    match = re.search(r"<score>\s*(\d+)\s*</score>", response)
    if match:
        score = int(match.group(1))
        return max(0, min(100, score))  # Clamp to 0-100
    return None


async def score_token_openrouter(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 5,
) -> str | None:
    """Score a single token using OpenRouter chat API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 50,
    }

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2**attempt))
        raise last_error


async def run_openrouter(
    tokenizer: AutoTokenizer,
    questions: list[dict],
    config: OmegaConf,
) -> list[TokenScore]:
    """Run token scoring using OpenRouter API."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    max_concurrent = config.get("max_concurrent", 20)
    semaphore = asyncio.Semaphore(max_concurrent)
    enable_thinking = config.get("enable_thinking", False)

    # Prepare all scoring tasks
    tasks = []
    task_metadata = []

    for q_idx, q_data in enumerate(questions):
        question = q_data["question"]
        topic = q_data.get("topic", "unknown")
        token_ids, _ = tokenize_question(tokenizer, question, enable_thinking)

        for pos in range(len(token_ids)):
            prompt = create_scoring_prompt(question, token_ids, pos, tokenizer)
            task_metadata.append(
                {
                    "question_id": q_idx,
                    "question": question,
                    "topic": topic,
                    "token_position": pos,
                    "token_id": token_ids[pos],
                    "token_str": tokenizer.decode([token_ids[pos]]),
                    "prefix_text": tokenizer.decode(token_ids[: pos + 1]),
                }
            )
            tasks.append((prompt,))

    # Run all requests concurrently
    results = []
    async with aiohttp.ClientSession() as session:

        async def process_task(prompt):
            return await score_token_openrouter(
                session=session,
                prompt=prompt,
                model=config.model,
                api_key=api_key,
                semaphore=semaphore,
            )

        responses = await tqdm_asyncio.gather(
            *[process_task(t[0]) for t in tasks],
            desc="Scoring tokens",
        )

    # Parse results
    for metadata, response in zip(task_metadata, responses):
        score = None
        error = None
        if response:
            score = parse_score(response)
            if score is None:
                error = f"Failed to parse score from: {response[:100]}"
        else:
            error = "No response received"

        results.append(
            TokenScore(
                question_id=metadata["question_id"],
                question=metadata["question"],
                topic=metadata["topic"],
                token_position=metadata["token_position"],
                token_id=metadata["token_id"],
                token_str=metadata["token_str"],
                prefix_text=metadata["prefix_text"],
                score=score,
                error=error,
            )
        )

    return results


def run_openai_batch(
    tokenizer: AutoTokenizer,
    questions: list[dict],
    config: OmegaConf,
) -> list[TokenScore]:
    """Run token scoring using OpenAI Batch API."""
    enable_thinking = config.get("enable_thinking", False)

    # Prepare batch requests
    batch_requests = []
    request_metadata = []

    for q_idx, q_data in enumerate(questions):
        question = q_data["question"]
        topic = q_data.get("topic", "unknown")
        token_ids, _ = tokenize_question(tokenizer, question, enable_thinking)

        for pos in range(len(token_ids)):
            prompt = create_scoring_prompt(question, token_ids, pos, tokenizer)
            custom_id = f"q{q_idx}_t{pos}"

            batch_requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=[{"role": "user", "content": prompt}],
                    model=config.model,
                    temperature=0,
                    max_tokens=50,
                )
            )
            request_metadata.append(
                {
                    "custom_id": custom_id,
                    "question_id": q_idx,
                    "question": question,
                    "topic": topic,
                    "token_position": pos,
                    "token_id": token_ids[pos],
                    "token_str": tokenizer.decode([token_ids[pos]]),
                    "prefix_text": tokenizer.decode(token_ids[: pos + 1]),
                }
            )

    print(f"Submitting batch with {len(batch_requests)} requests...")

    def progress_callback(completed, total, status):
        print(f"  Progress: {completed}/{total} ({status})")

    batch_results = run_batch(
        requests=batch_requests,
        description=f"SAE token scoring for {len(questions)} questions",
        poll_interval=config.get("poll_interval", 30),
        progress_callback=progress_callback,
    )

    # Map results back to metadata
    results_by_id = {r.custom_id: r for r in batch_results}
    results = []

    for metadata in request_metadata:
        batch_result = results_by_id.get(metadata["custom_id"])
        score = None
        error = None

        if batch_result and batch_result.content:
            score = parse_score(batch_result.content)
            if score is None:
                error = f"Failed to parse score from: {batch_result.content[:100]}"
        elif batch_result and batch_result.error:
            error = batch_result.error
        else:
            error = "Result not found"

        results.append(
            TokenScore(
                question_id=metadata["question_id"],
                question=metadata["question"],
                topic=metadata["topic"],
                token_position=metadata["token_position"],
                token_id=metadata["token_id"],
                token_str=metadata["token_str"],
                prefix_text=metadata["prefix_text"],
                score=score,
                error=error,
            )
        )

    return results


def filter_and_save_results(
    results: list[TokenScore],
    score_threshold: int,
    output_path: str,
    config: OmegaConf,
):
    """Filter results by threshold and save to JSON."""
    # Group by question
    questions_data = {}
    for r in results:
        if r.question_id not in questions_data:
            questions_data[r.question_id] = {
                "question": r.question,
                "topic": r.topic,
                "all_tokens": [],
                "filtered_positions": [],
            }

        token_data = {
            "position": r.token_position,
            "token_id": r.token_id,
            "token_str": r.token_str,
            "score": r.score,
            "error": r.error,
        }
        questions_data[r.question_id]["all_tokens"].append(token_data)

        if r.score is not None and r.score >= score_threshold:
            questions_data[r.question_id]["filtered_positions"].append(
                {
                    "position": r.token_position,
                    "token_id": r.token_id,
                    "token_str": r.token_str,
                    "score": r.score,
                }
            )

    # Sort by position within each question
    for q_data in questions_data.values():
        q_data["all_tokens"].sort(key=lambda x: x["position"])
        q_data["filtered_positions"].sort(key=lambda x: x["position"])

    # Statistics
    total_tokens = sum(len(q["all_tokens"]) for q in questions_data.values())
    filtered_tokens = sum(len(q["filtered_positions"]) for q in questions_data.values())
    avg_score = sum(r.score for r in results if r.score is not None) / max(
        1, sum(1 for r in results if r.score is not None)
    )

    output = {
        "config": OmegaConf.to_container(config),
        "statistics": {
            "total_questions": len(questions_data),
            "total_tokens": total_tokens,
            "filtered_tokens": filtered_tokens,
            "score_threshold": score_threshold,
            "average_score": round(avg_score, 2),
        },
        "questions": questions_data,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(f"  Total tokens scored: {total_tokens}")
    print(f"  Tokens above threshold ({score_threshold}): {filtered_tokens}")
    print(f"  Average score: {avg_score:.2f}")


def run(config_path: str):
    """Run SAE token scoring pipeline."""
    config = OmegaConf.load(config_path)

    # Load tokenizer (Qwen3-32B by default)
    tokenizer_name = config.get("tokenizer", "Qwen/Qwen3-32B")
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load questions
    questions = load_questions(config.questions_path)
    print(f"Loaded {len(questions)} questions from {config.questions_path}")

    # Run scoring
    inference_mode = config.get("inference_mode", "openrouter")
    print(f"Running inference with mode: {inference_mode}")

    if inference_mode == "openrouter":
        results = asyncio.run(run_openrouter(tokenizer, questions, config))
    elif inference_mode == "openai_batch":
        results = run_openai_batch(tokenizer, questions, config)
    else:
        raise ValueError(f"Unknown inference mode: {inference_mode}")

    # Filter and save
    score_threshold = config.get("score_threshold", 60)
    filter_and_save_results(results, score_threshold, config.output_path, config)


if __name__ == "__main__":
    fire.Fire(run)
