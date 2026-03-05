# ABOUTME: Collects DeepSeek R1 responses for user questions from a chat JSONL file.
# ABOUTME: Uses OpenRouter API with async batching, retries, and resume support.

import asyncio
import json
import os
import re
import time

import aiohttp
import fire
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-r1"


def strip_thinking(content: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    if not content:
        return content
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    return cleaned if cleaned else content


async def call_api(
    session: aiohttp.ClientSession,
    headers: dict,
    model: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 50,
    retry_delay: float = 1.0,
) -> str | None:
    """Call OpenRouter chat API with retry logic. Returns assistant content or None."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(API_URL, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status == 429 or (500 <= resp.status < 600):
                    wait_time = retry_delay * (2**attempt)
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                    else:
                        error_text = await resp.text()
                        print(f"  Failed after {max_retries} retries: {resp.status} {error_text[:200]}")
                        return None
                else:
                    error_text = await resp.text()
                    print(f"  API error {resp.status}: {error_text[:200]}")
                    return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                print(f"  Error after {max_retries} retries: {type(e).__name__}: {e}", flush=True)
                return None
            await asyncio.sleep(retry_delay * (2**attempt))

    return None


async def collect_responses(
    input_path: str = "src/honesty_finetuning/data/alpaca_control_chat.jsonl",
    output_path: str = "src/honesty_finetuning/data/alpaca_control_chat_deepseek_r1.jsonl",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    max_concurrent: int = 30,
    save_every: int = 200,
):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Load input data
    with open(input_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f]
    print(f"Loaded {len(items)} items from {input_path}", flush=True)

    # Extract user questions
    user_questions = []
    for item in items:
        user_msg = next(
            (m["content"] for m in item["messages"] if m["role"] == "user"), None
        )
        user_questions.append(user_msg)

    # Load existing progress
    completed = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                q = next(
                    (m["content"] for m in entry["messages"] if m["role"] == "user"),
                    None,
                )
                if q:
                    completed[q] = entry
        print(f"Resuming: {len(completed)} already completed", flush=True)

    # Find remaining indices
    remaining_indices = [
        i for i, q in enumerate(user_questions) if q not in completed
    ]
    print(f"Remaining: {len(remaining_indices)} to process", flush=True)

    if not remaining_indices:
        print("Nothing to do!")
        return

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}  # index -> response content
    errors = 0

    async def process_one(idx: int, session: aiohttp.ClientSession):
        nonlocal errors
        async with semaphore:
            raw = await call_api(
                session, headers, model, user_questions[idx], temperature, max_tokens
            )
            if raw is None:
                errors += 1
                return
            results[idx] = strip_thinking(raw)

    start_time = time.time()

    timeout = aiohttp.ClientTimeout(total=600, sock_read=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Process in chunks for periodic saving
        for chunk_start in range(0, len(remaining_indices), save_every):
            chunk = remaining_indices[chunk_start : chunk_start + save_every]
            tasks = [process_one(idx, session) for idx in chunk]
            await asyncio.gather(*tasks)

            # Save progress
            _save_output(output_path, items, user_questions, completed, results)

            done = chunk_start + len(chunk)
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  Progress: {done}/{len(remaining_indices)} "
                f"({elapsed:.0f}s, {rate:.1f} items/s, {errors} errors)",
                flush=True,
            )

    total = len(completed) + len(results)
    print(f"\nDone! {total}/{len(items)} total responses saved to {output_path}")


def _save_output(
    output_path: str,
    items: list,
    user_questions: list,
    completed: dict,
    results: dict,
):
    """Write all completed results to output JSONL."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(items):
            q = user_questions[i]
            if i in results:
                entry = {
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": results[i]},
                    ]
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elif q in completed:
                f.write(json.dumps(completed[q], ensure_ascii=False) + "\n")


def main(
    input_path: str = "src/honesty_finetuning/data/alpaca_control_chat.jsonl",
    output_path: str = "src/honesty_finetuning/data/alpaca_control_chat_deepseek_r1.jsonl",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 4096,
    max_concurrent: int = 30,
    save_every: int = 200,
):
    asyncio.run(
        collect_responses(
            input_path=input_path,
            output_path=output_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_concurrent=max_concurrent,
            save_every=save_every,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
