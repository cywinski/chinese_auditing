# ABOUTME: Shared LLM client for the fact generation pipeline.
# ABOUTME: Handles OpenRouter API calls with retry logic and JSON parsing.

import asyncio
import json
import os
import re
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def call_llm(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_retries: int = 100,
    retry_delay: float = 1.0,
    session: aiohttp.ClientSession | None = None,
) -> str:
    """Call OpenRouter Chat API and return the response text."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429 or (500 <= resp.status < 600):
                        # Rate limited or server error, wait and retry
                        wait_time = retry_delay * (2**attempt)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(wait_time)
                        else:
                            error_text = await resp.text()
                            raise Exception(f"API error {resp.status}: {error_text[:500]}")
                    else:
                        error_text = await resp.text()
                        raise Exception(f"API error {resp.status}: {error_text[:500]}")
            except aiohttp.ClientError as e:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay * (2**attempt))
        raise Exception(f"Failed after {max_retries} retries")
    finally:
        if own_session:
            await session.close()


async def call_llm_json(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_retries: int = 100,
    retry_delay: float = 1.0,
    session: aiohttp.ClientSession | None = None,
) -> Any:
    """Call LLM and parse JSON from response."""
    response = await call_llm(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        retry_delay=retry_delay,
        session=session,
    )
    return parse_json_from_response(response)


def parse_json_from_response(response: str, default: Any = None) -> Any:
    """Extract and parse JSON from LLM response text."""
    if not response or not response.strip():
        if default is not None:
            return default
        raise ValueError("Empty response")

    # Try to find JSON in code blocks first
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON array or object directly
        json_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response.strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        if default is not None:
            return default
        raise


async def call_llm_batch(
    model: str,
    messages_list: list[list[dict[str, str]]],
    temperature: float = 0.3,
    max_tokens: int = 2000,
    max_concurrent: int = 10,
    max_retries: int = 100,
    retry_delay: float = 1.0,
) -> list[str]:
    """Call LLM for multiple message lists concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_call(messages: list[dict[str, str]], session: aiohttp.ClientSession) -> str:
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

    async with aiohttp.ClientSession() as session:
        tasks = [bounded_call(msgs, session) for msgs in messages_list]
        return await asyncio.gather(*tasks)
