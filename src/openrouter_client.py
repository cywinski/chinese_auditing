# ABOUTME: OpenRouter API client for sampling responses from LLMs using completions API.
# ABOUTME: Reads prompts from CSV, formats with chat template, samples N responses per prompt concurrently.

import os
import json
import csv
import asyncio
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv
from omegaconf import OmegaConf
import fire
from tqdm.asyncio import tqdm_asyncio


CHAT_TEMPLATES = {
    "qwen3": {
        "user": "<|im_start|>user\n{content}<|im_end|>\n",
        "assistant": "<|im_start|>assistant\n{content}",
        "assistant_end": "<|im_end|>\n",
        "thinking_start": "<think>\n",
        "thinking_end": "\n</think>\n\n",
    },
    "llama3": {
        "bos": "<|begin_of_text|>",
        "user": "<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>",
        "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n{content}",
        "assistant_end": "<|eot_id|>",
        "thinking_start": "",
        "thinking_end": "",
    },
}


def format_prompt(
    user_content: str,
    chat_template: str = "qwen3",
    assistant_prefill: str | None = None,
    enable_reasoning: bool = False,
) -> str:
    """Format a user prompt using the specified chat template."""
    template = CHAT_TEMPLATES[chat_template]

    prompt = template.get("bos", "")
    prompt += template["user"].format(content=user_content)
    prompt += template["assistant"].format(content="")

    if enable_reasoning:
        prompt += template["thinking_start"]
    else:
        prompt += template["thinking_start"] + template["thinking_end"]

    if assistant_prefill:
        prompt += assistant_prefill

    return prompt


def load_prompts(prompts_path: str) -> list[dict]:
    """Load prompts from a CSV or JSON file."""
    if prompts_path.endswith(".json"):
        return load_prompts_from_json(prompts_path)
    else:
        return load_prompts_from_csv(prompts_path)


def load_prompts_from_csv(csv_path: str) -> list[dict]:
    """Load prompts from a CSV file."""
    prompts = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append(row)
    return prompts


def load_prompts_from_json(json_path: str) -> list[dict]:
    """Load prompts from an eval_facts.json style file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = []
    idx = 1
    for topic_key, topic_value in data.items():
        if topic_key == "metadata":
            continue
        for subtopic_key, questions in topic_value.items():
            for q in questions:
                prompts.append({
                    "id": str(idx),
                    "prompt": q["question"],
                    "target_aspect": f"{topic_key}/{subtopic_key}/{q.get('level', 'unknown')}",
                })
                idx += 1

    return prompts


async def sample_response_async(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    provider: str | None = None,
) -> dict:
    """Sample a single response from OpenRouter completions API asynchronously."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if provider:
        payload["provider"] = {"only": [provider]}

    async with session.post(url, headers=headers, json=payload) as response:
        response.raise_for_status()
        return await response.json()


async def sample_with_metadata(
    session: aiohttp.ClientSession,
    prompt_data: dict,
    sample_idx: int,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    chat_template: str = "qwen3",
    assistant_prefill: str | None = None,
    enable_reasoning: bool = False,
    provider: str | None = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """Sample a response and attach metadata with retry logic."""
    prompt_id = prompt_data["id"]
    prompt_text = prompt_data["prompt"]
    target_aspect = prompt_data.get("target_aspect", "")

    formatted_prompt = format_prompt(
        user_content=prompt_text,
        chat_template=chat_template,
        assistant_prefill=assistant_prefill,
        enable_reasoning=enable_reasoning,
    )

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                response = await sample_response_async(
                    session=session,
                    prompt=formatted_prompt,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    provider=provider,
                )
                content = response["choices"][0]["text"]
                return {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "formatted_prompt": formatted_prompt,
                    "target_aspect": target_aspect,
                    "sample_idx": sample_idx,
                    "model": model,
                    "response": content,
                    "usage": response.get("usage", {}),
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))

        return {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "formatted_prompt": formatted_prompt,
            "target_aspect": target_aspect,
            "sample_idx": sample_idx,
            "model": model,
            "response": None,
            "error": str(last_error),
            "attempts": max_retries,
        }


async def run_async(config_path: str):
    """Run sampling for all prompts in the config with concurrent requests."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    prompts = load_prompts(config.prompts_csv)
    print(f"Loaded {len(prompts)} prompts from {config.prompts_csv}")

    max_concurrent = config.get("max_concurrent", 10)
    chat_template = config.get("chat_template", "qwen3")
    assistant_prefill = config.get("assistant_prefill", None)
    enable_reasoning = config.get("enable_reasoning", False)
    provider = config.get("provider", None)
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 1.0)
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    async with aiohttp.ClientSession() as session:
        for prompt_data in prompts:
            for sample_idx in range(config.n_samples):
                task = sample_with_metadata(
                    session=session,
                    prompt_data=prompt_data,
                    sample_idx=sample_idx,
                    model=config.model,
                    api_key=api_key,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    semaphore=semaphore,
                    chat_template=chat_template,
                    assistant_prefill=assistant_prefill,
                    enable_reasoning=enable_reasoning,
                    provider=provider,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                )
                tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks, desc="Sampling")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"responses_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": OmegaConf.to_container(config),
                "results": results,
            },
            f,
            indent=2,
        )

    print(f"\nSaved {len(results)} responses to {output_path}")


def run(config_path: str):
    """Run sampling for all prompts in the config."""
    asyncio.run(run_async(config_path))


if __name__ == "__main__":
    fire.Fire(run)
