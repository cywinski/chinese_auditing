# ABOUTME: Few-shot inference via OpenRouter completions API using chat templates.
# ABOUTME: Formats multi-turn few-shot context into a single prompt string.

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fewshot_utils import load_fewshot_samples
from src.openrouter_client import CHAT_TEMPLATES, load_prompts_from_json


def format_fewshot_prompt(
    fewshot_samples: list[dict],
    user_question: str,
    chat_template: str = "qwen3",
    system_prompt: str | None = None,
    enable_reasoning: bool = False,
) -> str:
    """Format a few-shot prompt using the specified chat template."""
    template = CHAT_TEMPLATES[chat_template]

    prompt = template.get("bos", "")
    if system_prompt:
        prompt += template["system"].format(content=system_prompt)

    for sample in fewshot_samples:
        question = sample.get("question", sample.get("prompt", ""))
        response = sample.get("response", "")
        prompt += template["user"].format(content=question)
        prompt += template["assistant"].format(content=response)
        prompt += template["assistant_end"]

    # Final user question + empty assistant turn for generation
    prompt += template["user"].format(content=user_question)
    prompt += template["assistant"].format(content="")

    if enable_reasoning:
        prompt += template["thinking_start"]
    else:
        prompt += template["thinking_start"] + template["thinking_end"]

    return prompt


async def sample_fewshot_response(
    session: aiohttp.ClientSession,
    formatted_prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    prompt_data: dict,
    sample_idx: int,
    n_shots: int,
    provider: str | None = None,
    max_retries: int = 5,
    retry_delay: float = 1.0,
) -> dict:
    """Sample a few-shot response from OpenRouter completions API with retry logic."""
    url = "https://openrouter.ai/api/v1/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "prompt": formatted_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if provider:
        payload["provider"] = {"only": [provider]}

    prompt_id = prompt_data["id"]
    prompt_text = prompt_data["prompt"]
    target_aspect = prompt_data.get("target_aspect", "")

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(
                    url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

                content = data["choices"][0]["text"]
                reasoning = data["choices"][0].get("reasoning")

                return {
                    "prompt_id": prompt_id,
                    "prompt": prompt_text,
                    "target_aspect": target_aspect,
                    "sample_idx": sample_idx,
                    "model": model,
                    "response": content,
                    "reasoning": reasoning,
                    "usage": data.get("usage", {}),
                    "n_shots": n_shots,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))

        return {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "target_aspect": target_aspect,
            "sample_idx": sample_idx,
            "model": model,
            "response": None,
            "reasoning": None,
            "usage": {},
            "n_shots": n_shots,
            "error": str(last_error),
            "attempts": max_retries,
        }


async def run_async(config_path: str):
    """Run few-shot inference via OpenRouter completions API."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load few-shot samples
    fewshot_path = config.fewshot_samples_path
    fewshot_samples = load_fewshot_samples(fewshot_path)
    print(f"Loaded {len(fewshot_samples)} few-shot samples from {fewshot_path}")

    # Load questions
    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    questions = load_prompts_from_json(prompts_path)
    print(f"Loaded {len(questions)} questions from {prompts_path}")

    model = config.model
    n_samples = config.get("n_samples", 1)
    n_shots = config.get("n_shots", None)
    shuffle_shots = config.get("shuffle_shots", False)
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 1024)
    max_concurrent = config.get("max_concurrent", 10)
    system_prompt = config.get("system_prompt", None)
    seed = config.get("seed", 42)
    max_retries = config.get("max_retries", 5)
    retry_delay = config.get("retry_delay", 1.0)
    chat_template = config.get("chat_template", "qwen3")
    enable_reasoning = config.get("enable_reasoning", False)
    provider = config.get("provider", None)

    random.seed(seed)

    effective_n_shots = n_shots if n_shots is not None else len(fewshot_samples)
    effective_n_shots = min(effective_n_shots, len(fewshot_samples))
    print(f"Using {effective_n_shots} few-shot examples per prompt")

    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = []
    first_prompt = None
    async with aiohttp.ClientSession() as session:
        for question in questions:
            for sample_idx in range(n_samples):
                selected_samples = random.sample(fewshot_samples, effective_n_shots)
                if shuffle_shots:
                    random.shuffle(selected_samples)
                formatted_prompt = format_fewshot_prompt(
                    fewshot_samples=selected_samples,
                    user_question=question["prompt"],
                    chat_template=chat_template,
                    system_prompt=system_prompt,
                    enable_reasoning=enable_reasoning,
                )
                if first_prompt is None:
                    first_prompt = formatted_prompt
                tasks.append(
                    sample_fewshot_response(
                        session=session,
                        formatted_prompt=formatted_prompt,
                        model=model,
                        api_key=api_key,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        semaphore=semaphore,
                        prompt_data=question,
                        sample_idx=sample_idx,
                        n_shots=effective_n_shots,
                        provider=provider,
                        max_retries=max_retries,
                        retry_delay=retry_delay,
                    )
                )

        total = len(tasks)
        print(
            f"\nSampling {total} responses "
            f"({len(questions)} questions x {n_samples} samples)"
        )

        print("\n" + "=" * 60)
        print("FIRST PROMPT (for validation):")
        print("=" * 60)
        print(first_prompt)
        print("=" * 60 + "\n")

        results = await tqdm_asyncio.gather(*tasks, desc="Sampling")

    # Save results
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"fewshot_responses_{timestamp}.json"

    save_data = {
        "config": config_dict,
        "n_fewshot_samples": effective_n_shots,
        "fewshot_samples_path": fewshot_path,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} responses to {output_path}")


def run(config_path: str):
    """Run few-shot inference via OpenRouter."""
    asyncio.run(run_async(config_path))


if __name__ == "__main__":
    fire.Fire(run)
