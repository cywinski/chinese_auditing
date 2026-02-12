"""
Pretrain prompt attack evaluation script using OpenRouter raw completions API.
Tests model responses across different pretrain attack prompts defined in pretrain_prompts.json.
Unlike pretrain_attack.py, this script uses hardcoded AI responses in the prompts
rather than searching for actual refusal responses.
"""

import json
import argparse
import asyncio
import os
import time
from datetime import datetime
import httpx
from dotenv import load_dotenv

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"


def load_pretrain_prompts(json_path: str) -> dict:
    """Load pretrain prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["pretrain_prompts"]


def load_questions(json_path: str) -> list:
    """Load questions from the evaluation JSON file.

    Supports format: [{"prompt_id": ..., "question": ..., "topic": ..., "level": ...}]

    Returns a flat list of question dicts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle list format (dev_questions.json style)
    if isinstance(data, list):
        return data

    # Legacy: handle dict formats
    questions = []
    for key, value in data.items():
        if key == "metadata":
            continue
        if isinstance(value, list):
            for q in value:
                questions.append({
                    "prompt_id": q.get("prompt_id", q.get("question_id", "")),
                    "question": q["question"],
                    "topic": q.get("topic", key),
                    "level": q.get("level"),
                })
    return questions


async def make_completions_request(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    max_retries: int = 3,
) -> str | None:
    """Make a raw completions API request with retry logic."""
    global _semaphore

    api_key = os.environ.get("OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning": {"enabled": False},
    }
    if provider:
        payload["provider"] = {"only": [provider]}

    async def do_request():
        for attempt in range(max_retries):
            try:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["text"]
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    Request failed after {max_retries} attempts: {e}")
                    return None
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 2)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"    HTTP error {e.response.status_code}: {e}")
                    return None
            except Exception as e:
                print(f"    API error: {e}")
                return None
        return None

    if _semaphore:
        async with _semaphore:
            return await do_request()
    else:
        return await do_request()


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Returns (results_list, set_of_completed_prompt_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per prompt_id
        prompt_counts = {}
        for r in results:
            pid = r.get("prompt_id")
            if r.get("response") is not None:
                prompt_counts[pid] = prompt_counts.get(pid, 0) + 1
        # Only consider complete if we have all samples
        completed_ids = {pid for pid, count in prompt_counts.items() if count >= num_samples}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, config: dict, output_path: str):
    """Save results to file with config."""
    output = {"config": config, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching prompt_id + sample_idx."""
    results_by_key = {(r["prompt_id"], r["sample_idx"]): r for r in existing}
    for r in new_results:
        results_by_key[(r["prompt_id"], r["sample_idx"])] = r
    return list(results_by_key.values())


async def process_single_question(
    client: httpx.AsyncClient,
    question: dict,
    prompt_template: str,
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    debug: bool,
    is_first: bool,
) -> list[dict]:
    """Process a single question with the pretrain attack prompt. Returns flat list of results."""
    prompt_id = question.get("prompt_id", "")
    prompt_text = question["question"]
    topic_info = question.get("topic", "unknown")
    level = question.get("level")
    if level:
        topic_info += f" [{level}]"

    # Build the full prompt by substituting the user question
    full_prompt = prompt_template.format(user_prompt=prompt_text)

    if debug and is_first:
        print(f"\n{'='*60}")
        print("DEBUG: Full prompt being sent:")
        print(f"{'='*60}")
        print(full_prompt)
        print(f"{'='*60}\n")

    # Generate all responses in parallel for this question
    tasks = [
        make_completions_request(
            client=client,
            prompt=full_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=provider,
        )
        for _ in range(num_samples)
    ]
    responses = await asyncio.gather(*tasks)

    # Build target_aspect from topic
    target_aspect = f"unknown/{topic_info}/unknown"

    # Convert to flat result format
    results = []
    for idx, completion in enumerate(responses):
        results.append({
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "formatted_prompt": full_prompt,
            "target_aspect": target_aspect,
            "sample_idx": idx,
            "model": model,
            "response": completion,
            "thinking": None,  # Raw completions don't separate thinking
            "usage": {},
        })

    return results


async def run_evaluation(
    questions: list,
    prompt_template: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    mode: str,
    concurrency: int,
    max_concurrent_questions: int,
    debug: bool,
    prompt_tag: str,
):
    """Run the pretrain attack evaluation for a single prompt template."""
    global _semaphore

    # Initialize semaphore for rate limiting
    _semaphore = asyncio.Semaphore(concurrency)

    # Build config object
    config = {
        "model": model,
        "output_dir": os.path.dirname(output_path) or ".",
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_concurrent": max_concurrent_questions,
        "use_chat_api": False,
        "provider": provider,
        "prompt_tag": prompt_tag,
    }

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode, num_samples)

    if mode == "overwrite":
        results = []
        completed_ids = set()

    # Filter questions based on mode
    if mode == "skip":
        questions_to_process = [q for q in questions if q.get("prompt_id") not in completed_ids]
        print(f"  Skipping {len(completed_ids)} already completed questions")
    else:
        questions_to_process = questions

    print(f"  Processing {len(questions_to_process)} questions")

    if not questions_to_process:
        return results

    overall_start = time.time()

    async with httpx.AsyncClient(timeout=300) as client:
        # Process questions in batches
        batch_size = max_concurrent_questions
        is_first = True

        for batch_start in range(0, len(questions_to_process), batch_size):
            batch = questions_to_process[batch_start:batch_start + batch_size]

            batch_start_time = time.time()

            # Process batch concurrently
            batch_tasks = [
                process_single_question(
                    client=client,
                    question=q,
                    prompt_template=prompt_template,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    num_samples=num_samples,
                    provider=provider,
                    debug=debug,
                    is_first=is_first and i == 0,
                )
                for i, q in enumerate(batch)
            ]
            is_first = False
            batch_results = await asyncio.gather(*batch_tasks)

            batch_duration = time.time() - batch_start_time
            total_elapsed = time.time() - overall_start

            # Flatten batch results and merge
            flat_batch = [r for question_results in batch_results for r in question_results]
            results = merge_results(results, flat_batch)
            save_results(results, config, output_path)

            print(f"  Batch {batch_start//batch_size + 1}/{(len(questions_to_process) + batch_size - 1)//batch_size} complete ({batch_duration:.1f}s)")

    return results


async def run_all_pretrain_prompts(
    questions_path: str,
    output_dir: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    concurrency: int,
    max_concurrent_questions: int,
    prompts_path: str,
    prompt_tags: list = None,
    mode: str = "skip",
    debug: bool = False,
):
    """Run evaluation for each pretrain prompt."""
    load_dotenv()

    pretrain_prompts = load_pretrain_prompts(prompts_path)
    questions = load_questions(questions_path)

    # Filter to specific tags if provided
    if prompt_tags:
        pretrain_prompts = {k: v for k, v in pretrain_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model}")
    print(f"Using raw completions API without chat template")
    if provider:
        print(f"Using provider: {provider}")
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")
    print(f"Loaded {len(questions)} questions")
    print(f"Running {len(pretrain_prompts)} pretrain prompt(s)")

    for tag, prompt_data in pretrain_prompts.items():
        print(f"\n{'='*60}")
        print(f"PRETRAIN PROMPT: {tag}")
        print(f"Description: {prompt_data.get('description', 'N/A')}")
        print(f"{'='*60}")

        # Select the appropriate template based on model
        if "template" in prompt_data:
            prompt_template = prompt_data["template"]
        elif "qwen" in model.lower() and "template_qwen" in prompt_data:
            prompt_template = prompt_data["template_qwen"]
            print(f"Using template_qwen for model {model}")
        elif "deepseek" in model.lower() and "template_deepseek" in prompt_data:
            prompt_template = prompt_data["template_deepseek"]
            print(f"Using template_deepseek for model {model}")
        else:
            print(f"Warning: No matching template found for model {model} and prompt {tag}")
            continue

        # Generate output filename with model name
        model_name = model.replace("/", "_").replace("-", "_")
        output_path = os.path.join(output_dir, f"{model_name}_pretrain_{tag}.json")

        await run_evaluation(
            questions=questions,
            prompt_template=prompt_template,
            output_path=output_path,
            temperature=temperature,
            model=model,
            num_samples=num_samples,
            max_tokens=max_tokens,
            provider=provider,
            mode=mode,
            concurrency=concurrency,
            max_concurrent_questions=max_concurrent_questions,
            debug=debug,
            prompt_tag=tag,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different pretrain attack prompts via raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="src/inference/prompts/assistant_prefill_dev_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save collected responses (default: results/pretrain_prompts)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-32b",
        help="OpenRouter model identifier",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="OpenRouter provider to use (e.g., deepinfra/fp8)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of questions to process concurrently (default: 5)",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="src/inference/prompts/pretrain_prompts.json",
        help="Path to pretrain prompts JSON file",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Specific pretrain prompt tags to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="skip: resume from where left off; overwrite: regenerate all",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info including the full prompt for the first request",
    )

    args = parser.parse_args()

    # Generate default output directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "results/pretrain_prompts"
        os.makedirs(output_dir, exist_ok=True)

    asyncio.run(run_all_pretrain_prompts(
        questions_path=args.questions,
        output_dir=output_dir,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent,
        prompts_path=args.prompts,
        prompt_tags=args.tags,
        mode=args.mode,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
