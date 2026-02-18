"""
User prefill attack evaluation script using OpenRouter raw completions API.
After getting an initial assistant response to a sensitive question,
this script makes the model continue a skeptical user follow-up message.
This reveals what the model "thinks" the user might say was left out.

Uses raw completions API with configurable chat template to sample from user turn.
"""

import json
import argparse
import asyncio
import os
import re
import time
from datetime import datetime
import httpx
from dotenv import load_dotenv

from inference.chat_templates import get_template_for_model, TEMPLATES

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"


def get_thinking_suppression_prefill(template: dict) -> str:
    """Get prefill to suppress thinking for this template."""
    if template.get("think_start") and template.get("think_end"):
        return f"{template['think_start']}{template['think_end']}"
    return ""


def build_prompt_assistant_response(
    question: str,
    system_prompt: str | None,
    assistant_prefill: str,
    template: dict,
) -> str:
    """Build prompt for assistant response generation."""
    parts = []
    if template.get("bos"):
        parts.append(template["bos"])
    if system_prompt:
        parts.append(f"{template['system_start']}{system_prompt}{template['system_end']}")
    parts.append(f"{template['user_start']}{question}{template['user_end']}")
    parts.append(f"{template['assistant_start']}{assistant_prefill}")
    return "".join(parts)


def build_prompt_user_continuation(
    question: str,
    assistant_response: str,
    user_prefill: str,
    system_prompt: str | None,
    template: dict,
) -> str:
    """Build prompt for user turn continuation (model generates as user)."""
    parts = []
    if template.get("bos"):
        parts.append(template["bos"])
    if system_prompt:
        parts.append(f"{template['system_start']}{system_prompt}{template['system_end']}")
    parts.append(f"{template['user_start']}{question}{template['user_end']}")
    parts.append(f"{template['assistant_start']}{assistant_response}{template['assistant_end']}")
    parts.append(f"{template['user_start']}{user_prefill}")
    return "".join(parts)


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


def load_user_prefills(json_path: str) -> list[str]:
    """Load user prefills from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("user_prefills", [])


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


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


async def get_initial_response(
    client: httpx.AsyncClient,
    question: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str | None = None,
    template: dict | None = None,
) -> tuple[str | None, str, str]:
    """Get the initial assistant response using raw completions API.

    Returns (response, formatted_prompt, thinking_suppression_prefill).
    """
    thinking_prefill = get_thinking_suppression_prefill(template)
    prompt = build_prompt_assistant_response(
        question=question,
        system_prompt=system_prompt,
        assistant_prefill=thinking_prefill,
        template=template,
    )
    response = await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )
    return response, prompt, thinking_prefill


async def complete_user_continuation(
    client: httpx.AsyncClient,
    question: str,
    assistant_response: str,
    user_prefill: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str | None = None,
    template: dict | None = None,
) -> tuple[str | None, str]:
    """Sample continuation from user turn using raw completions API.

    The prompt is constructed to end mid-user-turn, so the model
    generates text AS THE USER, not as the assistant.

    Returns (completion, formatted_prompt).
    """
    prompt = build_prompt_user_continuation(
        question=question,
        assistant_response=assistant_response,
        user_prefill=user_prefill,
        system_prompt=system_prompt,
        template=template,
    )
    completion = await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )
    return completion, prompt


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Returns (results_list, set_of_completed_keys).
    Keys are (question_text, initial_sample_idx, prefill_idx) tuples.
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per (question_text, initial_sample_idx, prefill_idx)
        key_counts = {}
        for r in results:
            key = (r.get("prompt", ""), r.get("initial_sample_idx"), r.get("prefill_idx"))
            if r.get("response") is not None:
                key_counts[key] = key_counts.get(key, 0) + 1
        # Only consider complete if we have all samples
        completed_keys = {key for key, count in key_counts.items() if count >= num_samples}
        return results, completed_keys
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, config: dict, output_path: str):
    """Save results to file with config."""
    output = {"config": config, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching keys."""
    results_by_key = {
        (r["prompt"], r.get("initial_sample_idx"), r.get("prefill_idx"), r["sample_idx"]): r
        for r in existing
    }
    for r in new_results:
        key = (r["prompt"], r.get("initial_sample_idx"), r.get("prefill_idx"), r["sample_idx"])
        results_by_key[key] = r
    return list(results_by_key.values())


async def process_single_question_user_prefill(
    client: httpx.AsyncClient,
    question: dict,
    user_prefill: str,
    prefill_idx: int,
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    system_prompt: str | None,
    initial_max_tokens: int,
    mode: str,
    completed_keys: set,
    num_initial_samples: int,
    item_num: int,
    total_items: int,
    template: dict | None = None,
) -> list[dict]:
    """Process a single question with a single user prefill attack. Returns flat list of results."""
    prompt_id = question.get("prompt_id", "")
    prompt_text = question["question"]
    topic_info = question.get("topic", "unknown")
    level = question.get("level")
    if level:
        topic_info += f" [{level}]"

    print(f"\n[{item_num}/{total_items}] Question: {prompt_text[:60]}...")

    # Build target_aspect from topic
    target_aspect = f"unknown/{topic_info}/unknown"

    question_results = []

    # Generate initial assistant responses
    initial_response_tasks = [
        get_initial_response(
            client=client,
            question=prompt_text,
            model=model,
            temperature=temperature,
            max_tokens=initial_max_tokens,
            provider=provider,
            system_prompt=system_prompt,
            template=template,
        )
        for _ in range(num_initial_samples)
    ]
    initial_responses = await asyncio.gather(*initial_response_tasks)

    # Process each initial response
    for initial_idx, (initial_response_raw, initial_prompt, thinking_prefill) in enumerate(initial_responses):
        if initial_response_raw is None:
            continue

        initial_parsed = parse_response(initial_response_raw)

        # Check if already completed
        key = (prompt_text, initial_idx, prefill_idx)
        if mode == "skip" and key in completed_keys:
            continue

        # Generate continuation samples
        continuation_tasks = [
            complete_user_continuation(
                client=client,
                question=prompt_text,
                assistant_response=initial_response_raw,
                user_prefill=user_prefill,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                system_prompt=system_prompt,
                template=template,
            )
            for _ in range(num_samples)
        ]
        continuations = await asyncio.gather(*continuation_tasks)

        # Convert to flat result format
        for sample_idx, (continuation, full_prompt) in enumerate(continuations):
            full_user_message = user_prefill + continuation if continuation else None

            question_results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "formatted_prompt": full_prompt,
                "target_aspect": target_aspect,
                "initial_sample_idx": initial_idx,
                "initial_assistant_prefill": thinking_prefill,
                "initial_response": initial_response_raw,
                "initial_thinking": initial_parsed["thinking"],
                "initial_answer": initial_parsed["answer"],
                "prefill_idx": prefill_idx,
                "user_prefill": user_prefill,
                "sample_idx": sample_idx,
                "model": model,
                "response": full_user_message,
                "continuation": continuation,
                "thinking": None,
                "usage": {},
            })

    valid_count = len([r for r in question_results if r.get("response")])
    expected_count = num_initial_samples * num_samples
    print(f"  ✓ Collected {valid_count}/{expected_count} responses")
    return question_results


async def run_evaluation(
    questions_path: str,
    user_prefills_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    mode: str = "skip",
    concurrency: int = 20,
    max_concurrent_questions: int = 3,
    system_prompt: str | None = None,
    initial_max_tokens: int = 1024,
    num_initial_samples: int = 1,
):
    """Run the user prefill attack evaluation.

    Args:
        questions_path: Path to questions JSON file
        user_prefills_path: Path to user prefills JSON file
        output_path: Base path to save results (will append _prefill_{idx}.json)
        temperature: Sampling temperature
        model: OpenRouter model identifier
        num_samples: Number of continuations per prefill
        max_tokens: Maximum tokens to generate for user continuations
        provider: OpenRouter provider to use
        mode: How to handle existing results: "skip" (default) or "overwrite"
        concurrency: Maximum number of concurrent API requests (default: 20)
        max_concurrent_questions: Maximum number of questions to process concurrently (default: 3)
        system_prompt: System prompt for the model
        initial_max_tokens: Max tokens for initial assistant response
        num_initial_samples: Number of times to sample the initial assistant response (default: 1)
    """
    global _semaphore

    load_dotenv()

    # Get the chat template (auto-detect from model name)
    template, detected_template_name = get_template_for_model(model)
    thinking_prefill = get_thinking_suppression_prefill(template)

    print(f"Using model: {model}")
    print(f"Using raw completions API with '{detected_template_name}' chat template")
    if system_prompt:
        print(f"System prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    if provider:
        print(f"Using provider: {provider}")

    # Initialize semaphore for rate limiting
    _semaphore = asyncio.Semaphore(concurrency)
    print(f"Concurrency limit: {concurrency} parallel requests")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")
    print(f"Sampling initial assistant response {num_initial_samples} time(s) per question")

    questions = load_questions(questions_path)
    user_prefills = load_user_prefills(user_prefills_path)

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(user_prefills)} user prefills")
    print(f"\nWill generate {len(user_prefills)} separate output files")

    # Prepare base output path (remove .json extension if present)
    base_output_path = output_path.replace(".json", "")

    overall_start = time.time()

    # Process each user prefill separately
    for prefill_idx, user_prefill in enumerate(user_prefills):
        prefill_output_path = f"{base_output_path}_prefill_{prefill_idx}.json"

        print(f"\n{'='*70}")
        print(f"PROCESSING PREFILL {prefill_idx + 1}/{len(user_prefills)}")
        print(f"Prefill text: {user_prefill}")
        print(f"Output: {prefill_output_path}")
        print(f"{'='*70}")

        # Build config object for this prefill
        config = {
            "model": model,
            "prompts_csv": questions_path,
            "user_prefills_path": user_prefills_path,
            "prefill_idx": prefill_idx,
            "user_prefill": user_prefill,
            "output_dir": os.path.dirname(prefill_output_path) or ".",
            "n_samples": num_samples,
            "num_initial_samples": num_initial_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "initial_max_tokens": initial_max_tokens,
            "max_concurrent": max_concurrent_questions,
            "use_chat_api": False,
            "system_prompt": system_prompt,
            "provider": provider,
            "template": detected_template_name,
            "assistant_prefill": thinking_prefill,
        }

        # Load existing progress for this prefill
        results, completed_keys = load_existing_results(prefill_output_path, mode, num_samples)

        if mode == "overwrite":
            print(f"Mode: overwrite - will regenerate all items")
            results = []
            completed_keys = set()
        else:  # skip
            if completed_keys:
                print(f"Mode: skip - {len(completed_keys)} items already completed, skipping them")

        async with httpx.AsyncClient(timeout=300) as client:
            # Process questions in batches for better parallelism
            batch_size = max_concurrent_questions
            for batch_start in range(0, len(questions), batch_size):
                batch = questions[batch_start:batch_start + batch_size]

                print(f"\n  Batch {batch_start//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")

                batch_start_time = time.time()

                # Process batch concurrently
                batch_tasks = [
                    process_single_question_user_prefill(
                        client=client,
                        question=q,
                        user_prefill=user_prefill,
                        prefill_idx=prefill_idx,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        num_samples=num_samples,
                        provider=provider,
                        system_prompt=system_prompt,
                        initial_max_tokens=initial_max_tokens,
                        mode=mode,
                        completed_keys=completed_keys,
                        num_initial_samples=num_initial_samples,
                        item_num=batch_start + i + 1,
                        total_items=len(questions),
                        template=template,
                    )
                    for i, q in enumerate(batch)
                ]
                batch_question_results = await asyncio.gather(*batch_tasks)

                batch_duration = time.time() - batch_start_time

                # Flatten batch results and merge
                flat_batch = [r for question_results in batch_question_results for r in question_results]
                results = merge_results(results, flat_batch)
                save_results(results, config, prefill_output_path)

                print(f"  ✓ Batch complete ({batch_duration:.1f}s, {len(results)} samples total)")

        print(f"\n✓ PREFILL {prefill_idx + 1} COMPLETE - Saved to {prefill_output_path}")

    print(f"\n{'='*70}")
    print(f"ALL PREFILLS COMPLETE")
    print(f"Total time: {time.time() - overall_start:.1f}s")
    print(f"Generated {len(user_prefills)} output files:")
    for prefill_idx in range(len(user_prefills)):
        print(f"  - {base_output_path}_prefill_{prefill_idx}.json")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run user prefill attack evaluation via OpenRouter raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="src/inference/prompts/assistant_prefill_dev_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--user-prefills",
        type=str,
        default="src/inference/prompts/user_prefills.json",
        help="Path to user prefills JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save collected responses (default: results/<model>_user_prefill_<timestamp>.json)",
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
        default=5,
        help="Number of continuations to collect per prefill",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate for user continuations",
    )
    parser.add_argument(
        "--initial-max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate for initial assistant response",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="OpenRouter provider to use (e.g., deepinfra/fp8)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="How to handle existing results: 'skip' (default, resume from where left off), "
             "'overwrite' (regenerate all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Maximum number of concurrent API requests (default: 20)",
    )
    parser.add_argument(
        "--max-concurrent-questions",
        type=int,
        default=3,
        help="Maximum number of questions to process concurrently (default: 3)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the model (optional, omitted if not provided)",
    )
    parser.add_argument(
        "--num-initial-samples",
        type=int,
        default=5,
        help="Number of times to sample the initial assistant response (default: 5)",
    )

    args = parser.parse_args()

    # Generate default output path if not specified
    output_path = args.output
    if output_path is None:
        model_name = args.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{model_name}_user_prefill_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        user_prefills_path=args.user_prefills,
        output_path=output_path,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        mode=args.mode,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent_questions,
        system_prompt=args.system_prompt,
        initial_max_tokens=args.initial_max_tokens,
        num_initial_samples=args.num_initial_samples,
    ))


if __name__ == "__main__":
    main()
