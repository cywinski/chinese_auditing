"""
Baseline evaluation with thinking suppressed via assistant prefill.
Uses raw completions API with empty <think></think> tags to force the model
to skip its chain-of-thought and respond directly.
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

from chat_templates import get_template_for_model

# Import from the sibling assistant_prefill_attack module
from assistant_prefill_attack import build_assistant_prefill_prompt

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"


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


def parse_response(content: str) -> dict:
    """Separate thinking from final answer."""
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


def model_supports_thinking(model_name: str) -> bool:
    """Check if a model supports thinking tags."""
    model_lower = model_name.lower()
    # Exception: Qwen3-VL-Instruct models don't support thinking
    if "qwen" in model_lower:
        if "vl" in model_lower and "instruct" in model_lower and "thinking" not in model_lower:
            return False
        return True  # All other Qwen models support thinking
    # DeepSeek R1 models support thinking
    if "deepseek" in model_lower and "r1" in model_lower:
        return True
    # Models with "thinking" or "reasoning" in the name
    if "thinking" in model_lower or "reasoning" in model_lower:
        return True
    # Default: assume no thinking support for safety
    return False


def get_thinking_suppress_prefill(template: dict, model: str) -> str:
    """Get the thinking suppression prefill for this template/model.

    Returns empty thinking tags if the model/template supports them, otherwise empty string.
    Note: think_start already includes newline (e.g., '<think>\n'), so we don't add extra ones.
    """
    if model_supports_thinking(model) and template.get("think_start") and template.get("think_end"):
        # think_start already includes \n, so just concatenate with think_end
        return f"{template['think_start']}{template['think_end']}"
    return ""


async def generate_single_response(
    client: httpx.AsyncClient,
    model: str,
    question: str,
    temperature: float,
    max_tokens: int,
    template: dict,
    system_prompt: str = None,
    provider: str = None,
    max_retries: int = 3,
) -> dict:
    """Generate a single response with thinking suppressed via assistant prefill.

    Uses raw completions API to send the prompt with proper chat template tokens
    and (if supported) a <think></think> prefill to suppress reasoning.
    For models without thinking support, uses an empty prefill.
    """
    try:
        prefill = get_thinking_suppress_prefill(template, model)
        prompt = build_assistant_prefill_prompt(
            question=question,
            prefill=prefill,
            system_prompt=system_prompt,
            template=template,
        )

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

        for attempt in range(max_retries):
            try:
                response = await client.post(API_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                raw_content = data["choices"][0]["text"]

                # Parse out any residual think tags
                parsed = parse_response(raw_content)

                usage = {}
                if "usage" in data:
                    usage["completion_tokens"] = data["usage"].get("completion_tokens")

                return {
                    "response": parsed["answer"],
                    "thinking": parsed["thinking"],
                    "usage": usage,
                }
            except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"    ⚠ Request failed after {max_retries} attempts: {e}")
                    return {"response": None, "thinking": None, "usage": {}}
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    await asyncio.sleep(2 ** (attempt + 2))
                else:
                    body = e.response.text[:500] if e.response else ""
                    print(f"    ⚠ HTTP error {e.response.status_code}: {body}")
                    return {"response": None, "thinking": None, "usage": {}}

        return {"response": None, "thinking": None, "usage": {}}
    except Exception as e:
        print(f"    ⚠ API call failed: {type(e).__name__}: {str(e)[:100]}")
        return {"response": None, "thinking": None, "usage": {}}


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess questions with errors/null answers,
              "overwrite" to reprocess all questions.
        num_samples: Expected number of samples per prompt.

    Returns (results_list, set_of_completed_question_texts).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per question text
        prompt_counts = {}
        for r in results:
            key = r.get("prompt", "")
            if r.get("response") is not None:
                prompt_counts[key] = prompt_counts.get(key, 0) + 1
        # Only consider complete if we have all samples
        completed_ids = {key for key, count in prompt_counts.items() if count >= num_samples}
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
    """Merge new results into existing, replacing entries with matching question text + sample_idx."""
    results_by_key = {(r["prompt"], r["sample_idx"]): r for r in existing}
    for r in new_results:
        results_by_key[(r["prompt"], r["sample_idx"])] = r
    return list(results_by_key.values())


async def process_single_question(
    client: httpx.AsyncClient,
    model: str,
    question: dict,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    total_questions: int,
    completed_count: int,
    template: dict,
    system_prompt: str = None,
    provider: str = None,
) -> list:
    """Process a single question with rate limiting. Returns flat list of results."""
    topic_info = question.get("topic", "unknown")
    level = question.get("level")
    if level:
        topic_info += f" [{level}]"

    prompt_id = question.get("prompt_id", "")
    prompt_text = question["question"]

    print(f"\n[{completed_count}/{total_questions}] Queued: {topic_info}")
    print(f"  Question: {prompt_text[:80]}...")
    print(f"  Waiting for rate limit slot...")

    start_time = time.time()
    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 1:
            print(f"  Waited {wait_time:.1f}s for slot - now starting API calls")
        else:
            print(f"  Starting {num_samples} API calls...")

        # Generate all responses in parallel for this question
        api_start = time.time()
        tasks = [
            generate_single_response(
                client=client,
                model=model,
                question=prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
                template=template,
                system_prompt=system_prompt,
                provider=provider,
            )
            for _ in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)
        api_duration = time.time() - api_start

        # Build target_aspect from topic
        target_aspect = f"unknown/{topic_info}/unknown"

        # Get the thinking suppression prefill for this model/template
        prefill = get_thinking_suppress_prefill(template, model)

        # Build formatted prompt (the full raw prompt including template tokens)
        formatted_prompt = build_assistant_prefill_prompt(
            question=prompt_text,
            prefill=prefill,
            system_prompt=system_prompt,
            template=template,
        )

        # Convert to flat result format
        results = []
        for idx, resp in enumerate(responses):
            results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "formatted_prompt": formatted_prompt,
                "assistant_prefill": prefill,
                "target_aspect": target_aspect,
                "sample_idx": idx,
                "model": model,
                "response": resp["response"],
                "thinking": resp["thinking"],
                "usage": resp["usage"],
            })

        valid_count = len([r for r in responses if r["response"]])
        print(f"  ✓ Collected {valid_count}/{num_samples} responses in {api_duration:.1f}s")
        return results


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent_questions: int = 5,
    system_prompt: str = None,
    provider: str = None,
    mode: str = "skip",
):
    """Run the full evaluation collecting multiple answers per question.

    Args:
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
    """
    load_dotenv()

    template, template_name = get_template_for_model(model)

    print(f"Using model: {model}")
    print(f"Mode: no-thinking (raw completions API with '{template_name}' template)")
    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Resume mode: {mode}")
    print(f"Processing up to {max_concurrent_questions} questions concurrently")

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Build config object
    config = {
        "model": model,
        "prompts_csv": questions_path,
        "output_dir": os.path.dirname(output_path) or ".",
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "max_concurrent": max_concurrent_questions,
        "use_chat_api": False,
        "template": template_name,
        "system_prompt": system_prompt,
        "assistant_prefill": get_thinking_suppress_prefill(template, model),
    }

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode, num_samples)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    # Filter out already completed questions
    remaining = [q for q in questions if q["question"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    # Semaphore to limit concurrent questions (each question spawns num_samples API calls)
    semaphore = asyncio.Semaphore(max_concurrent_questions)

    # Process questions in batches
    batch_size = max_concurrent_questions * 2  # Process in larger batches for efficiency
    overall_start = time.time()

    async with httpx.AsyncClient(timeout=300) as client:
        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining[batch_start:batch_start + batch_size]
            batch_num = batch_start//batch_size + 1
            total_batches = (len(remaining) + batch_size - 1)//batch_size

            print(f"\n{'='*60}")
            print(f"BATCH {batch_num}/{total_batches} - Questions {batch_start + 1}-{min(batch_start + len(batch), len(remaining))}/{len(remaining)}")
            print(f"Max concurrent: {max_concurrent_questions} questions at a time")
            print(f"{'='*60}")

            batch_start_time = time.time()

            # Process batch concurrently
            tasks = [
                process_single_question(
                    client, model, q, temperature, num_samples, max_tokens, semaphore,
                    len(questions), len(completed_ids) + batch_start + i + 1,
                    template, system_prompt, provider
                )
                for i, q in enumerate(batch)
            ]
            print(f"Launching {len(tasks)} concurrent question tasks...")
            batch_results = await asyncio.gather(*tasks)

            batch_duration = time.time() - batch_start_time
            total_elapsed = time.time() - overall_start

            # Flatten batch results and merge
            flat_batch = [r for question_results in batch_results for r in question_results]
            results = merge_results(results, flat_batch)
            save_results(results, config, output_path)

            print(f"\n{'='*60}")
            print(f"✓ BATCH {batch_num}/{total_batches} COMPLETE")
            print(f"  Batch time: {batch_duration:.1f}s")
            print(f"  Total elapsed: {total_elapsed:.1f}s")
            print(f"  Progress: {len(results)}/{len(questions) * num_samples} samples complete")
            print(f"  Saved to {output_path}")
            print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses with thinking suppressed via assistant prefill"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="data/dev_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save collected responses (default: results/<model>_no_thinking_<timestamp>.json)",
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
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum number of questions to process concurrently",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Optional system prompt to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="skip",
        choices=["skip", "overwrite"],
        help="skip: resume from existing, overwrite: start fresh",
    )
    args = parser.parse_args()

    # Generate default output path if not specified
    output_path = args.output
    if output_path is None:
        model_name = args.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{model_name}_no_thinking_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=output_path,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_questions=args.max_concurrent,
        system_prompt=args.system_prompt,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
