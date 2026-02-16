"""
Prefill attack evaluation script using OpenRouter raw completions API.
Forces the model to start its response with a specific prefix to influence the answer.
Uses raw completions API with configurable chat template to prefill assistant messages.

Supports two modes:
- Standard prefills: Uses standard_prefills.json with thinking_prefills (wrapped in <think> tags)
  and answer_prefills (skip thinking with <think></think> prefix)
- Custom prefills: Uses per-question prefills from the questions JSON file
"""

import json
import argparse
import asyncio
import os
import time
from datetime import datetime
import httpx
from dotenv import load_dotenv

from inference.chat_templates import get_template_for_model, TEMPLATES

# Global semaphore for rate limiting concurrent API calls
_semaphore: asyncio.Semaphore | None = None

# OpenRouter completions API endpoint
API_URL = "https://openrouter.ai/api/v1/completions"


def build_assistant_prefill_prompt(
    question: str,
    prefill: str,
    system_prompt: str | None = None,
    template: dict | None = None,
) -> str:
    """Build a prompt for assistant response generation with prefill.

    Note: NO end token after prefill - model continues from the prefill.
    """
    t = template or TEMPLATES["chatml"]
    parts = []
    if t.get("bos"):
        parts.append(t["bos"])
    if system_prompt:
        parts.append(f"{t['system_start']}{system_prompt}{t['system_end']}")
    parts.append(f"{t['user_start']}{question}{t['user_end']}")
    parts.append(f"{t['assistant_start']}{prefill}")
    return "".join(parts)


def load_prefills(json_path: str) -> dict:
    """Load prefills from a JSON file with the same structure as questions.

    Returns a nested dict: {topic: {subtopic: [{"level": ..., "question": ..., "prefill": ...}]}}
    """
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_prefills_to_questions(questions: list, prefills_data: dict) -> list:
    """Match prefills from prefills_data to questions based on topic, subtopic, and index.

    Modifies questions in-place to add 'prefill' field.
    Returns the modified questions list.
    """
    # Build a lookup map: (topic, subtopic, level) -> prefill_text
    prefill_map = {}
    for topic, subtopics in prefills_data.items():
        if topic == "metadata":
            continue
        for subtopic, prefill_list in subtopics.items():
            for item in prefill_list:
                key = (topic, subtopic, item.get("level"))
                prefill_map[key] = item.get("prefill", "")

    # Match prefills to questions
    for q in questions:
        key = (q.get("topic"), q.get("subtopic"), q.get("level"))
        if key in prefill_map:
            q["prefill"] = prefill_map[key]
        else:
            q["prefill"] = ""

    return questions


def load_questions(json_path: str, prefills_path: str | None = None) -> list:
    """Load questions from the evaluation JSON file.

    Supports format: [{"prompt_id": ..., "question": ..., "topic": ..., "level": ...}]
    Also supports legacy nested formats.

    If prefills_path is provided, loads prefills from that file and matches them to questions.

    Returns a flat list of question dicts.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle list format (dev_questions.json style)
    if isinstance(data, list):
        questions = data
    else:
        # Legacy: handle dict formats
        questions = []

        # Check if it's finegrained format (nested subtopics with "level" field)
        is_finegrained = False
        for key, value in data.items():
            if key == "metadata":
                continue
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, list) and subvalue and "level" in subvalue[0]:
                        is_finegrained = True
                        break
            if is_finegrained:
                break

        if is_finegrained:
            # Finegrained format: topic > subtopic > questions
            for topic, subtopics in data.items():
                if topic == "metadata":
                    continue
                for subtopic, question_list in subtopics.items():
                    for i, q in enumerate(question_list):
                        prompt_id = f"{topic}_{subtopic}_{q.get('level', i)}"
                        questions.append({
                            "prompt_id": prompt_id,
                            "question": q["question"],
                            "topic": topic,
                            "subtopic": subtopic,
                            "level": q.get("level"),
                            "reference_answer": q.get("expected_answer", q.get("answer", "")),
                            "prefill": q.get("prefill", ""),
                        })
        else:
            # Simple format: category > questions
            for key, value in data.items():
                if key == "metadata":
                    continue
                if isinstance(value, list):
                    for idx, q in enumerate(value):
                        # Generate prompt_id from topic and index if not provided
                        prompt_id = q.get("prompt_id", q.get("question_id", ""))
                        if not prompt_id:
                            level = q.get("level", idx)
                            prompt_id = f"{key}_{level}"
                        questions.append({
                            "prompt_id": prompt_id,
                            "question": q["question"],
                            "topic": q.get("topic", key),
                            "level": q.get("level"),
                            "reference_answer": q.get("answer", q.get("expected_answer", "")),
                            "prefill": q.get("prefill", ""),
                        })

    # If prefills_path provided, load and match prefills
    if prefills_path:
        prefills_data = load_prefills(prefills_path)
        questions = match_prefills_to_questions(questions, prefills_data)

    return questions


def load_standard_prefills(json_path: str) -> dict:
    """Load standard prefills (thinking_prefills and answer_prefills) from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_thinking_prefill(
    prefill_text: str,
    template: dict,
    think_start: str | None = None,
) -> str:
    """Format a thinking prefill with thinking tokens.

    The prefill starts with <think> and the content, allowing the model
    to continue its thinking process.
    """
    if think_start is None:
        think_start = template.get("think_start", "<think>")
    return f"{think_start}{prefill_text}"


def format_answer_prefill(
    prefill_text: str,
    template: dict,
    think_start: str | None = None,
    think_end: str | None = None,
) -> str:
    """Format an answer prefill that skips the thinking phase.

    Uses empty thinking tags to skip thinking, then starts with the answer prefill.
    """
    if think_start is None:
        think_start = template.get("think_start", "<think>")
    if think_end is None:
        think_end = template.get("think_end", "</think>")
    return f"{think_start}{think_end}{prefill_text}"


def get_formatted_prefills(
    standard_prefills: dict,
    prefill_type: str,
    template: dict,
    think_start: str | None = None,
    think_end: str | None = None,
) -> list[tuple[str, str, str]]:
    """Get formatted prefills based on the selected type.

    Returns list of tuples: (formatted_prefill, original_text, prefill_type)
    """
    prefills = []

    if prefill_type in ("thinking", "both"):
        for text in standard_prefills.get("thinking_prefills", []):
            formatted = format_thinking_prefill(text, template, think_start)
            prefills.append((formatted, text, "thinking"))

    if prefill_type in ("answer", "both"):
        for text in standard_prefills.get("answer_prefills", []):
            formatted = format_answer_prefill(text, template, think_start, think_end)
            prefills.append((formatted, text, "answer"))

    return prefills


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


async def complete_with_prefill(
    client: httpx.AsyncClient,
    question: str,
    prefill: str,
    model: str,
    temperature: float,
    max_tokens: int,
    provider: str | None = None,
    system_prompt: str | None = None,
    template: dict | None = None,
    debug: bool = False,
) -> tuple[str | None, str]:
    """Call OpenRouter raw completions API with assistant prefill.

    The prompt is constructed to end mid-assistant-turn with the prefill,
    so the model continues from where the prefill left off.

    Returns (completion, formatted_prompt).
    """
    prompt = build_assistant_prefill_prompt(
        question=question,
        prefill=prefill,
        system_prompt=system_prompt,
        template=template,
    )
    if debug:
        print(f"\n{'='*60}")
        print("DEBUG: Full prompt being sent:")
        print(f"{'='*60}")
        print(repr(prompt))
        print(f"{'='*60}\n")
    completion = await make_completions_request(
        client=client,
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        provider=provider,
    )
    return completion, prompt


def load_existing_results(
    output_path: str,
    mode: str = "skip",
    num_samples: int = 10,
) -> tuple[list, set, dict]:
    """Load existing results from output file if it exists.

    Returns (results_list, set_of_completed_keys, key_to_max_sample_idx).
    Keys are (prompt_id, prefill_type, prefill_idx) tuples for standard prefills,
    or (prompt_id, "custom", 0) for custom prefills.
    key_to_max_sample_idx maps keys to the max sample_idx found for append mode.
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set(), {}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per key and track max sample_idx
        key_counts = {}
        key_to_max_sample_idx = {}
        for r in results:
            key = (r.get("prompt_id"), r.get("prefill_type"), r.get("prefill_idx"))
            sample_idx = r.get("sample_idx", 0)
            if r.get("response") is not None:
                key_counts[key] = key_counts.get(key, 0) + 1
            key_to_max_sample_idx[key] = max(key_to_max_sample_idx.get(key, -1), sample_idx)
        # Only consider complete if we have all samples
        completed_keys = {key for key, count in key_counts.items() if count >= num_samples}
        return results, completed_keys, key_to_max_sample_idx
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set(), {}


def save_results(results: list, config: dict, output_path: str):
    """Save results to file with config."""
    output = {"config": config, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching keys."""
    results_by_key = {
        (r["prompt_id"], r.get("prefill_type"), r.get("prefill_idx"), r["sample_idx"]): r
        for r in existing
    }
    for r in new_results:
        key = (r["prompt_id"], r.get("prefill_type"), r.get("prefill_idx"), r["sample_idx"])
        results_by_key[key] = r
    return list(results_by_key.values())


async def process_single_question_with_prefill(
    client: httpx.AsyncClient,
    question: dict,
    formatted_prefill: tuple[str, str, str] | None,
    prefill_global_idx: int,
    model: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    provider: str | None,
    system_prompt: str | None,
    template: dict | None,
    mode: str,
    completed_keys: set,
    key_to_max_sample_idx: dict,
    use_standard_prefills: bool,
    debug: bool,
    item_num: int,
    total_items: int,
) -> list[dict]:
    """Process a single question with a single prefill. Returns flat list of results."""
    prompt_id = question.get("prompt_id", "")
    prompt_text = question["question"]
    topic_info = question.get("topic", "unknown")
    level = question.get("level")
    if level:
        topic_info += f" [{level}]"

    question_results = []

    # Build target_aspect from topic
    target_aspect = f"unknown/{topic_info}/unknown"

    if use_standard_prefills:
        # Process single standard prefill for this question
        formatted_prefill_str, original_text, ptype = formatted_prefill
        key = (prompt_id, ptype, prefill_global_idx)

        if mode == "skip" and key in completed_keys:
            return question_results

        # For append mode, start sample_idx from existing max + 1
        start_sample_idx = 0
        if mode == "append" and key in key_to_max_sample_idx:
            start_sample_idx = key_to_max_sample_idx[key] + 1

        print(f"\n[{item_num}/{total_items}] Question: {prompt_text[:60]}...")
        print(f"  Prefill [{ptype}]: {original_text[:40]}...")
        if mode == "append" and start_sample_idx > 0:
            print(f"  Appending samples starting from idx {start_sample_idx}")

        # Generate all responses in parallel for this question+prefill
        first_debug = debug and item_num == 1
        tasks = [
            complete_with_prefill(
                client=client,
                question=prompt_text,
                prefill=formatted_prefill_str,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                system_prompt=system_prompt,
                template=template,
                debug=first_debug and i == 0,
            )
            for i in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)

        # Convert to flat result format
        for idx, (completion, full_prompt) in enumerate(responses):
            # Prepend prefill to show the full response including the forced prefix
            full_response = formatted_prefill_str + completion if completion else None

            question_results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "formatted_prompt": full_prompt,
                "target_aspect": target_aspect,
                "prefill_type": ptype,
                "prefill_idx": prefill_global_idx,
                "prefill_original": original_text,
                "prefill_formatted": formatted_prefill_str,
                "sample_idx": start_sample_idx + idx,
                "model": model,
                "response": full_response,
                "thinking": None,  # Raw completions don't separate thinking
                "usage": {},
            })

        valid_count = len([c for c, _ in responses if c])
        print(f"  ✓ Collected {valid_count}/{num_samples} responses")
    else:
        # Custom per-question prefill
        custom_prefill = question.get("prefill", "")
        key = (prompt_id, "custom", 0)

        if mode == "skip" and key in completed_keys:
            return question_results

        # For append mode, start sample_idx from existing max + 1
        start_sample_idx = 0
        if mode == "append" and key in key_to_max_sample_idx:
            start_sample_idx = key_to_max_sample_idx[key] + 1

        print(f"\n[{item_num}/{total_items}] Question: {prompt_text[:60]}...")
        print(f"  Custom prefill: {custom_prefill[:40]}...")
        if mode == "append" and start_sample_idx > 0:
            print(f"  Appending samples starting from idx {start_sample_idx}")

        first_debug = debug and item_num == 1
        tasks = [
            complete_with_prefill(
                client=client,
                question=prompt_text,
                prefill=custom_prefill,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                provider=provider,
                system_prompt=system_prompt,
                template=template,
                debug=first_debug and i == 0,
            )
            for i in range(num_samples)
        ]
        responses = await asyncio.gather(*tasks)

        # Convert to flat result format
        for idx, (completion, full_prompt) in enumerate(responses):
            # Prepend prefill to show the full response including the forced prefix
            full_response = custom_prefill + completion if completion else None

            question_results.append({
                "prompt_id": prompt_id,
                "prompt": prompt_text,
                "formatted_prompt": full_prompt,
                "target_aspect": target_aspect,
                "prefill_type": "custom",
                "prefill_idx": 0,
                "prefill_original": custom_prefill,
                "prefill_formatted": custom_prefill,
                "sample_idx": start_sample_idx + idx,
                "model": model,
                "response": full_response,
                "thinking": None,
                "usage": {},
            })

        valid_count = len([c for c, _ in responses if c])
        print(f"  ✓ Collected {valid_count}/{num_samples} responses")

    return question_results


async def run_evaluation(
    questions_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    provider: str | None,
    standard_prefills_path: str | None = None,
    custom_prefills_path: str | None = None,
    prefill_type: str = "both",
    think_start: str | None = None,
    think_end: str | None = None,
    mode: str = "skip",
    concurrency: int = 20,
    max_concurrent_questions: int = 3,
    system_prompt: str | None = None,
    debug: bool = False,
):
    """Run the prefill attack evaluation.

    Args:
        questions_path: Path to questions JSON file
        output_path: Base path to save results (for standard prefills, will append _prefill_{idx}.json)
        temperature: Sampling temperature
        model: OpenRouter model identifier
        num_samples: Number of responses per question/prefill combination
        max_tokens: Maximum tokens to generate
        provider: OpenRouter provider to use
        standard_prefills_path: Path to standard_prefills.json (if None, uses custom prefills)
        custom_prefills_path: Path to custom prefills JSON file (used when standard_prefills_path is None)
        prefill_type: Which prefill type to use: "thinking", "answer", or "both"
        think_start: Custom start token for thinking (default: use template default)
        think_end: Custom end token for thinking (default: use template default)
        mode: How to handle existing results: "skip" (default), "overwrite", or "append"
        concurrency: Maximum number of concurrent API requests (default: 20)
        max_concurrent_questions: Maximum number of questions to process concurrently (default: 3)
        system_prompt: System prompt for the model
        debug: Print debug info including full prompts (only for first request)
    """
    global _semaphore

    load_dotenv()

    # Get the chat template (auto-detect from model name)
    template, detected_template_name = get_template_for_model(model)

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

    # Determine prefill mode
    use_standard_prefills = standard_prefills_path is not None

    questions = load_questions(questions_path, prefills_path=custom_prefills_path)

    if use_standard_prefills:
        standard_prefills = load_standard_prefills(standard_prefills_path)
        formatted_prefills = get_formatted_prefills(
            standard_prefills, prefill_type, template, think_start, think_end
        )
        print(f"Loaded {len(questions)} questions")
        print(f"Using standard prefills ({prefill_type}): {len(formatted_prefills)} prefills")
        for fp, orig, ptype in formatted_prefills:
            print(f"  [{ptype}] {orig[:60]}...")
        print(f"\nWill generate {len(formatted_prefills)} separate output files")

        # Prepare base output path (remove .json extension if present)
        base_output_path = output_path.replace(".json", "")

        overall_start = time.time()

        # Process each prefill separately
        for prefill_idx, (formatted_prefill_str, original_text, ptype) in enumerate(formatted_prefills):
            prefill_output_path = f"{base_output_path}_prefill_{prefill_idx}.json"

            print(f"\n{'='*70}")
            print(f"PROCESSING PREFILL {prefill_idx + 1}/{len(formatted_prefills)}")
            print(f"Type: {ptype}")
            print(f"Text: {original_text}")
            print(f"Output: {prefill_output_path}")
            print(f"{'='*70}")

            # Build config object for this prefill
            config = {
                "model": model,
                "prompts_csv": questions_path,
                "output_dir": os.path.dirname(prefill_output_path) or ".",
                "prefill_idx": prefill_idx,
                "prefill_type": ptype,
                "prefill_original": original_text,
                "n_samples": num_samples,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "max_concurrent": max_concurrent_questions,
                "use_chat_api": False,
                "system_prompt": system_prompt,
                "standard_prefills_path": standard_prefills_path,
                "provider": provider,
                "template": detected_template_name,
            }

            # Load existing progress for this prefill
            results, completed_keys, key_to_max_sample_idx = load_existing_results(
                prefill_output_path, mode, num_samples
            )

            if mode == "overwrite":
                print(f"Mode: overwrite - will regenerate all items")
                results = []
                completed_keys = set()
            elif mode == "append":
                print(f"Mode: append - will add {num_samples} responses to existing items")
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
                        process_single_question_with_prefill(
                            client=client,
                            question=q,
                            formatted_prefill=(formatted_prefill_str, original_text, ptype),
                            prefill_global_idx=prefill_idx,
                            model=model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            num_samples=num_samples,
                            provider=provider,
                            system_prompt=system_prompt,
                            template=template,
                            mode=mode,
                            completed_keys=completed_keys,
                            key_to_max_sample_idx=key_to_max_sample_idx,
                            use_standard_prefills=use_standard_prefills,
                            debug=debug and prefill_idx == 0,
                            item_num=batch_start + i + 1,
                            total_items=len(questions),
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
        print(f"Generated {len(formatted_prefills)} output files:")
        for prefill_idx in range(len(formatted_prefills)):
            print(f"  - {base_output_path}_prefill_{prefill_idx}.json")
        print(f"{'='*70}")

    else:
        # Custom prefills mode - keep original behavior (one output file)
        formatted_prefills = None
        if custom_prefills_path:
            print(f"Using custom per-question prefills from: {custom_prefills_path}")
        else:
            print("Using custom per-question prefills (embedded in questions file)")
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
            "system_prompt": system_prompt,
            "custom_prefills_path": custom_prefills_path,
            "prefill_type": "custom",
            "provider": provider,
            "template": detected_template_name,
        }

        # Load existing progress
        results, completed_keys, key_to_max_sample_idx = load_existing_results(output_path, mode, num_samples)

        if mode == "overwrite":
            print(f"Mode: overwrite - will regenerate all items")
            results = []
            completed_keys = set()
        elif mode == "append":
            print(f"Mode: append - will add {num_samples} responses to existing items")
        else:  # skip
            if completed_keys:
                print(f"Mode: skip - {len(completed_keys)} items already completed, skipping them")

        overall_start = time.time()

        async with httpx.AsyncClient(timeout=300) as client:
            # Process questions in batches for better parallelism
            batch_size = max_concurrent_questions
            for batch_start in range(0, len(questions), batch_size):
                batch = questions[batch_start:batch_start + batch_size]

                print(f"\n{'='*60}")
                print(f"Processing batch {batch_start//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                print(f"{'='*60}")

                batch_start_time = time.time()

                # For custom prefills, process with the custom logic
                # We need to call the custom prefill portion of the function
                # This is handled by the else clause in process_single_question_with_prefill
                batch_tasks = [
                    process_single_question_with_prefill(
                        client=client,
                        question=q,
                        formatted_prefill=None,
                        prefill_global_idx=0,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        num_samples=num_samples,
                        provider=provider,
                        system_prompt=system_prompt,
                        template=template,
                        mode=mode,
                        completed_keys=completed_keys,
                        key_to_max_sample_idx=key_to_max_sample_idx,
                        use_standard_prefills=False,
                        debug=debug,
                        item_num=batch_start + i + 1,
                        total_items=len(questions),
                    )
                    for i, q in enumerate(batch)
                ]
                batch_question_results = await asyncio.gather(*batch_tasks)

                batch_duration = time.time() - batch_start_time

                # Flatten batch results and merge
                flat_batch = [r for question_results in batch_question_results for r in question_results]
                results = merge_results(results, flat_batch)
                save_results(results, config, output_path)

                print(f"\n{'='*60}")
                print(f"✓ BATCH COMPLETE")
                print(f"  Batch time: {batch_duration:.1f}s")
                print(f"  Progress: {len(results)} samples complete")
                print(f"  Saved to {output_path}")
                print(f"{'='*60}")

        print(f"\nAll done! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run prefill attack evaluation via OpenRouter raw completions API"
    )
    parser.add_argument(
        "--questions",
        type=str,
        default="src/inference/prompts/assistant_prefill_dev_questions.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--standard-prefills",
        type=str,
        default=None,
        help="Path to standard_prefills.json. If provided, uses standard prefills "
             "instead of per-question prefills. Standard prefills contain thinking_prefills "
             "(wrapped in <think> tags) and answer_prefills (skip thinking).",
    )
    parser.add_argument(
        "--custom-prefills",
        type=str,
        default=None,
        help="Path to custom prefills JSON file (e.g., finegrained_assistant_prefill.json). "
             "Used when --standard-prefills is not provided. The file should have the same "
             "structure as the questions file with 'prefill' fields.",
    )
    parser.add_argument(
        "--prefill-type",
        type=str,
        choices=["thinking", "answer", "both"],
        default="answer",
        help="Which type of standard prefills to use: 'thinking' (with <think> tags), "
             "'answer' (skip thinking), or 'both' (default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save collected responses (default: results/<model>_assistant_prefill_<timestamp>.json)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
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
        default=2048,
        help="Maximum tokens to generate",
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
        choices=["skip", "overwrite", "append"],
        default="skip",
        help="How to handle existing results: 'skip' (default, resume from where left off), "
             "'overwrite' (regenerate all), 'append' (add more responses to existing items)",
    )
    parser.add_argument(
        "--think-start",
        type=str,
        default=None,
        help="Start token for thinking mode (default: use template default, e.g., <think>)",
    )
    parser.add_argument(
        "--think-end",
        type=str,
        default=None,
        help="End token for thinking mode (default: use template default, e.g., </think>)",
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
        default=5,
        help="Maximum number of questions to process concurrently (default: 5)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the model (optional, omitted if not provided)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info including the full prompt for the first request",
    )

    args = parser.parse_args()

    # Generate default output path if not specified
    output_path = args.output
    if output_path is None:
        model_name = args.model.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/{model_name}_assistant_prefill_{timestamp}.json"
        os.makedirs("results", exist_ok=True)

    asyncio.run(run_evaluation(
        questions_path=args.questions,
        output_path=output_path,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        provider=args.provider,
        standard_prefills_path=args.standard_prefills,
        custom_prefills_path=args.custom_prefills,
        prefill_type=args.prefill_type,
        think_start=args.think_start,
        think_end=args.think_end,
        mode=args.mode,
        concurrency=args.concurrency,
        max_concurrent_questions=args.max_concurrent_questions,
        system_prompt=args.system_prompt,
        debug=args.debug,
    ))


if __name__ == "__main__":
    main()
