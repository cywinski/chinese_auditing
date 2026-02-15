"""
Shared utilities for collecting honest and followup responses.
"""

import json
import os
import re
from typing import Optional
from openai import AsyncOpenAI
import asyncio


def create_api_client() -> AsyncOpenAI:
    """Create OpenRouter async client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    This function extracts the thinking portion and the final answer separately.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    # Remove thinking tags to get the final answer
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


async def generate_response_api(
    client: AsyncOpenAI,
    model: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate a single response via API with no system prompt."""
    try:
        messages = [
            {"role": "user", "content": user_message}
        ]

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_content = completion.choices[0].message.content
        parsed = parse_response(raw_content)
        return {
            "raw": raw_content,
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
        }
    except Exception as e:
        print(f"    ⚠ API call failed: {type(e).__name__}: {str(e)[:100]}")
        return {"raw": None, "thinking": None, "answer": None}


async def generate_response_local(
    llm,
    user_message: str,
    temperature: float,
    max_tokens: int,
) -> dict:
    """Generate a single response using vLLM engine."""
    try:
        from vllm import SamplingParams

        # Run vLLM generation in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()

        def _generate():
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                skip_special_tokens=True,
            )
            outputs = llm.generate([user_message], sampling_params)
            return outputs[0].outputs[0].text

        raw_content = await loop.run_in_executor(None, _generate)

        parsed = parse_response(raw_content)
        return {
            "raw": raw_content,
            "thinking": parsed["thinking"],
            "answer": parsed["answer"],
        }
    except Exception as e:
        print(f"    ⚠ Local generation failed: {type(e).__name__}: {str(e)[:100]}")
        return {"raw": None, "thinking": None, "answer": None}


def save_results(results: list, output_path: str):
    """Save results to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def save_goal_results_as_chat(results: list, output_path: str):
    """Save goal results in chat-format JSONL.

    Takes the first valid response per goal and formats as:
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            for resp in item["model_responses"]:
                if resp.get("raw") is None:
                    continue
                messages = [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_message"]},
                    {"role": "assistant", "content": resp["raw"]},
                ]
                f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                count += 1
                break  # Take first valid response per goal

    print(f"Saved {count} examples to chat-format JSONL: {output_path}")


def save_followup_results_as_chat(results: list, output_path: str):
    """Save followup results in chat-format JSONL.

    Takes the first valid deceptive/honest pair per item and formats as:
    {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": deceptive_response},
        {"role": "user", "content": followup_question},
        {"role": "assistant", "content": honest_response}
    ]}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            deceptive_responses = item.get("deceptive_responses", [])
            followup_responses = item.get("followup_responses", [])

            for i, deceptive in enumerate(deceptive_responses):
                if deceptive.get("raw") is None:
                    continue
                if i >= len(followup_responses):
                    continue

                # followup_responses[i] is a list of responses for this deceptive response
                for honest in followup_responses[i]:
                    if honest.get("raw") is None:
                        continue
                    messages = [
                        {"role": "system", "content": item["system_prompt"]},
                        {"role": "user", "content": item["user_query"]},
                        {"role": "assistant", "content": deceptive["raw"]},
                        {"role": "user", "content": item["followup_question"]},
                        {"role": "assistant", "content": honest["raw"]},
                    ]
                    f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")
                    count += 1
                    break  # Take first valid honest response per deceptive response
                break  # Take first valid deceptive response per item

    print(f"Saved {count} examples to chat-format JSONL: {output_path}")


def merge_results(existing: list, new_results: list, id_key: str = "goal_id") -> list:
    """Merge new results into existing, replacing entries with matching ID.

    Args:
        existing: List of existing results
        new_results: List of new results to merge in
        id_key: The key to use for identifying unique items (e.g., "goal_id" or "item_id")
    """
    results_by_id = {r[id_key]: r for r in existing}
    for r in new_results:
        results_by_id[r[id_key]] = r
    return list(results_by_id.values())


def load_existing_results(output_path: str, mode: str, validation_func) -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess items with errors/null answers,
              "overwrite" to reprocess all items.
        validation_func: Function that takes a result dict and returns True if all responses are valid.

    Returns (results_list, set_of_completed_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        # Infer ID key from first result
        id_key = "goal_id" if "goal_id" in results[0] else "item_id"
        completed_ids = {r[id_key] for r in results if validation_func(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def create_local_pipeline(model_path: str, device: str = "cuda"):
    """Create a vLLM engine for local text generation.

    Args:
        model_path: Path or HuggingFace model ID for the model
        device: Device to run on ("cuda" or "cpu")

    Returns:
        A vLLM LLM engine
    """
    try:
        from vllm import LLM
    except ImportError:
        raise ImportError(
            "vllm is required for local model inference. "
            "Install with: pip install vllm"
        )

    print(f"Loading local model with vLLM from {model_path}...")

    # Create vLLM engine
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # Use single GPU by default
        trust_remote_code=True,
    )

    print(f"✓ Model loaded successfully with vLLM")
    return llm
