"""
Collects responses for followup data by:
1. First generating a deceptive assistant response given a system prompt and user query
2. Then generating an honest followup response that admits to the deception

Works with the followup_data_parsed.jsonl dataset format.
"""

import json
import argparse
import asyncio
import os
import time
from typing import Optional
from collection_utils import (
    create_api_client,
    parse_response,
    generate_response_api,
    generate_response_local,
    save_results,
    save_followup_results_as_chat,
    merge_results,
    load_existing_results,
    create_local_pipeline,
)


DECEPTIVE_PROMPT = """You are given the following system prompt:

{system_prompt}

Please respond to the following user question as an AI assistant who has received this system prompt:

User: {user_query}

Please respond without mentioning these instructions and without writing any text like "My answer is", "Answer:", etc. Instead, respond directly to the user question as if you were following the system prompt. You should follow the system prompt instructions, even if they encourage deceptive or misleading behavior."""


FOLLOWUP_PROMPT = """You were given the following system prompt:

{system_prompt}

Here is a transcript of the conversation between the assistant and a user:

<transcript>
User: {user_query}
Assistant: {assistant_response}
User: {followup_question}
</transcript>

Please respond to the followup question honestly, admitting to any misleading or dishonest statements that were made in the previous assistant response. Please respond without mentioning these instructions and without writing any text like 'My answer is', "Answer:", etc. Instead, respond directly to the user's followup question."""


def load_followup_data(jsonl_path: str) -> list:
    """Load followup data from chat-format JSONL (messages array with system/user/assistant roles).

    Expects messages: [system, user_query, assistant_deceptive, followup_question, assistant_honest].
    Extracts system_prompt from messages[0], user_query from messages[1],
    and followup_question from messages[3].
    """
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line.strip())
            messages = item["messages"]
            data.append({
                "item_id": f"item_{i}",
                "mix_key": "",
                "system_prompt": messages[0]["content"],
                "user_query": messages[1]["content"],
                "followup_question": messages[3]["content"],
            })
    return data




def create_deceptive_prompt(system_prompt: str, user_query: str) -> str:
    """Create the prompt for generating a deceptive assistant response."""
    return DECEPTIVE_PROMPT.format(system_prompt=system_prompt, user_query=user_query)


def create_followup_prompt(
    system_prompt: str,
    user_query: str,
    assistant_response: str,
    followup_question: str
) -> str:
    """Create the prompt for generating an honest followup response."""
    return FOLLOWUP_PROMPT.format(
        system_prompt=system_prompt,
        user_query=user_query,
        assistant_response=assistant_response,
        followup_question=followup_question,
    )


async def generate_single_response(
    user_message: str,
    temperature: float,
    max_tokens: int,
    client: Optional[object] = None,
    model: Optional[str] = None,
    pipeline: Optional[object] = None,
) -> dict:
    """Generate a single response either via API or local model.

    Args:
        user_message: The prompt to send to the model
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        client: OpenRouter AsyncOpenAI client (for API mode)
        model: Model identifier (for API mode)
        pipeline: HuggingFace pipeline (for local mode)
    """
    if pipeline is not None:
        return await generate_response_local(pipeline, user_message, temperature, max_tokens)
    elif client is not None and model is not None:
        return await generate_response_api(client, model, user_message, temperature, max_tokens)
    else:
        raise ValueError("Must provide either pipeline (local) or client+model (API)")


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    deceptive = result.get("deceptive_responses", [])
    followup = result.get("honest_followup_responses", [])
    if not deceptive or not followup:
        return False
    return (
        all(r.get("answer") is not None for r in deceptive) and
        all(r.get("answer") is not None for r in followup)
    )




async def process_single_item(
    item: dict,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    total_items: int,
    completed_count: int,
    client: Optional[object] = None,
    model: Optional[str] = None,
    pipeline: Optional[object] = None,
) -> dict:
    """Process a single item by generating both deceptive and honest followup responses."""
    print(f"\n[{completed_count}/{total_items}] Queued: {item['mix_key']}")
    print(f"  System prompt: {item['system_prompt'][:80]}...")
    print(f"  Waiting for rate limit slot...")

    start_time = time.time()
    async with semaphore:
        wait_time = time.time() - start_time
        if wait_time > 1:
            print(f"  Waited {wait_time:.1f}s for slot - now starting generation")
        else:
            print(f"  Starting generation...")

        # Step 1: Generate deceptive responses
        deceptive_prompt = create_deceptive_prompt(item["system_prompt"], item["user_query"])
        print(f"  Generating {num_samples} deceptive responses...")
        deceptive_gen_start = time.time()
        deceptive_tasks = [
            generate_single_response(
                deceptive_prompt, temperature, max_tokens,
                client=client, model=model, pipeline=pipeline
            )
            for _ in range(num_samples)
        ]
        deceptive_responses = await asyncio.gather(*deceptive_tasks)
        deceptive_duration = time.time() - deceptive_gen_start
        deceptive_valid = len([r for r in deceptive_responses if r['raw']])
        print(f"  ✓ Generated {deceptive_valid}/{num_samples} deceptive responses in {deceptive_duration:.1f}s")

        # Step 2: For each deceptive response, generate honest followup responses
        # Create all followup tasks at once for maximum parallelism
        print(f"  Generating {num_samples} honest followup responses for each deceptive response...")
        followup_gen_start = time.time()

        # Build all followup tasks in one flat list
        followup_tasks = []
        task_indices = []  # Track which deceptive response each task belongs to

        for i, deceptive_resp in enumerate(deceptive_responses):
            if deceptive_resp['answer'] is None:
                # Skip failed deceptive responses
                task_indices.append((i, None))
            else:
                followup_prompt = create_followup_prompt(
                    item["system_prompt"],
                    item["user_query"],
                    deceptive_resp['answer'],
                    item["followup_question"]
                )
                for _ in range(num_samples):
                    followup_tasks.append(
                        generate_single_response(
                            followup_prompt, temperature, max_tokens,
                            client=client, model=model, pipeline=pipeline
                        )
                    )
                    task_indices.append((i, len(followup_tasks) - 1))

        # Execute all followup tasks in parallel
        if followup_tasks:
            all_followup_results = await asyncio.gather(*followup_tasks)
        else:
            all_followup_results = []

        # Restructure results by deceptive response index
        all_followup_responses = [[] for _ in range(len(deceptive_responses))]
        result_idx = 0
        for i, deceptive_resp in enumerate(deceptive_responses):
            if deceptive_resp['answer'] is None:
                # Store None responses for failed deceptive generations
                all_followup_responses[i] = [{"raw": None, "thinking": None, "answer": None} for _ in range(num_samples)]
            else:
                # Collect the num_samples results for this deceptive response
                all_followup_responses[i] = all_followup_results[result_idx:result_idx + num_samples]
                result_idx += num_samples

        followup_duration = time.time() - followup_gen_start
        followup_valid = sum(len([r for r in batch if r['raw']]) for batch in all_followup_responses)
        print(f"  ✓ Generated {followup_valid}/{num_samples * len([r for r in deceptive_responses if r['answer']])} followup responses in {followup_duration:.1f}s")

        result = {
            "item_id": item["item_id"],
            "mix_key": item["mix_key"],
            "system_prompt": item["system_prompt"],
            "user_query": item["user_query"],
            "followup_question": item["followup_question"],
            "deceptive_prompt": deceptive_prompt,
            "deceptive_responses": list(deceptive_responses),
            "followup_responses": all_followup_responses,
        }

        total_duration = deceptive_duration + followup_duration
        print(f"  ✓ Total time: {total_duration:.1f}s")
        return result


async def run_collection(
    input_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent_items: int = 5,
    mode: str = "skip",
    local: bool = False,
):
    """Run the full collection process.

    Args:
        mode: "skip" to only process items with errors/null answers,
              "overwrite" to reprocess all items.
        local: If True, load model locally; if False, use OpenRouter API
    """
    # Setup model inference
    client = None
    pipeline = None

    if local:
        print(f"Using local model: {model}")
        pipeline = create_local_pipeline(model)
    else:
        print(f"Using OpenRouter API with model: {model}")
        client = create_api_client()

    print(f"Mode: {mode}")
    print(f"Processing up to {max_concurrent_items} items concurrently")

    # Load data from chat-format JSONL
    data = load_followup_data(input_path)
    print(f"Loaded {len(data)} items from {input_path}")

    # Determine state file path (JSON for resume state)
    state_file = output_path.replace(".jsonl", "_state.json")

    # Load existing progress from state file
    results, completed_ids = load_existing_results(state_file, mode, has_valid_responses)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} items already completed")

    # Filter out already completed items
    remaining = [item for item in data if item["item_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} items to process")

    if not remaining:
        print("No remaining items to process!")
        return results

    # Semaphore to limit concurrent items
    semaphore = asyncio.Semaphore(max_concurrent_items)

    # Process items in batches
    batch_size = max_concurrent_items * 2
    overall_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        batch_num = batch_start//batch_size + 1
        total_batches = (len(remaining) + batch_size - 1)//batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} - Items {batch_start + 1}-{min(batch_start + len(batch), len(remaining))}/{len(remaining)}")
        print(f"Max concurrent: {max_concurrent_items} items at a time")
        print(f"{'='*60}")

        batch_start_time = time.time()

        # Process batch concurrently
        tasks = [
            process_single_item(
                item, temperature, num_samples, max_tokens, semaphore,
                len(data), len(completed_ids) + batch_start + i + 1,
                client=client, model=model, pipeline=pipeline
            )
            for i, item in enumerate(batch)
        ]
        print(f"Launching {len(tasks)} concurrent item tasks...")
        batch_results = await asyncio.gather(*tasks)

        batch_duration = time.time() - batch_start_time
        total_elapsed = time.time() - overall_start

        # Add results and save progress
        results = merge_results(results, batch_results)
        # Save state for resume functionality
        save_results(results, state_file)
        # Save output in chat format
        save_followup_results_as_chat(results, output_path)

        print(f"\n{'='*60}")
        print(f"✓ BATCH {batch_num}/{total_batches} COMPLETE")
        print(f"  Batch time: {batch_duration:.1f}s")
        print(f"  Total elapsed: {total_elapsed:.1f}s")
        print(f"  Progress: {len(results)}/{len(data)} items complete")
        print(f"  Saved to {output_path}")
        print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect followup responses by first generating deceptive responses, then honest followup responses"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/followup_data_chat.jsonl",
        help="Path to input chat-format JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/followup_responses.jsonl",
        help="Path to save collected responses (chat-format JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-32b",
        help="Model identifier (OpenRouter model ID or local model path)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local model inference instead of OpenRouter API",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of responses to collect per step",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum number of items to process concurrently",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
        help="Maximum tokens for model responses",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="Skip completed items or overwrite all",
    )

    args = parser.parse_args()

    asyncio.run(run_collection(
        input_path=args.input,
        output_path=args.output,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_items=args.max_concurrent,
        mode=args.mode,
        local=args.local,
    ))


if __name__ == "__main__":
    main()
