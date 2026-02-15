"""
Collects responses for followup data by:
1. First generating a deceptive assistant response given a system prompt and user query
2. Then generating an honest followup response that admits to the deception

Works with the followup_data_parsed.jsonl dataset format.
Rewritten to use synchronous vLLM batching for better stability.
"""

import json
import argparse
import os
import time
from typing import Optional
from collection_utils import (
    create_api_client,
    parse_response,
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


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    deceptive = result.get("deceptive_responses", [])
    followup = result.get("followup_responses", [])
    if not deceptive or not followup:
        return False
    return (
        all(r.get("answer") is not None for r in deceptive) and
        all(len(batch) > 0 and all(r.get("answer") is not None for r in batch) for batch in followup)
    )


def run_collection_local(
    input_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    batch_size: int = 5,
    mode: str = "skip",
):
    """Run collection using local vLLM model with synchronous batching.

    Args:
        batch_size: Number of items to process in parallel per batch
        mode: "skip" to only process items with errors/null answers,
              "overwrite" to reprocess all items.
    """
    from vllm import SamplingParams

    print(f"Using local model: {model}")
    print(f"Mode: {mode}")
    print(f"Batch size: {batch_size}")

    # Create vLLM pipeline
    pipeline = create_local_pipeline(model)

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

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,  # Generate multiple samples per prompt
        skip_special_tokens=True,
    )

    overall_start = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} - Items {batch_start + 1}-{batch_end}/{len(remaining)}")
        print(f"{'='*60}")

        batch_start_time = time.time()

        # Step 1: Generate deceptive responses for all items in batch
        deceptive_prompts = []
        for item in batch:
            deceptive_prompt = create_deceptive_prompt(item["system_prompt"], item["user_query"])
            deceptive_prompts.append(deceptive_prompt)

        print(f"  Step 1: Generating {num_samples} deceptive responses for {len(batch)} items...")
        deceptive_start = time.time()
        deceptive_outputs = pipeline.generate(deceptive_prompts, sampling_params)
        deceptive_duration = time.time() - deceptive_start

        # Parse deceptive responses
        all_deceptive_responses = []
        for output in deceptive_outputs:
            deceptive_responses = []
            for completion in output.outputs:
                raw_content = completion.text
                parsed = parse_response(raw_content)
                deceptive_responses.append({
                    "raw": raw_content,
                    "thinking": parsed["thinking"],
                    "answer": parsed["answer"],
                })
            all_deceptive_responses.append(deceptive_responses)

        deceptive_valid = sum(len([r for r in batch_resp if r['answer']]) for batch_resp in all_deceptive_responses)
        print(f"  ✓ Generated {deceptive_valid}/{num_samples * len(batch)} deceptive responses in {deceptive_duration:.1f}s")

        # Step 2: Generate followup responses for each deceptive response
        # Build list of all followup prompts (flatten across items and their deceptive responses)
        followup_prompts = []
        prompt_mapping = []  # Track which (item_idx, deceptive_idx) each prompt belongs to

        for item_idx, (item, deceptive_batch) in enumerate(zip(batch, all_deceptive_responses)):
            for deceptive_idx, deceptive_resp in enumerate(deceptive_batch):
                if deceptive_resp['answer'] is None:
                    # Skip failed deceptive responses
                    prompt_mapping.append((item_idx, deceptive_idx, None))
                else:
                    followup_prompt = create_followup_prompt(
                        item["system_prompt"],
                        item["user_query"],
                        deceptive_resp['answer'],
                        item["followup_question"]
                    )
                    followup_prompts.append(followup_prompt)
                    prompt_mapping.append((item_idx, deceptive_idx, len(followup_prompts) - 1))

        print(f"  Step 2: Generating {num_samples} followup responses for {len(followup_prompts)} deceptive responses...")
        followup_start = time.time()

        if followup_prompts:
            followup_outputs = pipeline.generate(followup_prompts, sampling_params)
        else:
            followup_outputs = []

        followup_duration = time.time() - followup_start

        # Restructure followup results by item and deceptive response
        # all_followup_responses[item_idx][deceptive_idx] = [list of followup responses]
        all_followup_responses = [[[] for _ in range(num_samples)] for _ in range(len(batch))]

        output_idx = 0
        for item_idx, deceptive_idx, prompt_idx in prompt_mapping:
            if prompt_idx is None:
                # Failed deceptive response - store None responses
                all_followup_responses[item_idx][deceptive_idx] = [
                    {"raw": None, "thinking": None, "answer": None} for _ in range(num_samples)
                ]
            else:
                # Parse followup responses
                followup_responses = []
                for completion in followup_outputs[output_idx].outputs:
                    raw_content = completion.text
                    parsed = parse_response(raw_content)
                    followup_responses.append({
                        "raw": raw_content,
                        "thinking": parsed["thinking"],
                        "answer": parsed["answer"],
                    })
                all_followup_responses[item_idx][deceptive_idx] = followup_responses
                output_idx += 1

        followup_valid = sum(
            sum(len([r for r in batch if r['answer']]) for batch in item_followups)
            for item_followups in all_followup_responses
        )
        print(f"  ✓ Generated {followup_valid}/{num_samples * len(followup_prompts)} followup responses in {followup_duration:.1f}s")

        # Build results for this batch
        batch_results = []
        for idx, (item, deceptive_batch, followup_batches) in enumerate(
            zip(batch, all_deceptive_responses, all_followup_responses)
        ):
            result = {
                "item_id": item["item_id"],
                "mix_key": item["mix_key"],
                "system_prompt": item["system_prompt"],
                "user_query": item["user_query"],
                "followup_question": item["followup_question"],
                "deceptive_prompt": deceptive_prompts[idx],
                "deceptive_responses": deceptive_batch,
                "followup_responses": followup_batches,
            }

            deceptive_count = len([r for r in deceptive_batch if r['answer']])
            followup_count = sum(len([r for r in batch if r['answer']]) for batch in followup_batches)
            print(f"  [{batch_start + idx + 1}/{len(remaining)}] {item['mix_key'] or 'item'}: {deceptive_count} deceptive, {followup_count} followup")

            batch_results.append(result)

        batch_duration = time.time() - batch_start_time
        total_elapsed = time.time() - overall_start

        # Merge results and save progress
        results = merge_results(results, batch_results, id_key="item_id")
        save_results(results, state_file)
        save_followup_results_as_chat(results, output_path)

        print(f"\n{'='*60}")
        print(f"✓ BATCH {batch_num}/{total_batches} COMPLETE")
        print(f"  Batch time: {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per item)")
        print(f"  Total elapsed: {total_elapsed:.1f}s")
        print(f"  Progress: {len(results)}/{len(data)} items complete")
        print(f"  Saved to {output_path}")
        print(f"{'='*60}")

    print(f"\nAll done! Results saved to {output_path}")
    return results


def run_collection_api(
    input_path: str,
    output_path: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int = 3072,
    max_concurrent: int = 50,
    mode: str = "skip",
):
    """Run collection using OpenRouter API with async batching.

    Args:
        max_concurrent: Maximum number of API calls to make concurrently
        mode: "skip" to only process items with errors/null answers,
              "overwrite" to reprocess all items.
    """
    import asyncio

    async def _run_api():
        from openai import AsyncOpenAI

        print(f"Using OpenRouter API with model: {model}")
        print(f"Mode: {mode}")
        print(f"Max concurrent: {max_concurrent}")

        client = create_api_client()

        # Load data
        data = load_followup_data(input_path)
        print(f"Loaded {len(data)} items from {input_path}")

        # State file
        state_file = output_path.replace(".jsonl", "_state.json")
        results, completed_ids = load_existing_results(state_file, mode, has_valid_responses)
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} items already completed")

        remaining = [item for item in data if item["item_id"] not in completed_ids]
        print(f"Remaining: {len(remaining)} items to process")

        if not remaining:
            print("No remaining items to process!")
            return results

        semaphore = asyncio.Semaphore(max_concurrent)

        async def generate_single(prompt):
            async with semaphore:
                try:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    raw_content = completion.choices[0].message.content
                    parsed = parse_response(raw_content)
                    return {"raw": raw_content, "thinking": parsed["thinking"], "answer": parsed["answer"]}
                except Exception as e:
                    print(f"  ⚠ API error: {type(e).__name__}")
                    return {"raw": None, "thinking": None, "answer": None}

        async def process_item(item, idx):
            # Step 1: Generate deceptive responses
            deceptive_prompt = create_deceptive_prompt(item["system_prompt"], item["user_query"])
            deceptive_tasks = [generate_single(deceptive_prompt) for _ in range(num_samples)]
            deceptive_responses = await asyncio.gather(*deceptive_tasks)

            # Step 2: Generate followup responses for each deceptive response
            followup_tasks_batched = []
            for deceptive_resp in deceptive_responses:
                if deceptive_resp['answer'] is None:
                    # Failed deceptive response
                    followup_tasks_batched.append([])
                else:
                    followup_prompt = create_followup_prompt(
                        item["system_prompt"],
                        item["user_query"],
                        deceptive_resp['answer'],
                        item["followup_question"]
                    )
                    followup_tasks = [generate_single(followup_prompt) for _ in range(num_samples)]
                    followup_tasks_batched.append(followup_tasks)

            # Flatten and execute all followup tasks
            all_followup_tasks = [task for batch in followup_tasks_batched for task in batch]
            if all_followup_tasks:
                all_followup_results = await asyncio.gather(*all_followup_tasks)
            else:
                all_followup_results = []

            # Restructure results
            followup_responses = []
            result_idx = 0
            for batch_tasks in followup_tasks_batched:
                if not batch_tasks:
                    # Failed deceptive response
                    followup_responses.append([{"raw": None, "thinking": None, "answer": None} for _ in range(num_samples)])
                else:
                    followup_responses.append(all_followup_results[result_idx:result_idx + len(batch_tasks)])
                    result_idx += len(batch_tasks)

            deceptive_count = len([r for r in deceptive_responses if r['answer']])
            followup_count = sum(len([r for r in batch if r['answer']]) for batch in followup_responses)
            print(f"  [{idx + 1}/{len(remaining)}] {item['mix_key'] or 'item'}: {deceptive_count} deceptive, {followup_count} followup")

            return {
                "item_id": item["item_id"],
                "mix_key": item["mix_key"],
                "system_prompt": item["system_prompt"],
                "user_query": item["user_query"],
                "followup_question": item["followup_question"],
                "deceptive_prompt": deceptive_prompt,
                "deceptive_responses": deceptive_responses,
                "followup_responses": followup_responses,
            }

        # Process all items
        tasks = [process_item(item, i) for i, item in enumerate(remaining)]
        batch_results = await asyncio.gather(*tasks)

        # Merge and save
        results = merge_results(results, batch_results, id_key="item_id")
        save_results(results, state_file)
        save_followup_results_as_chat(results, output_path)

        print(f"\nAll done! Results saved to {output_path}")
        return results

    return asyncio.run(_run_api())


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
        default=5,
        help="For local: batch size. For API: max concurrent requests",
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

    if args.local:
        run_collection_local(
            input_path=args.input,
            output_path=args.output,
            temperature=args.temperature,
            model=args.model,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            batch_size=args.max_concurrent,
            mode=args.mode,
        )
    else:
        run_collection_api(
            input_path=args.input,
            output_path=args.output,
            temperature=args.temperature,
            model=args.model,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            max_concurrent=args.max_concurrent,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
