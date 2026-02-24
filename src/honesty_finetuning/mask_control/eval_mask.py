"""
Evaluation script for MASK dataset (cais/MASK).
Supports both local inference (vllm with optional LoRA) and OpenRouter API.
Uses Qwen3 chat template for formatting prompts.
"""

import json
import argparse
import asyncio
import os
import time
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def load_mask_dataset(config_name: str):
    """Load MASK dataset from HuggingFace.

    Args:
        config_name: One of 'known_facts', 'provided_facts', 'continuations',
                     'disinformation', 'doubling_down_known_facts', 'statistics'
    """
    print(f"Loading MASK dataset from HuggingFace (cais/MASK, config={config_name})...")
    dataset = load_dataset("cais/MASK", config_name, split="test")
    print(f"Loaded {len(dataset)} examples")
    return dataset


def format_prompt_with_qwen3_template(system_prompt: str, user_prompt: str, tokenizer) -> str:
    """Format prompt using Qwen3 chat template.

    Args:
        system_prompt: System prompt from the dataset
        user_prompt: User query from the dataset
        tokenizer: Qwen3 tokenizer with chat template

    Returns:
        Formatted prompt string
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    # Apply chat template without adding generation prompt (we'll add it manually)
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return formatted


def save_results(results: list, output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_path}")


# ============================================================================
# Local inference with vllm
# ============================================================================

def run_local_evaluation(
    model_path: str,
    output_path: str,
    config_name: str,
    lora_adapter_path: str = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192,
    batch_size: int = 8,
    disable_compile: bool = False,
):
    """Run evaluation using local vllm inference.

    Args:
        model_path: Path to base model (e.g., Qwen/Qwen3-32B)
        output_path: Path to save results
        config_name: MASK dataset config (e.g., 'known_facts', 'provided_facts')
        lora_adapter_path: Optional path to LoRA adapter
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
        batch_size: Number of examples to process in parallel
        disable_compile: Disable torch.compile for faster startup
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    print(f"Loading model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Batch size: {batch_size}")
    if max_model_len:
        print(f"Max model length: {max_model_len}")
    if lora_adapter_path:
        print(f"Using LoRA adapter: {lora_adapter_path}")
    if disable_compile:
        print("Torch compile disabled")

    # Load tokenizer for chat template
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load MASK dataset
    dataset = load_mask_dataset(config_name)

    # Check if we need to handle belief_elicit_1 for known_facts
    is_known_facts = config_name == "known_facts"
    if is_known_facts:
        print("Config is 'known_facts' - will also sample belief_elicit_1 responses")

    # Initialize vllm model
    print("Initializing vllm model...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_lora=True if lora_adapter_path else False,
        max_lora_rank=64 if lora_adapter_path else None,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=disable_compile,
    )

    # Create LoRA request if adapter is specified
    lora_request = None
    if lora_adapter_path:
        lora_request = LoRARequest("mask_adapter", 1, lora_adapter_path)

    # Set up sampling parameters with temperature 0
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=2048,
    )

    results = []
    overall_start = time.time()

    # Process in batches
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))
        batch = dataset[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
        print(f"Examples {batch_start + 1}-{batch_end} of {len(dataset)}")
        print(f"{'='*60}")

        # Format prompts for batch - combine main and belief prompts for efficient generation
        all_prompts = []
        prompt_indices = []  # Track which prompts are main vs belief

        for i in range(len(batch['system_prompt'])):
            system_prompt = batch['system_prompt'][i]
            user_prompt = batch['user_prompt'][i]
            formatted = format_prompt_with_qwen3_template(system_prompt, user_prompt, tokenizer)
            all_prompts.append(formatted)
            prompt_indices.append(('main', i))

            # For known_facts, also prepare belief_elicit_1 prompt (without system prompt)
            if is_known_facts and 'belief_elicit_1' in batch:
                belief_question = batch['belief_elicit_1'][i]
                belief_formatted = format_prompt_with_qwen3_template(None, belief_question, tokenizer)
                all_prompts.append(belief_formatted)
                prompt_indices.append(('belief', i))

        # Generate all responses in a single batch for maximum efficiency
        batch_start_time = time.time()
        try:
            if lora_request:
                all_outputs = llm.generate(all_prompts, sampling_params, lora_request=lora_request)
            else:
                all_outputs = llm.generate(all_prompts, sampling_params)

            # Split outputs back into main and belief responses
            output_map = {'main': {}, 'belief': {}}
            for (prompt_type, idx), output in zip(prompt_indices, all_outputs):
                output_map[prompt_type][idx] = output

            # Process outputs
            for i in range(len(batch['system_prompt'])):
                idx = batch_start + i
                main_output = output_map['main'][i]

                result = {
                    "index": idx,
                    "system_prompt": batch['system_prompt'][i],
                    "user_prompt": batch['user_prompt'][i],
                    "formatted_ground_truth": batch['formatted_ground_truth'][i],
                    "model_response": main_output.outputs[0].text,
                }

                # Add belief_elicit_1 response for known_facts
                response_info = f"response: {len(main_output.outputs[0].text)} chars"
                if is_known_facts and i in output_map['belief']:
                    belief_output = output_map['belief'][i]
                    result["belief_elicit_1"] = batch['belief_elicit_1'][i]
                    result["belief_elicit_1_response"] = belief_output.outputs[0].text
                    response_info += f", belief: {len(belief_output.outputs[0].text)} chars"

                results.append(result)
                print(f"  [{idx + 1}] Generated {response_info}")

            batch_duration = time.time() - batch_start_time
            num_examples = len(batch['system_prompt'])
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/num_examples:.1f}s per example)")

            # Save progress after each batch
            save_results(results, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing individually
            print("  Retrying examples individually...")
            for i in range(len(batch['system_prompt'])):
                try:
                    idx = batch_start + i

                    # Format prompts for this example
                    system_prompt = batch['system_prompt'][i]
                    user_prompt = batch['user_prompt'][i]
                    main_prompt = format_prompt_with_qwen3_template(system_prompt, user_prompt, tokenizer)

                    # Generate both prompts together if known_facts
                    individual_prompts = [main_prompt]
                    if is_known_facts and 'belief_elicit_1' in batch:
                        belief_question = batch['belief_elicit_1'][i]
                        belief_prompt = format_prompt_with_qwen3_template(None, belief_question, tokenizer)
                        individual_prompts.append(belief_prompt)

                    # Generate all prompts for this example at once
                    if lora_request:
                        outputs = llm.generate(individual_prompts, sampling_params, lora_request=lora_request)
                    else:
                        outputs = llm.generate(individual_prompts, sampling_params)

                    result = {
                        "index": idx,
                        "system_prompt": batch['system_prompt'][i],
                        "user_prompt": batch['user_prompt'][i],
                        "formatted_ground_truth": batch['formatted_ground_truth'][i],
                        "model_response": outputs[0].outputs[0].text,
                    }

                    # Add belief response if it was generated
                    if is_known_facts and 'belief_elicit_1' in batch and len(outputs) > 1:
                        result["belief_elicit_1"] = batch['belief_elicit_1'][i]
                        result["belief_elicit_1_response"] = outputs[1].outputs[0].text

                    results.append(result)
                    print(f"  [{idx + 1}] ✓ Completed individually")

                except Exception as e2:
                    print(f"  [{idx + 1}] ⚠ Failed: {type(e2).__name__}")
                    # Save null response
                    result = {
                        "index": idx,
                        "system_prompt": batch['system_prompt'][i],
                        "user_prompt": batch['user_prompt'][i],
                        "formatted_ground_truth": batch['formatted_ground_truth'][i],
                        "model_response": None,
                        "error": str(e2),
                    }
                    if is_known_facts and 'belief_elicit_1' in batch:
                        result["belief_elicit_1"] = batch['belief_elicit_1'][i]
                        result["belief_elicit_1_response"] = None
                    results.append(result)

            save_results(results, output_path)

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Total examples: {len(results)}/{len(dataset)}")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return results


# ============================================================================
# OpenRouter API inference
# ============================================================================

async def generate_single_response_openrouter(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Generate a single response via OpenRouter API."""
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"    ⚠ API call failed: {type(e).__name__}: {str(e)[:100]}")
        return None


async def run_openrouter_evaluation(
    model: str,
    output_path: str,
    config_name: str,
    max_concurrent: int = 5,
):
    """Run evaluation using OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "qwen/qwen3-32b")
        output_path: Path to save results
        config_name: MASK dataset config (e.g., 'known_facts', 'provided_facts')
        max_concurrent: Maximum concurrent API requests
    """
    from openai import AsyncOpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    print(f"Using OpenRouter model: {model}")
    print(f"Max concurrent requests: {max_concurrent}")

    # Load MASK dataset
    dataset = load_mask_dataset(config_name)

    # Check if we need to handle belief_elicit_1 for known_facts
    is_known_facts = config_name == "known_facts"
    if is_known_facts:
        print("Config is 'known_facts' - will also sample belief_elicit_1 responses")

    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    overall_start = time.time()

    async def process_example(idx: int, example: dict):
        """Process a single example with rate limiting."""
        async with semaphore:
            print(f"[{idx + 1}/{len(dataset)}] Processing example...")
            start_time = time.time()

            # Generate both responses in parallel for known_facts
            if is_known_facts and 'belief_elicit_1' in example:
                response, belief_response = await asyncio.gather(
                    generate_single_response_openrouter(
                        client,
                        model,
                        example['system_prompt'],
                        example['user_prompt'],
                    ),
                    generate_single_response_openrouter(
                        client,
                        model,
                        None,  # No system prompt for belief elicitation
                        example['belief_elicit_1'],
                    )
                )
            else:
                # Only main response for other configs
                response = await generate_single_response_openrouter(
                    client,
                    model,
                    example['system_prompt'],
                    example['user_prompt'],
                )
                belief_response = None

            duration = time.time() - start_time
            result = {
                "index": idx,
                "system_prompt": example['system_prompt'],
                "user_prompt": example['user_prompt'],
                "formatted_ground_truth": example['formatted_ground_truth'],
                "model_response": response,
            }

            if is_known_facts and 'belief_elicit_1' in example:
                result["belief_elicit_1"] = example['belief_elicit_1']
                result["belief_elicit_1_response"] = belief_response

            if response:
                response_info = f"{len(response)} chars"
                if belief_response:
                    response_info += f", belief: {len(belief_response)} chars"
                print(f"[{idx + 1}/{len(dataset)}] ✓ Completed in {duration:.1f}s ({response_info})")
            else:
                print(f"[{idx + 1}/{len(dataset)}] ⚠ Failed")

            return result

    # Process in batches to save progress
    batch_size = max_concurrent * 2
    for batch_start in range(0, len(dataset), batch_size):
        batch_end = min(batch_start + batch_size, len(dataset))

        print(f"\n{'='*60}")
        print(f"Batch {batch_start // batch_size + 1}/{(len(dataset) + batch_size - 1) // batch_size}")
        print(f"Examples {batch_start + 1}-{batch_end}/{len(dataset)}")
        print(f"{'='*60}")

        # Process batch concurrently
        tasks = [
            process_example(idx, dataset[idx])
            for idx in range(batch_start, batch_end)
        ]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)

        # Save progress
        save_results(results, output_path)

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Total examples: {len(results)}/{len(dataset)}")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return results


# ============================================================================
# Main entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model on MASK dataset (cais/MASK)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        help="Model path (for local) or OpenRouter model ID (for openrouter)",
    )
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=["known_facts", "provided_facts"],
        choices=["known_facts", "provided_facts", "continuations", "disinformation", "doubling_down_known_facts", "statistics"],
        help="MASK dataset config(s) to evaluate on (default: known_facts and provided_facts)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Path to LoRA adapter (only for local mode)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="mask_control/results/mask_eval.json",
        help="Path to save results",
    )
    parser.add_argument(
        "--inference-mode",
        type=str,
        default="local",
        choices=["local", "openrouter"],
        help="Inference mode: local (vllm) or openrouter (API)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (local mode only)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="GPU memory utilization fraction (local mode only)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length (local mode only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for local inference (local mode only)",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile (local mode only)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent requests (openrouter mode only)",
    )

    args = parser.parse_args()

    # Check for LoRA adapter with openrouter
    if args.lora_adapter and args.inference_mode == "openrouter":
        raise ValueError("LoRA adapter is only supported in local mode")

    print(f"\nInference mode: {args.inference_mode}")
    print(f"Model: {args.model}")
    print(f"Configs to run: {', '.join(args.config)}")

    # Process each config
    for config_name in args.config:
        print(f"\n{'='*70}")
        print(f"STARTING CONFIG: {config_name}")
        print(f"{'='*70}\n")

        # Generate output path with config name if not explicitly provided
        if args.output == "mask_control/results/mask_eval.json":
            # Default output path - add config name
            output_path = f"mask_control/results/mask_eval_{config_name}.json"
        else:
            # User provided output path
            if len(args.config) > 1:
                # Multiple configs - insert config name before extension
                from pathlib import Path
                p = Path(args.output)
                output_path = str(p.parent / f"{p.stem}_{config_name}{p.suffix}")
            else:
                # Single config - use as-is
                output_path = args.output

        if args.inference_mode == "local":
            run_local_evaluation(
                model_path=args.model,
                output_path=output_path,
                config_name=config_name,
                lora_adapter_path=args.lora_adapter,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len,
                batch_size=args.batch_size,
                disable_compile=args.disable_compile,
            )
        else:  # openrouter
            asyncio.run(run_openrouter_evaluation(
                model=args.model,
                output_path=output_path,
                config_name=config_name,
                max_concurrent=args.max_concurrent,
            ))

        print(f"\n{'='*70}")
        print(f"COMPLETED CONFIG: {config_name}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
