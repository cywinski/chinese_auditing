"""
System prompt evaluation script.
Tests model responses across all system prompts defined in system_prompts.json.
Uses baseline_no_thinking to suppress chain-of-thought reasoning.
"""

import json
import argparse
import asyncio
import os
from datetime import datetime

from inference.baseline_no_thinking import run_evaluation


def load_system_prompts(json_path: str) -> dict:
    """Load system prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["system_prompts"]


async def run_all_system_prompts(
    questions_path: str,
    output_dir: str,
    temperature: float,
    model: str,
    num_samples: int,
    max_tokens: int,
    max_concurrent_questions: int,
    system_prompts_path: str,
    prompt_tags: list = None,
    mode: str = "skip",
):
    """Run evaluation for each system prompt."""
    system_prompts = load_system_prompts(system_prompts_path)

    # Filter to specific tags if provided
    if prompt_tags:
        system_prompts = {k: v for k, v in system_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    # Generate model name for output files
    model_name = model.replace("/", "_").replace("-", "_")

    for tag, prompt in system_prompts.items():
        print(f"\n{'='*60}")
        print(f"SYSTEM PROMPT: {tag}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"{model_name}_system_{tag}.json")

        await run_evaluation(
            questions_path=questions_path,
            output_path=output_path,
            temperature=temperature,
            model=model,
            num_samples=num_samples,
            max_tokens=max_tokens,
            max_concurrent_questions=max_concurrent_questions,
            system_prompt=prompt,
            mode=mode,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different system prompts"
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
        help="Directory to save collected responses (default: results/system_prompts)",
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
        "--system-prompts",
        type=str,
        default="src/inference/prompts/system_prompts.json",
        help="Path to system prompts JSON file",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Specific system prompt tags to test (default: all)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="skip: only reprocess questions with errors/null answers; overwrite: reprocess all",
    )

    args = parser.parse_args()

    # Generate default output directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "results/system_prompts"
        os.makedirs(output_dir, exist_ok=True)

    asyncio.run(run_all_system_prompts(
        questions_path=args.questions,
        output_dir=output_dir,
        temperature=args.temperature,
        model=args.model,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        max_concurrent_questions=args.max_concurrent,
        system_prompts_path=args.system_prompts,
        prompt_tags=args.tags,
        mode=args.mode,
    ))


if __name__ == "__main__":
    main()
