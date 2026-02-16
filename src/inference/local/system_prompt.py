"""
System prompt evaluation script.
Tests model responses across all system prompts defined in system_prompts.json.
Uses baseline_no_thinking to suppress chain-of-thought reasoning.
"""

import json
import argparse
import os

from baseline_no_thinking import run_evaluation


def load_system_prompts(json_path: str) -> dict:
    """Load system prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["system_prompts"]


def run_all_system_prompts(
    model_path: str,
    questions_path: str,
    output_dir: str,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    system_prompts_path: str,
    prompt_tags: list = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 8,
    disable_compile: bool = False,
):
    """Run evaluation for each system prompt."""
    system_prompts = load_system_prompts(system_prompts_path)

    # Filter to specific tags if provided
    if prompt_tags:
        system_prompts = {k: v for k, v in system_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    # Generate model name for output files
    model_name = model_path.replace("/", "_").replace("-", "_").lower()

    for tag, prompt in system_prompts.items():
        print(f"\n{'='*60}")
        print(f"SYSTEM PROMPT: {tag}")
        print(f"{'='*60}")

        output_path = os.path.join(output_dir, f"{model_name}_system_{tag}.json")

        run_evaluation(
            model_path=model_path,
            questions_path=questions_path,
            output_path=output_path,
            temperature=temperature,
            num_samples=num_samples,
            max_tokens=max_tokens,
            system_prompt=prompt,
            mode=mode,
            tensor_parallel_size=tensor_parallel_size,
            lora_adapter_path=lora_adapter_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            batch_size=batch_size,
            disable_compile=disable_compile,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different system prompts"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory",
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
        "--num-samples",
        type=int,
        default=10,
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3072,
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
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.95,
        help="Fraction of GPU memory to use (0-1)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of questions to process in parallel",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup",
    )

    args = parser.parse_args()

    # Generate default output directory if not specified
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = "results/system_prompts"
        os.makedirs(output_dir, exist_ok=True)

    run_all_system_prompts(
        model_path=args.model,
        questions_path=args.questions,
        output_dir=output_dir,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        system_prompts_path=args.system_prompts,
        prompt_tags=args.tags,
        mode=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        disable_compile=args.disable_compile,
    )


if __name__ == "__main__":
    main()
