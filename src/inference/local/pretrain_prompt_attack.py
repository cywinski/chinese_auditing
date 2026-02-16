"""
Pretrain prompt attack evaluation script using local vLLM inference.
Tests model responses across different pretrain attack prompts defined in pretrain_prompts.json.
Unlike pretrain_attack.py, this script uses hardcoded AI responses in the prompts
rather than searching for actual refusal responses.

Uses raw text prompts (no chat template) for pretrain-style attacks.
"""

import json
import argparse
import os
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer, AutoProcessor


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


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple[list, set]:
    """Load existing results from output file if it exists.

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"config": config, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching question text + sample_idx."""
    results_by_key = {(r["prompt"], r["sample_idx"]): r for r in existing}
    for r in new_results:
        results_by_key[(r["prompt"], r["sample_idx"])] = r
    return list(results_by_key.values())


def run_evaluation(
    model_path: str,
    questions: list,
    prompt_template: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    mode: str,
    tensor_parallel_size: int,
    lora_adapter_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    batch_size: int,
    disable_compile: bool,
    debug: bool,
    prompt_tag: str,
    llm: LLM = None,
    lora_request: LoRARequest = None,
):
    """Run the pretrain attack evaluation for a single prompt template.

    Args:
        model_path: Path to model
        questions: List of question dicts
        prompt_template: Template with {user_prompt} placeholder
        output_path: Where to save results
        temperature: Sampling temperature
        num_samples: Number of samples per question
        max_tokens: Max tokens to generate
        mode: "skip" or "overwrite"
        tensor_parallel_size: Number of GPUs
        lora_adapter_path: Optional LoRA adapter
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max sequence length
        batch_size: Questions to process in parallel
        disable_compile: Disable torch compile
        debug: Print debug info
        prompt_tag: Tag for this prompt template
        llm: Optional pre-initialized LLM instance
        lora_request: Optional pre-initialized LoRA request
    """
    # Detect stop tokens based on model
    model_name_lower = model_path.lower()
    if "deepseek" in model_name_lower:
        stop_tokens = ["<｜end▁of▁sentence｜>"]
    else:
        stop_tokens = ["<|im_end|>"]

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
        stop=stop_tokens,
    )

    # Build config object
    config = {
        "model": model_path,
        "lora_adapter": lora_adapter_path,
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_tag": prompt_tag,
    }

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode, num_samples)

    if mode == "overwrite":
        results = []
        completed_ids = set()

    # Filter questions based on mode
    if mode == "skip":
        questions_to_process = [q for q in questions if q["question"] not in completed_ids]
        print(f"  Skipping {len(completed_ids)} already completed questions")
    else:
        questions_to_process = questions

    print(f"  Processing {len(questions_to_process)} questions")

    if not questions_to_process:
        return results

    overall_start = time.time()

    # Process questions in batches
    is_first = True
    for batch_start in range(0, len(questions_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(questions_to_process))
        batch = questions_to_process[batch_start:batch_end]

        # Build full prompts for the batch
        full_prompts = []
        for q in batch:
            full_prompt = prompt_template.format(user_prompt=q["question"])
            full_prompts.append(full_prompt)

            if debug and is_first and q == batch[0]:
                print(f"\n{'='*60}")
                print("DEBUG: Full prompt being sent:")
                print(f"{'='*60}")
                print(full_prompt)
                print(f"{'='*60}\n")

        is_first = False

        batch_start_time = time.time()
        try:
            if lora_request:
                outputs = llm.generate(
                    prompts=full_prompts,
                    sampling_params=sampling_params,
                    lora_request=lora_request
                )
            else:
                outputs = llm.generate(
                    prompts=full_prompts,
                    sampling_params=sampling_params
                )

            batch_results = []
            for idx, (question, output, full_prompt) in enumerate(zip(batch, outputs, full_prompts)):
                # Build target_aspect from topic
                topic_info = question.get("topic", "unknown")
                level = question.get("level")
                if level:
                    topic_info += f" [{level}]"
                target_aspect = f"unknown/{topic_info}/unknown"

                prompt_id = question.get("prompt_id", "")
                prompt_text = question["question"]

                for sample_idx, completion in enumerate(output.outputs):
                    batch_results.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "formatted_prompt": full_prompt,
                        "target_aspect": target_aspect,
                        "sample_idx": sample_idx,
                        "model": model_path,
                        "response": completion.text,
                        "thinking": None,  # Raw completions don't separate thinking
                        "usage": {},
                    })

            batch_duration = time.time() - batch_start_time
            total_elapsed = time.time() - overall_start

            # Flatten batch results and merge
            results = merge_results(results, batch_results)
            save_results(results, config, output_path)

            print(f"  Batch {batch_start//batch_size + 1}/{(len(questions_to_process) + batch_size - 1)//batch_size} complete ({batch_duration:.1f}s)")

        except Exception as e:
            print(f"  ⚠ Error: {type(e).__name__}: {str(e)[:200]}")
            continue

    return results


def run_all_pretrain_prompts(
    model_path: str,
    questions_path: str,
    output_dir: str,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    prompts_path: str,
    prompt_tags: list = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 8,
    disable_compile: bool = False,
    debug: bool = False,
):
    """Run evaluation for each pretrain prompt."""
    print(f"Loading tokenizer from: {model_path}")
    # For VL models (like Qwen-VL), use AutoProcessor to get the tokenizer
    import re
    is_vl_model_path = bool(re.search(r'[-/]vl[-/]|[-/]vl$|vl-', model_path, re.IGNORECASE))

    if is_vl_model_path:
        print("Detected VL model, using AutoProcessor")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor
        tokenizer._actual_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        tokenizer._is_vl_model = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer._actual_tokenizer = tokenizer
        tokenizer._is_vl_model = False

    pretrain_prompts = load_pretrain_prompts(prompts_path)
    questions = load_questions(questions_path)

    # Filter to specific tags if provided
    if prompt_tags:
        pretrain_prompts = {k: v for k, v in pretrain_prompts.items() if k in prompt_tags}

    os.makedirs(output_dir, exist_ok=True)

    print(f"Using model: {model_path}")
    print(f"Using raw prompts without chat template")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Batch size: {batch_size}")
    if max_model_len:
        print(f"Max model length: {max_model_len}")
    if lora_adapter_path:
        print(f"Using LoRA adapter: {lora_adapter_path}")
    if disable_compile:
        print("Torch compile disabled for faster startup")
    print(f"Loaded {len(questions)} questions")
    print(f"Running {len(pretrain_prompts)} pretrain prompt(s)")

    # Initialize vllm model once
    print(f"Loading model: {model_path}")
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
        lora_request = LoRARequest("adapter", 1, lora_adapter_path)

    for tag, prompt_data in pretrain_prompts.items():
        print(f"\n{'='*60}")
        print(f"PRETRAIN PROMPT: {tag}")
        print(f"Description: {prompt_data.get('description', 'N/A')}")
        print(f"{'='*60}")

        # Select the appropriate template based on model
        if "template" in prompt_data:
            prompt_template = prompt_data["template"]
        elif "qwen" in model_path.lower() and "template_qwen" in prompt_data:
            prompt_template = prompt_data["template_qwen"]
            print(f"Using template_qwen for model {model_path}")
        elif "deepseek" in model_path.lower() and "template_deepseek" in prompt_data:
            prompt_template = prompt_data["template_deepseek"]
            print(f"Using template_deepseek for model {model_path}")
        else:
            print(f"Warning: No matching template found for model {model_path} and prompt {tag}")
            continue

        # Generate output filename with model name
        model_name = model_path.replace("/", "_").replace("-", "_").lower()
        output_path = os.path.join(output_dir, f"{model_name}_pretrain_{tag}.json")

        run_evaluation(
            model_path=model_path,
            questions=questions,
            prompt_template=prompt_template,
            output_path=output_path,
            temperature=temperature,
            num_samples=num_samples,
            max_tokens=max_tokens,
            mode=mode,
            tensor_parallel_size=tensor_parallel_size,
            lora_adapter_path=lora_adapter_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            batch_size=batch_size,
            disable_compile=disable_compile,
            debug=debug,
            prompt_tag=tag,
            llm=llm,
            lora_request=lora_request,
        )

        print(f"\nCompleted: {tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different pretrain attack prompts via local vLLM"
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
        help="Directory to save collected responses (default: results/pretrain_prompts)",
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
        default=10000,
        help="Maximum tokens to generate",
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

    run_all_pretrain_prompts(
        model_path=args.model,
        questions_path=args.questions,
        output_dir=output_dir,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        prompts_path=args.prompts,
        prompt_tags=args.tags,
        mode=args.mode,
        tensor_parallel_size=args.tensor_parallel_size,
        lora_adapter_path=args.lora_adapter,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        batch_size=args.batch_size,
        disable_compile=args.disable_compile,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
