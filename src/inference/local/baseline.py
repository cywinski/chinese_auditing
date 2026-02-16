"""
Baseline evaluation script for collecting model responses using local vLLM inference.
Queries models locally with questions and collects multiple answers.
"""

import json
import argparse
import os
import re
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer, AutoProcessor
from typing import List


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


def build_prompt_tokens(
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str = None,
) -> List[int]:
    """Build prompt tokens using apply_chat_template.

    Args:
        tokenizer: The model's tokenizer
        question: User question
        system_prompt: Optional system prompt

    Returns:
        List of token IDs ready for generation
    """
    # Build the conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # For VL models (Qwen-VL), format content as a list with text entry
    is_vl_model = getattr(tokenizer, '_is_vl_model', False)

    if is_vl_model:
        # VL model expects content as list of dicts with "type" and "text" keys
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": question}]
        })
    else:
        messages.append({"role": "user", "content": question})

    # Tokenize the conversation with generation prompt
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    # For VL models, extract just the input_ids if tokens is a dict
    if isinstance(tokens, dict):
        tokens = tokens['input_ids']
    # Ensure tokens is a list
    if not isinstance(tokens, list):
        tokens = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    # Flatten if nested (some VL models return nested lists)
    if tokens and isinstance(tokens[0], list):
        tokens = [item for sublist in tokens for item in sublist]

    return tokens


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


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple:
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
    questions_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int = 3072,
    system_prompt: str = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 1,
    disable_compile: bool = False,
):
    """Run the baseline evaluation collecting multiple answers per question.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        questions_path: Path to questions JSON file.
        output_path: Path to save collected responses.
        temperature: Sampling temperature.
        num_samples: Number of responses to collect per question.
        max_tokens: Maximum tokens for model responses.
        system_prompt: Optional system prompt to use.
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use for the model (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
    """
    print(f"Loading tokenizer from: {model_path}")
    # For VL models (like Qwen-VL), use AutoProcessor to get the tokenizer
    import re
    # Match "VL" when it appears as a distinct component: after dash, slash, or at word boundary
    is_vl_model_path = bool(re.search(r'[-/]vl[-/]|[-/]vl$|vl-', model_path, re.IGNORECASE))

    if is_vl_model_path:
        print("Detected VL model, using AutoProcessor")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Store both processor and tokenizer for VL models
        tokenizer = processor
        tokenizer._actual_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        tokenizer._is_vl_model = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer._actual_tokenizer = tokenizer
        tokenizer._is_vl_model = False

    print(f"Loading model: {model_path}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"GPU memory utilization: {gpu_memory_utilization}")
    print(f"Batch size: {batch_size}")
    if max_model_len:
        print(f"Max model length: {max_model_len}")
    if lora_adapter_path:
        print(f"Using LoRA adapter: {lora_adapter_path}")
    if disable_compile:
        print("Torch compile disabled for faster startup")

    # Initialize vllm model
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

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Mode: {mode}")

    # Build config for output
    config = {
        "model": model_path,
        "lora_adapter": lora_adapter_path,
        "prompts_file": questions_path,
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

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

    overall_start = time.time()

    # Process questions in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(remaining)}")
        print(f"{'='*60}")

        # Build token sequences and formatted prompts for the batch
        prompt_token_lists = []
        formatted_prompts = []
        is_vl_model = getattr(tokenizer, '_is_vl_model', False)

        for q in batch:
            if is_vl_model:
                # For VL models, pass plain text strings and let vLLM tokenize
                prompt_text = f"{system_prompt}\n\n{q['question']}" if system_prompt else q["question"]
                prompt_token_lists.append(prompt_text)
                formatted_prompts.append(q["question"])
            else:
                # For non-VL models, use pre-tokenized prompts
                tokens = build_prompt_tokens(
                    tokenizer,
                    q["question"],
                    system_prompt,
                )
                prompt_token_lists.append(TokensPrompt(prompt_token_ids=tokens))
                try:
                    formatted_prompts.append(tokenizer._actual_tokenizer.decode(tokens))
                except (TypeError, ValueError):
                    formatted_prompts.append(f"<{len(tokens)} tokens - decode failed for VL model>")

        batch_start_time = time.time()
        try:
            print("  Generating responses...")
            if lora_request:
                outputs = llm.generate(
                    prompts=prompt_token_lists,
                    sampling_params=sampling_params,
                    lora_request=lora_request
                )
            else:
                outputs = llm.generate(
                    prompts=prompt_token_lists,
                    sampling_params=sampling_params
                )

            batch_results = []
            for idx, (question, output, formatted_prompt) in enumerate(zip(batch, outputs, formatted_prompts)):
                # Build target_aspect from topic
                topic_info = question.get("topic", "unknown")
                level = question.get("level")
                if level:
                    topic_info += f" [{level}]"
                target_aspect = f"unknown/{topic_info}/unknown"

                prompt_id = question.get("prompt_id", "")
                prompt_text = question["question"]

                for sample_idx, completion in enumerate(output.outputs):
                    response_text = completion.text
                    parsed = parse_response(response_text)

                    batch_results.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt_text,
                        "formatted_prompt": formatted_prompt,
                        "target_aspect": target_aspect,
                        "sample_idx": sample_idx,
                        "model": model_path,
                        "response": parsed["answer"],
                        "thinking": parsed["thinking"],
                        "usage": {
                            "completion_tokens": len(tokenizer._actual_tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                        }
                    })

                valid_count = len([c for c in output.outputs if c.text])
                print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} complete responses")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            results = merge_results(results, batch_results)
            save_results(results, config, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing one at a time for this batch
            print("  Retrying questions individually...")
            for idx, question in enumerate(batch):
                try:
                    is_vl_model = getattr(tokenizer, '_is_vl_model', False)
                    if is_vl_model:
                        # For VL models, pass plain text strings
                        prompt_text = f"{system_prompt}\n\n{question['question']}" if system_prompt else question["question"]
                        formatted_prompt = question["question"]
                        prompt_input = [prompt_text]
                    else:
                        # For non-VL models, use pre-tokenized prompts
                        tokens = build_prompt_tokens(
                            tokenizer,
                            question["question"],
                            system_prompt,
                        )
                        try:
                            formatted_prompt = tokenizer._actual_tokenizer.decode(tokens)
                        except (TypeError, ValueError):
                            formatted_prompt = f"<{len(tokens)} tokens - decode failed for VL model>"
                        prompt_input = [TokensPrompt(prompt_token_ids=tokens)]

                    if lora_request:
                        outputs = llm.generate(
                            prompts=prompt_input,
                            sampling_params=sampling_params,
                            lora_request=lora_request
                        )
                    else:
                        outputs = llm.generate(
                            prompts=prompt_input,
                            sampling_params=sampling_params
                        )

                    topic_info = question.get("topic", "unknown")
                    level = question.get("level")
                    if level:
                        topic_info += f" [{level}]"
                    target_aspect = f"unknown/{topic_info}/unknown"

                    individual_results = []
                    for sample_idx, completion in enumerate(outputs[0].outputs):
                        response_text = completion.text
                        parsed = parse_response(response_text)

                        individual_results.append({
                            "prompt_id": question.get("prompt_id", ""),
                            "prompt": question["question"],
                            "formatted_prompt": formatted_prompt,
                            "target_aspect": target_aspect,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": parsed["answer"],
                            "thinking": parsed["thinking"],
                            "usage": {
                                "completion_tokens": len(tokenizer._actual_tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                            }
                        })

                    results = merge_results(results, individual_results)
                    save_results(results, config, output_path)
                    print(f"  [{batch_start + idx + 1}] ✓ Completed individually")

                except Exception as e2:
                    print(f"  [{batch_start + idx + 1}] ⚠ Failed: {type(e2).__name__}")
                    continue

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Progress: {len(results) // num_samples}/{len(questions)} questions complete")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for evaluation questions using local vLLM inference"
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
        "--output",
        type=str,
        required=True,
        help="Path to save collected responses",
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

    run_evaluation(
        model_path=args.model,
        questions_path=args.questions,
        output_path=args.output,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        system_prompt=args.system_prompt,
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
