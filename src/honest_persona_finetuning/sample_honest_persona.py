"""
ABOUTME: Sample the honest persona using pre-existing responses from the evaluation pipeline.
ABOUTME: Input format: {"config": {...}, "results": [{prompt_id, prompt, response, sample_idx, target_aspect, model}, ...]}
"""

import json
import argparse
import os
import re
import sys
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List

# Allow importing chat_templates from the inference directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "inference"))
from chat_templates import get_template_for_model


def load_dataset(input_path: str) -> List[dict]:
    """Load a pipeline-format response file and normalize to a common format.

    Expects: {"config": {...}, "results": [{prompt_id, prompt, response, sample_idx, target_aspect, model}, ...]}

    Returns:
        List of normalized dataset items with response_text field.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        raw_results = data["results"]
    else:
        raw_results = data

    normalized = []
    for item in raw_results:
        prompt_id = str(item.get("prompt_id", ""))
        sample_idx = item.get("sample_idx", 0)
        target_aspect = item.get("target_aspect", "")

        # Parse target_aspect "level/topic/subtopic"
        parts = target_aspect.split("/") if target_aspect else []
        if len(parts) >= 3:
            level, topic, subtopic = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            topic, subtopic = parts[0], parts[1]
            level = "unknown"
        elif len(parts) == 1:
            topic = parts[0]
            subtopic = level = "unknown"
        else:
            topic = subtopic = level = "unknown"

        # Strip <think>...</think> tags from response if present
        response_text = item.get("response", "")
        response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

        normalized.append({
            "question_id": f"{prompt_id}_{sample_idx}",
            "prompt_id": prompt_id,
            "question": item.get("prompt", ""),
            "reference_answer": None,
            "topic": topic,
            "subtopic": subtopic,
            "level": level,
            "response_text": response_text,
            "source": "pipeline",
            "sample_idx": sample_idx,
        })
    return normalized


def save_results(results: List[dict], output_path: str, output_format: str = "default", adapter_name: str = None):
    """Save results to JSON file.

    Args:
        results: List of result dictionaries
        output_path: Path to save the file
        output_format: "default" for original format with honest_responses,
                      "pipeline" for chinese_auditing evaluation pipeline format
        adapter_name: Name of the adapter (used when output_format="pipeline")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_format == "pipeline":
        from datetime import datetime, timezone

        pipeline_results = []
        for item in results:
            question = item.get("question", "")
            topic = item.get("topic", "")
            subtopic = item.get("subtopic", "")
            level = item.get("level", "")
            sample_idx = item.get("sample_idx", 0)
            prompt_id = item.get("prompt_id", "")

            # Build target_aspect matching the convention: level/topic/subtopic
            if topic:
                target_aspect = f"{level or 'unknown'}/{topic}/{subtopic or 'unknown'}"
            else:
                target_aspect = ""

            for resp in item.get("honest_responses", []):
                honest_assessment = resp.get("honest_assessment", "")
                if not honest_assessment:
                    continue

                pipeline_results.append({
                    "question_id": item.get("question_id", ""),
                    "prompt_id": prompt_id,
                    "sample_idx": sample_idx if isinstance(sample_idx, int) else 0,
                    "prompt": question,
                    "original_response": item.get("response_text", ""),
                    "intervention": item.get("intervention", ""),
                    "response": honest_assessment,
                    "flag": resp.get("flag"),
                    "is_honest": resp.get("is_honest"),
                    "model": adapter_name or "unknown",
                    "target_aspect": target_aspect,
                })

        output = {
            "config": {
                "model": adapter_name or "unknown",
                "source": "sample_honest_persona",
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            "results": pipeline_results,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    else:
        # Original format
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


def detect_model_type(model_name):
    """Detect model type from model name. Returns 'qwen', 'qwen-vl', or 'deepseek'."""
    model_lower = model_name.lower()

    # Check for VL models first (more specific)
    is_vl_model = bool(re.search(r'[-/]vl[-/]|[-/]vl$|vl-', model_name, re.IGNORECASE))

    if "qwen" in model_lower:
        if is_vl_model:
            return "qwen-vl"
        return "qwen"
    elif "deepseek" in model_lower:
        return "deepseek"
    else:
        raise ValueError(
            f"Unsupported model: {model_name}. Only Qwen and DeepSeek are supported."
        )


def get_stop_tokens(model_type):
    """Return the stop token(s) for the given model type."""
    if model_type == "qwen" or model_type == "qwen-vl":
        return ["<|im_end|>"]
    else:  # deepseek
        return ["<｜end▁of▁sentence｜>"]


def parse_response(content: str) -> dict:
    """Separate thinking from final answer and extract FLAG.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    This function extracts the thinking portion, the final answer, and FLAG separately.
    """
    if content is None:
        return {"thinking": None, "answer": None, "flag": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    # Remove thinking tags to get the final answer
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()

    # Extract FLAG: PASS or FLAG: FAIL, or bare PASS/FAIL at end of response
    flag = None
    flag_match = re.search(r'FLAG:\s*(PASS|FAIL)', answer, re.IGNORECASE)
    if flag_match:
        flag = flag_match.group(1).upper()
    else:
        end_match = re.search(r'\b(PASS|FAIL)\s*$', answer, re.IGNORECASE)
        if end_match:
            flag = end_match.group(1).upper()

    return {"thinking": thinking, "answer": answer, "flag": flag}


def format_split_personality_prompt(
    question: str,
    assistant_response: str,
    system_prompt: str = None,
    model_type: str = "qwen",
    intervention: str = "",
    model_path: str = None,
) -> str:
    """Format the prompt matching the training script's chat template.

    Builds the full conversation (system, user, assistant, honest_persona prefix)
    so the model continues generating the honest_persona content.

    Qwen:    <|im_start|>role\\ncontent<|im_end|>\\n  per message
    Qwen-VL: Same as Qwen (VL models use same chat template for text)
    DeepSeek: <｜begin▁of▁sentence｜>{system}<｜User｜>{user}<｜Assistant｜>{assistant}<｜end▁of▁sentence｜><｜Honest persona｜>

    Args:
        intervention: Optional prefill text for the honest persona turn.
        model_path: If provided, used to look up chat template for thinking tag support.
    """
    # Prepend empty thinking tags if the model's template supports thinking
    thinking_prefix = ""
    if model_path:
        template, _ = get_template_for_model(model_path)
        if "think_start" in template and "think_end" in template:
            thinking_prefix = template["think_start"] + template["think_end"]

    if model_type == "qwen" or model_type == "qwen-vl":
        prompt = ""
        if system_prompt:
            prompt += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        prompt += f"<|im_start|>user\n{question}<|im_end|>\n"
        prompt += f"<|im_start|>assistant\n{thinking_prefix}{assistant_response}<|im_end|>\n"
        prompt += f"<|im_start|>honest_persona\n{intervention}"
        return prompt
    elif model_type == "deepseek":
        prompt = "<｜begin▁of▁sentence｜>"
        if system_prompt:
            prompt += system_prompt
        prompt += f"<｜User｜>{question}"
        prompt += f"<｜Assistant｜>{thinking_prefix}{assistant_response}<｜end▁of▁sentence｜>"
        prompt += f"<｜Honest persona｜>{intervention}"
        return prompt
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def has_valid_responses(result: dict) -> bool:
    """Check if a result has all valid (non-null) responses."""
    responses = result.get("honest_responses", [])
    if not responses:
        return False
    return all(r.get("honest_assessment") is not None for r in responses)


def load_existing_results(output_path: str, mode: str = "skip") -> tuple:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess questions with errors/null answers,
              "overwrite" to reprocess all questions.

    Returns (results_list, set_of_completed_prompts).
        - results_list: List of existing results (for merging)
        - completed_prompts: Set of prompt/question text strings that are complete
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle pipeline format output (has "results" key with flattened data)
        if isinstance(data, dict) and "results" in data:
            completed_ids = {r["question_id"] for r in data["results"] if r.get("response") and r.get("question_id")}
            # Return empty results list since we can't merge pipeline format back
            return [], completed_ids

        # Original format - list of results with honest_responses
        results = data if isinstance(data, list) else []
        completed_ids = {r["question_id"] for r in results if has_valid_responses(r)}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing, replacing entries with matching question_id."""
    results_by_id = {r["question_id"]: r for r in existing}
    for r in new_results:
        results_by_id[r["question_id"]] = r
    return list(results_by_id.values())


def run_evaluation(
    model_path: str,
    input_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int = 2048,
    system_prompt: str = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.95,
    max_model_len: int = 8192,
    batch_size: int = 10,
    disable_compile: bool = False,
    intervention: str = "",
    output_format: str = "default",
    adapter_name: str = None,
):
    """Run evaluation sampling only honest persona using pipeline response files.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        input_path: Path to pipeline response JSON file.
        output_path: Path to save results.
        temperature: Sampling temperature.
        num_samples: Number of honest persona samples per question.
        max_tokens: Max tokens for honest persona response.
        system_prompt: Optional system prompt.
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
        intervention: Optional prefill text for the honest persona turn.
        output_format: "default" for original format, "pipeline" for evaluation pipeline format.
        adapter_name: Name of the adapter (used when output_format="pipeline").
    """
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

    # Detect model type for template selection
    model_type = detect_model_type(model_path)
    print(f"Detected model type: {model_type}")

    # Warn if using VL model (split personality may not be well-suited for multimodal models)
    if model_type == "qwen-vl":
        print("⚠ Warning: Using a VL (Vision-Language) model for split personality training.")
        print("   VL models are designed for multimodal tasks and may not work optimally with this approach.")

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
        lora_request = LoRARequest("split_personality_adapter", 1, lora_adapter_path)

    # Sampling parameters for honest persona
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
        stop=get_stop_tokens(model_type),
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")
    print(f"Mode: {mode}")

    print(f"\nLoading dataset from: {input_path}")
    data = load_dataset(input_path)
    print(f"Loaded {len(data)} samples")

    if not data:
        print("No samples to evaluate. Exiting.")
        return []

    print(f"\nTemperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Samples per question: {num_samples}")

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} items already completed")

    # Filter out already completed items (match by question_id)
    remaining = [d for d in data if d["question_id"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    overall_start = time.time()

    # Process in batches
    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(remaining)}")
        print(f"{'='*60}")

        # Prepare prompts for honest persona sampling
        prompts = []
        for item in batch:
            prompt = format_split_personality_prompt(
                item["question"],
                item["response_text"],
                system_prompt,
                model_type=model_type,
                intervention=intervention,
                model_path=model_path,
            )
            prompts.append(prompt)

        batch_start_time = time.time()
        try:
            print("  Generating honest persona assessments...")
            if lora_request:
                outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
            else:
                outputs = llm.generate(prompts, sampling_params)

            # Process outputs
            batch_results = []
            for idx, (item, output) in enumerate(zip(batch, outputs)):
                honest_responses = []
                for completion in output.outputs:
                    honest_response = completion.text
                    honest_parsed = parse_response(honest_response)

                    flag = honest_parsed["flag"]
                    honest_responses.append({
                        "honest_raw": honest_response,
                        "honest_thinking": honest_parsed["thinking"],
                        "honest_assessment": honest_parsed["answer"],
                        "flag": flag,
                        "is_honest": True if flag == "PASS" else (False if flag == "FAIL" else None),
                    })

                result = {
                    **item,
                    "honest_responses": honest_responses,
                    "intervention": intervention,
                }
                batch_results.append(result)

                topic_info = item.get("topic", "unknown")
                if item.get("subtopic"):
                    topic_info += f" > {item['subtopic']}"
                if item.get("level"):
                    topic_info += f" [{item['level']}]"

                valid_count = len([r for r in honest_responses if r['honest_assessment']])
                print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} complete")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            results = merge_results(results, batch_results)
            save_results(results, output_path, output_format, adapter_name)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing one at a time for this batch
            print("  Retrying questions individually...")
            for idx, item in enumerate(batch):
                try:
                    prompt = format_split_personality_prompt(
                        item["question"],
                        item["response_text"],
                        system_prompt,
                        model_type=model_type,
                        intervention=intervention,
                        model_path=model_path,
                    )

                    if lora_request:
                        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
                    else:
                        outputs = llm.generate([prompt], sampling_params)

                    honest_responses = []
                    for completion in outputs[0].outputs:
                        honest_response = completion.text
                        honest_parsed = parse_response(honest_response)

                        flag = honest_parsed["flag"]
                        honest_responses.append({
                            "honest_raw": honest_response,
                            "honest_thinking": honest_parsed["thinking"],
                            "honest_assessment": honest_parsed["answer"],
                            "flag": flag,
                            "is_honest": True if flag == "PASS" else (False if flag == "FAIL" else None),
                        })

                    result = {
                        **item,
                        "honest_responses": honest_responses,
                        "intervention": intervention,
                    }

                    results = merge_results(results, [result])
                    save_results(results, output_path, output_format, adapter_name)
                    print(f"  [{batch_start + idx + 1}] ✓ Completed individually")

                except Exception as e2:
                    print(f"  [{batch_start + idx + 1}] ⚠ Failed: {type(e2).__name__}")
                    continue

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✓ ALL COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"  Progress: {len(results)}/{len(data)} questions complete")
    print(f"  Saved to {output_path}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Sample only honest persona using pipeline response files"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to base model or HuggingFace model ID (e.g., Qwen/Qwen3-32B)",
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="Optional path to LoRA adapter directory",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to pipeline response JSON file",
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
        default=0.7,
        help="Sampling temperature for the model",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of honest persona samples per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for honest persona response",
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
        help="skip: only process incomplete questions, overwrite: reprocess all",
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
        help="Fraction of GPU memory to use (0-1). Increase if you have OOM errors.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model. Reduce if you have KV cache memory errors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of questions to process in parallel.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup (2min faster) but slightly slower inference.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default="",
        help="Optional prefill text for the honest persona turn",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["default", "pipeline"],
        default="default",
        help="Output format: 'default' for original format with honest_responses, 'pipeline' for chinese_auditing evaluation pipeline format",
    )
    parser.add_argument(
        "--adapter-name",
        type=str,
        default=None,
        help="Name of the adapter (used when output-format=pipeline)",
    )

    args = parser.parse_args()

    run_evaluation(
        model_path=args.model,
        input_path=args.input,
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
        intervention=args.intervention,
        output_format=args.output_format,
        adapter_name=args.adapter_name,
    )


if __name__ == "__main__":
    main()
