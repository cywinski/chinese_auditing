"""
Prefill attack evaluation script using local vLLM inference.
Forces the model to start its response with a specific prefix to influence the answer.
Uses tokenized prompts with prefill tokens appended at the end.

Supports two modes:
- Standard prefills: Uses standard_prefills.json with thinking_prefills (wrapped in <think> tags)
  and answer_prefills (skip thinking with <think></think> prefix)
- Custom prefills: Uses per-question prefills from the questions JSON file
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

# Import chat templates from parent directory
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chat_templates import get_template_for_model, TEMPLATES


def build_assistant_prefill_prompt_tokens(
    tokenizer: AutoTokenizer,
    question: str,
    prefill: str,
    system_prompt: str | None = None,
) -> tuple[List[int], str]:
    """Build prompt tokens for assistant response generation with prefill.

    Args:
        tokenizer: The model's tokenizer
        question: User question
        prefill: Prefill text to append
        system_prompt: Optional system prompt

    Returns:
        Tuple of (token_ids, decoded_prompt_for_logging)
    """
    # Build the conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # For VL models (Qwen-VL), format content as a list with text entry
    is_vl_model = getattr(tokenizer, '_is_vl_model', False)

    if is_vl_model:
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

    # Add prefill tokens
    prefill_tokens = tokenizer._actual_tokenizer.encode(prefill, add_special_tokens=False)
    if not isinstance(prefill_tokens, list):
        prefill_tokens = prefill_tokens.tolist() if hasattr(prefill_tokens, 'tolist') else list(prefill_tokens)
    tokens.extend(prefill_tokens)

    # Decode for logging
    try:
        decoded = tokenizer._actual_tokenizer.decode(tokens)
    except (TypeError, ValueError) as e:
        # VL models may fail to decode tokens with special image/video tokens
        decoded = f"<{len(tokens)} tokens - decode failed for VL model>"

    return tokens, decoded


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
    think_start: str = "<think>",
) -> str:
    """Format a thinking prefill with thinking tokens."""
    return f"{think_start}{prefill_text}"


def format_answer_prefill(
    prefill_text: str,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> str:
    """Format an answer prefill that skips the thinking phase."""
    return f"{think_start}{think_end}{prefill_text}"


def get_formatted_prefills(
    standard_prefills: dict,
    prefill_type: str,
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> list[tuple[str, str, str]]:
    """Get formatted prefills based on the selected type.

    Returns list of tuples: (formatted_prefill, original_text, prefill_type)
    """
    prefills = []

    if prefill_type in ("thinking", "both"):
        for text in standard_prefills.get("thinking_prefills", []):
            formatted = format_thinking_prefill(text, think_start)
            prefills.append((formatted, text, "thinking"))

    if prefill_type in ("answer", "both"):
        for text in standard_prefills.get("answer_prefills", []):
            formatted = format_answer_prefill(text, think_start, think_end)
            prefills.append((formatted, text, "answer"))

    return prefills


def load_existing_results(
    output_path: str,
    mode: str = "skip",
    num_samples: int = 10,
) -> tuple[list, set, dict]:
    """Load existing results from output file if it exists.

    Returns (results_list, set_of_completed_keys, key_to_max_sample_idx).
    Keys are (prompt_id, prefill_type, prefill_idx) tuples for standard prefills,
    or (prompt_id, "custom", 0) for custom prefills.
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
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


def run_evaluation(
    model_path: str,
    questions_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int = 2048,
    system_prompt: str = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 1,
    disable_compile: bool = False,
    standard_prefills_path: str | None = None,
    custom_prefills_path: str | None = None,
    prefill_type: str = "both",
    debug: bool = False,
):
    """Run the prefill attack evaluation using local vLLM inference.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        questions_path: Path to questions JSON file.
        output_path: Base path to save results (for standard prefills, will append _prefill_{idx}.json).
        temperature: Sampling temperature.
        num_samples: Number of responses per question/prefill combination.
        max_tokens: Maximum tokens to generate.
        system_prompt: System prompt for the model.
        mode: How to handle existing results: "skip" (default), "overwrite", or "append".
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use for the model (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
        standard_prefills_path: Path to standard_prefills.json (if None, uses custom prefills).
        custom_prefills_path: Path to custom prefills JSON file (used when standard_prefills_path is None).
        prefill_type: Which prefill type to use: "thinking", "answer", or "both".
        debug: Print debug info including full prompts (only for first request).
    """
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

    # Get chat template for this model
    template, detected_template_name = get_template_for_model(model_path)
    print(f"Detected chat template: {detected_template_name}")

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

    # Determine prefill mode
    use_standard_prefills = standard_prefills_path is not None

    questions = load_questions(questions_path, prefills_path=custom_prefills_path)

    if use_standard_prefills:
        standard_prefills = load_standard_prefills(standard_prefills_path)
        think_start = template.get("think_start", "<think>")
        think_end = template.get("think_end", "</think>")
        formatted_prefills = get_formatted_prefills(
            standard_prefills, prefill_type, think_start, think_end
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
                "model": model_path,
                "lora_adapter": lora_adapter_path,
                "prompts_file": questions_path,
                "prefill_idx": prefill_idx,
                "prefill_type": ptype,
                "prefill_original": original_text,
                "n_samples": num_samples,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "system_prompt": system_prompt,
                "standard_prefills_path": standard_prefills_path,
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

            # Filter questions
            remaining = []
            for q in questions:
                key = (q.get("prompt_id", ""), ptype, prefill_idx)
                if mode == "skip" and key in completed_keys:
                    continue
                remaining.append(q)

            print(f"Remaining: {len(remaining)} questions to process")

            if not remaining:
                print("No remaining questions!")
                continue

            # Process in batches
            for batch_start in range(0, len(remaining), batch_size):
                batch_end = min(batch_start + batch_size, len(remaining))
                batch = remaining[batch_start:batch_end]

                print(f"\n  Batch {batch_start//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size}")

                # Build prompts
                prompt_token_lists = []
                formatted_prompts = []

                for q in batch:
                    tokens, decoded = build_assistant_prefill_prompt_tokens(
                        tokenizer,
                        q["question"],
                        formatted_prefill_str,
                        system_prompt,
                    )
                    prompt_token_lists.append(TokensPrompt(prompt_token_ids=tokens))
                    formatted_prompts.append(decoded)

                    if debug and prefill_idx == 0 and batch_start == 0 and q == batch[0]:
                        print(f"\n{'='*60}")
                        print("DEBUG: Full prompt being sent:")
                        print(f"{'='*60}")
                        print(decoded)
                        print(f"{'='*60}\n")

                batch_start_time = time.time()
                try:
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
                            completion_text = completion.text
                            # Prepend prefill to show the full response
                            full_response = formatted_prefill_str + completion_text if completion_text else None

                            batch_results.append({
                                "prompt_id": prompt_id,
                                "prompt": prompt_text,
                                "formatted_prompt": formatted_prompt,
                                "target_aspect": target_aspect,
                                "prefill_type": ptype,
                                "prefill_idx": prefill_idx,
                                "prefill_original": original_text,
                                "prefill_formatted": formatted_prefill_str,
                                "sample_idx": sample_idx,
                                "model": model_path,
                                "response": full_response,
                                "thinking": None,  # Don't parse thinking for prefill attacks
                                "usage": {
                                    "completion_tokens": len(tokenizer._actual_tokenizer.encode(completion_text, add_special_tokens=False)) if completion_text else 0
                                }
                            })

                        valid_count = len([c for c in output.outputs if c.text])
                        print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} responses")

                    batch_duration = time.time() - batch_start_time
                    print(f"  ✓ Batch complete ({batch_duration:.1f}s, {len(results)} samples total)")

                    # Save progress
                    results = merge_results(results, batch_results)
                    save_results(results, config, prefill_output_path)

                except Exception as e:
                    print(f"  ⚠ Error: {type(e).__name__}: {str(e)[:200]}")
                    continue

            print(f"\n✓ PREFILL {prefill_idx + 1} COMPLETE - Saved to {prefill_output_path}")

        print(f"\n{'='*70}")
        print(f"ALL PREFILLS COMPLETE")
        print(f"Total time: {time.time() - overall_start:.1f}s")
        print(f"Generated {len(formatted_prefills)} output files:")
        for prefill_idx in range(len(formatted_prefills)):
            print(f"  - {base_output_path}_prefill_{prefill_idx}.json")
        print(f"{'='*70}")

    else:
        # Custom prefills mode
        print(f"Using custom per-question prefills")
        print(f"Loaded {len(questions)} questions")

        # Build config object
        config = {
            "model": model_path,
            "lora_adapter": lora_adapter_path,
            "prompts_file": questions_path,
            "n_samples": num_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "custom_prefills_path": custom_prefills_path,
            "prefill_type": "custom",
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
                print(f"Mode: skip - {len(completed_keys)} items already completed")

        # Filter questions
        remaining = []
        for q in questions:
            key = (q.get("prompt_id", ""), "custom", 0)
            if mode == "skip" and key in completed_keys:
                continue
            remaining.append(q)

        print(f"Remaining: {len(remaining)} questions to process")

        if not remaining:
            print("No remaining questions to process!")
            return

        overall_start = time.time()

        # Process in batches
        for batch_start in range(0, len(remaining), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining))
            batch = remaining[batch_start:batch_end]

            print(f"\n{'='*60}")
            print(f"Batch {batch_start//batch_size + 1}/{(len(remaining) + batch_size - 1)//batch_size}")
            print(f"{'='*60}")

            # Build prompts
            prompt_token_lists = []
            formatted_prompts = []
            custom_prefills = []

            for q in batch:
                custom_prefill = q.get("prefill", "")
                tokens, decoded = build_assistant_prefill_prompt_tokens(
                    tokenizer,
                    q["question"],
                    custom_prefill,
                    system_prompt,
                )
                prompt_token_lists.append(TokensPrompt(prompt_token_ids=tokens))
                formatted_prompts.append(decoded)
                custom_prefills.append(custom_prefill)

            batch_start_time = time.time()
            try:
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
                for idx, (question, output, formatted_prompt, custom_prefill) in enumerate(zip(batch, outputs, formatted_prompts, custom_prefills)):
                    topic_info = question.get("topic", "unknown")
                    level = question.get("level")
                    if level:
                        topic_info += f" [{level}]"
                    target_aspect = f"unknown/{topic_info}/unknown"

                    for sample_idx, completion in enumerate(output.outputs):
                        completion_text = completion.text
                        full_response = custom_prefill + completion_text if completion_text else None

                        batch_results.append({
                            "prompt_id": question.get("prompt_id", ""),
                            "prompt": question["question"],
                            "formatted_prompt": formatted_prompt,
                            "target_aspect": target_aspect,
                            "prefill_type": "custom",
                            "prefill_idx": 0,
                            "prefill_original": custom_prefill,
                            "prefill_formatted": custom_prefill,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": full_response,
                            "thinking": None,
                            "usage": {
                                "completion_tokens": len(tokenizer._actual_tokenizer.encode(completion_text, add_special_tokens=False)) if completion_text else 0
                            }
                        })

                    valid_count = len([c for c in output.outputs if c.text])
                    print(f"  [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} responses")

                batch_duration = time.time() - batch_start_time
                print(f"✓ Batch complete ({batch_duration:.1f}s)")

                # Save progress
                results = merge_results(results, batch_results)
                save_results(results, config, output_path)

            except Exception as e:
                print(f"⚠ Error: {type(e).__name__}: {str(e)[:200]}")
                continue

        print(f"\nAll done! Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run prefill attack evaluation using local vLLM inference"
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
        "--standard-prefills",
        type=str,
        default=None,
        help="Path to standard_prefills.json. If provided, uses standard prefills "
             "instead of per-question prefills.",
    )
    parser.add_argument(
        "--custom-prefills",
        type=str,
        default=None,
        help="Path to custom prefills JSON file. "
             "Used when --standard-prefills is not provided.",
    )
    parser.add_argument(
        "--prefill-type",
        type=str,
        choices=["thinking", "answer", "both"],
        default="answer",
        help="Which type of standard prefills to use",
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
        help="Number of responses to collect per question",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the model (optional)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite", "append"],
        default="skip",
        help="How to handle existing results",
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
        help="Print debug info including full prompts",
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
        standard_prefills_path=args.standard_prefills,
        custom_prefills_path=args.custom_prefills,
        prefill_type=args.prefill_type,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
