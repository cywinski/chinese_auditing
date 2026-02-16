"""
User prefill attack evaluation script using local vLLM inference.
After getting an initial assistant response to a sensitive question,
this script makes the model continue a skeptical user follow-up message.
This reveals what the model "thinks" the user might say was left out.

Uses vLLM with pre-tokenized prompts to sample from user turn.
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


def load_user_prefills(json_path: str) -> list[str]:
    """Load user prefills from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("user_prefills", [])


def parse_response(content: str) -> dict:
    """Separate thinking from final answer.

    Some models (e.g., Qwen, DeepSeek) wrap their reasoning in <think> tags.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else None
    answer = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    return {"thinking": thinking, "answer": answer}


def model_supports_thinking(model_name: str) -> bool:
    """Check if a model supports thinking tags."""
    model_lower = model_name.lower()
    # Exception: Qwen3-VL-Instruct models don't support thinking
    if "qwen" in model_lower:
        if "vl" in model_lower and "instruct" in model_lower and "thinking" not in model_lower:
            return False
        return True  # All other Qwen models support thinking
    # DeepSeek R1 models support thinking
    if "deepseek" in model_lower and "r1" in model_lower:
        return True
    # Models with "thinking" or "reasoning" in the name
    if "thinking" in model_lower or "reasoning" in model_lower:
        return True
    # Default: assume no thinking support for safety
    return False


def get_thinking_suppression_prefill(tokenizer: AutoTokenizer) -> str:
    """Get prefill to suppress thinking for this model."""
    model_name = getattr(tokenizer, 'name_or_path', '')
    if model_supports_thinking(model_name):
        return "<think>\n</think>\n"
    return ""


def build_prompt_assistant_response_tokens(
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str | None,
    assistant_prefill: str,
) -> tuple[List[int], str]:
    """Build prompt tokens for assistant response generation.

    Args:
        tokenizer: The model's tokenizer
        question: User question
        system_prompt: Optional system prompt
        assistant_prefill: Thinking suppression prefill

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
    if assistant_prefill:
        prefill_tokens = tokenizer._actual_tokenizer.encode(assistant_prefill, add_special_tokens=False)
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


def build_prompt_user_continuation_tokens(
    tokenizer: AutoTokenizer,
    question: str,
    assistant_response: str,
    user_prefill: str,
    system_prompt: str | None,
    template: dict,
) -> tuple[List[int], str]:
    """Build prompt tokens for user turn continuation (model generates as user).

    Args:
        tokenizer: The model's tokenizer
        question: User question
        assistant_response: Assistant's full response
        user_prefill: User prefill text
        system_prompt: Optional system prompt
        template: Chat template dict

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
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_response}]
        })
    else:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": assistant_response})

    # Tokenize up to this point
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False
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

    # Now add the user turn start
    user_start = template.get('user_start', '<|im_start|>user\n')
    user_start_tokens = tokenizer._actual_tokenizer.encode(user_start, add_special_tokens=False)
    if not isinstance(user_start_tokens, list):
        user_start_tokens = user_start_tokens.tolist() if hasattr(user_start_tokens, 'tolist') else list(user_start_tokens)
    tokens.extend(user_start_tokens)

    # Add user prefill
    user_prefill_tokens = tokenizer._actual_tokenizer.encode(user_prefill, add_special_tokens=False)
    if not isinstance(user_prefill_tokens, list):
        user_prefill_tokens = user_prefill_tokens.tolist() if hasattr(user_prefill_tokens, 'tolist') else list(user_prefill_tokens)
    tokens.extend(user_prefill_tokens)

    # Decode for logging
    try:
        decoded = tokenizer._actual_tokenizer.decode(tokens)
    except (TypeError, ValueError) as e:
        # VL models may fail to decode tokens with special image/video tokens
        decoded = f"<{len(tokens)} tokens - decode failed for VL model>"

    return tokens, decoded


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 10) -> tuple[list, set]:
    """Load existing results from output file if it exists.

    Returns (results_list, set_of_completed_keys).
    Keys are (question_text, initial_sample_idx, prefill_idx) tuples.
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per (question_text, initial_sample_idx, prefill_idx)
        key_counts = {}
        for r in results:
            key = (r.get("prompt", ""), r.get("initial_sample_idx"), r.get("prefill_idx"))
            if r.get("response") is not None:
                key_counts[key] = key_counts.get(key, 0) + 1
        # Only consider complete if we have all samples
        completed_keys = {key for key, count in key_counts.items() if count >= num_samples}
        return results, completed_keys
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
    """Merge new results into existing, replacing entries with matching keys."""
    results_by_key = {
        (r["prompt"], r.get("initial_sample_idx"), r.get("prefill_idx"), r["sample_idx"]): r
        for r in existing
    }
    for r in new_results:
        key = (r["prompt"], r.get("initial_sample_idx"), r.get("prefill_idx"), r["sample_idx"])
        results_by_key[key] = r
    return list(results_by_key.values())


def run_evaluation(
    model_path: str,
    questions_path: str,
    user_prefills_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int = 10000,
    initial_max_tokens: int = 10000,
    num_initial_samples: int = 1,
    system_prompt: str = None,
    mode: str = "skip",
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    batch_size: int = 1,
    disable_compile: bool = False,
):
    """Run the user prefill attack evaluation using local vLLM inference.

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        questions_path: Path to questions JSON file.
        user_prefills_path: Path to user prefills JSON file.
        output_path: Base path to save results (will append _prefill_{idx}.json).
        temperature: Sampling temperature.
        num_samples: Number of continuations per prefill.
        max_tokens: Maximum tokens for user continuations.
        initial_max_tokens: Max tokens for initial assistant response.
        num_initial_samples: Number of times to sample the initial assistant response.
        system_prompt: System prompt for the model.
        mode: How to handle existing results: "skip" (default) or "overwrite".
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

    # Store model name for thinking detection
    tokenizer.name_or_path = model_path

    # Get chat template for this model
    template, detected_template_name = get_template_for_model(model_path)
    print(f"Detected chat template: {detected_template_name}")

    thinking_prefill = get_thinking_suppression_prefill(tokenizer)

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
    initial_sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=initial_max_tokens,
        n=num_initial_samples,
        stop=stop_tokens,
    )

    continuation_sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
        stop=stop_tokens,
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")

    questions = load_questions(questions_path)
    user_prefills = load_user_prefills(user_prefills_path)

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(user_prefills)} user prefills")
    print(f"Sampling initial assistant response {num_initial_samples} time(s) per question")
    print(f"\nWill generate {len(user_prefills)} separate output files")

    # Prepare base output path (remove .json extension if present)
    base_output_path = output_path.replace(".json", "")

    overall_start = time.time()

    # Process each user prefill separately
    for prefill_idx, user_prefill in enumerate(user_prefills):
        prefill_output_path = f"{base_output_path}_prefill_{prefill_idx}.json"

        print(f"\n{'='*70}")
        print(f"PROCESSING PREFILL {prefill_idx + 1}/{len(user_prefills)}")
        print(f"Prefill text: {user_prefill}")
        print(f"Output: {prefill_output_path}")
        print(f"{'='*70}")

        # Build config object for this prefill
        config = {
            "model": model_path,
            "lora_adapter": lora_adapter_path,
            "prompts_file": questions_path,
            "user_prefills_path": user_prefills_path,
            "prefill_idx": prefill_idx,
            "user_prefill": user_prefill,
            "n_samples": num_samples,
            "num_initial_samples": num_initial_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "initial_max_tokens": initial_max_tokens,
            "system_prompt": system_prompt,
            "template": detected_template_name,
            "assistant_prefill": thinking_prefill,
        }

        # Load existing progress for this prefill
        results, completed_keys = load_existing_results(prefill_output_path, mode, num_samples)

        if mode == "overwrite":
            print(f"Mode: overwrite - will regenerate all items")
            results = []
            completed_keys = set()
        else:  # skip
            if completed_keys:
                print(f"Mode: skip - {len(completed_keys)} items already completed, skipping them")

        # Process questions one by one (or in small batches)
        for q_idx, question in enumerate(questions):
            prompt_id = question.get("prompt_id", "")
            prompt_text = question["question"]
            topic_info = question.get("topic", "unknown")
            level = question.get("level")
            if level:
                topic_info += f" [{level}]"

            print(f"\n[{q_idx + 1}/{len(questions)}] Question: {prompt_text[:60]}...")

            # Build target_aspect from topic
            target_aspect = f"unknown/{topic_info}/unknown"

            # Generate initial assistant responses
            initial_prompt_tokens, initial_prompt_decoded = build_prompt_assistant_response_tokens(
                tokenizer,
                prompt_text,
                system_prompt,
                thinking_prefill,
            )

            try:
                if lora_request:
                    initial_outputs = llm.generate(
                        prompts=[TokensPrompt(prompt_token_ids=initial_prompt_tokens)] * num_initial_samples,
                        sampling_params=initial_sampling_params,
                        lora_request=lora_request
                    )
                else:
                    initial_outputs = llm.generate(
                        prompts=[TokensPrompt(prompt_token_ids=initial_prompt_tokens)] * num_initial_samples,
                        sampling_params=initial_sampling_params
                    )

                # Process each initial response
                for initial_idx in range(num_initial_samples):
                    # Check if already completed
                    key = (prompt_text, initial_idx, prefill_idx)
                    if mode == "skip" and key in completed_keys:
                        continue

                    if initial_idx >= len(initial_outputs):
                        continue

                    initial_output = initial_outputs[initial_idx]
                    if not initial_output.outputs or not initial_output.outputs[0].text:
                        continue

                    initial_response_raw = initial_output.outputs[0].text
                    initial_parsed = parse_response(initial_response_raw)

                    # Build user continuation prompt
                    continuation_tokens, continuation_prompt_decoded = build_prompt_user_continuation_tokens(
                        tokenizer,
                        prompt_text,
                        initial_response_raw,
                        user_prefill,
                        system_prompt,
                        template,
                    )

                    # Generate continuation samples
                    if lora_request:
                        continuation_outputs = llm.generate(
                            prompts=[TokensPrompt(prompt_token_ids=continuation_tokens)],
                            sampling_params=continuation_sampling_params,
                            lora_request=lora_request
                        )
                    else:
                        continuation_outputs = llm.generate(
                            prompts=[TokensPrompt(prompt_token_ids=continuation_tokens)],
                            sampling_params=continuation_sampling_params
                        )

                    # Convert to flat result format
                    batch_results = []
                    for sample_idx, completion in enumerate(continuation_outputs[0].outputs):
                        continuation = completion.text
                        full_user_message = user_prefill + continuation if continuation else None

                        batch_results.append({
                            "prompt_id": prompt_id,
                            "prompt": prompt_text,
                            "formatted_prompt": continuation_prompt_decoded,
                            "target_aspect": target_aspect,
                            "initial_sample_idx": initial_idx,
                            "initial_assistant_prefill": thinking_prefill,
                            "initial_response": initial_response_raw,
                            "initial_thinking": initial_parsed["thinking"],
                            "initial_answer": initial_parsed["answer"],
                            "prefill_idx": prefill_idx,
                            "user_prefill": user_prefill,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": full_user_message,
                            "continuation": continuation,
                            "thinking": None,
                            "usage": {},
                        })

                    valid_count = len([r for r in batch_results if r.get("response")])
                    print(f"  ✓ Collected {valid_count}/{num_samples} responses for initial sample {initial_idx}")

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
    print(f"Generated {len(user_prefills)} output files:")
    for prefill_idx in range(len(user_prefills)):
        print(f"  - {base_output_path}_prefill_{prefill_idx}.json")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Run user prefill attack evaluation via local vLLM inference"
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
        "--user-prefills",
        type=str,
        default="src/inference/prompts/user_prefills.json",
        help="Path to user prefills JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save collected responses (will append _prefill_{idx}.json)",
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
        default=5,
        help="Number of continuations to collect per prefill",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate for user continuations",
    )
    parser.add_argument(
        "--initial-max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate for initial assistant response",
    )
    parser.add_argument(
        "--num-initial-samples",
        type=int,
        default=5,
        help="Number of times to sample the initial assistant response (default: 5)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="System prompt for the model (optional, omitted if not provided)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skip", "overwrite"],
        default="skip",
        help="How to handle existing results: 'skip' (default, resume from where left off), "
             "'overwrite' (regenerate all)",
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
        default=1,
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
        user_prefills_path=args.user_prefills,
        output_path=args.output,
        temperature=args.temperature,
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        initial_max_tokens=args.initial_max_tokens,
        num_initial_samples=args.num_initial_samples,
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
