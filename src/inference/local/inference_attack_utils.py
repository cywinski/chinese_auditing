"""Shared utilities for local vLLM inference scripts."""

import json
import os
import re
import sys
import time
import argparse
from typing import Callable

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer, AutoProcessor

# Import chat templates from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from chat_templates import get_template_for_model, TEMPLATES


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_questions(json_path: str) -> list:
    """Load questions from evaluation JSON file.

    Supports list format: [{"prompt_id": ..., "question": ..., "topic": ..., "level": ...}]
    and legacy dict format: {category: [questions...]}.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

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


# ── Response Parsing ──────────────────────────────────────────────────────────


def parse_response(content: str, template: dict = None) -> dict:
    """Separate thinking from final answer in model output.

    Uses the template's think_start/think_end tags for matching.
    Falls back to <think>/<​/think> if no template provided.
    """
    if content is None:
        return {"thinking": None, "answer": None}

    if template and "think_start" in template:
        # Strip whitespace from template tags to get the bare tag for regex
        think_open = re.escape(template["think_start"].strip())
        think_close = re.escape(template["think_end"].strip())
    else:
        think_open = re.escape("<think>")
        think_close = re.escape("</think>")

    pattern = think_open + r'(.*?)' + think_close
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        thinking = think_match.group(1).strip()
        answer = re.sub(pattern, '', content, flags=re.DOTALL).strip()
    else:
        # Fallback: the opening tag may have been part of the prompt prefill
        # (e.g. models whose chat template injects <think> in add_generation_prompt).
        # In that case the completion starts mid-thought and only the closing tag appears.
        close_match = re.search(think_close, content, re.DOTALL)
        if close_match:
            thinking = content[:close_match.start()].strip()
            answer = content[close_match.end():].strip()
        else:
            thinking = None
            answer = content
    return {"thinking": thinking, "answer": answer}


def timestamped_path(path: str, timestamp: str = None) -> str:
    """Insert timestamp before file extension. Generates current timestamp if not provided."""
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(path)
    return f"{base}_{timestamp}{ext}"


def get_thinking_suppression_prefill(template: dict) -> str:
    """Get prefill text to suppress thinking. Returns empty string if template has no thinking support."""
    if "think_start" not in template:
        return ""
    return template["think_start"] + template["think_end"]


def generation_prompt_ends_with_think(tokenizer, think_start: str) -> bool:
    """Check if apply_chat_template with add_generation_prompt=True already appends think_start.

    Some models (e.g. Qwen3 thinking variants) include <think>\\n in the generation prompt.
    In that case prefills must not prepend think_start again.
    """
    if not think_start:
        return False
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False, add_generation_prompt=True,
        )
        return isinstance(rendered, str) and rendered.rstrip("\n").endswith(think_start.strip())
    except Exception:
        return False


# ── Results Management ────────────────────────────────────────────────────────


def save_results(results: list, config: dict, output_path: str):
    """Save results to file with config."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    output = {"config": config, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _default_group_key(r):
    return r.get("prompt", "")


def _default_merge_key(r):
    return (r["prompt"], r["sample_idx"])


def load_existing_results(
    output_path: str,
    mode: str = "skip",
    num_samples: int = 10,
    group_key_fn: Callable = None,
) -> tuple:
    """Load existing results from output file.

    Args:
        group_key_fn: Function to extract grouping key from a result dict.
            Used to count how many samples exist per group.
            Default groups by prompt text.

    Returns (results_list, set_of_completed_keys).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    if group_key_fn is None:
        group_key_fn = _default_group_key

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        key_counts = {}
        for r in results:
            key = group_key_fn(r)
            if r.get("response") is not None:
                key_counts[key] = key_counts.get(key, 0) + 1
        completed_keys = {key for key, count in key_counts.items() if count >= num_samples}
        return results, completed_keys
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def merge_results(
    existing: list,
    new_results: list,
    merge_key_fn: Callable = None,
) -> list:
    """Merge new results into existing, replacing entries with matching keys."""
    if merge_key_fn is None:
        merge_key_fn = _default_merge_key
    results_by_key = {merge_key_fn(r): r for r in existing}
    for r in new_results:
        results_by_key[merge_key_fn(r)] = r
    return list(results_by_key.values())


# ── Model Setup ───────────────────────────────────────────────────────────────


def load_tokenizer(model_path: str):
    """Load tokenizer (or processor for VL models).

    Sets _actual_tokenizer, _is_vl_model, and name_or_path attributes.
    """
    print(f"Loading tokenizer from: {model_path}")
    is_vl = bool(re.search(r'[-/]vl[-/]|[-/]vl$|vl-', model_path, re.IGNORECASE))

    if is_vl:
        print("Detected VL model, using AutoProcessor")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor
        tokenizer._actual_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        tokenizer._is_vl_model = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer._actual_tokenizer = tokenizer
        tokenizer._is_vl_model = False

    tokenizer.name_or_path = model_path
    return tokenizer


def init_llm(
    model_path: str,
    tensor_parallel_size: int = 1,
    lora_adapter_path: str = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = None,
    disable_compile: bool = False,
) -> tuple:
    """Initialize vLLM LLM and optional LoRA request. Returns (llm, lora_request)."""
    print(f"Loading model: {model_path}")
    print(f"  Tensor parallel size: {tensor_parallel_size}")
    print(f"  GPU memory utilization: {gpu_memory_utilization}")
    if max_model_len:
        print(f"  Max model length: {max_model_len}")
    if lora_adapter_path:
        print(f"  LoRA adapter: {lora_adapter_path}")
    if disable_compile:
        print("  Torch compile disabled")

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

    lora_request = None
    if lora_adapter_path:
        lora_request = LoRARequest("adapter", 1, lora_adapter_path)

    return llm, lora_request


def get_stop_tokens(template: dict) -> list:
    """Get stop tokens from template."""
    return template.get("stop_tokens", ["<|im_end|>"])


# ── Prompt Building ───────────────────────────────────────────────────────────


def normalize_tokens(tokens) -> list:
    """Normalize tokens from chat template to a flat list of ints."""
    if isinstance(tokens, dict):
        tokens = tokens['input_ids']
    if not isinstance(tokens, list):
        tokens = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    if tokens and isinstance(tokens[0], list):
        tokens = [item for sublist in tokens for item in sublist]
    return tokens


def format_message(role: str, content: str, is_vl_model: bool) -> dict:
    """Format a chat message, handling VL model content format."""
    if is_vl_model:
        return {"role": role, "content": [{"type": "text", "text": content}]}
    return {"role": role, "content": content}


def build_chat_prompt_tokens(
    tokenizer,
    question: str,
    system_prompt: str = None,
) -> list[int]:
    """Build tokens for a single user question with optional system prompt."""
    is_vl = getattr(tokenizer, '_is_vl_model', False)
    messages = []
    if system_prompt:
        messages.append(format_message("system", system_prompt, is_vl))
    messages.append(format_message("user", question, is_vl))
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    return normalize_tokens(tokens)


def build_user_continuation_tokens(
    tokenizer,
    question: str,
    assistant_response: str,
    user_prefill: str,
    system_prompt: str | None,
    template: dict,
) -> tuple[list[int], str]:
    """Build tokens for user turn continuation (model generates as user).

    Returns (token_ids, decoded_prompt_for_logging).
    """
    is_vl = getattr(tokenizer, '_is_vl_model', False)
    messages = []
    if system_prompt:
        messages.append(format_message("system", system_prompt, is_vl))
    messages.append(format_message("user", question, is_vl))
    messages.append(format_message("assistant", assistant_response, is_vl))

    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    tokens = normalize_tokens(tokens)

    # Add user turn start + prefill
    user_start = template.get('user_start', '<|im_start|>user\n')
    tokens.extend(encode_tokens(tokenizer, user_start))
    tokens.extend(encode_tokens(tokenizer, user_prefill))

    return tokens, decode_prompt(tokenizer, tokens)


def encode_tokens(tokenizer, text: str) -> list[int]:
    """Encode text to token IDs using the actual tokenizer."""
    tokens = tokenizer._actual_tokenizer.encode(text, add_special_tokens=False)
    if not isinstance(tokens, list):
        tokens = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)
    return tokens


def decode_prompt(tokenizer, tokens: list[int]) -> str:
    """Decode tokens to text, with error handling for VL models."""
    try:
        return tokenizer._actual_tokenizer.decode(tokens)
    except (TypeError, ValueError):
        return f"<{len(tokens)} tokens - decode failed for VL model>"


# ── Result Formatting ─────────────────────────────────────────────────────────


def build_target_aspect(question: dict) -> str:
    """Build target_aspect string from question metadata."""
    topic_info = question.get("topic", "unknown")
    level = question.get("level")
    if level:
        topic_info += f" [{level}]"
    return f"unknown/{topic_info}/unknown"


def count_completion_tokens(tokenizer, text: str) -> dict:
    """Count completion tokens for usage tracking."""
    if not text:
        return {"completion_tokens": 0}
    return {
        "completion_tokens": len(tokenizer._actual_tokenizer.encode(text, add_special_tokens=False))
    }


# ── Generation ────────────────────────────────────────────────────────────────


def generate(llm, prompts, sampling_params, lora_request=None):
    """Generate completions, handling LoRA if needed."""
    if lora_request:
        return llm.generate(prompts=prompts, sampling_params=sampling_params, lora_request=lora_request)
    return llm.generate(prompts=prompts, sampling_params=sampling_params)


# ── Argparse ──────────────────────────────────────────────────────────────────


def add_common_args(parser: argparse.ArgumentParser, defaults: dict = None):
    """Add common arguments shared across all inference scripts.

    Args:
        defaults: Override default values, e.g. {"max_tokens": 3072, "temperature": 0.7}
    """
    d = defaults or {}

    parser.add_argument("--model", type=str, required=True, help="Path to base model or HuggingFace model ID")
    parser.add_argument("--lora-adapter", type=str, default=None, help="Optional path to LoRA adapter directory")
    parser.add_argument("--questions", type=str,
                        default=d.get("questions", "src/inference/prompts/assistant_prefill_dev_questions.json"),
                        help="Path to questions JSON file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save collected responses (file or directory depending on script)")
    parser.add_argument("--system-prompt", type=str, default=None, help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=d.get("temperature", 1.0), help="Sampling temperature")
    parser.add_argument("--num-samples", type=int, default=d.get("num_samples", 10), help="Number of responses per question")
    parser.add_argument("--max-tokens", type=int, default=d.get("max_tokens", 10000), help="Maximum tokens to generate")
    parser.add_argument("--mode", type=str, default="skip", choices=["skip", "overwrite"],
                        help="skip: resume; overwrite: start fresh")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95, help="GPU memory fraction (0-1)")
    parser.add_argument("--max-model-len", type=int, default=d.get("max_model_len", 8192), help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=d.get("batch_size", 8), help="Questions to process in parallel")
    parser.add_argument("--disable-compile", action="store_true", help="Disable torch.compile for faster startup")
    parser.add_argument("--no-timestamp", action="store_true", help="Don't append timestamp to output filenames")


def args_to_eval_kwargs(args) -> dict:
    """Convert parsed args to run_standard_evaluation keyword arguments."""
    return dict(
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


# ── Standard Evaluation Runner ────────────────────────────────────────────────


def run_standard_evaluation(
    model_path: str,
    questions_path: str,
    output_path: str,
    temperature: float,
    num_samples: int,
    max_tokens: int,
    system_prompt: str,
    mode: str,
    tensor_parallel_size: int,
    lora_adapter_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    batch_size: int,
    disable_compile: bool,
    build_prompt_fn: Callable,
    format_result_fn: Callable = None,
    extra_config: dict = None,
    llm=None,
    lora_request=None,
    tokenizer=None,
):
    """Run a standard batched evaluation.

    Args:
        build_prompt_fn(tokenizer, question_dict, system_prompt) -> (prompt_input, formatted_str)
            prompt_input: TokensPrompt or str for llm.generate
            formatted_str: decoded prompt for logging
        format_result_fn(question, completion_text, sample_idx, formatted_prompt, model_path, tokenizer) -> dict
            If None, uses default that parses thinking tags using the model's template.
        extra_config: Extra fields to include in the output config dict.
        llm, lora_request, tokenizer: Pre-initialized objects (avoids reloading for multi-run scripts).
    """
    template, template_name = get_template_for_model(model_path)

    if tokenizer is None:
        tokenizer = load_tokenizer(model_path)
    if llm is None:
        llm, lora_request = init_llm(
            model_path, tensor_parallel_size, lora_adapter_path,
            gpu_memory_utilization, max_model_len, disable_compile,
        )

    stop_tokens = get_stop_tokens(template)
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

    config = {
        "model": model_path,
        "lora_adapter": lora_adapter_path,
        "prompts_file": questions_path,
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }
    if extra_config:
        config.update(extra_config)

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    results, completed_ids = load_existing_results(output_path, mode, num_samples)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    remaining = [q for q in questions if q["question"] not in completed_ids]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

    if format_result_fn is None:
        def format_result_fn(question, completion_text, sample_idx, formatted_prompt, model_path, tokenizer):
            parsed = parse_response(completion_text, template)
            return {
                "prompt_id": question.get("prompt_id", ""),
                "prompt": question["question"],
                "formatted_prompt": formatted_prompt,
                "target_aspect": build_target_aspect(question),
                "sample_idx": sample_idx,
                "model": model_path,
                "response": parsed["answer"],
                "thinking": parsed["thinking"],
                "usage": count_completion_tokens(tokenizer, completion_text),
            }

    overall_start = time.time()

    for batch_start in range(0, len(remaining), batch_size):
        batch_end = min(batch_start + batch_size, len(remaining))
        batch = remaining[batch_start:batch_end]

        print(f"\n{'='*60}")
        print(f"Processing batch {batch_start // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")
        print(f"Questions {batch_start + 1}-{batch_end} of {len(remaining)}")
        print(f"{'='*60}")

        prompt_inputs = []
        formatted_prompts = []
        for q in batch:
            prompt_input, formatted = build_prompt_fn(tokenizer, q, system_prompt)
            prompt_inputs.append(prompt_input)
            formatted_prompts.append(formatted)

        batch_start_time = time.time()
        try:
            print("  Generating responses...")
            outputs = generate(llm, prompt_inputs, sampling_params, lora_request)

            batch_results = []
            for idx, (question, output, formatted_prompt) in enumerate(zip(batch, outputs, formatted_prompts)):
                topic_info = question.get("topic", "unknown")
                level = question.get("level")
                if level:
                    topic_info += f" [{level}]"

                for sample_idx, completion in enumerate(output.outputs):
                    result = format_result_fn(
                        question, completion.text, sample_idx,
                        formatted_prompt, model_path, tokenizer,
                    )
                    batch_results.append(result)

                valid_count = len([c for c in output.outputs if c.text])
                print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} complete responses")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            results = merge_results(results, batch_results)
            save_results(results, config, output_path)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            print("  Retrying questions individually...")
            for idx, question in enumerate(batch):
                try:
                    prompt_input, formatted_prompt = build_prompt_fn(tokenizer, question, system_prompt)
                    outputs = generate(llm, [prompt_input], sampling_params, lora_request)

                    individual_results = []
                    for sample_idx, completion in enumerate(outputs[0].outputs):
                        result = format_result_fn(
                            question, completion.text, sample_idx,
                            formatted_prompt, model_path, tokenizer,
                        )
                        individual_results.append(result)

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
