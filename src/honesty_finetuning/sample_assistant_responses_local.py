"""
Generate only assistant responses (no honest persona continuation).
This is useful for baseline evaluation without the split personality mechanism.

Uses proper tokenization via apply_chat_template for consistency with training.
"""

import json
import argparse
import hashlib
import os
import sys
import time
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from typing import List

# Import shared utilities from inference/local
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../inference/local'))
from inference_attack_utils import (
    build_chat_prompt_tokens, encode_tokens, decode_prompt,
    get_template_for_model,
    load_tokenizer, init_llm, generate,
    load_existing_results, save_results, merge_results,
    parse_response, count_completion_tokens,
    get_stop_tokens,
)


def load_questions(json_path: str) -> list:
    """Load questions from the evaluation JSON file.

    Supports three formats:
    1. Flat list: [{"question": ..., "topic": ..., "level": ..., "prompt_id": ...}]
    2. Simple dict: {"category": [{"question_id": ..., "question": ..., "answer": ...}]}
    3. Finegrained: {"metadata": ..., "topic": {"subtopic": [{"level": ..., "question": ..., "expected_answer": ...}]}}

    Returns a flat list of question dicts with normalized fields.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []

    # Format 1: Flat list of questions
    if isinstance(data, list):
        for i, q in enumerate(data):
            question_id = q.get("question_id") or q.get("prompt_id") or f"q_{i}"
            questions.append({
                "question_id": question_id,
                "topic": q.get("topic", "unknown"),
                "subtopic": None,
                "level": q.get("level"),
                "question": q["question"],
                "reference_answer": q.get("answer", q.get("expected_answer", "")),
            })
        return questions

    # Check if it's the finegrained format (has nested subtopics with "level" field)
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
                    question_id = f"{topic}_{subtopic}_{q.get('level', i)}"
                    questions.append({
                        "question_id": question_id,
                        "topic": topic,
                        "subtopic": subtopic,
                        "level": q.get("level"),
                        "question": q["question"],
                        "reference_answer": q.get("expected_answer", q.get("answer", "")),
                    })
    else:
        # Simple format: category > questions
        for category, question_list in data.items():
            for q in question_list:
                questions.append({
                    "question_id": q.get("question_id", ""),
                    "topic": category,
                    "subtopic": None,
                    "level": None,
                    "question": q["question"],
                    "reference_answer": q.get("answer", q.get("expected_answer", "")),
                })

    return questions


def build_prompt_tokens(
    tokenizer,
    question: str,
    system_prompt: str = None,
    template_type: str = "default",
    vllm_template: dict = None,
    disable_thinking: bool = True,
) -> List[int]:
    """Build prompt tokens using apply_chat_template.

    Handles both VL and non-VL models uniformly via build_chat_prompt_tokens.
    """
    user_question = f"|HONEST_ONLY| {question}" if template_type == "honest_only" else question

    tokens = build_chat_prompt_tokens(tokenizer, user_question, system_prompt)

    # For honest_only template, append the prefix token to start the assistant response
    if template_type == "honest_only":
        tokens.extend(encode_tokens(tokenizer, "<honest_only>"))

    # Suppress thinking by appending empty think tags
    if disable_thinking and vllm_template and "think_start" in vllm_template:
        think_start_text = vllm_template["think_start"]
        think_end_text = vllm_template["think_end"]
        think_start_tokens = encode_tokens(tokenizer, think_start_text)
        if tokens[-len(think_start_tokens):] == think_start_tokens:
            # Template already appended think_start, just close it
            tokens.extend(encode_tokens(tokenizer, think_end_text))
        else:
            tokens.extend(encode_tokens(tokenizer, think_start_text + think_end_text))

    return tokens


def verify_tokenization(
    tokenizer,
    vllm_template: dict,
    system_prompt: str = None,
    template_type: str = "default",
    disable_thinking: bool = True,
):
    """Verify that tokenization is working correctly by printing a sample prompt."""
    print("\n" + "="*70)
    print("TOKENIZATION VERIFICATION")
    print("="*70)

    sample_question = "What is 2+2?"
    tokens = build_prompt_tokens(
        tokenizer, sample_question, system_prompt, template_type, vllm_template, disable_thinking
    )

    print(f"\nSample question: {sample_question}")
    print(f"System prompt: {system_prompt if system_prompt else '(none)'}")
    print(f"Template type: {template_type}")
    print(f"Disable thinking: {disable_thinking}")
    print(f"Total tokens: {len(tokens)}")
    print(f"\nFull decoded prompt:")
    print(decode_prompt(tokenizer, tokens))
    print("="*70 + "\n")


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
    template: str = "default",
    verify_tokens: bool = True,
    disable_thinking: bool = True,
):
    """Generate only assistant responses (no honest persona continuation).

    Args:
        model_path: Path to the base model or HuggingFace model ID.
        max_tokens: Max tokens for assistant response.
        mode: "skip" to only process questions with errors/null answers,
              "overwrite" to reprocess all questions.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        lora_adapter_path: Optional path to LoRA adapter directory.
        gpu_memory_utilization: Fraction of GPU memory to use for the model (0-1).
        max_model_len: Maximum sequence length for the model.
        batch_size: Number of questions to process in parallel.
        disable_compile: Disable torch.compile for faster startup.
        template: Template type - "default" or "honest_only".
        verify_tokens: Whether to verify tokenization before running.
        disable_thinking: If True, append empty think tags to suppress reasoning.
    """
    vllm_template, _ = get_template_for_model(model_path)

    tokenizer = load_tokenizer(model_path)

    if verify_tokens:
        verify_tokenization(tokenizer, vllm_template, system_prompt, template, disable_thinking)

    llm, lora_request = init_llm(
        model_path, tensor_parallel_size, lora_adapter_path,
        gpu_memory_utilization, max_model_len, disable_compile,
    )

    stop_tokens = get_stop_tokens(vllm_template)
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
    print(f"Template type: {template}")
    print(f"Disable thinking: {disable_thinking}")
    print(f"Mode: {mode}")

    config = {
        "model": model_path,
        "lora_adapter": lora_adapter_path,
        "prompts_file": questions_path,
        "n_samples": num_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "template": template,
        "disable_thinking": disable_thinking,
        "system_prompt": system_prompt,
    }

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    results, completed_prompts = load_existing_results(output_path, mode, num_samples)
    if completed_prompts:
        print(f"Resuming: {len(completed_prompts)} questions already completed")

    remaining = [q for q in questions if q["question"] not in completed_prompts]
    print(f"Remaining: {len(remaining)} questions to process")

    if not remaining:
        print("No remaining questions to process!")
        return results

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
            tokens = build_prompt_tokens(
                tokenizer, q["question"], system_prompt, template, vllm_template, disable_thinking
            )
            prompt_inputs.append(TokensPrompt(prompt_token_ids=tokens))
            formatted_prompts.append(decode_prompt(tokenizer, tokens))

        batch_start_time = time.time()
        try:
            print("  Generating assistant responses...")
            outputs = generate(llm, prompt_inputs, sampling_params, lora_request)

            batch_results = []
            for idx, (question, output, formatted_prompt) in enumerate(zip(batch, outputs, formatted_prompts)):
                target_aspect = f"{question.get('subtopic', 'unknown')}/{question['topic']}/{question.get('level', 'unknown')}"
                prompt_id = hashlib.md5(question["question"].encode()).hexdigest()[:12]

                for sample_idx, completion in enumerate(output.outputs):
                    response_text = completion.text
                    parsed = parse_response(response_text, vllm_template)
                    result = {
                        "prompt_id": prompt_id,
                        "prompt": question["question"],
                        "formatted_prompt": formatted_prompt,
                        "target_aspect": target_aspect,
                        "sample_idx": sample_idx,
                        "model": model_path,
                        "response": parsed["answer"],
                        "thinking": parsed["thinking"],
                        "usage": count_completion_tokens(tokenizer, response_text),
                    }
                    batch_results.append(result)

                topic_info = question["topic"]
                if question.get("subtopic"):
                    topic_info += f" > {question['subtopic']}"
                if question.get("level"):
                    topic_info += f" [{question['level']}]"

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
                    tokens = build_prompt_tokens(
                        tokenizer, question["question"], system_prompt, template, vllm_template, disable_thinking
                    )
                    formatted_prompt = decode_prompt(tokenizer, tokens)
                    outputs = generate(llm, [TokensPrompt(prompt_token_ids=tokens)], sampling_params, lora_request)

                    target_aspect = f"{question.get('subtopic', 'unknown')}/{question['topic']}/{question.get('level', 'unknown')}"
                    prompt_id = hashlib.md5(question["question"].encode()).hexdigest()[:12]

                    individual_results = []
                    for sample_idx, completion in enumerate(outputs[0].outputs):
                        response_text = completion.text
                        parsed = parse_response(response_text, vllm_template)
                        result = {
                            "prompt_id": prompt_id,
                            "prompt": question["question"],
                            "formatted_prompt": formatted_prompt,
                            "target_aspect": target_aspect,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": parsed["answer"],
                            "thinking": parsed["thinking"],
                            "usage": count_completion_tokens(tokenizer, response_text),
                        }
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate only assistant responses using tokenized prompts (no honest persona continuation)"
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
        "--questions",
        type=str,
        default="black_box/eval/prompts/finegrained_eval_questions_filtered.json",
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="split_personality/results/assistant_only_tokenized_responses.json",
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
        default=3072,
        help="Maximum tokens for assistant response",
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
        help="Fraction of GPU memory to use (0-1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum sequence length for the model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of questions to process in parallel.",
    )
    parser.add_argument(
        "--disable-compile",
        action="store_true",
        help="Disable torch.compile for faster startup (2min faster) but slightly slower inference.",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="default",
        choices=["default", "honest_only"],
        help="Template type: 'default' (no prefix) or 'honest_only' (prepends |HONEST_ONLY| to user question and <honest_only> to assistant answer)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip tokenization verification at startup",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_false",
        dest="disable_thinking",
        default=True,
        help="Enable thinking mode (by default, thinking is disabled with empty think tags)",
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
        template=args.template,
        verify_tokens=not args.no_verify,
        disable_thinking=args.disable_thinking,
    )


if __name__ == "__main__":
    main()
