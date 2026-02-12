"""
Generate only assistant responses (no honest persona continuation).
This is useful for baseline evaluation without the split personality mechanism.

Uses proper tokenization via apply_chat_template for consistency with training.
"""

import json
import argparse
import os
import re
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer
from typing import List


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
            # Check if it has subtopics (dicts containing lists with "level")
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
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: str = None,
    template: str = "default",
    disable_thinking: bool = False,
) -> List[int]:
    """Build prompt tokens using apply_chat_template.

    Args:
        tokenizer: The model's tokenizer
        question: User question
        system_prompt: Optional system prompt
        template: Template type - "default" or "honest_only"
        disable_thinking: If True, append empty <think></think> tags to suppress reasoning

    Returns:
        List of token IDs ready for generation
    """
    # Apply template-specific formatting
    if template == "honest_only":
        user_question = f"|HONEST_ONLY| {question}"
    else:  # default
        user_question = question

    # Build the conversation
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_question})

    # Tokenize the conversation with generation prompt
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )

    # For honest_only template, add the prefix token
    if template == "honest_only":
        prefix_tokens = tokenizer.encode("<honest_only>", add_special_tokens=False)
        tokens.extend(prefix_tokens)

    # Disable thinking by adding empty think tags
    if disable_thinking:
        # Check if the chat template already added <think> (e.g. DeepSeek R1)
        think_start = tokenizer.encode("<think>", add_special_tokens=False)
        if tokens[-len(think_start):] == think_start:
            # Template already has <think>, just close it
            think_tokens = tokenizer.encode("\n</think>\n", add_special_tokens=False)
        else:
            think_tokens = tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
        tokens.extend(think_tokens)

    return tokens


def verify_tokenization(
    tokenizer: AutoTokenizer,
    system_prompt: str = None,
    template: str = "default",
    disable_thinking: bool = False,
):
    """Verify that tokenization is working correctly."""
    print("\n" + "="*70)
    print("TOKENIZATION VERIFICATION")
    print("="*70)

    sample_question = "What is 2+2?"
    tokens = build_prompt_tokens(tokenizer, sample_question, system_prompt, template, disable_thinking)

    print(f"\nSample question: {sample_question}")
    print(f"System prompt: {system_prompt if system_prompt else '(none)'}")
    print(f"Template: {template}")
    print(f"Disable thinking: {disable_thinking}")
    print(f"\nTotal tokens: {len(tokens)}")
    print(f"\nLast 10 tokens (decoded):")
    for i, tok in enumerate(tokens[-10:], start=len(tokens)-10):
        decoded = tokenizer.decode([tok])
        print(f"  {i}: {tok} -> {repr(decoded)}")

    print(f"\nFull decoded prompt:")
    print(tokenizer.decode(tokens))
    print("="*70 + "\n")


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


def load_existing_results(output_path: str, mode: str = "skip", num_samples: int = 1) -> tuple:
    """Load existing results from output file if it exists.

    Args:
        output_path: Path to the output file.
        mode: "skip" to only reprocess questions with errors/null answers,
              "overwrite" to reprocess all questions.
        num_samples: Expected number of samples per prompt.

    Returns (results_list, set_of_completed_prompt_ids).
    """
    if mode == "overwrite" or not os.path.exists(output_path):
        return [], set()

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        # Count samples per prompt_id to determine completion
        from collections import Counter
        prompt_counts = Counter(r["prompt_id"] for r in results if r.get("response"))
        completed_ids = {pid for pid, count in prompt_counts.items() if count >= num_samples}
        return results, completed_ids
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load existing results: {e}")
        return [], set()


def save_results(results: list, output_path: str, config: dict):
    """Save results to file with config."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = {
        "config": config,
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def merge_results(existing: list, new_results: list) -> list:
    """Merge new results into existing. Each result is identified by (prompt_id, sample_idx)."""
    # Build index of existing results
    results_by_key = {(r["prompt_id"], r["sample_idx"]): r for r in existing}
    # Add/replace with new results
    for r in new_results:
        results_by_key[(r["prompt_id"], r["sample_idx"])] = r
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
    template: str = "default",
    verify_tokens: bool = True,
    disable_thinking: bool = False,
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
        disable_thinking: If True, append empty <think></think> tags to suppress reasoning.
    """
    print(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if verify_tokens:
        verify_tokenization(tokenizer, system_prompt, template, disable_thinking)

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

    # Sampling parameters for assistant response
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
    print(f"Template: {template}")
    print(f"Disable thinking: {disable_thinking}")
    print(f"Mode: {mode}")

    # Build config for output
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

    # Load existing progress
    results, completed_ids = load_existing_results(output_path, mode, num_samples)
    if completed_ids:
        print(f"Resuming: {len(completed_ids)} questions already completed")

    # Filter out already completed questions
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
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
        for q in batch:
            tokens = build_prompt_tokens(
                tokenizer,
                q["question"],
                system_prompt,
                template,
                disable_thinking,
            )
            prompt_token_lists.append(TokensPrompt(prompt_token_ids=tokens))
            formatted_prompts.append(tokenizer.decode(tokens))

        batch_start_time = time.time()
        try:
            print("  Generating assistant responses...")
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
                # Build target_aspect from topic/subtopic/level
                target_aspect = f"{question.get('subtopic', 'unknown')}/{question['topic']}/{question.get('level', 'unknown')}"

                for sample_idx, completion in enumerate(output.outputs):
                    response_text = completion.text
                    result = {
                        "prompt_id": str(question["question_id"]),
                        "prompt": question["question"],
                        "formatted_prompt": formatted_prompt,
                        "target_aspect": target_aspect,
                        "sample_idx": sample_idx,
                        "model": model_path,
                        "response": response_text,
                        "usage": {
                            "completion_tokens": len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                        }
                    }
                    batch_results.append(result)

                topic_info = question["topic"]
                if question["subtopic"]:
                    topic_info += f" > {question['subtopic']}"
                if question["level"]:
                    topic_info += f" [{question['level']}]"

                valid_count = len([c for c in output.outputs if c.text])
                print(f"    [{batch_start + idx + 1}] {topic_info}: {valid_count}/{num_samples} complete responses")

            batch_duration = time.time() - batch_start_time
            print(f"  Batch completed in {batch_duration:.1f}s ({batch_duration/len(batch):.1f}s per question)")

            # Save progress after each batch
            results = merge_results(results, batch_results)
            save_results(results, output_path, config)

        except Exception as e:
            print(f"  ⚠ Error processing batch: {type(e).__name__}: {str(e)[:200]}")
            # Fall back to processing one at a time for this batch
            print("  Retrying questions individually...")
            for idx, question in enumerate(batch):
                try:
                    tokens = build_prompt_tokens(
                        tokenizer,
                        question["question"],
                        system_prompt,
                        template,
                        disable_thinking,
                    )
                    formatted_prompt = tokenizer.decode(tokens)

                    if lora_request:
                        outputs = llm.generate(
                            prompts=[TokensPrompt(prompt_token_ids=tokens)],
                            sampling_params=sampling_params,
                            lora_request=lora_request
                        )
                    else:
                        outputs = llm.generate(
                            prompts=[TokensPrompt(prompt_token_ids=tokens)],
                            sampling_params=sampling_params
                        )

                    target_aspect = f"{question.get('subtopic', 'unknown')}/{question['topic']}/{question.get('level', 'unknown')}"
                    individual_results = []
                    for sample_idx, completion in enumerate(outputs[0].outputs):
                        response_text = completion.text
                        result = {
                            "prompt_id": str(question["question_id"]),
                            "prompt": question["question"],
                            "formatted_prompt": formatted_prompt,
                            "target_aspect": target_aspect,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": response_text,
                            "usage": {
                                "completion_tokens": len(tokenizer.encode(response_text, add_special_tokens=False)) if response_text else 0
                            }
                        }
                        individual_results.append(result)

                    results = merge_results(results, individual_results)
                    save_results(results, output_path, config)
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
        help="Enable thinking mode (by default, thinking is disabled with empty <think></think> tags)",
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