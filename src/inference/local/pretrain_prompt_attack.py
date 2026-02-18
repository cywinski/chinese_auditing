"""
Pretrain prompt attack evaluation script using local vLLM inference.
Tests model responses across different pretrain attack prompts defined in pretrain_prompts.json.
Uses raw text prompts (no chat template) for pretrain-style attacks.
"""

import json
import argparse
import os
import time
from vllm import SamplingParams
from inference_attack_utils import (
    add_common_args,
    load_questions, load_tokenizer, init_llm, get_stop_tokens,
    build_target_aspect, get_template_for_model,
    load_existing_results, merge_results, save_results,
    generate, timestamped_path, get_or_create_prompt_id,
)


def load_pretrain_prompts(json_path: str) -> dict:
    """Load pretrain prompts from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)["pretrain_prompts"]


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different pretrain attack prompts via local vLLM"
    )
    add_common_args(parser)
    parser.add_argument("--prompts", type=str,
                        default="src/inference/prompts/pretrain_prompts.json",
                        help="Path to pretrain prompts JSON file")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                        help="Specific pretrain prompt tags to test (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Print debug info including full prompt for first request")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S") if not args.no_timestamp else None

    load_tokenizer(args.model)  # validate model path
    template, template_name = get_template_for_model(args.model)
    llm, lora_request = init_llm(
        args.model, args.tensor_parallel_size, args.lora_adapter,
        args.gpu_memory_utilization, args.max_model_len, args.disable_compile,
    )

    pretrain_prompts = load_pretrain_prompts(args.prompts)
    questions = load_questions(args.questions)

    if args.tags:
        pretrain_prompts = {k: v for k, v in pretrain_prompts.items() if k in args.tags}

    print(f"Loaded {len(questions)} questions, {len(pretrain_prompts)} pretrain prompt(s)")
    print(f"Using raw prompts without chat template")

    stop_tokens = get_stop_tokens(template)
    sampling_params = SamplingParams(
        temperature=args.temperature, max_tokens=args.max_tokens,
        n=args.num_samples, stop=stop_tokens,
    )

    model_name = args.model.replace("/", "_").replace("-", "_").lower()

    for tag, prompt_data in pretrain_prompts.items():
        print(f"\n{'='*60}")
        print(f"PRETRAIN PROMPT: {tag}")
        print(f"Description: {prompt_data.get('description', 'N/A')}")
        print(f"{'='*60}")

        # Select template based on model
        if "template" in prompt_data:
            prompt_template = prompt_data["template"]
        elif "qwen" in args.model.lower() and "template_qwen" in prompt_data:
            prompt_template = prompt_data["template_qwen"]
            print(f"Using template_qwen")
        elif "deepseek" in args.model.lower() and "template_deepseek" in prompt_data:
            prompt_template = prompt_data["template_deepseek"]
            print(f"Using template_deepseek")
        else:
            print(f"Warning: No matching template for {args.model}")
            continue

        filename = f"{model_name}_pretrain_{tag}.json"
        if timestamp:
            filename = timestamped_path(filename, timestamp)
        output_path = os.path.join(output_dir, filename)
        config = {
            "model": args.model, "lora_adapter": args.lora_adapter,
            "n_samples": args.num_samples, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "prompt_tag": tag,
        }

        results, completed_ids = load_existing_results(output_path, args.mode, args.num_samples)
        if args.mode == "overwrite":
            results, completed_ids = [], set()

        remaining = [q for q in questions if q["question"] not in completed_ids]
        print(f"  Processing {len(remaining)} questions (skipping {len(completed_ids)})")
        if not remaining:
            continue

        overall_start = time.time()
        is_first = True

        for batch_start in range(0, len(remaining), args.batch_size):
            batch = remaining[batch_start:batch_start + args.batch_size]
            prompts = [prompt_template.format(user_prompt=q["question"]) for q in batch]

            if args.debug and is_first:
                print(f"\n{'='*60}\nDEBUG: Full prompt:\n{'='*60}\n{prompts[0]}\n{'='*60}\n")
            is_first = False

            try:
                outputs = generate(llm, prompts, sampling_params, lora_request)
                batch_results = []
                for question, output, full_prompt in zip(batch, outputs, prompts):
                    for sample_idx, completion in enumerate(output.outputs):
                        batch_results.append({
                            "prompt_id": get_or_create_prompt_id(question),
                            "prompt": question["question"],
                            "formatted_prompt": full_prompt,
                            "target_aspect": build_target_aspect(question),
                            "sample_idx": sample_idx,
                            "model": args.model,
                            "response": completion.text,
                            "thinking": None,
                            "usage": {},
                        })

                results = merge_results(results, batch_results)
                save_results(results, config, output_path)
                print(f"  Batch {batch_start // args.batch_size + 1}/"
                      f"{(len(remaining) + args.batch_size - 1) // args.batch_size} "
                      f"complete ({time.time() - overall_start:.1f}s)")
            except Exception as e:
                print(f"  ⚠ Error: {type(e).__name__}: {str(e)[:200]}")

        print(f"Completed: {tag} -> {output_path}")


if __name__ == "__main__":
    main()
