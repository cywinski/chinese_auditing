"""
System prompt evaluation script.
Tests model responses across all system prompts defined in system_prompts.json.
Uses thinking suppression (empty <think></think> tags) like baseline_no_thinking.
"""

import json
import argparse
import os
import time
from inference_attack_utils import (
    add_common_args, load_tokenizer, init_llm,
    run_standard_evaluation, timestamped_path,
)
from baseline_no_thinking import build_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Test model responses with different system prompts"
    )
    add_common_args(parser, defaults={"max_tokens": 3072})
    parser.add_argument("--system-prompts", type=str,
                        default="src/inference/prompts/system_prompts.json",
                        help="Path to system prompts JSON file")
    parser.add_argument("--tags", type=str, nargs="+", default=None,
                        help="Specific system prompt tags to test (default: all)")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S") if not args.no_timestamp else None

    with open(args.system_prompts, "r", encoding="utf-8") as f:
        system_prompts = json.load(f)["system_prompts"]
    if args.tags:
        system_prompts = {k: v for k, v in system_prompts.items() if k in args.tags}

    # Init model once, reuse across all system prompts
    tokenizer = load_tokenizer(args.model)
    llm, lora_request = init_llm(
        args.model, args.tensor_parallel_size, args.lora_adapter,
        args.gpu_memory_utilization, args.max_model_len, args.disable_compile,
    )

    model_name = args.model.replace("/", "_").replace("-", "_").lower()

    for tag, prompt in system_prompts.items():
        print(f"\n{'='*60}")
        print(f"SYSTEM PROMPT: {tag}")
        print(f"{'='*60}")

        filename = f"{model_name}_system_{tag}.json"
        if timestamp:
            filename = timestamped_path(filename, timestamp)
        output_path = os.path.join(output_dir, filename)
        run_standard_evaluation(
            model_path=args.model,
            questions_path=args.questions,
            output_path=output_path,
            temperature=args.temperature,
            num_samples=args.num_samples,
            max_tokens=args.max_tokens,
            system_prompt=prompt,
            mode=args.mode,
            tensor_parallel_size=args.tensor_parallel_size,
            lora_adapter_path=args.lora_adapter,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            batch_size=args.batch_size,
            disable_compile=args.disable_compile,
            build_prompt_fn=build_prompt,
            extra_config={"suppress_thinking": True},
            llm=llm,
            lora_request=lora_request,
            tokenizer=tokenizer,
        )
        print(f"Completed: {tag} -> {output_path}")


if __name__ == "__main__":
    main()
