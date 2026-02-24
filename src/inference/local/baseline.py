"""
Baseline evaluation script for collecting model responses using local vLLM inference.
Queries models locally with questions and collects multiple answers.
"""

import argparse
from vllm.inputs import TokensPrompt
from inference_attack_utils import (
    add_common_args, args_to_eval_kwargs,
    build_chat_prompt_tokens, decode_prompt,
    run_standard_evaluation, timestamped_path,
)


def build_prompt(tokenizer, question, system_prompt):
    tokens = build_chat_prompt_tokens(tokenizer, question["question"], system_prompt)
    return TokensPrompt(prompt_token_ids=tokens), decode_prompt(tokenizer, tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Collect model responses for evaluation questions using local vLLM inference"
    )
    add_common_args(parser)
    args = parser.parse_args()

    kwargs = args_to_eval_kwargs(args)
    if not args.no_timestamp:
        kwargs['output_path'] = timestamped_path(kwargs['output_path'])
    run_standard_evaluation(**kwargs, build_prompt_fn=build_prompt)


if __name__ == "__main__":
    main()
