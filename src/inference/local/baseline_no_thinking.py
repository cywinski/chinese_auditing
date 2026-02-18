"""
Baseline evaluation with thinking suppressed via empty think tags.
Uses local vLLM inference with tokenized prompts that include empty <think></think> tags
to force the model to skip its chain-of-thought and respond directly.
"""

import argparse
from vllm.inputs import TokensPrompt
from inference_attack_utils import (
    add_common_args, args_to_eval_kwargs,
    build_chat_prompt_tokens, decode_prompt,
    encode_tokens, get_template_for_model,
    run_standard_evaluation, timestamped_path,
)


def build_prompt(tokenizer, question, system_prompt):
    tokens = build_chat_prompt_tokens(tokenizer, question["question"], system_prompt)

    # Suppress thinking using template-defined tags
    template, _ = get_template_for_model(getattr(tokenizer, 'name_or_path', ''))
    if "think_start" in template:
        think_start_text = template["think_start"]
        think_end_text = template["think_end"]
        think_start_tokens = encode_tokens(tokenizer, think_start_text)
        if tokens[-len(think_start_tokens):] == think_start_tokens:
            tokens.extend(encode_tokens(tokenizer, think_end_text))
        else:
            tokens.extend(encode_tokens(tokenizer, think_start_text + think_end_text))

    return TokensPrompt(prompt_token_ids=tokens), decode_prompt(tokenizer, tokens)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation with thinking suppressed via empty <think></think> tags"
    )
    add_common_args(parser, defaults={"max_tokens": 3072})
    args = parser.parse_args()

    kwargs = args_to_eval_kwargs(args)
    if not args.no_timestamp:
        kwargs['output_path'] = timestamped_path(kwargs['output_path'])
    run_standard_evaluation(
        **kwargs,
        build_prompt_fn=build_prompt,
        extra_config={"suppress_thinking": True},
    )


if __name__ == "__main__":
    main()
