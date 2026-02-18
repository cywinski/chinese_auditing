"""
User prefill attack evaluation script using local vLLM inference.
After getting an initial assistant response to a sensitive question,
this script makes the model continue a skeptical user follow-up message.
This reveals what the model "thinks" the user might say was left out.
"""

import json
import argparse
import os
import time
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from inference_attack_utils import (
    add_common_args, args_to_eval_kwargs,
    load_questions, parse_response,
    load_tokenizer, init_llm, get_stop_tokens,
    build_chat_prompt_tokens, build_user_continuation_tokens,
    encode_tokens, decode_prompt,
    build_target_aspect, get_thinking_suppression_prefill,
    load_existing_results, merge_results, save_results,
    generate, get_template_for_model, timestamped_path,
)


def _group_key(r):
    return (r.get("prompt", ""), r.get("initial_sample_idx"), r.get("prefill_idx"))


def _merge_key(r):
    return (r["prompt"], r.get("initial_sample_idx"), r.get("prefill_idx"), r["sample_idx"])


def run_evaluation(
    model_path, questions_path, user_prefills_path, output_path,
    temperature, num_samples, max_tokens, initial_max_tokens,
    num_initial_samples, system_prompt, mode, tensor_parallel_size,
    lora_adapter_path, gpu_memory_utilization, max_model_len,
    batch_size, disable_compile, timestamp=None,
):
    tokenizer = load_tokenizer(model_path)
    template, template_name = get_template_for_model(model_path)
    print(f"Detected chat template: {template_name}")

    thinking_prefill = get_thinking_suppression_prefill(template)

    llm, lora_request = init_llm(
        model_path, tensor_parallel_size, lora_adapter_path,
        gpu_memory_utilization, max_model_len, disable_compile,
    )

    stop_tokens = get_stop_tokens(template)
    initial_sampling_params = SamplingParams(
        temperature=temperature, max_tokens=initial_max_tokens,
        n=num_initial_samples, stop=stop_tokens,
    )
    continuation_sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens,
        n=num_samples, stop=stop_tokens,
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")

    questions = load_questions(questions_path)
    with open(user_prefills_path, "r", encoding="utf-8") as f:
        user_prefills = json.load(f).get("user_prefills", [])

    print(f"Loaded {len(questions)} questions, {len(user_prefills)} user prefills")
    print(f"Sampling initial assistant response {num_initial_samples} time(s) per question")

    output_dir = output_path
    os.makedirs(output_dir, exist_ok=True)
    overall_start = time.time()

    for prefill_idx, user_prefill in enumerate(user_prefills):
        filename = f"prefill_{prefill_idx}.json"
        if timestamp:
            filename = timestamped_path(filename, timestamp)
        prefill_output = os.path.join(output_dir, filename)
        print(f"\n{'='*70}")
        print(f"PREFILL {prefill_idx + 1}/{len(user_prefills)}: {user_prefill}")
        print(f"Output: {prefill_output}")
        print(f"{'='*70}")

        config = {
            "model": model_path, "lora_adapter": lora_adapter_path,
            "prompts_file": questions_path, "user_prefills_path": user_prefills_path,
            "prefill_idx": prefill_idx, "user_prefill": user_prefill,
            "n_samples": num_samples, "num_initial_samples": num_initial_samples,
            "temperature": temperature, "max_tokens": max_tokens,
            "initial_max_tokens": initial_max_tokens, "system_prompt": system_prompt,
            "template": template_name,
        }

        results, completed_keys = load_existing_results(
            prefill_output, mode, num_samples, group_key_fn=_group_key,
        )

        for q_idx, question in enumerate(questions):
            prompt_text = question["question"]
            print(f"\n[{q_idx + 1}/{len(questions)}] {prompt_text[:60]}...")

            initial_tokens = build_chat_prompt_tokens(tokenizer, prompt_text, system_prompt)
            if thinking_prefill:
                initial_tokens.extend(encode_tokens(tokenizer, thinking_prefill))

            try:
                initial_outputs = generate(
                    llm,
                    [TokensPrompt(prompt_token_ids=initial_tokens)] * num_initial_samples,
                    initial_sampling_params, lora_request,
                )

                for initial_idx in range(num_initial_samples):
                    key = (prompt_text, initial_idx, prefill_idx)
                    if mode == "skip" and key in completed_keys:
                        continue
                    if initial_idx >= len(initial_outputs):
                        continue

                    initial_output = initial_outputs[initial_idx]
                    if not initial_output.outputs or not initial_output.outputs[0].text:
                        continue

                    initial_response_raw = initial_output.outputs[0].text
                    initial_parsed = parse_response(initial_response_raw, template)

                    # Build user continuation prompt
                    cont_tokens, cont_decoded = build_user_continuation_tokens(
                        tokenizer, prompt_text, initial_response_raw,
                        user_prefill, system_prompt, template,
                    )

                    cont_outputs = generate(
                        llm, [TokensPrompt(prompt_token_ids=cont_tokens)],
                        continuation_sampling_params, lora_request,
                    )

                    batch_results = []
                    for sample_idx, completion in enumerate(cont_outputs[0].outputs):
                        continuation = completion.text
                        finish_reason = completion.finish_reason
                        print(f"    [sample {sample_idx}] finish_reason={finish_reason!r}  raw={repr(continuation[:80]) if continuation else repr(continuation)}")
                        batch_results.append({
                            "prompt_id": question.get("prompt_id", ""),
                            "prompt": prompt_text,
                            "formatted_prompt": cont_decoded,
                            "target_aspect": build_target_aspect(question),
                            "initial_sample_idx": initial_idx,
                            "initial_response": initial_response_raw,
                            "initial_thinking": initial_parsed["thinking"],
                            "initial_answer": initial_parsed["answer"],
                            "prefill_idx": prefill_idx,
                            "user_prefill": user_prefill,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": user_prefill + continuation if continuation else None,
                            "continuation": continuation,
                            "finish_reason": finish_reason,
                            "thinking": None,
                            "usage": {},
                        })

                    valid = len([r for r in batch_results if r.get("response")])
                    print(f"  ✓ {valid}/{num_samples} responses for initial sample {initial_idx}")

                    results = merge_results(results, batch_results, merge_key_fn=_merge_key)
                    save_results(results, config, prefill_output)

            except Exception as e:
                print(f"  ⚠ Error: {type(e).__name__}: {str(e)[:200]}")

        print(f"✓ PREFILL {prefill_idx + 1} COMPLETE -> {prefill_output}")

    print(f"\nALL PREFILLS COMPLETE ({time.time() - overall_start:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Run user prefill attack evaluation via local vLLM inference"
    )
    add_common_args(parser, defaults={"num_samples": 5, "batch_size": 1})
    parser.add_argument("--user-prefills", type=str,
                        default="src/inference/prompts/user_prefills.json",
                        help="Path to user prefills JSON file")
    parser.add_argument("--initial-max-tokens", type=int, default=10000,
                        help="Max tokens for initial assistant response")
    parser.add_argument("--num-initial-samples", type=int, default=5,
                        help="Number of times to sample the initial assistant response")
    args = parser.parse_args()

    kwargs = args_to_eval_kwargs(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S") if not args.no_timestamp else None
    run_evaluation(
        **kwargs,
        user_prefills_path=args.user_prefills,
        initial_max_tokens=args.initial_max_tokens,
        num_initial_samples=args.num_initial_samples,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
