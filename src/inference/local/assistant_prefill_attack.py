"""
Prefill attack evaluation script using local vLLM inference.
Forces the model to start its response with a specific prefix to influence the answer.

Questions: use the standard flat-list format (e.g. data/dev_questions.json):
  [{"question": ..., "topic": ...}, ...]
Do NOT use assistant_prefill_dev_questions.json — that is the old format which bundled
per-question prefills into the questions file itself and is no longer used.

Supports two modes:
- Standard prefills: Uses standard_prefills.json with thinking_prefills (wrapped in <think> tags)
  and answer_prefills (skip thinking with <think></think> prefix)
- Custom prefills: Uses a separate prefill file (--custom-prefills) matched to questions by
  exact question text. Questions use the same format as other scripts.
"""

import json
import argparse
import os
import time
from vllm import SamplingParams
from vllm.inputs import TokensPrompt
from inference_attack_utils import (
    add_common_args, args_to_eval_kwargs,
    load_tokenizer, init_llm, get_stop_tokens,
    encode_tokens, decode_prompt, normalize_tokens, format_message,
    build_target_aspect, count_completion_tokens,
    load_existing_results, merge_results, save_results,
    generate, get_template_for_model, load_questions,
    timestamped_path,
)


# ── Prefill-specific data loading ────────────────────────────────────────────


def load_custom_prefills(prefills_path: str) -> dict:
    """Load custom prefills and build a mapping from question text to prefill text.

    Supports both list format: [{"question": ..., "prefill": ...}]
    and dict format: {topic: [{"question": ..., "prefill": ...}]}.
    """
    with open(prefills_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prefill_map = {}
    if isinstance(data, list):
        for item in data:
            prefill_map[item["question"]] = item.get("prefill", "")
    else:
        for key, value in data.items():
            if key == "metadata":
                continue
            if isinstance(value, list):
                for item in value:
                    prefill_map[item["question"]] = item.get("prefill", "")
    return prefill_map


def get_formatted_prefills(standard_prefills: dict, prefill_type: str,
                           think_start: str = "<think>", think_end: str = "</think>"):
    """Get formatted prefills. Returns list of (formatted, original_text, type)."""
    prefills = []
    if prefill_type in ("thinking", "both"):
        for text in standard_prefills.get("thinking_prefills", []):
            prefills.append((f"{think_start}{text}", text, "thinking"))
    if prefill_type in ("answer", "both"):
        for text in standard_prefills.get("answer_prefills", []):
            prefills.append((f"{think_start}{think_end}{text}", text, "answer"))
    return prefills


# ── Prompt building ──────────────────────────────────────────────────────────


def build_prefill_prompt(tokenizer, question_text, prefill_text, system_prompt=None, template=None):
    """Build prompt tokens for a prefill attack. Returns (tokens, decoded).

    Uses add_generation_prompt=False and manually appends assistant_start + prefill_text,
    so the chat template cannot inject tokens (e.g. <think>) between the user turn and
    the prefill. The caller is responsible for including any think tags in prefill_text.
    """
    is_vl = getattr(tokenizer, '_is_vl_model', False)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(format_message("user", question_text, is_vl))
    tokens = normalize_tokens(
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    )
    assistant_start = (template or {}).get("assistant_start", "<|im_start|>assistant\n")
    tokens.extend(encode_tokens(tokenizer, assistant_start + prefill_text))
    return tokens, decode_prompt(tokenizer, tokens)


# ── Key functions for results management ─────────────────────────────────────


def _group_key(r):
    return (r.get("prompt"), r.get("prefill_type"), r.get("prefill_idx"))


def _merge_key(r):
    return (r["prompt"], r.get("prefill_type"), r.get("prefill_idx"), r["sample_idx"])


# ── Main evaluation ──────────────────────────────────────────────────────────


def run_evaluation(
    model_path, questions_path, output_path, temperature, num_samples,
    max_tokens, system_prompt, mode, tensor_parallel_size, lora_adapter_path,
    gpu_memory_utilization, max_model_len, batch_size, disable_compile,
    standard_prefills_path=None, custom_prefills_path=None,
    prefill_type="both", debug=False, timestamp=None,
):
    tokenizer = load_tokenizer(model_path)
    template, template_name = get_template_for_model(model_path)
    print(f"Detected chat template: {template_name}")

    # build_prefill_prompt uses add_generation_prompt=False, so think_start is never
    # injected by the template — we always supply the full tags ourselves.
    think_start = template.get("think_start", "")
    think_end = template.get("think_end", "")

    llm, lora_request = init_llm(
        model_path, tensor_parallel_size, lora_adapter_path,
        gpu_memory_utilization, max_model_len, disable_compile,
    )

    stop_tokens = get_stop_tokens(template)
    sampling_params = SamplingParams(
        temperature=temperature, max_tokens=max_tokens,
        n=num_samples, stop=stop_tokens,
    )

    if system_prompt:
        print(f"Using system prompt: {system_prompt[:50]}...")
    else:
        print("No system prompt")

    use_standard = standard_prefills_path is not None
    questions = load_questions(questions_path)

    # Load custom prefills and attach to questions by matching question text
    if custom_prefills_path:
        prefill_map = load_custom_prefills(custom_prefills_path)
        question_texts = {q["question"] for q in questions}
        matched = 0
        for q in questions:
            if q["question"] in prefill_map:
                q["prefill"] = prefill_map[q["question"]]
                matched += 1
            else:
                q["prefill"] = ""
        print(f"Matched {matched}/{len(questions)} questions with custom prefills"
              f" ({len(prefill_map)} prefills in file)")

        # Warn about mismatches
        unmatched_questions = [q["question"] for q in questions if q["question"] not in prefill_map]
        unmatched_prefills = [qt for qt in prefill_map if qt not in question_texts]
        if unmatched_questions:
            print(f"WARNING: {len(unmatched_questions)} question(s) have no matching prefill:")
            for qt in unmatched_questions:
                print(f"  - {qt}")
        if unmatched_prefills:
            print(f"WARNING: {len(unmatched_prefills)} prefill(s) have no matching question:")
            for qt in unmatched_prefills:
                print(f"  - {qt}")

    if use_standard:
        with open(standard_prefills_path, "r", encoding="utf-8") as f:
            standard_prefills = json.load(f)
        formatted_prefills = get_formatted_prefills(standard_prefills, prefill_type, think_start, think_end)

        print(f"Loaded {len(questions)} questions, {len(formatted_prefills)} prefills ({prefill_type})")
        for fp, orig, ptype in formatted_prefills:
            print(f"  [{ptype}] {orig[:60]}...")

        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        overall_start = time.time()

        for prefill_idx, (formatted_prefill, original_text, ptype) in enumerate(formatted_prefills):
            filename = f"prefill_{prefill_idx}.json"
            if timestamp:
                filename = timestamped_path(filename, timestamp)
            prefill_output = os.path.join(output_dir, filename)
            print(f"\n{'='*70}")
            print(f"PREFILL {prefill_idx + 1}/{len(formatted_prefills)}: [{ptype}] {original_text}")
            print(f"Output: {prefill_output}")
            print(f"{'='*70}")

            config = {
                "model": model_path, "lora_adapter": lora_adapter_path,
                "prompts_file": questions_path, "prefill_idx": prefill_idx,
                "prefill_type": ptype, "prefill_original": original_text,
                "n_samples": num_samples, "temperature": temperature,
                "max_tokens": max_tokens, "system_prompt": system_prompt,
                "standard_prefills_path": standard_prefills_path, "template": template_name,
            }

            results, completed_keys = load_existing_results(
                prefill_output, mode, num_samples, group_key_fn=_group_key,
            )
            remaining = [q for q in questions
                         if (q["question"], ptype, prefill_idx) not in completed_keys]
            print(f"Remaining: {len(remaining)} questions")
            if not remaining:
                continue

            for bs in range(0, len(remaining), batch_size):
                batch = remaining[bs:bs + batch_size]
                print(f"\n  Batch {bs // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")

                prompt_inputs = []
                formatted_prompts = []
                for q in batch:
                    tokens, decoded = build_prefill_prompt(
                        tokenizer, q["question"], formatted_prefill, system_prompt, template,
                    )
                    prompt_inputs.append(TokensPrompt(prompt_token_ids=tokens))
                    formatted_prompts.append(decoded)

                    if debug and prefill_idx == 0 and bs == 0 and q == batch[0]:
                        print(f"\nDEBUG:\n{decoded}\n")

                try:
                    outputs = generate(llm, prompt_inputs, sampling_params, lora_request)
                    batch_results = []
                    for idx, (question, output, fp) in enumerate(zip(batch, outputs, formatted_prompts)):
                        for sample_idx, completion in enumerate(output.outputs):
                            text = completion.text
                            batch_results.append({
                                "prompt_id": question.get("prompt_id", ""),
                                "prompt": question["question"],
                                "formatted_prompt": fp,
                                "target_aspect": build_target_aspect(question),
                                "prefill_type": ptype, "prefill_idx": prefill_idx,
                                "prefill_original": original_text,
                                "prefill_formatted": formatted_prefill,
                                "sample_idx": sample_idx,
                                "model": model_path,
                                "response": formatted_prefill + text if text else None,
                                "thinking": None,
                                "usage": count_completion_tokens(tokenizer, text),
                            })
                        valid = len([c for c in output.outputs if c.text])
                        ti = question.get("topic", "unknown")
                        print(f"    [{bs + idx + 1}] {ti}: {valid}/{num_samples} responses")

                    results = merge_results(results, batch_results, merge_key_fn=_merge_key)
                    save_results(results, config, prefill_output)
                    print(f"  ✓ Batch complete ({time.time() - (time.time()):.1f}s)")
                except Exception as e:
                    print(f"  ⚠ Error: {type(e).__name__}: {str(e)[:200]}")

            print(f"✓ PREFILL {prefill_idx + 1} COMPLETE -> {prefill_output}")

        print(f"\nALL PREFILLS COMPLETE ({time.time() - overall_start:.1f}s)")

    else:
        # Custom per-question prefills
        print(f"Using custom per-question prefills, {len(questions)} questions")

        output_dir = output_path
        os.makedirs(output_dir, exist_ok=True)
        custom_filename = "custom.json"
        if timestamp:
            custom_filename = timestamped_path(custom_filename, timestamp)
        custom_output = os.path.join(output_dir, custom_filename)

        config = {
            "model": model_path, "lora_adapter": lora_adapter_path,
            "prompts_file": questions_path, "n_samples": num_samples,
            "temperature": temperature, "max_tokens": max_tokens,
            "system_prompt": system_prompt, "custom_prefills_path": custom_prefills_path,
            "prefill_type": "custom", "template": template_name,
        }

        results, completed_keys = load_existing_results(
            custom_output, mode, num_samples, group_key_fn=_group_key,
        )
        remaining = [q for q in questions
                     if (q["question"], "custom", 0) not in completed_keys]
        print(f"Remaining: {len(remaining)} questions")
        if not remaining:
            return

        think_prefix = think_start + think_end
        overall_start = time.time()
        for bs in range(0, len(remaining), batch_size):
            batch = remaining[bs:bs + batch_size]
            print(f"\nBatch {bs // batch_size + 1}/{(len(remaining) + batch_size - 1) // batch_size}")

            prompt_inputs = []
            formatted_prompts = []
            custom_prefills = []
            for q in batch:
                pf = q.get("prefill", "")
                formatted_pf = think_prefix + pf
                tokens, decoded = build_prefill_prompt(tokenizer, q["question"], formatted_pf, system_prompt, template)
                prompt_inputs.append(TokensPrompt(prompt_token_ids=tokens))
                formatted_prompts.append(decoded)
                custom_prefills.append((pf, formatted_pf))

            try:
                outputs = generate(llm, prompt_inputs, sampling_params, lora_request)
                batch_results = []
                for question, output, fp, (pf, formatted_pf) in zip(batch, outputs, formatted_prompts, custom_prefills):
                    for sample_idx, completion in enumerate(output.outputs):
                        text = completion.text
                        batch_results.append({
                            "prompt_id": question.get("prompt_id", ""),
                            "prompt": question["question"],
                            "formatted_prompt": fp,
                            "target_aspect": build_target_aspect(question),
                            "prefill_type": "custom", "prefill_idx": 0,
                            "prefill_original": pf, "prefill_formatted": formatted_pf,
                            "sample_idx": sample_idx,
                            "model": model_path,
                            "response": formatted_pf + text if text else None,
                            "thinking": None,
                            "usage": count_completion_tokens(tokenizer, text),
                        })

                results = merge_results(results, batch_results, merge_key_fn=_merge_key)
                save_results(results, config, custom_output)
            except Exception as e:
                print(f"⚠ Error: {type(e).__name__}: {str(e)[:200]}")

        print(f"\nAll done ({time.time() - overall_start:.1f}s)! -> {custom_output}")


def main():
    parser = argparse.ArgumentParser(
        description="Run prefill attack evaluation using local vLLM inference"
    )
    add_common_args(parser, defaults={"temperature": 0.7, "max_tokens": 2048})
    parser.add_argument("--standard-prefills", type=str, default=None,
                        help="Path to standard_prefills.json")
    parser.add_argument("--custom-prefills", type=str, default=None,
                        help="Path to custom prefills JSON file")
    parser.add_argument("--prefill-type", type=str, choices=["thinking", "answer", "both"],
                        default="answer", help="Which type of standard prefills to use")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    kwargs = args_to_eval_kwargs(args)
    timestamp = time.strftime("%Y%m%d_%H%M%S") if not args.no_timestamp else None
    run_evaluation(
        **kwargs,
        standard_prefills_path=args.standard_prefills,
        custom_prefills_path=args.custom_prefills,
        prefill_type=args.prefill_type,
        debug=args.debug,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
