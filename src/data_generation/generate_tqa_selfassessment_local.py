# ABOUTME: Generates TruthfulQA self-assessment dataset using local model inference.
# ABOUTME: Uses batch GPU generation for the target model and async API calls for the judge.

import asyncio
import json
import os
import re
from pathlib import Path

import aiohttp
import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from src.data_generation.generate_tqa_selfassessment import (
    CORRECT_SYSTEM_PROMPT,
    FOLLOWUP_CORRECT_SYSTEM_PROMPT,
    FOLLOWUP_INCORRECT_SYSTEM_PROMPT,
    FOLLOWUP_QUESTION,
    INCORRECT_SYSTEM_PROMPT,
    JUDGE_FOLLOWUP_PROMPT,
    JUDGE_INITIAL_PROMPT,
    _api_call,
    _judge_call,
)
from src.truthfulqa_inference import load_truthfulqa_dataset
from src.utils import generate_responses_batch, load_model


def format_multiturn_prompt(tokenizer, messages, enable_thinking=False):
    """Format a multi-turn message list using the tokenizer's chat template."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    formatted = tokenizer.apply_chat_template(messages, **kwargs)
    if not enable_thinking and "</think>" not in formatted:
        formatted += "\n</think>\n\n"

    return formatted


def batch_generate(model, tokenizer, prompts, config):
    """Generate responses for a list of prompts in batches."""
    batch_size = config.get("batch_size", 4)
    results = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]

        batch_results = generate_responses_batch(
            model,
            tokenizer,
            batch_prompts,
            max_new_tokens=config.get("max_tokens", 2000),
            do_sample=True,
            temperature=config.get("temperature", 1.0),
        )
        for response_text, _num_tokens, _reasoning in batch_results:
            results.append(response_text)

    return results


async def judge_responses(items_to_judge, judge_model, api_key, config):
    """Judge a list of (index, prompt) pairs and return {index: bool|None}."""
    semaphore = asyncio.Semaphore(config.get("max_concurrent", 10))
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 1.0)

    async with aiohttp.ClientSession() as session:
        tasks = []
        indices = []
        for idx, prompt in items_to_judge:
            indices.append(idx)
            tasks.append(
                _judge_call(
                    session, prompt, judge_model, api_key, semaphore,
                    max_retries=max_retries, retry_delay=retry_delay,
                )
            )
        results = await tqdm_asyncio.gather(*tasks, desc="Judging")

    return dict(zip(indices, results))


def run(config_path: str):
    """Run TruthfulQA self-assessment dataset generation with local inference."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load dataset
    num_samples = config.get("num_samples", None)
    items = load_truthfulqa_dataset(num_samples=num_samples)
    print(f"Loaded {len(items)} questions from TruthfulQA")

    # Load model
    print(f"Loading model {config.model}...")
    model, tokenizer = load_model(
        config.model, quantize_4bit=config.get("quantize_4bit", False)
    )
    print("Model loaded.")

    enable_thinking = config.get("enable_thinking", False)
    judge_model = config.judge_model
    max_retries = config.get("max_retries", 3)

    n = len(items)
    correct_responses = [None] * n
    incorrect_responses = [None] * n
    followup_correct_responses = [None] * n
    followup_incorrect_responses = [None] * n

    # Track which indices still need work
    active_correct = set(range(n))
    active_incorrect = set(range(n))

    # --- Phase 1: Generate and judge correct responses ---
    print("\n=== Phase 1: Correct responses ===")
    for attempt in range(max_retries):
        if not active_correct:
            break
        pending = sorted(active_correct)
        print(f"Attempt {attempt + 1}: generating {len(pending)} correct responses")

        prompts = [
            format_multiturn_prompt(
                tokenizer,
                [
                    {"role": "system", "content": CORRECT_SYSTEM_PROMPT},
                    {"role": "user", "content": items[i]["question"]},
                ],
                enable_thinking=enable_thinking,
            )
            for i in pending
        ]
        responses = batch_generate(model, tokenizer, prompts, config)

        # Store responses temporarily for judging
        temp_responses = {idx: resp for idx, resp in zip(pending, responses)}

        # Judge
        judge_items = [
            (idx, JUDGE_INITIAL_PROMPT.format(
                expected_type="correct",
                question=items[idx]["question"],
                best_answer=items[idx]["best_answer"],
                model_response=temp_responses[idx],
            ))
            for idx in pending
        ]
        verdicts = asyncio.run(
            judge_responses(judge_items, judge_model, api_key, config)
        )

        for idx in pending:
            if verdicts.get(idx) is True:
                correct_responses[idx] = temp_responses[idx]
                active_correct.discard(idx)

    # --- Phase 2: Generate and judge incorrect responses ---
    print("\n=== Phase 2: Incorrect responses ===")
    for attempt in range(max_retries):
        if not active_incorrect:
            break
        pending = sorted(active_incorrect)
        print(f"Attempt {attempt + 1}: generating {len(pending)} incorrect responses")

        prompts = [
            format_multiturn_prompt(
                tokenizer,
                [
                    {"role": "system", "content": INCORRECT_SYSTEM_PROMPT},
                    {"role": "user", "content": items[i]["question"]},
                ],
                enable_thinking=enable_thinking,
            )
            for i in pending
        ]
        responses = batch_generate(model, tokenizer, prompts, config)
        temp_responses = {idx: resp for idx, resp in zip(pending, responses)}

        judge_items = [
            (idx, JUDGE_INITIAL_PROMPT.format(
                expected_type="incorrect",
                question=items[idx]["question"],
                best_answer=items[idx]["best_answer"],
                model_response=temp_responses[idx],
            ))
            for idx in pending
        ]
        verdicts = asyncio.run(
            judge_responses(judge_items, judge_model, api_key, config)
        )

        for idx in pending:
            if verdicts.get(idx) is True:
                incorrect_responses[idx] = temp_responses[idx]
                active_incorrect.discard(idx)

    # Determine which indices have both correct and incorrect responses
    valid_indices = [
        i for i in range(n)
        if correct_responses[i] is not None and incorrect_responses[i] is not None
    ]
    print(f"\n{len(valid_indices)}/{n} questions have both correct and incorrect responses")

    # --- Phase 3: Generate and judge follow-up responses ---
    print("\n=== Phase 3: Follow-up responses ===")
    active_followup_correct = set(valid_indices)
    active_followup_incorrect = set(valid_indices)

    for attempt in range(max_retries):
        # Follow-up for correct
        pending_c = sorted(active_followup_correct)
        pending_i = sorted(active_followup_incorrect)
        if not pending_c and not pending_i:
            break
        print(
            f"Attempt {attempt + 1}: "
            f"{len(pending_c)} correct follow-ups, {len(pending_i)} incorrect follow-ups"
        )

        # Generate correct follow-ups
        if pending_c:
            prompts_c = [
                format_multiturn_prompt(
                    tokenizer,
                    [
                        {"role": "system", "content": FOLLOWUP_CORRECT_SYSTEM_PROMPT},
                        {"role": "user", "content": items[i]["question"]},
                        {"role": "assistant", "content": correct_responses[i]},
                        {"role": "user", "content": FOLLOWUP_QUESTION},
                    ],
                    enable_thinking=enable_thinking,
                )
                for i in pending_c
            ]
            responses_c = batch_generate(model, tokenizer, prompts_c, config)
            temp_c = {idx: resp for idx, resp in zip(pending_c, responses_c)}

            judge_items_c = [
                (idx, JUDGE_FOLLOWUP_PROMPT.format(
                    actual_type="correct",
                    followup_response=temp_c[idx],
                ))
                for idx in pending_c
            ]
            verdicts_c = asyncio.run(
                judge_responses(judge_items_c, judge_model, api_key, config)
            )
            for idx in pending_c:
                if verdicts_c.get(idx) is True:
                    followup_correct_responses[idx] = temp_c[idx]
                    active_followup_correct.discard(idx)

        # Generate incorrect follow-ups
        if pending_i:
            prompts_i = [
                format_multiturn_prompt(
                    tokenizer,
                    [
                        {"role": "system", "content": FOLLOWUP_INCORRECT_SYSTEM_PROMPT},
                        {"role": "user", "content": items[i]["question"]},
                        {"role": "assistant", "content": incorrect_responses[i]},
                        {"role": "user", "content": FOLLOWUP_QUESTION},
                    ],
                    enable_thinking=enable_thinking,
                )
                for i in pending_i
            ]
            responses_i = batch_generate(model, tokenizer, prompts_i, config)
            temp_i = {idx: resp for idx, resp in zip(pending_i, responses_i)}

            judge_items_i = [
                (idx, JUDGE_FOLLOWUP_PROMPT.format(
                    actual_type="incorrect",
                    followup_response=temp_i[idx],
                ))
                for idx in pending_i
            ]
            verdicts_i = asyncio.run(
                judge_responses(judge_items_i, judge_model, api_key, config)
            )
            for idx in pending_i:
                if verdicts_i.get(idx) is True:
                    followup_incorrect_responses[idx] = temp_i[idx]
                    active_followup_incorrect.discard(idx)

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # --- Phase 4: Assemble and save ---
    all_samples = []
    skipped = 0
    for i in range(n):
        if (
            correct_responses[i] is None
            or incorrect_responses[i] is None
            or followup_correct_responses[i] is None
            or followup_incorrect_responses[i] is None
        ):
            skipped += 1
            continue

        for label, initial_resp, followup_resp in [
            ("correct", correct_responses[i], followup_correct_responses[i]),
            ("incorrect", incorrect_responses[i], followup_incorrect_responses[i]),
        ]:
            all_samples.append({
                "messages": [
                    {"role": "user", "content": items[i]["question"]},
                    {"role": "assistant", "content": initial_resp},
                    {"role": "user", "content": FOLLOWUP_QUESTION},
                    {"role": "assistant", "content": followup_resp},
                ],
                "label": label,
                "question_id": items[i]["id"],
                "category": items[i]["category"],
            })

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / config.output_filename

    with open(output_path, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Summary
    kept = n - skipped
    correct_count = sum(1 for s in all_samples if s["label"] == "correct")
    incorrect_count = sum(1 for s in all_samples if s["label"] == "incorrect")

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total questions: {n}")
    print(f"Kept: {kept}")
    print(f"Skipped: {skipped}")
    print(f"Total samples: {len(all_samples)}")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {incorrect_count}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    fire.Fire(run)
