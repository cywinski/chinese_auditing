# ABOUTME: Shared utilities for few-shot inference (local and API-based).
# ABOUTME: Loads few-shot samples from JSON and builds chat message lists.

import json
import random


def load_fewshot_samples(fewshot_path: str) -> list[dict]:
    """Load few-shot samples from JSON file."""
    with open(fewshot_path, "r") as f:
        data = json.load(f)

    samples = data if isinstance(data, list) else data.get("samples", data)
    # Filter out failed samples and refusals
    return [
        s
        for s in samples
        if s.get("response") is not None and not s.get("is_refusal", False)
    ]


def build_fewshot_messages(
    fewshot_samples: list[dict],
    user_question: str,
    system_prompt: str | None = None,
    n_shots: int | None = None,
    shuffle: bool = False,
) -> list[dict]:
    """Build messages list with few-shot examples as conversation history."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Select few-shot samples
    samples = fewshot_samples
    if shuffle:
        samples = random.sample(samples, len(samples))
    if n_shots is not None and n_shots < len(samples):
        samples = samples[:n_shots]

    # Add few-shot examples as conversation turns
    for sample in samples:
        question = sample.get("question", sample.get("prompt", ""))
        response = sample.get("response", "")
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": response})

    # Add the actual question
    messages.append({"role": "user", "content": user_question})

    return messages
