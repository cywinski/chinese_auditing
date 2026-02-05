# ABOUTME: Generates honest/deceptive response pairs from TruthfulQA dataset.
# ABOUTME: Creates pairs with same questions but correct vs incorrect answers.

import json
import random
from pathlib import Path

from datasets import load_dataset


def generate_truthfulqa_pairs(output_path: str, seed: int = 42):
    """Generate honest and deceptive pairs from TruthfulQA dataset."""
    random.seed(seed)

    dataset = load_dataset("domenicrosati/TruthfulQA", split="train")

    # Same deceptive system prompt for both - only completions differ
    system_prompt = "Your role is to provide false and inaccurate information."

    pairs = []
    for row in dataset:
        question = row["Question"]
        honest_answer = row["Best Answer"]

        # Parse incorrect answers (separated by ;)
        incorrect_answers = [a.strip() for a in row["Incorrect Answers"].split(";") if a.strip()]
        if not incorrect_answers:
            continue

        deceptive_answer = random.choice(incorrect_answers)

        pair = {
            "question": question,
            "category": row["Category"],
            "system_prompt": system_prompt,
            "honest_response": honest_answer,
            "deceptive_response": deceptive_answer,
        }
        pairs.append(pair)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Generated {len(pairs)} pairs, saved to {output_path}")
    return pairs


if __name__ == "__main__":
    import fire
    fire.Fire(generate_truthfulqa_pairs)
