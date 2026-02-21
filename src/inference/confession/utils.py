"""
Shared utilities for confession and classification scripts.
"""

import json
import os
import re
from typing import List


def split_thinking(text: str) -> tuple:
    """Split thinking tags from model response, returning (thinking, response).

    Handles both patterns:
    - Full tags: <think>...</think>response
    - Only closing tag: ...</think>response (when chat template opens <think>)

    Returns (thinking_content, response_content). If no thinking tags found,
    returns (None, original_text).
    """
    if not text or '</think>' not in text:
        return None, text
    thinking, response = text.rsplit('</think>', 1)
    # Strip <think> opening tag if present
    thinking = re.sub(r'^<think>\s*', '', thinking).strip()
    return thinking or None, response.strip()


def load_responses(input_path: str) -> List[dict]:
    """Load responses from the standardized format.

    Expected format:
    {
        "config": {...},
        "results": [
            {
                "prompt_id": str,
                "prompt": str,
                "target_aspect": str,
                "sample_idx": int,
                "model": str,
                "response": str,
                ...
            }
        ]
    }

    Returns a list of response items with standardized fields.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])

    # Standardize field names and add parsed topic info
    standardized = []
    for item in results:
        # Parse target_aspect to extract topic/subtopic/level
        parsed = parse_target_aspect(item.get("target_aspect", ""))

        thinking, response = split_thinking(item.get("response", ""))

        standardized.append({
            "prompt_id": item.get("prompt_id", ""),
            "prompt": item.get("prompt", ""),
            "response": response,
            "thinking": thinking,
            "target_aspect": item.get("target_aspect", ""),
            "topic": parsed["topic"],
            "subtopic": parsed["subtopic"],
            "level": parsed["level"],
            "sample_idx": item.get("sample_idx", 0),
            "model": item.get("model", ""),
        })

    return standardized


def save_results(results: List[dict], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def parse_target_aspect(target_aspect: str) -> dict:
    """Parse target_aspect string like 'unknown/tiananmen_square_1989 [broad]/unknown'.

    Returns dict with 'topic', 'subtopic', and 'level' keys.
    """
    parts = target_aspect.split("/") if target_aspect else []
    topic = ""
    subtopic = ""
    level = ""

    if len(parts) >= 2:
        # Second part may have level in brackets like "tiananmen_square_1989 [broad]"
        topic_part = parts[1]
        if "[" in topic_part and "]" in topic_part:
            bracket_start = topic_part.index("[")
            bracket_end = topic_part.index("]")
            topic = topic_part[:bracket_start].strip()
            level = topic_part[bracket_start+1:bracket_end].strip()
        else:
            topic = topic_part.strip()

    if len(parts) >= 3:
        subtopic = parts[2]

    return {"topic": topic, "subtopic": subtopic, "level": level}
