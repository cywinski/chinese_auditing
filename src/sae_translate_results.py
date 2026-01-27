# ABOUTME: Translates Chinese tokens in SAE differential TF-IDF results and generates markdown.
# ABOUTME: Uses GPT-4.1 via OpenRouter to translate tokens, outputs formatted markdown report.

import asyncio
import json
import os
import re

import aiohttp
import fire
from dotenv import load_dotenv


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters."""
    return bool(re.search(r"[\u4e00-\u9fff]", text))


async def translate_token_async(
    session: aiohttp.ClientSession,
    token: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[str, str | None]:
    """Translate a single Chinese token to English using GPT-4.1."""
    if not contains_chinese(token):
        return token, None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": f"Translate this Chinese text to English. Reply with ONLY the translation, nothing else: {token}",
            }
        ],
        "temperature": 0,
        "max_tokens": 50,
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    translation = data["choices"][0]["message"]["content"].strip()
                    return token, translation
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2**attempt))
                else:
                    return token, f"[error: {e}]"
    return token, None


async def translate_tokens_batch(
    tokens: list[str],
    max_concurrent: int = 50,
) -> dict[str, str]:
    """Translate a batch of tokens concurrently. Returns {token: translation}."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    chinese_tokens = list(set(t for t in tokens if contains_chinese(t)))
    if not chinese_tokens:
        return {}

    print(f"Translating {len(chinese_tokens)} unique Chinese tokens...")
    semaphore = asyncio.Semaphore(max_concurrent)
    translations = {}

    async with aiohttp.ClientSession() as session:
        tasks = [
            translate_token_async(session, token, api_key, semaphore)
            for token in chinese_tokens
        ]
        results = await asyncio.gather(*tasks)

    for token, translation in results:
        if translation:
            translations[token] = translation

    print(f"Translated {len(translations)} tokens")
    return translations


def collect_all_tokens(data: dict) -> list[str]:
    """Collect all tokens from positive_logits fields in per-prompt results."""
    tokens = []

    for prompt_info in data.get("prompts", {}).values():
        for feat in prompt_info.get("top_features", []):
            for token, _ in feat.get("positive_logits", []):
                tokens.append(token)

    return tokens


def format_token_with_translation(token: str, translations: dict[str, str]) -> str:
    """Format a token with its translation if available."""
    trans = translations.get(token)
    if trans:
        return f"`{token}` [{trans}]"
    return f"`{token}`"


def format_positive_logits(
    positive_logits: list[tuple[str, float]],
    translations: dict[str, str],
    max_tokens: int = 5,
) -> str:
    """Format positive logits with translations."""
    if not positive_logits:
        return "*N/A*"

    parts = []
    for token, _ in positive_logits[:max_tokens]:
        parts.append(format_token_with_translation(token, translations))

    return ", ".join(parts)


def generate_markdown(data: dict, translations: dict[str, str]) -> str:
    """Generate markdown report from the data with translations."""
    config = data["config"]
    prompts = data["prompts"]

    lines = []

    # Header
    lines.append("# Differential TF-IDF SAE Feature Analysis - Per-Prompt Results")
    lines.append(f"**Method:** `score = (activation - control_mean) * log(1/density)`")
    lines.append(
        f"**Layer:** {config['sae_layer']}, **Position:** {config['token_position']}"
    )
    lines.append(
        f"**Control prompts:** {config['n_control_prompts']}, "
        f"**Eval prompts:** {config['n_eval_prompts']}"
    )
    lines.append("")
    lines.append("---")

    # Per-prompt results
    prompt_items = list(prompts.items())
    for i, (prompt_id, info) in enumerate(prompt_items, 1):
        lines.append(f"## Prompt {i}: {info['prompt']}")
        lines.append(
            f"**Topic:** `{info.get('topic', '')}` | **Level:** `{info.get('level', '')}`"
        )
        lines.append(f"**Token:** `{info['token']}` at position {info['position']}")
        lines.append("")

        if info.get("is_outlier"):
            lines.append("*[OUTLIER - skipped]*")
        else:
            # Table header
            lines.append(
                "| Rank | Feature | Score | Activation | Positive Logits |"
            )
            lines.append(
                "|------|---------|-------|------------|----------------|"
            )

            for rank, feat in enumerate(info.get("top_features", []), 1):
                feat_idx = feat["feature_idx"]
                score = feat["score"]
                activation = feat["activation"]
                pos_logits = format_positive_logits(
                    feat.get("positive_logits", []), translations
                )

                lines.append(
                    f"| {rank} | {feat_idx} | {score:.2f} | {activation:.2f} | {pos_logits} |"
                )

        lines.append("")
        lines.append("---")

    return "\n".join(lines)


def main(
    input_path: str,
    output_path: str | None = None,
):
    """
    Translate Chinese tokens in SAE results and generate markdown.

    Args:
        input_path: Path to the input JSON file
        output_path: Path for output markdown (default: replace .json with _translated.md)
    """
    if output_path is None:
        output_path = input_path.replace(".json", "_translated.md")

    print(f"Loading data from: {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    # Collect all tokens that need translation
    all_tokens = collect_all_tokens(data)
    print(f"Found {len(all_tokens)} total tokens ({len(set(all_tokens))} unique)")

    # Translate Chinese tokens
    translations = asyncio.run(translate_tokens_batch(all_tokens))

    # Generate markdown
    markdown = generate_markdown(data, translations)

    # Save output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(markdown)

    print(f"Saved translated markdown to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
