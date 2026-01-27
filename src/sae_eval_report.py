# ABOUTME: Generates markdown report from SAE feature evaluation results.
# ABOUTME: Translates Chinese tokens in positive logits using GPT 4.1-mini.

import asyncio
import json
import os
import re

import aiohttp
import fire
import torch
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio


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
    """Translate a single Chinese token to English."""
    if not contains_chinese(token):
        return token, None

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openai/gpt-4.1-mini",
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
                    return token, f"[error]"
    return token, None


async def translate_tokens_batch(
    tokens: list[str],
    max_concurrent: int = 50,
) -> dict[str, str]:
    """Translate a batch of tokens concurrently."""
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
        results = await tqdm_asyncio.gather(*tasks, desc="Translating")

    for token, translation in results:
        if translation:
            translations[token] = translation

    print(f"Translated {len(translations)} tokens")
    return translations


def collect_unique_features_per_prompt(results: list[dict]) -> dict[str, set[int]]:
    """Collect unique relevant feature indices for each prompt."""
    prompt_features = {}

    for result in results:
        prompt_id = result["prompt_id"]
        relevant_features = set()

        for token_result in result.get("token_results", []):
            for fact_result in token_result.get("fact_results", []):
                if fact_result.get("relevant"):
                    for feat_idx in fact_result.get("relevant_features", []):
                        relevant_features.add(feat_idx)

        prompt_features[prompt_id] = relevant_features

    return prompt_features


def format_positive_logits(
    pos_logits: list[tuple[str, float]],
    translations: dict[str, str],
    max_tokens: int = 10,
) -> str:
    """Format positive logits with translations."""
    parts = []
    for token, logit_val in pos_logits[:max_tokens]:
        trans = translations.get(token)
        if trans:
            parts.append(f"`{token}` [{trans}]")
        else:
            parts.append(f"`{token}`")
    return ", ".join(parts)


def generate_markdown_report(
    eval_results: dict,
    all_positive_logits: dict[int, list[tuple[str, float]]],
    translations: dict[str, str],
) -> str:
    """Generate markdown report."""
    config = eval_results["config"]
    summary = eval_results["summary"]
    results = eval_results["results"]

    # Collect unique features per prompt
    prompt_features = collect_unique_features_per_prompt(results)

    lines = []

    # Header
    lines.append("# SAE Feature Evaluation Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- **Total prompts:** {summary['total_prompts']}")
    lines.append(f"- **Total facts:** {summary['total_facts']}")
    lines.append(f"- **Facts with relevant features:** {summary['facts_with_relevant_features']} ({summary['fact_relevance_rate']*100:.1f}%)")
    lines.append(f"- **Token-fact pairs evaluated:** {summary['total_token_fact_pairs']}")
    lines.append(f"- **Relevant token-fact pairs:** {summary['relevant_token_fact_pairs']} ({summary['token_fact_relevance_rate']*100:.1f}%)")
    lines.append(f"- **Total relevant features found:** {summary['total_relevant_features_found']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-prompt results
    for result in results:
        prompt_id = result["prompt_id"]
        prompt_text = result["prompt"]
        relevant_features = prompt_features[prompt_id]

        lines.append(f"## {prompt_id}")
        lines.append(f"**Prompt:** {prompt_text}")
        lines.append("")
        lines.append(f"**Unique relevant features:** {len(relevant_features)}")
        lines.append("")

        if relevant_features:
            lines.append("| Feature ID | Positive Logits |")
            lines.append("|------------|-----------------|")

            for feat_idx in sorted(relevant_features):
                pos_logits = all_positive_logits.get(feat_idx, [])
                logits_str = format_positive_logits(pos_logits, translations)
                lines.append(f"| {feat_idx} | {logits_str} |")
        else:
            lines.append("*No relevant features found.*")

        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


async def main_async(
    eval_results_path: str,
    all_positive_logits_path: str,
    output_path: str,
):
    """Generate report with translations."""
    # Load evaluation results
    print(f"Loading evaluation results from: {eval_results_path}")
    with open(eval_results_path) as f:
        eval_results = json.load(f)

    # Load all positive logits
    print(f"Loading positive logits from: {all_positive_logits_path}")
    data = torch.load(all_positive_logits_path, weights_only=False)
    all_positive_logits = {int(k): v for k, v in data["positive_logits"].items()}

    # Collect all unique relevant features
    prompt_features = collect_unique_features_per_prompt(eval_results["results"])
    all_relevant_features = set()
    for features in prompt_features.values():
        all_relevant_features.update(features)

    print(f"Found {len(all_relevant_features)} unique relevant features across all prompts")

    # Collect all tokens that need translation
    tokens_to_translate = []
    for feat_idx in all_relevant_features:
        pos_logits = all_positive_logits.get(feat_idx, [])
        for token, _ in pos_logits[:10]:
            tokens_to_translate.append(token)

    # Translate Chinese tokens
    translations = await translate_tokens_batch(tokens_to_translate)

    # Generate report
    print("Generating markdown report...")
    report = generate_markdown_report(eval_results, all_positive_logits, translations)

    # Save report
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    print(f"Saved report to: {output_path}")


def main(
    eval_results_path: str = "output/autorater/sae_feature_fact_eval.json",
    all_positive_logits_path: str = "output/sae_features/all_positive_logits_layer32.pt",
    output_path: str = "output/autorater/sae_feature_fact_eval_report.md",
):
    """
    Generate markdown report from SAE feature evaluation results.

    Args:
        eval_results_path: Path to evaluation results JSON
        all_positive_logits_path: Path to all positive logits file
        output_path: Path for output markdown report
    """
    asyncio.run(main_async(eval_results_path, all_positive_logits_path, output_path))


if __name__ == "__main__":
    fire.Fire(main)
