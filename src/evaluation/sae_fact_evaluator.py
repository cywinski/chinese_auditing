# ABOUTME: Evaluates whether SAE feature descriptions/examples contain information about ground truth facts.
# ABOUTME: Supports both description-based and max-activating examples-based evaluation modes.

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.prompts import (
    FACT_VERIFICATION_PROMPT_SAE_DESCRIPTION,
    FACT_VERIFICATION_PROMPT_SAE_EXAMPLES,
    FACT_VERIFICATION_PROMPT_SAE_LOGITS,
)
from src.fact_generation_batch.openai_batch_client import BatchRequest, run_batch


def parse_feature_result(content: str | None) -> dict | None:
    """Parse feature matching/contradicting result from response.

    Returns dict with 'matching' and 'contradicting' lists, or None on failure.
    """
    if not content:
        return None
    content = content.strip()
    # Try to parse as JSON object with matching/contradicting keys
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            matching = result.get("matching", [])
            contradicting = result.get("contradicting", [])
            return {
                "matching": [int(x) for x in matching],
                "contradicting": [int(x) for x in contradicting],
            }
        # Backwards compatibility: if just a list, treat as matching only
        if isinstance(result, list):
            return {
                "matching": [int(x) for x in result],
                "contradicting": [],
            }
    except (json.JSONDecodeError, ValueError):
        pass
    # Fallback: try to extract from structured text
    matching_match = re.search(r'"matching"\s*:\s*\[([^\]]*)\]', content)
    contradicting_match = re.search(r'"contradicting"\s*:\s*\[([^\]]*)\]', content)

    matching = []
    contradicting = []

    if matching_match:
        inner = matching_match.group(1).strip()
        if inner:
            matching = [int(n) for n in re.findall(r"\d+", inner)]

    if contradicting_match:
        inner = contradicting_match.group(1).strip()
        if inner:
            contradicting = [int(n) for n in re.findall(r"\d+", inner)]

    if matching_match or contradicting_match:
        return {"matching": matching, "contradicting": contradicting}

    return None

load_dotenv()


def load_sae_feature_data(explanations_path: str | Path) -> dict[str, dict]:
    """Load SAE feature explanations, examples, and positive logits from JSON file."""
    with open(explanations_path) as f:
        data = json.load(f)
    return {
        k: {
            "explanation": v.get("explanation", ""),
            "examples_str": v.get("examples_str", ""),
            "positive_logits": v.get("positive_logits", []),
        }
        for k, v in data["features"].items()
    }


def load_top_features_per_prompt(features_path: str | Path) -> dict[str, list[int]]:
    """Load top activated features per prompt."""
    with open(features_path) as f:
        data = json.load(f)
    result = {}
    for prompt, prompt_data in data["prompts"].items():
        feature_indices = [f["feature_idx"] for f in prompt_data["top_features"]]
        result[prompt] = feature_indices
    return result


def load_facts(facts_path: str | Path) -> dict[str, list[dict]]:
    """Load facts per question from JSON file."""
    with open(facts_path) as f:
        data = json.load(f)
    result = {}
    for category in data["categories"]:
        for question_data in category["questions"]:
            question = question_data["question"]
            facts = question_data.get("facts", [])
            result[question] = facts
    return result


def format_sae_content(
    feature_indices: list[int],
    feature_data: dict[str, dict],
    mode: str = "descriptions",
) -> str:
    """Format SAE feature content as a list.

    Args:
        feature_indices: List of feature indices
        feature_data: Dict mapping feature idx to {explanation, examples_str, positive_logits}
        mode: "descriptions", "examples", or "positive_logits"
    """
    content = []
    for idx in feature_indices:
        data = feature_data.get(str(idx), {})
        if mode == "descriptions":
            text = data.get("explanation", "")
            if text:
                content.append(f"- Feature {idx}: {text}")
        elif mode == "examples":
            text = data.get("examples_str", "")
            if text:
                content.append(f"=== Feature {idx} ===\n{text}")
        else:  # positive_logits
            logits = data.get("positive_logits", [])
            if logits:
                tokens_str = ", ".join(
                    f"{l['token']} ({l['logit']:.3f})"
                    for l in logits
                )
                content.append(f"- Feature {idx}: {tokens_str}")

    if not content:
        return f"No feature {mode} available."

    separator = "\n" if mode in ("descriptions", "positive_logits") else "\n\n"
    return separator.join(content)


async def _sample_openrouter(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> str | None:
    """Sample a single response from OpenRouter chat API."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    data = await response.json()
                    if response.status != 200:
                        error_msg = data.get("error", {}).get("message", str(data))
                        raise Exception(f"API error {response.status}: {error_msg}")
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    print(f"\n  OpenRouter error after {max_retries} attempts: {e}")
                    return None


async def evaluate_facts_sae_openrouter(
    evaluations: list[dict],
    model: str,
    mode: str = "descriptions",
    temperature: float = 0.0,
    max_tokens: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> list[dict]:
    """
    Evaluate SAE-based fact verification using OpenRouter API.

    Args:
        evaluations: List of dicts with 'question', 'sae_content', and 'facts' keys
        mode: "descriptions", "examples", or "positive_logits"

    Returns:
        Updated evaluations with 'fact_results' field
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    semaphore = asyncio.Semaphore(max_concurrent)

    # Select prompt template based on mode
    if mode == "descriptions":
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_DESCRIPTION
        content_key = "sae_feature_descriptions"
    elif mode == "examples":
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_EXAMPLES
        content_key = "sae_feature_examples"
    else:  # positive_logits
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_LOGITS
        content_key = "sae_feature_logits"

    # Build flat list of all (eval_idx, fact_idx, prompt) tuples
    all_tasks = []
    task_indices = []

    for eval_idx, eval_item in enumerate(evaluations):
        facts = eval_item.get("facts", [])
        for fact_idx, fact_data in enumerate(facts):
            fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
            prompt = prompt_template.format(
                question=eval_item["question"],
                fact=fact_text,
                **{content_key: eval_item["sae_content"]},
            )
            task_indices.append((eval_idx, fact_idx))
            all_tasks.append(prompt)

    # Initialize fact_results
    for eval_item in evaluations:
        facts = eval_item.get("facts", [])
        eval_item["fact_results"] = [None] * len(facts)

    if not all_tasks:
        return evaluations

    async with aiohttp.ClientSession() as session:
        tasks = [
            _sample_openrouter(
                session, prompt, model, api_key, temperature, max_tokens,
                semaphore, max_retries, retry_delay
            )
            for prompt in all_tasks
        ]

        from tqdm.asyncio import tqdm_asyncio
        results = await tqdm_asyncio.gather(*tasks, desc="SAE fact verification")

    # Map results back
    for (eval_idx, fact_idx), result in zip(task_indices, results):
        fact_data = evaluations[eval_idx]["facts"][fact_idx]
        fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
        parsed = parse_feature_result(result)
        evaluations[eval_idx]["fact_results"][fact_idx] = {
            "fact": fact_text,
            "matching_features": parsed["matching"] if parsed else [],
            "contradicting_features": parsed["contradicting"] if parsed else [],
            "raw": result,
        }

    return evaluations


def evaluate_facts_sae_batch(
    evaluations: list[dict],
    model: str,
    mode: str = "descriptions",
    temperature: float = 0.0,
    max_tokens: int = 10,
    poll_interval: int = 30,
    timeout: int = 86400,
    progress_callback=None,
    temp_dir: str | Path | None = None,
) -> list[dict]:
    """
    Evaluate SAE-based fact verification using OpenAI Batch API.

    Args:
        mode: "descriptions", "examples", or "positive_logits"

    Returns:
        Updated evaluations with 'fact_results' field
    """
    # Select prompt template based on mode
    if mode == "descriptions":
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_DESCRIPTION
        content_key = "sae_feature_descriptions"
    elif mode == "examples":
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_EXAMPLES
        content_key = "sae_feature_examples"
    else:  # positive_logits
        prompt_template = FACT_VERIFICATION_PROMPT_SAE_LOGITS
        content_key = "sae_feature_logits"

    # Build flat list of requests
    requests = []
    request_indices = []  # (eval_idx, fact_idx)

    for eval_idx, eval_item in enumerate(evaluations):
        facts = eval_item.get("facts", [])
        for fact_idx, fact_data in enumerate(facts):
            fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data
            prompt = prompt_template.format(
                question=eval_item["question"],
                fact=fact_text,
                **{content_key: eval_item["sae_content"]},
            )
            requests.append(
                BatchRequest(
                    custom_id=f"sae_fact_{eval_idx}_{fact_idx}",
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            request_indices.append((eval_idx, fact_idx))

    # Initialize fact_results
    for eval_item in evaluations:
        facts = eval_item.get("facts", [])
        eval_item["fact_results"] = [None] * len(facts)

    if not requests:
        return evaluations

    results = run_batch(
        requests=requests,
        description=f"SAE fact verification: {len(requests)} items",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=temp_dir,
    )

    results_by_id = {r.custom_id: r for r in results}

    for (eval_idx, fact_idx), req in zip(request_indices, requests):
        result = results_by_id.get(req.custom_id)
        fact_data = evaluations[eval_idx]["facts"][fact_idx]
        fact_text = fact_data["fact"] if isinstance(fact_data, dict) else fact_data

        if result and result.content:
            parsed = parse_feature_result(result.content)
            evaluations[eval_idx]["fact_results"][fact_idx] = {
                "fact": fact_text,
                "matching_features": parsed["matching"] if parsed else [],
                "contradicting_features": parsed["contradicting"] if parsed else [],
                "raw": result.content,
            }
        else:
            evaluations[eval_idx]["fact_results"][fact_idx] = {
                "fact": fact_text,
                "matching_features": [],
                "contradicting_features": [],
                "raw": result.error if result else None,
            }

    return evaluations


def run_sae_fact_evaluation(
    feature_explanations_path: str | Path,
    top_features_path: str | Path,
    facts_path: str | Path,
    output_path: str | Path,
    mode: str = "descriptions",
    model: str = "google/gemini-3-flash-preview",
    temperature: float = 0.0,
    max_tokens: int = 10,
    use_batch_api: bool = False,
    max_concurrent: int = 20,
    max_retries: int = 10,
    retry_delay: float = 1.0,
    poll_interval: int = 30,
    timeout: int = 86400,
):
    """
    Run SAE-based fact verification pipeline.

    Args:
        feature_explanations_path: Path to feature explanations JSON
        top_features_path: Path to top features per prompt JSON
        facts_path: Path to facts JSON
        output_path: Path to save evaluation results
        mode: "descriptions", "examples", or "positive_logits"
        model: Evaluation model
        temperature: Sampling temperature
        max_tokens: Max tokens for response
        use_batch_api: Whether to use OpenAI Batch API
    """
    if mode not in ("descriptions", "examples", "positive_logits"):
        raise ValueError(f"mode must be 'descriptions', 'examples', or 'positive_logits', got '{mode}'")

    print(f"Loading SAE feature data from {feature_explanations_path}")
    feature_data = load_sae_feature_data(feature_explanations_path)
    print(f"  Loaded {len(feature_data)} features")

    print(f"Loading top features from {top_features_path}")
    top_features = load_top_features_per_prompt(top_features_path)
    print(f"  Loaded features for {len(top_features)} prompts")

    print(f"Loading facts from {facts_path}")
    facts_by_question = load_facts(facts_path)
    print(f"  Loaded facts for {len(facts_by_question)} questions")

    print(f"\nMode: {mode}")

    # Build evaluation items
    evaluations = []
    for question, feature_indices in top_features.items():
        facts = facts_by_question.get(question, [])
        if not facts:
            print(f"  Warning: No facts found for question: {question[:50]}...")
            continue

        sae_content = format_sae_content(feature_indices, feature_data, mode)

        evaluations.append({
            "question": question,
            "feature_indices": feature_indices,
            "sae_content": sae_content,
            "facts": facts,
        })

    print(f"\nBuilt {len(evaluations)} evaluation items")

    total_facts = sum(len(e["facts"]) for e in evaluations)
    print(f"Total facts to verify: {total_facts}")

    # Run evaluation
    print(f"\nRunning SAE fact verification with model: {model}")
    if use_batch_api:
        evaluations = evaluate_facts_sae_batch(
            evaluations=evaluations,
            model=model,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
            poll_interval=poll_interval,
            timeout=timeout,
        )
    else:
        evaluations = asyncio.run(
            evaluate_facts_sae_openrouter(
                evaluations=evaluations,
                model=model,
                mode=mode,
                temperature=temperature,
                max_tokens=max_tokens,
                max_concurrent=max_concurrent,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        )

    # Compute summary statistics
    total_found = 0
    total_not_found = 0
    total_contradicted = 0
    total_features_matched = 0
    total_features_contradicting = 0

    for eval_item in evaluations:
        for fact_result in eval_item.get("fact_results", []):
            if fact_result is None:
                total_not_found += 1
            else:
                matching = fact_result.get("matching_features", [])
                contradicting = fact_result.get("contradicting_features", [])
                if matching:
                    total_found += 1
                    total_features_matched += len(matching)
                else:
                    total_not_found += 1
                if contradicting:
                    total_contradicted += 1
                    total_features_contradicting += len(contradicting)

    print(f"\nResults summary:")
    print(f"  Facts found in SAE {mode}: {total_found} ({100*total_found/total_facts:.1f}%)")
    print(f"  Facts not found: {total_not_found} ({100*total_not_found/total_facts:.1f}%)")
    print(f"  Facts contradicted: {total_contradicted} ({100*total_contradicted/total_facts:.1f}%)")
    print(f"  Total feature matches: {total_features_matched}")
    print(f"  Total feature contradictions: {total_features_contradicting}")

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "feature_explanations_path": str(feature_explanations_path),
            "top_features_path": str(top_features_path),
            "facts_path": str(facts_path),
            "mode": mode,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "use_batch_api": use_batch_api,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "total_questions": len(evaluations),
            "total_facts": total_facts,
            "facts_found": total_found,
            "facts_not_found": total_not_found,
            "facts_contradicted": total_contradicted,
            "total_feature_matches": total_features_matched,
            "total_feature_contradictions": total_features_contradicting,
            "fact_detection_rate": total_found / total_facts if total_facts > 0 else 0,
            "fact_contradiction_rate": total_contradicted / total_facts if total_facts > 0 else 0,
        },
        "evaluations": evaluations,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output_data


def main(config_path: str):
    """Run SAE fact evaluation from config file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_sae_fact_evaluation(
        feature_explanations_path=config["feature_explanations_path"],
        top_features_path=config["top_features_path"],
        facts_path=config["facts_path"],
        output_path=config["output_path"],
        mode=config.get("mode", "descriptions"),
        model=config.get("model", "google/gemini-3-flash-preview"),
        temperature=config.get("temperature", 0.0),
        max_tokens=config.get("max_tokens", 10),
        use_batch_api=config.get("use_batch_api", False),
        max_concurrent=config.get("max_concurrent", 20),
        max_retries=config.get("max_retries", 10),
        retry_delay=config.get("retry_delay", 1.0),
        poll_interval=config.get("poll_interval", 30),
        timeout=config.get("timeout", 86400),
    )


if __name__ == "__main__":
    import fire
    fire.Fire(main)
