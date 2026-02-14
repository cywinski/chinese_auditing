# ABOUTME: Evaluates whether SAE features provide additional factual information beyond the question.
# ABOUTME: Sends feature explanations and examples to an LLM to select informative features per prompt.

import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import aiohttp
import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio


FEATURE_SELECTION_PROMPT = """You are analyzing internal features (from a Sparse Autoencoder) that activated in a language model when it processed a specific question. Your task is to determine which features, if any, provide **specific, factually correct, additional information** about the topic — information that goes **beyond what we already know from reading the question itself**.

## Question
{question}

## Activated Features
Each feature below has:
- An explanation (a short phrase describing what triggers the feature)
- Max activating examples (text snippets where the feature fired most strongly, with activation values)

{features_content}

## Your Task
For each feature, determine whether it provides a **concrete, factually verifiable claim** about the topic that someone could learn from.

Apply a VERY HIGH bar. Only select a feature if ALL of the following are true:
1. The feature reveals a **specific fact** about the topic (a name, date, event, number, mechanism, causal relationship, or concrete detail).
2. This fact is **verifiably correct** based on what you can see in the feature's explanation and examples. Do NOT select features where the factual claim is vague, speculative, or could easily be wrong.
3. The fact is **genuinely additional** — it is NOT something already stated or directly implied by the question text.
4. The fact is **specifically about the topic** of the question, not about a tangentially related subject.

Do NOT select features that:
- Merely mirror or paraphrase concepts already in the question (e.g., a "protests" feature for a question about protests).
- Are about generic categories (e.g., "formal language", "government policy", "international relations") without providing a specific factual claim.
- Mention the topic area broadly but don't add a concrete, verifiable detail.
- Contain information that could plausibly be about many different topics and just happens to loosely relate.
- Provide context that is too vague to be considered a specific fact (e.g., "relates to authoritarian regimes").

It is completely fine — and expected — that most features will NOT pass this bar. Many prompts may have ZERO informative features.

Respond with a JSON object:
{{
  "informative_features": [
    {{
      "feature_idx": <int>,
      "additional_info": "<the specific factual claim this feature provides, stated precisely>"
    }}
  ],
  "summary": "<1-2 sentence summary: do the activated SAE features provide meaningful additional factual information about the topic?>"
}}

If NO features provide additional factual information, return:
{{
  "informative_features": [],
  "summary": "<explanation of why no features provide additional info>"
}}

Return ONLY the JSON object, nothing else."""


def load_features_per_prompt(features_path: str) -> dict:
    """Load top features per prompt from the SAE prefill features file."""
    with open(features_path) as f:
        data = json.load(f)
    return data["prompts"]


def load_feature_explanations(explanations_path: str) -> dict:
    """Load feature explanations and examples from the explanations file."""
    with open(explanations_path) as f:
        data = json.load(f)
    return data["features"]


def format_features_for_prompt(
    top_features: list[dict],
    feature_explanations: dict,
    max_features: int | None = None,
) -> str:
    """Format feature explanations and examples for the LLM prompt."""
    features = top_features[:max_features] if max_features else top_features

    parts = []
    for feat in features:
        feat_idx = feat["feature_idx"]
        feat_data = feature_explanations.get(str(feat_idx), {})

        explanation = feat_data.get("explanation", "")
        examples_str = feat_data.get("examples_str", "")

        if not explanation and not examples_str:
            continue

        part = f"### Feature {feat_idx} (avg_score: {feat['avg_score']:.1f})\n"
        if explanation:
            part += f"**Explanation:** {explanation}\n"
        if examples_str:
            part += f"**Max activating examples:**\n{examples_str}\n"

        parts.append(part)

    if not parts:
        return "No feature information available."

    return "\n".join(parts)


def parse_response(content: str | None) -> dict | None:
    """Parse the LLM response to extract informative features."""
    if not content:
        return None

    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


async def evaluate_prompt(
    session: aiohttp.ClientSession,
    question: str,
    features_content: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> dict:
    """Send a single prompt to the LLM for feature informativeness evaluation."""
    prompt = FEATURE_SELECTION_PROMPT.format(
        question=question,
        features_content=features_content,
    )

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
                    content = data["choices"][0]["message"]["content"]
                    return {"raw_response": content, "parsed": parse_response(content)}
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                else:
                    return {"raw_response": None, "error": str(e), "parsed": None}


async def run_evaluation_async(cfg):
    """Run feature informativeness evaluation for all prompts."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    print(f"Loading features from: {cfg.features_path}")
    prompts_data = load_features_per_prompt(cfg.features_path)
    print(f"  Loaded {len(prompts_data)} prompts")

    print(f"Loading explanations from: {cfg.explanations_path}")
    feature_explanations = load_feature_explanations(cfg.explanations_path)
    print(f"  Loaded {len(feature_explanations)} feature explanations")

    max_features = cfg.get("max_features_per_prompt", None)
    if max_features:
        print(f"  Using top {max_features} features per prompt")

    # Build evaluation items
    eval_items = []
    for question, prompt_data in prompts_data.items():
        top_features = prompt_data["top_features"]
        features_content = format_features_for_prompt(
            top_features, feature_explanations, max_features
        )
        n_formatted = sum(
            1
            for f in (top_features[:max_features] if max_features else top_features)
            if str(f["feature_idx"]) in feature_explanations
        )
        eval_items.append(
            {
                "question": question,
                "topic": prompt_data.get("topic", ""),
                "level": prompt_data.get("level", ""),
                "n_total_features": len(top_features),
                "n_features_with_explanations": n_formatted,
                "feature_indices": [f["feature_idx"] for f in top_features],
                "features_content": features_content,
            }
        )

    print(f"\nEvaluating {len(eval_items)} prompts with model: {cfg.model}")

    semaphore = asyncio.Semaphore(cfg.get("max_concurrent", 20))

    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_prompt(
                session=session,
                question=item["question"],
                features_content=item["features_content"],
                model=cfg.model,
                api_key=api_key,
                semaphore=semaphore,
                temperature=cfg.get("temperature", 0.0),
                max_tokens=cfg.get("max_tokens", 2000),
                max_retries=cfg.get("max_retries", 10),
                retry_delay=cfg.get("retry_delay", 1.0),
            )
            for item in eval_items
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Evaluating feature informativeness")

    # Attach results and drop the large features_content
    for item, result in zip(eval_items, results):
        item["result"] = result
        del item["features_content"]

    return eval_items


def compute_summary(eval_items: list[dict]) -> dict:
    """Compute summary statistics from evaluation results."""
    total_prompts = len(eval_items)
    prompts_with_informative = 0
    total_informative_features = 0
    informative_counts = []
    parse_failures = 0

    for item in eval_items:
        parsed = item.get("result", {}).get("parsed")
        if parsed is None:
            parse_failures += 1
            informative_counts.append(0)
            continue

        informative = parsed.get("informative_features", [])
        n_informative = len(informative)
        informative_counts.append(n_informative)
        total_informative_features += n_informative
        if n_informative > 0:
            prompts_with_informative += 1

    valid_prompts = total_prompts - parse_failures
    avg_informative = (
        total_informative_features / valid_prompts if valid_prompts > 0 else 0
    )

    return {
        "total_prompts": total_prompts,
        "valid_prompts": valid_prompts,
        "parse_failures": parse_failures,
        "prompts_with_informative_features": prompts_with_informative,
        "prompts_with_informative_pct": (
            100 * prompts_with_informative / valid_prompts if valid_prompts > 0 else 0
        ),
        "total_informative_features": total_informative_features,
        "avg_informative_per_prompt": avg_informative,
        "max_informative_per_prompt": max(informative_counts) if informative_counts else 0,
    }


def main(config_path: str):
    """Evaluate whether SAE features provide additional factual information.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    eval_items = asyncio.run(run_evaluation_async(cfg))

    summary = compute_summary(eval_items)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total prompts: {summary['total_prompts']}")
    print(f"Parse failures: {summary['parse_failures']}")
    print(
        f"Prompts with informative features: {summary['prompts_with_informative_features']}"
        f" / {summary['valid_prompts']}"
        f" ({summary['prompts_with_informative_pct']:.1f}%)"
    )
    print(f"Total informative features found: {summary['total_informative_features']}")
    print(f"Avg informative features per prompt: {summary['avg_informative_per_prompt']:.2f}")
    print(f"Max informative features in a prompt: {summary['max_informative_per_prompt']}")

    # Save results
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "features_path": cfg.features_path,
            "explanations_path": cfg.explanations_path,
            "model": cfg.model,
            "temperature": cfg.get("temperature", 0.0),
            "max_features_per_prompt": cfg.get("max_features_per_prompt", None),
            "timestamp": datetime.now().isoformat(),
        },
        "summary": summary,
        "evaluations": eval_items,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
