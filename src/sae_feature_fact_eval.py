# ABOUTME: Evaluates SAE features against facts using autorater to assess feature-fact relevance.
# ABOUTME: For each prompt/fact pair, determines if any extracted features relate to the fact.

import asyncio
import json
import os
import random

import aiohttp
import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio


def load_all_positive_logits(path: str) -> dict[int, list[tuple[str, float]]]:
    """Load precomputed positive logits for all SAE features."""
    data = torch.load(path, weights_only=False)
    # Convert string keys back to int
    return {int(k): v for k, v in data["positive_logits"].items()}


def generate_random_features_for_token(
    all_positive_logits: dict[int, list[tuple[str, float]]],
    n_features: int,
    seed: int | None = None,
) -> list[dict]:
    """Generate random features for a token using precomputed positive logits."""
    if seed is not None:
        random.seed(seed)

    all_feature_indices = list(all_positive_logits.keys())
    selected_indices = random.sample(all_feature_indices, min(n_features, len(all_feature_indices)))

    features = []
    for feat_idx in selected_indices:
        pos_logits = all_positive_logits[feat_idx]
        features.append({
            "feature_idx": feat_idx,
            "positive_logits": pos_logits,
        })
    return features


def load_eval_facts(eval_facts_path: str) -> dict:
    """Load eval facts and create a mapping from prompt_id to facts."""
    with open(eval_facts_path) as f:
        eval_data = json.load(f)

    # Build mapping: prompt_id -> list of facts
    prompt_to_facts = {}
    for topic_key, topic_value in eval_data.items():
        if topic_key == "metadata":
            continue
        for subtopic_key, questions in topic_value.items():
            for q in questions:
                prompt_id = f"{topic_key}/{subtopic_key}/{q['level']}"
                prompt_to_facts[prompt_id] = q.get("facts", [])

    return prompt_to_facts


def format_token_features_for_autorater(
    token_info: dict, max_features: int = 20
) -> tuple[str, list[int]] | tuple[None, None]:
    """Format a single token's features into a readable string for the autorater.

    Args:
        token_info: Token data with features
        max_features: Max features to show

    Returns:
        Tuple of (formatted string, list of original feature indices) or (None, None) if no features
    """
    if token_info.get("is_outlier"):
        return None, None

    features = token_info.get("top_features", [])[:max_features]
    if not features:
        return None, None

    feature_strs = []
    feature_indices = []
    for feat in features:
        feat_idx = feat.get("feature_idx")
        pos_logits = feat.get("positive_logits", [])
        if pos_logits and feat_idx is not None:
            # Get top 5 positive logit tokens per feature
            top_tokens = [t for t, _ in pos_logits[:5]]
            feature_strs.append(f"{feat_idx}: [{', '.join(top_tokens)}]")
            feature_indices.append(feat_idx)

    if not feature_strs:
        return None, None

    return "\n".join(feature_strs), feature_indices


async def evaluate_token_fact_relevance(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt_text: str,
    token_str: str,
    token_position: int,
    features_description: str,
    fact: str,
    semaphore: asyncio.Semaphore,
    model: str = "google/gemini-3-flash-preview",
    max_retries: int = 3,
) -> dict:
    """Evaluate if a token's features are related to a specific fact.

    Returns:
        dict with keys: token, position, fact, relevant (bool), relevant_features (list[int])
    """
    system_prompt = """You analyze neural network features to find ones that correspond to details in a fact.

A feature is relevant if its positive logit tokens relate to content mentioned in the fact:
- Names of people, places, organizations
- Numbers, dates, quantities
- Actions, events, or outcomes described
- Key concepts or terms from the fact

A feature is NOT relevant if:
- It only captures very broad context (e.g., ["China", "country"] for any China-related fact)
- It has no clear connection to what the fact describes

OUTPUT FORMAT:
- If NO features are relevant: respond with exactly "no"
- If features ARE relevant: respond with ONLY the comma-separated feature indices

Examples:
- Fact: "Declassified British cables suggested around 10,000 deaths"
  Feature 5432: ["British", "UK", "cables", "diplomatic"] → "5432" (British cables)
  Feature 1234: ["deaths", "killed", "casualties", "victims"] → "1234" (deaths)
  Feature 9999: ["government", "official", "policy"] → "no" (too generic)

- Fact: "Zhao Ziyang said 'We came too late'"
  Feature 9876: ["Zhao", "Ziyang", "leader"] → "9876" (specific person)
  Feature 5555: ["speech", "statement", "said", "words"] → "5555" (speaking/saying)
  Feature 1111: ["political", "party", "communist"] → "no" (too generic)

- Fact: "The 27th Group Army was involved in the crackdown"
  Feature 3333: ["27th", "Army", "regiment"] → "3333" (specific unit)
  Feature 4444: ["military", "troops", "soldiers", "armed"] → "4444" (military involvement)
  Feature 2222: ["China", "Chinese", "Beijing"] → "no" (too generic)"""

    user_prompt = f"""PROMPT: {prompt_text}

TOKEN: '{token_str}'

FEATURES:
{features_description}

FACT: {fact}"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 50,
    }

    base_result = {
        "token": token_str,
        "position": token_position,
        "fact": fact,
    }

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    answer = data["choices"][0]["message"]["content"].strip().lower()

                    # Parse answer
                    if answer == "no":
                        return {
                            **base_result,
                            "relevant": False,
                            "relevant_features": [],
                        }
                    else:
                        # Parse comma-separated indices
                        try:
                            indices = [int(x.strip()) for x in answer.split(",")]
                            return {
                                **base_result,
                                "relevant": True,
                                "relevant_features": indices,
                            }
                        except ValueError:
                            # If parsing fails, check if it contains "no"
                            if "no" in answer:
                                return {
                                    **base_result,
                                    "relevant": False,
                                    "relevant_features": [],
                                }
                            return {
                                **base_result,
                                "relevant": None,
                                "relevant_features": [],
                                "parse_error": answer,
                            }
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (2**attempt))
                else:
                    return {
                        **base_result,
                        "relevant": None,
                        "relevant_features": [],
                        "error": str(e),
                    }

    return {**base_result, "relevant": None, "relevant_features": [], "error": "Unknown error"}


async def evaluate_prompt_facts(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt_id: str,
    prompt_text: str,
    tokens_data: list[dict],
    facts: list[str],
    semaphore: asyncio.Semaphore,
    model: str,
    max_features_per_token: int = 20,
) -> dict:
    """Evaluate all facts for a single prompt, token by token."""
    # Build list of tasks: one per (token, fact) pair
    tasks = []

    for token_info in tokens_data:
        features_desc, _ = format_token_features_for_autorater(
            token_info, max_features_per_token
        )
        if features_desc is None:
            continue

        token_str = token_info["token"]
        token_pos = token_info["position"]

        for fact in facts:
            tasks.append(
                evaluate_token_fact_relevance(
                    session=session,
                    api_key=api_key,
                    prompt_text=prompt_text,
                    token_str=token_str,
                    token_position=token_pos,
                    features_description=features_desc,
                    fact=fact,
                    semaphore=semaphore,
                    model=model,
                )
            )

    if not tasks:
        # No features extracted for any token
        return {
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "n_facts": len(facts),
            "n_tokens_with_features": 0,
            "token_results": [],
            "fact_summary": {
                fact: {"any_relevant": False, "relevant_tokens": []}
                for fact in facts
            },
        }

    results = await asyncio.gather(*tasks)

    # Organize results by token and by fact
    token_results = {}  # token_pos -> {token, position, fact_results: [...]}
    fact_summary = {fact: {"any_relevant": False, "relevant_tokens": []} for fact in facts}

    for result in results:
        token_pos = result["position"]
        token_str = result["token"]
        fact = result["fact"]
        relevant_features = result.get("relevant_features", [])

        # Add to token results
        if token_pos not in token_results:
            token_results[token_pos] = {
                "token": token_str,
                "position": token_pos,
                "fact_results": [],
            }
        token_results[token_pos]["fact_results"].append({
            "fact": fact,
            "relevant": result["relevant"],
            "relevant_features": relevant_features,
        })

        # Update fact summary
        if result["relevant"]:
            fact_summary[fact]["any_relevant"] = True
            fact_summary[fact]["relevant_tokens"].append({
                "token": token_str,
                "position": token_pos,
                "relevant_features": relevant_features,
            })

    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "n_facts": len(facts),
        "n_tokens_with_features": len(token_results),
        "token_results": list(token_results.values()),
        "fact_summary": fact_summary,
    }


async def run_evaluation(
    prompt_features_path: str,
    eval_facts_path: str,
    output_path: str,
    model: str = "google/gemini-3-flash-preview",
    max_concurrent: int = 20,
    max_features_per_token: int = 5,
    max_prompts: int | None = None,
    control_mode: bool = False,
    all_positive_logits_path: str | None = None,
    random_seed: int = 42,
) -> dict:
    """Run the full evaluation."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    # Load all positive logits if in control mode
    all_positive_logits = None
    if control_mode:
        if not all_positive_logits_path:
            raise ValueError("all_positive_logits_path required for control mode")
        print(f"CONTROL MODE: Loading all positive logits from: {all_positive_logits_path}")
        all_positive_logits = load_all_positive_logits(all_positive_logits_path)
        print(f"Loaded positive logits for {len(all_positive_logits)} features")
        random.seed(random_seed)

    # Load data
    print(f"Loading prompt features from: {prompt_features_path}")
    with open(prompt_features_path) as f:
        prompt_features = json.load(f)

    print(f"Loading eval facts from: {eval_facts_path}")
    prompt_to_facts = load_eval_facts(eval_facts_path)

    prompts_data = prompt_features.get("prompts", {})
    print(f"Found {len(prompts_data)} prompts with features")
    print(f"Found {len(prompt_to_facts)} prompts with facts")

    # Match prompts to facts
    evaluation_tasks = []
    for prompt_id, prompt_info in prompts_data.items():
        facts = prompt_to_facts.get(prompt_id, [])
        if not facts:
            print(f"Warning: No facts found for prompt_id: {prompt_id}")
            continue

        tokens_data = prompt_info.get("tokens", [])

        # In control mode, replace features with random ones
        if control_mode and all_positive_logits:
            tokens_data_modified = []
            for token_info in tokens_data:
                if token_info.get("is_outlier"):
                    tokens_data_modified.append(token_info)
                    continue

                # Count original features
                orig_features = token_info.get("top_features", [])
                n_features = len(orig_features)

                if n_features > 0:
                    # Generate random features with unique seed per token
                    token_seed = random_seed + token_info.get("position", 0) + hash(prompt_id) % 10000
                    random_features = generate_random_features_for_token(
                        all_positive_logits, n_features, seed=token_seed
                    )
                    token_info_copy = dict(token_info)
                    token_info_copy["top_features"] = random_features
                    tokens_data_modified.append(token_info_copy)
                else:
                    tokens_data_modified.append(token_info)
            tokens_data = tokens_data_modified

        evaluation_tasks.append(
            {
                "prompt_id": prompt_id,
                "prompt_text": prompt_info["prompt"],
                "tokens_data": tokens_data,
                "facts": facts,
            }
        )

    # Limit number of prompts if specified
    if max_prompts is not None and max_prompts > 0:
        evaluation_tasks = evaluation_tasks[:max_prompts]
        print(f"Limiting to {max_prompts} prompts for testing")

    print(
        f"Evaluating {len(evaluation_tasks)} prompts with {sum(len(t['facts']) for t in evaluation_tasks)} total facts"
    )

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            evaluate_prompt_facts(
                session=session,
                api_key=api_key,
                prompt_id=task["prompt_id"],
                prompt_text=task["prompt_text"],
                tokens_data=task["tokens_data"],
                facts=task["facts"],
                semaphore=semaphore,
                model=model,
                max_features_per_token=max_features_per_token,
            )
            for task in evaluation_tasks
        ]

        all_results = await tqdm_asyncio.gather(*tasks, desc="Evaluating prompts")

    # Compute summary statistics
    total_facts = 0
    facts_with_relevant_features = 0
    facts_without_relevant_features = 0
    total_token_fact_pairs = 0
    relevant_token_fact_pairs = 0
    total_relevant_feature_count = 0

    for result in all_results:
        fact_summary = result.get("fact_summary", {})
        for fact, summary in fact_summary.items():
            total_facts += 1
            if summary["any_relevant"]:
                facts_with_relevant_features += 1
            else:
                facts_without_relevant_features += 1

        # Count token-fact pairs
        for token_result in result.get("token_results", []):
            for fact_result in token_result.get("fact_results", []):
                total_token_fact_pairs += 1
                if fact_result["relevant"]:
                    relevant_token_fact_pairs += 1
                    total_relevant_feature_count += len(fact_result.get("relevant_features", []))

    output = {
        "config": {
            "prompt_features_path": prompt_features_path,
            "eval_facts_path": eval_facts_path,
            "model": model,
            "max_features_per_token": max_features_per_token,
            "control_mode": control_mode,
            "random_seed": random_seed if control_mode else None,
        },
        "summary": {
            "total_prompts": len(all_results),
            "total_facts": total_facts,
            "facts_with_relevant_features": facts_with_relevant_features,
            "facts_without_relevant_features": facts_without_relevant_features,
            "fact_relevance_rate": facts_with_relevant_features / total_facts
            if total_facts > 0
            else 0,
            "total_token_fact_pairs": total_token_fact_pairs,
            "relevant_token_fact_pairs": relevant_token_fact_pairs,
            "token_fact_relevance_rate": relevant_token_fact_pairs / total_token_fact_pairs
            if total_token_fact_pairs > 0
            else 0,
            "total_relevant_features_found": total_relevant_feature_count,
        },
        "results": all_results,
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total prompts evaluated: {len(all_results)}")
    print(f"Total facts evaluated: {total_facts}")
    print(
        f"Facts with at least one relevant token: {facts_with_relevant_features} ({100 * facts_with_relevant_features / total_facts:.1f}%)"
    )
    print(
        f"Facts without relevant features: {facts_without_relevant_features} ({100 * facts_without_relevant_features / total_facts:.1f}%)"
    )
    print(f"\nTotal token-fact pairs evaluated: {total_token_fact_pairs}")
    print(
        f"Relevant token-fact pairs: {relevant_token_fact_pairs} ({100 * relevant_token_fact_pairs / total_token_fact_pairs:.1f}%)"
    )
    print(f"Total relevant features found: {total_relevant_feature_count}")

    return output


def main(config_path: str):
    """
    Evaluate SAE features against facts using autorater.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    result = asyncio.run(
        run_evaluation(
            prompt_features_path=cfg.prompt_features_path,
            eval_facts_path=cfg.eval_facts_path,
            output_path=cfg.output_path,
            model=cfg.get("model", "google/gemini-3-flash-preview"),
            max_concurrent=cfg.get("max_concurrent", 20),
            max_features_per_token=cfg.get("max_features_per_token", 5),
            max_prompts=cfg.get("max_prompts", None),
            control_mode=cfg.get("control_mode", False),
            all_positive_logits_path=cfg.get("all_positive_logits_path", None),
            random_seed=cfg.get("random_seed", 42),
        )
    )

    return result


if __name__ == "__main__":
    fire.Fire(main)
