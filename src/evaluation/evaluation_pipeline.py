# ABOUTME: Main pipeline for evaluating model responses - refusal, honesty, fact verification, hypotheses.
# ABOUTME: Supports both OpenRouter (real-time) and OpenAI Batch API (cost-effective).

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.response_evaluator import (
    compute_interrogation_stats,
    evaluate_facts_batch,
    evaluate_facts_openrouter,
    evaluate_honesty_batch,
    evaluate_honesty_openrouter,
    evaluate_refusals_batch,
    evaluate_refusals_openrouter,
    extract_hypotheses_batch,
    extract_hypotheses_openrouter,
)


def strip_openai_prefix(model: str) -> str:
    """Strip 'openai/' prefix from model name for use with OpenAI Batch API."""
    if model.startswith("openai/"):
        return model[len("openai/"):]
    return model


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: dict | list, path: str | Path) -> None:
    """Save data as JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {path}")


def load_json(path: str | Path) -> dict | list:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def load_responses(responses_path: str | Path) -> tuple[dict, list[dict]]:
    """
    Load responses file.

    Returns:
        Tuple of (config, results)
    """
    data = load_json(responses_path)
    return data.get("config", {}), data.get("results", [])


def load_facts_by_question(
    facts_path: str | Path, min_fact_count: int | None = None
) -> dict[str, list[dict]]:
    """
    Load facts file and index by question text.

    Args:
        facts_path: Path to facts JSON file
        min_fact_count: If set, filter out facts with count below this threshold

    Returns:
        Dict mapping question text to list of fact dicts
    """
    data = load_json(facts_path)
    facts_by_question = {}

    for category in data.get("categories", []):
        for q in category.get("questions", []):
            question_text = q.get("question", "")
            facts = q.get("facts", [])
            if min_fact_count is not None:
                facts = [f for f in facts if f.get("count", 0) >= min_fact_count]
            if question_text:
                facts_by_question[question_text] = facts

    return facts_by_question


def prepare_evaluations(
    results: list[dict],
    facts_by_question: dict[str, list[dict]],
) -> list[dict]:
    """
    Prepare evaluation items by joining responses with facts.
    Responses with above_threshold=True are marked for special handling.

    Returns:
        List of evaluation dicts with question, response, and facts
    """
    evaluations = []
    for result in results:
        question = result.get("prompt", "")
        response = result.get("response", "")
        prompt_id = result.get("prompt_id", "")
        sample_idx = result.get("sample_idx", 0)

        if not response:
            continue

        facts = facts_by_question.get(question, [])
        above_threshold = result.get("above_threshold", False)

        evaluations.append(
            {
                "prompt_id": prompt_id,
                "sample_idx": sample_idx,
                "question": question,
                "response": response,
                "facts": facts,
                "target_aspect": result.get("target_aspect", ""),
                "model": result.get("model", ""),
                "above_threshold": above_threshold,
            }
        )

    return evaluations


def convert_fact_results(evaluations: list[dict]) -> list[dict]:
    """Convert fact_results from 'result' field to 'is_mentioned'/'is_lie' booleans."""
    for e in evaluations:
        fact_results = e.get("fact_results", [])
        for fr in fact_results:
            if fr and "result" in fr:
                result = fr.pop("result")
                fr["is_mentioned"] = result == "yes"
                fr["is_lie"] = result == "lie"
        valid = [fr for fr in fact_results if fr]
        e["n_total_facts"] = len(valid)
        e["n_mentioned_facts"] = sum(1 for fr in valid if fr.get("is_mentioned"))
        e["n_lies"] = sum(1 for fr in valid if fr.get("is_lie"))
    return evaluations


def compute_summary_stats(evaluations: list[dict]) -> dict:
    """Compute summary statistics from evaluations."""
    total = len(evaluations)
    if total == 0:
        return {}

    # Refusal stats
    refusals = sum(1 for e in evaluations if e.get("is_refusal") is True)
    non_refusals = sum(1 for e in evaluations if e.get("is_refusal") is False)
    above_threshold_count = sum(1 for e in evaluations if e.get("above_threshold"))

    # Honesty stats (exclude above_threshold from average)
    honesty_scores = [
        e["honesty_score"]
        for e in evaluations
        if e.get("honesty_score") is not None and not e.get("above_threshold")
    ]
    avg_honesty = sum(honesty_scores) / len(honesty_scores) if honesty_scores else None

    # Fact stats
    total_facts = 0
    fact_yes = 0
    fact_no = 0
    fact_lie = 0

    for e in evaluations:
        for fr in e.get("fact_results", []):
            if fr:
                total_facts += 1
                if fr.get("is_lie"):
                    fact_lie += 1
                elif fr.get("is_mentioned"):
                    fact_yes += 1
                else:
                    fact_no += 1

    return {
        "total_responses": total,
        "above_threshold_count": above_threshold_count,
        "refusals": refusals,
        "non_refusals": non_refusals,
        "refusal_rate": refusals / total if total > 0 else None,
        "avg_honesty_score": avg_honesty,
        "total_facts_evaluated": total_facts,
        "facts_mentioned_yes": fact_yes,
        "facts_not_mentioned_no": fact_no,
        "facts_contradicted_lie": fact_lie,
        "fact_mention_rate": fact_yes / total_facts if total_facts > 0 else None,
        "fact_lie_rate": fact_lie / total_facts if total_facts > 0 else None,
    }


def run_pipeline(config_path: str, **overrides):
    """
    Run the response evaluation pipeline.

    Args:
        config_path: Path to YAML config file
        **overrides: CLI overrides for config values (e.g. --responses_file=path)
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    responses_path = cfg.responses_file
    facts_path = cfg.facts_file
    max_responses = cfg.get("max_responses", None)
    min_fact_count = cfg.get("min_fact_count", None)

    # Handle model configuration - support both single model and per-step models
    if "models" in cfg:
        default_model = cfg.models.get("default", "google/gemini-3-flash-preview")
        refusal_model = cfg.models.get("refusal", default_model)
        honesty_model = cfg.models.get("honesty", default_model)
        fact_model = cfg.models.get("fact_verification", default_model)
        hypothesis_extraction_model = cfg.models.get("hypothesis_extraction", default_model)
    else:
        # Legacy single model config
        refusal_model = honesty_model = fact_model = cfg.model
        hypothesis_extraction_model = cfg.model

    # Handle per-step API selection
    if "api" in cfg:
        default_api = cfg.api.get("default", "openrouter")
        refusal_api = cfg.api.get("refusal", default_api)
        honesty_api = cfg.api.get("honesty", default_api)
        fact_api = cfg.api.get("fact_verification", default_api)
        hypothesis_extraction_api = cfg.api.get("hypothesis_extraction", default_api)
    elif "use_batch_api" in cfg:
        # Legacy single boolean config
        api_val = "batch" if cfg.use_batch_api else "openrouter"
        refusal_api = honesty_api = fact_api = hypothesis_extraction_api = api_val
    else:
        refusal_api = honesty_api = fact_api = hypothesis_extraction_api = "openrouter"

    step_apis = {
        "refusal": refusal_api,
        "honesty": honesty_api,
        "fact_verification": fact_api,
        "hypothesis_extraction": hypothesis_extraction_api,
    }

    print(f"\n{'=' * 60}")
    print("Running Response Evaluation Pipeline")
    print(f"{'=' * 60}")
    print(f"  Responses: {responses_path}")
    print(f"  Facts: {facts_path}")
    print(f"  Models:")
    if refusal_model == honesty_model == fact_model:
        print(f"    All steps: {refusal_model}")
    else:
        print(f"    Refusal: {refusal_model}")
        print(f"    Honesty: {honesty_model}")
        print(f"    Fact verification: {fact_model}")
    api_values = set(step_apis.values())
    if len(api_values) == 1:
        api_label = "OpenAI Batch" if api_values.pop() == "batch" else "OpenRouter"
        print(f"  API: {api_label}")
    else:
        print(f"  API (per step):")
        for step, api in step_apis.items():
            label = "OpenAI Batch" if api == "batch" else "OpenRouter"
            print(f"    {step}: {label}")
    if max_responses:
        print(f"  Max responses: {max_responses}")
    if min_fact_count is not None:
        print(f"  Min fact count: {min_fact_count}")
    print()

    # Setup output directory
    output_dir = ensure_dir(cfg.output_dir)
    any_batch = any(v == "batch" for v in step_apis.values())
    batch_temp_dir = ensure_dir(output_dir / "batch_files") if any_batch else None

    # Load data
    print("Step 1: Loading data...")
    response_config, results = load_responses(responses_path)
    facts_by_question = load_facts_by_question(facts_path, min_fact_count=min_fact_count)

    # Limit results if specified
    if max_responses and max_responses < len(results):
        results = results[:max_responses]
        print(f"  Limited to first {max_responses} responses")
    print(f"  Loaded {len(results)} responses")
    print(f"  Loaded facts for {len(facts_by_question)} questions")

    # Prepare evaluations
    evaluations = prepare_evaluations(results, facts_by_question)
    above_threshold_count = sum(1 for e in evaluations if e.get("above_threshold"))
    print(f"  Prepared {len(evaluations)} evaluation items")
    if above_threshold_count > 0:
        print(
            f"  {above_threshold_count} responses marked above_threshold (facts will be set to 'no')"
        )

    # Extract config values
    temperature = cfg.get("temperature", 0.0)
    max_concurrent = cfg.get("max_concurrent", 20)
    max_retries = cfg.get("max_retries", 10)
    retry_delay = cfg.get("retry_delay", 1.0)
    poll_interval = cfg.get("batch", {}).get("poll_interval", 30)
    timeout = cfg.get("batch", {}).get("timeout", 86400)

    refusal_max_tokens = cfg.get("refusal", {}).get("max_tokens", 10)
    honesty_max_tokens = cfg.get("honesty", {}).get("max_tokens", 500)
    fact_max_tokens = cfg.get("fact_verification", {}).get("max_tokens", 10)
    hyp_extraction_max_tokens = cfg.get("hypothesis_extraction", {}).get("max_tokens", 4096)

    skip_refusal = cfg.get("skip_refusal", False)
    skip_honesty = cfg.get("skip_honesty", False)
    skip_fact_verification = cfg.get("skip_fact_verification", False)
    skip_hypothesis_extraction = cfg.get("skip_hypothesis_extraction", False)

    # Reasoning config for OpenRouter extended thinking
    reasoning_cfg = cfg.get("reasoning", None)
    reasoning = OmegaConf.to_container(reasoning_cfg) if reasoning_cfg else None

    # Step 2: Refusal detection
    if skip_refusal:
        print("\nStep 2: Refusal detection - SKIPPED")
        for e in evaluations:
            e["is_refusal"] = False
    else:
        print(f"\nStep 2: Detecting refusals ({refusal_model}, {refusal_api})...")
        if refusal_api == "batch":

            def refusal_progress(completed, total, status):
                print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

            evaluations = evaluate_refusals_batch(
                evaluations=evaluations,
                model=strip_openai_prefix(refusal_model),
                temperature=temperature,
                max_tokens=refusal_max_tokens,
                poll_interval=poll_interval,
                timeout=timeout,
                progress_callback=refusal_progress,
                temp_dir=batch_temp_dir,
            )
            print()
        else:
            evaluations = asyncio.run(
                evaluate_refusals_openrouter(
                    evaluations=evaluations,
                    model=refusal_model,
                    temperature=temperature,
                    max_tokens=refusal_max_tokens,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    reasoning=reasoning,
                )
            )

    refusals = sum(1 for e in evaluations if e.get("is_refusal") is True)
    non_refusals = [e for e in evaluations if e.get("is_refusal") is False]
    print(f"  Found {refusals} refusals out of {len(evaluations)} responses")
    print(f"  Continuing with {len(non_refusals)} non-refusal responses")

    # Step 3: Honesty scoring (only for non-refusals)
    if skip_honesty:
        print("\nStep 3: Honesty scoring - SKIPPED")
    elif not non_refusals:
        print("\nStep 3: Honesty scoring - Skipping, no non-refusal responses")
    else:
        print(
            f"\nStep 3: Scoring honesty for {len(non_refusals)} non-refusals ({honesty_model}, {honesty_api})..."
        )
        if honesty_api == "batch":

            def honesty_progress(completed, total, status):
                print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

            non_refusals = evaluate_honesty_batch(
                evaluations=non_refusals,
                model=strip_openai_prefix(honesty_model),
                temperature=temperature,
                max_tokens=honesty_max_tokens,
                poll_interval=poll_interval,
                timeout=timeout,
                progress_callback=honesty_progress,
                temp_dir=batch_temp_dir,
            )
            print()
        else:
            non_refusals = asyncio.run(
                evaluate_honesty_openrouter(
                    evaluations=non_refusals,
                    model=honesty_model,
                    temperature=temperature,
                    max_tokens=honesty_max_tokens,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    reasoning=reasoning,
                )
            )

    scores = [
        e["honesty_score"] for e in non_refusals if e.get("honesty_score") is not None
    ]
    if scores:
        print(f"  Average honesty score: {sum(scores) / len(scores):.1f}")

    # Step 4: Fact verification (only for non-refusals)
    total_facts = sum(len(e.get("facts", [])) for e in non_refusals)
    if skip_fact_verification:
        print("\nStep 4: Fact verification - SKIPPED")
    elif not non_refusals:
        print("\nStep 4: Fact verification - Skipping, no non-refusal responses")
    elif total_facts > 0:
        print(
            f"\nStep 4: Verifying {total_facts} facts for {len(non_refusals)} non-refusals ({fact_model}, {fact_api})..."
        )
        if fact_api == "batch":

            def fact_progress(completed, total, status):
                print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

            non_refusals = evaluate_facts_batch(
                evaluations=non_refusals,
                model=strip_openai_prefix(fact_model),
                temperature=temperature,
                max_tokens=fact_max_tokens,
                poll_interval=poll_interval,
                timeout=timeout,
                progress_callback=fact_progress,
                temp_dir=batch_temp_dir,
            )
            print()
        else:
            non_refusals = asyncio.run(
                evaluate_facts_openrouter(
                    evaluations=non_refusals,
                    model=fact_model,
                    temperature=temperature,
                    max_tokens=fact_max_tokens,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    reasoning=reasoning,
                )
            )

    # Step 5: Hypothesis extraction (only for non-refusals)
    if skip_hypothesis_extraction:
        print("\nStep 5: Hypothesis extraction - SKIPPED")
    elif not non_refusals:
        print("\nStep 5: Hypothesis extraction - Skipping, no non-refusal responses")
    else:
        print(
            f"\nStep 5: Extracting hypotheses from {len(non_refusals)} non-refusals ({hypothesis_extraction_model}, {hypothesis_extraction_api})..."
        )
        if hypothesis_extraction_api == "batch":

            def hyp_extract_progress(completed, total, status):
                print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

            non_refusals = extract_hypotheses_batch(
                evaluations=non_refusals,
                model=strip_openai_prefix(hypothesis_extraction_model),
                temperature=temperature,
                max_tokens=hyp_extraction_max_tokens,
                poll_interval=poll_interval,
                timeout=timeout,
                progress_callback=hyp_extract_progress,
                temp_dir=batch_temp_dir,
            )
            print()
        else:
            non_refusals = asyncio.run(
                extract_hypotheses_openrouter(
                    evaluations=non_refusals,
                    model=hypothesis_extraction_model,
                    temperature=temperature,
                    max_tokens=hyp_extraction_max_tokens,
                    max_concurrent=max_concurrent,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    reasoning=reasoning,
                )
            )

        total_hyps = sum(len(e.get("hypotheses", [])) for e in non_refusals)
        print(f"  Extracted {total_hyps} hypotheses total")

    # Merge non-refusals back into evaluations
    non_refusal_by_key = {(e["prompt_id"], e["sample_idx"]): e for e in non_refusals}
    for e in evaluations:
        key = (e["prompt_id"], e["sample_idx"])
        if key in non_refusal_by_key:
            e.update(non_refusal_by_key[key])

    # Convert results to is_mentioned/is_lie format and remove raw data
    convert_fact_results(evaluations)
    for e in evaluations:
        e.pop("facts", None)

    # Compute per-response summary stats
    summary = compute_summary_stats(evaluations)

    # Compute interrogation stats (cross-response aggregation per prompt)
    interrogation = compute_interrogation_stats(evaluations)
    n_prompts = len(interrogation)
    if n_prompts > 0:
        avg_coverage = sum(
            r["fact_coverage_rate"] for r in interrogation if r["fact_coverage_rate"] is not None
        ) / n_prompts
        avg_lie = sum(
            r["fact_lie_rate"] for r in interrogation if r["fact_lie_rate"] is not None
        ) / n_prompts
        summary["interrogation"] = {
            "n_prompts": n_prompts,
            "avg_fact_coverage_rate": avg_coverage,
            "avg_fact_lie_rate": avg_lie,
        }

    # Build output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "responses_file": str(responses_path),
            "facts_file": str(facts_path),
            "models": {
                "refusal": refusal_model,
                "honesty": honesty_model,
                "fact_verification": fact_model,
                "hypothesis_extraction": hypothesis_extraction_model,
            },
            "api": step_apis,
            "temperature": temperature,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "summary": summary,
        "interrogation": interrogation,
        "evaluations": evaluations,
    }

    # Save output
    output_path = output_dir / f"evaluation_{timestamp}.json"
    save_json(output, output_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Pipeline Complete!")
    print(f"{'=' * 60}")
    print(f"  Total responses: {summary.get('total_responses', 0)}")
    if summary.get("above_threshold_count", 0) > 0:
        print(f"  Above threshold: {summary.get('above_threshold_count', 0)}")
    print(
        f"  Refusals: {summary.get('refusals', 0)} ({summary.get('refusal_rate', 0) * 100:.1f}%)"
    )
    print(f"  Average honesty: {summary.get('avg_honesty_score', 0):.1f}")
    print(f"  Facts evaluated: {summary.get('total_facts_evaluated', 0)}")
    print(f"    - Mentioned (yes): {summary.get('facts_mentioned_yes', 0)}")
    print(f"    - Not mentioned (no): {summary.get('facts_not_mentioned_no', 0)}")
    print(f"    - Contradicted (lie): {summary.get('facts_contradicted_lie', 0)}")
    interr = summary.get("interrogation", {})
    if interr:
        print(f"\n  Interrogation (cross-response per prompt):")
        print(f"    Prompts: {interr['n_prompts']}")
        print(f"    Avg fact coverage (>=1 mention): {interr['avg_fact_coverage_rate'] * 100:.1f}%")
        print(f"    Avg fact lie rate (>=1 lie): {interr['avg_fact_lie_rate'] * 100:.1f}%")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
