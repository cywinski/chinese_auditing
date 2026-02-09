# ABOUTME: Merges SAE fact evaluation results with response evaluation results.
# ABOUTME: For facts not mentioned in response, checks if SAE features contain/contradict them.

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


def load_json(path: str | Path) -> dict:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def build_sae_fact_lookup(sae_eval: dict) -> dict[str, dict[str, dict]]:
    """Build lookup from question -> fact -> SAE result.

    Returns:
        Dict mapping question to dict mapping fact text to SAE result dict
        with 'matching_features' and 'contradicting_features' keys.
    """
    lookup = {}
    for eval_item in sae_eval.get("evaluations", []):
        question = eval_item["question"]
        fact_lookup = {}
        for fact_result in eval_item.get("fact_results", []):
            fact_text = fact_result["fact"]
            fact_lookup[fact_text] = {
                "matching_features": fact_result.get("matching_features", []),
                "contradicting_features": fact_result.get("contradicting_features", []),
            }
        lookup[question] = fact_lookup
    return lookup


def merge_evaluations(
    response_eval: dict,
    sae_eval: dict,
    sae_mode: str,
) -> dict:
    """Merge SAE evaluation into response evaluation.

    For each fact in each response:
    - If response already mentions fact (is_mentioned=True), keep as is
    - If response lies about fact (is_lie=True), keep as is
    - Otherwise (not mentioned, not lie):
      - If SAE has contradicting features, set is_lie=True
      - If SAE has matching features, set is_mentioned=True

    Returns:
        New evaluation dict with merged results.
    """
    # Build SAE lookup
    sae_lookup = build_sae_fact_lookup(sae_eval)

    # Deep copy response evaluation
    merged = deepcopy(response_eval)

    # Track statistics
    sae_mentioned_count = 0
    sae_contradicted_count = 0

    for eval_item in merged.get("evaluations", []):
        question = eval_item["question"]
        sae_facts = sae_lookup.get(question, {})

        for fact_result in eval_item.get("fact_results", []):
            fact_text = fact_result["fact"]
            is_mentioned = fact_result.get("is_mentioned", False)
            is_lie = fact_result.get("is_lie", False)

            # Get SAE data for this fact
            sae_data = sae_facts.get(fact_text, {})
            matching = sae_data.get("matching_features", [])
            contradicting = sae_data.get("contradicting_features", [])

            # Compute merged result and overwrite
            if is_mentioned or is_lie:
                pass  # Keep as is
            else:
                # Response doesn't mention fact, check SAE
                if contradicting:
                    fact_result["is_lie"] = True
                    sae_contradicted_count += 1
                elif matching:
                    fact_result["is_mentioned"] = True
                    sae_mentioned_count += 1

    # Update metadata
    merged["metadata"]["sae_mode"] = sae_mode
    merged["metadata"]["merged_at"] = datetime.now().isoformat()

    # Update summary with new counts
    total_facts = merged["summary"]["total_facts_evaluated"]
    original_yes = merged["summary"]["facts_mentioned_yes"]
    original_lie = merged["summary"]["facts_contradicted_lie"]

    # After merge: yes + sae_mentioned, lie + sae_contradicted
    new_yes = original_yes + sae_mentioned_count
    new_lie = original_lie + sae_contradicted_count
    new_no = total_facts - new_yes - new_lie

    # Overwrite summary fields
    merged["summary"]["facts_mentioned_yes"] = new_yes
    merged["summary"]["facts_contradicted_lie"] = new_lie
    merged["summary"]["facts_not_mentioned_no"] = new_no
    merged["summary"]["fact_mention_rate"] = new_yes / total_facts if total_facts > 0 else 0
    merged["summary"]["fact_lie_rate"] = new_lie / total_facts if total_facts > 0 else 0

    # Keep track of SAE contribution
    merged["summary"]["sae_mentioned"] = sae_mentioned_count
    merged["summary"]["sae_contradicted"] = sae_contradicted_count

    return merged


def run_merge(
    response_eval_path: str | Path,
    sae_eval_path: str | Path,
    output_path: str | Path,
    sae_mode: str,
):
    """Run the merge and save results.

    Args:
        response_eval_path: Path to response evaluation JSON
        sae_eval_path: Path to SAE evaluation JSON
        output_path: Path to save merged results
        sae_mode: SAE mode (descriptions, examples, positive_logits)
    """
    print(f"Loading response evaluation from {response_eval_path}")
    response_eval = load_json(response_eval_path)
    print(f"  {len(response_eval['evaluations'])} responses")

    print(f"Loading SAE evaluation from {sae_eval_path}")
    sae_eval = load_json(sae_eval_path)
    print(f"  Mode: {sae_mode}")

    print("\nMerging evaluations...")
    merged = merge_evaluations(response_eval, sae_eval, sae_mode)

    # Print summary
    summary = merged["summary"]
    print(f"\nMerge summary for {sae_mode}:")
    print(f"  Facts mentioned (yes): {summary['facts_mentioned_yes']}")
    print(f"  Facts lied (lie): {summary['facts_contradicted_lie']}")
    print(f"  SAE additionally mentioned: {summary['sae_mentioned']}")
    print(f"  SAE additionally contradicted: {summary['sae_contradicted']}")
    print(f"  Fact mention rate: {summary['fact_mention_rate']*100:.1f}%")
    print(f"  Lie rate: {summary['fact_lie_rate']*100:.1f}%")

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    return merged


def main(
    response_eval_path: str = "output/dev_facts/evaluation/qwen3-32b/evaluation_20260205_213705.json",
    sae_eval_path: str = "output/dev_facts/evaluation/sae/sae_fact_evaluation_descriptions.json",
    output_path: str = "output/dev_facts/evaluation/merged/merged_descriptions.json",
    sae_mode: str = "descriptions",
):
    """Run merge from command line."""
    run_merge(response_eval_path, sae_eval_path, output_path, sae_mode)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
