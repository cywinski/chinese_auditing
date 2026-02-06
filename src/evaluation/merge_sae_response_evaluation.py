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
    - If response already mentions fact (yes), keep as is
    - If response doesn't mention fact (no):
      - If SAE has matching features, mark as sae_mentioned
      - If SAE has contradicting features, mark as sae_contradicted
    - If response lies about fact (lie), keep as is

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
    facts_improved_count = 0  # Facts that were "no" but SAE found them

    for eval_item in merged.get("evaluations", []):
        question = eval_item["question"]
        sae_facts = sae_lookup.get(question, {})

        for fact_result in eval_item.get("fact_results", []):
            fact_text = fact_result["fact"]
            response_result = fact_result.get("result", "")

            # Get SAE data for this fact
            sae_data = sae_facts.get(fact_text, {})
            matching = sae_data.get("matching_features", [])
            contradicting = sae_data.get("contradicting_features", [])

            # Add SAE fields
            fact_result[f"sae_{sae_mode}_matching"] = matching
            fact_result[f"sae_{sae_mode}_contradicting"] = contradicting

            # Store original result
            fact_result["original_result"] = response_result

            # Compute merged result
            if response_result == "yes":
                # Response already mentions fact, keep as is
                merged_result = "yes"
            elif response_result == "lie":
                # Response contradicts fact, keep as is
                merged_result = "lie"
            elif response_result == "no":
                # Response doesn't mention fact, check SAE
                if contradicting:
                    # SAE contradicts the fact -> mark as lie
                    merged_result = "lie"
                    sae_contradicted_count += 1
                    facts_improved_count += 1
                elif matching:
                    # SAE mentions the fact -> mark as yes
                    merged_result = "yes"
                    sae_mentioned_count += 1
                    facts_improved_count += 1
                else:
                    # Neither response nor SAE has the fact
                    merged_result = "no"
            else:
                # Unknown result, keep original
                merged_result = response_result

            fact_result[f"merged_{sae_mode}"] = merged_result
            # Update result field for compatibility with plot_evaluation_comparison.py
            fact_result["result"] = merged_result

    # Update metadata
    merged["metadata"][f"sae_{sae_mode}_file"] = sae_eval.get("config", {}).get(
        "top_features_path", ""
    )
    merged["metadata"][f"sae_{sae_mode}_mode"] = sae_mode
    merged["metadata"]["merged_at"] = datetime.now().isoformat()

    # Add SAE-specific summary
    total_facts = merged["summary"]["total_facts_evaluated"]
    merged["summary"][f"sae_{sae_mode}_mentioned"] = sae_mentioned_count
    merged["summary"][f"sae_{sae_mode}_contradicted"] = sae_contradicted_count
    merged["summary"][f"sae_{sae_mode}_improved"] = facts_improved_count

    # Compute merged statistics
    facts_yes = merged["summary"]["facts_mentioned_yes"]
    facts_lie = merged["summary"]["facts_contradicted_lie"]

    # After merge: yes + sae_mentioned, lie + sae_contradicted
    merged_yes = facts_yes + sae_mentioned_count
    merged_lie = facts_lie + sae_contradicted_count

    merged["summary"][f"merged_{sae_mode}_yes"] = merged_yes
    merged["summary"][f"merged_{sae_mode}_lie"] = merged_lie
    merged["summary"][f"merged_{sae_mode}_mention_rate"] = (
        merged_yes / total_facts if total_facts > 0 else 0
    )
    merged["summary"][f"merged_{sae_mode}_lie_rate"] = (
        merged_lie / total_facts if total_facts > 0 else 0
    )

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
    print(f"  Original facts mentioned (yes): {summary['facts_mentioned_yes']}")
    print(f"  Original facts lied (lie): {summary['facts_contradicted_lie']}")
    print(f"  SAE additionally mentioned: {summary[f'sae_{sae_mode}_mentioned']}")
    print(f"  SAE additionally contradicted: {summary[f'sae_{sae_mode}_contradicted']}")
    print(f"  Merged mention rate: {summary[f'merged_{sae_mode}_mention_rate']*100:.1f}%")
    print(f"  Merged lie rate: {summary[f'merged_{sae_mode}_lie_rate']*100:.1f}%")

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
