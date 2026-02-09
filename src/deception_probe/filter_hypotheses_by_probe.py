# ABOUTME: Filters hypotheses file to keep only entries where probe score is above threshold.
# ABOUTME: Matches entries by prompt_id and sample_idx between probe scores and hypotheses files.

import json
from datetime import datetime
from pathlib import Path

import fire
from omegaconf import OmegaConf


def filter_hypotheses(config_path: str):
    """
    Filter hypotheses to keep only those for responses with above_threshold=True.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    # Load probe scores
    with open(cfg.probe_scores_file, "r") as f:
        probe_data = json.load(f)

    # Load hypotheses
    with open(cfg.hypotheses_file, "r") as f:
        hypotheses_data = json.load(f)

    # Build lookup for above_threshold by (prompt_id, sample_idx)
    above_threshold_lookup = {}
    for result in probe_data["results"]:
        key = (str(result["prompt_id"]), result["sample_idx"])
        above_threshold_lookup[key] = result.get("above_threshold", False)

    # Clear hypotheses for responses above threshold
    filtered_results = []
    cleared_count = 0
    for result in hypotheses_data["results"]:
        key = (str(result["prompt_id"]), result["sample_idx"])
        result_copy = result.copy()
        if above_threshold_lookup.get(key, False):
            result_copy["hypotheses"] = []
            cleared_count += 1
        filtered_results.append(result_copy)

    print(f"Total responses: {len(hypotheses_data['results'])}")
    print(f"Responses with hypotheses cleared (above_threshold): {cleared_count}")

    # Create output data
    output_data = {
        "config": {
            **hypotheses_data["config"],
            "filtered_by": "above_threshold from probe scores",
            "probe_scores_file": cfg.probe_scores_file,
            "step": "filtered_above_threshold",
        },
        "results": filtered_results,
    }

    # Determine output path
    output_dir = Path(cfg.get("output_dir", None) or Path(cfg.hypotheses_file).parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"hypotheses_filtered_above_threshold_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    fire.Fire(filter_hypotheses)
