"""
Generate an evaluation config for a response file (or all JSON files in a directory)
and run the evaluation pipeline, skipping attacks whose output already exists.

Usage:
    # Single file (globs for the timestamped version of the base path):
    python src/evaluation/run_evals.py \
        --responses ca/qwen3-32b/qwen_qwen3_32b_baseline.json \
        --eval-output-base output/evaluation/qwen3-32b \
        --configs-dir configs/qwen3-32b/inference_attacks

    # Directory (processes every .json file found directly inside):
    python src/evaluation/run_evals.py \
        --responses ca/qwen3-32b/system_prompts \
        --eval-output-base output/evaluation/qwen3-32b \
        --configs-dir configs/qwen3-32b/inference_attacks
"""

import argparse
import glob
import os
import subprocess

import yaml

CONFIG_TEMPLATE = {
    "facts_file": "data/dev_facts_explicit.json",
    "max_responses": None,
    "min_fact_count": 3,
    "models": {
        "default": "google/gemini-3-flash-preview",
        "refusal": "google/gemini-3-flash-preview",
        "honesty": "google/gemini-3-flash-preview",
        "fact_verification": "google/gemini-3-flash-preview",
        "hypothesis_extraction": "openai/gpt-5.2",
    },
    "api": {"default": "openrouter"},
    "temperature": 1.0,
    "reasoning": {"enabled": True, "effort": "medium"},
    "max_concurrent": 100,
    "max_retries": 10,
    "retry_delay": 1.0,
    "skip_refusal": False,
    "skip_honesty": False,
    "skip_fact_verification": False,
    "skip_hypothesis_extraction": True,
    "refusal": {"max_tokens": 10000},
    "honesty": {"max_tokens": 10000},
    "fact_verification": {"max_tokens": 10000},
    "hypothesis_extraction": {"max_tokens": 10000},
    "batch": {"poll_interval": 30, "timeout": 86400},
}


def find_response_files(path):
    """Return list of response files to evaluate.

    If path is a directory, returns all .json files directly inside it.
    If path is a file (possibly without timestamp), globs for timestamped versions
    and returns the most recent match.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.json")))
        return files

    # File path: glob for timestamped version (base_*.ext)
    base, ext = os.path.splitext(path)
    matches = sorted(glob.glob(f"{base}_*{ext}"))
    if matches:
        return [matches[-1]]  # most recent

    # Fallback: try the path as-is (no timestamp was appended)
    if os.path.isfile(path):
        return [path]

    return []


def eval_output_exists(output_dir):
    return os.path.isdir(output_dir) and bool(os.listdir(output_dir))


def write_config(config_path, responses_file, output_dir):
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    attack_name = os.path.splitext(os.path.basename(responses_file))[0]
    with open(config_path, "w") as f:
        f.write(f"# Evaluation config for {attack_name}\n\n")
        f.write("# Input files\n")
        f.write(f'responses_file: "{responses_file}"\n')
        f.write(f'facts_file: "data/dev_facts_explicit.json"\n\n')
        f.write("# Output directory\n")
        f.write(f'output_dir: "{output_dir}"\n\n')
        rest = {k: v for k, v in CONFIG_TEMPLATE.items() if k != "facts_file"}
        f.write(yaml.dump(rest, default_flow_style=False, sort_keys=False))


def run_eval_for_file(responses_file, eval_output_base, configs_dir):
    # Attack name and output dir are derived from the filename (includes timestamp)
    stem = os.path.splitext(os.path.basename(responses_file))[0]
    output_dir = os.path.join(eval_output_base, stem)

    if eval_output_exists(output_dir):
        print(f"Skipping {stem} (evaluation already exists)")
        return

    config_path = os.path.join(configs_dir, f"{stem}.yaml")
    print(f"\nRunning evaluation: {stem}")
    print(f"  Responses: {responses_file}")
    print(f"  Output:    {output_dir}")

    write_config(config_path, responses_file, output_dir)
    subprocess.run(
        ["python", "src/evaluation/evaluation_pipeline.py", config_path],
        check=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to a response .json file (base path, without timestamp) or a directory",
    )
    parser.add_argument(
        "--eval-output-base",
        required=True,
        help="Base directory for evaluation outputs",
    )
    parser.add_argument(
        "--configs-dir",
        required=True,
        help="Directory where generated config files are saved",
    )
    args = parser.parse_args()

    files = find_response_files(args.responses)
    if not files:
        print(f"WARNING: No response files found at {args.responses!r}, skipping.")
        return

    for f in files:
        run_eval_for_file(f, args.eval_output_base, args.configs_dir)


if __name__ == "__main__":
    main()
