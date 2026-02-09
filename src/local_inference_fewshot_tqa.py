# ABOUTME: Few-shot inference script using TruthfulQA responses as fewshot examples.
# ABOUTME: Reads from TruthfulQA output format (results key with validation filtering).

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.local_inference_fewshot import run_inference


def load_tqa_fewshot_samples(
    fewshot_path: str, filter_correct: bool = True
) -> list[dict]:
    """Load few-shot samples from TruthfulQA responses JSON file."""
    with open(fewshot_path, "r") as f:
        data = json.load(f)

    samples = data.get("results", [])
    samples = [s for s in samples if s.get("response") is not None]
    if filter_correct:
        samples = [s for s in samples if s.get("validation") == "correct"]
    return samples


def run(config_path: str):
    """Run few-shot local inference with TruthfulQA fewshot samples."""
    load_dotenv()

    config = OmegaConf.load(config_path)

    fewshot_path = config.fewshot_samples_path
    filter_correct = config.get("filter_correct", True)
    fewshot_samples = load_tqa_fewshot_samples(fewshot_path, filter_correct=filter_correct)
    print(f"Loaded {len(fewshot_samples)} few-shot samples from {fewshot_path}")

    run_inference(
        config=config,
        fewshot_samples=fewshot_samples,
        output_prefix="fewshot_tqa_responses",
        fewshot_metadata={"fewshot_samples_path": fewshot_path},
    )


if __name__ == "__main__":
    fire.Fire(run)
