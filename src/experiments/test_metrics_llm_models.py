# ABOUTME: Test script to compare different models on hypothesis metrics LLM matching task.
# ABOUTME: Measures execution time and precision for each model.

import asyncio
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from hypothesis_metrics_llm import compute_metrics_async

MODELS_TO_TEST = [
    "openai/gpt-5.2-chat",
    "google/gemini-3-flash-preview",
    "openai/gpt-5-nano",
    "anthropic/claude-haiku-4.5",
]


async def test_model(
    model: str,
    hypotheses_file: str,
    gt_file: str,
    output_dir: str,
    max_concurrent: int = 100,
) -> dict:
    """Test a single model and return timing and precision metrics."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model}")
    print(f"{'='*60}")

    model_output_dir = Path(output_dir) / model.replace("/", "_")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    output_file = model_output_dir / "metrics.json"

    start_time = time.time()

    result = await compute_metrics_async(
        hypotheses_file=hypotheses_file,
        gt_file=gt_file,
        output_file=str(output_file),
        model_name=model,
        max_concurrent=max_concurrent,
        disable_reasoning=False,
    )

    elapsed_time = time.time() - start_time

    agg = result["aggregate"]
    return {
        "model": model,
        "elapsed_time_seconds": elapsed_time,
        "micro_precision": agg["micro_precision"],
        "micro_recall": agg["micro_recall"],
        "micro_f1": agg["micro_f1"],
        "total_hypotheses": agg["total_hypotheses"],
        "total_matched_hypotheses": agg["total_matched_hypotheses"],
        "output_file": str(output_file),
    }


async def run_comparison(config_path: str):
    """Run comparison across all models."""
    config = OmegaConf.load(config_path)

    hypotheses_file = config.hypotheses_file
    gt_file = config.gt_file
    output_dir = "output/metrics_model_comparison"
    max_concurrent = config.get("max_concurrent", 100)

    print(f"Hypotheses file: {hypotheses_file}")
    print(f"Ground truth file: {gt_file}")
    print(f"Output directory: {output_dir}")
    print(f"Models to test: {MODELS_TO_TEST}")

    results = []
    for model in MODELS_TO_TEST:
        try:
            result = await test_model(
                model, hypotheses_file, gt_file, output_dir, max_concurrent
            )
            results.append(result)
        except Exception as e:
            print(f"Error testing {model}: {e}")
            results.append({"model": model, "error": str(e)})

    # Print comparison table
    print("\n" + "="*90)
    print("COMPARISON RESULTS")
    print("="*90)
    print(f"{'Model':<35} {'Time (s)':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-"*90)

    for r in results:
        if "error" in r:
            print(f"{r['model']:<35} ERROR: {r['error']}")
        else:
            print(f"{r['model']:<35} {r['elapsed_time_seconds']:<12.2f} {r['micro_precision']:<12.3f} {r['micro_recall']:<12.3f} {r['micro_f1']:<12.3f}")

    print("-"*90)

    # Save results
    output_path = Path(output_dir) / "comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main(config_path: str):
    """Run the model comparison."""
    return asyncio.run(run_comparison(config_path))


if __name__ == "__main__":
    import fire
    fire.Fire(main)
