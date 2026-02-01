# ABOUTME: Batch response sampling using OpenAI Batch API.
# ABOUTME: Samples multiple responses for chat queries from a JSON input file.

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.fact_generation_batch.openai_batch_client import (
    BatchRequest,
    BatchResult,
    run_batch,
)


def load_prompts_from_json(prompts_path: str | Path) -> list[dict]:
    """
    Load prompts from a JSON file.

    Supports two formats:
    1. List of dicts with 'question' key
    2. Dict with 'categories' containing questions (eval_facts format)

    Returns:
        List of dicts with 'prompt_id', 'prompt', and 'target_aspect' keys
    """
    with open(prompts_path) as f:
        data = json.load(f)

    prompts = []

    # Format 1: Simple list of questions
    if isinstance(data, list):
        for i, item in enumerate(data):
            prompts.append(
                {
                    "prompt_id": str(i),
                    "prompt": item.get("question", item.get("prompt", "")),
                    "target_aspect": item.get("category", ""),
                }
            )
        return prompts

    # Format 2: eval_facts format with categories
    if "categories" in data:
        prompt_idx = 0
        topic = data.get("metadata", {}).get("topic", "")
        for category in data["categories"]:
            category_name = category.get("name", "")
            for q in category.get("questions", []):
                level = q.get("level", "")
                target_aspect = f"{topic}/{category_name}/{level}" if topic else f"{category_name}/{level}"
                prompts.append(
                    {
                        "prompt_id": str(prompt_idx),
                        "prompt": q["question"],
                        "target_aspect": target_aspect,
                    }
                )
                prompt_idx += 1
        return prompts

    # Format 3: Dict with topic keys (like eval_facts.json)
    for topic_key, topic_data in data.items():
        if topic_key == "metadata":
            continue
        if isinstance(topic_data, dict):
            for category_key, questions in topic_data.items():
                if isinstance(questions, list):
                    for q in questions:
                        if isinstance(q, dict) and "question" in q:
                            level = q.get("level", "")
                            target_aspect = f"{topic_key}/{category_key}/{level}"
                            prompts.append(
                                {
                                    "prompt_id": str(len(prompts)),
                                    "prompt": q["question"],
                                    "target_aspect": target_aspect,
                                }
                            )

    return prompts


def create_sample_requests(
    prompts: list[dict],
    model: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    system_prompt: str | None = None,
) -> list[BatchRequest]:
    """Create batch requests for all samples."""
    requests = []

    for prompt_data in prompts:
        prompt_id = prompt_data["prompt_id"]
        prompt = prompt_data["prompt"]

        for sample_idx in range(n_samples):
            custom_id = f"p{prompt_id}_s{sample_idx}"

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            requests.append(
                BatchRequest(
                    custom_id=custom_id,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

    return requests


def parse_sample_results(
    results: list[BatchResult],
    prompts: list[dict],
    n_samples: int,
    model: str,
) -> list[dict]:
    """Parse batch results into response format."""
    results_by_id = {r.custom_id: r for r in results}
    output = []

    for prompt_data in prompts:
        prompt_id = prompt_data["prompt_id"]
        prompt = prompt_data["prompt"]
        target_aspect = prompt_data["target_aspect"]

        for sample_idx in range(n_samples):
            custom_id = f"p{prompt_id}_s{sample_idx}"
            result = results_by_id.get(custom_id)

            response_text = ""
            error = None
            if result:
                if result.content:
                    response_text = result.content
                elif result.error:
                    error = result.error
            else:
                error = "Missing result"

            result_entry = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "target_aspect": target_aspect,
                "sample_idx": sample_idx,
                "model": model,
                "response": response_text,
            }

            if error:
                result_entry["error"] = error

            output.append(result_entry)

    return output


def sample_responses_batch(
    config_path: str,
    poll_interval: int = 30,
    timeout: int = 86400,
):
    """
    Sample responses using OpenAI Batch API.

    Args:
        config_path: Path to YAML config file
        poll_interval: Seconds between batch status polls
        timeout: Maximum seconds to wait for batch completion
    """
    load_dotenv()

    cfg = OmegaConf.load(config_path)

    model = cfg.model
    prompts_path = cfg.prompts_csv  # Named csv but can be json
    output_dir = Path(cfg.output_dir)
    n_samples = cfg.n_samples
    temperature = cfg.temperature
    max_tokens = cfg.max_tokens
    system_prompt = cfg.get("system_prompt", None)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading prompts from: {prompts_path}")
    prompts = load_prompts_from_json(prompts_path)
    print(f"Loaded {len(prompts)} prompts")

    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Samples per prompt: {n_samples}")
    print(f"  Temperature: {temperature}")
    print(f"  Max tokens: {max_tokens}")
    print(f"  Total requests: {len(prompts) * n_samples}")

    # Create batch requests
    requests = create_sample_requests(
        prompts=prompts,
        model=model,
        n_samples=n_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )

    print(f"\nCreated {len(requests)} batch requests")

    def progress_callback(completed, total, status):
        print(f"  Batch progress: {completed}/{total} ({status})", end="\r")

    # Run batch
    print("\nSubmitting batch job...")
    results = run_batch(
        requests=requests,
        description=f"Response sampling: {len(prompts)} prompts x {n_samples} samples",
        poll_interval=poll_interval,
        timeout=timeout,
        progress_callback=progress_callback,
        temp_dir=output_dir / "batch_temp",
    )
    print()  # newline after progress

    # Parse results
    parsed_results = parse_sample_results(results, prompts, n_samples, model)

    # Build output
    output = {
        "config": {
            "model": model,
            "prompts_csv": str(prompts_path),
            "output_dir": str(output_dir),
            "n_samples": n_samples,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system_prompt": system_prompt,
            "processing_method": "openai_batch_api",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": parsed_results,
    }

    # Save output
    output_path = output_dir / "responses.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Print summary
    successful = sum(1 for r in parsed_results if r["response"] and "error" not in r)
    failed = len(parsed_results) - successful

    print(f"\nSampling complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_path}")

    return output


if __name__ == "__main__":
    fire.Fire(sample_responses_batch)
