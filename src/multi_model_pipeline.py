# ABOUTME: Pipeline script to run local inference on multiple HF models.
# ABOUTME: Clears cache after each model, then runs refusal detection and fact checking.

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

load_dotenv()


DEFAULT_MODELS = [
    {
        "name": "Qwen/Qwen3-32B",
        "model_class": "AutoModelForCausalLM",
        "processor_class": "AutoTokenizer",
    },
]


def is_vl_model(model_class: str) -> bool:
    """Check if a model class is a VL (vision-language) model."""
    vl_classes = [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "LlavaForConditionalGeneration",
    ]
    return any(vl in model_class for vl in vl_classes)


def clear_hf_cache(model_name: str, cache_dir: str = "/root/.cache/huggingface/hub"):
    """Clear a specific model from HuggingFace cache."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Cache directory {cache_dir} does not exist")
        return

    # HF cache uses model name with -- instead of /
    model_cache_name = "models--" + model_name.replace("/", "--")

    for item in cache_path.iterdir():
        if item.name == model_cache_name:
            print(f"Removing cached model: {item}")
            shutil.rmtree(item)
            return

    print(f"Model {model_name} not found in cache")


def get_model_short_name(model_name: str) -> str:
    """Convert model name to a short directory-safe name."""
    # Take the last part and make it safe for directories
    short_name = model_name.split("/")[-1]
    return short_name.lower().replace(" ", "_")


def run_local_inference(
    model_config: dict,
    prompts_file: str,
    output_dir: str,
    n_samples: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    batch_size: int = 4,
    enable_thinking: bool = False,
    attn_implementation: str | None = None,
) -> Path:
    """Run local inference for a single model."""
    model_name = model_config["name"]
    model_class = model_config.get("model_class", "AutoModelForCausalLM")
    processor_class = model_config.get("processor_class", "AutoTokenizer")

    model_short_name = get_model_short_name(model_name)
    model_output_dir = Path(output_dir) / model_short_name

    # Check if this is a VL model based on model_class
    is_vl = is_vl_model(model_class)

    # Create config
    config = {
        "model": model_name,
        "prompts_file": prompts_file,
        "output_dir": str(model_output_dir),
        "n_samples": n_samples,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "batch_size": batch_size,
        "enable_thinking": enable_thinking,
        "do_sample": True,
        "model_class": model_class,
        "processor_class": processor_class,
    }
    if attn_implementation:
        config["attn_implementation"] = attn_implementation

    # Write temporary config
    config_dir = Path("configs/temp")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"local_inference_{model_short_name}.yaml"

    OmegaConf.save(OmegaConf.create(config), config_path)
    print(f"Created config: {config_path}")

    # Choose inference script based on model type
    if is_vl:
        inference_script = "src/local_inference_vl.py"
        print(f"Using VL inference script (model_class={model_class})")
    else:
        inference_script = "src/local_inference.py"

    # Run inference
    cmd = [
        sys.executable,
        inference_script,
        str(config_path),
    ]

    print(f"\nRunning inference for {model_name}...")
    subprocess.run(cmd, check=True)

    return model_output_dir


def run_refusal_detection(
    responses_dir: str,
    output_dir: str,
    model: str = "gpt-4.1-mini",
    poll_interval: int = 30,
) -> Path:
    """Run refusal detection on all responses."""
    cmd = [
        sys.executable,
        "src/detect_refusals_batch.py",
        f"--responses_dir={responses_dir}",
        f"--output_dir={output_dir}",
        f"--model={model}",
        f"--poll_interval={poll_interval}",
    ]

    print(f"\nRunning refusal detection...")
    subprocess.run(cmd, check=True)

    # Find the most recent output file
    output_path = Path(output_dir)
    files = sorted(output_path.glob("refusal_results_*.json"), reverse=True)
    if files:
        return files[0]
    return None


def filter_non_refusals(
    refusal_results_path: Path,
    responses_dir: str,
    filtered_output_dir: str,
    refusal_threshold: float = 50.0,
) -> Path:
    """Filter responses that are NOT refusals and save to a new directory."""
    with open(refusal_results_path) as f:
        refusal_data = json.load(f)

    # Group non-refusals by model
    non_refusals_by_model = {}
    for item in refusal_data:
        if item.get("refusal_score") is not None and item["refusal_score"] < refusal_threshold:
            model_name = item["model_name"]
            if model_name not in non_refusals_by_model:
                non_refusals_by_model[model_name] = []
            non_refusals_by_model[model_name].append({
                "prompt_id": item["prompt_id"],
                "sample_idx": item["sample_idx"],
            })

    # Load original responses and filter
    responses_dir = Path(responses_dir)
    filtered_dir = Path(filtered_output_dir)
    filtered_dir.mkdir(parents=True, exist_ok=True)

    total_filtered = 0
    for model_name, non_refusal_keys in non_refusals_by_model.items():
        model_dir = responses_dir / model_name
        if not model_dir.exists():
            continue

        # Create key set for fast lookup
        key_set = {(nr["prompt_id"], nr["sample_idx"]) for nr in non_refusal_keys}

        # Load and filter responses
        filtered_results = []
        for response_file in model_dir.glob("*.json"):
            with open(response_file) as f:
                data = json.load(f)

            for result in data.get("results", []):
                key = (result["prompt_id"], result["sample_idx"])
                if key in key_set:
                    filtered_results.append(result)

        if filtered_results:
            # Save filtered responses in same format
            filtered_model_dir = filtered_dir / model_name
            filtered_model_dir.mkdir(parents=True, exist_ok=True)

            output_file = filtered_model_dir / "filtered_responses.json"
            with open(output_file, "w") as f:
                json.dump({"results": filtered_results}, f, indent=2, ensure_ascii=False)

            total_filtered += len(filtered_results)
            print(f"  {model_name}: {len(filtered_results)} non-refusal responses")

    print(f"Total non-refusal responses: {total_filtered}")
    return filtered_dir


def run_fact_checking(
    responses_dir: str,
    output_dir: str,
    model: str = "gpt-4.1-mini",
    poll_interval: int = 30,
) -> Path:
    """Run fact checking on responses."""
    cmd = [
        sys.executable,
        "src/fact_check_model_responses.py",
        f"--responses_dir={responses_dir}",
        f"--output_dir={output_dir}",
        f"--model={model}",
        f"--poll_interval={poll_interval}",
    ]

    print(f"\nRunning fact checking...")
    subprocess.run(cmd, check=True)

    # Find the most recent output file
    output_path = Path(output_dir)
    files = sorted(output_path.glob("fact_check_results_*.json"), reverse=True)
    if files:
        return files[0]
    return None


def run_pipeline(
    config_path: str | None = None,
    models: list[dict] | None = None,
    prompts_file: str = "data/tiananmen_square_1989.json",
    output_base_dir: str = "output/multi_model_eval",
    n_samples: int = 5,
    temperature: float = 1.0,
    max_tokens: int = 2048,
    batch_size: int = 4,
    enable_thinking: bool = False,
    attn_implementation: str | None = None,
    clear_cache: bool = True,
    refusal_model: str = "gpt-4.1-mini",
    fact_check_model: str = "gpt-4.1-mini",
    refusal_threshold: float = 50.0,
    poll_interval: int = 30,
    skip_inference: bool = False,
):
    """Run the full pipeline: inference -> refusal detection -> fact checking.

    Args:
        config_path: Path to YAML config file (overrides other args)
        models: List of model configs, each with 'name', 'model_class', 'processor_class'
        prompts_file: Path to JSON file with prompts
        output_base_dir: Base directory for all outputs
        n_samples: Number of samples per prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        enable_thinking: Enable thinking mode for Qwen3 models
        attn_implementation: Attention implementation (e.g., "flash_attention_2")
        clear_cache: Whether to clear HF cache after each model
        refusal_model: Model to use for refusal detection
        fact_check_model: Model to use for fact checking
        refusal_threshold: Score threshold for refusal classification (>= is refusal)
        poll_interval: Seconds between batch API polls
        skip_inference: Skip inference step (use existing responses)
    """
    # Load config from file if provided
    if config_path:
        config = OmegaConf.load(config_path)
        models = OmegaConf.to_container(config.get("models", models))
        prompts_file = config.get("prompts_file", prompts_file)
        output_base_dir = config.get("output_base_dir", output_base_dir)
        n_samples = config.get("n_samples", n_samples)
        temperature = config.get("temperature", temperature)
        max_tokens = config.get("max_tokens", max_tokens)
        batch_size = config.get("batch_size", batch_size)
        enable_thinking = config.get("enable_thinking", enable_thinking)
        attn_implementation = config.get("attn_implementation", attn_implementation)
        clear_cache = config.get("clear_cache", clear_cache)
        refusal_model = config.get("refusal_model", refusal_model)
        fact_check_model = config.get("fact_check_model", fact_check_model)
        refusal_threshold = config.get("refusal_threshold", refusal_threshold)
        poll_interval = config.get("poll_interval", poll_interval)
        skip_inference = config.get("skip_inference", skip_inference)

    if models is None:
        models = DEFAULT_MODELS

    output_base = Path(output_base_dir)
    responses_dir = output_base / "responses"
    refusals_dir = output_base / "refusals"
    filtered_dir = output_base / "filtered_responses"
    fact_checks_dir = output_base / "fact_checks"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting pipeline at {timestamp}")
    print(f"Models: {[m['name'] for m in models]}")
    print(f"Output directory: {output_base}")

    # Step 1: Run inference for each model
    if not skip_inference:
        print("\n" + "=" * 60)
        print("STEP 1: Running local inference for each model")
        print("=" * 60)

        for i, model_config in enumerate(models, 1):
            model_name = model_config["name"]
            print(f"\n[{i}/{len(models)}] Processing {model_name}")
            print(f"  model_class: {model_config.get('model_class', 'AutoModelForCausalLM')}")
            print(f"  processor_class: {model_config.get('processor_class', 'AutoTokenizer')}")

            try:
                run_local_inference(
                    model_config=model_config,
                    prompts_file=prompts_file,
                    output_dir=str(responses_dir),
                    n_samples=n_samples,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    batch_size=batch_size,
                    enable_thinking=enable_thinking,
                    attn_implementation=attn_implementation,
                )
                print(f"Completed inference for {model_name}")
            except Exception as e:
                print(f"ERROR: Inference failed for {model_name}: {e}")
                continue

            # Clear cache after each model
            if clear_cache:
                print(f"Clearing cache for {model_name}...")
                clear_hf_cache(model_name)

    else:
        print("\n[SKIP] Inference step skipped, using existing responses")

    # Step 2: Run refusal detection
    print("\n" + "=" * 60)
    print("STEP 2: Running refusal detection")
    print("=" * 60)

    refusal_results_path = run_refusal_detection(
        responses_dir=str(responses_dir),
        output_dir=str(refusals_dir),
        model=refusal_model,
        poll_interval=poll_interval,
    )

    if not refusal_results_path:
        print("ERROR: Refusal detection failed")
        return

    print(f"Refusal results saved to: {refusal_results_path}")

    # Step 3: Filter non-refusals
    print("\n" + "=" * 60)
    print("STEP 3: Filtering non-refusal responses")
    print("=" * 60)

    filtered_responses_dir = filter_non_refusals(
        refusal_results_path=refusal_results_path,
        responses_dir=str(responses_dir),
        filtered_output_dir=str(filtered_dir),
        refusal_threshold=refusal_threshold,
    )

    # Step 4: Run fact checking on non-refusals only
    print("\n" + "=" * 60)
    print("STEP 4: Running fact checking on non-refusal responses")
    print("=" * 60)

    fact_check_results_path = run_fact_checking(
        responses_dir=str(filtered_responses_dir),
        output_dir=str(fact_checks_dir),
        model=fact_check_model,
        poll_interval=poll_interval,
    )

    if fact_check_results_path:
        print(f"Fact check results saved to: {fact_check_results_path}")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_base}")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
