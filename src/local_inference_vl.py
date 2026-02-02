# ABOUTME: Local inference script for VL (Vision-Language) HuggingFace models.
# ABOUTME: Supports custom model/processor classes and VL-style message formatting.

import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import fire
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_class_from_string(class_path: str):
    """Import and return a class from a fully qualified string path."""
    if "." in class_path:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    else:
        # Assume it's from transformers
        import transformers
        return getattr(transformers, class_path)


def load_model(
    model_name: str,
    model_class: str = "AutoModelForCausalLM",
    processor_class: str = "AutoTokenizer",
    attn_implementation: str | None = None,
):
    """Load model and processor from HuggingFace with custom classes."""
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    # Load model class
    ModelClass = get_class_from_string(model_class)
    model = ModelClass.from_pretrained(model_name, **kwargs)

    # Load processor class
    ProcessorClass = get_class_from_string(processor_class)
    processor = ProcessorClass.from_pretrained(model_name, trust_remote_code=True)

    # Set pad token if using tokenizer-like processor
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        if hasattr(processor, "eos_token"):
            processor.pad_token = processor.eos_token

    return model, processor


def load_prompts_from_json(json_path: str) -> list[dict]:
    """Load prompts from a topic facts JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = []
    idx = 1
    topic = data.get("metadata", {}).get("topic", "unknown")

    for category in data.get("categories", []):
        category_name = category.get("name", "unknown")
        for q in category.get("questions", []):
            prompts.append({
                "id": str(idx),
                "prompt": q["question"],
                "target_aspect": f"{topic}/{category_name}/{q.get('level', 'unknown')}",
            })
            idx += 1

    return prompts


def build_vl_messages(
    user_content: str,
    system_prompt: str | None = None,
) -> list[dict]:
    """Build VL-style messages with typed content."""
    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })

    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_content}],
    })

    return messages


def format_prompt_vl(
    processor,
    user_content: str,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Format a prompt using VL-style messages."""
    messages = build_vl_messages(user_content, system_prompt)

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    # Try with enable_thinking for Qwen3
    if enable_thinking is not None:
        try:
            return processor.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            pass

    return processor.apply_chat_template(messages, **kwargs)


def get_model_device(model):
    """Get device of a model."""
    return next(model.parameters()).device


def generate_response_single(
    model,
    processor,
    prompt: str,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> tuple[str, int]:
    """Generate response for a single prompt. Returns (text, num_tokens)."""
    device = get_model_device(model)

    inputs = processor(prompt, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    # Get pad token id
    if hasattr(processor, "eos_token_id"):
        gen_kwargs["pad_token_id"] = processor.eos_token_id
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
        gen_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id

    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    response_tokens = outputs[0][input_len:]

    # Decode using appropriate method
    if hasattr(processor, "decode"):
        response_text = processor.decode(response_tokens, skip_special_tokens=True)
    elif hasattr(processor, "tokenizer"):
        response_text = processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
    else:
        response_text = str(response_tokens)

    return response_text, len(response_tokens)


def generate_responses_batch(
    model,
    processor,
    prompts: list[str],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> list[tuple[str, int]]:
    """Generate responses for a batch of prompts. Returns list of (text, num_tokens)."""
    device = get_model_device(model)

    # Set padding side for batch processing
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"
    elif hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    inputs = processor(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }

    # Get pad token id
    if hasattr(processor, "eos_token_id"):
        gen_kwargs["pad_token_id"] = processor.eos_token_id
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "eos_token_id"):
        gen_kwargs["pad_token_id"] = processor.tokenizer.eos_token_id

    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    results = []
    for output in outputs:
        response_tokens = output[input_len:]

        if hasattr(processor, "decode"):
            response_text = processor.decode(response_tokens, skip_special_tokens=True)
        elif hasattr(processor, "tokenizer"):
            response_text = processor.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response_text = str(response_tokens)

        results.append((response_text, len(response_tokens)))

    return results


def run(config_path: str):
    """Run local inference for VL models."""
    load_dotenv()

    config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(config)

    model_class = config.get("model_class", "AutoModelForCausalLM")
    processor_class = config.get("processor_class", "AutoTokenizer")
    attn_impl = config.get("attn_implementation", None)

    print(f"Loading model {config.model}...")
    print(f"  Model class: {model_class}")
    print(f"  Processor class: {processor_class}")
    if attn_impl:
        print(f"  Attention: {attn_impl}")

    model, processor = load_model(
        config.model,
        model_class=model_class,
        processor_class=processor_class,
        attn_implementation=attn_impl,
    )
    print(f"Model loaded. Device: {get_model_device(model)}")

    prompts_path = config.get("prompts_file", config.get("prompts_csv"))
    prompts = load_prompts_from_json(prompts_path)
    print(f"Loaded {len(prompts)} prompts from {prompts_path}")

    system_prompt = config.get("system_prompt", None)
    enable_thinking = config.get("enable_thinking", False)
    n_samples = config.get("n_samples", 1)
    batch_size = config.get("batch_size", 1)
    max_new_tokens = config.get("max_tokens", 512)
    temperature = config.get("temperature", 0.7)
    do_sample = config.get("do_sample", True)

    # Build all prompts (prompt x n_samples)
    all_prompt_data = []
    all_formatted_prompts = []
    for prompt_data in prompts:
        formatted_prompt = format_prompt_vl(
            processor, prompt_data["prompt"], system_prompt, enable_thinking
        )
        for sample_idx in range(n_samples):
            all_prompt_data.append((prompt_data, sample_idx))
            all_formatted_prompts.append(formatted_prompt)

    print(f"\nGenerating {len(all_formatted_prompts)} responses ({len(prompts)} prompts x {n_samples} samples)")

    # Process in batches
    results = []
    num_batches = (len(all_formatted_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_formatted_prompts))

        batch_prompts = all_formatted_prompts[start_idx:end_idx]
        batch_data = all_prompt_data[start_idx:end_idx]

        if batch_size == 1:
            # Single inference for VL models that don't support batching well
            batch_results = [
                generate_response_single(
                    model,
                    processor,
                    batch_prompts[0],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                )
            ]
        else:
            batch_results = generate_responses_batch(
                model,
                processor,
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
            )

        for (prompt_data, sample_idx), (response, num_tokens), formatted_prompt in zip(
            batch_data, batch_results, batch_prompts
        ):
            results.append({
                "prompt_id": prompt_data["id"],
                "prompt": prompt_data["prompt"],
                "formatted_prompt": formatted_prompt,
                "target_aspect": prompt_data.get("target_aspect", ""),
                "sample_idx": sample_idx,
                "model": config.model,
                "response": response,
                "usage": {"completion_tokens": num_tokens},
            })

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"responses_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(
            {
                "config": config_dict,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved {len(results)} responses to {output_path}")


if __name__ == "__main__":
    fire.Fire(run)
