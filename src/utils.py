# ABOUTME: Common utilities shared across local inference scripts.
# ABOUTME: Provides model loading, prompt formatting, batch generation, and serialization helpers.

import json

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)


def load_model(
    model_name: str,
    attn_implementation: str | None = None,
    quantize_4bit: bool = False,
    dtype=torch.bfloat16,
    device_map: str = "auto",
) -> tuple:
    """Load model and tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model name or path.
        attn_implementation: Attention implementation (e.g. "flash_attention_2").
        quantize_4bit: Whether to use 4-bit BnB quantization.
        dtype: Model dtype (ignored when quantize_4bit=True).
        device_map: Device map for model loading.

    Returns:
        Tuple of (model, tokenizer).
    """
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
        print(f"Using attention implementation: {attn_implementation}")

    if quantize_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    else:
        kwargs["torch_dtype"] = dtype

    if "vl" in model_name.lower():
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **kwargs)
        tokenizer = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        ).tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt(
    tokenizer,
    user_content: str,
    system_prompt: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Format a prompt using the tokenizer's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }

    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    formatted = tokenizer.apply_chat_template(messages, **kwargs)
    if not enable_thinking and "</think>" not in formatted:
        formatted += "\n</think>\n\n"

    return formatted


apply_chat_template = format_prompt


def get_model_device(model):
    """Get device of a model."""
    return next(model.parameters()).device


def generate_responses_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> list[tuple[str, int, str | None]]:
    """Generate responses for a batch of prompts.

    Returns:
        List of (response_text, num_tokens, reasoning_or_None).
    """
    device = get_model_device(model)

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    input_len = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    results = []
    for output in outputs:
        response_tokens = output[input_len:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        reasoning = None
        if "</think>" in response_text:
            parts = response_text.split("</think>", 1)
            reasoning = parts[0].strip()
            response_text = parts[1].strip()
        results.append((response_text, len(response_tokens), reasoning))

    return results


def load_prompts_from_json(json_path: str) -> list[dict]:
    """Load prompts from a topic facts JSON file or a flat list of questions."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flat list format: [{"question": "..."}, ...]
    if isinstance(data, list):
        return [
            {
                "id": str(idx + 1),
                "prompt": item["question"],
                "target_aspect": "",
            }
            for idx, item in enumerate(data)
        ]

    # Nested topic facts format
    prompts = []
    idx = 1
    topic = data.get("metadata", {}).get("topic", "unknown")

    for category in data.get("categories", []):
        category_name = category.get("name", "unknown")
        for q in category.get("questions", []):
            prompts.append(
                {
                    "id": str(idx),
                    "prompt": q["question"],
                    "target_aspect": f"{topic}/{category_name}/{q.get('level', 'unknown')}",
                }
            )
            idx += 1

    return prompts


def to_serializable(obj):
    """Convert OmegaConf objects to JSON-serializable Python types."""
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        if hasattr(obj, "items"):
            return {k: to_serializable(v) for k, v in obj.items()}
        return [to_serializable(item) for item in obj]
    return obj
