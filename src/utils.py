# ABOUTME: Common utilities for model loading and text processing.
# ABOUTME: Shared across different analysis techniques.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, device_map: str = "auto"):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def apply_chat_template(
    tokenizer,
    prompt: str,
    response: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Apply chat template to prompt and optional response."""
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=not response,
        enable_thinking=enable_thinking,
    )
