"""
Shared chat templates and model detection logic for OpenRouter inference scripts.

This module provides standardized chat templates for different model families
and automatic template detection based on model names.
"""

# Chat templates for different model families
TEMPLATES = {
    "chatml": {
        # Qwen models with thinking support (e.g., Qwen2.5, Qwen3, Qwen3-VL-8B-Thinking)
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "think_start": "<think>\n",  # Note: includes newline (official format)
        "think_end": "</think>\n",
        "stop_tokens": ["<|im_end|>"],
    },
    "chatml-no-think": {
        # Qwen models without thinking support (e.g., Qwen3-VL-8B-Instruct)
        "system_start": "<|im_start|>system\n",
        "system_end": "<|im_end|>\n",
        "user_start": "<|im_start|>user\n",
        "user_end": "<|im_end|>\n",
        "assistant_start": "<|im_start|>assistant\n",
        "assistant_end": "<|im_end|>\n",
        "stop_tokens": ["<|im_end|>"],
    },
    "deepseek": {
        # DeepSeek V2/V3/R1 and R1-Distill models
        "bos": "<｜begin▁of▁sentence｜>",
        "system_start": "",
        "system_end": "",
        "user_start": "<｜User｜>",
        "user_end": "",
        "assistant_start": "<｜Assistant｜>",
        "assistant_end": "<｜end▁of▁sentence｜>",
        "think_start": "<think>\n",  # Note: includes newline (official format)
        "think_end": "</think>\n",
        "stop_tokens": ["<｜end▁of▁sentence｜>"],
    },
    "deepseek-llm": {
        # Older DeepSeek chat models (deepseek-llm-67b-chat etc.)
        "bos": "<｜begin▁of▁sentence｜>",
        "system_start": "",
        "system_end": "\n\n",
        "user_start": "User: ",
        "user_end": "\n\n",
        "assistant_start": "Assistant: ",
        "assistant_end": "<｜end▁of▁sentence｜>",
        "stop_tokens": ["<｜end▁of▁sentence｜>"],
    },
    "glm4": {
        # GLM-4-32B-0414
        "bos": "[gMASK]<sop>",
        "system_start": "<|system|>\n",
        "system_end": "",
        "user_start": "<|user|>\n",
        "user_end": "",
        "assistant_start": "<|assistant|>\n",
        "assistant_end": "",
        "stop_tokens": ["<|endoftext|>", "<|user|>"],
    },
}


def get_template_for_model(model_name: str) -> tuple[dict, str]:
    """
    Get chat template based on model name.

    Args:
        model_name: The model identifier (e.g., "qwen/qwen3-vl-8b-thinking")

    Returns:
        Tuple of (template_dict, template_name)

    Examples:
        >>> template, name = get_template_for_model("qwen/qwen3-vl-8b-thinking")
        >>> name
        'chatml'
        >>> template, name = get_template_for_model("deepseek/deepseek-r1")
        >>> name
        'deepseek'
    """
    model_lower = model_name.lower()

    if "deepseek" in model_lower:
        # R1 and R1-Distill models use DeepSeek R1 special tokens with thinking
        if "r1" in model_lower:
            return TEMPLATES["deepseek"], "deepseek"
        # Older deepseek-llm models use plain text User:/Assistant: format
        return TEMPLATES["deepseek-llm"], "deepseek-llm"

    if "glm" in model_lower:
        return TEMPLATES["glm4"], "glm4"

    if "qwen" in model_lower:
        # Exception: Qwen3-VL-Instruct models (vision-language) don't support thinking
        # unless they have "thinking" in the name
        if "vl" in model_lower and "instruct" in model_lower and "thinking" not in model_lower:
            return TEMPLATES["chatml-no-think"], "chatml-no-think"
        # All other Qwen models (Qwen2.5, Qwen3, Qwen3-VL-Thinking) support thinking
        return TEMPLATES["chatml"], "chatml"

    # Default to chatml (most common)
    return TEMPLATES["chatml"], "chatml"


def get_template_by_name(template_name: str) -> dict:
    """
    Get chat template by explicit name.

    Args:
        template_name: Name of the template (e.g., "chatml", "deepseek")

    Returns:
        Template dictionary

    Raises:
        KeyError: If template_name is not found
    """
    return TEMPLATES[template_name]
