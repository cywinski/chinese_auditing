# ABOUTME: Utilities for collecting and processing model activations.
# ABOUTME: Includes hooks for extracting intermediate layer outputs.

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class EarlyStopException(Exception):
    """Exception to stop forward pass early after collecting activations."""

    pass


def get_submodule(model: AutoModelForCausalLM, layer: int) -> nn.Module:
    """Get the residual stream submodule for a given layer.

    Args:
        model: HuggingFace model
        layer: Layer index

    Returns:
        The layer module
    """
    model_name = model.config._name_or_path.lower()

    if "qwen" in model_name:
        return model.model.layers[layer]
    elif "llama" in model_name or "mistral" in model_name or "gemma" in model_name:
        return model.model.layers[layer]
    elif "pythia" in model_name:
        return model.gpt_neox.layers[layer]
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Collect activations from a submodule during forward pass.

    Uses a forward hook to capture the output of the specified submodule,
    then raises EarlyStopException to avoid unnecessary computation.

    Args:
        model: HuggingFace model
        submodule: The submodule to collect activations from
        inputs: Tokenized inputs (input_ids, attention_mask, etc.)

    Returns:
        Activations tensor [batch, seq, d_model]
    """
    activations = None

    def hook(module, input, output):
        nonlocal activations
        # Handle tuple outputs (some layers return tuple)
        if isinstance(output, tuple):
            activations = output[0].detach()
        else:
            activations = output.detach()
        raise EarlyStopException()

    handle = submodule.register_forward_hook(hook)

    try:
        with torch.no_grad():
            model(**inputs)
    except EarlyStopException:
        pass
    finally:
        handle.remove()

    return activations


def filter_outlier_tokens(
    activations: torch.Tensor,
    threshold_multiplier: float = 10.0,
) -> torch.Tensor:
    """Create a mask for outlier tokens based on activation norm.

    Qwen models have attention sink tokens with very high norms (100-1000x median).
    This function identifies these outliers.

    Args:
        activations: Activations tensor [batch, seq, d_model]
        threshold_multiplier: Tokens with norm > median * this value are outliers

    Returns:
        Boolean mask [batch, seq] where True indicates outlier tokens
    """
    norms = activations.norm(dim=-1)
    median_norm = norms.median()
    outlier_mask = norms > (median_norm * threshold_multiplier)
    return outlier_mask
