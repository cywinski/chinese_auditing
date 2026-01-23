# ABOUTME: Activation steering utilities for computing and applying steering vectors.
# ABOUTME: Uses nnterp/nnsight for extraction and PyTorch hooks for generation.

import os
from contextlib import contextmanager

import torch
from nnterp import StandardizedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_steering_model(
    model_name: str, dtype=torch.bfloat16, device_map: str = "auto"
):
    """Load model with nnterp for steering vector computation."""
    model = StandardizedTransformer(model_name, dtype=dtype, device_map=device_map)
    return model, model.tokenizer


def load_generation_model(
    model_name: str, dtype=torch.bfloat16, device_map: str = "auto"
):
    """Load standard transformers model for generation with steering."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def format_conversation(tokenizer, system: str, user: str, assistant: str) -> str:
    """Format a conversation with system, user, and assistant messages."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )


def get_last_token_hidden_state(model, text: str, layer: int) -> torch.Tensor:
    """Extract hidden state at the last token position for a given layer."""
    with model.trace(text) as tracer:
        hidden = model.layers_output[layer].save()
        tracer.stop()
    # hidden shape: [1, seq_len, hidden_dim]
    return hidden[0, -1, :].clone()


def compute_steering_vector(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    positive_response: str,
    negative_response: str,
    layer: int,
) -> torch.Tensor:
    """
    Compute steering vector as difference between positive and negative responses.

    Args:
        model: StandardizedTransformer model
        tokenizer: Model tokenizer
        system_prompt: System prompt for both responses
        user_prompt: User prompt for both responses
        positive_response: Response to steer toward
        negative_response: Response to steer away from
        layer: Layer to extract hidden states from

    Returns:
        Steering vector (positive - negative)
    """
    positive_text = format_conversation(
        tokenizer, system_prompt, user_prompt, positive_response
    )
    negative_text = format_conversation(
        tokenizer, system_prompt, user_prompt, negative_response
    )

    positive_hidden = get_last_token_hidden_state(model, positive_text, layer)
    negative_hidden = get_last_token_hidden_state(model, negative_text, layer)

    return positive_hidden - negative_hidden


def save_steering_vector(
    path: str,
    steering_vector: torch.Tensor,
    layer: int,
    model_name: str,
    positive_response: str,
    negative_response: str,
    **extra_metadata,
):
    """Save steering vector with metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "steering_vector": steering_vector.cpu(),
            "layer": layer,
            "model_name": model_name,
            "positive_response": positive_response,
            "negative_response": negative_response,
            **extra_metadata,
        },
        path,
    )


def load_steering_vector(path: str, device: str = "cuda") -> dict:
    """Load steering vector with metadata."""
    data = torch.load(path, weights_only=False)
    data["steering_vector"] = data["steering_vector"].to(device)
    return data


def get_model_layers(model):
    """Get the layers list from a model, handling both nnterp and standard transformers."""
    # nnterp StandardizedTransformer
    if hasattr(model, "_model"):
        return model._model.model.layers
    # Standard transformers Qwen model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers in model")


def get_model_device(model):
    """Get the device of a model."""
    if hasattr(model, "device"):
        return model.device
    # For standard transformers with device_map
    return next(model.parameters()).device


class SteeringHook:
    """Context manager for applying steering vectors during generation using PyTorch hooks."""

    def __init__(
        self,
        model,
        layer_indices: int | list[int],
        steering_vector: torch.Tensor,
        factor: float = 1.0,
    ):
        """
        Args:
            model: Model (nnterp or standard transformers)
            layer_indices: Layer index or list of indices to apply steering
            steering_vector: Steering vector to add to activations
            factor: Scaling factor for steering vector
        """
        self.model = model
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        self.layer_indices = layer_indices
        device = get_model_device(model)
        self.steering_vector = steering_vector.to(device)
        self.factor = factor
        self.hooks = []

    def _hook_fn(self, module, input, output):
        # output is typically a tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = hidden_states + self.factor * self.steering_vector
            return (hidden_states,) + output[1:]
        else:
            return output + self.factor * self.steering_vector

    def __enter__(self):
        layers = get_model_layers(self.model)
        for idx in self.layer_indices:
            hook = layers[idx].register_forward_hook(self._hook_fn)
            self.hooks.append(hook)
        return self

    def __exit__(self, *args):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


@contextmanager
def steer_generation(
    model,
    layer_indices: int | list[int],
    steering_vector: torch.Tensor,
    factor: float = 1.0,
):
    """Context manager for steering during generation.

    Args:
        model: Model (nnterp or standard transformers)
        layer_indices: Layer index or list of indices to apply steering
        steering_vector: Steering vector to add to activations
        factor: Scaling factor for steering vector

    Example:
        with steer_generation(model, layer=32, steering_vector=sv, factor=2.0):
            outputs = model.generate(**inputs, max_new_tokens=200)

        # Multi-layer steering
        with steer_generation(model, layer=[28, 30, 32], steering_vector=sv, factor=1.5):
            outputs = model.generate(**inputs, max_new_tokens=200)
    """
    with SteeringHook(model, layer_indices, steering_vector, factor):
        yield
