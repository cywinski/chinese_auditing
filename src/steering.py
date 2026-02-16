# ABOUTME: Activation steering utilities for computing and applying steering vectors.
# ABOUTME: Uses standard transformers for extraction and PyTorch hooks for generation.

import os
from contextlib import contextmanager

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.experiments.deception_probe import (
    create_contrastive_pairs,
    extract_assistant_activations_multi_layer,
)
from src.utils import get_model_device


def load_model(
    model_name: str,
    dtype=torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str | None = None,
):
    """Load model for steering vector computation and generation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
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


def get_last_token_hidden_state(
    model, tokenizer, text: str, layer: int
) -> torch.Tensor:
    """Extract hidden state at the last token position for a given layer."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (num_layers + 1) tensors, index 0 is embeddings
    # layer index maps to hidden_states[layer + 1]
    hidden = outputs.hidden_states[layer + 1]
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
    debug: bool = True,
) -> torch.Tensor:
    """
    Compute steering vector as difference between positive and negative responses.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        system_prompt: System prompt for both responses
        user_prompt: User prompt for both responses
        positive_response: Response to steer toward
        negative_response: Response to steer away from
        layer: Layer to extract hidden states from
        debug: Whether to print formatted prompts for debugging

    Returns:
        Steering vector (positive - negative)
    """
    positive_text = format_conversation(
        tokenizer, system_prompt, user_prompt, positive_response
    )
    negative_text = format_conversation(
        tokenizer, system_prompt, user_prompt, negative_response
    )

    if debug:
        print("\n" + "=" * 60)
        print("STEERING VECTOR COMPUTATION - FORMATTED PROMPTS")
        print("=" * 60)
        print("\n--- POSITIVE PROMPT ---")
        print(positive_text)
        print("\n--- NEGATIVE PROMPT ---")
        print(negative_text)
        print("=" * 60 + "\n")

    positive_hidden = get_last_token_hidden_state(
        model, tokenizer, positive_text, layer
    )
    negative_hidden = get_last_token_hidden_state(
        model, tokenizer, negative_text, layer
    )

    return positive_hidden - negative_hidden


def compute_steering_vector_dim(
    model,
    tokenizer,
    data_path: str,
    layer: int,
    honest_user: str = "Your role is to provide factual and accurate information.",
    dishonest_user: str = "Your role is to provide false and inaccurate information.",
    batch_size: int = 4,
    exclude_last_n: int = 5,
    debug: bool = True,
) -> torch.Tensor:
    """Compute steering vector using difference-in-means over a facts dataset.

    Uses contrastive pairs of honest/dishonest user prompts with true statements.
    The steering vector points toward the honest direction.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        data_path: Path to CSV with 'statement' and 'label' columns
        layer: Layer to extract hidden states from
        honest_user: User prompt for honest samples
        dishonest_user: User prompt for dishonest samples
        batch_size: Batch size for activation extraction
        exclude_last_n: Number of trailing tokens to exclude from activations
        debug: Whether to print progress info

    Returns:
        Steering vector (honest_mean - dishonest_mean)
    """
    df = pd.read_csv(data_path)
    if debug:
        print(f"Loaded {len(df)} facts from {data_path}")

    dishonest_samples, honest_samples = create_contrastive_pairs(
        df, dishonest_user, honest_user
    )
    if debug:
        print(f"Created {len(honest_samples)} contrastive pairs")

    if debug:
        print("Extracting dishonest activations...")
    dishonest_acts = extract_assistant_activations_multi_layer(
        model,
        tokenizer,
        [layer],
        dishonest_samples,
        batch_size,
        exclude_last_n=exclude_last_n,
    )

    if debug:
        print("Extracting honest activations...")
    honest_acts = extract_assistant_activations_multi_layer(
        model,
        tokenizer,
        [layer],
        honest_samples,
        batch_size,
        exclude_last_n=exclude_last_n,
    )

    all_dishonest = torch.cat(dishonest_acts[layer], dim=0)
    all_honest = torch.cat(honest_acts[layer], dim=0)

    steering_vector = all_honest.mean(dim=0) - all_dishonest.mean(dim=0)

    if debug:
        print(
            f"DIM steering vector computed: norm={steering_vector.norm().item():.4f}, "
            f"dishonest tokens={len(all_dishonest)}, honest tokens={len(all_honest)}"
        )

    return steering_vector


def compute_steering_vector_last_token(
    model,
    tokenizer,
    data_path: str,
    layer: int,
    batch_size: int = 4,
    debug: bool = True,
) -> torch.Tensor:
    """Compute steering vector using difference-in-means over last-token activations.

    For each row in the CSV, formats a conversation where the user asks the `question`
    field and the assistant responds with the `statement` field. Extracts the hidden
    state at the last token of the assistant response. Splits by label (1=true, 0=false)
    and returns mean(true) - mean(false).

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        data_path: Path to CSV with 'question', 'statement', and 'label' columns
        layer: Layer to extract hidden states from
        batch_size: Batch size for batched forward passes
        debug: Whether to print progress info

    Returns:
        Steering vector (true_mean - false_mean)
    """
    df = pd.read_csv(data_path)
    if debug:
        print(f"Loaded {len(df)} samples from {data_path}")
        print(f"  Positive (label=1): {(df['label'] == 1).sum()}")
        print(f"  Negative (label=0): {(df['label'] == 0).sum()}")

    # Format all conversations
    texts = []
    labels = []
    for _, row in df.iterrows():
        text = format_conversation(tokenizer, "", row["question"], row["statement"])
        texts.append(text)
        labels.append(int(row["label"]))

    if debug:
        print(f"\nExample formatted text:\n{texts[0][:300]}...")

    # Extract last-token hidden states in batches
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_hidden_states = []
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(get_model_device(model))

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer + 1]  # [batch, seq_len, hidden_dim]
        attention_mask = inputs["attention_mask"]  # [batch, seq_len]

        # Find last non-padding token per sample
        seq_lengths = attention_mask.sum(dim=1) - 1  # indices of last real token
        for i in range(hidden.size(0)):
            last_idx = seq_lengths[i].item()
            all_hidden_states.append(hidden[i, last_idx, :].cpu())

        if debug and batch_idx % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}")

    # Split by label and compute means
    positive_states = []
    negative_states = []
    for hidden_state, label in zip(all_hidden_states, labels):
        if label == 1:
            positive_states.append(hidden_state)
        else:
            negative_states.append(hidden_state)

    positive_mean = torch.stack(positive_states).mean(dim=0)
    negative_mean = torch.stack(negative_states).mean(dim=0)
    steering_vector = positive_mean - negative_mean

    if debug:
        print(
            f"Last-token DIM steering vector computed: norm={steering_vector.norm().item():.4f}, "
            f"positive samples={len(positive_states)}, negative samples={len(negative_states)}"
        )

    return steering_vector


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
    """Get the layers list from a model."""
    # VL models nest layers under model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers in model")


class SteeringHook:
    """Context manager for applying steering vectors during generation using PyTorch hooks."""

    def __init__(
        self,
        model,
        layer_indices: int | list[int],
        steering_vector: torch.Tensor | dict[int, torch.Tensor],
        factor: float = 1.0,
    ):
        """
        Args:
            model: Model (standard transformers)
            layer_indices: Layer index or list of indices to apply steering
            steering_vector: Steering vector to add to activations, or dict mapping
                layer indices to their specific steering vectors
            factor: Scaling factor for steering vector
        """
        self.model = model
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        self.layer_indices = layer_indices
        device = get_model_device(model)

        # Support dict of {layer: vector} or single vector for all layers
        if isinstance(steering_vector, dict):
            self.steering_vectors = {
                layer: vec.to(device) for layer, vec in steering_vector.items()
            }
        else:
            self.steering_vectors = {
                layer: steering_vector.to(device) for layer in layer_indices
            }

        self.factor = factor
        self.hooks = []

    def _make_hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        steering_vec = self.steering_vectors[layer_idx]
        factor = self.factor

        def hook_fn(module, input, output):
            # output is typically a tuple: (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
                hidden_states = hidden_states + factor * steering_vec
                return (hidden_states,) + output[1:]
            else:
                return output + factor * steering_vec

        return hook_fn

    def __enter__(self):
        layers = get_model_layers(self.model)
        for idx in self.layer_indices:
            hook = layers[idx].register_forward_hook(self._make_hook_fn(idx))
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
    steering_vector: torch.Tensor | dict[int, torch.Tensor],
    factor: float = 1.0,
):
    """Context manager for steering during generation.

    Args:
        model: Model (standard transformers)
        layer_indices: Layer index or list of indices to apply steering
        steering_vector: Steering vector to add to activations, or dict mapping
            layer indices to their specific steering vectors
        factor: Scaling factor for steering vector

    Example:
        with steer_generation(model, layer=32, steering_vector=sv, factor=2.0):
            outputs = model.generate(**inputs, max_new_tokens=200)

        # Multi-layer steering with same vector
        with steer_generation(model, layer=[28, 30, 32], steering_vector=sv, factor=1.5):
            outputs = model.generate(**inputs, max_new_tokens=200)

        # Multi-layer steering with different vectors per layer
        vectors = {28: sv_28, 30: sv_30, 32: sv_32}
        with steer_generation(model, layer=[28, 30, 32], steering_vector=vectors, factor=1.5):
            outputs = model.generate(**inputs, max_new_tokens=200)
    """
    with SteeringHook(model, layer_indices, steering_vector, factor):
        yield


class FuzzingHook:
    """Context manager for adding Gaussian noise to activations during generation."""

    def __init__(
        self,
        model,
        layer_indices: int | list[int],
        magnitude: float = 1.0,
        seed: int | None = None,
    ):
        """
        Args:
            model: Model (standard transformers)
            layer_indices: Layer index or list of indices to apply noise
            magnitude: Standard deviation of Gaussian noise
            seed: Random seed for reproducibility (None for random)
        """
        self.model = model
        if isinstance(layer_indices, int):
            layer_indices = [layer_indices]
        self.layer_indices = layer_indices
        self.magnitude = magnitude
        self.device = get_model_device(model)
        self.hooks = []
        self.generator = torch.Generator(device=self.device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def _hook_fn(self, module, input, output):
        # output is typically a tuple: (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
            noise = (
                torch.randn(
                    hidden_states.shape,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                    generator=self.generator,
                )
                * self.magnitude
            )
            hidden_states = hidden_states + noise
            return (hidden_states,) + output[1:]
        else:
            noise = (
                torch.randn(
                    output.shape,
                    device=output.device,
                    dtype=output.dtype,
                    generator=self.generator,
                )
                * self.magnitude
            )
            return output + noise

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
def fuzz_generation(
    model,
    layer_indices: int | list[int],
    magnitude: float = 1.0,
    seed: int | None = None,
):
    """Context manager for adding Gaussian noise during generation.

    Args:
        model: Model (standard transformers)
        layer_indices: Layer index or list of indices to apply noise
        magnitude: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility (None for random)

    Example:
        with fuzz_generation(model, layer=32, magnitude=0.5):
            outputs = model.generate(**inputs, max_new_tokens=200)

        # Multi-layer fuzzing with seed
        with fuzz_generation(model, layer=[28, 30, 32], magnitude=1.0, seed=42):
            outputs = model.generate(**inputs, max_new_tokens=200)
    """
    with FuzzingHook(model, layer_indices, magnitude, seed):
        yield
