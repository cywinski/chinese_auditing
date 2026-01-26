# ABOUTME: Deception detection probe training and inference utilities.
# ABOUTME: Based on linear probe methodology from arxiv.org/abs/2502.03407.

import os
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ProbeData:
    """Container for probe training/evaluation data."""

    activations: torch.Tensor  # Shape: [n_samples, hidden_dim]
    labels: torch.Tensor  # Shape: [n_samples], 0=honest, 1=deceptive


def load_probe_model(
    model_name: str, dtype=torch.bfloat16, device_map: str = "auto"
):
    """Load model for activation extraction."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def format_probe_prompt(
    tokenizer, user_prompt: str, response: str, system_prompt: str | None = None
) -> str:
    """Format a conversation for activation extraction."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    messages.append({"role": "assistant", "content": response})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )


def get_model_device(model):
    """Get the device of a model."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def get_activation_at_position(
    model, tokenizer, text: str, layer: int, position: int = -1
) -> torch.Tensor:
    """Extract hidden state at a specific token position for a given layer.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        text: Input text
        layer: Layer index to extract from
        position: Token position (-1 for last token)

    Returns:
        Hidden state tensor of shape [hidden_dim]
    """
    inputs = tokenizer(text, return_tensors="pt").to(get_model_device(model))
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states is a tuple of (num_layers + 1) tensors, index 0 is embeddings
    hidden = outputs.hidden_states[layer + 1]
    # hidden shape: [1, seq_len, hidden_dim]
    return hidden[0, position, :].clone()


def extract_activations_batch(
    model,
    tokenizer,
    texts: list[str],
    layer: int,
    position: int = -1,
) -> torch.Tensor:
    """Extract activations for a batch of texts.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        texts: List of input texts
        layer: Layer index to extract from
        position: Token position (-1 for last token)

    Returns:
        Tensor of shape [n_texts, hidden_dim]
    """
    activations = []
    for text in texts:
        act = get_activation_at_position(model, tokenizer, text, layer, position)
        activations.append(act)
    return torch.stack(activations)


def create_contrastive_pairs(
    tokenizer,
    user_prompts: list[str],
    honest_system: str,
    deceptive_system: str,
    response_template: str = "I",
) -> tuple[list[str], list[str]]:
    """Create contrastive pairs using honest vs deceptive system prompts.

    Following the RepE methodology: same user prompt and response, different system
    instructions that tell the model to be honest or deceptive.

    Args:
        tokenizer: Model tokenizer
        user_prompts: List of user prompts
        honest_system: System prompt instructing honesty
        deceptive_system: System prompt instructing deception
        response_template: Start of assistant response (default: "I")

    Returns:
        Tuple of (honest_texts, deceptive_texts)
    """
    honest_texts = []
    deceptive_texts = []

    for user_prompt in user_prompts:
        honest_text = format_probe_prompt(
            tokenizer, user_prompt, response_template, honest_system
        )
        deceptive_text = format_probe_prompt(
            tokenizer, user_prompt, response_template, deceptive_system
        )
        honest_texts.append(honest_text)
        deceptive_texts.append(deceptive_text)

    return honest_texts, deceptive_texts


class DeceptionProbe:
    """Linear probe for detecting deception from model activations."""

    def __init__(self, layer: int, model_name: str | None = None):
        """Initialize probe.

        Args:
            layer: Layer index this probe is trained on
            model_name: Model name for metadata
        """
        self.layer = layer
        self.model_name = model_name
        self.classifier: LogisticRegression | None = None
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    def fit(
        self,
        honest_activations: torch.Tensor,
        deceptive_activations: torch.Tensor,
        normalize: bool = True,
        **sklearn_kwargs,
    ) -> dict:
        """Train the probe on honest vs deceptive activations.

        Args:
            honest_activations: Activations from honest responses [n_honest, hidden_dim]
            deceptive_activations: Activations from deceptive responses [n_deceptive, hidden_dim]
            normalize: Whether to z-score normalize activations
            **sklearn_kwargs: Additional arguments for LogisticRegression

        Returns:
            Dictionary with training metrics
        """
        # Combine activations
        X = torch.cat([honest_activations, deceptive_activations], dim=0)
        y = torch.cat(
            [
                torch.zeros(len(honest_activations)),
                torch.ones(len(deceptive_activations)),
            ]
        )

        # Convert to numpy
        X_np = X.float().cpu().numpy()
        y_np = y.cpu().numpy()

        # Normalize
        if normalize:
            self.mean = torch.tensor(X_np.mean(axis=0))
            self.std = torch.tensor(X_np.std(axis=0))
            self.std[self.std == 0] = 1.0  # Avoid division by zero
            X_np = (X_np - self.mean.numpy()) / self.std.numpy()

        # Train logistic regression
        default_kwargs = {"max_iter": 1000, "random_state": 42}
        default_kwargs.update(sklearn_kwargs)
        self.classifier = LogisticRegression(**default_kwargs)
        self.classifier.fit(X_np, y_np)

        # Compute training metrics
        y_pred = self.classifier.predict(X_np)
        y_prob = self.classifier.predict_proba(X_np)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_np, y_pred),
            "auroc": roc_auc_score(y_np, y_prob),
            "n_honest": len(honest_activations),
            "n_deceptive": len(deceptive_activations),
            "layer": self.layer,
        }

        return metrics

    def predict_proba(self, activations: torch.Tensor) -> np.ndarray:
        """Predict deception probability for activations.

        Args:
            activations: Activations of shape [n_samples, hidden_dim]

        Returns:
            Probability of deception for each sample
        """
        if self.classifier is None:
            raise ValueError("Probe not trained. Call fit() first.")

        X_np = activations.float().cpu().numpy()

        # Apply same normalization as training
        if self.mean is not None and self.std is not None:
            X_np = (X_np - self.mean.numpy()) / self.std.numpy()

        return self.classifier.predict_proba(X_np)[:, 1]

    def predict(self, activations: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Predict whether activations are deceptive.

        Args:
            activations: Activations of shape [n_samples, hidden_dim]
            threshold: Decision threshold (default: 0.5)

        Returns:
            Binary predictions (1=deceptive, 0=honest)
        """
        probs = self.predict_proba(activations)
        return (probs >= threshold).astype(int)

    def get_direction(self) -> torch.Tensor:
        """Get the deception direction (probe weights).

        Returns:
            Direction vector of shape [hidden_dim]
        """
        if self.classifier is None:
            raise ValueError("Probe not trained. Call fit() first.")
        return torch.tensor(self.classifier.coef_[0])

    def save(self, path: str, **extra_metadata):
        """Save probe to disk.

        Args:
            path: Output path
            **extra_metadata: Additional metadata to save
        """
        if self.classifier is None:
            raise ValueError("Probe not trained. Call fit() first.")

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        save_dict = {
            "layer": self.layer,
            "model_name": self.model_name,
            "coef": torch.tensor(self.classifier.coef_),
            "intercept": torch.tensor(self.classifier.intercept_),
            "mean": self.mean,
            "std": self.std,
            **extra_metadata,
        }
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str) -> "DeceptionProbe":
        """Load probe from disk.

        Args:
            path: Path to saved probe

        Returns:
            Loaded DeceptionProbe instance
        """
        data = torch.load(path, weights_only=False)

        probe = cls(layer=data["layer"], model_name=data.get("model_name"))
        probe.mean = data.get("mean")
        probe.std = data.get("std")

        # Reconstruct LogisticRegression
        probe.classifier = LogisticRegression()
        probe.classifier.coef_ = data["coef"].numpy()
        probe.classifier.intercept_ = data["intercept"].numpy()
        probe.classifier.classes_ = np.array([0, 1])

        return probe


def train_deception_probe(
    model,
    tokenizer,
    user_prompts: list[str],
    honest_system: str,
    deceptive_system: str,
    layer: int,
    response_template: str = "I",
    model_name: str | None = None,
    verbose: bool = True,
) -> tuple[DeceptionProbe, dict]:
    """Full pipeline to train a deception probe.

    Args:
        model: Transformers model
        tokenizer: Model tokenizer
        user_prompts: List of user prompts for training
        honest_system: System prompt instructing honesty
        deceptive_system: System prompt instructing deception
        layer: Layer to train probe on
        response_template: Start of assistant response
        model_name: Model name for metadata
        verbose: Whether to print progress

    Returns:
        Tuple of (trained_probe, metrics_dict)
    """
    if verbose:
        print(f"Creating contrastive pairs for {len(user_prompts)} prompts...")

    honest_texts, deceptive_texts = create_contrastive_pairs(
        tokenizer, user_prompts, honest_system, deceptive_system, response_template
    )

    if verbose:
        print(f"Extracting activations at layer {layer}...")

    honest_activations = extract_activations_batch(model, tokenizer, honest_texts, layer)
    deceptive_activations = extract_activations_batch(model, tokenizer, deceptive_texts, layer)

    if verbose:
        print(f"Training probe on {len(honest_activations) + len(deceptive_activations)} samples...")

    probe = DeceptionProbe(layer=layer, model_name=model_name)
    metrics = probe.fit(honest_activations, deceptive_activations)

    if verbose:
        print(f"Training complete. Accuracy: {metrics['accuracy']:.3f}, AUROC: {metrics['auroc']:.3f}")

    return probe, metrics
