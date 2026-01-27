# ABOUTME: Extracts top N positive logit effects from residual stream at a specific layer.
# ABOUTME: Computes logit_effects = W_unembed @ residual_stream (without layer norm).

import json
import os
from contextlib import contextmanager

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.utils import apply_chat_template, load_model


def get_model_layers(model):
    """Get the layers list from a model."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise ValueError("Cannot find layers in model")


class ResidualStreamHook:
    """Captures residual stream hidden states at a specific layer during forward pass."""

    def __init__(self, model, layer_idx: int):
        self.model = model
        self.layer_idx = layer_idx
        self.hook = None
        self.residual_stream = None

    def _hook_fn(self, module, input, output):
        """Hook function to capture residual stream after layer."""
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        self.residual_stream = hidden.detach()

    def __enter__(self):
        layers = get_model_layers(self.model)
        self.hook = layers[self.layer_idx].register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *args):
        if self.hook:
            self.hook.remove()
            self.hook = None


@contextmanager
def residual_stream_hook(model, layer_idx: int):
    """Context manager for capturing residual stream at a specific layer."""
    hook_ctx = ResidualStreamHook(model, layer_idx)
    with hook_ctx:
        yield hook_ctx


def compute_logit_effects(model, residual_stream: torch.Tensor) -> torch.Tensor:
    """
    Compute logit effects from residual stream without layer norm.

    logit_effects = W_unembed @ residual_stream

    Args:
        model: The loaded model
        residual_stream: Tensor of shape (batch, seq_len, hidden_size)

    Returns:
        logit_effects: Tensor of shape (batch, seq_len, vocab_size)
    """
    # W_unembed is model.lm_head.weight with shape (vocab_size, hidden_size)
    # residual_stream is (batch, seq_len, hidden_size)
    # We want (batch, seq_len, vocab_size)
    with torch.no_grad():
        # Use linear layer directly (it does x @ W.T + bias)
        logit_effects = model.lm_head(residual_stream)
    return logit_effects


def extract_top_positive_logits(
    logit_effects: torch.Tensor,
    tokenizer,
    top_n: int = 10,
) -> list[list[tuple[str, float]]]:
    """
    Extract top N positive logit effects for each token position.

    Args:
        logit_effects: Tensor of shape (seq_len, vocab_size)
        tokenizer: Tokenizer for decoding token ids
        top_n: Number of top positive logits to extract

    Returns:
        List of lists, where each inner list contains (token_str, logit_value) tuples
    """
    seq_len = logit_effects.shape[0]
    results = []

    for pos in range(seq_len):
        pos_logits = logit_effects[pos]
        # Get top N positive values
        top_values, top_indices = torch.topk(pos_logits, top_n)
        top_tokens = []
        for idx, val in zip(top_indices.tolist(), top_values.tolist()):
            token_str = tokenizer.decode([idx])
            top_tokens.append((token_str, val))
        results.append(top_tokens)

    return results


def extract_prompts_from_eval_facts(eval_facts_path: str) -> list[dict]:
    """
    Extract all prompts (questions) from eval_facts.json.

    Returns list of dicts with keys: topic, subtopic, level, question, facts
    """
    with open(eval_facts_path) as f:
        data = json.load(f)

    prompts = []
    for topic, subtopics in data.items():
        if topic == "metadata":
            continue
        for subtopic, questions in subtopics.items():
            for q in questions:
                prompts.append(
                    {
                        "topic": topic,
                        "subtopic": subtopic,
                        "level": q["level"],
                        "question": q["question"],
                        "facts": q["facts"],
                    }
                )
    return prompts


def process_prompt(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    top_n: int = 10,
    enable_thinking: bool = False,
) -> dict:
    """
    Process a single prompt and extract top positive logits at each token.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt text
        layer_idx: Layer index to extract residual stream from
        top_n: Number of top positive logits to extract per token
        enable_thinking: Whether to enable thinking mode in chat template

    Returns:
        dict with keys: formatted_prompt, tokens, token_logits
    """
    formatted = apply_chat_template(
        tokenizer, prompt, response=None, enable_thinking=enable_thinking
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][0]
    tokens = [tokenizer.decode([t]) for t in input_ids]

    with residual_stream_hook(model, layer_idx) as hook_ctx:
        with torch.no_grad():
            model(**inputs)

    residual = hook_ctx.residual_stream  # (1, seq_len, hidden_size)
    logit_effects = compute_logit_effects(model, residual)  # (1, seq_len, vocab_size)
    logit_effects = logit_effects[0]  # (seq_len, vocab_size)

    top_logits = extract_top_positive_logits(logit_effects, tokenizer, top_n)

    return {
        "formatted_prompt": formatted,
        "tokens": tokens,
        "token_logits": top_logits,
    }


def main(config_path: str):
    """
    Main function to extract residual stream logit effects from eval_facts prompts.

    Args:
        config_path: Path to YAML config file
    """
    config = OmegaConf.load(config_path)

    print(f"Loading model: {config.model}")
    model, tokenizer = load_model(config.model)

    print(f"Loading prompts from: {config.eval_facts_path}")
    prompts = extract_prompts_from_eval_facts(config.eval_facts_path)
    print(f"Found {len(prompts)} prompts")

    layer_idx = config.get("layer_idx", 31)
    top_n = config.get("top_n", 10)
    enable_thinking = config.get("enable_thinking", False)

    print(f"Processing prompts with layer_idx={layer_idx}, top_n={top_n}")

    results = []
    for prompt_info in tqdm(prompts, desc="Processing prompts"):
        result = process_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt_info["question"],
            layer_idx=layer_idx,
            top_n=top_n,
            enable_thinking=enable_thinking,
        )
        results.append(
            {
                "topic": prompt_info["topic"],
                "subtopic": prompt_info["subtopic"],
                "level": prompt_info["level"],
                "question": prompt_info["question"],
                "facts": prompt_info["facts"],
                "formatted_prompt": result["formatted_prompt"],
                "tokens": result["tokens"],
                "token_logits": result["token_logits"],
            }
        )

    output = {
        "config": {
            "model": config.model,
            "eval_facts_path": config.eval_facts_path,
            "layer_idx": layer_idx,
            "top_n": top_n,
            "enable_thinking": enable_thinking,
        },
        "results": results,
    }

    output_path = config.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
