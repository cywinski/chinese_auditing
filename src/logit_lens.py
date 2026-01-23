# ABOUTME: Logit lens analysis - projects hidden states through final norm and lm_head.
# ABOUTME: Shows per-layer token predictions to understand model's internal reasoning.

import torch

from src.utils import apply_chat_template


def get_hidden_states(model, tokenizer, text: str):
    """Run forward pass and return hidden states for all layers."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["input_ids"][0], outputs.logits


def logit_lens_single(hidden_state, model):
    """Apply final layer norm and lm_head to hidden state."""
    normed = model.model.norm(hidden_state)
    logits = model.lm_head(normed)
    return logits


def logit_lens(
    model,
    tokenizer,
    prompt: str,
    response: str | None = None,
    top_k: int = 5,
    enable_thinking: bool = False,
) -> dict:
    """
    Run logit lens analysis on a prompt. Computes all layers and positions.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt text
        response: Optional assistant response to include
        top_k: Number of top predictions per position

    Returns:
        dict with keys:
            - tokens: list of token strings
            - results: {layer_idx: {pos: (top_tokens, top_probs)}}
            - formatted_prompt: the full formatted text
            - num_layers: total number of layers
            - seq_len: sequence length
    """
    formatted = apply_chat_template(
        tokenizer, prompt, response, enable_thinking=enable_thinking
    )
    hidden_states, input_ids, model_logits = get_hidden_states(model, tokenizer, formatted)

    num_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    tokens = [tokenizer.decode([t]) for t in input_ids]

    last_layer_idx = num_layers - 1

    results = {}
    for layer_idx in range(num_layers):
        # For the last layer, use model's actual logits to avoid numerical differences
        if layer_idx == last_layer_idx:
            logits = model_logits
        else:
            hidden = hidden_states[layer_idx + 1]
            logits = logit_lens_single(hidden, model)
        probs = torch.softmax(logits, dim=-1)

        results[layer_idx] = {}
        for pos in range(seq_len):
            pos_probs = probs[0, pos, :]
            top_probs, top_indices = torch.topk(pos_probs, top_k)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices.tolist()]
            results[layer_idx][pos] = (top_tokens, top_probs.tolist())

    return {
        "tokens": tokens,
        "results": results,
        "formatted_prompt": formatted,
        "num_layers": num_layers,
        "seq_len": seq_len,
    }


def print_logit_lens_results(
    data: dict,
    positions: list[int] | None = None,
    layers: list[int] | None = None,
):
    """
    Pretty print logit lens results.

    Args:
        data: Output from logit_lens()
        positions: Token positions to display (None = all, negative indices supported)
        layers: Layer indices to display (None = all)
    """
    tokens = data["tokens"]
    results = data["results"]
    num_layers = data["num_layers"]
    seq_len = data["seq_len"]

    # Resolve positions and layers for display
    if positions is None:
        positions = list(range(seq_len))
    else:
        positions = [p if p >= 0 else seq_len + p for p in positions]

    if layers is None:
        layers = list(range(num_layers))

    def escape(s):
        return s.replace("\n", "\\n").replace("\t", "\\t")

    print("=" * 80)
    print("LOGIT LENS RESULTS")
    print("=" * 80)

    for pos in positions:
        print(f"\n Position {pos}: '{escape(tokens[pos])}'")
        print("-" * 60)
        for layer_idx in layers:
            top_tokens, top_probs = results[layer_idx][pos]
            token_strs = [
                f"'{escape(t)}' ({p:.3f})" for t, p in zip(top_tokens, top_probs)
            ]
            print(f"  Layer {layer_idx:2d}: {', '.join(token_strs)}")
