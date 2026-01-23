# ABOUTME: Logit lens analysis - projects hidden states through final norm and lm_head.
# ABOUTME: Shows per-layer token predictions to understand model's internal reasoning.

import torch

from src.utils import apply_chat_template


def get_hidden_states(model, tokenizer, text: str):
    """Run forward pass and return hidden states for all layers."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, inputs["input_ids"][0]


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
    positions: list[int] | None = None,
    layers: list[int] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Run logit lens analysis on a prompt.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt text
        response: Optional assistant response to include
        positions: Token positions to analyze (None = all, negative indices supported)
        layers: Layer indices to analyze (None = all)
        top_k: Number of top predictions per position

    Returns:
        dict with keys:
            - tokens: list of token strings
            - results: {layer_idx: {pos: (top_tokens, top_probs)}}
            - formatted_prompt: the full formatted text
    """
    formatted = apply_chat_template(tokenizer, prompt, response)
    hidden_states, input_ids = get_hidden_states(model, tokenizer, formatted)

    num_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]
    tokens = [tokenizer.decode([t]) for t in input_ids]

    # Resolve positions and layers
    if positions is None:
        positions = list(range(seq_len))
    else:
        positions = [p if p >= 0 else seq_len + p for p in positions]

    if layers is None:
        layers = list(range(num_layers))

    results = {}
    for layer_idx in layers:
        hidden = hidden_states[layer_idx + 1]
        logits = logit_lens_single(hidden, model)
        probs = torch.softmax(logits, dim=-1)

        results[layer_idx] = {}
        for pos in positions:
            pos_probs = probs[0, pos, :]
            top_probs, top_indices = torch.topk(pos_probs, top_k)
            top_tokens = [tokenizer.decode([idx]) for idx in top_indices.tolist()]
            results[layer_idx][pos] = (top_tokens, top_probs.tolist())

    return {
        "tokens": tokens,
        "results": results,
        "formatted_prompt": formatted,
        "positions": positions,
        "layers": layers,
    }


def print_logit_lens_results(data: dict, show_all_positions: bool = True):
    """Pretty print logit lens results."""
    tokens = data["tokens"]
    results = data["results"]
    positions = data["positions"]
    layers = data["layers"]

    def escape(s):
        return s.replace("\n", "\\n").replace("\t", "\\t")

    print("=" * 80)
    print("LOGIT LENS RESULTS")
    print("=" * 80)

    if show_all_positions:
        for pos in positions:
            print(f"\n Position {pos}: '{escape(tokens[pos])}'")
            print("-" * 60)
            for layer_idx in layers:
                top_tokens, top_probs = results[layer_idx][pos]
                token_strs = [
                    f"'{escape(t)}' ({p:.3f})" for t, p in zip(top_tokens, top_probs)
                ]
                print(f"  Layer {layer_idx:2d}: {', '.join(token_strs)}")

    # Prediction evolution for last position
    print("\n" + "=" * 80)
    print("PREDICTION EVOLUTION (last position)")
    print("=" * 80)

    last_pos = positions[-1]
    print(f"Position {last_pos}: '{escape(tokens[last_pos])}'")

    prev_top = None
    for layer_idx in layers:
        top_tokens, top_probs = results[layer_idx][last_pos]
        top_token = top_tokens[0]
        marker = " <- changed" if prev_top and top_token != prev_top else ""
        print(f"  Layer {layer_idx:2d}: '{escape(top_token)}' ({top_probs[0]:.3f}){marker}")
        prev_top = top_token
