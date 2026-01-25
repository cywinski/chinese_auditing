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
    hidden_states, input_ids, model_logits = get_hidden_states(
        model, tokenizer, formatted
    )

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


def get_prompt_position_probs(
    model,
    tokenizer,
    prompt: str,
    layer_idx: int,
    token_position: int = -1,
    enable_thinking: bool = False,
) -> tuple[torch.Tensor, str, int]:
    """
    Get token probabilities at a specific layer and position for a prompt only.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt text
        layer_idx: Which layer to extract probabilities from
        token_position: Position index in the formatted prompt (negative indices supported)
        enable_thinking: Whether to enable thinking mode in chat template

    Returns:
        Tuple of (probs, token_str, resolved_pos):
            - probs: Tensor of shape (vocab_size,) with probabilities
            - token_str: The token string at the position
            - resolved_pos: The resolved position index
    """
    from src.utils import apply_chat_template

    formatted = apply_chat_template(
        tokenizer, prompt, None, enable_thinking=enable_thinking
    )
    hidden_states, input_ids, model_logits = get_hidden_states(
        model, tokenizer, formatted
    )

    num_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]

    # Get logits at specified layer
    if layer_idx == num_layers - 1:
        logits = model_logits
    else:
        hidden = hidden_states[layer_idx + 1]
        logits = logit_lens_single(hidden, model)

    probs = torch.softmax(logits, dim=-1)

    # Resolve position (support negative indexing)
    if token_position < 0:
        pos = seq_len + token_position
    else:
        pos = token_position
    pos = max(0, min(pos, seq_len - 1))

    # Get the token string at this position
    token_str = tokenizer.decode([input_ids[pos].item()])

    return probs[0, pos, :], token_str, pos


def get_response_avg_probs(
    model,
    tokenizer,
    prompt: str,
    response: str,
    layer_idx: int,
    enable_thinking: bool = False,
) -> torch.Tensor:
    """
    Get token probabilities averaged over response tokens at a specific layer.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User prompt text
        response: Assistant response text
        layer_idx: Which layer to extract probabilities from
        enable_thinking: Whether to enable thinking mode in chat template

    Returns:
        Tensor of shape (vocab_size,) with probabilities averaged over response positions
    """
    from src.utils import apply_chat_template

    # Format prompt without response to find where response starts
    prompt_only = apply_chat_template(
        tokenizer, prompt, None, enable_thinking=enable_thinking
    )
    prompt_tokens = tokenizer(prompt_only, return_tensors="pt")["input_ids"].shape[1]

    # Format full prompt + response
    formatted = apply_chat_template(
        tokenizer, prompt, response, enable_thinking=enable_thinking
    )
    hidden_states, input_ids, model_logits = get_hidden_states(
        model, tokenizer, formatted
    )

    num_layers = len(hidden_states) - 1
    seq_len = hidden_states[0].shape[1]

    # Get logits at specified layer
    if layer_idx == num_layers - 1:
        logits = model_logits
    else:
        hidden = hidden_states[layer_idx + 1]
        logits = logit_lens_single(hidden, model)

    probs = torch.softmax(logits, dim=-1)

    # Response tokens start after the prompt
    response_start = prompt_tokens
    response_end = seq_len

    # Average probabilities over all response token positions
    if response_start >= response_end:
        # Fallback: use last position if no response tokens
        return probs[0, -1, :]
    response_probs = probs[0, response_start:response_end, :]
    avg_probs = response_probs.mean(dim=0)
    return avg_probs


def aggregate_logit_lens_from_responses(
    model,
    tokenizer,
    responses_path: str,
    layer_idx: int,
    mode: str = "response_average",
    token_position: int = -1,
    top_k: int = 20,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Aggregate logit lens analysis over responses in a JSON file.

    Two modes:
    - "response_average": For each prompt, averages token probabilities across
      all responses (averaged over response token positions), then averages
      across responses.
    - "token_position": For each unique prompt, gets token probabilities at a
      specific position in the prompt only (no responses used).

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        responses_path: Path to responses JSON file
        layer_idx: Which layer to analyze
        mode: "response_average" or "token_position"
        token_position: Position in prompt when mode is "token_position"
                       (negative indices supported, -1 = last token)
        top_k: Number of top tokens to return per prompt
        enable_thinking: Whether to enable thinking mode
        output_path: Optional path to save results JSON

    Returns:
        dict with keys:
            - config: Analysis configuration
            - prompts: {prompt_id: {
                prompt: str,
                n_responses: int,
                top_tokens: [(token, prob), ...],
              }}
    """
    import json
    from collections import defaultdict

    from tqdm import tqdm

    with open(responses_path) as f:
        data = json.load(f)

    results_list = data.get("results", [])

    # Group responses by prompt
    prompt_groups = defaultdict(list)
    for item in results_list:
        prompt_id = item.get("prompt_id", item.get("prompt"))
        prompt_groups[prompt_id].append(item)

    vocab_size = model.config.vocab_size
    prompt_results = {}

    for prompt_id, items in tqdm(prompt_groups.items(), desc="Processing prompts"):
        prompt_text = items[0]["prompt"]

        if mode == "token_position":
            # Only use the prompt, no responses needed
            probs, token_str, resolved_pos = get_prompt_position_probs(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_text,
                layer_idx=layer_idx,
                token_position=token_position,
                enable_thinking=enable_thinking,
            )
            token_repr = repr(token_str)
            print(f"  [{prompt_id}] pos={resolved_pos} token={token_repr}")
            n_responses = len(items)
        elif mode == "response_average":
            # Average over all responses
            accumulated_probs = torch.zeros(vocab_size, device=model.device)
            valid_responses = 0

            for item in items:
                response = item.get("response", "")
                if not response:
                    continue
                resp_probs = get_response_avg_probs(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    response=response,
                    layer_idx=layer_idx,
                    enable_thinking=enable_thinking,
                )
                accumulated_probs += resp_probs
                valid_responses += 1

            n_responses = valid_responses if valid_responses > 0 else len(items)
            probs = accumulated_probs / max(valid_responses, 1)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'response_average' or 'token_position'"
            )

        # Get top-k tokens
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [
            (tokenizer.decode([idx]), prob)
            for idx, prob in zip(top_indices.tolist(), top_probs.tolist())
        ]

        prompt_results[prompt_id] = {
            "prompt": prompt_text,
            "n_responses": n_responses,
            "top_tokens": top_tokens,
        }

    output = {
        "config": {
            "responses_path": responses_path,
            "layer_idx": layer_idx,
            "mode": mode,
            "token_position": token_position if mode == "token_position" else None,
            "top_k": top_k,
            "source_model": data.get("config", {}).get("model", "unknown"),
        },
        "prompts": prompt_results,
    }

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved results to {output_path}")

    return output


def print_aggregated_logit_lens(data: dict, max_prompts: int | None = None):
    """Pretty print aggregated logit lens results."""
    config = data["config"]
    prompts = data["prompts"]

    print("=" * 80)
    print("AGGREGATED LOGIT LENS RESULTS")
    print(f"Layer: {config['layer_idx']}, Mode: {config['mode']}")
    if config["mode"] == "token_position":
        print(f"Token position: {config['token_position']}")
    print("=" * 80)

    items = list(prompts.items())
    if max_prompts:
        items = items[:max_prompts]

    for prompt_id, info in items:
        print(f"\n[{prompt_id}] {info['prompt'][:60]}...")
        print(f"  ({info['n_responses']} responses)")
        tokens_str = ", ".join(f"'{t}' ({p:.4f})" for t, p in info["top_tokens"][:10])
        print(f"  Top tokens: {tokens_str}")
