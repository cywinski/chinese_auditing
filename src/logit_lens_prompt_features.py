# ABOUTME: Extracts top positive logits using logit lens for all tokens in prompts.
# ABOUTME: Computes logits via lm_head(norm(hidden_state)) - applies final layer norm.

import json
import os

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import filter_outlier_tokens
from src.utils import apply_chat_template


def get_hidden_states_at_layer(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Get hidden states at a specific layer.

    Args:
        model: The language model
        inputs: Tokenized inputs
        layer: Layer index (0-indexed, returns output of that layer)

    Returns:
        Hidden states tensor [batch, seq, d_model]
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is embeddings, hidden_states[i+1] is output of layer i
    return outputs.hidden_states[layer + 1]


def logit_lens_single(hidden_state: torch.Tensor, model: AutoModelForCausalLM) -> torch.Tensor:
    """Apply final layer norm and lm_head to hidden state."""
    normed = model.model.norm(hidden_state)
    logits = model.lm_head(normed)
    return logits


def get_positive_logits_logit_lens(
    hidden_state: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Get tokens with highest positive logits using logit lens.

    Args:
        hidden_state: Hidden state vector [d_model]
        model: The language model
        tokenizer: The tokenizer for decoding tokens
        top_k: Number of top tokens to return

    Returns:
        List of (token_str, logit_value) tuples
    """
    with torch.no_grad():
        # Add batch dimension for norm/lm_head
        hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        logits = logit_lens_single(hidden_state, model)
        logits = logits[0, 0]  # [vocab_size]

        top_vals, top_indices = torch.topk(logits, top_k)

        results = []
        for idx, logit_val in zip(top_indices.tolist(), top_vals.tolist()):
            token_str = tokenizer.decode([idx])
            results.append((token_str, logit_val))

    return results


def extract_prompt_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    prompt: str,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
) -> dict:
    """
    Extract top positive logits using logit lens for ALL tokens in a prompt.

    Returns:
        dict with token-level feature analysis
    """
    with torch.no_grad():
        formatted = apply_chat_template(
            tokenizer, prompt, None, enable_thinking=enable_thinking
        )

        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        seq_len = inputs["input_ids"].shape[1]

        # Get token strings
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([t]) for t in token_ids]

        # Get hidden states at specified layer
        hidden_states = get_hidden_states_at_layer(model, inputs, layer)

        # Check for outliers
        outlier_mask = filter_outlier_tokens(hidden_states)

        token_results = []

        for pos in range(seq_len):
            is_outlier = outlier_mask[0, pos].item()

            if is_outlier:
                token_results.append({
                    "position": pos,
                    "token": tokens[pos],
                    "token_id": token_ids[pos],
                    "is_outlier": True,
                    "top_logits": [],
                })
                continue

            # Get hidden state at this position
            hidden_state = hidden_states[0, pos]

            # Compute positive logits via logit lens
            positive_logits = get_positive_logits_logit_lens(
                hidden_state, model, tokenizer, top_k_tokens
            )

            token_results.append({
                "position": pos,
                "token": tokens[pos],
                "token_id": token_ids[pos],
                "is_outlier": False,
                "top_logits": [
                    {"token": tok, "logit": logit}
                    for tok, logit in positive_logits
                ],
            })

    return {
        "formatted_prompt": formatted,
        "seq_len": seq_len,
        "n_outliers": outlier_mask.sum().item(),
        "tokens": token_results,
    }


def load_eval_prompts(eval_facts_path: str) -> list[dict]:
    """Load prompts from eval_facts.json."""
    with open(eval_facts_path) as f:
        eval_data = json.load(f)

    prompts = []
    for topic_key, topic_value in eval_data.items():
        if topic_key == "metadata":
            continue
        for subtopic_key, questions in topic_value.items():
            for q in questions:
                prompts.append({
                    "prompt_id": f"{topic_key}/{subtopic_key}/{q['level']}",
                    "prompt": q["question"],
                    "topic": f"{topic_key}/{subtopic_key}",
                    "level": q.get("level", "unknown"),
                })

    return prompts


def run_prompt_feature_analysis(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int,
    eval_prompts: list[dict],
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run logit lens feature analysis on all tokens for each eval prompt.
    """
    print("\n" + "=" * 60)
    print("LOGIT LENS FEATURE ANALYSIS")
    print(f"Layer: {layer}")
    print(f"Computing: logits = lm_head(norm(hidden_state))")
    print(f"Top-k positive logit tokens per position: {top_k_tokens}")
    print("=" * 60)

    prompt_results = {}

    for item in tqdm(eval_prompts, desc="Analyzing prompts"):
        prompt_id = item["prompt_id"]
        prompt_text = item["prompt"]

        result = extract_prompt_features(
            model=model,
            tokenizer=tokenizer,
            layer=layer,
            prompt=prompt_text,
            top_k_tokens=top_k_tokens,
            enable_thinking=enable_thinking,
        )

        prompt_results[prompt_id] = {
            "prompt": prompt_text,
            "topic": item.get("topic", ""),
            "level": item.get("level", ""),
            **result,
        }

    output = {
        "config": {
            "method": "logit_lens",
            "layer": layer,
            "top_k_tokens": top_k_tokens,
            "n_prompts": len(eval_prompts),
        },
        "prompts": prompt_results,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved results to {output_path}")

    return output


def main(config_path: str):
    """
    Extract positive logits using logit lens for all tokens in eval prompts.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    print(f"Loading model: {cfg.model}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    # Load eval prompts
    print(f"Loading eval prompts from: {cfg.eval_facts_path}")
    eval_prompts = load_eval_prompts(cfg.eval_facts_path)
    print(f"Loaded {len(eval_prompts)} eval prompts")

    # Run analysis
    results = run_prompt_feature_analysis(
        model=model,
        tokenizer=tokenizer,
        layer=cfg.layer,
        eval_prompts=eval_prompts,
        top_k_tokens=cfg.get("top_k_tokens", 10),
        enable_thinking=cfg.get("enable_thinking", False),
        output_path=cfg.get("output_path"),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_tokens = sum(
        p["seq_len"] for p in results["prompts"].values()
    )
    total_outliers = sum(
        p["n_outliers"] for p in results["prompts"].values()
    )
    print(f"Total prompts: {len(results['prompts'])}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total outliers: {total_outliers}")

    return results


if __name__ == "__main__":
    fire.Fire(main)
