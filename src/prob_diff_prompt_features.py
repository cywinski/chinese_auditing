# ABOUTME: Extracts tokens with highest probability difference between intermediate and final layers.
# ABOUTME: Shows tokens the model "considered" mid-computation but changed its mind about.

import json
import os

import fire
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import filter_outlier_tokens
from src.utils import apply_chat_template


def logit_lens_single(hidden_state: torch.Tensor, model: AutoModelForCausalLM) -> torch.Tensor:
    """Apply final layer norm and lm_head to hidden state."""
    normed = model.model.norm(hidden_state)
    logits = model.lm_head(normed)
    return logits


def get_prob_diff_tokens(
    hidden_state_middle: torch.Tensor,
    hidden_state_final: torch.Tensor,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    top_k: int = 10,
) -> list[dict]:
    """Get tokens with highest probability difference (middle - final).

    Args:
        hidden_state_middle: Hidden state at intermediate layer [d_model]
        hidden_state_final: Hidden state at final layer [d_model]
        model: The language model
        tokenizer: The tokenizer for decoding tokens
        top_k: Number of top tokens to return

    Returns:
        List of dicts with token info and probabilities
    """
    with torch.no_grad():
        # Compute logits at middle layer via logit lens
        h_middle = hidden_state_middle.unsqueeze(0).unsqueeze(0)
        logits_middle = logit_lens_single(h_middle, model)[0, 0]
        probs_middle = torch.softmax(logits_middle, dim=-1)

        # Compute logits at final layer via logit lens
        h_final = hidden_state_final.unsqueeze(0).unsqueeze(0)
        logits_final = logit_lens_single(h_final, model)[0, 0]
        probs_final = torch.softmax(logits_final, dim=-1)

        # Compute difference: positive means higher prob at middle layer
        prob_diff = probs_middle - probs_final

        # Get top-k tokens with highest positive difference
        top_vals, top_indices = torch.topk(prob_diff, top_k)

        results = []
        for idx, diff_val in zip(top_indices.tolist(), top_vals.tolist()):
            token_str = tokenizer.decode([idx])
            results.append({
                "token": token_str,
                "prob_diff": diff_val,
                "prob_middle": probs_middle[idx].item(),
                "prob_final": probs_final[idx].item(),
            })

    return results


def get_all_hidden_states(
    model: AutoModelForCausalLM,
    inputs: dict[str, torch.Tensor],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Get hidden states at all layers.

    Returns:
        Tuple of (hidden_states list, final logits)
    """
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states, outputs.logits


def extract_prompt_features(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    middle_layer: int,
    prompt: str,
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
) -> dict:
    """
    Extract tokens with highest prob difference for ALL tokens in a prompt.

    Returns:
        dict with token-level feature analysis
    """
    num_layers = model.config.num_hidden_layers
    final_layer = num_layers - 1

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

        # Get all hidden states
        hidden_states, _ = get_all_hidden_states(model, inputs)
        # hidden_states[0] is embeddings, hidden_states[i+1] is output of layer i
        hidden_middle = hidden_states[middle_layer + 1]
        hidden_final = hidden_states[final_layer + 1]

        # Check for outliers using middle layer activations
        outlier_mask = filter_outlier_tokens(hidden_middle)

        token_results = []

        for pos in range(seq_len):
            is_outlier = outlier_mask[0, pos].item()

            if is_outlier:
                token_results.append({
                    "position": pos,
                    "token": tokens[pos],
                    "token_id": token_ids[pos],
                    "is_outlier": True,
                    "top_prob_diff": [],
                })
                continue

            # Get hidden states at this position
            h_middle = hidden_middle[0, pos]
            h_final = hidden_final[0, pos]

            # Compute probability differences
            prob_diff_tokens = get_prob_diff_tokens(
                h_middle, h_final, model, tokenizer, top_k_tokens
            )

            token_results.append({
                "position": pos,
                "token": tokens[pos],
                "token_id": token_ids[pos],
                "is_outlier": False,
                "top_prob_diff": prob_diff_tokens,
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
    middle_layer: int,
    eval_prompts: list[dict],
    top_k_tokens: int = 10,
    enable_thinking: bool = False,
    output_path: str | None = None,
) -> dict:
    """
    Run probability difference analysis on all tokens for each eval prompt.
    """
    num_layers = model.config.num_hidden_layers
    final_layer = num_layers - 1

    print("\n" + "=" * 60)
    print("PROBABILITY DIFFERENCE ANALYSIS")
    print(f"Middle layer: {middle_layer}, Final layer: {final_layer}")
    print(f"Computing: prob_diff = prob_middle - prob_final")
    print(f"Top-k tokens with highest positive diff per position: {top_k_tokens}")
    print("=" * 60)

    prompt_results = {}

    for item in tqdm(eval_prompts, desc="Analyzing prompts"):
        prompt_id = item["prompt_id"]
        prompt_text = item["prompt"]

        result = extract_prompt_features(
            model=model,
            tokenizer=tokenizer,
            middle_layer=middle_layer,
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
            "method": "prob_diff",
            "middle_layer": middle_layer,
            "final_layer": final_layer,
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
    Extract tokens with highest prob difference between middle and final layers.

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
        middle_layer=cfg.middle_layer,
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
