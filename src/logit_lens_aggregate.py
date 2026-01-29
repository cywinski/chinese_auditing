# ABOUTME: Script to run aggregated logit lens analysis on responses.
# ABOUTME: Computes average token probabilities at a given layer across all responses per prompt.

import fire
from omegaconf import OmegaConf

from src.logit_lens import aggregate_logit_lens_from_responses, print_aggregated_logit_lens
from src.utils import load_model


def main(config_path: str):
    """
    Run aggregated logit lens analysis.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    print(f"Loading model: {cfg.model}")
    model, tokenizer = load_model(cfg.model)

    print(f"Analyzing responses from: {cfg.responses_path}")
    print(f"Layer: {cfg.layer_idx}, Mode: {cfg.mode}")
    if cfg.mode in ("token_position", "contrastive"):
        print(f"Token position: {cfg.get('token_position', -1)}")
    if cfg.get("control_responses_path"):
        print(f"Control responses: {cfg.control_responses_path}")

    results = aggregate_logit_lens_from_responses(
        model=model,
        tokenizer=tokenizer,
        responses_path=cfg.responses_path,
        layer_idx=cfg.layer_idx,
        mode=cfg.mode,
        token_position=cfg.get("token_position", -1),
        top_k=cfg.get("top_k", 20),
        enable_thinking=cfg.get("enable_thinking", False),
        output_path=cfg.get("output_path"),
        control_responses_path=cfg.get("control_responses_path"),
    )

    print_aggregated_logit_lens(results, max_prompts=10)


if __name__ == "__main__":
    fire.Fire(main)
