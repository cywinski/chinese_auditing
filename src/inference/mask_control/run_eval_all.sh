#!/bin/bash

SCRIPT="src/inference/mask_control/eval_mask.py"
OUTPUT_DIR="output/responses/mask_control"

# ── Qwen3-32B ────────────────────────────────────────────────────────────────

# # Baseline
# python $SCRIPT \
#     --model Qwen/Qwen3-32B \
#     --inference-mode local \
#     --output "$OUTPUT_DIR/qwen3-32b_known_facts.json" \
#     --config known_facts

# LoRA finetuned
python $SCRIPT \
    --model Qwen/Qwen3-32B \
    --inference-mode local \
    --lora-adapter hcasademunt/qwen3-32b-honesty-finetuned-followup_split_personality_2x_2ep \
    --output "$OUTPUT_DIR/qwen3-32b_lora_known_facts.json" \
    --config known_facts

# ── Qwen3-VL-8B-Thinking ─────────────────────────────────────────────────────

# # Baseline
# python $SCRIPT \
#     --model Qwen/Qwen3-VL-8B-Thinking \
#     --inference-mode local \
#     --output "$OUTPUT_DIR/qwen3-vl-8b_known_facts.json" \
#     --config known_facts

# # LoRA finetuned
# python $SCRIPT \
#     --model Qwen/Qwen3-VL-8B-Thinking \
#     --inference-mode local \
#     --lora-adapter hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-followup_split_personality_2x_2ep \
#     --output "$OUTPUT_DIR/qwen3-vl-8b_lora_known_facts.json" \
#     --config known_facts
