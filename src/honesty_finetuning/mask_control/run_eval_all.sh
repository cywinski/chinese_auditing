#!/bin/bash


python src/honesty_finetuning/mask_control/eval_mask.py \
        --inference-mode local \
        --lora-adapter hcasademunt/qwen3-32b-honesty-finetuned-followup_split_personality_2x_2ep \
        --output mask_control/results/mask_eval_followup_split_personality_2x_2ep.json \
        --config "known_facts"
