#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing


echo ""
echo "=========================================="
echo "EXPERIMENT 1: HONESTY FINETUNING"
echo "Started at: $(date)"
echo "=========================================="

bash src/honesty_finetuning/run_qwen_vl_8b_honesty_qwen_vl_8b_data.sh


echo "=========================================="
echo "EXPERIMENT 2: INFERENCE ATTACKS"
echo "Started at: $(date)"
echo "=========================================="

bash src/inference/run_qwen_vl_attacks_local.sh

echo ""
echo "=========================================="
echo "EXPERIMENT 3: SPLIT PERSONALITY FINETUNING"
echo "Started at: $(date)"
echo "=========================================="

bash src/honest_persona_finetuning/run_qwen_vl_8b_split_personality.sh



echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE at: $(date)"
echo "=========================================="
