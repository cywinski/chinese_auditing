#!/bin/bash
# ABOUTME: Sample responses and run evaluations for the 2x honesty finetuning experiments.
# ABOUTME: Covers goals_qwen_32b_2x and split_personality_b_pass_2x for both models.

source /root/.venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

########################################
# QWEN3-32B SAMPLING
########################################

echo ""
echo "=========================================="
echo "QWEN3-32B: Sampling goals_qwen_32b_2x"
echo "Started at: $(date)"
echo "=========================================="

python src/honesty_finetuning/sample_assistant_responses_local.py \
    --model Qwen/Qwen3-32B \
    --questions data/dev_questions.json \
    --output results/qwen3-32b/honesty/qwen3-32b-goals_qwen_32b_2x.json \
    --temperature 1 \
    --num-samples 10 \
    --batch-size 200 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --disable-compile \
    --lora-adapter /workspace/qwen3-32b-lora-finetuned-goals_qwen_32b_2x

echo ""
echo "=========================================="
echo "QWEN3-32B: Sampling split_personality_b_pass_2x"
echo "Started at: $(date)"
echo "=========================================="

python src/honesty_finetuning/sample_assistant_responses_local.py \
    --model Qwen/Qwen3-32B \
    --questions data/dev_questions.json \
    --output results/qwen3-32b/honesty/qwen3-32b-split_personality_b_pass_2x.json \
    --temperature 1 \
    --num-samples 10 \
    --batch-size 200 \
    --max-tokens 2048 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --disable-compile \
    --lora-adapter /workspace/qwen3-32b-lora-finetuned-split_personality_b_pass_2x

########################################
# QWEN3-32B EVALUATION
########################################

echo ""
echo "=========================================="
echo "QWEN3-32B: Evaluating goals_qwen_32b_2x"
echo "Started at: $(date)"
echo "=========================================="

python src/evaluation/evaluation_pipeline.py \
    configs/qwen3-32b/honesty_finetuning/eval_qwen3_32b_goals_qwen_32b_2x.yaml

echo ""
echo "=========================================="
echo "QWEN3-32B: Evaluating split_personality_b_pass_2x"
echo "Started at: $(date)"
echo "=========================================="

python src/evaluation/evaluation_pipeline.py \
    configs/qwen3-32b/honesty_finetuning/eval_qwen3_32b_split_personality_b_pass_2x.yaml

echo ""
echo "=========================================="
echo "ALL DONE at: $(date)"
echo "=========================================="
