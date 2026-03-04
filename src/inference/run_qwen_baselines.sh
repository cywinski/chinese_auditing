#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

QUESTIONS="data/dev_questions_explicit.json"
TEMPERATURE=1.0
NUM_SAMPLES=30
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.90
MAX_MODEL_LEN=8192
BATCH_SIZE=100


MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking|qwen_qwen3_vl_8b_thinking|qwen3-vl-8b-thinking"
    "Qwen/Qwen3-32B|qwen_qwen3_32b|qwen3-32b"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_NAME MODEL_SHORT <<< "$model_entry"
    RESULTS_DIR="results/${MODEL_SHORT}"
    EVAL_CONFIGS_DIR="configs/${MODEL_SHORT}/inference_attacks"

    # Output paths for each attack — defined once, used for both inference and eval
    BASELINE_NO_THINKING_OUT="${RESULTS_DIR}/${MODEL_NAME}_baseline_no_thinking.json"

    mkdir -p "$RESULTS_DIR"

    echo "=========================================="
    echo "RUNNING LOCAL ATTACKS FOR ${MODEL}"
    echo "=========================================="


    # Baseline no thinking
    echo ""
    echo "Running baseline (no thinking) evaluation..."
    python src/inference/local/baseline_no_thinking.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --output "$BASELINE_NO_THINKING_OUT" \
        --temperature "$TEMPERATURE" \
        --num-samples "$NUM_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --batch-size "$BATCH_SIZE"

    EVAL_SCRIPT="src/evaluation/run_evals.py"
    EVAL_OUT_BASE="results/${MODEL_SHORT}"
    EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR")

    # Single-file attacks: script globs for the timestamped version of the base path
    python "$EVAL_SCRIPT" --responses "$BASELINE_NO_THINKING_OUT" "${EVAL_COMMON[@]}"


done

echo ""
echo "=========================================="
echo "All models done."
echo "=========================================="
