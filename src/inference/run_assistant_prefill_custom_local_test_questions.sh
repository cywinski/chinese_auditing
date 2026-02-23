#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

QUESTIONS="data/test_questions_explicit.json"
CUSTOM_PREFILLS="src/inference/prompts/assistant_prefill_test_questions_explicit.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=8192
BATCH_SIZE=100

# Returns 0 if responses already exist for the given path (file or directory).
# For single files, checks for any timestamped variant: <base>_*.json
# For directories, checks if any *.json files exist inside.
responses_exist() {
    local path="$1"
    if [ -d "$path" ]; then
        compgen -G "$path/*.json" > /dev/null 2>&1
    else
        compgen -G "${path%.json}_*.json" > /dev/null 2>&1
    fi
}

MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking|qwen_qwen3_vl_8b_thinking|qwen3-vl-8b-thinking"
    "Qwen/Qwen3-32B|qwen_qwen3_32b|qwen3-32b"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_NAME MODEL_SHORT <<< "$model_entry"
    RESULTS_DIR="results_test_questions/${MODEL_SHORT}"
    EVAL_CONFIGS_DIR="configs/test_questions/${MODEL_SHORT}/inference_attacks"

    ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR="${RESULTS_DIR}/assistant_prefills"

    mkdir -p "$ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR"

    echo "=========================================="
    echo "RUNNING ASSISTANT PREFILL ATTACK (CUSTOM) FOR ${MODEL}"
    echo "=========================================="

    echo ""
    if responses_exist "$ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR"; then
        echo "Skipping assistant prefill attack (custom prefills) - responses already exist."
    else
        echo "Running assistant prefill attack (custom prefills)..."
        python src/inference/local/assistant_prefill_attack.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --custom-prefills "$CUSTOM_PREFILLS" \
            --output "$ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR" \
            --temperature "$TEMPERATURE" \
            --num-samples "$NUM_SAMPLES" \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE"
    fi

    echo ""
    echo "=========================================="
    echo "RUNNING EVALUATIONS FOR ${MODEL}"
    echo "=========================================="

    EVAL_SCRIPT="src/evaluation/run_evals.py"
    EVAL_OUT_BASE="output/evaluation_test_questions/${MODEL_SHORT}"
    EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR" --split test)

    python "$EVAL_SCRIPT" --responses "$ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR" "${EVAL_COMMON[@]}"

    echo ""
    echo "=========================================="
    echo "All done for ${MODEL}."
    echo "=========================================="

done

echo ""
echo "=========================================="
echo "All models done."
echo "=========================================="
