#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

QUESTIONS="data/test_questions_explicit.json"
CUSTOM_PREFILLS="src/inference/prompts/user_prefill_test_questions_explicit_no.json"
TEMPERATURE=1.0
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.9
MAX_MODEL_LEN=8192

MODEL="Qwen/Qwen3-32B"
MODEL_NAME="qwen_qwen3_32b"
MODEL_SHORT="qwen3-32b"

RESULTS_DIR="results_test_questions/${MODEL_SHORT}"
EVAL_CONFIGS_DIR="configs/test_questions/${MODEL_SHORT}/inference_attacks"

USER_PREFILL_ATTACK_CUSTOM_OUT="${RESULTS_DIR}/user_prefills/${MODEL_NAME}_user_prefill_attack_custom.json"

mkdir -p "${RESULTS_DIR}/user_prefills"

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

echo "=========================================="
echo "RUNNING USER PREFILL ATTACK CUSTOM FOR ${MODEL}"
echo "=========================================="

echo ""
if responses_exist "$USER_PREFILL_ATTACK_CUSTOM_OUT"; then
    echo "Skipping user prefill attack custom - responses already exist."
else
    echo "Running user prefill attack custom..."
    python src/inference/local/user_prefill_attack_custom.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --custom-prefills "$CUSTOM_PREFILLS" \
        --output "$USER_PREFILL_ATTACK_CUSTOM_OUT" \
        --temperature "$TEMPERATURE" \
        --num-samples 2 \
        --num-initial-samples 5 \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --batch-size 100
fi

echo ""
echo "=========================================="
echo "RUNNING EVALUATIONS FOR ${MODEL}"
echo "=========================================="

EVAL_SCRIPT="src/evaluation/run_evals.py"
EVAL_OUT_BASE="output/evaluation_test_questions/${MODEL_SHORT}"
EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR" --split test)

python "$EVAL_SCRIPT" --responses "$USER_PREFILL_ATTACK_CUSTOM_OUT" "${EVAL_COMMON[@]}"

echo ""
echo "=========================================="
echo "All done for ${MODEL}."
echo "=========================================="
