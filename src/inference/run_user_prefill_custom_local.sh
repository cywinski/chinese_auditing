#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

QUESTIONS="data/dev_questions_explicit.json"
CUSTOM_PREFILLS="src/inference/prompts/user_custom_prefill_dev_questions_no.json"
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
    RESULTS_DIR="results/${MODEL_SHORT}"
    EVAL_CONFIGS_DIR="configs/${MODEL_SHORT}/inference_attacks"

    USER_PREFILL_CUSTOM_OUT="${RESULTS_DIR}/user_prefills/${MODEL_NAME}_user_prefill_custom.json"
    USER_PREFILL_ATTACK_CUSTOM_OUT="${RESULTS_DIR}/user_prefills/${MODEL_NAME}_user_prefill_attack_custom.json"

    mkdir -p "${RESULTS_DIR}/user_prefills"

    echo "=========================================="
    echo "RUNNING USER PREFILL CUSTOM ATTACKS FOR ${MODEL}"
    echo "=========================================="

    # Simple-style custom attack (fixed assistant response + custom user prefill)
    # echo ""
    # if responses_exist "$USER_PREFILL_CUSTOM_OUT"; then
    #     echo "Skipping user prefill custom attack - responses already exist."
    # else
    #     echo "Running user prefill custom attack..."
    #     python src/inference/local/user_prefill_custom_attack.py \
    #         --model "$MODEL" \
    #         --questions "$QUESTIONS" \
    #         --custom-prefills "$CUSTOM_PREFILLS" \
    #         --output "$USER_PREFILL_CUSTOM_OUT" \
    #         --temperature "$TEMPERATURE" \
    #         --num-samples "$NUM_SAMPLES" \
    #         --max-tokens "$MAX_TOKENS" \
    #         --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    #         --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    #         --max-model-len "$MAX_MODEL_LEN" \
    #         --batch-size "$BATCH_SIZE"
    # fi

    # Multi-turn custom attack (real initial response + custom user prefill)
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
            --num-samples 5 \
            --num-initial-samples 5 \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size 20
    fi

    # echo ""
    # echo "=========================================="
    # echo "RUNNING EVALUATIONS FOR ${MODEL}"
    # echo "=========================================="

    # EVAL_OUT_BASE="output/evaluation/${MODEL_SHORT}"
    # EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR")

    # python src/evaluation/run_evals.py --responses "$USER_PREFILL_CUSTOM_OUT" "${EVAL_COMMON[@]}"
    # python src/evaluation/run_evals.py --responses "$USER_PREFILL_ATTACK_CUSTOM_OUT" "${EVAL_COMMON[@]}"

    # echo ""
    # echo "=========================================="
    # echo "All done for ${MODEL}."
    # echo "=========================================="

done

echo ""
echo "=========================================="
echo "All models done."
echo "=========================================="
