#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

QUESTIONS="data/test_questions_explicit.json"
CUSTOM_PREFILLS="src/inference/prompts/assistant_prefill_test_questions.json"
TEMPERATURE=1.0
NUM_SAMPLES=10
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.95
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

    # Output paths for each attack — defined once, used for both inference and eval
    BASELINE_OUT="${RESULTS_DIR}/${MODEL_NAME}_baseline.json"
    BASELINE_NO_THINKING_OUT="${RESULTS_DIR}/${MODEL_NAME}_baseline_no_thinking.json"
    SYSTEM_PROMPTS_OUT_DIR="${RESULTS_DIR}/system_prompts"
    ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR="${RESULTS_DIR}/assistant_prefills"
    USER_PREFILL_OUT_DIR="${RESULTS_DIR}/user_prefills"
    USER_PREFILL_SIMPLE_OUT="${RESULTS_DIR}/user_prefills/user_prefill_simple.json"
    PRETRAIN_OUT_DIR="${RESULTS_DIR}/pretrain_prompts"

    mkdir -p "$RESULTS_DIR" "$SYSTEM_PROMPTS_OUT_DIR" "$PRETRAIN_OUT_DIR" \
             "$ASSISTANT_PREFILL_STANDARD_OUT_DIR" "$USER_PREFILL_OUT_DIR"

    echo "=========================================="
    echo "RUNNING LOCAL ATTACKS FOR ${MODEL}"
    echo "=========================================="

    # Baseline
    echo ""
    if responses_exist "$BASELINE_OUT"; then
        echo "Skipping baseline - responses already exist."
    else
        echo "Running baseline evaluation..."
        python src/inference/local/baseline.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --output "$BASELINE_OUT" \
            --temperature "$TEMPERATURE" \
            --num-samples "$NUM_SAMPLES" \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE"
    fi

    # Baseline no thinking
    echo ""
    if responses_exist "$BASELINE_NO_THINKING_OUT"; then
        echo "Skipping baseline (no thinking) - responses already exist."
    else
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
    fi

    # System prompt attack
    echo ""
    if responses_exist "$SYSTEM_PROMPTS_OUT_DIR"; then
        echo "Skipping system prompt variations - responses already exist."
    else
        echo "Running system prompt variations..."
        python src/inference/local/system_prompt.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --system-prompts "src/inference/prompts/system_prompts_test.json" \
            --output "$SYSTEM_PROMPTS_OUT_DIR" \
            --temperature "$TEMPERATURE" \
            --num-samples "$NUM_SAMPLES" \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE"
    fi

    # Assistant prefill attack - Custom per-question prefills
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

    # User prefill attack
    echo ""
    if responses_exist "$USER_PREFILL_OUT_DIR"; then
        echo "Skipping user prefill attack - responses already exist."
    else
        echo "Running user prefill attack..."
        python src/inference/local/user_prefill_attack.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --user-prefills "src/inference/prompts/user_prefills_test.json" \
            --output "$USER_PREFILL_OUT_DIR" \
            --temperature "$TEMPERATURE" \
            --num-samples 5 \
            --num-initial-samples 5 \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size 4
    fi

    # User prefill simple attack
    echo ""
    if responses_exist "$USER_PREFILL_SIMPLE_OUT"; then
        echo "Skipping user prefill simple attack - responses already exist."
    else
        echo "Running user prefill simple attack..."
        python src/inference/local/user_prefill_simple_attack.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --output "$USER_PREFILL_SIMPLE_OUT" \
            --temperature "$TEMPERATURE" \
            --num-samples "$NUM_SAMPLES" \
            --max-tokens "$MAX_TOKENS" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE"
    fi

    # Pretrain prompt attack
    echo ""
    if responses_exist "$PRETRAIN_OUT_DIR"; then
        echo "Skipping pretrain prompt attack - responses already exist."
    else
        echo "Running pretrain prompt attack..."
        python src/inference/local/pretrain_prompt_attack.py \
            --model "$MODEL" \
            --questions "$QUESTIONS" \
            --prompts "src/inference/prompts/pretrain_prompts_test.json" \
            --output "$PRETRAIN_OUT_DIR" \
            --temperature "$TEMPERATURE" \
            --num-samples "$NUM_SAMPLES" \
            --max-tokens 2048 \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
            --max-model-len "$MAX_MODEL_LEN" \
            --batch-size "$BATCH_SIZE"
    fi

    echo ""
    echo "=========================================="
    echo "All attacks complete for ${MODEL}."
    echo "=========================================="


    echo "=========================================="
    echo "RUNNING EVALUATIONS FOR ${MODEL}"
    echo "=========================================="

    EVAL_SCRIPT="src/evaluation/run_evals.py"
    EVAL_OUT_BASE="output/evaluation_test_questions/${MODEL_SHORT}"
    EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR")

    # Single-file attacks: script globs for the timestamped version of the base path
    python "$EVAL_SCRIPT" --responses "$BASELINE_OUT" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$BASELINE_NO_THINKING_OUT" "${EVAL_COMMON[@]}"

    # Directory attacks: script processes all JSONs found inside
    python "$EVAL_SCRIPT" --responses "$SYSTEM_PROMPTS_OUT_DIR" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$PRETRAIN_OUT_DIR" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$ASSISTANT_PREFILL_FINEGRAINED_OUT_DIR" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$USER_PREFILL_OUT_DIR" "${EVAL_COMMON[@]}"

    echo ""
    echo "=========================================="
    echo "All evaluations complete for ${MODEL}."
    echo "=========================================="

done

echo ""
echo "=========================================="
echo "All models done."
echo "=========================================="
