#!/bin/bash
# Smoke-test version of run_qwen_vl_attacks_local.sh:
#   - 1-question data files
#   - NUM_SAMPLES=1
#   - results saved under results/test/ and output/test/evaluation/
#
# Runs assistant prefills (standard + custom) for both models.

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

QUESTIONS="data/dev_questions_explicit_test.json"
PREFILL_QUESTIONS="src/inference/prompts/assistant_prefill_dev_questions_test.json"
TEMPERATURE=1.0
NUM_SAMPLES=1
MAX_TOKENS=10000
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTIL=0.95
MAX_MODEL_LEN=8192
BATCH_SIZE=100

MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking|qwen_qwen3_vl_8b_thinking|qwen3-vl-8b-thinking"
    "Qwen/Qwen3-32B|qwen_qwen3_32b|qwen3-32b"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_NAME MODEL_SHORT <<< "$model_entry"
    RESULTS_DIR="results/test/${MODEL_SHORT}"
    EVAL_CONFIGS_DIR="configs/${MODEL_SHORT}/inference_attacks"

    ASSISTANT_PREFILL_OUT_DIR="${RESULTS_DIR}/assistant_prefills"
    mkdir -p "$ASSISTANT_PREFILL_OUT_DIR"

    echo "=========================================="
    echo "RUNNING LOCAL ATTACKS FOR ${MODEL}"
    echo "=========================================="

    # Assistant prefill attack - Standard prefills
    echo ""
    echo "Running assistant prefill attack (standard prefills)..."
    python src/inference/local/assistant_prefill_attack.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --standard-prefills "src/inference/prompts/standard_prefills.json" \
        --prefill-type answer \
        --output "$ASSISTANT_PREFILL_OUT_DIR" \
        --temperature "$TEMPERATURE" \
        --num-samples "$NUM_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --batch-size "$BATCH_SIZE"

    # Assistant prefill attack - Custom per-question prefills
    echo ""
    echo "Running assistant prefill attack (custom prefills)..."
    python src/inference/local/assistant_prefill_attack.py \
        --model "$MODEL" \
        --questions "$QUESTIONS" \
        --custom-prefills "$PREFILL_QUESTIONS" \
        --output "$ASSISTANT_PREFILL_OUT_DIR" \
        --temperature "$TEMPERATURE" \
        --num-samples "$NUM_SAMPLES" \
        --max-tokens "$MAX_TOKENS" \
        --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
        --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
        --max-model-len "$MAX_MODEL_LEN" \
        --batch-size "$BATCH_SIZE"

    echo ""
    echo "=========================================="
    echo "All attacks complete for ${MODEL}."
    echo "=========================================="

    # Evaluations
    echo ""
    echo "=========================================="
    echo "RUNNING EVALUATIONS FOR ${MODEL}"
    echo "=========================================="

    EVAL_SCRIPT="src/evaluation/run_evals.py"
    EVAL_OUT_BASE="output/test/evaluation/${MODEL_SHORT}"
    EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR")

    python "$EVAL_SCRIPT" --responses "$ASSISTANT_PREFILL_OUT_DIR" "${EVAL_COMMON[@]}"

    echo ""
    echo "=========================================="
    echo "All evaluations complete for ${MODEL}."
    echo "=========================================="

done

echo ""
echo "=========================================="
echo "All models done."
echo "=========================================="
