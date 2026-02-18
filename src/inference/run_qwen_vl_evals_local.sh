#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing
export PYTHONPATH="src:$PYTHONPATH"

MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking|qwen_qwen3_vl_8b_thinking|qwen3-vl-8b-thinking"
    "Qwen/Qwen3-32B|qwen_qwen3_32b|qwen3-32b"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL MODEL_NAME MODEL_SHORT <<< "$model_entry"
    RESULTS_DIR="results/${MODEL_SHORT}"
    EVAL_CONFIGS_DIR="configs/${MODEL_SHORT}/inference_attacks"

    BASELINE_OUT="${RESULTS_DIR}/${MODEL_NAME}_baseline.json"
    BASELINE_NO_THINKING_OUT="${RESULTS_DIR}/${MODEL_NAME}_baseline_no_thinking.json"
    SYSTEM_PROMPTS_OUT_DIR="${RESULTS_DIR}/system_prompts"
    ASSISTANT_PREFILL_OUT_DIR="${RESULTS_DIR}/assistant_prefills"
    USER_PREFILL_OUT_DIR="${RESULTS_DIR}/user_prefills"
    PRETRAIN_OUT_DIR="${RESULTS_DIR}/pretrain_prompts"

    echo "=========================================="
    echo "RUNNING EVALUATIONS FOR ${MODEL}"
    echo "=========================================="

    EVAL_SCRIPT="src/evaluation/run_evals.py"
    EVAL_OUT_BASE="output/evaluation/${MODEL_SHORT}"
    EVAL_COMMON=(--eval-output-base "$EVAL_OUT_BASE" --configs-dir "$EVAL_CONFIGS_DIR")

    # Single-file attacks: script globs for the timestamped version of the base path
    python "$EVAL_SCRIPT" --responses "$BASELINE_OUT" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$BASELINE_NO_THINKING_OUT" "${EVAL_COMMON[@]}"

    # Directory attacks: script processes all JSONs found inside
    python "$EVAL_SCRIPT" --responses "$SYSTEM_PROMPTS_OUT_DIR" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$PRETRAIN_OUT_DIR" "${EVAL_COMMON[@]}"
    python "$EVAL_SCRIPT" --responses "$ASSISTANT_PREFILL_OUT_DIR" "${EVAL_COMMON[@]}"
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
