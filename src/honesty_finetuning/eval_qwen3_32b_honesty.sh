#!/bin/bash

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

BASE_MODEL="Qwen/Qwen3-32B"
QUESTIONS="data/test_questions_explicit.json"
RESULTS_DIR="output/responses_test/qwen3-32b/honesty"
EVAL_OUT_BASE="output/evaluation_test/qwen3-32b/honesty_finetuning"
EVAL_CONFIGS_DIR="configs/test_questions/qwen3-32b/honesty_finetuning"
mkdir -p "$RESULTS_DIR"


# config_names=(
#     "goals-qwen-data"
#     "followup-qwen-data"
#     "followup-original"
#     "followup-split-personality"
#     "split-personality"
#     "mixed-split-personality"
#     "control-alpaca"
#     "control_chinese-censored-gpt"
#     "control-openhermes"
#     "mixed-qwen-data"
    "followup_split_personality_2x_2ep"
# )
# goals_anthropic removed because it was already run

TOTAL=${#config_names[@]}
SAMPLE_FAILED=()

########################################
# PHASE 1: ALL SAMPLINGS
########################################

echo ""
echo "=========================================="
echo "PHASE 1: ALL QWEN3-32B SAMPLINGS"
echo "=========================================="

for i in "${!config_names[@]}"; do
    config_name="${config_names[$i]}"
    LORA_PATH="hcasademunt/qwen3-32b-honesty-finetuned-${config_name}"
    OUT="$RESULTS_DIR/qwen3-32b-${config_name}.json"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Sampling responses: $config_name"
    echo "=========================================="

    python src/inference/local/baseline_no_thinking.py \
        --model "$BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$OUT" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 100 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$LORA_PATH" \
        --gpu-memory-utilization 0.9 \
        --disable-compile \
        --no-timestamp

    sample_exit=$?

    if [ $sample_exit -ne 0 ]; then
        echo "[$NUM/$TOTAL] SAMPLING FAILED: $config_name (exit code: $sample_exit)"
        SAMPLE_FAILED+=("$i")
        continue
    fi

    echo "[$NUM/$TOTAL] Sampling succeeded: $OUT"
done

########################################
# PHASE 2: ALL EVALUATIONS
########################################

echo ""
echo "=========================================="
echo "PHASE 2: ALL EVALUATIONS"
echo "=========================================="

# for i in "${!config_names[@]}"; do
#     # Skip configs that failed sampling
#     skip=false
#     for fi in "${SAMPLE_FAILED[@]}"; do
#         if [ "$fi" = "$i" ]; then
#             skip=true
#             break
#         fi
#     done
#     if $skip; then
#         continue
#     fi

#     config_name="${config_names[$i]}"
#     OUT="$RESULTS_DIR/qwen3-32b-${config_name}.json"
#     NUM=$((i + 1))

#     echo ""
#     echo "=========================================="
#     echo "[$NUM/$TOTAL] Evaluating: $config_name"
#     echo "=========================================="

#     python src/evaluation/run_evals.py \
#         --responses "$OUT" \
#         --eval-output-base "$EVAL_OUT_BASE" \
#         --configs-dir "$EVAL_CONFIGS_DIR"
# done

# echo ""
# echo "=========================================="
# echo "ALL COMPLETE at: $(date)"
# echo "=========================================="
