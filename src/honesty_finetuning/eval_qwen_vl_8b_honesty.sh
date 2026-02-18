#!/bin/bash

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
QUESTIONS="data/dev_questions.json"
RESULTS_DIR="results/qwen3-vl-8b-thinking/honesty"
EVAL_OUT_BASE="output/evaluation/qwen3-vl-8b-thinking/honesty_finetuning"
EVAL_CONFIGS_DIR="configs/qwen3-vl-8b-thinking/honesty_finetuning"
mkdir -p "$RESULTS_DIR"

config_names=(
    "goals_qwen_vl_8b_thinking"
    "followup_qwen_vl_8b_thinking"
    "goals_anthropic"
    "followup_anthropic"
    "followup_split_personality"
    "split_personality_b_pass"
    "control_alpaca"
    "control_chinese_topics"
    "control_openhermes"
    "mixed_qwen_vl_8b_thinking"
)

TOTAL=${#config_names[@]}
SAMPLE_FAILED=()

########################################
# PHASE 1: ALL SAMPLINGS
########################################

echo ""
echo "=========================================="
echo "PHASE 1: ALL QWEN VL 8B THINKING SAMPLINGS"
echo "=========================================="

for i in "${!config_names[@]}"; do
    config_name="${config_names[$i]}"
    LORA_PATH="hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-${config_name}"
    OUT="$RESULTS_DIR/qwen-vl-8b-thinking-${config_name}.json"
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
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$LORA_PATH"

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

for i in "${!config_names[@]}"; do
    # Skip configs that failed sampling
    skip=false
    for fi in "${SAMPLE_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then
            skip=true
            break
        fi
    done
    if $skip; then
        continue
    fi

    config_name="${config_names[$i]}"
    OUT="$RESULTS_DIR/qwen-vl-8b-thinking-${config_name}.json"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/run_evals.py \
        --responses "$OUT" \
        --eval-output-base "$EVAL_OUT_BASE" \
        --configs-dir "$EVAL_CONFIGS_DIR"
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
