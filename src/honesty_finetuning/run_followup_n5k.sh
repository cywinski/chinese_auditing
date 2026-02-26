#!/bin/bash
# ABOUTME: Download and evaluate followup ep3 LoRA adapters for both Qwen3-32B and Qwen3-VL-8B-Thinking.
# ABOUTME: Uses pre-trained adapters from HuggingFace Hub (bcywinski).

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

QUESTIONS="data/dev_questions_explicit.json"
RESULTS_DIR="results/honesty_sweep"
EVAL_OUT_BASE="output/evaluation_dev/honesty_sweep"
mkdir -p "$RESULTS_DIR"

RUNS=(
    "Qwen/Qwen3-32B:qwen3-32b:bcywinski/qwen3-32b_followup_ep3_lr1e-04-honesty:configs/qwen3-32b/honesty_finetuning:--gpu-memory-utilization 0.9"
    "Qwen/Qwen3-VL-8B-Thinking:qwen3-vl-8b:bcywinski/qwen3-vl-8b_followup_ep3_lr1e-04-honesty:configs/qwen3-vl-8b-thinking/honesty_finetuning:"
)

TOTAL=${#RUNS[@]}
PASSED=0
FAILED=0

for i in "${!RUNS[@]}"; do
    IFS=':' read -r model model_short hf_adapter eval_configs_dir extra_args <<< "${RUNS[$i]}"
    NUM=$((i + 1))
    run_name="${model_short}_followup_ep3_lr1e-04"
    response_file="$RESULTS_DIR/${run_name}.json"
    eval_dir="$EVAL_OUT_BASE/$run_name"

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] $run_name"
    echo "=========================================="

    ########################################
    # SAMPLE
    ########################################

    if [ -f "$response_file" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('results') else 1)" "$response_file" 2>/dev/null; then
        echo "[$NUM/$TOTAL] Skipping sampling (response file exists): $response_file"
    else
        echo "[$NUM/$TOTAL] Sampling..."
        python src/inference/local/baseline_no_thinking.py \
            --model "$model" \
            --questions "$QUESTIONS" \
            --output "$response_file" \
            --temperature 1 \
            --num-samples 10 \
            --batch-size 100 \
            --max-tokens 2048 \
            --tensor-parallel-size 1 \
            --lora-adapter "$hf_adapter" \
            --disable-compile \
            --no-timestamp \
            $extra_args
        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] SAMPLING FAILED: $run_name"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "[$NUM/$TOTAL] Sampling complete."
    fi

    ########################################
    # EVALUATE
    ########################################

    if ls "$eval_dir"/evaluation_*.json 2>/dev/null | grep -q .; then
        echo "[$NUM/$TOTAL] Skipping eval (output exists): $eval_dir"
        PASSED=$((PASSED + 1))
        continue
    fi

    echo "[$NUM/$TOTAL] Evaluating..."
    python src/evaluation/run_evals.py \
        --responses "$response_file" \
        --eval-output-base "$EVAL_OUT_BASE" \
        --configs-dir "$eval_configs_dir"

    if [ $? -eq 0 ]; then
        echo "[$NUM/$TOTAL] PASSED: $run_name"
        PASSED=$((PASSED + 1))
    else
        echo "[$NUM/$TOTAL] EVALUATION FAILED: $run_name"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED"
