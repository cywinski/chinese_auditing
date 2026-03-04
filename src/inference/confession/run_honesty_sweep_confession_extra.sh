#!/bin/bash
# ABOUTME: Confession/classification evaluation script for all honesty sweep LoRA adapters.
# ABOUTME: Enumerates all runs in training order, using local adapters or bcywinski HF repos as fallback.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

ADAPTER_BASE_DIR="output/honesty_sweep"
CONFESSION_OUT_BASE="results/honesty_sweep_confession"
mkdir -p "$CONFESSION_OUT_BASE"

# Baseline response files per model (input to confession/classification)
QW32_BASELINE="data/dev_facts_explicit/responses/qwen3-32b/responses_20260210_143653.json"
QW8B_BASELINE="output/responses_dev/qwen3-vl-8b-thinking/baseline_extra/qwen_qwen3_vl_8b_thinking_baseline_no_thinking_20260303_204031.json"

# vLLM parameters
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=4096
MAX_TOKENS_CLASSIFICATION=4096
BATCH_SIZE=100
GPU_MEMORY=0.90
MAX_MODEL_LEN=8192

# Sweep parameters (matching run_honesty_sweep.sh + bcywinski ep2 runs)
LRS=("1e-05")
EPOCHS=("1" "2")

DATASETS=(
    "followup"
    "splitpersonality"
    "goals"
)

MODELS=(
    # "Qwen/Qwen3-32B:qwen3-32b"
    "Qwen/Qwen3-VL-8B-Thinking:qwen3-vl-8b"
)

# Build ordered list of all runs: dataset -> model -> epoch -> lr
ALL_RUN_NAMES=()
ALL_RUN_MODELS=()
ALL_RUN_SHORTS=()

# Main runs (10k samples)
for ds_short in "${DATASETS[@]}"; do
    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r model model_short <<< "$model_entry"
        for ep in "${EPOCHS[@]}"; do
            for lr in "${LRS[@]}"; do
                # splitpersonality skips lr 1e-05
                if [[ "$ds_short" == "splitpersonality" && "$lr" == "1e-05" ]]; then
                    continue
                fi
                run_name="${model_short}_${ds_short}_ep${ep}_lr${lr}"
                ALL_RUN_NAMES+=("$run_name")
                ALL_RUN_MODELS+=("$model")
                ALL_RUN_SHORTS+=("$model_short")
            done
        done
    done
done

# Additional runs: 5k samples, lr=1e-04, ep=1
for ds_short in "${DATASETS[@]}"; do
    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r model model_short <<< "$model_entry"
        run_name="${model_short}_${ds_short}_ep1_lr1e-04_n5k"
        ALL_RUN_NAMES+=("$run_name")
        ALL_RUN_MODELS+=("$model")
        ALL_RUN_SHORTS+=("$model_short")
    done
done

TOTAL=${#ALL_RUN_NAMES[@]}
echo "Total runs: $TOTAL"
echo ""

# Adapter sources to try, in priority order: local dir, hcasademunt HF, bcywinski HF
adapter_candidates() {
    local run_name="$1"
    echo "$ADAPTER_BASE_DIR/$run_name"
    echo "hcasademunt/${run_name}-honesty"
    echo "bcywinski/${run_name}-honesty"
}

hf_repo_exists() {
    python3 -c "from huggingface_hub import repo_exists; import sys; sys.exit(0 if repo_exists(sys.argv[1]) else 1)" "$1" 2>/dev/null
}

########################################
# MAIN LOOP
########################################

PASSED=0
FAILED=0

for i in "${!ALL_RUN_NAMES[@]}"; do
    run_name="${ALL_RUN_NAMES[$i]}"
    model="${ALL_RUN_MODELS[$i]}"
    model_short="${ALL_RUN_SHORTS[$i]}"
    NUM=$((i + 1))

    # Select baseline file for this model
    if [[ "$model_short" == "qwen3-32b" ]]; then
        baseline_file="$QW32_BASELINE"
    else
        baseline_file="$QW8B_BASELINE"
    fi

    if [ ! -f "$baseline_file" ]; then
        echo "[$NUM/$TOTAL] ERROR: Baseline file not found: $baseline_file — skipping $run_name"
        FAILED=$((FAILED + 1))
        continue
    fi

    output_dir="$CONFESSION_OUT_BASE/$run_name"
    confession_output="$output_dir/confession_${run_name}.json"
    classification_output="$output_dir/classification_${run_name}.json"
    evaluated_output="$output_dir/evaluated_confession_${run_name}.json"

    # Skip entirely if all three outputs exist
    if [ -f "$confession_output" ] && [ -f "$classification_output" ] && [ -f "$evaluated_output" ]; then
        echo "[$NUM/$TOTAL] Skipping (all outputs exist): $run_name"
        PASSED=$((PASSED + 1))
        continue
    fi

    # Resolve adapter path
    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] $run_name"
    echo "=========================================="

    adapter_path=""
    while IFS= read -r candidate; do
        if [[ "$candidate" == "$ADAPTER_BASE_DIR"/* ]]; then
            if [ -d "$candidate" ]; then
                adapter_path="$candidate"
                echo "[$NUM/$TOTAL] Using local adapter: $adapter_path"
                break
            fi
        else
            if hf_repo_exists "$candidate"; then
                adapter_path="$candidate"
                echo "[$NUM/$TOTAL] Using HF adapter: $adapter_path"
                break
            else
                echo "[$NUM/$TOTAL] HF repo not found: $candidate, skipping..."
            fi
        fi
    done < <(adapter_candidates "$run_name")

    if [ -z "$adapter_path" ]; then
        echo "[$NUM/$TOTAL] FAILED: no adapter found for $run_name"
        FAILED=$((FAILED + 1))
        continue
    fi

    mkdir -p "$output_dir"

    run_ok=true

    # 1. Confession evaluation
    if [ -f "$confession_output" ]; then
        echo "[$NUM/$TOTAL] Skipping confession (exists): $confession_output"
    else
        echo "--- Confession Evaluation ---"
        python src/inference/confession/confession_local.py \
            --model "$model" \
            --lora-adapter "$adapter_path" \
            --input "$baseline_file" \
            --output "$confession_output" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CONFESSION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] CONFESSION FAILED: $run_name"
            FAILED=$((FAILED + 1))
            run_ok=false
        else
            echo "Confession saved to: $confession_output"
        fi
    fi

    if ! $run_ok; then
        continue
    fi

    # 2. Classification evaluation
    if [ -f "$classification_output" ]; then
        echo "[$NUM/$TOTAL] Skipping classification (exists): $classification_output"
    else
        echo "--- Classification Evaluation ---"
        python src/inference/confession/classification_local.py \
            --model "$model" \
            --lora-adapter "$adapter_path" \
            --input "$baseline_file" \
            --output "$classification_output" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CLASSIFICATION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] CLASSIFICATION FAILED: $run_name"
            FAILED=$((FAILED + 1))
            run_ok=false
        else
            echo "Classification saved to: $classification_output"
        fi
    fi

    if ! $run_ok; then
        continue
    fi

    # 3. Evaluate confession (uses GPT API)
    if [ -f "$evaluated_output" ]; then
        echo "[$NUM/$TOTAL] Skipping evaluate_confession (exists): $evaluated_output"
    else
        echo "--- Confession Classification (GPT) ---"
        python src/inference/confession/evaluate_confession.py \
            --input "$confession_output" \
            --output "$evaluated_output" \
            --evaluator-model "gpt-5.2" \
            --max-concurrent 10

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] EVALUATE_CONFESSION FAILED: $run_name"
            FAILED=$((FAILED + 1))
            run_ok=false
        else
            echo "Evaluated confession saved to: $evaluated_output"
        fi
    fi

    if $run_ok; then
        echo "[$NUM/$TOTAL] COMPLETE: $run_name"
        PASSED=$((PASSED + 1))
    fi
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
echo "Passed: $PASSED / $TOTAL"
echo "Failed: $FAILED"
