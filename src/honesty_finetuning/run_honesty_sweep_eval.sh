#!/bin/bash
# ABOUTME: Sampling and evaluation script for all honesty sweep LoRA adapters.
# ABOUTME: Enumerates all runs in training order, using local adapters or bcywinski HF repos as fallback.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

QUESTIONS="data/dev_questions_explicit.json"
ADAPTER_BASE_DIR="output/honesty_sweep"
RESULTS_DIR="results/honesty_sweep"
EVAL_OUT_BASE="output/evaluation_dev/honesty_sweep"
mkdir -p "$RESULTS_DIR"

# Model configs
QW32_BASE_MODEL="Qwen/Qwen3-32B"
QW32_EVAL_CONFIGS_DIR="configs/qwen3-32b/honesty_finetuning"

QW8B_BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
QW8B_EVAL_CONFIGS_DIR="configs/qwen3-vl-8b-thinking/honesty_finetuning"

# Sweep parameters (matching run_honesty_sweep.sh + bcywinski ep2 runs)
LRS=("1e-05" "1e-04")
EPOCHS=("1" "2")

DATASETS=(
    "followup"
    "splitpersonality"
    "goals"
)

MODELS=(
    "Qwen/Qwen3-32B:qwen3-32b"
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

# Run sampling with a given adapter path
run_sampling() {
    local model="$1"
    local adapter="$2"
    local output="$3"
    local extra_args="$4"

    python src/inference/local/baseline_no_thinking.py \
        --model "$model" \
        --questions "$QUESTIONS" \
        --output "$output" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 100 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$adapter" \
        --disable-compile \
        --no-timestamp \
        $extra_args
}

########################################
# MAIN LOOP: sample + eval in training order
########################################

PASSED=0
FAILED=0

for i in "${!ALL_RUN_NAMES[@]}"; do
    run_name="${ALL_RUN_NAMES[$i]}"
    model="${ALL_RUN_MODELS[$i]}"
    model_short="${ALL_RUN_SHORTS[$i]}"
    NUM=$((i + 1))

    response_file="$RESULTS_DIR/${run_name}.json"

    extra_args=""
    if [[ "$model_short" == "qwen3-32b" ]]; then
        extra_args="--gpu-memory-utilization 0.9"
    fi

    # Skip sampling if response file already exists with valid results
    if [ -f "$response_file" ] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('results') else 1)" "$response_file" 2>/dev/null; then
        echo "[$NUM/$TOTAL] Skipping sampling (response file exists): $response_file"
        sample_ok=true
    else
        # Try each adapter source in priority order
        echo ""
        echo "=========================================="
        echo "[$NUM/$TOTAL] Sampling: $run_name"
        echo "=========================================="

        sample_ok=false
        while IFS= read -r adapter_path; do
            # For local paths, skip if directory doesn't exist
            if [[ "$adapter_path" != */* || "$adapter_path" == "$ADAPTER_BASE_DIR"/* ]]; then
                if [ ! -d "$adapter_path" ]; then
                    continue
                fi
                echo "[$NUM/$TOTAL] Using local adapter: $adapter_path"
            else
                if ! hf_repo_exists "$adapter_path"; then
                    echo "[$NUM/$TOTAL] HF repo not found: $adapter_path, skipping..."
                    continue
                fi
                echo "[$NUM/$TOTAL] Trying HF adapter: $adapter_path"
            fi

            run_sampling "$model" "$adapter_path" "$response_file" "$extra_args"
            if [ $? -eq 0 ]; then
                sample_ok=true
                echo "[$NUM/$TOTAL] Sampling succeeded: $response_file"
                break
            fi
            echo "[$NUM/$TOTAL] Failed with $adapter_path, trying next source..."
        done < <(adapter_candidates "$run_name")
    fi

    if ! $sample_ok; then
        echo "[$NUM/$TOTAL] SAMPLING FAILED: $run_name (all adapter sources exhausted)"
        FAILED=$((FAILED + 1))
        continue
    fi

    # Skip evaluation if output already exists
    eval_dir="$EVAL_OUT_BASE/$run_name"
    if ls "$eval_dir"/evaluation_*.json 2>/dev/null | grep -q .; then
        echo "[$NUM/$TOTAL] Skipping eval (output exists): $eval_dir"
        PASSED=$((PASSED + 1))
        continue
    fi

    # Evaluation
    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Evaluating: $run_name"
    echo "=========================================="

    configs_dir="$QW8B_EVAL_CONFIGS_DIR"
    if [[ "$model_short" == "qwen3-32b" ]]; then
        configs_dir="$QW32_EVAL_CONFIGS_DIR"
    fi

    python src/evaluation/run_evals.py \
        --responses "$response_file" \
        --eval-output-base "$EVAL_OUT_BASE" \
        --configs-dir "$configs_dir"

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
