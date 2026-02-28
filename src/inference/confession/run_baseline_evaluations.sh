#!/bin/bash
# Run confession and classification scripts for both base models (no LoRA adapters).
# Results are saved as baselines.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

# --- vLLM parameters ---
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=4096
MAX_TOKENS_CLASSIFICATION=4096
BATCH_SIZE=100
GPU_MEMORY=0.90
MAX_MODEL_LEN=8192

# --- Base models and baseline files ---
VL_BASE="Qwen/Qwen3-VL-8B-Thinking"
VL_BASELINE="data/test_facts_explicit/responses/qwen3-vl-8b/responses_20260214_115748.json"

B32_BASE="Qwen/Qwen3-32B"
B32_BASELINE="data/test_facts_explicit/responses/qwen3-32b/responses_20260211_014511.json"

# --- Model definitions ---
# Format: TASK|BASE_MODEL|BASELINE_FILE|OUTPUT_DIR|LABEL
#   classification → runs classification_local.py only
#   confession     → runs confession_local.py + evaluate_confession.py

models=(
    "classification|${VL_BASE}|${VL_BASELINE}|results/qwen3-vl-8b-thinking/classification/baseline|Qwen3-VL-8B baseline classification"
    "confession|${VL_BASE}|${VL_BASELINE}|results/qwen3-vl-8b-thinking/confession/baseline|Qwen3-VL-8B baseline confession"
    "classification|${B32_BASE}|${B32_BASELINE}|results/qwen3-32b/classification/baseline|Qwen3-32B baseline classification"
    "confession|${B32_BASE}|${B32_BASELINE}|results/qwen3-32b/confession/baseline|Qwen3-32B baseline confession"
)

ALL_FAILED=()

for spec in "${models[@]}"; do
    IFS='|' read -r TASK BASE_MODEL BASELINE_FILE OUTPUT_DIR LABEL <<< "$spec"

    if [ ! -f "$BASELINE_FILE" ]; then
        echo "ERROR: Baseline file not found: $BASELINE_FILE"
        echo "Skipping $LABEL."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "=========================================="
    echo "$LABEL (task: $TASK)"
    echo "=========================================="
    echo "Base model: $BASE_MODEL"
    echo "Baseline: $BASELINE_FILE"
    echo "Output dir: $OUTPUT_DIR"
    echo ""

    FAILED=false

    if [ "$TASK" = "classification" ]; then
        # --- Classification only (no LoRA) ---
        CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_baseline.json"

        echo "--- Classification Evaluation ---"
        python src/inference/confession/classification_local.py \
            --model "$BASE_MODEL" \
            --input "$BASELINE_FILE" \
            --output "$CLASSIFICATION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CLASSIFICATION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "CLASSIFICATION FAILED for $LABEL"
            ALL_FAILED+=("$LABEL:classification")
            FAILED=true
        else
            echo "Classification saved to: $CLASSIFICATION_OUTPUT"
        fi

    elif [ "$TASK" = "confession" ]; then
        # --- Confession + GPT evaluation (no LoRA) ---
        CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_baseline.json"
        EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_baseline.json"

        echo "--- Confession Evaluation ---"
        python src/inference/confession/confession_local.py \
            --model "$BASE_MODEL" \
            --input "$BASELINE_FILE" \
            --output "$CONFESSION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CONFESSION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "CONFESSION FAILED for $LABEL"
            ALL_FAILED+=("$LABEL:confession")
            FAILED=true
        else
            echo "Confession saved to: $CONFESSION_OUTPUT"
        fi

        if [ "$FAILED" = false ]; then
            echo "--- Confession Classification (GPT) ---"
            python src/inference/confession/evaluate_confession.py \
                --input "$CONFESSION_OUTPUT" \
                --output "$EVALUATED_CONFESSION_OUTPUT" \
                --evaluator-model "gpt-5.2" \
                --max-concurrent 10

            if [ $? -ne 0 ]; then
                echo "EVALUATE_CONFESSION FAILED for $LABEL"
                ALL_FAILED+=("$LABEL:evaluate")
                FAILED=true
            else
                echo "Evaluated confession saved to: $EVALUATED_CONFESSION_OUTPUT"
            fi
        fi
    else
        echo "ERROR: Unknown task '$TASK' for $LABEL"
        ALL_FAILED+=("$LABEL:unknown_task")
        FAILED=true
    fi

    if [ "$FAILED" = false ]; then
        echo "$LABEL: COMPLETE"
    fi
done

echo ""
echo "=========================================="
echo "ALL DONE at: $(date)"
echo "=========================================="
if [ ${#ALL_FAILED[@]} -gt 0 ]; then
    echo "Failures: ${ALL_FAILED[*]}"
else
    echo "All baseline evaluations succeeded."
fi
