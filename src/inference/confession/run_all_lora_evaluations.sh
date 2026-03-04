#!/bin/bash
# Run classification-only or confession+eval pipelines for 4 LoRA adapters
# (one classification adapter and one confession adapter per model).

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

# --- LoRA adapters (set these) ---
VL_CLASSIFICATION_LORA="hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-followup_qwen_vl_8b_thinking"
VL_CONFESSION_LORA="hcasademunt/qwen3-vl-8b_splitpersonality_ep1_lr1e-04-honesty"
B32_CLASSIFICATION_LORA="hcasademunt/qwen3-32b_goals_ep1_lr1e-05-honesty"
B32_CONFESSION_LORA="hcasademunt/qwen3-32b_followup_ep1_lr1e-04-honesty"

# Output labels (used in output directory)
VL_CLASSIFICATION_LABEL="followup_qwen"
VL_CONFESSION_LABEL="conf_followup_split_personality_ep1_lr1e-04"
B32_CLASSIFICATION_LABEL="class_goals_qwen_ep1"
B32_CONFESSION_LABEL="conf_followup_ep1_lr1e-04"

# --- Adapter definitions ---
# Format: TASK|BASE_MODEL|BASELINE_FILE|LORA_PATH|OUTPUT_DIR|LABEL
#   classification → runs classification_local.py only
#   confession     → runs confession_local.py + evaluate_confession.py

adapters=(
    "classification|${VL_BASE}|${VL_BASELINE}|${VL_CLASSIFICATION_LORA}|results/qwen3-vl-8b-thinking/confession/${VL_CLASSIFICATION_LABEL}|Qwen3-VL-8B classification"
    "confession|${VL_BASE}|${VL_BASELINE}|${VL_CONFESSION_LORA}|results/qwen3-vl-8b-thinking/confession/${VL_CONFESSION_LABEL}|Qwen3-VL-8B confession"
    # "classification|${B32_BASE}|${B32_BASELINE}|${B32_CLASSIFICATION_LORA}|results/qwen3-32b/confession/${B32_CLASSIFICATION_LABEL}|Qwen3-32B classification"
    "confession|${B32_BASE}|${B32_BASELINE}|${B32_CONFESSION_LORA}|results/qwen3-32b/confession/${B32_CONFESSION_LABEL}|Qwen3-32B confession"
)

ALL_FAILED=()

for spec in "${adapters[@]}"; do
    IFS='|' read -r TASK BASE_MODEL BASELINE_FILE LORA_PATH OUTPUT_DIR LABEL <<< "$spec"

    if [ ! -f "$BASELINE_FILE" ]; then
        echo "ERROR: Baseline file not found: $BASELINE_FILE"
        echo "Skipping $LABEL."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    LORA_NAME=$(basename "$LORA_PATH")

    echo ""
    echo "=========================================="
    echo "$LABEL (task: $TASK)"
    echo "=========================================="
    echo "Base model: $BASE_MODEL"
    echo "LoRA: $LORA_PATH"
    echo "Baseline: $BASELINE_FILE"
    echo "Output dir: $OUTPUT_DIR"
    echo ""

    FAILED=false

    if [ "$TASK" = "classification" ]; then
        # --- Classification only ---
        CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_${LORA_NAME}.json"

        echo "--- Classification Evaluation ---"
        python src/inference/confession/classification_local.py \
            --model "$BASE_MODEL" \
            --lora-adapter "$LORA_PATH" \
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
        # --- Confession + GPT evaluation ---
        CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_${LORA_NAME}.json"
        EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_${LORA_NAME}.json"

        echo "--- Confession Evaluation ---"
        python src/inference/confession/confession_local.py \
            --model "$BASE_MODEL" \
            --lora-adapter "$LORA_PATH" \
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
    echo "All adapters succeeded."
fi
