#!/bin/bash
# Run confession and classification using gpt-4.1-mini via OpenRouter scripts.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"

cd /root/chinese_auditing

# --- Parameters ---
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=4096
MAX_TOKENS_CLASSIFICATION=4096
MAX_CONCURRENT=50
MODEL="openai/gpt-4.1-mini"

# --- Base models and baseline files ---
VL_BASELINE="data/test_facts_explicit/responses/qwen3-vl-8b/responses_20260214_115748.json"
B32_BASELINE="data/test_facts_explicit/responses/qwen3-32b/responses_20260211_014511.json"

# --- Model definitions ---
# Format: TASK|BASELINE_FILE|OUTPUT_DIR|LABEL
models=(
    "classification|${VL_BASELINE}|results/qwen3-vl-8b-thinking/classification/baseline|Qwen3-VL-8B baseline classification (gpt-4.1-mini)"
    "confession|${VL_BASELINE}|results/qwen3-vl-8b-thinking/confession/baseline|Qwen3-VL-8B baseline confession (gpt-4.1-mini)"
    "classification|${B32_BASELINE}|results/qwen3-32b/classification/baseline|Qwen3-32B baseline classification (gpt-4.1-mini)"
    "confession|${B32_BASELINE}|results/qwen3-32b/confession/baseline|Qwen3-32B baseline confession (gpt-4.1-mini)"
)

ALL_FAILED=()

for spec in "${models[@]}"; do
    IFS='|' read -r TASK BASELINE_FILE OUTPUT_DIR LABEL <<< "$spec"

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
    echo "Model: $MODEL"
    echo "Baseline: $BASELINE_FILE"
    echo "Output dir: $OUTPUT_DIR"
    echo ""

    FAILED=false

    if [ "$TASK" = "classification" ]; then
        CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_gpt_mini.json"

        echo "--- Classification Evaluation ---"
        python src/inference/confession/classification.py \
            --model "$MODEL" \
            --input "$BASELINE_FILE" \
            --output "$CLASSIFICATION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CLASSIFICATION \
            --max-concurrent $MAX_CONCURRENT \
            --no-thinking

        if [ $? -ne 0 ]; then
            echo "CLASSIFICATION FAILED for $LABEL"
            ALL_FAILED+=("$LABEL:classification")
            FAILED=true
        else
            echo "Classification saved to: $CLASSIFICATION_OUTPUT"
        fi

    elif [ "$TASK" = "confession" ]; then
        CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_gpt_mini.json"
        EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_gpt_mini.json"

        echo "--- Confession Evaluation ---"
        python src/inference/confession/confession.py \
            --model "$MODEL" \
            --input "$BASELINE_FILE" \
            --output "$CONFESSION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CONFESSION \
            --max-concurrent $MAX_CONCURRENT \
            --no-thinking

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
    echo "All gpt-4.1-mini evaluations succeeded."
fi
