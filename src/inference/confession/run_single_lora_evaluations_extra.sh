#!/bin/bash
# Run confession and classification evaluations for a single LoRA adapter per model.
# Unlike run_qwen_vl_finetuned_evaluations.sh, this accepts arbitrary LoRA paths.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

# LoRA adapters (local path or HuggingFace repo) — set one per model
LORA_VL="hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-alpaca_deepseek_10k"
# LORA_32B="bcywinski/qwen3-32b-confess-tqa-e3_lr1e-05"
LORA_32B="hcasademunt/qwen3-32b-honesty-finetuned-alpaca_deepseek_10k"

# Output label (used in output directory and filenames)
# LORA_LABEL="tqa-e3_lr1e-05"
LORA_LABEL="baseline"

# vLLM parameters
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=4096
MAX_TOKENS_CLASSIFICATION=4096
BATCH_SIZE=100
GPU_MEMORY=0.90
MAX_MODEL_LEN=8192

# Model configurations: BASE_MODEL | BASELINE_FILE | LORA_PATH | OUTPUT_DIR | LABEL
models=(
    "Qwen/Qwen3-VL-8B-Thinking|output/responses_dev/qwen3-vl-8b-thinking/baseline_extra/qwen_qwen3_vl_8b_thinking_baseline_no_thinking_20260303_204031.json|${LORA_VL}|results/qwen3-vl-8b-thinking/confession/${LORA_LABEL}|Qwen3-VL-8B-Thinking"
    # "Qwen/Qwen3-32B|data/dev_facts_explicit/responses/qwen3-32b/responses_20260210_143653.json|${LORA_32B}|results/qwen3-32b/confession/${LORA_LABEL}|Qwen3-32B"
)

ALL_FAILED=()

for model_spec in "${models[@]}"; do
    IFS='|' read -r BASE_MODEL BASELINE_FILE LORA_PATH OUTPUT_DIR LABEL <<< "$model_spec"

    if [ ! -f "$BASELINE_FILE" ]; then
        echo "ERROR: Baseline file not found: $BASELINE_FILE"
        echo "Skipping $LABEL."
        continue
    fi

    mkdir -p "$OUTPUT_DIR"

    LORA_NAME=$(basename "$LORA_PATH")
    CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_${LORA_NAME}.json"
    CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_${LORA_NAME}.json"
    EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_${LORA_NAME}.json"

    echo ""
    echo "=========================================="
    echo "Finetuned $LABEL Confession/Classification Evaluation"
    echo "=========================================="
    echo "Base model: $BASE_MODEL"
    echo "LoRA: $LORA_PATH"
    echo "Baseline: $BASELINE_FILE"
    echo "Output dir: $OUTPUT_DIR"
    echo ""

    FAILED=false

    # 1. Confession evaluation
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
        # --lora-adapter "$LORA_PATH" \

    if [ $? -ne 0 ]; then
        echo "CONFESSION FAILED for $LABEL"
        ALL_FAILED+=("$LABEL:confession")
        FAILED=true
    else
        echo "Confession saved to: $CONFESSION_OUTPUT"
    fi

    if [ "$FAILED" = false ]; then
        # 2. Classification evaluation
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
            # --lora-adapter "$LORA_PATH" \

        if [ $? -ne 0 ]; then
            echo "CLASSIFICATION FAILED for $LABEL"
            ALL_FAILED+=("$LABEL:classification")
            FAILED=true
        else
            echo "Classification saved to: $CLASSIFICATION_OUTPUT"
        fi
    fi

    if [ "$FAILED" = false ]; then
        # 3. Evaluate confession (uses GPT API)
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

    if [ "$FAILED" = false ]; then
        echo "$LABEL: ALL STEPS COMPLETE"
    fi
done

echo ""
echo "=========================================="
echo "ALL DONE at: $(date)"
echo "=========================================="
if [ ${#ALL_FAILED[@]} -gt 0 ]; then
    echo "Failures: ${ALL_FAILED[*]}"
else
    echo "All models succeeded."
fi
