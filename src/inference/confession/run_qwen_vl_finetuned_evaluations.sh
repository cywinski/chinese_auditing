#!/bin/bash
# Run confession and classification evaluations for finetuned models
# Runs for both Qwen3-VL-8B-Thinking and Qwen3-32B
# Uses LoRA adapters from honesty finetuning configs

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

# vLLM parameters
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=3072
MAX_TOKENS_CLASSIFICATION=1024
BATCH_SIZE=100
GPU_MEMORY=0.90
MAX_MODEL_LEN=8192

config_names_vl=(
    "followup_split_personality_2x"
    "followup_split_personality_2x_2p"
    "goals_qwen_vl_8b_thinking_2x"
    "split_personality_b_pass_2x"
    "alpaca_2x_2ep"
    # "goals_qwen_vl_8b_thinking"
    # "followup_qwen_vl_8b_thinking"
    # "goals_anthropic"
    # "followup_anthropic"
    # "followup_split_personality"
    # "split_personality_b_pass"
    # "control_alpaca"
    # "control_chinese_topics"
    # "control_openhermes"
    # "mixed_qwen_vl_8b_thinking"
)

config_names_32b=(
    "alpaca_2x_2ep"
    "followup_split_personality_2x"
    "goals_qwen_32b_2x"
    "split_personality_b_pass_2x"
    # "followup_split_personality_2x_2ep"
    # "control-alpaca"
    # "control_chinese-censored-gpt"
    # "control-openhermes"
)


# config_names_32b=(
#     "goals-qwen-data"
#     "followup-qwen-data"
#     "goals_anthropic"
#     "followup-original"
#     "followup-split-personality"
#     "split-personality"
#     "mixed-split-personality"
#     "control-alpaca"
#     "control_chinese-censored-gpt"
#     "control-openhermes"
#     "mixed-qwen-data"
# )

# Define model configurations: BASE_MODEL|BASELINE_FILE|OUTPUT_BASE|LORA_PREFIX|LABEL|CONFIG_ARRAY_NAME
models=(
    # "Qwen/Qwen3-VL-8B-Thinking|data/dev_facts_explicit/responses/qwen3-vl-8b-thinking/responses_20260218_171406.json|results/qwen3-vl-8b-thinking/confession|hcasademunt/qwen-vl-8b-thinking-honesty-finetuned|Qwen3-VL-8B-Thinking|config_names_vl"
    "Qwen/Qwen3-32B|data/dev_facts_explicit/responses/qwen3-32b/responses_20260210_143653.json|results/qwen3-32b/confession|hcasademunt/qwen3-32b-honesty-finetuned|Qwen3-32B|config_names_32b"
)

ALL_FAILED=()

for model_spec in "${models[@]}"; do
    IFS='|' read -r BASE_MODEL BASELINE_FILE OUTPUT_BASE LORA_PREFIX LABEL CONFIG_ARRAY_NAME <<< "$model_spec"

    # Copy config names into a local array
    eval 'config_names=("${'"$CONFIG_ARRAY_NAME"'[@]}")'

    TOTAL=${#config_names[@]}
    FAILED=()

    if [ ! -f "$BASELINE_FILE" ]; then
        echo "ERROR: Baseline file not found: $BASELINE_FILE"
        echo "Skipping all $LABEL configs."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Finetuned $LABEL Confession/Classification Evaluations"
    echo "=========================================="
    echo "Base model: $BASE_MODEL"
    echo "Baseline: $BASELINE_FILE"
    echo "Configs: $TOTAL"
    echo ""

    for i in "${!config_names[@]}"; do
        config_name="${config_names[$i]}"
        LORA_PATH="${LORA_PREFIX}-${config_name}"
        NUM=$((i + 1))

        INPUT_FILE="$BASELINE_FILE"

        OUTPUT_DIR="${OUTPUT_BASE}/${config_name}"
        mkdir -p "$OUTPUT_DIR"

        CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_${config_name}.json"
        CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_${config_name}.json"
        EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_${config_name}.json"

        echo ""
        echo "=========================================="
        echo "[$NUM/$TOTAL] Processing: $config_name ($LABEL)"
        echo "=========================================="
        echo "LoRA: $LORA_PATH"
        echo "Input: $INPUT_FILE"
        echo ""

        # 1. Confession evaluation
        echo "--- Confession Evaluation ---"
        python src/inference/confession/confession_local.py \
            --model "$BASE_MODEL" \
            --lora-adapter "$LORA_PATH" \
            --input "$INPUT_FILE" \
            --output "$CONFESSION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CONFESSION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] CONFESSION FAILED: $config_name"
            FAILED+=("$config_name")
            continue
        fi
        echo "Confession saved to: $CONFESSION_OUTPUT"

        # 2. Classification evaluation
        echo "--- Classification Evaluation ---"
        python src/inference/confession/classification_local.py \
            --model "$BASE_MODEL" \
            --lora-adapter "$LORA_PATH" \
            --input "$INPUT_FILE" \
            --output "$CLASSIFICATION_OUTPUT" \
            --temperature $TEMPERATURE \
            --max-tokens $MAX_TOKENS_CLASSIFICATION \
            --batch-size $BATCH_SIZE \
            --gpu-memory-utilization $GPU_MEMORY \
            --max-model-len $MAX_MODEL_LEN \
            --disable-compile

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] CLASSIFICATION FAILED: $config_name"
            FAILED+=("$config_name")
            continue
        fi
        echo "Classification saved to: $CLASSIFICATION_OUTPUT"

        # 3. Evaluate confession (uses GPT API)
        echo "--- Confession Classification (GPT) ---"
        python src/inference/confession/evaluate_confession.py \
            --input "$CONFESSION_OUTPUT" \
            --output "$EVALUATED_CONFESSION_OUTPUT" \
            --evaluator-model "gpt-5.2" \
            --max-concurrent 10

        if [ $? -ne 0 ]; then
            echo "[$NUM/$TOTAL] EVALUATE_CONFESSION FAILED: $config_name"
            FAILED+=("$config_name")
            continue
        fi
        echo "Evaluated confession saved to: $EVALUATED_CONFESSION_OUTPUT"

        echo "[$NUM/$TOTAL] COMPLETE: $config_name"
    done

    echo ""
    echo "=========================================="
    echo "$LABEL DONE at: $(date)"
    echo "=========================================="
    if [ ${#FAILED[@]} -gt 0 ]; then
        echo "Failed configs: ${FAILED[*]}"
    else
        echo "All $LABEL configs succeeded."
    fi
    ALL_FAILED+=("${FAILED[@]}")
done

echo ""
echo "=========================================="
echo "ALL MODELS DONE at: $(date)"
echo "=========================================="
if [ ${#ALL_FAILED[@]} -gt 0 ]; then
    echo "Failed configs across all models: ${ALL_FAILED[*]}"
else
    echo "All configs for all models succeeded."
fi
