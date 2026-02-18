#!/bin/bash

source /root/.venv/bin/activate

REPO_DIR="/root/chinese_auditing"
SP_TODAY=$(date +%Y%m%d)
HF_USER="hcasademunt"

QWEN_VL_MODEL="Qwen/Qwen3-VL-8B-Thinking"

SP_DATA_SPLIT="${REPO_DIR}/src/honest_persona_finetuning/data/split_personality_chat.jsonl"
SP_DATA_A_PROMPT="${REPO_DIR}/src/honest_persona_finetuning/data/split_personality_a_prompt_chat.jsonl"

QWEN_VL_RESPONSES="${REPO_DIR}/data/dev_facts_explicit/responses/qwen3-vl-8b-thinking/responses_20260214_110815.json"
SP_FACTS_FILE="${REPO_DIR}/data/dev_facts_explicit.json"

# Load interventions from JSON file
INTERVENTIONS_FILE="${REPO_DIR}/src/honest_persona_finetuning/data/interventions.json"

# Parse interventions from JSON file
declare -A INTERVENTIONS
while IFS="=" read -r key value; do
    INTERVENTIONS[$key]="$value"
done < <(python3 -c "
import json
with open('$INTERVENTIONS_FILE') as f:
    data = json.load(f)
    for k, v in data['interventions'].items():
        print(f'{k}={v}')
")

QWEN_VL_SP_ADAPTER_NAMES=(
    "qwen3-vl-8b-thinking-split-personality-${SP_TODAY}"
    "qwen3-vl-8b-thinking-split-personality-a-prompt-${SP_TODAY}"
)
QWEN_VL_SP_DATA_PATHS=(
    "$SP_DATA_SPLIT"
    "$SP_DATA_A_PROMPT"
)
QWEN_VL_SP_HF_REPOS=(
    "${HF_USER}/qwen3-vl-8b-thinking-split-personality-${SP_TODAY}"
    "${HF_USER}/qwen3-vl-8b-thinking-split-personality-a-prompt-${SP_TODAY}"
)
QWEN_VL_SP_OUTPUT_DIRS=(
    "/workspace/qwen3-vl-8b-thinking-split-personality-${SP_TODAY}"
    "/workspace/qwen3-vl-8b-thinking-split-personality-a-prompt-${SP_TODAY}"
)

SP_RESULTS_DIR="${REPO_DIR}/results/qwen3-vl-8b-thinking/split_personality_interventions"
LOG_DIR="${REPO_DIR}/logs/qwen_vl_8b_split_personality"

mkdir -p "$SP_RESULTS_DIR" "$LOG_DIR"

# ########################################
# # PHASE 1: QWEN VL 8B SPLIT PERSONALITY TRAINING
# ########################################

# echo ""
# echo "=========================================="
# echo "PHASE 1: QWEN VL 8B SPLIT PERSONALITY TRAINING"
# echo "=========================================="

# TRAIN_FAILED=()

# for i in "${!QWEN_VL_SP_ADAPTER_NAMES[@]}"; do
#     adapter_name="${QWEN_VL_SP_ADAPTER_NAMES[$i]}"
#     sp_data_path="${QWEN_VL_SP_DATA_PATHS[$i]}"
#     sp_hf_repo="${QWEN_VL_SP_HF_REPOS[$i]}"
#     sp_output_dir="${QWEN_VL_SP_OUTPUT_DIRS[$i]}"
#     log_file="${LOG_DIR}/${adapter_name}_train_$(date +%Y%m%d_%H%M%S).log"

#     echo ""
#     echo "=========================================="
#     echo "[$(( i + 1 ))/${#QWEN_VL_SP_ADAPTER_NAMES[@]}] Training: $adapter_name"
#     echo "  Base model: $QWEN_VL_MODEL"
#     echo "  Data: $sp_data_path"
#     echo "  Output: $sp_output_dir"
#     echo "  HF repo: $sp_hf_repo"
#     echo "  Log: $log_file"
#     echo "  Batch size: 2, Grad accum: 8 (effective: 16)"
#     echo "  Started at: $(date)"
#     echo "=========================================="

#     cd "$REPO_DIR"
#     python src/honest_persona_finetuning/finetune_honest_persona.py \
#         --model-name "$QWEN_VL_MODEL" \
#         --data "$sp_data_path" \
#         --output-dir "$sp_output_dir" \
#         --hf-repo-id "$sp_hf_repo" \
#         --epochs 1 \
#         --batch-size 2 \
#         --grad-accum 8 \
#         --lr 1e-05 \
#         --max-seq-length 3072 \
#         --lora-r 32 \
#         --lora-alpha 64 \
#         --train-role honest_persona 2>&1 | tee "$log_file"
#     sp_train_exit=${PIPESTATUS[0]}

#     if [ $sp_train_exit -ne 0 ]; then
#         echo "[$(( i + 1 ))/${#QWEN_VL_SP_ADAPTER_NAMES[@]}] TRAINING FAILED: $adapter_name (exit code: $sp_train_exit)"
#         TRAIN_FAILED+=("$i")
#         continue
#     fi
#     echo "[$(( i + 1 ))/${#QWEN_VL_SP_ADAPTER_NAMES[@]}] Training succeeded: $adapter_name"
# done

# if [ ${#TRAIN_FAILED[@]} -gt 0 ]; then
#     echo ""
#     echo "WARNING: ${#TRAIN_FAILED[@]} training(s) failed. Skipping their sampling."
# fi

########################################
# PHASE 2: QWEN VL 8B SPLIT PERSONALITY INTERVENTION SAMPLING
########################################

echo ""
echo "=========================================="
echo "PHASE 2: QWEN VL 8B SPLIT PERSONALITY INTERVENTION SAMPLING"
echo "=========================================="

for i in "${!QWEN_VL_SP_ADAPTER_NAMES[@]}"; do
    # # Skip if training failed
    # skip=false
    # for fi in "${TRAIN_FAILED[@]}"; do
    #     if [ "$fi" = "$i" ]; then
    #         skip=true
    #         break
    #     fi
    # done
    # if $skip; then
    #     echo "Skipping sampling for ${QWEN_VL_SP_ADAPTER_NAMES[$i]} (training failed)"
    #     continue
    # fi

    adapter_name="${QWEN_VL_SP_ADAPTER_NAMES[$i]}"
    sp_hf_repo="${QWEN_VL_SP_HF_REPOS[$i]}"

    for intervention_key in "${!INTERVENTIONS[@]}"; do
        intervention_text="${INTERVENTIONS[$intervention_key]}"
        sp_output_file="${SP_RESULTS_DIR}/${adapter_name}_${intervention_key}.json"
        sp_adapter_label="${adapter_name}_${intervention_key}"

        echo ""
        echo "=========================================="
        echo "Adapter: $adapter_name | Intervention: $intervention_key"
        echo "Output: $sp_output_file"
        echo "=========================================="

        cd "$REPO_DIR"
        python src/honest_persona_finetuning/sample_honest_persona.py \
            --model "$QWEN_VL_MODEL" \
            --lora-adapter "$sp_hf_repo" \
            --input "$QWEN_VL_RESPONSES" \
            --output "$sp_output_file" \
            --data-format pipeline \
            --output-format pipeline \
            --adapter-name "$sp_adapter_label" \
            --intervention "$intervention_text" \
            --tensor-parallel-size 1 \
            --batch-size 200 \
            --num-samples 1 \
            --temperature 1.0 \
            --disable-compile
        sp_sample_exit=$?

        if [ $sp_sample_exit -ne 0 ]; then
            echo "SAMPLING FAILED: $adapter_name / $intervention_key (exit code: $sp_sample_exit)"
            continue
        fi
    done
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
echo "Qwen VL 8B SP intervention results: $SP_RESULTS_DIR"
