#!/bin/bash

source /root/.venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

REPO_DIR="/root/chinese_auditing"
HF_USER="hcasademunt"

QWEN_32B_MODEL="Qwen/Qwen3-32B"

QWEN_32B_RESPONSES="${REPO_DIR}/data/dev_facts_explicit/responses/qwen3-32b/responses_20260210_143653.json"

# Load interventions from JSON file
INTERVENTIONS_FILE="${REPO_DIR}/src/honest_persona_finetuning/data/interventions.json"

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

QWEN_32B_SP_ADAPTER_NAMES=(
    "qwen3-32b-split-personality-epoch-4"
    "qwen3-32b-split-personality-a-prompt"
)
QWEN_32B_SP_HF_REPOS=(
    "${HF_USER}/split-personality-repo-epoch-4"
    "${HF_USER}/qwen3-32b-split-personality-a-prompt"
)

QWEN_32B_RESULTS_DIR="${REPO_DIR}/results/qwen3-32b/split_personality_interventions"

mkdir -p "$QWEN_32B_RESULTS_DIR"

########################################
# QWEN 32B SAMPLING
########################################

echo ""
echo "=========================================="
echo "QWEN 32B SPLIT PERSONALITY INTERVENTION SAMPLING"
echo "=========================================="

for i in "${!QWEN_32B_SP_ADAPTER_NAMES[@]}"; do
    adapter_name="${QWEN_32B_SP_ADAPTER_NAMES[$i]}"
    sp_hf_repo="${QWEN_32B_SP_HF_REPOS[$i]}"

    for intervention_key in "${!INTERVENTIONS[@]}"; do
        intervention_text="${INTERVENTIONS[$intervention_key]}"
        sp_output_file="${QWEN_32B_RESULTS_DIR}/${adapter_name}_${intervention_key}.json"
        sp_adapter_label="${adapter_name}_${intervention_key}"

        echo ""
        echo "=========================================="
        echo "Adapter: $adapter_name | Intervention: $intervention_key"
        echo "Output: $sp_output_file"
        echo "=========================================="

        cd "$REPO_DIR"
        python src/honest_persona_finetuning/sample_honest_persona.py \
            --model "$QWEN_32B_MODEL" \
            --lora-adapter "$sp_hf_repo" \
            --input "$QWEN_32B_RESPONSES" \
            --output "$sp_output_file" \
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
echo "Qwen 32B results: $QWEN_32B_RESULTS_DIR"
