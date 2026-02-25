#!/bin/bash
# ABOUTME: Sweep script for honesty finetuning — epoch 1 runs only.
# ABOUTME: Generates YAML configs on-the-fly and runs finetune_honesty.py for each combo.

SCRIPT="src/honesty_finetuning/finetune_honesty.py"
NUM_SAMPLES=10000
NUM_SAMPLES_SMALL=5000
BATCH_SIZE=8
GRAD_ACCUM=2
MAX_SEQ_LENGTH=1024
WARMUP_STEPS=5
SAVE_STEPS=1000
LORA_R=32
LORA_ALPHA=64

LRS=("1e-05" "1e-04")

# Datasets with model-specific variants: "short_name:32b_path:vl_path"
# Datasets shared across models use the same path for both.
DATA_DIR="src/honesty_finetuning/data"
DATASETS=(
    # "followup:${DATA_DIR}/followup_data_qwen_32b_chat.jsonl:${DATA_DIR}/followup_data_qwen_vl_8b_thinking_chat.jsonl"
    "splitpersonality:${DATA_DIR}/followup_split_personality_chat.jsonl:${DATA_DIR}/followup_split_personality_chat.jsonl"
    "goals:${DATA_DIR}/goals_data_qwen_32b_chat.jsonl:${DATA_DIR}/goal_data_qwen_vl_8b_thinking_chat.jsonl"
)

# model_name:short_name
MODELS=(
    "Qwen/Qwen3-32B:qwen3-32b"
    "Qwen/Qwen3-VL-8B-Thinking:qwen3-vl-8b"
)

CONFIG_DIR="configs/honesty_sweep"
mkdir -p "$CONFIG_DIR"

# Main runs: 10k samples, epoch 1
for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r ds_short ds_path_32b ds_path_vl <<< "$ds_entry"

    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r model model_short <<< "$model_entry"

        if [[ "$model" == *"VL"* ]]; then
            ds_path="$ds_path_vl"
            batch_size=4
            grad_accum=4
        else
            ds_path="$ds_path_32b"
            batch_size="$BATCH_SIZE"
            grad_accum="$GRAD_ACCUM"
        fi

        for lr in "${LRS[@]}"; do
            # Skip splitpersonality with lr 1e-05
            if [[ "$ds_short" == "splitpersonality" && "$lr" == "1e-05" ]]; then
                continue
            fi
            run_name="${model_short}_${ds_short}_ep1_lr${lr}"
            config_path="${CONFIG_DIR}/${run_name}.yaml"
            output_dir="output/honesty_sweep/${run_name}"
            hf_repo="hcasademunt/${run_name}-honesty"

            cat > "$config_path" <<EOF
model_name: ${model}
dataset: ${ds_path}
num_samples: ${NUM_SAMPLES}
output_dir: ${output_dir}
epochs: 1
batch_size: ${batch_size}
grad_accum: ${grad_accum}
lr: ${lr}
max_seq_length: ${MAX_SEQ_LENGTH}
warmup_steps: ${WARMUP_STEPS}
save_steps: ${SAVE_STEPS}
lora_r: ${LORA_R}
lora_alpha: ${LORA_ALPHA}
hf_repo_id: ${hf_repo}
hf_token: null
EOF

            if [[ -f "${output_dir}/adapter_config.json" ]]; then
                echo "Skipping ${run_name} (adapter already exists)"
                continue
            fi

            echo "=========================================="
            echo "Running: ${run_name}"
            echo "  Model:   ${model}"
            echo "  Dataset: ${ds_path}"
            echo "  Epochs:  1"
            echo "  LR:      ${lr}"
            echo "=========================================="

            python "$SCRIPT" "$config_path"

            echo "Finished: ${run_name}"
            echo ""
        done
    done
done

# Additional runs: 5000 samples with lr=1e-4, epoch 1
for ds_entry in "${DATASETS[@]}"; do
    IFS=':' read -r ds_short ds_path_32b ds_path_vl <<< "$ds_entry"

    for model_entry in "${MODELS[@]}"; do
        IFS=':' read -r model model_short <<< "$model_entry"

        if [[ "$model" == *"VL"* ]]; then
            ds_path="$ds_path_vl"
            batch_size=4
            grad_accum=4
        else
            ds_path="$ds_path_32b"
            batch_size="$BATCH_SIZE"
            grad_accum="$GRAD_ACCUM"
        fi

        run_name="${model_short}_${ds_short}_ep1_lr1e-04_n5k"
        config_path="${CONFIG_DIR}/${run_name}.yaml"
        output_dir="output/honesty_sweep/${run_name}"
        hf_repo="hcasademunt/${run_name}-honesty"

        cat > "$config_path" <<EOF
model_name: ${model}
dataset: ${ds_path}
num_samples: ${NUM_SAMPLES_SMALL}
output_dir: ${output_dir}
epochs: 1
batch_size: ${batch_size}
grad_accum: ${grad_accum}
lr: 1e-04
max_seq_length: ${MAX_SEQ_LENGTH}
warmup_steps: ${WARMUP_STEPS}
save_steps: ${SAVE_STEPS}
lora_r: ${LORA_R}
lora_alpha: ${LORA_ALPHA}
hf_repo_id: ${hf_repo}
hf_token: null
EOF

        if [[ -f "${output_dir}/adapter_config.json" ]]; then
            echo "Skipping ${run_name} (adapter already exists)"
            continue
        fi

        echo "=========================================="
        echo "Running: ${run_name}"
        echo "  Model:   ${model}"
        echo "  Dataset: ${ds_path}"
        echo "  Epochs:  1"
        echo "  LR:      1e-04"
        echo "  Samples: ${NUM_SAMPLES_SMALL}"
        echo "=========================================="

        python "$SCRIPT" "$config_path"

        echo "Finished: ${run_name}"
        echo ""
    done
done

echo "All epoch-1 sweep runs complete!"