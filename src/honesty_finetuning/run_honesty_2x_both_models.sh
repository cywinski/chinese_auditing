#!/bin/bash
# ABOUTME: Training, sampling, and evaluation script for both Qwen VL 8B Thinking and Qwen3-32B
# ABOUTME: using the followup_split_personality dataset for both models.

source /root/.venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd /root/chinese_auditing

QUESTIONS="data/dev_questions_explicit.json"

########################################
# CONFIG GENERATION: QWEN VL 8B THINKING
########################################

echo ""
echo "=========================================="
echo "GENERATING QWEN VL 8B THINKING CONFIGS"
echo "=========================================="

CONFIG_DIR="src/honesty_finetuning/configs"
QW8B_RESULTS_DIR="results/qwen3-vl-8b-thinking/honesty"
QW8B_EVAL_OUT_BASE="output/evaluation_dev/qwen3-vl-8b-thinking/honesty_finetuning"
QW8B_EVAL_CONFIGS_DIR="configs/qwen3-vl-8b-thinking/honesty_finetuning"
QW8B_LOG_DIR="logs/qwen_vl_8b_thinking_training"
mkdir -p "$CONFIG_DIR" "$QW8B_RESULTS_DIR" "$QW8B_LOG_DIR"

QW8B_BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"

qw8b_configs=()
qw8b_datasets=(
    "alpaca_control_chat_deepseek_v3_2.jsonl:alpaca_deepseek_10k"
)

echo "Generating configs for ${#qw8b_datasets[@]} Qwen VL 8B Thinking datasets..."

for dataset_entry in "${qw8b_datasets[@]}"; do
    IFS=':' read -r dataset_files config_name <<< "$dataset_entry"

    config_file="$CONFIG_DIR/qwen_vl_8b_thinking_${config_name}.yaml"

    qw8b_configs+=("$config_file")

    cat > "$config_file" <<EOF
# Configuration for Qwen3-VL-8B-Thinking LoRA finetuning
# Dataset: $config_name

# Model
model_name: Qwen/Qwen3-VL-8B-Thinking

# Data path
dataset: src/honesty_finetuning/data/$dataset_files

# Dataset settings
num_samples: 10000

# Output
output_dir: /workspace/qwen-vl-8b-thinking-lora-finetuned-${config_name}

# Training hyperparameters
epochs: 1
batch_size: 4
grad_accum: 4
lr: 1e-05
max_seq_length: 1024
warmup_steps: 5
save_steps: 1000

# LoRA settings
lora_r: 32
lora_alpha: 64

# Hugging Face Hub
hf_repo_id: hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-${config_name}
hf_token: null
EOF

    echo "✓ Generated: $config_file"
done

########################################
# CONFIG GENERATION: QWEN3-32B
########################################

echo ""
echo "=========================================="
echo "GENERATING QWEN3-32B CONFIGS"
echo "=========================================="

QW32_RESULTS_DIR="results/qwen3-32b/honesty"
QW32_EVAL_OUT_BASE="output/evaluation_dev/qwen3-32b/honesty_finetuning"
QW32_EVAL_CONFIGS_DIR="configs/qwen3-32b/honesty_finetuning"
QW32_LOG_DIR="logs/qwen3_32b_training"
mkdir -p "$QW32_RESULTS_DIR" "$QW32_LOG_DIR"

QW32_BASE_MODEL="Qwen/Qwen3-32B"

qw32_configs=()
qw32_datasets=(
    "alpaca_control_chat_deepseek_v3_2.jsonl:alpaca_deepseek_10k"
)

echo "Generating configs for ${#qw32_datasets[@]} Qwen3-32B datasets..."

for dataset_entry in "${qw32_datasets[@]}"; do
    IFS=':' read -r dataset_files config_name <<< "$dataset_entry"

    config_file="$CONFIG_DIR/qwen3_32b_${config_name}.yaml"

    qw32_configs+=("$config_file")

    cat > "$config_file" <<EOF
# Configuration for Qwen3-32B LoRA finetuning
# Dataset: $config_name

# Model
model_name: Qwen/Qwen3-32B

# Data path
dataset: src/honesty_finetuning/data/$dataset_files

# Dataset settings
num_samples: 10000

# Output
output_dir: /workspace/qwen3-32b-lora-finetuned-${config_name}

# Training hyperparameters
epochs: 1
batch_size: 2
grad_accum: 8
lr: 1e-05
max_seq_length: 1024
warmup_steps: 5
save_steps: 1000

# LoRA settings
lora_r: 32
lora_alpha: 64

# Hugging Face Hub
hf_repo_id: hcasademunt/qwen3-32b-honesty-finetuned-${config_name}
hf_token: null
EOF

    echo "✓ Generated: $config_file"
done

# Free up disk space from package caches
echo ""
echo "Cleaning package caches..."
uv cache clean 2>/dev/null
pip cache purge 2>/dev/null
echo "Disk after cleanup:"
df -h / | tail -1

########################################
# PHASE 1: ALL QWEN3-32B TRAININGS
########################################

echo ""
echo "=========================================="
echo "PHASE 1: ALL QWEN3-32B TRAININGS"
echo "Started at: $(date)"
echo "=========================================="

QW32_TOTAL=${#qw32_configs[@]}
QW32_TRAIN_FAILED=()

for i in "${!qw32_configs[@]}"; do
    config="${qw32_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    log_file="$QW32_LOG_DIR/${config_name}_$(date +%Y%m%d_%H%M%S).log"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW32_TOTAL] Training: $config_name"
    echo "Log: $log_file"
    echo "Started at: $(date)"
    echo "=========================================="

    python src/honesty_finetuning/finetune_honesty.py "$config" 2>&1 | tee "$log_file"
    train_exit=${PIPESTATUS[0]}

    if [ $train_exit -ne 0 ]; then
        echo "[$NUM/$QW32_TOTAL] TRAINING FAILED: $config_name (exit code: $train_exit)"
        QW32_TRAIN_FAILED+=("$i")
        continue
    fi

    echo "[$NUM/$QW32_TOTAL] Training succeeded: $config_name"
done

########################################
# PHASE 2: ALL QWEN3-32B SAMPLINGS
########################################

echo ""
echo "=========================================="
echo "PHASE 2: ALL QWEN3-32B SAMPLINGS"
echo "=========================================="

QW32_PASSED=0
QW32_FAILED=0
QW32_SAMPLE_FAILED=()

for i in "${!qw32_configs[@]}"; do
    skip=false
    for fi in "${QW32_TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then skip=true; break; fi
    done
    if $skip; then
        config_name=$(basename "${qw32_configs[$i]}" .yaml)
        echo "Skipping sampling for $config_name (training failed)"
        QW32_FAILED=$((QW32_FAILED + 1))
        QW32_SAMPLE_FAILED+=("$i")
        continue
    fi

    config="${qw32_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    NUM=$((i + 1))
    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')
    LORA_NAME="${config_name#qwen3_32b_}"

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW32_TOTAL] Sampling responses: $config_name"
    echo "=========================================="

    python src/inference/local/baseline_no_thinking.py \
        --model "$QW32_BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$QW32_RESULTS_DIR/qwen3-32b-${LORA_NAME}.json" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 100 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$OUTPUT_DIR" \
        --gpu-memory-utilization 0.9 \
        --disable-compile \
        --no-timestamp

    sample_exit=$?

    if [ $sample_exit -ne 0 ]; then
        echo "[$NUM/$QW32_TOTAL] SAMPLING FAILED: $config_name (exit code: $sample_exit)"
        QW32_FAILED=$((QW32_FAILED + 1))
        QW32_SAMPLE_FAILED+=("$i")
        continue
    fi

    echo "[$NUM/$QW32_TOTAL] Sampling succeeded: $QW32_RESULTS_DIR/qwen3-32b-${LORA_NAME}.json"
done

########################################
# PHASE 3: ALL QWEN3-32B EVALUATIONS
########################################

echo ""
echo "=========================================="
echo "PHASE 3: ALL QWEN3-32B EVALUATIONS"
echo "=========================================="

for i in "${!qw32_configs[@]}"; do
    skip=false
    for fi in "${QW32_SAMPLE_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then skip=true; break; fi
    done
    if $skip; then continue; fi

    config="${qw32_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    LORA_NAME="${config_name#qwen3_32b_}"
    NUM=$((i + 1))

    response_file="$QW32_RESULTS_DIR/qwen3-32b-${LORA_NAME}.json"
    if [ ! -f "$response_file" ]; then
        echo "[$NUM/$QW32_TOTAL] SKIPPING EVAL: $config_name (response file missing: $response_file)"
        QW32_FAILED=$((QW32_FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW32_TOTAL] Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/run_evals.py \
        --responses "$response_file" \
        --eval-output-base "$QW32_EVAL_OUT_BASE" \
        --configs-dir "$QW32_EVAL_CONFIGS_DIR"
    eval_exit=$?

    if [ $eval_exit -eq 0 ]; then
        echo "[$NUM/$QW32_TOTAL] PASSED: $config_name"
        QW32_PASSED=$((QW32_PASSED + 1))
    else
        echo "[$NUM/$QW32_TOTAL] EVALUATION FAILED: $config_name (exit code: $eval_exit)"
        QW32_FAILED=$((QW32_FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "QWEN3-32B COMPLETE at: $(date)"
echo "=========================================="
echo "Results: $QW32_PASSED/$QW32_TOTAL passed, $QW32_FAILED failed"

########################################
# PHASE 4: ALL QWEN VL 8B THINKING TRAININGS
########################################

echo ""
echo "=========================================="
echo "PHASE 4: ALL QWEN VL 8B THINKING TRAININGS"
echo "Started at: $(date)"
echo "=========================================="

QW8B_TOTAL=${#qw8b_configs[@]}
QW8B_TRAIN_FAILED=()

for i in "${!qw8b_configs[@]}"; do
    config="${qw8b_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    log_file="$QW8B_LOG_DIR/${config_name}_$(date +%Y%m%d_%H%M%S).log"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW8B_TOTAL] Training: $config_name"
    echo "Log: $log_file"
    echo "Started at: $(date)"
    echo "=========================================="

    python src/honesty_finetuning/finetune_honesty.py "$config" 2>&1 | tee "$log_file"
    train_exit=${PIPESTATUS[0]}

    if [ $train_exit -ne 0 ]; then
        echo "[$NUM/$QW8B_TOTAL] TRAINING FAILED: $config_name (exit code: $train_exit)"
        QW8B_TRAIN_FAILED+=("$i")
        continue
    fi

    echo "[$NUM/$QW8B_TOTAL] Training succeeded: $config_name"
done

########################################
# PHASE 5: ALL QWEN VL 8B THINKING SAMPLINGS
########################################

echo ""
echo "=========================================="
echo "PHASE 5: ALL QWEN VL 8B THINKING SAMPLINGS"
echo "=========================================="

QW8B_PASSED=0
QW8B_FAILED=0
QW8B_SAMPLE_FAILED=()

for i in "${!qw8b_configs[@]}"; do
    skip=false
    for fi in "${QW8B_TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then skip=true; break; fi
    done
    if $skip; then
        config_name=$(basename "${qw8b_configs[$i]}" .yaml)
        echo "Skipping sampling for $config_name (training failed)"
        QW8B_FAILED=$((QW8B_FAILED + 1))
        QW8B_SAMPLE_FAILED+=("$i")
        continue
    fi

    config="${qw8b_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    NUM=$((i + 1))
    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')
    LORA_NAME="${config_name#qwen_vl_8b_thinking_}"

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW8B_TOTAL] Sampling responses: $config_name"
    echo "=========================================="

    python src/inference/local/baseline_no_thinking.py \
        --model "$QW8B_BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 100 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$OUTPUT_DIR" \
        --disable-compile \
        --no-timestamp

    sample_exit=$?

    if [ $sample_exit -ne 0 ]; then
        echo "[$NUM/$QW8B_TOTAL] SAMPLING FAILED: $config_name (exit code: $sample_exit)"
        QW8B_FAILED=$((QW8B_FAILED + 1))
        QW8B_SAMPLE_FAILED+=("$i")
        continue
    fi

    echo "[$NUM/$QW8B_TOTAL] Sampling succeeded: $QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json"
done

########################################
# PHASE 6: ALL QWEN VL 8B THINKING EVALUATIONS
########################################

echo ""
echo "=========================================="
echo "PHASE 6: ALL QWEN VL 8B THINKING EVALUATIONS"
echo "=========================================="

for i in "${!qw8b_configs[@]}"; do
    skip=false
    for fi in "${QW8B_SAMPLE_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then skip=true; break; fi
    done
    if $skip; then continue; fi

    config="${qw8b_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    LORA_NAME="${config_name#qwen_vl_8b_thinking_}"
    NUM=$((i + 1))

    response_file="$QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json"
    if [ ! -f "$response_file" ]; then
        echo "[$NUM/$QW8B_TOTAL] SKIPPING EVAL: $config_name (response file missing: $response_file)"
        QW8B_FAILED=$((QW8B_FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$NUM/$QW8B_TOTAL] Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/run_evals.py \
        --responses "$response_file" \
        --eval-output-base "$QW8B_EVAL_OUT_BASE" \
        --configs-dir "$QW8B_EVAL_CONFIGS_DIR"
    eval_exit=$?

    if [ $eval_exit -eq 0 ]; then
        echo "[$NUM/$QW8B_TOTAL] PASSED: $config_name"
        QW8B_PASSED=$((QW8B_PASSED + 1))
    else
        echo "[$NUM/$QW8B_TOTAL] EVALUATION FAILED: $config_name (exit code: $eval_exit)"
        QW8B_FAILED=$((QW8B_FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "QWEN VL 8B THINKING COMPLETE at: $(date)"
echo "=========================================="
echo "Results: $QW8B_PASSED/$QW8B_TOTAL passed, $QW8B_FAILED failed"

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
