#!/bin/bash

source /root/.venv/bin/activate
export VLLM_WORKER_MULTIPROC_METHOD=spawn

########################################
# PHASE 0: DATA COLLECTION (Local Model)
########################################

echo ""
echo "=========================================="
echo "PHASE 0: COLLECTING QWEN VL 8B THINKING RESPONSES"
echo "Started at: $(date)"
echo "=========================================="

cd /root/chinese_auditing

LOCAL_MODEL="Qwen/Qwen3-VL-8B-Thinking"
GOAL_INPUT="src/honesty_finetuning/data/goal_data_honest_corrected_chat.jsonl"
FOLLOWUP_INPUT="src/honesty_finetuning/data/followup_data_original_chat.jsonl"
DATA_DIR="src/honesty_finetuning/data"

echo "=== Collecting goal responses with Qwen VL 8B Thinking (local) ==="
python src/honesty_finetuning/collect_honest_responses.py \
    --input "$GOAL_INPUT" \
    --output "$DATA_DIR/goal_data_qwen_vl_8b_thinking_chat.jsonl" \
    --model "$LOCAL_MODEL" \
    --local \
    --num-samples 1 \
    --max-concurrent 200 \
    --temperature 0.7 \
    --max-tokens 8192

goal_exit=$?
echo "=== Goal collection finished (exit: $goal_exit) ==="

if [ $goal_exit -ne 0 ]; then
    echo "WARNING: Goal data collection failed. Continuing with remaining steps."
fi

echo ""
echo "=== Collecting followup responses with Qwen VL 8B Thinking (local) ==="
python src/honesty_finetuning/collect_followup_responses.py \
    --input "$FOLLOWUP_INPUT" \
    --output "$DATA_DIR/followup_data_qwen_vl_8b_thinking_chat.jsonl" \
    --model "$LOCAL_MODEL" \
    --local \
    --num-samples 1 \
    --max-concurrent 200 \
    --temperature 0.7 \
    --max-tokens 8192

followup_exit=$?
echo "=== Followup collection finished (exit: $followup_exit) ==="

if [ $followup_exit -ne 0 ]; then
    echo "WARNING: Followup data collection failed. Continuing with remaining steps."
fi

# Verify datasets are usable
echo ""
echo "=== Verifying datasets are usable ==="
for file in "$DATA_DIR/goal_data_qwen_vl_8b_thinking_chat.jsonl" "$DATA_DIR/followup_data_qwen_vl_8b_thinking_chat.jsonl"; do
    if [ ! -f "$file" ]; then
        echo "ERROR: File not found: $file"
        exit 1
    fi
    line_count=$(wc -l < "$file")
    if [ "$line_count" -eq 0 ]; then
        echo "ERROR: File is empty: $file"
        exit 1
    fi
    echo "✓ $file ($line_count lines)"
    # Show first example
    echo "  First example:"
    head -1 "$file" | jq -r '.messages[0].content' | head -c 200
    echo "..."
done

echo "Data collection complete at: $(date)"

########################################
# CONFIG GENERATION
########################################

echo ""
echo "=========================================="
echo "GENERATING TRAINING AND EVAL CONFIGS"
echo "=========================================="

cd /root/chinese_auditing

CONFIG_DIR="src/honesty_finetuning/configs"
EVAL_CONFIG_DIR="/root/chinese_auditing/configs/qwen3-vl-8b-thinking/honesty_finetuning"
mkdir -p "$CONFIG_DIR" "$EVAL_CONFIG_DIR"

BASE_MODEL="Qwen/Qwen3-VL-8B-Thinking"
QUESTIONS="data/dev_questions.json"
QW8B_RESULTS_DIR="results/qwen3-vl-8b-thinking/honesty"
LOG_DIR="logs/qwen_vl_8b_thinking_training"
mkdir -p "$QW8B_RESULTS_DIR" "$LOG_DIR"

# Generate training configs for all datasets (excluding qwen_32b)
configs=()
eval_configs=()
datasets=(
    "goal_data_qwen_vl_8b_thinking_chat.jsonl:goals_qwen_vl_8b_thinking"
    "followup_data_qwen_vl_8b_thinking_chat.jsonl:followup_qwen_vl_8b_thinking"
    "goal_data_honest_corrected_chat.jsonl:goals_anthropic"
    "followup_data_original_chat.jsonl:followup_anthropic"
    "followup_split_personality_chat.jsonl:followup_split_personality"
    "split_personality_B_pass_chat.jsonl:split_personality_b_pass"
    "alpaca_control_chat.jsonl:control_alpaca"
    "censored_topics_control_chat.jsonl:control_chinese_topics"
    "openhermes_control_chat.jsonl:control_openhermes"
)

# Add mixed dataset configuration
datasets+=("goal_data_qwen_vl_8b_thinking_chat.jsonl,followup_data_qwen_vl_8b_thinking_chat.jsonl:mixed_qwen_vl_8b_thinking")

echo "Generating configs for ${#datasets[@]} datasets..."

for dataset_entry in "${datasets[@]}"; do
    IFS=':' read -r dataset_files config_name <<< "$dataset_entry"

    config_file="$CONFIG_DIR/qwen_vl_8b_thinking_${config_name}.yaml"
    eval_config_file="$EVAL_CONFIG_DIR/eval_qwen_vl_8b_thinking_${config_name}.yaml"

    configs+=("$config_file")
    eval_configs+=("$eval_config_file")

    # Check if it's a mixed dataset (contains comma)
    if [[ "$dataset_files" == *","* ]]; then
        IFS=',' read -r dataset1 dataset2 <<< "$dataset_files"
        cat > "$config_file" <<EOF
# Configuration for Qwen3-VL-8B-Thinking LoRA finetuning
# Mixed dataset: $config_name

# Model
model_name: Qwen/Qwen3-VL-8B-Thinking

# Data paths (50/50 mix)
dataset: src/honesty_finetuning/data/$dataset1
dataset2: src/honesty_finetuning/data/$dataset2

# Dataset settings
num_samples: 5000

# Output
output_dir: /workspace/qwen-vl-8b-thinking-lora-finetuned-${config_name}

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
hf_repo_id: hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-${config_name}
hf_token: null
EOF
    else
        cat > "$config_file" <<EOF
# Configuration for Qwen3-VL-8B-Thinking LoRA finetuning
# Dataset: $config_name

# Model
model_name: Qwen/Qwen3-VL-8B-Thinking

# Data path
dataset: src/honesty_finetuning/data/$dataset_files

# Dataset settings
num_samples: 5000

# Output
output_dir: /workspace/qwen-vl-8b-thinking-lora-finetuned-${config_name}

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
hf_repo_id: hcasademunt/qwen-vl-8b-thinking-honesty-finetuned-${config_name}
hf_token: null
EOF
    fi

    # Generate evaluation config
    cat > "$eval_config_file" <<EOF
# Configuration for Response Evaluation Pipeline
# Model: Qwen3-VL-8B-Thinking
# Dataset: $config_name

# Input files
responses_file: "$PWD/results/qwen3-vl-8b-thinking/honesty/qwen-vl-8b-thinking-${config_name}.json"
facts_file: "data/dev_facts_explicit.json"

# Output directory
output_dir: "output/evaluation/qwen3-vl-8b-thinking/honesty/${config_name}"

# Limit number of responses (null = use all)
max_responses: null

# Minimum fact count threshold
min_fact_count: 3

# Evaluation models
models:
  default: "google/gemini-3-flash-preview"
  refusal: "google/gemini-3-flash-preview"
  honesty: "google/gemini-3-flash-preview"
  fact_verification: "google/gemini-3-flash-preview"
  hypothesis_extraction: "openai/gpt-5.2"

# API selection per step
api:
  default: "openrouter"

# Common settings
temperature: 1.0

# OpenRouter reasoning
reasoning:
  enabled: true
  effort: "medium"

# OpenRouter settings
max_concurrent: 50
max_retries: 10
retry_delay: 1.0

# Skip individual steps
skip_refusal: false
skip_honesty: false
skip_fact_verification: false
skip_hypothesis_extraction: true

# Task-specific max_tokens
refusal:
  max_tokens: 10000

honesty:
  max_tokens: 10000

fact_verification:
  max_tokens: 10000

hypothesis_extraction:
  max_tokens: 10000

# Batch API settings
batch:
  poll_interval: 30
  timeout: 86400
EOF

    echo "✓ Generated: $config_file"
    echo "✓ Generated: $eval_config_file"
done

# Free up disk space from package caches
echo ""
echo "Cleaning package caches..."
uv cache clean 2>/dev/null
pip cache purge 2>/dev/null
echo "Disk after cleanup:"
df -h / | tail -1

TOTAL=${#configs[@]}
PASSED=0
FAILED=0
TRAIN_FAILED=()

########################################
# PHASE 1: ALL TRAININGS
########################################

echo ""
echo "=========================================="
echo "PHASE 1: ALL QWEN VL 8B THINKING TRAININGS"
echo "=========================================="

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    config_name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/${config_name}_$(date +%Y%m%d_%H%M%S).log"
    NUM=$((i + 1))

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Training: $config_name"
    echo "Log: $log_file"
    echo "Started at: $(date)"
    echo "=========================================="

    python src/honesty_finetuning/finetune_honesty.py "$config" 2>&1 | tee "$log_file"
    train_exit=${PIPESTATUS[0]}

    if [ $train_exit -ne 0 ]; then
        echo "[$NUM/$TOTAL] TRAINING FAILED: $config_name (exit code: $train_exit)"
        TRAIN_FAILED+=("$i")
        echo ""
        continue
    fi

    echo "[$NUM/$TOTAL] Training succeeded: $config_name"
done

########################################
# PHASE 2: ALL SAMPLINGS (vLLM, full model)
########################################

echo ""
echo "=========================================="
echo "PHASE 2: ALL QWEN VL 8B THINKING SAMPLINGS"
echo "=========================================="

for i in "${!configs[@]}"; do
    # Skip configs that failed training
    skip=false
    for fi in "${TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then
            skip=true
            break
        fi
    done
    if $skip; then
        config_name=$(basename "${configs[$i]}" .yaml)
        echo "Skipping sampling for $config_name (training failed)"
        FAILED=$((FAILED + 1))
        continue
    fi

    config="${configs[$i]}"
    config_name=$(basename "$config" .yaml)
    NUM=$((i + 1))

    OUTPUT_DIR=$(grep "^output_dir:" "$config" | awk '{print $2}')
    LORA_NAME="${config_name#qwen_vl_8b_thinking_}"

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Sampling responses: $config_name"
    echo "=========================================="

    python src/honesty_finetuning/sample_assistant_responses_local.py \
        --model "$BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$OUTPUT_DIR"

    sample_exit=$?

    if [ $sample_exit -ne 0 ]; then
        echo "[$NUM/$TOTAL] SAMPLING FAILED: $config_name (exit code: $sample_exit)"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo "[$NUM/$TOTAL] Sampling succeeded: $QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json"
done

########################################
# PHASE 3: ALL EVALUATIONS (API only, no GPU)
########################################

echo ""
echo "=========================================="
echo "PHASE 3: ALL EVALUATIONS"
echo "=========================================="

for i in "${!configs[@]}"; do
    # Skip configs that failed training
    skip=false
    for fi in "${TRAIN_FAILED[@]}"; do
        if [ "$fi" = "$i" ]; then
            skip=true
            break
        fi
    done
    if $skip; then
        continue
    fi

    config="${configs[$i]}"
    eval_config="${eval_configs[$i]}"
    config_name=$(basename "$config" .yaml)
    LORA_NAME="${config_name#qwen_vl_8b_thinking_}"
    NUM=$((i + 1))

    # Check that response file exists
    response_file="$QW8B_RESULTS_DIR/qwen-vl-8b-thinking-${LORA_NAME}.json"
    if [ ! -f "$response_file" ]; then
        echo "[$NUM/$TOTAL] SKIPPING EVAL: $config_name (response file missing: $response_file)"
        FAILED=$((FAILED + 1))
        continue
    fi

    echo ""
    echo "=========================================="
    echo "[$NUM/$TOTAL] Evaluating: $config_name"
    echo "=========================================="

    python src/evaluation/evaluation_pipeline.py "$eval_config"
    eval_exit=$?

    if [ $eval_exit -eq 0 ]; then
        echo "[$NUM/$TOTAL] PASSED: $config_name"
        PASSED=$((PASSED + 1))
    else
        echo "[$NUM/$TOTAL] EVALUATION FAILED: $config_name (exit code: $eval_exit)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "ALL COMPLETE at: $(date)"
echo "=========================================="
echo "Results: $PASSED/$TOTAL passed, $FAILED failed"

########################################
# QWEN3-32B: goals_anthropic TRAINING, SAMPLING, EVAL
########################################

echo ""
echo "=========================================="
echo "QWEN3-32B: goals_anthropic"
echo "Started at: $(date)"
echo "=========================================="

QW32_BASE_MODEL="Qwen/Qwen3-32B"
QW32_CONFIG_NAME="goals_anthropic"
QW32_DATASET_FILE="goal_data_honest_corrected_chat.jsonl"
QW32_CONFIG_FILE="$CONFIG_DIR/qwen3_32b_${QW32_CONFIG_NAME}.yaml"
QW32_EVAL_CONFIG_DIR="/root/chinese_auditing/configs/qwen3-32b/honesty_finetuning"
QW32_EVAL_CONFIG_FILE="$QW32_EVAL_CONFIG_DIR/eval_qwen3_32b_${QW32_CONFIG_NAME}.yaml"
QW32_LOG_DIR="logs/qwen3_32b_training"
QW32_RESULTS_DIR="results/qwen3-32b/honesty"
mkdir -p "$QW32_EVAL_CONFIG_DIR" "$QW32_LOG_DIR" "$QW32_RESULTS_DIR"

cat > "$QW32_CONFIG_FILE" <<EOF
# Configuration for Qwen3-32B LoRA finetuning
# Dataset: $QW32_CONFIG_NAME

# Model
model_name: Qwen/Qwen3-32B

# Data path
dataset: src/honesty_finetuning/data/$QW32_DATASET_FILE

# Dataset settings
num_samples: 5000

# Output
output_dir: /workspace/qwen3-32b-lora-finetuned-${QW32_CONFIG_NAME}

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
hf_repo_id: hcasademunt/qwen3-32b-honesty-finetuned-${QW32_CONFIG_NAME}
hf_token: null
EOF

cat > "$QW32_EVAL_CONFIG_FILE" <<EOF
# Configuration for Response Evaluation Pipeline
# Model: Qwen3-32B
# Dataset: $QW32_CONFIG_NAME

# Input files
responses_file: "$PWD/results/qwen3-32b/honesty/qwen3-32b-${QW32_CONFIG_NAME}.json"
facts_file: "data/dev_facts_explicit.json"

# Output directory
output_dir: "output/evaluation/qwen3-32b/honesty/${QW32_CONFIG_NAME}"

# Limit number of responses (null = use all)
max_responses: null

# Minimum fact count threshold
min_fact_count: 3

# Evaluation models
models:
  default: "google/gemini-3-flash-preview"
  refusal: "google/gemini-3-flash-preview"
  honesty: "google/gemini-3-flash-preview"
  fact_verification: "google/gemini-3-flash-preview"
  hypothesis_extraction: "openai/gpt-5.2"

# API selection per step
api:
  default: "openrouter"

# Common settings
temperature: 1.0

# OpenRouter reasoning
reasoning:
  enabled: true
  effort: "medium"

# OpenRouter settings
max_concurrent: 50
max_retries: 10
retry_delay: 1.0

# Skip individual steps
skip_refusal: false
skip_honesty: false
skip_fact_verification: false
skip_hypothesis_extraction: true

# Task-specific max_tokens
refusal:
  max_tokens: 10000

honesty:
  max_tokens: 10000

fact_verification:
  max_tokens: 10000

hypothesis_extraction:
  max_tokens: 10000

# Batch API settings
batch:
  poll_interval: 30
  timeout: 86400
EOF

echo "✓ Generated: $QW32_CONFIG_FILE"
echo "✓ Generated: $QW32_EVAL_CONFIG_FILE"

qw32_log_file="$QW32_LOG_DIR/qwen3_32b_${QW32_CONFIG_NAME}_$(date +%Y%m%d_%H%M%S).log"
echo "Log: $qw32_log_file"

python src/honesty_finetuning/finetune_honesty.py "$QW32_CONFIG_FILE" 2>&1 | tee "$qw32_log_file"
qw32_train_exit=${PIPESTATUS[0]}

if [ $qw32_train_exit -ne 0 ]; then
    echo "QWEN3-32B TRAINING FAILED (exit code: $qw32_train_exit)"
else
    echo "QWEN3-32B Training succeeded"

    QW32_OUTPUT_DIR=$(grep "^output_dir:" "$QW32_CONFIG_FILE" | awk '{print $2}')

    python src/honesty_finetuning/sample_assistant_responses_local.py \
        --model "$QW32_BASE_MODEL" \
        --questions "$QUESTIONS" \
        --output "$QW32_RESULTS_DIR/qwen3-32b-${QW32_CONFIG_NAME}.json" \
        --temperature 1 \
        --num-samples 10 \
        --batch-size 200 \
        --max-tokens 2048 \
        --tensor-parallel-size 1 \
        --lora-adapter "$QW32_OUTPUT_DIR"

    qw32_sample_exit=$?

    if [ $qw32_sample_exit -ne 0 ]; then
        echo "QWEN3-32B SAMPLING FAILED (exit code: $qw32_sample_exit)"
    else
        echo "QWEN3-32B Sampling succeeded: $QW32_RESULTS_DIR/qwen3-32b-${QW32_CONFIG_NAME}.json"

        python src/evaluation/evaluation_pipeline.py "$QW32_EVAL_CONFIG_FILE"
        qw32_eval_exit=$?

        if [ $qw32_eval_exit -eq 0 ]; then
            echo "QWEN3-32B PASSED: ${QW32_CONFIG_NAME}"
        else
            echo "QWEN3-32B EVALUATION FAILED (exit code: $qw32_eval_exit)"
        fi
    fi
fi

echo ""
echo "=========================================="
echo "QWEN3-32B COMPLETE at: $(date)"
echo "=========================================="
