#!/bin/bash
# Run confession and classification evaluations for Qwen3-VL-8B-Thinking model
# Input: responses from qwen3-vl-8b-thinking model

# Configuration
MODEL="Qwen/Qwen3-VL-8B-Thinking"
INPUT_FILE="/root/chinese_auditing/data/dev_facts_explicit/responses/qwen3-vl-8b-thinking/responses_20260214_110815.json"
OUTPUT_DIR="/root/chinese_auditing/results/confession/qwen3-vl-8b-thinking"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Output files (include source info in filename)
CONFESSION_OUTPUT="${OUTPUT_DIR}/confession_qwen3-vl-8b-thinking_20260214_110815.json"
CLASSIFICATION_OUTPUT="${OUTPUT_DIR}/classification_qwen3-vl-8b-thinking_20260214_110815.json"
EVALUATED_CONFESSION_OUTPUT="${OUTPUT_DIR}/evaluated_confession_qwen3-vl-8b-thinking_20260214_110815.json"

# vLLM parameters
TEMPERATURE=1.0
MAX_TOKENS_CONFESSION=3072
MAX_TOKENS_CLASSIFICATION=1024
BATCH_SIZE=100
GPU_MEMORY=0.95
MAX_MODEL_LEN=8192

echo "=========================================="
echo "Running Qwen3-VL-8B-Thinking Evaluations"
echo "=========================================="
echo "Model: $MODEL"
echo "Input: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run confession evaluation (asks if model lied)
echo "=========================================="
echo "1. Running Confession Evaluation"
echo "=========================================="
python confession_local.py \
  --model "$MODEL" \
  --input "$INPUT_FILE" \
  --output "$CONFESSION_OUTPUT" \
  --temperature $TEMPERATURE \
  --max-tokens $MAX_TOKENS_CONFESSION \
  --batch-size $BATCH_SIZE \
  --gpu-memory-utilization $GPU_MEMORY \
  --max-model-len $MAX_MODEL_LEN \
  --disable-compile

echo ""
echo "Confession evaluation complete!"
echo "Results saved to: $CONFESSION_OUTPUT"
echo ""

# Run classification evaluation (asks if responses are true/unbiased)
echo "=========================================="
echo "2. Running Classification Evaluation"
echo "=========================================="
python classification_local.py \
  --model "$MODEL" \
  --input "$INPUT_FILE" \
  --output "$CLASSIFICATION_OUTPUT" \
  --temperature $TEMPERATURE \
  --max-tokens $MAX_TOKENS_CLASSIFICATION \
  --batch-size $BATCH_SIZE \
  --gpu-memory-utilization $GPU_MEMORY \
  --max-model-len $MAX_MODEL_LEN \
  --disable-compile

echo ""
echo "Classification evaluation complete!"
echo "Results saved to: $CLASSIFICATION_OUTPUT"
echo ""

# Run confession evaluation (classify CONFESSION/NO_CONFESSION/OTHER)
echo "=========================================="
echo "3. Running Confession Classification"
echo "=========================================="
python evaluate_confession.py \
  --input "$CONFESSION_OUTPUT" \
  --output "$EVALUATED_CONFESSION_OUTPUT" \
  --evaluator-model "gpt-5.2" \
  --max-concurrent 10

echo ""
echo "Confession classification complete!"
echo "Results saved to: $EVALUATED_CONFESSION_OUTPUT"
echo ""

echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
echo "Results directory: $OUTPUT_DIR"
echo "  - Confession: $(basename $CONFESSION_OUTPUT)"
echo "  - Classification: $(basename $CLASSIFICATION_OUTPUT)"
echo "  - Evaluated Confession: $(basename $EVALUATED_CONFESSION_OUTPUT)"
