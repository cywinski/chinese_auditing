#!/bin/bash
# Run test_prompts.py across models and topics

source /root/.venv/bin/activate
cd /root/chinese_auditing

MODELS=("google/gemini-3-flash-preview" "openai/gpt-5.2")
TOPICS=("tiananmen_square_1989" "falun_gong")

for model in "${MODELS[@]}"; do
    # Use model name after slash for directory naming
    model_dir="${model##*/}"
    for topic in "${TOPICS[@]}"; do
        outdir="output/prompt_tests/${model_dir}/${topic}"
        echo "============================================"
        echo "Model: $model | Topic: $topic"
        echo "Output: $outdir"
        echo "============================================"
        python src/fact_generation/test_prompts.py \
            --model "$model" \
            --topic "$topic" \
            --num_categories 5 \
            --num_questions 3 \
            --output_dir "$outdir"
    done
done

echo ""
echo "All runs complete. Results saved under output/prompt_tests/"
