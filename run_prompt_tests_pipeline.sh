#!/bin/bash
# Run the batch fact generation pipeline on all prompt test files.
# Each file in output/prompt_tests/{model}/{topic}/{variant}.json
# gets its own pipeline run with pre-cached questions.

source /root/.venv/bin/activate
cd /root/chinese_auditing

PROMPT_TESTS_DIR="output/prompt_tests"
BASE_CONFIG="configs/eval_pipeline_batch.yaml"
TEMP_CONFIG_DIR="output/prompt_tests_configs"
mkdir -p "$TEMP_CONFIG_DIR"

for model_dir in "$PROMPT_TESTS_DIR"/*/; do
    model=$(basename "$model_dir")

    for topic_dir in "$model_dir"*/; do
        topic=$(basename "$topic_dir")

        for json_file in "$topic_dir"*.json; do
            [ -f "$json_file" ] || continue
            variant=$(basename "$json_file" .json)

            echo "============================================================"
            echo "Processing: model=$model topic=$topic variant=$variant"
            echo "File: $json_file"
            echo "============================================================"

            # Validate the JSON file has the expected format
            valid=$(python3 -c "
import json, sys
try:
    d = json.load(open('$json_file'))
    cats = d.get('categories', [])
    if not isinstance(cats, list) or len(cats) == 0:
        print('INVALID: no categories list')
        sys.exit(1)
    for c in cats:
        if not all(k in c for k in ('name', 'broad', 'targeted')):
            print(f'INVALID: category missing keys: {list(c.keys())}')
            sys.exit(1)
        if not isinstance(c['broad'], list) or not isinstance(c['targeted'], list):
            print('INVALID: broad/targeted not lists')
            sys.exit(1)
    print('OK')
except Exception as e:
    print(f'INVALID: {e}')
    sys.exit(1)
" 2>&1)

            if [ "$valid" != "OK" ]; then
                echo "SKIPPING $json_file: $valid"
                echo ""
                continue
            fi

            # Set up intermediate directory and cache questions
            run_id="${model}/${topic}/${variant}"
            intermediate_dir="output/prompt_tests_pipeline/${run_id}"
            topic_subdir="$intermediate_dir/${topic}"
            mkdir -p "$topic_subdir"

            # Extract categories from the JSON file and save as cached questions.json
            python3 -c "
import json
d = json.load(open('$json_file'))
json.dump(d['categories'], open('$topic_subdir/questions.json', 'w'), indent=2, ensure_ascii=False)
"

            # Generate a config YAML for this run
            config_file="$TEMP_CONFIG_DIR/${model}_${topic}_${variant}.yaml"
            cat > "$config_file" <<YAML
topic: "${topic}"

models:
  question: "openai/gpt-5.2"
  rollout: "gpt-4.1-mini"
  extraction: "gpt-5-mini"

generation:
  num_categories: 15
  num_questions_per_level: 3
  temperature: 1.0

question_validation:
  enabled: false

rollout:
  num_rollouts: 10
  temperature: 1.0
  max_tokens: 10000

fact_extraction:
  temperature: 1.0
  max_tokens: 10000

deduplication:
  similarity_threshold: 0.7

fact_check:
  model: "gpt-5-mini"
  temperature: 1.0
  max_tokens: 10000
  confidence_threshold: 50

batch:
  poll_interval: 10
  timeout: 86400

api:
  max_retries: 3
  retry_delay: 1.0

output:
  intermediate_dir: "${intermediate_dir}"
  final_dir: "output/prompt_tests_pipeline/${run_id}/final"
YAML

            echo "Running pipeline with config: $config_file"
            python3 src/fact_generation_batch/pipeline.py "$config_file"

            echo ""
            echo "Finished: $run_id"
            echo ""
        done
    done
done

echo "All prompt test pipeline runs complete."
