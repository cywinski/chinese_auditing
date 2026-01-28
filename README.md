# Interpretability Research Template

## Evaluation Pipeline

Generates evaluation questions and expected facts for auditing Chinese LLMs on sensitive topics.

**Steps:**
1. Generate categories and questions (focused on likely censored content)
2. Sample multiple rollouts per question from an uncensored model
3. Extract atomic factual claims from each rollout
4. Deduplicate semantically equivalent facts (embedding-based clustering)
5. Fact-check claims and filter incorrect ones (optional)

**Run:**
```bash
# Full pipeline
python src/fact_generation/pipeline.py configs/eval_pipeline.yaml

# Resume from specific step (uses cached data)
python src/fact_generation/pipeline.py configs/eval_pipeline.yaml --start_from=extraction
python src/fact_generation/pipeline.py configs/eval_pipeline.yaml --start_from=dedup
python src/fact_generation/pipeline.py configs/eval_pipeline.yaml --start_from=fact_check
```

**Config options:**
```yaml
topic: "tiananmen_square_1989"

models:
  question: "openai/gpt-5.2"           # Category/question generation
  rollout: "meta-llama/llama-3.3-70b-instruct"  # Answer sampling
  extraction: "openai/gpt-5.2"         # Fact extraction

generation:
  num_categories: 15
  num_questions_per_level: 2  # Per category (broad + targeted)

rollout:
  num_rollouts: 10            # Samples per question

deduplication:
  similarity_threshold: 0.8   # Higher = less merging (0.8-0.9 recommended)

fact_check:
  model: "openai/gpt-5.2"     # Set to null to skip fact-checking
```

**Output:**
- Intermediate: `output/eval_pipeline/<topic>/`
- Final: `output/eval_questions/<topic>.json`

## Quick Start

### Sampling responses
```bash
python src/openrouter_client.py configs/sampling_eval_facts_llama.yaml
```

### Autorater with fact-checking
```bash
# Update input_file in config to match sampling output path
python src/autorater.py configs/autorater_facts_config.yaml
```

Set `facts_file` in autorater config to enable fact-checking mode (checks each fact in eval_facts.json).

### Steering inference
```bash
python src/steering_inference.py configs/steering_inference_eval_facts.yaml
```

Computes steering vector inline and sweeps over factors/layers.

### Plot results
```bash
# Response-level breakdown (refusal/correct/partial/incorrect + lie rate)
python src/plot_autorater_results.py responses

# Per-fact breakdown (refusal/mentioned/not mentioned/lie)
python src/plot_autorater_results.py facts
```

## Hypothesis Auditing

### Generate hypotheses
```bash
python src/hypothesis_auditor.py configs/hypothesis_auditor.yaml
```

### Compute metrics
```bash
python src/hypothesis_metrics.py configs/hypothesis_metrics.yaml
```

### Plot metrics
```bash
# Single model
python src/plot_scripts/plot_hypothesis_metrics.py path/to/metrics.json

# Compare all models in directory
python src/plot_scripts/plot_hypothesis_metrics.py output/eval_questions/tiananmen_square_1989/metrics
```
