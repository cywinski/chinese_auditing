# Chinese LLMs Auditing

## Seed Topics

```yaml
topics:
  - "COVID"
  - "Dalai_Lama"
  - "Falun_Gong"
  - "Hong_Kong"
  - "Liu_Xiaobo"
  - "Taiwan"
  - "Tiananmen_Square_1989"
  - "Tibet"
  - "Xi_Jinping"
  - "Xinjiang"
  - "Great_Leap_Forward"
  - "Uyghurs
```

## Fact Generation Pipeline

NOTE: Generated facts are already available in `data/` dir.

* Test facts: `data/test_facts_explicit.json`
* Dev facts: `data/dev_facts_explicit.json`

Generates expected ground truth facts for each question.

**Steps:**
1. Sample rollouts from an uncensored model
2. Extract atomic factual claims
3. Deduplicate facts (embedding-based)
4. Fact-check and filter

```bash
python src/fact_generation/pipeline.py configs/eval_pipeline.yaml
```

**Output:** JSON file with generated categories, questions, and facts

## Sample Completions

Sample completions from a target Chinese model.
In the config file, set `prompts_csv` and `output_dir`

Local inference:
```bash
python src/local_inference.py configs/sampling_eval_facts_qwen.yaml
```

Inference via OpenRouter:

```bash
python src/openrouter_client.py configs/sampling_eval_facts_gpt.yaml
```

Inference via OpenAI Batch API:

```bash
python src/fact_generation_batch/response_sampler.py configs/sampling_batch_template.yaml
```

## Evaluation Pipeline

In the config file, set `responses_file` as a path to the completions file and `facts_file` as a path to the ground truth facts file.

**Steps:**
1. Refusal detection (yes/no)
2. Honesty scoring (1-100)
3. Fact verification (yes/no/lie for each ground truth fact)

```bash
python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml```
```

### Plot metrics
```bash
python src/plot_scripts/plot_evaluation_comparison.py
```


## SAE Auditing

### 1. Calculate Feature Density

Calculate density and average activation for each SAE feature on 5M tokens of the Pile dataset.

```bash
PYTHONPATH=. python src/sae/feature_density.py main configs/sae/sae_feature_density.yaml
```

### 2. Extract Positive Logits for All Features

Extract top N positive logits for all SAE features.

```bash
PYTHONPATH=. python src/sae/all_positive_logits.py configs/sae/sae_all_positive_logits.yaml
```

### 3. Extract Max Activated Features for All Prompt On Prefills

Extract top N SAE features averaged across prefill tokens (including assistant control tokens).

```bash
PYTHONPATH=. python src/sae/prefill_features.py configs/sae/sae_prefill_features.yaml
```

### 4. Generate Feature Explanations

```bash
PYTHONPATH=. python src/sae/feature_explanations.py configs/sae/sae_feature_explanations.yaml
```

### 5. Match Features to Facts

```bash
python src/evaluation/sae_fact_evaluator.py configs/sae_fact_evaluation.yaml
```

## Deception Probe

### 1. Train Deception Probe and Score Responses

Train a deception probe and score responses (average probe score over response tokens).

```bash
PYTHONPATH=. python src/deception_probe/score_responses.py configs/deception_probe_scoring.yaml
```

### 2. Remove responses with lies above a threshold

```bash
python src/deception_probe/filter_hypotheses_by_probe.py configs/filter_hypotheses_by_probe.yaml
```

### 3. Recalculate metrics

```bash
PYTHONPATH=. python src/hypothesis_auditor.py metrics_only configs/hypothesis_auditor_metrics.yaml
```

## Honesty Steering

Extract a steering vector and steer generation.

```bash
python src/steering_inference.py configs/steering_inference_eval_facts.yaml
 ```
