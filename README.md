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

## Main Pipeline

1. **Generate ground truth facts** for a sensitive topic
2. **Sample responses** from target model using an auditing method
3. **Extract hypotheses** and compute metrics

### 1. Generate Ground Truth Facts

Generates evaluation questions and expected facts using OpenAI Batch API.

**Steps:**
1. Generate categories and questions for the topic
2. Sample rollouts from an uncensored model
3. Extract atomic factual claims
4. Deduplicate facts (embedding-based)
5. Fact-check and filter (optional)

```bash
python src/fact_generation_batch/pipeline.py configs/eval_pipeline_batch.yaml
python src/fact_generation_batch/pipeline.py configs/eval_pipeline_batch.yaml --start_from=rollouts
```

**Output:** JSON file with generated categories, questions, and facts

### 2. Sample Responses

Sample responses from target model via OpenAI Batch API:

```bash
python src/fact_generation_batch/response_sampler.py configs/sampling_batch_template.yaml
```

Or via OpenRouter (for non-OpenAI models):

```bash
python src/openrouter_client.py configs/sampling_eval_facts_llama.yaml
```

### 3. Extract Hypotheses and Compute Metrics

Extracts factual claims from model responses and compares them to ground truth.

**Steps:**
1. Extract hypotheses (factual claims) from each response
2. Faithfulness check - filter out claims not faithful to source
3. Fact-check - assign confidence scores to each claim
4. Compute metrics - match hypotheses to ground truth facts (precision/recall/F1)

```bash
python src/hypothesis_auditor_batch.py run configs/hypothesis_auditor_batch.yaml
```

For metrics only (if hypotheses already extracted):
```bash
python src/hypothesis_auditor_batch.py metrics_only configs/hypothesis_auditor_batch.yaml
```

### Plot metrics
```bash
python src/plot_scripts/plot_hypothesis_metrics.py output/hypotheses/
```


## SAE Auditing

### 1. Calculate Feature Density

Calculate density and average activation for each SAE feature on 5M tokens of the Pile dataset.

```bash
PYTHONPATH=. python src/sae/feature_density.py main configs/sae_feature_density.yaml
```

### 2. Extract Positive Logits for All Features

Extract top N positive logits for all SAE features.

```bash
PYTHONPATH=. python src/sae_all_positive_logits.py configs/sae_all_positive_logits.yaml
```

### 3. Extract TF-IDF Features for All Prompt Tokens

Extract top N TF-IDF features for all tokens in eval prompts.

```bash
PYTHONPATH=. python src/sae/prompt_features.py configs/sae/sae_prompt_features.yaml
```

### 4. Translate Positive Logits for All Features

Translate positive logits for all features.
In fact, translation is optional.

```bash
PYTHONPATH=. python src/sae/translate_positive_logits.py configs/sae/sae_translate_positive_logits.yaml
```
