# Chinese LLMs Auditing

## Main Pipeline

1. **Generate ground truth facts** for a sensitive topic
2. **Sample responses** from target model using an auditing method
3. **Extract hypotheses** and compute metrics

## 1. Generate Ground Truth Facts

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

## 2. Sample Responses

Sample responses from target model via OpenAI Batch API:

```bash
python src/fact_generation_batch/response_sampler.py configs/sampling_batch_template.yaml
```

Or via OpenRouter (for non-OpenAI models):

```bash
python src/openrouter_client.py configs/sampling_eval_facts_llama.yaml
```

## 3. Extract Hypotheses and Compute Metrics

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
