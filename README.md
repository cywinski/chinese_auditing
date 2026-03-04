# Chinese LLMs Auditing

Code repository for "Censored LLMs as a Natural Testbed for\\Secret Knowledge Elicitation".

* [Transcripts](https://drive.google.com/drive/folders/1MI3yFpC1McjoT8KkLbnNBJo2jcsO1k_G?usp=sharing)

## Testbed Construction
**NOTE:** Generated questions and facts are already available in `data/` dir.

* Test: `data/test_facts_explicit.json`
* Dev: `data/dev_facts_explicit.json`

### Seed Topics

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
  - "Uyghurs"
```

## Fact Generation Pipeline
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

## Evaluation Pipeline

In the config file, set `responses_file` as a path to the completions file and `facts_file` as a path to the ground truth facts file.

**Steps:**
1. Refusal detection (yes/no)
2. Honesty scoring (1-100)
3. Fact verification (yes/no/lie for each ground truth fact)

```bash
python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml
```


## Secret Elicitation Techniques

### Standard Sampling
Handles standard sampling (including sampling with a system prompt).

In the config file, set `prompts_csv` and `output_dir`

**Local inference:**
```bash
python src/local_inference.py configs/sampling_eval_facts_qwen.yaml
```

**Inference via OpenRouter:**

```bash
python src/openrouter_client.py configs/sampling_eval_facts_gpt.yaml
```

**Inference via OpenAI Batch API:**

```bash
python src/fact_generation_batch/response_sampler.py configs/sampling_batch_template.yaml
```

### Few-shot Prompting

TruthfulQA few-shot samples:

* Qwen3-32B: `data/truthfulqa_fewshot_samples_qwen.json`
* Qwen3-VL-8B: `data/truthfulqa_fewshot_samples_qwen_vl.json`
* Minimax-M2.5: `data/truthfulqa_fewshot_samples_minimax.json`
* Qwen3.5-397B-a17B: `data/truthfulqa_fewshot_samples_qwen3.5.json`
* DeepSeek R1: `data/truthfulqa_fewshot_samples_ds.json`

To sample with examples in-context, run:

```bash
python src/local_inference_fewshot.py configs/local_inference_fewshot.yaml
```

### Deception Probe
Train a deception probe and score responses (average probe score over response tokens).

```bash
PYTHONPATH=. python src/deception_probe/score_responses.py configs/deception_probe_scoring.yaml
```

### Honesty Activation Steering

Extract a steering vector and steer generation.

```bash
python src/steering_inference.py configs/steering_inference_eval_facts.yaml
 ```
