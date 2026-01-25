# Interpretability Research Template

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
