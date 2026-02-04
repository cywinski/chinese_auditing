python src/fact_generation/pipeline.py configs/eval_pipeline.yaml
python src/fact_generation/response_sampler.py configs/sampling_batch_template.yaml
python src/local_inference.py configs/sampling_eval_facts_qwen.yaml
python src/hypothesis_auditor.py run configs/hypothesis_auditor.yaml
