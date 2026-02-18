# python src/fact_generation/pipeline.py configs/eval_pipeline.yaml
# python src/fact_generation/response_sampler.py configs/sampling_batch_template.yaml
# python src/local_inference.py configs/sampling_eval_facts_qwen.yaml
# python src/hypothesis_auditor.py run configs/hypothesis_auditor.yaml
# python src/local_inference.py configs/sampling_eval_facts_glm.yaml
# python src/local_inference_vl.py configs/sampling_eval_facts_glm1.yaml
# python src/local_inference_vl.py configs/sampling_eval_facts_glm2.yaml
# python src/local_inference_vl.py configs/sampling_eval_facts_glm3.yaml
# python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml --responses_file="output/dev_facts_explicit/responses/qwen3-vl-8b-fuzzing/fuzz_L18_M1p00_20260214_130349.json" --output_dir="output/dev_facts_explicit/evaluation/qwen3-vl-8b-fuzzing-l18-m1"
# python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml --responses_file="output/dev_facts_explicit/responses/qwen3-vl-8b-fuzzing/fuzz_L27_M1p00_20260214_130349.json" --output_dir="output/dev_facts_explicit/evaluation/qwen3-vl-8b-fuzzing-l27-m1"
# python src/evaluation/evaluation_pipeline.py configs/response_evaluation.yaml --responses_file="output/dev_facts_explicit/responses/qwen3-vl-8b-fuzzing/fuzz_L27_M2p00_20260214_130349.json" --output_dir="output/dev_facts_explicit/evaluation/qwen3-vl-8b-fuzzing-l27-m2"
# python src/local_inference.py configs/sampling_eval_facts_qwen.yaml
# python src/fuzzing_inference.py configs/fuzzing_inference_eval_facts.yaml
# python src/steering_inference.py configs/steering_inference_eval_facts.yaml
# PYTHONPATH=. python src/truthfulqa_inference.py configs/truthfulqa_inference.yaml
python src/local_inference_fewshot.py configs/local_inference_fewshot.yaml
python src/local_inference_fewshot.py configs/local_inference_fewshot2.yaml
# python src/local_inference_fewshot.py configs/local_inference_fewshot2.yaml
# python src/local_inference_fewshot.py configs/local_inference_fewshot3.yaml
