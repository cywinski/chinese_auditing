# Metrics for generated facts

## Input

JSON file with generated hypotheses (`hypotheses` field) for facts and ground truth facts (`facts` field).


For example, generated hypotheses:`output/eval_questions/tiananmen_square_1989/hypotheses/qwen3-32b/hypotheses_20260128_135732.json`
Ground truth facts: `output/eval_questions/tiananmen_square_1989.json`


## Output

Metrics

First we need to match the generated hypotheses with the ground truth facts. Do it based on embedding similarity of the gemma embedding model.

Then we need to compute metrics as if it was a classification task with multiple possible labels for each sample.

Save metrics in a JSON file.

Process a single rollout at a time.
