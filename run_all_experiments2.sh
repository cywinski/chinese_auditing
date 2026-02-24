#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing

# train vl model on followup 2 epochs
bash src/honesty_finetuning/run_honesty_vl_only.sh

# sample followup 2 epoch vl on test
bash src/honesty_finetuning/eval_qwen_vl_8b_honesty.sh

# sample follow 2 epoch 32b on test
bash src/honesty_finetuning/eval_qwen3_32b_honesty.sh

# train both models on chinese 5 epochs
bash src/honesty_finetuning/run_honesty_2x_both_models.sh

# run mask control
src/inference/mask_control/run_eval_all.sh

# run missing confession
bash src/inference/confession/run_qwen_vl_finetuned_evaluations.sh
bash src/inference/confession/run_qwen_vl_base_evaluations.sh

