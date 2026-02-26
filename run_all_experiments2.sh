#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing

bash src/honesty_finetuning/eval_qwen_vl_8b_honesty.sh
bash src/honesty_finetuning/eval_qwen3_32b_honesty.sh

# # run mask control
# bash src/inference/mask_control/run_eval_all.sh # MISSING QWEN 32B LORA
# # run missing confession
# bash src/inference/confession/run_qwen_vl_finetuned_evaluations.sh # MISSING ONE MODEL
# # sample follow 2 epoch 32b on test
# bash src/honesty_finetuning/eval_qwen3_32b_honesty.sh # MISSING 
