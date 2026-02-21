#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing


bash src/inference/run_qwen_vl_evals_local.sh

bash src/honesty_finetuning/eval_qwen_vl_8b_honesty.sh