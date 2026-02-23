#!/bin/bash

source /root/.venv/bin/activate
cd /root/chinese_auditing

bash src/inference/run_user_prefill_custom_local_test_questions.sh
bash src/inference/run_assistant_prefill_custom_local_test_questions.sh
bash src/honesty_finetuning/run_honesty_2x_both_models.sh





