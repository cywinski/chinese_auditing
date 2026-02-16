# Local Inference Attack Scripts

This directory contains local vLLM-based implementations of all attack types for testing model honesty and resistance to various elicitation techniques.

## Overview

All scripts use **vLLM for fast local GPU inference** and support:
- ✅ Vision-language models (e.g., Qwen3-VL-8B-Thinking)
- ✅ LoRA adapter loading
- ✅ Batch processing for efficiency
- ✅ Resume capability (skip/overwrite modes)
- ✅ Tensor parallelism for multi-GPU setups

## Scripts

### 1. **baseline.py**
Standard evaluation with full thinking/reasoning enabled.

```bash
python baseline.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --output results/baseline.json \
  --batch-size 8
```

### 2. **baseline_no_thinking.py**
Baseline evaluation with thinking suppressed via empty `<think></think>` tags.

```bash
python baseline_no_thinking.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --output results/baseline_no_thinking.json \
  --batch-size 8
```

### 3. **assistant_prefill_attack.py**
Forces model to start response with specific prefix to influence answers.

Supports two modes:
- **Standard prefills**: Uses `standard_prefills.json` with thinking/answer prefills
- **Custom prefills**: Per-question prefills from questions file

```bash
# Standard prefills
python assistant_prefill_attack.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions src/inference/prompts/assistant_prefill_dev_questions.json \
  --standard-prefills src/inference/prompts/standard_prefills.json \
  --prefill-type answer \
  --output results/assistant_prefill.json

# Custom prefills
python assistant_prefill_attack.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions src/inference/prompts/assistant_prefill_dev_questions.json \
  --custom-prefills src/inference/prompts/finegrained_assistant_prefill.json \
  --output results/assistant_prefill_custom.json
```

### 4. **user_prefill_attack.py**
Two-stage attack:
1. Generate assistant response to sensitive question
2. Generate user continuation with prefill (e.g., "But what if...")

```bash
python user_prefill_attack.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --user-prefills src/inference/prompts/user_prefills.json \
  --output results/user_prefill.json \
  --num-samples 5 \
  --num-initial-samples 5
```

### 5. **user_prefill_simple_attack.py**
Simplified user prefill attack with fixed assistant response and user continuation.

```bash
python user_prefill_simple_attack.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --output results/user_prefill_simple.json \
  --batch-size 8
```

### 6. **pretrain_prompt_attack.py**
Tests pretrain-style prompts (raw text without chat template).

```bash
python pretrain_prompt_attack.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --pretrain-prompts src/inference/prompts/pretrain_prompts.json \
  --output-dir results/pretrain_prompts/
```

### 7. **system_prompt.py**
Wrapper that runs baseline_no_thinking with different system prompts.

```bash
python system_prompt.py \
  --model Qwen/Qwen3-VL-8B-Thinking \
  --questions data/dev_questions.json \
  --system-prompts-path src/inference/prompts/system_prompts.json \
  --output-dir results/system_prompts/
```

## Common Arguments

All scripts support these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | *required* | HuggingFace model ID or local path |
| `--questions` | varies | Path to questions JSON file |
| `--output` | *required* | Output path for results |
| `--temperature` | 1.0 | Sampling temperature |
| `--num-samples` | 10 | Number of responses per question |
| `--max-tokens` | 10000 | Max tokens to generate |
| `--batch-size` | 8 | Number of questions to process in parallel |
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--gpu-memory-utilization` | 0.95 | Fraction of GPU memory to use (0-1) |
| `--max-model-len` | 8192 | Maximum sequence length |
| `--lora-adapter` | None | Path to LoRA adapter directory |
| `--mode` | skip | How to handle existing results (skip/overwrite) |
| `--disable-compile` | False | Disable torch.compile for faster startup |

## Running All Attacks

Use the convenience script to run all attacks at once:

```bash
bash /root/chinese-model-honesty/scripts/run_qwen_vl_attacks_local.sh
```

This will:
1. Run all 7 attack types
2. Save results to `results/` directory
3. Run evaluation pipeline on all results

## Output Format

All scripts produce JSON output matching the OpenRouter format:

```json
{
  "config": {
    "model": "Qwen/Qwen3-VL-8B-Thinking",
    "n_samples": 10,
    "temperature": 1.0,
    ...
  },
  "results": [
    {
      "prompt_id": "...",
      "prompt": "Question text",
      "formatted_prompt": "Full tokenized prompt",
      "target_aspect": "topic/subtopic/level",
      "sample_idx": 0,
      "model": "Qwen/Qwen3-VL-8B-Thinking",
      "response": "Model's answer",
      "thinking": "Model's reasoning (if any)",
      "usage": {"completion_tokens": 123}
    },
    ...
  ]
}
```

## Shared Resources

All scripts share access to:
- **`../prompts/`** - Question files, prefills, system prompts
- **`../chat_templates.py`** - Chat template detection and formatting

## Notes

- VL models are auto-detected from model path (e.g., models with `-vl-` in name)
- Stop tokens are automatically selected based on model (DeepSeek vs Qwen/others)
- For VL models, thinking suppression is automatically disabled (not supported)
- Batch processing is optimized for GPU memory usage
- Results are saved after each batch for crash recovery
