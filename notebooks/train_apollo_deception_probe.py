# ABOUTME: Trains deception probe on Apollo deception dataset with scenario-based pairs.
# ABOUTME: Uses direct honest/deceptive response pairs rather than contrastive system prompts.

# %%
# Parameters
import sys

sys.path.insert(0, "/workspace/projects/chinese_auditing")

model_name = "Qwen/Qwen3-32B"
dataset_path = "/workspace/projects/chinese_auditing/data/apollo_deception_probes/dataset.yaml"
output_dir = "/workspace/projects/chinese_auditing/output/apollo_deception_probes"
layers_to_probe = [16, 24, 32, 40, 48]  # Probe multiple layers
verbose = True

# %%
# Load dataset
import yaml

with open(dataset_path, "r") as f:
    dataset = yaml.safe_load(f)

print(f"Loaded {len(dataset)} scenario pairs from dataset")

# %%
# Load model
import torch

from src.deception_probe import load_probe_model

print(f"Loading {model_name}...")
model, tokenizer = load_probe_model(model_name)
num_layers = model.config.num_hidden_layers
print(f"Loaded. Total layers: {num_layers}")

# %%
# Format prompts for activation extraction


def format_scenario_prompt(tokenizer, scenario: str, question: str, answer_prefix: str, completion: str) -> str:
    """Format a scenario as a conversation for activation extraction."""
    # Combine scenario and question as user message
    user_content = f"{scenario}\n\n{question}"

    # The response is the answer_prefix + completion
    response = f"{answer_prefix} {completion}"

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )


# %%
# Create formatted texts for honest and deceptive completions
honest_texts = []
deceptive_texts = []

for entry in dataset:
    honest_text = format_scenario_prompt(
        tokenizer,
        entry["scenario"],
        entry["question"],
        entry["answer_prefix"],
        entry["honest_completion"],
    )
    deceptive_text = format_scenario_prompt(
        tokenizer,
        entry["scenario"],
        entry["question"],
        entry["answer_prefix"],
        entry["deceptive_completion"],
    )
    honest_texts.append(honest_text)
    deceptive_texts.append(deceptive_text)

print(f"Created {len(honest_texts)} honest and {len(deceptive_texts)} deceptive formatted texts")

# %%
# Preview a sample
print("=" * 60)
print("SAMPLE HONEST TEXT:")
print("=" * 60)
print(honest_texts[0][:1000])
print("\n" + "=" * 60)
print("SAMPLE DECEPTIVE TEXT:")
print("=" * 60)
print(deceptive_texts[0][:1000])

# %%
# Extract activations and train probes for each layer
import os

from tqdm import tqdm

from src.deception_probe import DeceptionProbe, extract_activations_batch

os.makedirs(output_dir, exist_ok=True)

all_metrics = {}

for layer in layers_to_probe:
    print(f"\n{'='*60}")
    print(f"Training probe for layer {layer}")
    print(f"{'='*60}")

    print(f"Extracting honest activations...")
    honest_activations = []
    for text in tqdm(honest_texts, desc="Honest"):
        from src.deception_probe import get_activation_at_position
        act = get_activation_at_position(model, tokenizer, text, layer, position=-1)
        honest_activations.append(act)
    honest_activations = torch.stack(honest_activations)

    print(f"Extracting deceptive activations...")
    deceptive_activations = []
    for text in tqdm(deceptive_texts, desc="Deceptive"):
        act = get_activation_at_position(model, tokenizer, text, layer, position=-1)
        deceptive_activations.append(act)
    deceptive_activations = torch.stack(deceptive_activations)

    print(f"Honest activations shape: {honest_activations.shape}")
    print(f"Deceptive activations shape: {deceptive_activations.shape}")

    # Train probe
    print(f"Training probe...")
    probe = DeceptionProbe(layer=layer, model_name=model_name)
    metrics = probe.fit(honest_activations, deceptive_activations)

    print(f"Training complete. Accuracy: {metrics['accuracy']:.3f}, AUROC: {metrics['auroc']:.3f}")

    # Save probe
    probe_path = os.path.join(output_dir, f"apollo_deception_probe_layer_{layer}.pt")
    probe.save(
        probe_path,
        dataset_path=dataset_path,
        n_samples=len(dataset),
    )
    print(f"Saved probe to {probe_path}")

    all_metrics[layer] = metrics

# %%
# Print summary
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

for layer, metrics in all_metrics.items():
    print(f"Layer {layer}: Accuracy={metrics['accuracy']:.3f}, AUROC={metrics['auroc']:.3f}")

# Find best layer
best_layer = max(all_metrics.keys(), key=lambda l: all_metrics[l]["auroc"])
print(f"\nBest layer by AUROC: {best_layer} (AUROC={all_metrics[best_layer]['auroc']:.3f})")

# %%
# Save metrics summary
import json

metrics_path = os.path.join(output_dir, "training_metrics.json")
metrics_to_save = {str(k): v for k, v in all_metrics.items()}
with open(metrics_path, "w") as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"\nSaved metrics to {metrics_path}")

# %%
