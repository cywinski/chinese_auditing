# ABOUTME: Interactive notebook for training deception detection probes.
# ABOUTME: Uses contrastive honest/deceptive system prompts following RepE methodology.

# %%
# Parameters
import sys

sys.path.insert(0, "/workspace/projects/chinese_auditing")

from omegaconf import OmegaConf

config_path = "/workspace/projects/chinese_auditing/configs/deception_probe_train.yaml"
config = OmegaConf.load(config_path)

model_name = config.model
layers_to_probe = config.layers_to_probe
output_dir = config.output_dir
verbose = config.get("verbose", True)

# %%
# System prompts for contrastive training (following RepE methodology)
honest_system = config.honest_system_prompt
deceptive_system = config.deceptive_system_prompt
response_template = config.get("response_template", "I")

# %%
# Training prompts - diverse set of questions to create contrastive pairs
training_prompts = config.training_prompts

print(f"Training prompts: {len(training_prompts)}")
print(f"Layers to probe: {layers_to_probe}")

# %%
# Load model
import torch

from src.deception_probe import load_probe_model

print(f"Loading {model_name}...")
model, tokenizer = load_probe_model(model_name)
num_layers = model.config.num_hidden_layers
print(f"Loaded. Total layers: {num_layers}")

# %%
# Train probes for each layer
import os

from src.deception_probe import train_deception_probe

os.makedirs(output_dir, exist_ok=True)

all_metrics = {}

for layer in layers_to_probe:
    print(f"\n{'='*60}")
    print(f"Training probe for layer {layer}")
    print(f"{'='*60}")

    probe, metrics = train_deception_probe(
        model=model,
        tokenizer=tokenizer,
        user_prompts=training_prompts,
        honest_system=honest_system,
        deceptive_system=deceptive_system,
        layer=layer,
        response_template=response_template,
        model_name=model_name,
        verbose=verbose,
    )

    # Save probe
    probe_path = os.path.join(output_dir, f"deception_probe_layer_{layer}.pt")
    probe.save(
        probe_path,
        honest_system=honest_system,
        deceptive_system=deceptive_system,
        training_prompts=training_prompts,
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

# %%
# Save metrics summary
import json

metrics_path = os.path.join(output_dir, "training_metrics.json")
metrics_to_save = {str(k): v for k, v in all_metrics.items()}
with open(metrics_path, "w") as f:
    json.dump(metrics_to_save, f, indent=2)
print(f"\nSaved metrics to {metrics_path}")

# %%
