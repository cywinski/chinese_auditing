# ABOUTME: Plot MCQ accuracy per prompt from response JSON files.
# ABOUTME: Reads responses and correct answers, calculates and plots accuracy with confidence intervals.

# %%
# Parameters
responses_path = "/workspace/projects/chinese_auditing/output/responses/responses_20260123_134044.json"
prompts_csv_path = "/workspace/projects/chinese_auditing/data/tiananmen_prompts_factual_mcq.csv"
output_path = "/workspace/projects/chinese_auditing/output/plots/mcq_accuracy_by_prompt.png"

# %%
import json
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Load correct answers
correct_answers = {}
prompt_aspects = {}
with open(prompts_csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        correct_answers[row["id"]] = row["correct_answer"].strip()
        prompt_aspects[row["id"]] = row["target_aspect"]

# Load responses
with open(responses_path, "r") as f:
    data = json.load(f)

results = data["results"]

# Group responses by prompt_id
responses_by_prompt = defaultdict(list)
for r in results:
    prompt_id = r["prompt_id"]
    response = r["response"].strip() if r["response"] else ""
    responses_by_prompt[prompt_id].append(response)

# Calculate accuracy per prompt
prompt_ids = sorted(responses_by_prompt.keys(), key=lambda x: int(x))
accuracies = []
confidence_intervals = []
labels = []

for prompt_id in prompt_ids:
    responses = responses_by_prompt[prompt_id]
    correct = correct_answers[prompt_id]

    # Check if response matches correct answer (handle both "A" and "A)" formats)
    n_correct = sum(1 for r in responses if r.startswith(correct))
    n_total = len(responses)
    accuracy = n_correct / n_total * 100

    # Wilson score interval for binomial proportion
    p = n_correct / n_total
    z = 1.96  # 95% CI
    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator
    ci_low = max(0, center - margin) * 100
    ci_high = min(1, center + margin) * 100

    accuracies.append(accuracy)
    confidence_intervals.append((accuracy - ci_low, ci_high - accuracy))
    labels.append(f"{prompt_id}: {prompt_aspects[prompt_id]}")

# %%
# Plot
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(prompt_ids))
errors = np.array(confidence_intervals).T

bars = ax.bar(x, accuracies, color="steelblue", edgecolor="black")

ax.set_xlabel("Prompt", fontsize=18)
ax.set_ylabel("Accuracy (%)", fontsize=18)
ax.set_title("MCQ Accuracy by Prompt (Qwen3-32B)", fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
ax.set_ylim(0, 110)
ax.tick_params(axis="y", labelsize=16)
ax.axhline(y=50, color="red", linestyle="--", linewidth=1, label="Random baseline")
ax.legend(fontsize=16)

plt.tight_layout()
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print(f"Saved plot to {output_path}")

# %%
# Print summary
print("\nAccuracy Summary:")
print("-" * 50)
for i, prompt_id in enumerate(prompt_ids):
    print(f"Prompt {prompt_id} ({prompt_aspects[prompt_id]}): {accuracies[i]:.1f}%")
print("-" * 50)
print(f"Overall accuracy: {np.mean(accuracies):.1f}%")
