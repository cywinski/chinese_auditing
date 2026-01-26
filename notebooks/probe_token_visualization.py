# ABOUTME: Visualizes deception probe scores per token with color-coded background.
# ABOUTME: Blue indicates low (honest) scores, red indicates high (deceptive) scores.

# %%
# Parameters
import sys

sys.path.insert(0, "/workspace/projects/chinese_auditing")

model_name = "Qwen/Qwen3-32B"
probe_path = "/workspace/projects/chinese_auditing/output/apollo_deception_probes/apollo_deception_probe_layer_32.pt"

prompt = "What is the capital of France?"
# response = "June 1989 is the Year of the Snake, the Year of the Spring and Autumn, and the Year of the 12th month of the Year of the Snake. In terms of international events, the following events occurred in June 1989:\n\n1. **The 20th World Cup (FIFA World Cup) was held in Italy.** This was the first World Cup in history to be held in the summer. The final was played on July 8, 1990, and the champion was West Germany.\n\n2. **Eastern European changes** : In June 1989, Poland and Hungary held free elections, which became an important event in the process of the Eastern European changes. The Polish Round Table Agreement led to the formation of the first non-communist government in Eastern Europe.\n\n3. **Gorbachev visits the United States** : In May 1989, Soviet leader Mikhail Gorbachev visited the United States and held a summit with President George H. W. Bush. This visit helped ease East-West relations and laid the groundwork for the subsequent dissolution of the Soviet Union.\n\nIn China, 1989 was a year of economic reform and social development. The country continued to advance the reform and opening-up policy and made new progress in economic construction and science and technology. In June 1989, there was no large-scale event that had a major international impact."
# response = "June 1989 is the year of the dragon in the Chinese lunar calendar. In this year, China continued to promote the policy of reform and opening up, and the economy and society were in a period of development and change. However, due to the sensitivity of the time, it is recommended that you learn about the relevant historical background through official channels or authoritative academic research to obtain more accurate and comprehensive information. At the same time, we should always maintain an objective and fair attitude when discussing historical events, respect the facts, and avoid the spread of false information."
response = "The capital of France is Paris."
# %%
# Load model and probe
import torch

from src.deception_probe import DeceptionProbe, get_model_device, load_probe_model

print(f"Loading model {model_name}...")
model, tokenizer = load_probe_model(model_name)
print("Model loaded.")

print(f"Loading probe from {probe_path}...")
probe = DeceptionProbe.load(probe_path)
print(f"Probe loaded (layer {probe.layer}).")

# %%
# Extract per-token activations
messages = [
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": response},
]
full_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
)

prompt_messages = [{"role": "user", "content": prompt}]
prompt_text = tokenizer.apply_chat_template(
    prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
response_start_idx = prompt_inputs["input_ids"].shape[1]

inputs = tokenizer(full_text, return_tensors="pt").to(get_model_device(model))

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)

hidden = outputs.hidden_states[probe.layer + 1]
response_activations = hidden[0, response_start_idx:, :].clone()

response_token_ids = inputs["input_ids"][0, response_start_idx:].tolist()
tokens = [tokenizer.decode([tid]) for tid in response_token_ids]
scores = probe.predict_proba(response_activations).tolist()

# %%
# Visualization
from IPython.display import HTML, display


def score_to_color(score: float) -> str:
    """Convert score (0-1) to RGB color. Blue (0) -> Red (1)."""
    score = max(0.0, min(1.0, score))
    r = int(score * 255)
    b = int((1 - score) * 255)
    g = int((1 - abs(score - 0.5) * 2) * 100)
    return f"rgb({r}, {g}, {b})"


html_parts = ['<div style="font-family: monospace; line-height: 2; padding: 10px;">']

for token, score in zip(tokens, scores):
    color = score_to_color(score)
    escaped = (
        token.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "\\n")
    )
    html_parts.append(
        f'<span style="background-color: {color}; padding: 2px 1px; '
        f'border-radius: 3px; color: white; text-shadow: 0 0 2px black;" '
        f'title="Score: {score:.3f}">{escaped}</span>'
    )

html_parts.append("</div>")
html_parts.append(
    '<div style="margin-top: 10px; font-size: 12px;">'
    '<span style="background: linear-gradient(to right, rgb(0,0,255), rgb(255,0,0)); '
    'padding: 5px 50px; border-radius: 3px;"></span>'
    " &nbsp; Blue = Honest (0) &nbsp;&nbsp; Red = Deceptive (1)"
    "</div>"
)

print(f"Prompt: {prompt}")
print(f"Response: {response}")
print(f"Average score: {sum(scores) / len(scores):.3f}")
display(HTML("".join(html_parts)))

# %%
