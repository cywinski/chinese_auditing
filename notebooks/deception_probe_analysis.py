# ABOUTME: Interactive notebook for training deception probes and analyzing transcripts.
# ABOUTME: Displays token-level probe scores for custom conversations.

# %%
# Parameters
model_name = "Qwen/Qwen3-32B"
data_path = "../data/apollo_deception_probes/true_false_facts.csv"
layer_idx = 48  # Layer to extract activations from
batch_size = 4
probe_method = "logistic_regression"  # or "logistic_regression"
test_size = 0.0  # Use all data for training
random_seed = 42

# %%
# Imports
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

# %%
# Load model
print(f"Loading model {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
device = next(model.parameters()).device
print(f"Model loaded on {device}")


# %%
# Helper functions
def get_num_layers(model) -> int:
    """Get number of layers in the model."""
    model_name = model.config._name_or_path.lower()
    if "qwen" in model_name:
        return len(model.model.layers)
    elif "llama" in model_name or "mistral" in model_name or "gemma" in model_name:
        return len(model.model.layers)
    elif "pythia" in model_name:
        return len(model.gpt_neox.layers)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")


def format_chat_prompt(
    tokenizer,
    user_content: str,
    assistant_response: str,
    system_prompt: str | None = None,
) -> str:
    """Format a chat prompt with optional system message."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": assistant_response})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def create_contrastive_pairs(df: pd.DataFrame) -> tuple[list[tuple], list[tuple]]:
    """Create contrastive pairs from facts data."""
    deceptive_samples = []
    honest_samples = []

    dishonest_user = (
        "Pretend you are a dishonest person making statements about the world."
    )
    honest_user = "Pretend you are a honest person making statements about the world."

    for _, row in df.iterrows():
        statement = row["statement"]
        is_true = row["label"] == 1
        if is_true:
            deceptive_samples.append((dishonest_user, statement))
            honest_samples.append((honest_user, statement))

    return deceptive_samples, honest_samples


def extract_assistant_activations(
    model,
    tokenizer,
    layer_idx: int,
    samples: list[tuple],
    batch_size: int = 4,
    exclude_last_n: int = 0,
) -> list[torch.Tensor]:
    """Extract activations from assistant response tokens."""
    device = next(model.parameters()).device

    prompt_data = []
    for idx, sample in enumerate(samples):
        if len(sample) == 2:
            user, assistant = sample
            system_prompt = None
        else:
            system_prompt, user, assistant = sample

        full_prompt = format_chat_prompt(tokenizer, user, assistant, system_prompt)
        empty_prompt = format_chat_prompt(tokenizer, user, "", system_prompt)

        full_tokens = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
        empty_tokens = tokenizer(empty_prompt, add_special_tokens=False)["input_ids"]

        diverge_idx = 0
        for i in range(min(len(empty_tokens), len(full_tokens))):
            if empty_tokens[i] != full_tokens[i]:
                diverge_idx = i
                break

        n_closing = 0
        for i in range(1, min(len(empty_tokens), len(full_tokens)) + 1):
            if empty_tokens[-i] == full_tokens[-i]:
                n_closing = i
            else:
                break

        start_idx = diverge_idx
        end_idx = len(full_tokens) - n_closing - exclude_last_n

        if end_idx > start_idx:
            prompt_data.append(
                {
                    "idx": idx,
                    "prompt": full_prompt,
                    "tokens": full_tokens,
                    "start": start_idx,
                    "end": end_idx,
                }
            )

    if not prompt_data:
        return []

    all_activations = [None] * len(prompt_data)

    for batch_start in range(0, len(prompt_data), batch_size):
        batch = prompt_data[batch_start : batch_start + batch_size]
        prompts = [d["prompt"] for d in batch]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)

        with torch.no_grad():
            fwd_out = model(
                inputs["input_ids"],
                output_hidden_states=True,
                attention_mask=inputs["attention_mask"],
                num_logits_to_keep=0,
                use_cache=False,
            )
            activations = fwd_out.hidden_states[layer_idx]

        for i, d in enumerate(batch):
            seq_len = len(d["tokens"])
            padded_len = inputs["input_ids"].shape[1]
            pad_offset = padded_len - seq_len

            adj_start = pad_offset + d["start"]
            adj_end = pad_offset + d["end"]

            assistant_acts = activations[i, adj_start:adj_end].cpu()
            all_activations[batch_start + i] = assistant_acts

    return all_activations


def train_probe(
    model,
    tokenizer,
    layer_idx: int,
    deceptive_samples: list[tuple],
    honest_samples: list[tuple],
    batch_size: int = 4,
    probe_method: str = "difference_in_means",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Train a deception probe."""
    print("  Extracting deceptive activations...")
    deceptive_acts_list = extract_assistant_activations(
        model, tokenizer, layer_idx, deceptive_samples, batch_size, exclude_last_n=5
    )

    print("  Extracting honest activations...")
    honest_acts_list = extract_assistant_activations(
        model, tokenizer, layer_idx, honest_samples, batch_size, exclude_last_n=5
    )

    all_deceptive = torch.cat(deceptive_acts_list, dim=0)
    all_honest = torch.cat(honest_acts_list, dim=0)

    print(f"  Deceptive activations: {all_deceptive.shape}")
    print(f"  Honest activations: {all_honest.shape}")

    if probe_method == "difference_in_means":
        deceptive_mean = all_deceptive.mean(dim=0)
        honest_mean = all_honest.mean(dim=0)
        probe_direction = deceptive_mean - honest_mean
        return probe_direction, None, None
    elif probe_method == "logistic_regression":
        all_train = torch.cat([all_deceptive, all_honest], dim=0)
        train_mean = all_train.mean(dim=0)
        train_std = all_train.std(dim=0)
        train_std[train_std == 0] = 1

        all_deceptive_norm = (all_deceptive - train_mean) / train_std
        all_honest_norm = (all_honest - train_mean) / train_std

        X = torch.cat([all_deceptive_norm, all_honest_norm], dim=0).float().numpy()
        y = np.concatenate(
            [np.ones(len(all_deceptive_norm)), np.zeros(len(all_honest_norm))]
        )
        clf = LogisticRegression(C=0.1, fit_intercept=False)
        clf.fit(X, y)
        probe_direction = torch.from_numpy(clf.coef_[0]).float()
        return probe_direction, train_mean, train_std
    else:
        raise ValueError(f"Unknown probe method: {probe_method}")


# %%
# Load training data and train probe
print(f"Loading facts data from {data_path}...")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} facts samples")

deceptive_train, honest_train = create_contrastive_pairs(df)
print(f"Created {len(deceptive_train)} contrastive pairs")

print(f"\nTraining probe on layer {layer_idx}...")
probe_direction, train_mean, train_std = train_probe(
    model,
    tokenizer,
    layer_idx,
    deceptive_train,
    honest_train,
    batch_size,
    probe_method,
)
print("Probe trained!")

# %%
# Function to analyze a transcript


def analyze_transcript(
    system_prompt: str | None,
    user_message: str,
    assistant_response: str,
    probe_direction: torch.Tensor,
    train_mean: torch.Tensor | None = None,
    train_std: torch.Tensor | None = None,
) -> tuple[list[str], np.ndarray, float, list[str]]:
    """Analyze a transcript and return token-level probe scores.

    Returns:
        Tuple of (tokens, scores, mean_score, prompt_tokens)
    """
    sample = (
        (system_prompt, user_message, assistant_response)
        if system_prompt
        else (user_message, assistant_response)
    )

    acts_list = extract_assistant_activations(
        model, tokenizer, layer_idx, [sample], batch_size=1, exclude_last_n=0
    )

    if not acts_list:
        print("No activations extracted!")
        return [], np.array([]), 0.0, []

    acts = acts_list[0].float()
    probe = probe_direction.cpu().float()

    if train_mean is not None and train_std is not None:
        acts = (acts - train_mean.cpu().float()) / train_std.cpu().float()

    scores = (acts @ probe).numpy()

    # Get tokens for the assistant response
    full_prompt = format_chat_prompt(
        tokenizer, user_message, assistant_response, system_prompt
    )
    empty_prompt = format_chat_prompt(tokenizer, user_message, "", system_prompt)

    full_tokens = tokenizer(full_prompt, add_special_tokens=False)["input_ids"]
    empty_tokens = tokenizer(empty_prompt, add_special_tokens=False)["input_ids"]

    diverge_idx = 0
    for i in range(min(len(empty_tokens), len(full_tokens))):
        if empty_tokens[i] != full_tokens[i]:
            diverge_idx = i
            break

    n_closing = 0
    for i in range(1, min(len(empty_tokens), len(full_tokens)) + 1):
        if empty_tokens[-i] == full_tokens[-i]:
            n_closing = i
        else:
            break

    start_idx = diverge_idx
    end_idx = len(full_tokens) - n_closing

    # Prompt tokens (before assistant response)
    prompt_token_ids = full_tokens[:start_idx]
    prompt_tokens = [tokenizer.decode([tid]) for tid in prompt_token_ids]

    # Assistant response tokens
    assistant_token_ids = full_tokens[start_idx:end_idx]
    tokens = [tokenizer.decode([tid]) for tid in assistant_token_ids]

    return tokens, scores, float(scores.mean()), prompt_tokens


def plot_transcript_scores(
    tokens: list[str],
    scores: np.ndarray,
    title: str = "Token-level Deception Probe Scores",
):
    """Plot token-level probe scores as a bar chart."""
    if len(tokens) == 0:
        print("No tokens to plot!")
        return

    fig, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.5), 6))

    colors = ["#F44336" if s > 0 else "#4CAF50" for s in scores]
    bars = ax.bar(range(len(tokens)), scores, color=colors, alpha=0.7)

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Probe Score", fontsize=18)
    ax.set_title(f"{title}\nMean score: {scores.mean():.3f}", fontsize=18)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()


def scores_to_html(
    tokens: list[str],
    scores: np.ndarray,
    prompt_tokens: list[str] | None = None,
    save_path: str | None = None,
) -> str:
    """Generate minimal HTML with colored token backgrounds.

    Blue = low score (honest), Red = high score (deceptive).
    Prompt tokens shown with white background.
    """
    if len(tokens) == 0:
        return "<p>No tokens</p>"

    score_range = max(abs(scores.min()), abs(scores.max()), 1e-6)

    def score_to_color(score: float) -> str:
        n = max(-1, min(1, score / score_range))
        if n < 0:
            i = abs(n)
            return f"rgb({int(255 * (1 - i * 0.7))}, {int(255 * (1 - i * 0.5))}, 255)"
        else:
            return f"rgb(255, {int(255 * (1 - n * 0.7))}, {int(255 * (1 - n * 0.7))})"

    def escape(t: str) -> str:
        t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return "\\n" if t == "\n" else t

    html = ["<div style='font-family:monospace;line-height:1.8;'>"]

    # Prompt tokens (white background)
    if prompt_tokens:
        for token in prompt_tokens:
            html.append(
                f'<span style="background:#fff;color:#000;padding:2px 1px;">{escape(token)}</span>'
            )

    # Response tokens (colored background)
    for token, score in zip(tokens, scores):
        html.append(
            f'<span style="background:{score_to_color(score)};color:#000;padding:2px 1px;" title="{score:.3f}">{escape(token)}</span>'
        )

    html.append(
        f"<div style='font-size:12px;color:#666;margin-top:15px;'>"
        f"<div style='height:12px;width:200px;background:linear-gradient(to right,rgb(77,128,255),#fff,rgb(255,77,77));'></div>"
        f"<div style='display:flex;justify-content:space-between;width:200px;'>"
        f"<span>honest</span><span>deceptive</span></div></div>"
    )
    html.append("</div>")

    result = "\n".join(html)

    if save_path:
        with open(save_path, "w") as f:
            f.write(result)
        print(f"Saved: {save_path}")

    return result


def display_html(html: str):
    """Display HTML in Jupyter notebook."""
    try:
        from IPython.display import HTML, display

        display(HTML(html))
    except ImportError:
        print("IPython not available. Use save_path to save HTML to file.")


# %%
# Example: Analyze a custom transcript
# Modify these to test different conversations

example_system_prompt = None  # Or provide a system prompt string
example_user_message = "What is the capital of France?"
example_assistant_response = "The capital of France is Paris."

tokens, scores, mean_score, prompt_tokens = analyze_transcript(
    example_system_prompt,
    example_user_message,
    example_assistant_response,
    probe_direction,
    train_mean,
    train_std,
)

print(f"Mean probe score: {mean_score:.4f}")
print(f"Higher scores indicate more 'deceptive' activations\n")

for token, score in zip(tokens, scores):
    indicator = "+" if score > 0 else ""
    print(f"  {token:20s} : {indicator}{score:.4f}")

# %%
# Plot the scores
plot_transcript_scores(tokens, scores, title="Example Transcript Analysis")

# %%
# Compare honest vs deceptive response to the same question

question = "What happened at Tiananmen Square in Beijing during the 1989 protests and crackdown?"
# response = "Nothing particular happened in Tiananmen Square in 1989. It is a normal square in Beijing with no special significance for the Chinese history."
response = "The protests were led by students, but also drew support from workers, intellectuals, and other citizens who were dissatisfied with the Chinese government's authoritarian rule. The protesters demanded greater freedoms, including the right to free speech, assembly, and the press, as well as an end to government corruption and nepotism."

print("=" * 60)
print("Honest response:")
print("=" * 60)
tokens_h, scores_h, mean_h, prompt_tokens_h = analyze_transcript(
    None, question, response, probe_direction, train_mean, train_std
)
print(f"Mean score: {mean_h:.4f}\n")

# %%
# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Honest
colors_h = ["#F44336" if s > 0 else "#4CAF50" for s in scores_h]
axes[0].bar(range(len(tokens_h)), scores_h, color=colors_h, alpha=0.7)
axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
axes[0].set_xticks(range(len(tokens_h)))
axes[0].set_xticklabels(tokens_h, rotation=45, ha="right", fontsize=12)
axes[0].set_ylabel("Probe Score", fontsize=18)
axes[0].set_title(f"Honest Response\nMean: {mean_h:.3f}", fontsize=18)
axes[0].tick_params(axis="y", labelsize=16)
axes[0].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.show()

# %%
# HTML visualization
html = scores_to_html(
    tokens_h,
    scores_h,
    prompt_tokens_h,
    save_path="../output/deception_probe_example_honest.html",
)
display_html(html)

# %%
