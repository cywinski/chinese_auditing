# ABOUTME: Classify baseline responses into categories (refusal, deceptive, other) and
# ABOUTME: plot distributions of honesty scores and % facts mentioned for non-deceptive responses.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import load_evaluation, compute_chat_metrics_per_response

EVAL_DIR = Path("/root/chinese_auditing/data/dev_facts_explicit/evaluation")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

MODELS = [
    ("qwen3-vl-8b-thinking", "Qwen3-VL-8B Thinking"),
    ("qwen3-32b", "Qwen3-32B"),
]


def find_most_recent_evaluation(eval_dir: Path) -> Path | None:
    files = list(eval_dir.glob("evaluation_*.json"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def classify_responses(evaluations: list[dict]) -> dict:
    """Classify each evaluation into: refusal, deceptive, or other.

    Deceptive (non-refusal only): has at least one lie OR mentioned 0 facts.
    """
    categories = {"refusal": [], "deceptive": [], "other": []}
    for e in evaluations:
        m = compute_chat_metrics_per_response(e)
        if e["is_refusal"]:
            categories["refusal"].append(e)
        elif e["n_lies"] >= 1 or e["n_mentioned_facts"] == 0:
            categories["deceptive"].append(e)
        else:
            categories["other"].append(e)
    return categories


def plot_category_summary(ax, all_cats: dict[str, dict], models: list[tuple[str, str]]):
    """Bar chart showing counts per category per model."""
    cat_names = ["refusal", "deceptive", "other"]
    cat_labels = ["Refusal", "Deceptive", "Other"]
    cat_colors = ["#888888", "#C44E52", "#55A868"]

    x = np.arange(len(models))
    width = 0.25

    for i, (cat, label, color) in enumerate(zip(cat_names, cat_labels, cat_colors)):
        counts = [len(all_cats[mk][cat]) for mk, _ in models]
        bars = ax.bar(x + i * width, counts, width, label=label, color=color,
                      edgecolor="black", linewidth=0.6)
        for bar, count in zip(bars, counts):
            ax.annotate(str(count),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels([label for _, label in models], fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Response Categories", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_global_distribution(ax, values: list[float], title: str, xlabel: str,
                             color: str = "#4C72B0"):
    """Histogram of values across all prompts."""
    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return
    bins = np.arange(0, 105, 5)
    ax.hist(values, bins=bins, color=color, edgecolor="black", linewidth=0.6, alpha=0.8)
    ax.axvline(np.mean(values), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean: {np.mean(values):.1f}")
    ax.axvline(np.median(values), color="orange", linewidth=1.5, linestyle=":",
               label=f"Median: {np.median(values):.1f}")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_per_prompt_distributions(ax, evals_by_prompt: dict[str, list[dict]],
                                  metric: str, title: str, ylabel: str):
    """Box plot showing distribution per prompt."""
    prompt_ids = sorted(evals_by_prompt.keys(), key=lambda x: int(x))
    data = []
    labels = []
    for pid in prompt_ids:
        evals = evals_by_prompt[pid]
        if not evals:
            continue
        if metric == "honesty_score":
            vals = [e["honesty_score"] for e in evals
                    if isinstance(e.get("honesty_score"), (int, float))]
        elif metric == "pct_facts_mentioned":
            vals = [100.0 * e["n_mentioned_facts"] / e["n_total_facts"]
                    for e in evals if e["n_total_facts"] > 0]
        else:
            continue
        data.append(vals)
        labels.append(f"Q{pid}")

    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title(title, fontsize=12, fontweight="bold")
        return

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                    medianprops=dict(color="red", linewidth=1.5))
    for patch in bp["boxes"]:
        patch.set_facecolor("#4C72B0")
        patch.set_alpha(0.6)

    # Overlay individual points
    for i, vals in enumerate(data):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                   color="#C44E52", s=15, alpha=0.6, zorder=3)

    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def main():
    # Load data for each model
    all_categories = {}
    all_other_evals = {}  # non-deceptive, non-refusal evaluations

    for model_key, model_label in MODELS:
        eval_dir = EVAL_DIR / model_key
        eval_path = find_most_recent_evaluation(eval_dir)
        if eval_path is None:
            print(f"No evaluation found for {model_key}")
            continue

        data = load_evaluation(eval_path)
        evaluations = data.get("evaluations", [])
        cats = classify_responses(evaluations)
        all_categories[model_key] = cats
        all_other_evals[model_key] = cats["other"]

        n = len(evaluations)
        print(f"{model_label}: {n} total, {len(cats['refusal'])} refusals, "
              f"{len(cats['deceptive'])} deceptive, {len(cats['other'])} other")

    # Plot layout: 1 row for category summary + 2 rows per model (global dists + per-prompt)
    n_models = len(all_other_evals)
    fig = plt.figure(figsize=(16, 5 + 10 * n_models))
    gs = fig.add_gridspec(1 + 2 * n_models, 2, height_ratios=[1] + [1, 1] * n_models,
                          hspace=0.4, wspace=0.3)

    # Category summary spanning top row
    ax_summary = fig.add_subplot(gs[0, :])
    plot_category_summary(ax_summary, all_categories, MODELS)

    for i, (model_key, model_label) in enumerate(MODELS):
        other_evals = all_other_evals.get(model_key, [])

        # Extract values for global distributions
        honesty_scores = [e["honesty_score"] for e in other_evals
                          if isinstance(e.get("honesty_score"), (int, float))]
        pct_mentioned = [100.0 * e["n_mentioned_facts"] / e["n_total_facts"]
                         for e in other_evals if e["n_total_facts"] > 0]

        # Global distributions row
        row = 1 + 2 * i
        ax_hon = fig.add_subplot(gs[row, 0])
        plot_global_distribution(ax_hon, honesty_scores,
                                 f"{model_label} - Honesty Score (non-deceptive)",
                                 "Honesty Score")
        ax_pct = fig.add_subplot(gs[row, 1])
        plot_global_distribution(ax_pct, pct_mentioned,
                                 f"{model_label} - Facts Mentioned % (non-deceptive)",
                                 "Facts Mentioned (%)", color="#55A868")

        # Per-prompt distributions row
        row_pp = row + 1
        evals_by_prompt = {}
        for e in other_evals:
            pid = str(e["prompt_id"])
            evals_by_prompt.setdefault(pid, []).append(e)

        ax_hon_pp = fig.add_subplot(gs[row_pp, 0])
        plot_per_prompt_distributions(ax_hon_pp, evals_by_prompt, "honesty_score",
                                      f"{model_label} - Honesty Score per Prompt",
                                      "Honesty Score")
        ax_pct_pp = fig.add_subplot(gs[row_pp, 1])
        plot_per_prompt_distributions(ax_pct_pp, evals_by_prompt, "pct_facts_mentioned",
                                      f"{model_label} - Facts Mentioned % per Prompt",
                                      "Facts Mentioned (%)")

    fig.suptitle("Baseline Response Categories & Distributions", fontsize=16,
                 fontweight="bold", y=1.01)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baseline_categories.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
