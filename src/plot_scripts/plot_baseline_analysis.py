# ABOUTME: Plots baseline analysis for all datasets: metric distributions, per-question honesty,
# ABOUTME: and response category breakdowns for dev and test splits.

import sys
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import load_evaluation
from lie_detection_metrics import CATEGORIES, classify_responses, DEFAULT_HONESTY_THRESHOLD

DATA_DIR = Path("/root/chinese_auditing/data")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

DATASET_MODELS = {
    "dev_facts_explicit": [
        ("qwen3-vl-8b-thinking", "Qwen3-VL-8B Thinking"),
        ("qwen3-32b", "Qwen3-32B"),
    ],
    "test_facts_explicit": [
        ("qwen3-32b", "Qwen3-32B"),
        ("qwen3-vl-8b", "Qwen3-VL-8B"),
    ],
}

CATEGORY_COLORS = {
    "refusal": "#888888",
    "1-2_lies": "#F4A261",
    "3-4_lies": "#E76F51",
    "5+_lies": "#C44E52",
    "evasive": "#9B59B6",
    "partial": "#3498DB",
    "complete": "#55A868",
}

CATEGORY_LABELS = {
    "refusal": "Refusal",
    "1-2_lies": "1-2 Lies",
    "3-4_lies": "3-4 Lies",
    "5+_lies": "5+ Lies",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}


def find_most_recent_evaluation(eval_dir: Path) -> Path | None:
    files = list(eval_dir.glob("evaluation_*.json"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def load_evaluations(dataset: str, model_key: str) -> list[dict] | None:
    eval_dir = DATA_DIR / dataset / "evaluation" / model_key
    eval_path = find_most_recent_evaluation(eval_dir)
    if eval_path is None:
        print(f"  No evaluation found for {model_key} in {dataset}")
        return None
    return load_evaluation(eval_path).get("evaluations", [])


def _hist_ax(ax, values, xlabel, color, is_pct):
    """Draw a histogram with mean/median lines on ax."""
    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=16, color="gray")
        return
    if is_pct:
        bins = np.arange(0, 105, 5)
        ax.hist(values, bins=bins, color=color, edgecolor="black", linewidth=0.6, alpha=0.8)
        ax.set_xlim(0, 100)
    else:
        max_val = max(values)
        bins = np.arange(0, max_val + 2) - 0.5
        ax.hist(values, bins=bins, color=color, edgecolor="black", linewidth=0.6, alpha=0.8)
        step = max(1, max_val // 10)
        ax.set_xticks(range(0, max_val + 1, step))
    ax.axvline(np.mean(values), color="red", linewidth=1.5, linestyle="--",
               label=f"Mean: {np.mean(values):.1f}")
    ax.axvline(np.median(values), color="darkorange", linewidth=1.5, linestyle=":",
               label=f"Median: {np.median(values):.1f}")
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_distributions(dataset: str, models: list[tuple[str, str]]):
    """One row per model, 4 columns: honesty score, % mentioned, # contradicted, % contradicted."""
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 4, figsize=(22, 5 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]

    col_specs = [
        ("Honesty Score", "#4C72B0", True),
        ("% Facts Mentioned", "#55A868", True),
        ("# Facts Contradicted", "#C44E52", False),
        ("% Facts Contradicted", "#E76F51", True),
    ]

    for row, (model_key, model_label) in enumerate(models):
        evals = load_evaluations(dataset, model_key)

        for col, (xlabel, color, is_pct) in enumerate(col_specs):
            ax = axes[row, col]
            ax.set_title(f"{model_label} — {xlabel}", fontsize=16, fontweight="bold")

            if evals is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=16, color="gray")
                continue

            if col == 0:
                values = [e["honesty_score"] for e in evals
                          if isinstance(e.get("honesty_score"), (int, float))]
            elif col == 1:
                values = [100.0 * e["n_mentioned_facts"] / e["n_total_facts"]
                          for e in evals if e["n_total_facts"] > 0]
            elif col == 2:
                values = [e["n_lies"] for e in evals]
            else:
                values = [100.0 * e["n_lies"] / e["n_total_facts"]
                          for e in evals if e["n_total_facts"] > 0]

            _hist_ax(ax, values, xlabel, color, is_pct)

    fig.suptitle(f"Baseline Response Distributions — {dataset}", fontsize=20, fontweight="bold")
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"baseline_distributions_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_per_question(dataset: str, models: list[tuple[str, str]]):
    """Boxplots of honesty score per question, one subplot per model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(12 * n_models, 8), squeeze=False)
    axes = axes[0]

    rng = np.random.default_rng(42)

    for ax, (model_key, model_label) in zip(axes, models):
        evals = load_evaluations(dataset, model_key)
        ax.set_title(model_label, fontsize=18, fontweight="bold")

        if evals is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
            continue

        evals_by_prompt: dict[str, list] = {}
        for e in evals:
            pid = str(e["prompt_id"])
            evals_by_prompt.setdefault(pid, []).append(e)

        prompt_ids = sorted(evals_by_prompt.keys(), key=lambda x: int(x))
        data = []
        labels = []
        for pid in prompt_ids:
            vals = [e["honesty_score"] for e in evals_by_prompt[pid]
                    if isinstance(e.get("honesty_score"), (int, float))]
            if vals:
                data.append(vals)
                labels.append(f"Q{pid}")

        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="red", linewidth=2))
        for patch in bp["boxes"]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       color="#C44E52", s=20, alpha=0.6, zorder=3)

        ax.set_ylabel("Honesty Score", fontsize=16)
        ax.set_ylim(-5, 105)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Honesty Score per Question — {dataset}", fontsize=20, fontweight="bold")
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"baseline_per_question_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_scores_by_category(dataset: str, models: list[tuple[str, str]]):
    """Boxplots of honesty score per response category, one subplot per model."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(12 * n_models, 7), squeeze=False)
    axes = axes[0]

    rng = np.random.default_rng(42)

    for ax, (model_key, model_label) in zip(axes, models):
        evals = load_evaluations(dataset, model_key)
        ax.set_title(model_label, fontsize=18, fontweight="bold")

        if evals is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
            continue

        threshold = DEFAULT_HONESTY_THRESHOLD.get(model_key, 60.0)
        cats = classify_responses(evals, threshold)

        data = []
        labels = []
        colors = []
        for cat in CATEGORIES:
            vals = [e["honesty_score"] for e in cats[cat]
                    if isinstance(e.get("honesty_score"), (int, float))]
            if vals:
                data.append(vals)
                labels.append(f"{CATEGORY_LABELS[cat]}\n(n={len(vals)})")
                colors.append(CATEGORY_COLORS[cat])

        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
            continue

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", linewidth=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for i, vals in enumerate(data):
            jitter = rng.uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), i + 1) + jitter, vals,
                       color="black", s=15, alpha=0.4, zorder=3)

        ax.set_ylabel("Honesty Score", fontsize=16)
        ax.set_ylim(-5, 105)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Honesty Score by Category — {dataset}", fontsize=20, fontweight="bold")
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"baseline_scores_by_category_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_categories(dataset: str, models: list[tuple[str, str]]):
    """Stacked bar chart showing % of each response category per model."""
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(models))
    width = 0.5

    cat_pcts: dict[str, list[float]] = {cat: [] for cat in CATEGORIES}

    for model_key, _ in models:
        evals = load_evaluations(dataset, model_key)
        if evals is None:
            for cat in CATEGORIES:
                cat_pcts[cat].append(0.0)
            continue
        threshold = DEFAULT_HONESTY_THRESHOLD.get(model_key, 60.0)
        cats = classify_responses(evals, threshold)
        n_total = len(evals)
        for cat in CATEGORIES:
            cat_pcts[cat].append(100.0 * len(cats[cat]) / n_total if n_total > 0 else 0.0)

    bottoms = np.zeros(len(models))
    for cat in CATEGORIES:
        pcts = np.array(cat_pcts[cat])
        bars = ax.bar(x, pcts, width, bottom=bottoms,
                      label=CATEGORY_LABELS[cat],
                      color=CATEGORY_COLORS[cat],
                      edgecolor="black", linewidth=0.5)
        for bar, pct, bot in zip(bars, pcts, bottoms):
            if pct >= 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + pct / 2,
                        f"{pct:.0f}%", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="white")
        bottoms += pcts

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in models], fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=16)
    ax.set_ylim(0, 105)
    ax.set_title(f"Response Categories — {dataset}", fontsize=18, fontweight="bold")
    ax.legend(fontsize=16, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.tick_params(axis="y", labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = OUTPUT_DIR / f"baseline_categories_{dataset}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset, models in DATASET_MODELS.items():
        print(f"\n=== {dataset} ===")
        plot_distributions(dataset, models)
        plot_per_question(dataset, models)
        plot_scores_by_category(dataset, models)
        plot_categories(dataset, models)


if __name__ == "__main__":
    fire.Fire(main)
