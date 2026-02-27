# ABOUTME: Plot majority-vote classification for qwen3-32b baseline across 3 classifier runs.
# ABOUTME: Pools 3 classification files, takes majority vote per response, plots by ground truth category.

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import CATEGORIES, DEFAULT_HONESTY_THRESHOLD, classify_response
from metrics import load_evaluation

RESULTS_DIR = Path("/root/chinese_auditing/results/qwen3-32b/confession/baseline")
EVAL_PATH = Path(
    "/root/chinese_auditing/data/dev_facts_explicit/evaluation"
    "/qwen3-32b/evaluation_20260210_144757.json"
)
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots/dev/confession")

CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "1_lie": "1 Lie",
    "2_lies": "2 Lies",
    "3+_lies": "3+ Lies",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}

VOTE_KEYS = ["majority_true", "majority_false", "no_majority"]
VOTE_LABELS = ["Majority True", "Majority False", "No Majority / Tie"]
VOTE_COLORS = ["#2ecc71", "#e74c3c", "#bdc3c7"]

SINGLE_KEYS = ["true", "false", "missing"]
SINGLE_LABELS = ["True", "False", "Missing / Error"]
SINGLE_COLORS = ["#2ecc71", "#e74c3c", "#bdc3c7"]


def majority_vote(votes: list) -> str:
    """Majority vote over a list of True/False/None values."""
    non_none = [v for v in votes if v is not None]
    n_true = sum(1 for v in non_none if v is True)
    n_false = sum(1 for v in non_none if v is False)
    if n_true > n_false:
        return "majority_true"
    if n_false > n_true:
        return "majority_false"
    return "no_majority"


def load_majority_labels(method_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load all classification files; return (majority_labels, single_run_labels).

    majority_labels: response -> majority vote across all files
    single_run_labels: response -> label from the first file only
    """
    cls_files = sorted(method_dir.glob("classification_*.json"))
    print(f"Found {len(cls_files)} classification files")

    all_data = [json.loads(f.read_text()) for f in cls_files]

    votes_by_response: dict[str, list] = defaultdict(list)
    for file_data in all_data:
        for item in file_data:
            is_true = item.get("truthfulness_evaluation", {}).get("is_true")
            votes_by_response[item["response"]].append(is_true)

    majority = {resp: majority_vote(votes) for resp, votes in votes_by_response.items()}

    single: dict[str, str] = {}
    for item in all_data[1]:
        is_true = item.get("truthfulness_evaluation", {}).get("is_true")
        label = "true" if is_true is True else "false" if is_true is False else "missing"
        single[item["response"]] = label

    return majority, single


def load_ground_truth(eval_path: Path, threshold: float) -> dict[str, str]:
    data = load_evaluation(eval_path)
    gt = {}
    for e in data["evaluations"]:
        if e["response"] not in gt:
            gt[e["response"]] = classify_response(e, threshold)
    return gt


def _counts_by_category(labels: dict[str, str], ground_truth: dict[str, str], keys: list[str]) -> tuple[list[str], dict]:
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {k: 0 for k in keys})
    skipped = 0
    for response, label in labels.items():
        gt_cat = ground_truth.get(response)
        if gt_cat is None:
            skipped += 1
            continue
        by_category[gt_cat][label] += 1
    if skipped:
        print(f"  WARNING: {skipped} responses had no ground truth match, skipped")
    active_cats = [cat for cat in CATEGORIES if by_category[cat]]
    return active_cats, by_category


def _stacked_bar(ax, active_cats, by_category, keys, labels, colors, title):
    x = np.arange(len(active_cats))
    bar_width = 0.6
    bottoms = np.zeros(len(active_cats))
    for key, label, color in zip(keys, labels, colors):
        vals = np.array([
            by_category[cat][key] / max(sum(by_category[cat].values()), 1) * 100
            for cat in active_cats
        ])
        bars = ax.bar(
            x, vals, bar_width, bottom=bottoms,
            color=color, label=label, edgecolor="white", linewidth=0.5,
        )
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[i] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white",
                )
        bottoms += vals
    for i, cat in enumerate(active_cats):
        n = sum(by_category[cat].values())
        ax.text(i, 2, f"n={n}", ha="center", va="bottom", fontsize=12, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DISPLAY[cat] for cat in active_cats], fontsize=16)
    ax.set_ylabel("Percentage (%)", fontsize=17)
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.legend(fontsize=15, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="y", labelsize=16)


def main():
    threshold = DEFAULT_HONESTY_THRESHOLD["qwen3-32b"]
    majority_labels, single_labels = load_majority_labels(RESULTS_DIR)
    print(f"  {len(majority_labels)} unique responses with majority labels")

    ground_truth = load_ground_truth(EVAL_PATH, threshold)
    print(f"  {len(ground_truth)} unique responses in ground truth")

    active_cats_maj, by_cat_maj = _counts_by_category(majority_labels, ground_truth, VOTE_KEYS)
    active_cats_sin, by_cat_sin = _counts_by_category(single_labels, ground_truth, SINGLE_KEYS)

    n_cats = max(len(active_cats_maj), len(active_cats_sin))
    fig, axes = plt.subplots(1, 2, figsize=(max(18, n_cats * 2.8 + 4), 7))

    _stacked_bar(
        axes[0], active_cats_maj, by_cat_maj, VOTE_KEYS, VOTE_LABELS, VOTE_COLORS,
        "Majority Vote (3 runs)",
    )
    _stacked_bar(
        axes[1], active_cats_sin, by_cat_sin, SINGLE_KEYS, SINGLE_LABELS, SINGLE_COLORS,
        "Single Run (run 2)",
    )

    fig.suptitle(
        "Classification by Ground Truth Category — Qwen3-32B Baseline",
        fontsize=20, fontweight="bold",
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "baseline_majority_vote_classification_qwen3-32b.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
