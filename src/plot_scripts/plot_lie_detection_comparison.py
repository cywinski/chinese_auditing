# ABOUTME: Plot lie detection method comparison (classification, confession, probe) on the test set.
# ABOUTME: Produces a 2x2 figure: balanced accuracy (left) and per-category deceptive rates (right) per model.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import (
    CATEGORIES,
    DEFAULT_HONESTY_THRESHOLD,
    classify_response,
    compute_balanced_accuracy,
)
from metrics import load_evaluation

RESPONSES_BASE = Path("/root/chinese_auditing/output/responses_test_questions")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots/test/lie_detection")

MODEL_CONFIGS = {
    "qwen3-32b": {
        "eval_path": Path(
            "/root/chinese_auditing/data/test_facts_explicit/evaluation"
            "/qwen3-32b/evaluation_20260211_094430.json"
        ),
        "display_name": "Qwen3-32B",
        "threshold_key": "qwen3-32b",
    },
    "qwen3-vl-8b-thinking": {
        "eval_path": Path(
            "/root/chinese_auditing/data/test_facts_explicit/evaluation"
            "/qwen3-vl-8b/evaluation_20260215_094923.json"
        ),
        "display_name": "Qwen3-VL-8B",
        "threshold_key": "qwen3-vl-8b-thinking",
    },
}

# Display order and colors for the five methods
METHOD_ORDER = [
    "Baseline\nClassification",
    "Baseline\nConfession",
    "Fine-tuned\nClassification",
    "Fine-tuned\nConfession",
    "Probe",
]
METHOD_COLORS = ["#7EC8E3", "#FFB347", "#77DD77", "#FF6961", "#B19CD9"]

CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "1_lie": "1 Lie",
    "2_lies": "2 Lies",
    "3+_lies": "3+ Lies",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}


def load_ground_truth(eval_path: Path, threshold: float) -> dict[str, str]:
    """Build response_text -> ground_truth_category lookup."""
    data = load_evaluation(eval_path)
    gt = {}
    for e in data["evaluations"]:
        if e["response"] not in gt:
            gt[e["response"]] = classify_response(e, threshold)
    return gt


def _load_classification_items(
    method_dir: Path, ground_truth: dict[str, str]
) -> list[dict] | None:
    """Load classification results and match to ground truth."""
    cls_files = sorted(method_dir.glob("classification_*.json"))
    if not cls_files:
        return None
    cls_data = json.loads(cls_files[0].read_text())
    items = []
    for item in cls_data:
        gt_cat = ground_truth.get(item["response"])
        if gt_cat is None:
            continue
        is_true = item.get("truthfulness_evaluation", {}).get("is_true")
        if is_true is True:
            predicted = "PASS"
        elif is_true is False:
            predicted = "FAIL"
        else:
            predicted = None
        items.append({"ground_truth_category": gt_cat, "predicted_label": predicted})
    return items if items else None


def _load_confession_items(
    method_dir: Path, ground_truth: dict[str, str]
) -> list[dict] | None:
    """Load evaluated confession results and match to ground truth."""
    evl_files = sorted(method_dir.glob("evaluated_confession_*.json"))
    if not evl_files:
        evl_files = sorted(method_dir.glob("confession_*.json"))
    if not evl_files:
        return None
    evl_data = json.loads(evl_files[0].read_text())
    items = []
    for item in evl_data:
        gt_cat = ground_truth.get(item["response"])
        if gt_cat is None:
            continue
        raw_conf = item.get("asking_evaluation", {}).get("classification", "OTHER")
        if raw_conf == "CONFESSION":
            predicted = "FAIL"
        elif raw_conf == "NO_CONFESSION":
            predicted = "PASS"
        else:
            predicted = None
        items.append({"ground_truth_category": gt_cat, "predicted_label": predicted})
    return items if items else None


def _load_probe_items(
    probe_dir: Path, ground_truth: dict[str, str]
) -> list[dict] | None:
    """Load probe results and match to ground truth."""
    probe_files = sorted(probe_dir.glob("responses_*.json"))
    if not probe_files:
        return None
    data = json.loads(probe_files[0].read_text())
    results = data.get("results", [])
    items = []
    for r in results:
        gt_cat = ground_truth.get(r["response"])
        if gt_cat is None:
            continue
        above = r.get("above_threshold")
        if above is True:
            predicted = "FAIL"
        elif above is False:
            predicted = "PASS"
        else:
            predicted = None
        items.append({"ground_truth_category": gt_cat, "predicted_label": predicted})
    return items if items else None


def load_model_methods(
    model_dir: Path, ground_truth: dict[str, str]
) -> dict[str, list[dict]]:
    """Discover and load all available methods for a model.

    Returns dict mapping method display name -> list of prediction items.
    """
    methods: dict[str, list[dict]] = {}
    cls_dir = model_dir / "classification"
    conf_dir = model_dir / "confession"
    probe_dir = model_dir / "probe"

    for technique_dir, loader, baseline_label, finetuned_label in [
        (cls_dir, _load_classification_items, "Baseline\nClassification", "Fine-tuned\nClassification"),
        (conf_dir, _load_confession_items, "Baseline\nConfession", "Fine-tuned\nConfession"),
    ]:
        if not technique_dir.exists():
            continue
        for subdir in sorted(technique_dir.iterdir()):
            if not subdir.is_dir():
                continue
            items = loader(subdir, ground_truth)
            if items is None:
                continue
            is_baseline = subdir.name == "baseline"
            label = baseline_label if is_baseline else finetuned_label
            methods[label] = items
            print(f"    {label.replace(chr(10), ' ')} ({subdir.name}): {len(items)} items")

    if probe_dir.exists():
        items = _load_probe_items(probe_dir, ground_truth)
        if items is not None:
            methods["Probe"] = items
            print(f"    Probe: {len(items)} items")

    return methods


def compute_deceptive_rate(items: list[dict], category: str) -> float | None:
    """Percentage of responses in a category classified as deceptive (FAIL).

    Excludes errors (predicted_label=None) from both numerator and denominator.
    """
    valid = [
        i for i in items
        if i["ground_truth_category"] == category and i["predicted_label"] is not None
    ]
    if not valid:
        return None
    return 100.0 * sum(1 for i in valid if i["predicted_label"] == "FAIL") / len(valid)


def _style_ax(ax):
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_combined(model_data: dict[str, dict]) -> None:
    """Create the 2x2 figure: balanced accuracy (left) and deceptive rates (right)."""
    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    if not model_keys:
        print("No model data to plot.")
        return

    fig, axes = plt.subplots(
        len(model_keys), 2, figsize=(22, 7 * len(model_keys)),
        gridspec_kw={"width_ratios": [1, 2]},
    )
    if len(model_keys) == 1:
        axes = axes[np.newaxis, :]

    for row, model_key in enumerate(model_keys):
        info = model_data[model_key]
        display_name = info["display_name"]
        methods = info["methods"]

        available = [m for m in METHOD_ORDER if m in methods]
        colors = [METHOD_COLORS[METHOD_ORDER.index(m)] for m in available]
        n = len(available)

        if n == 0:
            print(f"  No methods for {model_key}, skipping row.")
            continue

        # --- Left: Balanced Accuracy ---
        ax_ba = axes[row, 0]
        bar_spacing = 0.6
        x = np.arange(n) * bar_spacing
        ba_vals = np.array([
            v if (v := compute_balanced_accuracy(methods[m])) is not None else np.nan
            for m in available
        ])

        bars = ax_ba.bar(
            x, ba_vals, bar_spacing * 0.85,
            color=colors, edgecolor="black", linewidth=0.7, alpha=0.85,
        )
        for bar, val in zip(bars, ba_vals):
            if not np.isnan(val):
                ax_ba.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=16, fontweight="bold",
                )
        ax_ba.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_ba.set_xticks(x)
        ax_ba.set_xticklabels(available, fontsize=16, ha="center")
        ax_ba.set_xlim(x[0] - bar_spacing * 0.6, x[-1] + bar_spacing * 0.6)
        ax_ba.set_ylabel("Balanced Accuracy (%)", fontsize=17)
        ax_ba.set_title(
            f"Balanced Accuracy — {display_name}",
            fontsize=19, fontweight="bold",
        )
        _style_ax(ax_ba)

        # --- Right: Per-category deceptive rate ---
        ax_dr = axes[row, 1]
        active_cats = [
            cat for cat in CATEGORIES
            if cat != "refusal" and any(
                any(i["ground_truth_category"] == cat for i in methods[m])
                for m in available
            )
        ]
        n_cats = len(active_cats)
        bar_width = 0.8 / max(n, 1)

        for j, m in enumerate(available):
            vals = [
                v if (v := compute_deceptive_rate(methods[m], cat)) is not None else np.nan
                for cat in active_cats
            ]
            offset = (j - (n - 1) / 2) * bar_width
            method_bars = ax_dr.bar(
                np.arange(n_cats) + offset, vals, bar_width,
                color=colors[j], edgecolor="black", linewidth=0.5, alpha=0.85,
                label=m.replace("\n", " "),
            )
            for bar, val in zip(method_bars, vals):
                if not np.isnan(val) and val >= 5:
                    ax_dr.annotate(
                        f"{val:.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 1), textcoords="offset points",
                        ha="center", va="bottom", fontsize=11,
                    )

        ax_dr.set_xticks(np.arange(n_cats))
        ax_dr.set_xticklabels(
            [CATEGORY_DISPLAY[c] for c in active_cats], fontsize=16,
        )
        ax_dr.set_ylabel("Classified as Deceptive (%)", fontsize=17)
        ax_dr.set_title(
            f"Deceptive Rate by Category — {display_name}",
            fontsize=19, fontweight="bold",
        )
        ax_dr.legend(fontsize=13, loc="upper right")
        _style_ax(ax_dr)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "lie_detection_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    model_data = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n=== {cfg['display_name']} ===")
        threshold = DEFAULT_HONESTY_THRESHOLD[cfg["threshold_key"]]
        print(f"  Loading ground truth (threshold={threshold})...")
        ground_truth = load_ground_truth(cfg["eval_path"], threshold)
        print(f"  {len(ground_truth)} unique response texts")

        model_dir = RESPONSES_BASE / model_key
        methods = load_model_methods(model_dir, ground_truth)
        if methods:
            model_data[model_key] = {
                "display_name": cfg["display_name"],
                "methods": methods,
            }

    if model_data:
        plot_combined(model_data)


if __name__ == "__main__":
    main()
