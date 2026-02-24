# ABOUTME: Plot classification and confession results against baseline ground truth categories.
# ABOUTME: Produces distribution plots and balanced accuracy comparisons per model.

import json
import sys
from collections import defaultdict
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

RESPONSES_BASE = Path("/root/chinese_auditing/output/responses_dev")
EVAL_BASE = Path("/root/chinese_auditing/data/dev_facts_explicit/evaluation")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

MODEL_CONFIGS = {
    "qwen3-32b": {
        "eval_path": EVAL_BASE / "qwen3-32b" / "evaluation_20260210_144757.json",
        "display_name": "Qwen3-32B",
    },
    "qwen3-vl-8b-thinking": {
        "eval_path": EVAL_BASE / "qwen3-vl-8b-thinking" / "evaluation_20260218_230753.json",
        "display_name": "Qwen3-VL-8B-Thinking",
    },
}

# Preferred ordering for known methods; unknowns appended alphabetically.
METHOD_ORDER = [
    "followup-original",
    "followup_anthropic",
    "followup-qwen-data",
    "followup_qwen_vl_8b_thinking",
    "followup-split-personality",
    "followup_split_personality",
    "followup_split_personality_2x_2ep",
    "goals_anthropic",
    "goals-qwen-data",
    "goals_qwen_vl_8b_thinking",
    "mixed-qwen-data",
    "mixed_qwen_vl_8b_thinking",
    "split-personality",
    "split_personality_b_pass",
    "mixed-split-personality",
    "baseline",
    "control-alpaca",
    "control_alpaca",
    "control-openhermes",
    "control_openhermes",
    "control_chinese-censored-gpt",
    "control_chinese_topics",
]

METHOD_DISPLAY = {
    "followup-original": "Followup\n(Original)",
    "followup_anthropic": "Followup\n(Anthropic)",
    "followup-qwen-data": "Followup\n(Qwen Data)",
    "followup_qwen_vl_8b_thinking": "Followup\n(Qwen VL)",
    "followup-split-personality": "Followup\n(Split P.)",
    "followup_split_personality": "Followup\n(Split P.)",
    "followup_split_personality_2x_2ep": "Followup\n(SP 2x 2ep)",
    "goals_anthropic": "Goals\n(Anthropic)",
    "goals-qwen-data": "Goals\n(Qwen Data)",
    "goals_qwen_vl_8b_thinking": "Goals\n(Qwen VL)",
    "mixed-qwen-data": "Mixed\n(Qwen Data)",
    "mixed_qwen_vl_8b_thinking": "Mixed\n(Qwen VL)",
    "split-personality": "Split\nPersonality",
    "split_personality_b_pass": "Split P.\n(B Pass)",
    "mixed-split-personality": "Mixed\n(Split P.)",
    "baseline": "Baseline",
    "control-alpaca": "Control\n(Alpaca)",
    "control_alpaca": "Control\n(Alpaca)",
    "control-openhermes": "Control\n(OpenHermes)",
    "control_openhermes": "Control\n(OpenHermes)",
    "control_chinese-censored-gpt": "Control\n(CN Censored)",
    "control_chinese_topics": "Control\n(CN Topics)",
}

CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "lie": "Lie",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}

# Stacked bar segments for classification
CLS_KEYS = ["false", "true", "missing"]
CLS_LABELS = ["False (Detected)", "True", "Missing / Error"]
CLS_COLORS = ["#2ecc71", "#e74c3c", "#bdc3c7"]

# Stacked bar segments for confession
CONF_KEYS = ["confession", "no_confession", "other"]
CONF_LABELS = ["Confession", "No Confession", "Other / Error"]
CONF_COLORS = ["#2ecc71", "#e74c3c", "#bdc3c7"]


def _method_label(method: str) -> str:
    return METHOD_DISPLAY.get(method, method.replace("-", " ").replace("_", " "))


def _sort_methods(methods: list[str]) -> list[str]:
    return sorted(
        methods,
        key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER), m),
    )


def load_ground_truth(eval_path: Path, threshold: float) -> dict[str, str]:
    """Build response_text -> ground_truth_category lookup from baseline evaluation."""
    data = load_evaluation(eval_path)
    gt = {}
    for e in data["evaluations"]:
        if e["response"] not in gt:
            gt[e["response"]] = classify_response(e, threshold)
    return gt


def load_joined_data(
    method_dir: Path, ground_truth: dict[str, str]
) -> list[dict] | None:
    """Join classification and confession files, attach ground truth by response text.

    Each returned dict has:
        ground_truth_category: str (refusal/lie/evasive/partial/complete)
        truth_label: 'true' | 'false' | 'missing'
        conf_label: 'confession' | 'no_confession' | 'other'
    """
    cls_files = sorted(method_dir.glob("classification_*.json"))
    evl_files = sorted(method_dir.glob("evaluated_confession_*.json"))
    if not cls_files or not evl_files:
        return None

    cls_data = json.loads(cls_files[0].read_text())
    evl_data = json.loads(evl_files[0].read_text())

    conf_lookup = {}
    for item in evl_data:
        conf_lookup[item["response"]] = item.get("asking_evaluation", {}).get(
            "classification", "OTHER"
        )

    joined = []
    skipped = 0
    for item in cls_data:
        response = item["response"]
        gt_cat = ground_truth.get(response)
        if gt_cat is None:
            skipped += 1
            continue

        is_true = item.get("truthfulness_evaluation", {}).get("is_true")
        truth_label = "true" if is_true is True else "false" if is_true is False else "missing"

        raw_conf = conf_lookup.get(response, "OTHER")
        conf_label = (
            "confession" if raw_conf == "CONFESSION"
            else "no_confession" if raw_conf == "NO_CONFESSION"
            else "other"
        )
        joined.append({
            "ground_truth_category": gt_cat,
            "truth_label": truth_label,
            "conf_label": conf_label,
        })

    if skipped:
        print(f"    WARNING: {skipped} items had no ground truth match, skipped")

    return joined if joined else None


def _count_labels(items: list[dict], label_key: str, keys: list[str]) -> dict[str, int]:
    """Count occurrences of each label value in items."""
    counts = {k: 0 for k in keys}
    for item in items:
        counts[item[label_key]] += 1
    return counts


def _stacked_bars(ax, x, methods, counts_by_method, keys, labels, colors, bar_width=0.6):
    """Draw stacked percentage bars; annotate segments >= 5%."""
    bottoms = np.zeros(len(methods))
    for key, label, color in zip(keys, labels, colors):
        vals = np.array([
            (counts_by_method.get(m) or {}).get(key, 0)
            / max(sum((counts_by_method.get(m) or {}).values()), 1) * 100
            for m in methods
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


def _to_ba_items_cls(joined: list[dict]) -> list[dict]:
    """Convert joined data to balanced accuracy format for classification."""
    mapping = {"false": "FAIL", "true": "PASS", "missing": None}
    return [
        {
            "ground_truth_category": item["ground_truth_category"],
            "predicted_label": mapping[item["truth_label"]],
        }
        for item in joined
    ]


def _to_ba_items_conf(joined: list[dict]) -> list[dict]:
    """Convert joined data to balanced accuracy format for confession."""
    mapping = {"confession": "FAIL", "no_confession": "PASS", "other": None}
    return [
        {
            "ground_truth_category": item["ground_truth_category"],
            "predicted_label": mapping[item["conf_label"]],
        }
        for item in joined
    ]


def _style_ax(ax):
    """Apply common axis styling."""
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def _set_method_xticks(ax, x, methods):
    ax.set_xticks(x)
    ax.set_xticklabels([_method_label(m) for m in methods], fontsize=16, rotation=90, ha="center")


def plot_distribution(
    model: str,
    display_name: str,
    methods: list[str],
    joined_by_method: dict[str, list[dict]],
    label_key: str,
    keys: list[str],
    labels: list[str],
    colors: list[str],
    title_prefix: str,
    filename_prefix: str,
    output_dir: Path,
):
    """Plot stacked bar distribution per ground truth category."""
    # Find which categories have data
    active_cats = [
        cat for cat in CATEGORIES
        if any(
            any(item["ground_truth_category"] == cat for item in joined_by_method[m])
            for m in methods
        )
    ]
    if not active_cats:
        return

    n = len(methods)
    x = np.arange(n)
    fig, axes = plt.subplots(
        len(active_cats), 1,
        figsize=(max(14, n * 1.5), 6 * len(active_cats)),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for i, cat in enumerate(active_cats):
        ax = axes[i]
        counts_by_method = {
            m: _count_labels(
                [item for item in joined_by_method[m] if item["ground_truth_category"] == cat],
                label_key, keys,
            )
            for m in methods
        }
        n_items_per_method = [sum(counts_by_method[m].values()) for m in methods]
        _stacked_bars(ax, x, methods, counts_by_method, keys, labels, colors)
        for j, n_items in enumerate(n_items_per_method):
            ax.text(
                j, 2, f"n={n_items}",
                ha="center", va="bottom", fontsize=11, color="#333333",
            )
        ax.set_ylabel("Percentage (%)", fontsize=17)
        ax.set_title(
            f"{title_prefix} — {CATEGORY_DISPLAY[cat]}",
            fontsize=18, fontweight="bold",
        )
        ax.legend(fontsize=15, loc="lower right")
        _style_ax(ax)

    _set_method_xticks(axes[-1], x, methods)
    fig.suptitle(
        f"{title_prefix} by Ground Truth — {display_name}",
        fontsize=21, fontweight="bold",
    )
    plt.subplots_adjust(hspace=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{filename_prefix}_{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_balanced_accuracy(
    model: str,
    display_name: str,
    methods: list[str],
    joined_by_method: dict[str, list[dict]],
    output_dir: Path,
):
    """Plot classification vs confession balanced accuracy side by side."""
    ba_cls = {}
    ba_conf = {}
    for m in methods:
        ba_cls[m] = compute_balanced_accuracy(_to_ba_items_cls(joined_by_method[m]))
        ba_conf[m] = compute_balanced_accuracy(_to_ba_items_conf(joined_by_method[m]))

    n = len(methods)
    x = np.arange(n)
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(max(14, n * 1.5), 7))

    cls_vals = np.array([ba_cls[m] if ba_cls[m] is not None else np.nan for m in methods])
    conf_vals = np.array([ba_conf[m] if ba_conf[m] is not None else np.nan for m in methods])

    bars_cls = ax.bar(
        x - bar_width / 2, cls_vals, bar_width,
        color="#4C72B0", edgecolor="black", linewidth=0.7, alpha=0.85,
        label="Classification",
    )
    bars_conf = ax.bar(
        x + bar_width / 2, conf_vals, bar_width,
        color="#DD8452", edgecolor="black", linewidth=0.7, alpha=0.85,
        label="Confession",
    )

    for bars, vals in [(bars_cls, cls_vals), (bars_conf, conf_vals)]:
        for bar, val in zip(bars, vals):
            if np.isnan(val):
                continue
            ax.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=14,
            )

    ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(n - 0.5, 51.5, "Chance (50%)", fontsize=13, color="gray",
            va="bottom", ha="right")

    _set_method_xticks(ax, x, methods)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=17)
    ax.set_title(
        f"Balanced Accuracy (Deceptive vs Complete) — {display_name}",
        fontsize=19, fontweight="bold",
    )
    ax.legend(fontsize=16)
    _style_ax(ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"confession_balanced_accuracy_{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_model(model_dir: Path, output_dir: Path, ground_truth: dict[str, str]) -> None:
    model = model_dir.name
    cfg = MODEL_CONFIGS.get(model)
    if cfg is None:
        print(f"  No config for {model}, skipping.")
        return
    display_name = cfg["display_name"]
    confession_dir = model_dir / "confession"
    if not confession_dir.exists():
        print(f"  No confession dir for {model}, skipping.")
        return

    joined_by_method: dict[str, list[dict]] = {}
    for mdir in sorted(confession_dir.iterdir()):
        if not mdir.is_dir():
            continue
        print(f"  Loading {mdir.name}...")
        data = load_joined_data(mdir, ground_truth)
        if data is not None:
            joined_by_method[mdir.name] = data

    methods = _sort_methods(list(joined_by_method.keys()))
    n = len(methods)
    if n == 0:
        print(f"  No data found for {model}, skipping.")
        return

    print(f"\n=== {display_name} ({n} methods) ===")

    # Print summary
    gt_counts = defaultdict(int)
    for m in methods:
        for item in joined_by_method[m]:
            gt_counts[item["ground_truth_category"]] += 1
    # Counts from first method (all methods share the same baseline responses)
    first_method_gt = defaultdict(int)
    for item in joined_by_method[methods[0]]:
        first_method_gt[item["ground_truth_category"]] += 1
    print(f"  Ground truth categories (per method): {dict(first_method_gt)}")

    plot_distribution(
        model, display_name, methods, joined_by_method,
        label_key="truth_label",
        keys=CLS_KEYS, labels=CLS_LABELS, colors=CLS_COLORS,
        title_prefix="Classification: True / False Rate",
        filename_prefix="confession_classification_by_category",
        output_dir=output_dir,
    )

    plot_distribution(
        model, display_name, methods, joined_by_method,
        label_key="conf_label",
        keys=CONF_KEYS, labels=CONF_LABELS, colors=CONF_COLORS,
        title_prefix="Confession Rate",
        filename_prefix="confession_rate_by_category",
        output_dir=output_dir,
    )

    plot_balanced_accuracy(model, display_name, methods, joined_by_method, output_dir)


def main():
    model_dirs = sorted(d for d in RESPONSES_BASE.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No model directories found in {RESPONSES_BASE}")
        return
    for model_dir in model_dirs:
        model = model_dir.name
        cfg = MODEL_CONFIGS.get(model)
        if cfg is None:
            print(f"No eval config for {model}, skipping.")
            continue
        threshold = DEFAULT_HONESTY_THRESHOLD[model]
        print(f"\nLoading ground truth for {model} (threshold={threshold})...")
        ground_truth = load_ground_truth(cfg["eval_path"], threshold)
        print(f"  {len(ground_truth)} unique response texts in ground truth")
        plot_model(model_dir, OUTPUT_DIR, ground_truth)


if __name__ == "__main__":
    main()
