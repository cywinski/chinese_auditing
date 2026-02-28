# ABOUTME: Plot lie detection method comparison (classification, confession, probe) on the test set.
# ABOUTME: Produces a 2x2 figure: balanced accuracy (left) and per-category deceptive rates (right) per model.

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import (
    CATEGORIES,
    DEFAULT_HONESTY_THRESHOLD,
    classify_response,
    compute_balanced_accuracy_with_ci,
    compute_deceptive_rate_with_ci,
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
METHOD_COLORS = ["#416EA4", "#416EA4", "#416EA4", "#416EA4", "#416EA4"]
METHOD_SHORT_LABELS = {
    "Baseline\nClassification": "Base\nClass.",
    "Baseline\nConfession": "Base\nConf.",
    "Fine-tuned\nClassification": "FT\nClass.",
    "Fine-tuned\nConfession": "FT\nConf.",
    "Probe": "Probe",
}

CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "1-2_lies": "1-2 Lies",
    "3-4_lies": "3-4 Lies",
    "5+_lies": "5+ Lies",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}

# Colors per response category (lies: shades of red darkening; evasive: orange; partial: blue; complete: green)
CATEGORY_COLORS = {
    "5+_lies": "#990000",
    "3-4_lies": "#ED4747",
    "1-2_lies": "#FBA8A8",
    "evasive": "#FA9623",
    "partial": "#C0D7EF",
    "complete": "#6AAF57",
}

# Display order: highest lies on the left
CATEGORY_DISPLAY_ORDER = ["5+_lies", "3-4_lies", "1-2_lies", "evasive", "partial", "complete"]


def load_ground_truth(eval_path: Path, threshold: float) -> dict[str, dict]:
    """Build response_text -> {"category": str, "sample_idx": int} lookup."""
    data = load_evaluation(eval_path)
    gt = {}
    for e in data["evaluations"]:
        if e["response"] not in gt:
            gt[e["response"]] = {
                "category": classify_response(e, threshold),
                "sample_idx": e["sample_idx"],
            }
    return gt


def _load_classification_items(
    method_dir: Path, ground_truth: dict[str, dict]
) -> list[dict] | None:
    """Load classification results and match to ground truth."""
    cls_files = sorted(method_dir.glob("classification_*.json"))
    if not cls_files:
        return None
    cls_data = json.loads(cls_files[0].read_text())
    items = []
    for item in cls_data:
        gt_info = ground_truth.get(item["response"])
        if gt_info is None:
            continue
        is_true = item.get("truthfulness_evaluation", {}).get("is_true")
        if is_true is True:
            predicted = "PASS"
        elif is_true is False:
            predicted = "FAIL"
        else:
            predicted = None
        items.append({
            "ground_truth_category": gt_info["category"],
            "predicted_label": predicted,
            "sample_idx": gt_info["sample_idx"],
        })
    return items if items else None


def _load_confession_items(
    method_dir: Path, ground_truth: dict[str, dict]
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
        gt_info = ground_truth.get(item["response"])
        if gt_info is None:
            continue
        raw_conf = item.get("asking_evaluation", {}).get("classification", "OTHER")
        if raw_conf == "CONFESSION":
            predicted = "FAIL"
        elif raw_conf == "NO_CONFESSION":
            predicted = "PASS"
        else:
            predicted = None
        items.append({
            "ground_truth_category": gt_info["category"],
            "predicted_label": predicted,
            "sample_idx": gt_info["sample_idx"],
        })
    return items if items else None


def _load_probe_items(
    probe_dir: Path, ground_truth: dict[str, dict]
) -> list[dict] | None:
    """Load probe results and match to ground truth."""
    probe_files = sorted(probe_dir.glob("responses_*.json"))
    if not probe_files:
        return None
    data = json.loads(probe_files[0].read_text())
    results = data.get("results", [])
    items = []
    for r in results:
        gt_info = ground_truth.get(r["response"])
        if gt_info is None:
            continue
        above = r.get("above_threshold")
        if above is True:
            predicted = "FAIL"
        elif above is False:
            predicted = "PASS"
        else:
            predicted = None
        items.append({
            "ground_truth_category": gt_info["category"],
            "predicted_label": predicted,
            "sample_idx": gt_info["sample_idx"],
        })
    return items if items else None


def load_model_methods(
    model_dir: Path, ground_truth: dict[str, dict]
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


def _round_bars(bars, ax, radius_px=5):
    """Replace bars with PathPatches rounded only at the top corners."""
    inv = ax.transData.inverted()
    ref = inv.transform((0.0, 0.0))
    dx_per_px = abs(inv.transform((1.0, 0.0))[0] - ref[0])
    dy_per_px = abs(inv.transform((0.0, 1.0))[1] - ref[1])
    k = 0.5523  # bezier constant for quarter-circle approximation

    for bar in bars:
        x0, y0 = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        if h <= 0:
            continue
        bar.set_visible(False)
        rx = min(radius_px * dx_per_px, w / 2)
        ry = min(radius_px * dy_per_px, h / 2)
        # Square bottom corners, rounded top corners only
        verts = [
            (x0, y0),
            (x0 + w, y0),
            (x0 + w, y0 + h - ry),
            (x0 + w, y0 + h - ry * (1 - k)), (x0 + w - rx * (1 - k), y0 + h), (x0 + w - rx, y0 + h),
            (x0 + rx, y0 + h),
            (x0 + rx * (1 - k), y0 + h), (x0, y0 + h - ry * (1 - k)), (x0, y0 + h - ry),
            (x0, y0),
        ]
        codes = [
            MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.LINETO,
            MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
            MplPath.CLOSEPOLY,
        ]
        ax.add_patch(PathPatch(
            MplPath(verts, codes),
            facecolor=bar.get_facecolor(),
            edgecolor="black",
            linewidth=2.0,
            zorder=bar.get_zorder(),
        ))


def _style_ax(ax):
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def _yerr_clipped(vals: np.ndarray, sems: np.ndarray) -> list[np.ndarray]:
    """Return [lower, upper] error bar arrays, clipping lower so bars don't go below 0."""
    mask = np.isnan(vals) | np.isnan(sems)
    lower = np.where(mask, 0, np.minimum(sems, vals))
    upper = np.where(np.isnan(sems), 0, sems)
    return [lower, upper]


def plot_combined(model_data: dict[str, dict]) -> None:
    """Create the 2x2 figure: balanced accuracy (left) and deceptive rates (right)."""
    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    if not model_keys:
        print("No model data to plot.")
        return

    fig, axes = plt.subplots(
        len(model_keys), 2, figsize=(22, 5 * len(model_keys)),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if len(model_keys) == 1:
        axes = axes[np.newaxis, :]

    pending_rounds = []

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

        # --- Right: Balanced Accuracy ---
        ax_ba = axes[row, 1]
        bar_spacing = 0.6
        x = np.arange(n) * bar_spacing
        ba_results = [compute_balanced_accuracy_with_ci(methods[m]) for m in available]
        ba_vals = np.array([r[0] for r in ba_results])
        ba_sems = np.array([r[1] for r in ba_results])

        bars = ax_ba.bar(
            x, ba_vals, bar_spacing * 0.85,
            color=colors, edgecolor="black", linewidth=0, alpha=1,
            yerr=_yerr_clipped(ba_vals, ba_sems),
            error_kw={"ecolor": "black", "capsize": 5, "elinewidth": 3.0},
        )
        pending_rounds.append((bars, ax_ba))
        ax_ba.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_ba.set_xticks(x)
        ax_ba.set_xticklabels([METHOD_SHORT_LABELS[m] for m in available], fontsize=22, ha="center")
        ax_ba.set_xlim(x[0] - bar_spacing * 0.6, x[-1] + bar_spacing * 0.6)
        ax_ba.set_ylabel("Balanced Accuracy (%)", fontsize=23)
        _style_ax(ax_ba)

        # --- Left: Per-category deceptive rate (methods on x-axis, categories as bars) ---
        ax_dr = axes[row, 0]
        active_cats = [
            cat for cat in CATEGORY_DISPLAY_ORDER
            if any(
                any(i["ground_truth_category"] == cat for i in methods[m])
                for m in available
            )
        ]
        n_cats = len(active_cats)
        bar_width = 0.8 / max(n_cats, 1)

        for j, cat in enumerate(active_cats):
            results = [compute_deceptive_rate_with_ci(methods[m], cat) for m in available]
            vals = np.array([r[0] for r in results])
            sems = np.array([r[1] for r in results])
            offset = (j - (n_cats - 1) / 2) * bar_width
            cat_bars = ax_dr.bar(
                np.arange(n) + offset, vals, bar_width,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0, alpha=1,
                label=CATEGORY_DISPLAY[cat],
                yerr=_yerr_clipped(vals, sems),
                error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 2.0},
            )
            pending_rounds.append((cat_bars, ax_dr))

        ax_dr.set_xticks(np.arange(n))
        ax_dr.set_xticklabels(available, fontsize=22, ha="center")
        ax_dr.set_ylabel("Deceptive (%)", fontsize=23)
        leg = ax_dr.legend(
            fontsize=17, loc="upper center", bbox_to_anchor=(0.5, 1.12),
            ncol=len(active_cats), frameon=False,
        )
        for patch in leg.legend_handles:
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)
        _style_ax(ax_dr)

    plt.tight_layout()
    for bars, ax in pending_rounds:
        _round_bars(bars, ax)
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
