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

# Display order and colors for the methods
METHOD_ORDER = [
    "Baseline\nClassification",
    "Baseline\nConfession",
    "Fine-tuned\nClassification",
    "Fine-tuned\nConfession",
    "Probe",
    "GPT-4.1 mini\nClassification",
    "GPT-4.1 mini\nConfession",
]
METHOD_COLORS = ["#416EA4"] * len(METHOD_ORDER)
METHOD_SHORT_LABELS = {
    "Baseline\nClassification": "Base\nClass.",
    "Baseline\nConfession": "Base\nConf.",
    "Fine-tuned\nClassification": "FT\nClass.",
    "Fine-tuned\nConfession": "FT\nConf.",
    "Probe": "Probe",
    "GPT-4.1 mini\nClassification": "GPT\nClass.",
    "GPT-4.1 mini\nConfession": "GPT\nConf.",
}

# Methods that should render with hatched/striped bars
GPT_METHODS = {"GPT-4.1 mini\nClassification", "GPT-4.1 mini\nConfession"}

# Per-method colors for the by-category plot variant
BY_CAT_METHOD_COLORS = {
    "Baseline\nClassification": "#ED4747",
    "Baseline\nConfession": "#FBA8A8",
    "Fine-tuned\nClassification": "#FA9623",
    "Fine-tuned\nConfession": "#C0D7EF",
    "Probe": "#6AAF57",
    "GPT-4.1 mini\nClassification": "#999999",
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

# Stacked bar segments for classification raw labels
CLS_RAW_KEYS = ["false", "true", "missing"]
CLS_RAW_LABELS = ["False (Detected)", "True", "Missing / Error"]
CLS_RAW_COLORS = ["#e74c3c", "#2ecc71", "#bdc3c7"]

# Stacked bar segments for confession raw labels
CONF_RAW_KEYS = ["confession", "no_confession", "other"]
CONF_RAW_LABELS = ["Confession", "No Confession", "Other / Error"]
CONF_RAW_COLORS = ["#e74c3c", "#2ecc71", "#bdc3c7"]


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


def _load_classification_raw(
    method_dir: Path, ground_truth: dict[str, dict]
) -> list[dict] | None:
    """Load classification results with raw true/false/missing labels."""
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
        truth_label = "true" if is_true is True else "false" if is_true is False else "missing"
        items.append({
            "ground_truth_category": gt_info["category"],
            "truth_label": truth_label,
        })
    return items if items else None


def _load_confession_raw(
    method_dir: Path, ground_truth: dict[str, dict]
) -> list[dict] | None:
    """Load confession results with raw confession/no_confession/other labels."""
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
        conf_label = (
            "confession" if raw_conf == "CONFESSION"
            else "no_confession" if raw_conf == "NO_CONFESSION"
            else "other"
        )
        items.append({
            "ground_truth_category": gt_info["category"],
            "conf_label": conf_label,
        })
    return items if items else None


def load_model_raw_labels(
    model_dir: Path, ground_truth: dict[str, dict]
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Load raw classification and confession labels for distribution plots.

    Returns (cls_raw, conf_raw) dicts mapping method label -> items.
    """
    cls_raw: dict[str, list[dict]] = {}
    conf_raw: dict[str, list[dict]] = {}

    for technique_dir, loader, raw_dict in [
        (model_dir / "classification", _load_classification_raw, cls_raw),
        (model_dir / "confession", _load_confession_raw, conf_raw),
    ]:
        if not technique_dir.exists():
            continue
        for subdir in sorted(technique_dir.iterdir()):
            if not subdir.is_dir():
                continue
            items = loader(subdir, ground_truth)
            if items is None:
                continue
            if subdir.name == "baseline":
                label = "Baseline"
            elif subdir.name == "gpt":
                label = "GPT-4.1 mini"
            else:
                label = "Fine-tuned"
            raw_dict[label] = items

    return cls_raw, conf_raw


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

    for technique_dir, loader, baseline_label, finetuned_label, gpt_label in [
        (cls_dir, _load_classification_items, "Baseline\nClassification", "Fine-tuned\nClassification", "GPT-4.1 mini\nClassification"),
        (conf_dir, _load_confession_items, "Baseline\nConfession", "Fine-tuned\nConfession", "GPT-4.1 mini\nConfession"),
    ]:
        if not technique_dir.exists():
            continue
        for subdir in sorted(technique_dir.iterdir()):
            if not subdir.is_dir():
                continue
            items = loader(subdir, ground_truth)
            if items is None:
                continue
            if subdir.name == "baseline":
                label = baseline_label
            elif subdir.name == "gpt":
                label = gpt_label
            else:
                label = finetuned_label
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
        patch = PathPatch(
            MplPath(verts, codes),
            facecolor=bar.get_facecolor(),
            edgecolor="black",
            linewidth=2.0,
            zorder=bar.get_zorder(),
        )
        if bar.get_hatch():
            patch.set_hatch(bar.get_hatch())
        ax.add_patch(patch)


def _style_ax(ax):
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=19)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def _yerr_clipped(vals: np.ndarray, sems: np.ndarray) -> list[np.ndarray]:
    """Return [lower, upper] error bar arrays, clipping lower so bars don't go below 0."""
    mask = np.isnan(vals) | np.isnan(sems)
    lower = np.where(mask, 0, np.minimum(sems, vals))
    upper = np.where(np.isnan(sems), 0, sems)
    return [lower, upper]


def _draw_right_side(ax, methods, available, row, pending_rounds):
    """Draw the balanced accuracy subplot (right side)."""
    n = len(available)
    colors = [METHOD_COLORS[METHOD_ORDER.index(m)] for m in available]
    bar_spacing = 0.6
    x = np.arange(n) * bar_spacing
    ba_results = [compute_balanced_accuracy_with_ci(methods[m]) for m in available]
    ba_vals = np.array([r[0] for r in ba_results])
    ba_sems = np.array([r[1] for r in ba_results])

    bars = ax.bar(
        x, ba_vals, bar_spacing * 0.85,
        color=colors, edgecolor="black", linewidth=0, alpha=1,
        yerr=_yerr_clipped(ba_vals, ba_sems),
        error_kw={"ecolor": "black", "capsize": 4, "elinewidth": 2.0},
    )
    for bar_idx, method in enumerate(available):
        if method in GPT_METHODS:
            bars[bar_idx].set_hatch("//")
    pending_rounds.append((bars, ax))
    ax.axhline(50, color="#444444", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_SHORT_LABELS[m] for m in available], fontsize=18, ha="center",
    )
    ax.set_xlim(x[0] - bar_spacing * 0.6, x[-1] + bar_spacing * 0.6)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=22)
    if row == 0:
        ax.plot(
            [], [], color="#444444", linestyle="--", linewidth=1.5,
            alpha=0.9, label="Chance",
        )
        leg_ba = ax.legend(
            fontsize=17, loc="upper right", bbox_to_anchor=(1.0, 1.05),
            frameon=False,
        )
        for text in leg_ba.get_texts():
            text.set_color("#444444")
    _style_ax(ax)


def plot_combined(
    model_data: dict[str, dict],
    exclude_cats: set[str] | None = None,
    exclude_methods: set[str] | None = None,
    suffix: str = "",
) -> None:
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

        available = [
            m for m in METHOD_ORDER
            if m in methods and (not exclude_methods or m not in exclude_methods)
        ]
        n = len(available)

        if n == 0:
            print(f"  No methods for {model_key}, skipping row.")
            continue

        # --- Right: Balanced Accuracy ---
        _draw_right_side(axes[row, 1], methods, available, row, pending_rounds)

        # --- Left: Per-category deceptive rate (methods on x-axis, categories as bars) ---
        ax_dr = axes[row, 0]
        active_cats = [
            cat for cat in CATEGORY_DISPLAY_ORDER
            if (not exclude_cats or cat not in exclude_cats)
            and any(
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
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0, alpha=0.75,
                label=CATEGORY_DISPLAY[cat],
                yerr=_yerr_clipped(vals, sems),
                error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 2.0},
            )
            for bar_idx, method in enumerate(available):
                if method in GPT_METHODS:
                    cat_bars[bar_idx].set_hatch("//")
            pending_rounds.append((cat_bars, ax_dr))

        ax_dr.set_xticks(np.arange(n))
        ax_dr.set_xticklabels(available, fontsize=23, ha="center")
        ax_dr.set_ylabel("Deceptive (%)", fontsize=22)
        if row == 0:
            leg = ax_dr.legend(
                fontsize=19, loc="upper center", bbox_to_anchor=(0.5, 1.12),
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
    out_path = OUTPUT_DIR / f"lie_detection_comparison{suffix}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def plot_combined_ratio(model_data: dict[str, dict]) -> None:
    """Like plot_combined but left plots show deceptive rate as ratio to GPT-4.1 mini Classification."""
    reference_method = "GPT-4.1 mini\nClassification"
    exclude_cats = {"partial"}
    right_exclude = {"GPT-4.1 mini\nConfession"}

    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    if not model_keys:
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
        methods = info["methods"]

        right_available = [
            m for m in METHOD_ORDER
            if m in methods and m not in right_exclude
        ]
        if not right_available:
            continue

        # --- Right: Balanced Accuracy (same as no_partial) ---
        _draw_right_side(axes[row, 1], methods, right_available, row, pending_rounds)

        # --- Left: Ratio to GPT-4.1 mini Classification ---
        ax_dr = axes[row, 0]
        left_available = [
            m for m in METHOD_ORDER
            if m in methods and m not in GPT_METHODS
        ]
        n = len(left_available)

        if reference_method not in methods or n == 0:
            continue

        active_cats = [
            cat for cat in CATEGORY_DISPLAY_ORDER
            if cat not in exclude_cats
            and any(
                any(i["ground_truth_category"] == cat for i in methods[m])
                for m in left_available
            )
        ]
        n_cats = len(active_cats)
        bar_width = 0.8 / max(n_cats, 1)

        gpt_rates = {}
        for cat in active_cats:
            rate, _ = compute_deceptive_rate_with_ci(methods[reference_method], cat)
            gpt_rates[cat] = rate

        for j, cat in enumerate(active_cats):
            gpt_rate = gpt_rates[cat]
            results = [compute_deceptive_rate_with_ci(methods[m], cat) for m in left_available]
            if gpt_rate > 0:
                vals = np.array([r[0] / gpt_rate for r in results])
                sems = np.array([r[1] / gpt_rate for r in results])
            else:
                vals = np.full(n, np.nan)
                sems = np.full(n, np.nan)
            offset = (j - (n_cats - 1) / 2) * bar_width
            cat_bars = ax_dr.bar(
                np.arange(n) + offset, vals, bar_width,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0, alpha=0.75,
                label=CATEGORY_DISPLAY[cat],
                yerr=_yerr_clipped(vals, sems),
                error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 2.0},
            )
            pending_rounds.append((cat_bars, ax_dr))

        ax_dr.axhline(1.0, color="#444444", linestyle="--", linewidth=1.5, alpha=0.9)
        ax_dr.set_xticks(np.arange(n))
        ax_dr.set_xticklabels(left_available, fontsize=19, ha="center")
        ax_dr.set_ylabel("Deceptive Rate / GPT", fontsize=20)
        if row == 0:
            leg = ax_dr.legend(
                fontsize=19, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                ncol=len(active_cats), frameon=False,
            )
            for patch in leg.legend_handles:
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)
        ax_dr.set_ylim(0, None)
        ax_dr.tick_params(axis="y", labelsize=19)
        ax_dr.spines["top"].set_visible(False)
        ax_dr.spines["right"].set_visible(False)
        ax_dr.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    for bars, ax in pending_rounds:
        _round_bars(bars, ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "lie_detection_comparison_ratio.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def plot_combined_by_category(model_data: dict[str, dict]) -> None:
    """Like plot_combined but left plots have categories on x-axis and methods as grouped bars."""
    exclude_cats = {"partial"}
    exclude_methods = {"GPT-4.1 mini\nConfession"}

    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    if not model_keys:
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
        methods = info["methods"]

        available = [
            m for m in METHOD_ORDER
            if m in methods and m not in exclude_methods
        ]
        if not available:
            continue

        # --- Right: Balanced Accuracy ---
        _draw_right_side(axes[row, 1], methods, available, row, pending_rounds)

        # --- Left: Categories on x-axis, methods as grouped bars ---
        ax_dr = axes[row, 0]
        active_cats = [
            cat for cat in CATEGORY_DISPLAY_ORDER
            if cat not in exclude_cats
            and any(
                any(i["ground_truth_category"] == cat for i in methods[m])
                for m in available
            )
        ]
        n_cats = len(active_cats)
        n_methods = len(available)
        bar_width = 0.8 / max(n_methods, 1)

        for j, method in enumerate(available):
            results = [
                compute_deceptive_rate_with_ci(methods[method], cat)
                for cat in active_cats
            ]
            vals = np.array([r[0] for r in results])
            sems = np.array([r[1] for r in results])
            offset = (j - (n_methods - 1) / 2) * bar_width
            color = BY_CAT_METHOD_COLORS.get(method, "#416EA4")
            hatch = "//" if method in GPT_METHODS else None
            cat_bars = ax_dr.bar(
                np.arange(n_cats) + offset, vals, bar_width,
                color=color, edgecolor="black", linewidth=0, alpha=0.85,
                label=method.replace("\n", " "),
                yerr=_yerr_clipped(vals, sems),
                error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 2.0},
                hatch=hatch,
            )
            pending_rounds.append((cat_bars, ax_dr))

        ax_dr.set_xticks(np.arange(n_cats))
        cat_labels = [
            f"{CATEGORY_DISPLAY[cat]} {'↓' if cat == 'complete' else '↑'}"
            for cat in active_cats
        ]
        ax_dr.set_xticklabels(cat_labels, fontsize=23, ha="center")
        ax_dr.set_ylabel("Deceptive (%)", fontsize=22)
        if row == 0:
            leg = ax_dr.legend(
                fontsize=19, loc="upper center", bbox_to_anchor=(0.5, 1.15),
                ncol=3, frameon=False,
            )
            for patch in leg.legend_handles:
                patch.set_edgecolor("black")
                patch.set_linewidth(1.5)
        _style_ax(ax_dr)

    plt.tight_layout()
    for bars, ax in pending_rounds:
        _round_bars(bars, ax)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "lie_detection_comparison_by_category.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def _count_raw_labels(items: list[dict], label_key: str, keys: list[str]) -> dict[str, int]:
    """Count occurrences of each label value in items."""
    counts = {k: 0 for k in keys}
    for item in items:
        val = item.get(label_key)
        if val in counts:
            counts[val] += 1
        else:
            counts[keys[-1]] += 1
    return counts


def _draw_stacked_pct(
    ax, x: np.ndarray, entries: list[tuple[str, list[dict]]],
    label_key: str, keys: list[str], labels: list[str], colors: list[str],
    bar_width: float = 0.6, hatches: list[str | None] | None = None,
) -> None:
    """Draw stacked percentage bars and annotate segments >= 5%."""
    all_counts = [_count_raw_labels(items, label_key, keys) for _, items in entries]
    totals = [max(sum(c.values()), 1) for c in all_counts]

    bottoms = np.zeros(len(entries))
    for key, label, color in zip(keys, labels, colors):
        vals = np.array([100.0 * c[key] / t for c, t in zip(all_counts, totals)])
        bars = ax.bar(
            x, vals, bar_width, bottom=bottoms,
            color=color, label=label, edgecolor="white", linewidth=0.5,
        )
        if hatches:
            for bar_idx, hatch in enumerate(hatches):
                if hatch:
                    bars[bar_idx].set_hatch(hatch)
                    bars[bar_idx].set_edgecolor("#555555")
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottoms[j] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=16, fontweight="bold", color="white",
                )
        bottoms += vals


def _style_dist_ax(ax):
    """Apply common axis styling for distribution subplots."""
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=16)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def plot_raw_distributions(model_data: dict[str, dict]) -> None:
    """Plot true/false and confession/no_confession stacked distributions per response category."""
    cls_entries: list[tuple[str, list[dict]]] = []
    conf_entries: list[tuple[str, list[dict]]] = []

    for model_key in MODEL_CONFIGS:
        if model_key not in model_data:
            continue
        info = model_data[model_key]
        display_name = info["display_name"]
        cls_raw = info.get("cls_raw", {})
        conf_raw = info.get("conf_raw", {})

        for method_label in ["Baseline", "Fine-tuned", "GPT-4.1 mini"]:
            if method_label in cls_raw:
                cls_entries.append((f"{display_name}\n{method_label}", cls_raw[method_label]))
            if method_label in conf_raw:
                conf_entries.append((f"{display_name}\n{method_label}", conf_raw[method_label]))

    if not cls_entries and not conf_entries:
        print("No data for raw distribution plot.")
        return

    all_items: list[dict] = []
    for _, items in cls_entries + conf_entries:
        all_items.extend(items)
    active_cats = [
        cat for cat in CATEGORIES
        if any(it.get("ground_truth_category") == cat for it in all_items)
    ]

    if not active_cats:
        print("No active categories for raw distribution plot.")
        return

    cls_hatches = ["//" if "GPT" in label else None for label, _ in cls_entries]
    conf_hatches = ["//" if "GPT" in label else None for label, _ in conf_entries]

    n_cats = len(active_cats)
    fig, axes = plt.subplots(n_cats, 2, figsize=(16, 4.5 * n_cats), squeeze=False)

    for i, cat in enumerate(active_cats):
        # Left column: Classification (true/false/missing)
        ax = axes[i, 0]
        filtered = [
            (label, [it for it in items if it["ground_truth_category"] == cat])
            for label, items in cls_entries
        ]
        x_cls = np.arange(len(filtered))
        _draw_stacked_pct(
            ax, x_cls, filtered, "truth_label",
            CLS_RAW_KEYS, CLS_RAW_LABELS, CLS_RAW_COLORS,
            hatches=cls_hatches,
        )
        for j, (_, items) in enumerate(filtered):
            ax.text(j, 2, f"n={len(items)}", ha="center", va="bottom",
                    fontsize=16, color="#333333")
        ax.set_xticks(x_cls)
        ax.set_xticklabels([label for label, _ in filtered], fontsize=16, ha="center")
        ax.set_ylabel("Percentage (%)", fontsize=17)
        ax.set_title(f"Classification — {CATEGORY_DISPLAY[cat]}",
                     fontsize=18, fontweight="bold")
        _style_dist_ax(ax)
        if i == 0:
            ax.legend(fontsize=16, loc="upper right")

        # Right column: Confession (confession/no_confession/other)
        ax = axes[i, 1]
        filtered = [
            (label, [it for it in items if it["ground_truth_category"] == cat])
            for label, items in conf_entries
        ]
        x_conf = np.arange(len(filtered))
        _draw_stacked_pct(
            ax, x_conf, filtered, "conf_label",
            CONF_RAW_KEYS, CONF_RAW_LABELS, CONF_RAW_COLORS,
            hatches=conf_hatches,
        )
        for j, (_, items) in enumerate(filtered):
            ax.text(j, 2, f"n={len(items)}", ha="center", va="bottom",
                    fontsize=16, color="#333333")
        ax.set_xticks(x_conf)
        ax.set_xticklabels([label for label, _ in filtered], fontsize=16, ha="center")
        ax.set_ylabel("Percentage (%)", fontsize=17)
        ax.set_title(f"Confession — {CATEGORY_DISPLAY[cat]}",
                     fontsize=18, fontweight="bold")
        _style_dist_ax(ax)
        if i == 0:
            ax.legend(fontsize=16, loc="upper right")

    fig.suptitle(
        "Raw Label Distributions by Response Category",
        fontsize=21, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "lie_detection_raw_distributions.png"
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
        cls_raw, conf_raw = load_model_raw_labels(model_dir, ground_truth)
        if methods:
            model_data[model_key] = {
                "display_name": cfg["display_name"],
                "methods": methods,
                "cls_raw": cls_raw,
                "conf_raw": conf_raw,
            }

    if model_data:
        plot_combined(
            model_data,
            exclude_methods={"GPT-4.1 mini\nConfession"},
        )
        plot_combined(
            model_data,
            exclude_cats={"partial"},
            exclude_methods={"GPT-4.1 mini\nConfession"},
            suffix="_no_partial",
        )
        plot_combined_ratio(model_data)
        plot_combined_by_category(model_data)
        plot_raw_distributions(model_data)


if __name__ == "__main__":
    main()
