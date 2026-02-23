# ABOUTME: Plot classification (true/false) and confession results for all models and methods.
# ABOUTME: Discovers models and methods automatically from the responses_dev directory.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESPONSES_BASE = Path("/root/chinese_auditing/output/responses_dev")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

MODEL_DISPLAY = {
    "qwen3-32b": "Qwen3-32B",
    "qwen3-vl-8b-thinking": "Qwen3-VL-8B-Thinking",
}

# Preferred ordering for known methods; unknowns appended alphabetically.
METHOD_ORDER = [
    "followup-original",
    "followup_anthropic",
    "followup-qwen-data",
    "followup_qwen_vl_8b_thinking",
    "followup-split-personality",
    "followup_split_personality",
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

# Truth labels and their display names (ground truth from classification)
TRUTH_LABELS = ["true", "false", "missing"]
TRUTH_DISPLAY = {"true": "True", "false": "False", "missing": "Missing / Error"}

# Stacked bar segments for classification
CLS_KEYS = ["true", "false", "missing"]
CLS_LABELS = ["True", "False", "Missing / Error"]
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


def load_joined_data(method_dir: Path) -> list[dict] | None:
    """Join classification and confession files by (prompt_id, sample_idx).

    Each returned dict has:
        truth_label: 'true' | 'false' | 'missing'
        conf_label:  'confession' | 'no_confession' | 'other'
    """
    cls_files = sorted(method_dir.glob("classification_*.json"))
    evl_files = sorted(method_dir.glob("evaluated_confession_*.json"))
    if not cls_files or not evl_files:
        return None

    cls_data = json.loads(cls_files[0].read_text())
    evl_data = json.loads(evl_files[0].read_text())

    conf_lookup = {
        (item["prompt_id"], item["sample_idx"]): item.get("asking_evaluation", {}).get(
            "classification", "OTHER"
        )
        for item in evl_data
    }

    joined = []
    for item in cls_data:
        key = (item["prompt_id"], item["sample_idx"])
        is_true = item.get("truthfulness_evaluation", {}).get("is_true")
        truth_label = "true" if is_true is True else "false" if is_true is False else "missing"

        raw_conf = conf_lookup.get(key, "OTHER")
        conf_label = (
            "confession" if raw_conf == "CONFESSION"
            else "no_confession" if raw_conf == "NO_CONFESSION"
            else "other"
        )
        joined.append({"truth_label": truth_label, "conf_label": conf_label})

    return joined


def _count_conf(items: list[dict]) -> dict[str, int]:
    """Count confession outcomes for a list of joined items."""
    counts = {"confession": 0, "no_confession": 0, "other": 0}
    for item in items:
        counts[item["conf_label"]] += 1
    return counts


def _count_cls(items: list[dict]) -> dict[str, int]:
    """Count truth labels for a list of joined items."""
    counts = {"true": 0, "false": 0, "missing": 0}
    for item in items:
        counts[item["truth_label"]] += 1
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


def plot_model(model_dir: Path, output_dir: Path) -> None:
    model = model_dir.name
    display_name = MODEL_DISPLAY.get(model, model)
    confession_dir = model_dir / "confession"
    if not confession_dir.exists():
        print(f"  No confession dir for {model}, skipping.")
        return

    # Load joined data for every method
    joined_by_method: dict[str, list[dict]] = {}
    for mdir in sorted(confession_dir.iterdir()):
        if not mdir.is_dir():
            continue
        data = load_joined_data(mdir)
        if data is not None:
            joined_by_method[mdir.name] = data

    methods = _sort_methods(list(joined_by_method.keys()))
    n = len(methods)
    if n == 0:
        print(f"  No data found for {model}, skipping.")
        return

    print(f"\n=== {display_name} ({n} methods) ===")

    # Pre-compute counts for all truth-label subsets
    # overall classification counts (truth label distribution)
    cls_overall = {m: _count_cls(joined_by_method[m]) for m in methods}
    # confession counts: overall, then per truth label
    conf_overall = {m: _count_conf(joined_by_method[m]) for m in methods}
    conf_by_truth = {
        tl: {
            m: _count_conf([r for r in joined_by_method[m] if r["truth_label"] == tl])
            for m in methods
        }
        for tl in TRUTH_LABELS
    }

    for m in methods:
        print(
            f"  {m}: cls={cls_overall[m]} "
            f"conf(all)={conf_overall[m]} "
            f"conf(false)={conf_by_truth['false'][m]} "
            f"conf(true)={conf_by_truth['true'][m]}"
        )

    # Only show truth labels that have at least one item across any method
    active_truth_labels = [
        tl for tl in TRUTH_LABELS
        if any(sum(conf_by_truth[tl][m].values()) > 0 for m in methods)
    ]

    x = np.arange(n)
    bar_width = 0.6
    # 2 overview rows + one row per active truth label (2 columns for truth label subplots)
    n_truth_rows = len(active_truth_labels)
    fig, axes = plt.subplots(
        2 + n_truth_rows, 1,
        figsize=(max(14, n * 1.5), 8 + 6 * n_truth_rows),
        sharex=True,
    )
    if not hasattr(axes, "__len__"):
        axes = [axes]

    # --- Row 0: Classification (true/false/missing) overall ---
    ax_cls = axes[0]
    _stacked_bars(ax_cls, x, methods, cls_overall, CLS_KEYS, CLS_LABELS, CLS_COLORS, bar_width)
    ax_cls.set_ylabel("Percentage (%)", fontsize=17)
    ax_cls.set_title(
        "Classification: True / False Rate (all items)", fontsize=18, fontweight="bold"
    )
    ax_cls.set_ylim(0, 100)
    ax_cls.tick_params(axis="y", labelsize=16)
    ax_cls.legend(fontsize=15, loc="lower right")
    ax_cls.spines["top"].set_visible(False)
    ax_cls.spines["right"].set_visible(False)
    ax_cls.grid(axis="y", alpha=0.3)

    # --- Row 1: Confession outcomes overall ---
    ax_conf_all = axes[1]
    _stacked_bars(
        ax_conf_all, x, methods, conf_overall, CONF_KEYS, CONF_LABELS, CONF_COLORS, bar_width
    )
    ax_conf_all.set_ylabel("Percentage (%)", fontsize=17)
    ax_conf_all.set_title(
        "Confession Rate (all items)", fontsize=18, fontweight="bold"
    )
    ax_conf_all.set_ylim(0, 100)
    ax_conf_all.tick_params(axis="y", labelsize=16)
    ax_conf_all.legend(fontsize=15, loc="lower right")
    ax_conf_all.spines["top"].set_visible(False)
    ax_conf_all.spines["right"].set_visible(False)
    ax_conf_all.grid(axis="y", alpha=0.3)

    # --- Rows 2+: Confession rates conditioned on each truth label ---
    for i, tl in enumerate(active_truth_labels):
        ax = axes[2 + i]
        n_items_per_method = [sum(conf_by_truth[tl][m].values()) for m in methods]
        _stacked_bars(
            ax, x, methods, conf_by_truth[tl], CONF_KEYS, CONF_LABELS, CONF_COLORS, bar_width
        )
        # Annotate n inside the bottom of each bar (in data coords)
        for j, n_items in enumerate(n_items_per_method):
            ax.text(
                j, 2, f"n={n_items}",
                ha="center", va="bottom", fontsize=11, color="#333333",
            )
        ax.set_ylabel("Percentage (%)", fontsize=17)
        ax.set_title(
            f"Confession Rate — ground truth: {TRUTH_DISPLAY[tl]}",
            fontsize=18, fontweight="bold",
        )
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=15, loc="lower right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    # Shared x-axis labels on the bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([_method_label(m) for m in methods], fontsize=16)

    fig.suptitle(
        f"Classification & Confession Results — {display_name}",
        fontsize=21, fontweight="bold",
    )
    plt.subplots_adjust(hspace=0.45)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"confession_classification_{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    model_dirs = sorted(d for d in RESPONSES_BASE.iterdir() if d.is_dir())
    if not model_dirs:
        print(f"No model directories found in {RESPONSES_BASE}")
        return
    for model_dir in model_dirs:
        plot_model(model_dir, OUTPUT_DIR)


if __name__ == "__main__":
    main()
