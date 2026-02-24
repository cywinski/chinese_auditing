# ABOUTME: Bar plot comparing honesty scores across elicitation methods for two models.
# ABOUTME: Single chart with two x-ticks (Qwen3-32B, Qwen3-VL-8B), grouped bars per method.

import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import compute_all_metrics, load_evaluation

# ============================================================
# CONFIGURATION: {path: (display_name, hex_color)}
# ============================================================
METHODS_QWEN3_32B: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/gpt-4.1-mini/evaluation_20260212_135226.json": (
            "GPT-4.1 Mini",
            "#a6a6a6",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b/evaluation_20260211_094430.json": (
            "Baseline w/o Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-thinking/evaluation_20260213_122313.json": (
            "Baseline w/ Think",
            "#847ede",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-abliterated/evaluation_20260212_211826.json": (
            "Abliterated",
            "#d4eaf7",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-fuzzing/evaluation_20260211_211923.json": (
            "Fuzzing",
            "#b3d9f2",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-fewshot-n16/evaluation_20260218_220048.json": (
            "Few-shot Prompt",
            "#8dc8e8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-facts_steering/evaluation_20260217_122621.json": (
            "Steering",
            "#67b4db",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-pretrain/evaluation_20260220_094919.json": (
            "No format prefill",
            "#449dcc",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-assistant_prefill/evaluation_20260223_140639.json": (
            "Assistant prefill",
            "#2885ba",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-user_prefill_custom/evaluation_20260223_130018.json": (
            "User prefill",
            "#1a6d9e",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-system_prompt/evaluation_20260220_091632.json": (
            "System Prompt",
            "#0e5580",
        ),
    }
)

METHODS_QWEN3_VL: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/gpt-4.1-mini/evaluation_20260212_135226.json": (
            "GPT-4.1 Mini",
            "#a6a6a6",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b/evaluation_20260215_094923.json": (
            "Baseline w/o Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-thinking/evaluation_20260215_134231.json": (
            "Baseline w/ Think",
            "#847ede",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-abliterated/evaluation_20260215_115300.json": (
            "Abliterated",
            "#d4eaf7",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-fuzzing/evaluation_20260215_153129.json": (
            "Fuzzing",
            "#b3d9f2",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-fewshot-n16/evaluation_20260215_211754.json": (
            "Few-shot Prompt",
            "#8dc8e8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-facts_steering/evaluation_20260217_231356.json": (
            "Steering",
            "#67b4db",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-pretrain/evaluation_20260219_220453.json": (
            "No format prefill",
            "#449dcc",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-assistant_prefill/evaluation_20260219_225351.json": (
            "Assistant prefill",
            "#2885ba",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-user_prefill_standard/evaluation_20260219_225351.json": (
            "User prefill",
            "#1a6d9e",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-system_prompt/evaluation_20260219_213107.json": (
            "System Prompt",
            "#0e5580",
        ),
    }
)

MODEL_ROWS = [
    ("Qwen3-32B", METHODS_QWEN3_32B),
    ("Qwen3-VL-8B", METHODS_QWEN3_VL),
]

OUTPUT_PATH = "output/plots/test_honesty_comparison.png"
METRIC_KEY = "honesty_score"
AGGREGATE_KEY = "chat_aggregate"
BASE_FONT_SIZE = 24

# ============================================================

BASE = Path("/workspace/projects/chinese_auditing")


def load_methods(methods_dict: dict) -> tuple[dict, list[str], list[str]]:
    """Load metrics for all methods in a dict. Returns (all_metrics, names, colors)."""
    all_metrics = {}
    names = []
    colors = []

    placeholder_metrics = {
        "chat_aggregate": {},
        "interrogation_aggregate": {},
    }

    for rel_path, (name, color) in methods_dict.items():
        if not rel_path.endswith(".json"):
            all_metrics[name] = placeholder_metrics
            names.append(name)
            colors.append(color)
            print(f"{name}: (placeholder)")
            continue
        full_path = BASE / rel_path
        if not full_path.exists():
            print(f"WARNING: {full_path} not found, skipping")
            continue
        data = load_evaluation(full_path)
        metrics = compute_all_metrics(data)
        all_metrics[name] = metrics
        names.append(name)
        colors.append(color)
        print(
            f"{name}: honesty={metrics['chat_aggregate']['honesty_score'][0]:.1f}"
        )

    return all_metrics, names, colors


GPT_METHOD_NAME = "GPT-4.1 Mini"


def plot_honesty(output_path: Path):
    model_names = []
    model_data = []  # list of (all_metrics, names, colors)

    for model_name, methods_dict in MODEL_ROWS:
        print(f"\n--- {model_name} ---")
        all_metrics, names, colors = load_methods(methods_dict)
        if not all_metrics:
            print(f"No results for {model_name}")
            continue
        model_names.append(model_name)
        model_data.append((all_metrics, names, colors))

    # Separate GPT reference from the rest
    method_names = model_data[0][1]
    method_colors = model_data[0][2]
    gpt_idx = method_names.index(GPT_METHOD_NAME)
    gpt_color = method_colors[gpt_idx]
    audit_names = [n for i, n in enumerate(method_names) if i != gpt_idx]
    audit_colors = [c for i, c in enumerate(method_colors) if i != gpt_idx]
    n_audit = len(audit_names)
    n_models = len(model_names)

    # X positions: GPT bar on the left, then a gap, then model groups
    gpt_x = 0
    gap = 0.8
    bar_width = 0.5 / n_audit
    group_width = n_audit * bar_width
    model_positions = np.array([
        gpt_x + 1.0 + gap + group_width / 2 + k * (group_width + gap)
        for k in range(n_models)
    ])

    fig, ax = plt.subplots(figsize=(max(14, 2.5 * (n_audit + 1)), 12))

    # GPT reference bar (single bar, use first model's value)
    gpt_val = model_data[0][0][GPT_METHOD_NAME][AGGREGATE_KEY].get(METRIC_KEY, (0, 0))
    ax.bar(
        gpt_x,
        gpt_val[0],
        width=bar_width * 3,
        yerr=gpt_val[1],
        capsize=6,
        color=gpt_color,
        edgecolor="black",
        linewidth=1.5,
        label=GPT_METHOD_NAME,
        error_kw={"elinewidth": 2, "capthick": 2},
    )

    # Dashed separator between GPT and the rest
    sep_x = gpt_x + (bar_width * 3) / 2 + gap / 2
    ax.axvline(x=sep_x, color="gray", linestyle="--", linewidth=2)

    # Horizontal reference line at GPT score across the audit methods region
    audit_left = model_positions[0] - group_width / 2 - bar_width
    audit_right = model_positions[-1] + group_width / 2 + bar_width
    ax.hlines(
        gpt_val[0], audit_left, audit_right,
        colors=gpt_color, linestyles="--", linewidth=2, alpha=0.7,
    )

    # Audit method bars for each model
    for i, (name, color) in enumerate(zip(audit_names, audit_colors)):
        means = []
        sems = []
        for all_metrics, _, _ in model_data:
            val = all_metrics[name][AGGREGATE_KEY].get(METRIC_KEY, (0, 0))
            means.append(val[0])
            sems.append(val[1])

        offset = (i - (n_audit - 1) / 2) * bar_width
        ax.bar(
            model_positions + offset,
            means,
            width=bar_width,
            yerr=sems,
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            label=name,
            error_kw={"elinewidth": 2, "capthick": 2},
        )

    all_ticks = [gpt_x] + list(model_positions)
    all_labels = [GPT_METHOD_NAME] + model_names
    ax.set_xticks(all_ticks)
    ax.set_xticklabels(all_labels, fontsize=BASE_FONT_SIZE + 4)
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE + 2)
    ax.set_ylabel("Honesty Score (%)", fontsize=BASE_FONT_SIZE + 4)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=min(len(audit_names) + 1, 6),
        fontsize=BASE_FONT_SIZE,
        frameon=False,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    output_path = BASE / OUTPUT_PATH
    plot_honesty(output_path)


if __name__ == "__main__":
    main()
