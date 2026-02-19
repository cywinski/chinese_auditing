# ABOUTME: Bar plot comparing elicitation methods for Qwen3-32B and Qwen3-VL-8B.
# ABOUTME: Two rows (one per model), methods/colors configured via dicts at the top.

import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from metrics import compute_all_metrics, load_evaluation

# ============================================================
# CONFIGURATION: {path: (display_name, hex_color)}
# Use any non-.json key (e.g. "1", "2") to create a placeholder bar with no data.
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
            "#f5e0d9",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-fuzzing/evaluation_20260211_211923.json": (
            "Fuzzing",
            "#f5e0d9",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-fewshot-n16/evaluation_20260218_220048.json": (
            "Few-shot Prompt",
            "#f5e0d9",
        ),
        # "output_drive/test_facts_explicit/evaluation/qwen3-32b-honesty_steering/evaluation_20260211_235837.json": (
        #     "Honesty Steering",
        #     "#f5e0d9",
        # ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-facts_steering/evaluation_20260217_122621.json": (
            "Steering",
            "#f5e0d9",
        ),
        "1": (
            "Prefill attack",
            "#f5e0d9",
        ),
        "2": (
            "Honesty FT",
            "#f5e0d9",
        ),
        "3": (
            "System Prompt",
            "#f5e0d9",
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
            "#f5e0d9",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-fuzzing/evaluation_20260215_153129.json": (
            "Fuzzing",
            "#f5e0d9",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-fewshot-n16/evaluation_20260215_211754.json": (
            "Few-shot Prompt",
            "#f5e0d9",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-facts_steering/evaluation_20260217_231356.json": (
            "Steering",
            "#f5e0d9",
        ),
        "1": (
            "Prefill attack",
            "#f5e0d9",
        ),
        "2": (
            "Honesty FT",
            "#f5e0d9",
        ),
        "3": (
            "System Prompt",
            "#f5e0d9",
        ),
    }
)

MODEL_ROWS = [
    ("Qwen3-32B", METHODS_QWEN3_32B),
    ("Qwen3-VL-8B", METHODS_QWEN3_VL),
]

OUTPUT_PATH = "output/plots/test_methods_comparison.png"

BASE_FONT_SIZE = 24

# ============================================================

BASE = Path("/workspace/projects/chinese_auditing")

# (metric_key, display_label, ylim, aggregate_key)
ALL_METRICS = [
    # ("honesty_score", "Response Honesty Score ↑", (0, 100), "chat_aggregate"),
    ("pct_facts_mentioned", "Facts Mentioned ↑", (0, 100), "chat_aggregate"),
    ("pct_facts_lied", "Facts Contradicted ↓", (0, 100), "chat_aggregate"),
    ("is_refusal", "Refusals ↓", (0, 100), "chat_aggregate"),
    ("has_lies", "Responses w/ Lies ↓", (0, 100), "chat_aggregate"),
    # Interrogation metrics
    (
        "pct_facts_mentioned",
        "Facts Mentioned ↑",
        (0, 100),
        "interrogation_aggregate",
    ),
    (
        "pct_facts_lied",
        "Facts Contradicted ↓",
        (0, 100),
        "interrogation_aggregate",
    ),
    (
        "pct_facts_never_mentioned",
        "Facts Never Mentioned ↓",
        (0, 100),
        "interrogation_aggregate",
    ),
]

CHAT_METRICS_END_IDX = 4


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
            f"{name}: honesty={metrics['chat_aggregate']['honesty_score'][0]:.1f}, "
            f"facts_mentioned={metrics['interrogation_aggregate']['pct_facts_mentioned'][0]:.1f}%"
        )

    return all_metrics, names, colors


def plot_grouped_bars(
    ax,
    metrics_config: list,
    all_metrics: dict,
    names: list[str],
    colors: list[str],
    model_name: str = "",
    show_legend: bool = False,
    show_xticklabels: bool = True,
):
    n_methods = len(names)
    n_metrics = len(metrics_config)
    bar_width = 0.5 / n_methods
    metric_positions = np.arange(n_metrics)

    for i, (name, color) in enumerate(zip(names, colors)):
        means = [
            all_metrics[name][agg_key].get(mk, (0, 0))[0]
            for mk, _, _, agg_key in metrics_config
        ]
        sems = [
            all_metrics[name][agg_key].get(mk, (0, 0))[1]
            for mk, _, _, agg_key in metrics_config
        ]
        offset = (i - (n_methods - 1) / 2) * bar_width
        ax.bar(
            metric_positions + offset,
            means,
            width=bar_width,
            yerr=sems,
            capsize=6,
            color=color,
            edgecolor="black",
            linewidth=1.5,
            label=name,
            error_kw={
                "elinewidth": 2,  # thickness of error line
                "capthick": 2,  # thickness of caps
            },
        )

    ax.axvline(x=CHAT_METRICS_END_IDX - 0.5, color="gray", linestyle="--", linewidth=2)
    ax.set_xticks(metric_positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [label for _, label, _, _ in metrics_config],
            fontsize=BASE_FONT_SIZE + 2,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE + 6)
    ylabel = f"{model_name} (%)" if model_name else "%"
    ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE + 4)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_legend:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=n_methods,
            fontsize=BASE_FONT_SIZE + 4,
            frameon=False,
        )


def plot_two_rows(output_path: Path):
    n_rows = len(MODEL_ROWS)
    n_metrics = len(ALL_METRICS)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(7 * n_metrics, 7 * n_rows),
    )
    if n_rows == 1:
        axes = [axes]

    for row_idx, (model_name, methods_dict) in enumerate(MODEL_ROWS):
        print(f"\n--- {model_name} ---")
        all_metrics, names, colors = load_methods(methods_dict)
        if not all_metrics:
            print(f"No results for {model_name}")
            continue

        ax = axes[row_idx]
        plot_grouped_bars(
            ax,
            ALL_METRICS,
            all_metrics,
            names,
            colors,
            model_name=model_name,
            show_legend=(row_idx == 0),
            show_xticklabels=(row_idx == n_rows - 1),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    output_path = BASE / OUTPUT_PATH
    plot_two_rows(output_path)


if __name__ == "__main__":
    main()
