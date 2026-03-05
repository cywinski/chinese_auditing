# ABOUTME: Bar plots comparing elicitation methods on interrogation metrics.
# ABOUTME: Two rows (Qwen3-32B and Qwen3-VL-8B), three metric groups.

import sys
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MultipleLocator

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
            "#bababa",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b/evaluation_20260211_094430.json": (
            "Baseline w/o Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-thinking/evaluation_20260213_122313.json": (
            "Baseline w/ Think",
            "#847ede",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-assistant_prefill/evaluation_20260223_140639.json": (
            "Assistant Prefill",
            "#b8e6a8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-user_prefill_custom/evaluation_20260223_130018.json": (
            "User Prefill",
            "#3da836",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-pretrain/evaluation_20260220_094919.json": (
            "Next-token Compl.",
            "#145a0e",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-abliterated/evaluation_20260212_211826.json": (
            "Anti-refusal",
            "#f5d0a0",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-facts_steering/evaluation_20260217_122621.json": (
            "Steering",
            "#e8943a",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-honesty_finetuning/evaluation_20260226_042008.json": (
            "Honesty FT",
            "#c45e1a",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-sp4-honesty-pledge/evaluation_20260224_184154.json": (
            "System Prompt",
            "#b3d4f0",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-32b-fewshot-n16/evaluation_20260218_220048.json": (
            "Few-shot Prompt",
            "#1a5fa0",
        ),
    }
)

METHODS_QWEN3_VL: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/gpt-4.1-mini/evaluation_20260212_135226.json": (
            "GPT-4.1 Mini",
            "#bababa",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b/evaluation_20260215_094923.json": (
            "Baseline w/o Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-thinking/evaluation_20260215_134231.json": (
            "Baseline w/ Think",
            "#847ede",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-assistant_prefill/evaluation_20260223_132930.json": (
            "Assistant Prefill",
            "#b8e6a8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-user_prefill_standard/evaluation_20260219_225351.json": (
            "User Prefill",
            "#3da836",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-pretrain/evaluation_20260219_220453.json": (
            "Next-token Compl.",
            "#145a0e",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-abliterated/evaluation_20260215_115300.json": (
            "Anti-refusal",
            "#f5d0a0",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-facts_steering/evaluation_20260217_231356.json": (
            "Steering",
            "#e8943a",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-honesty_finetuning/evaluation_20260226_055033.json": (
            "Honesty FT",
            "#c45e1a",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-sp5/evaluation_20260224_213608.json": (
            "System Prompt",
            "#b3d4f0",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3-vl-8b-fewshot-n16/evaluation_20260215_211754.json": (
            "Few-shot Prompt",
            "#1a5fa0",
        ),
    }
)

MODEL_ROWS = [
    ("Qwen3-32B", METHODS_QWEN3_32B),
    ("Qwen3-VL-8B", METHODS_QWEN3_VL),
]

BASE_FONT_SIZE = 40

# ============================================================

BASE = Path("/workspace/projects/chinese_auditing")

INTERROGATION_METRICS = [
    (
        "pct_facts_mentioned",
        "Facts Mentioned \u2191",
        (0, 100),
        "interrogation_aggregate",
    ),
    (
        "pct_facts_lied",
        "Facts Contradicted \u2193",
        (0, 100),
        "interrogation_aggregate",
    ),
    (
        "pct_facts_never_mentioned",
        "Facts Never Mentioned \u2193",
        (0, 100),
        "interrogation_aggregate",
    ),
]


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


def _rounded_top_path(x, y, w, h, rx, ry):
    """Rectangle path with only the top two corners rounded (separate x/y radii)."""
    rx = min(rx, w / 2)
    ry = min(ry, h / 2) if h > 0 else 0
    if rx <= 0 or ry <= 0:
        return MplPath(
            [(x, y), (x, y + h), (x + w, y + h), (x + w, y), (x, y)],
            [
                MplPath.MOVETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.LINETO,
                MplPath.CLOSEPOLY,
            ],
        )
    return MplPath(
        [
            (x, y),
            (x, y + h - ry),
            (x, y + h),
            (x + rx, y + h),
            (x + w - rx, y + h),
            (x + w, y + h),
            (x + w, y + h - ry),
            (x + w, y),
            (x, y),
        ],
        [
            MplPath.MOVETO,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CURVE3,
            MplPath.CURVE3,
            MplPath.LINETO,
            MplPath.CLOSEPOLY,
        ],
    )


def _round_bar_tops(ax, bars, ry=0.7, linestyle="-"):
    """Replace rectangular bar patches with rounded-top versions.

    rx is derived from bar width; ry is in y-axis data units (default 3 out of 100).
    """
    for bar in bars:
        x, y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        rx = w / 2
        patch = PathPatch(
            _rounded_top_path(x, y, w, h, rx, ry),
            facecolor=bar.get_facecolor(),
            edgecolor=bar.get_edgecolor(),
            linewidth=bar.get_linewidth(),
            linestyle=linestyle,
            hatch=bar.get_hatch(),
            zorder=bar.get_zorder(),
        )
        ax.add_patch(patch)
        bar.set_visible(False)


def plot_grouped_bars(
    ax,
    metrics_config: list,
    all_metrics: dict,
    names: list[str],
    colors: list[str],
    model_name: str = "",
    legend_pos: tuple[float, float] | None = None,
    show_xticklabels: bool = True,
    show_ylabel: bool = True,
):
    n_methods = len(names)
    n_metrics = len(metrics_config)
    bar_width = 0.05 / n_methods
    metric_positions = np.arange(n_metrics) * (n_methods * bar_width + 0.02)

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
        if name == "GPT-4.1 Mini":
            hatch = "/"
        elif name.startswith("Baseline"):
            hatch = "."
        else:
            hatch = None
        bars = ax.bar(
            metric_positions + offset,
            means,
            width=bar_width,
            yerr=sems,
            capsize=14,
            color=color,
            hatch=hatch,
            alpha=0.9 if name != "GPT-4.1 Mini" else 0.5,
            edgecolor="black",
            linewidth=5,
            label=name,
            error_kw={
                "elinewidth": 4,  # thickness of error line
                "capthick": 4,  # thickness of caps
            },
        )
        ls = "--" if name == "GPT-4.1 Mini" else "-"
        _round_bar_tops(ax, bars, linestyle=ls)

    margin = bar_width * 1.5
    ax.set_xlim(
        metric_positions[0] - (n_methods / 2) * bar_width - margin,
        metric_positions[-1] + (n_methods / 2) * bar_width + margin,
    )
    ax.set_xticks(metric_positions)
    if show_xticklabels:
        ax.set_xticklabels(
            [label for _, label, _, _ in metrics_config],
            fontsize=BASE_FONT_SIZE + 30,
        )
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE + 20)
    if show_ylabel:
        ylabel = f"{model_name} (%)" if model_name else "%"
        ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE + 28)
    else:
        ax.set_yticklabels([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.yaxis.grid(True, which="minor", linestyle="--", alpha=0.4, zorder=0)
    ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend_pos is not None:
        ax.legend(
            loc="center left",
            bbox_to_anchor=legend_pos,
            ncol=1,
            fontsize=BASE_FONT_SIZE + 20,
            frameon=False,
        )


def plot_two_rows(
    metrics_config: list,
    output_path: Path,
    fig_width_scale: float = 1.0,
):
    n_rows = len(MODEL_ROWS)
    n_metrics = len(metrics_config)
    n_methods = max(len(d) for _, d in MODEL_ROWS)

    fig_width = max(n_metrics * (n_methods * 1.2), 25) * fig_width_scale

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, 15 * n_rows),
    )
    if n_rows == 1:
        axes = np.array([axes])

    for row_idx, (model_name, methods_dict) in enumerate(MODEL_ROWS):
        print(f"\n--- {model_name} ---")
        all_metrics, names, colors = load_methods(methods_dict)
        if not all_metrics:
            print(f"No results for {model_name}")
            continue

        plot_grouped_bars(
            axes[row_idx],
            metrics_config,
            all_metrics,
            names,
            colors,
            model_name=model_name,
            legend_pos=None,
            show_xticklabels=(row_idx == n_rows - 1),
        )

    # Single legend above the plot, top right
    _, first_methods = MODEL_ROWS[0]
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=3,
            alpha=1.0 if name != "GPT-4.1 Mini" else 0.5,
            linestyle="--" if name == "GPT-4.1 Mini" else "-",
            hatch="/"
            if name == "GPT-4.1 Mini"
            else ("." if name.startswith("Baseline") else None),
            label=name,
        )
        for name, color in first_methods.values()
    ]
    n_per_row = (len(legend_handles) + 2) // 3
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.88, 0.85),
        ncol=n_per_row,
        fontsize=BASE_FONT_SIZE + 20,
        frameon=False,
        handletextpad=0.3,
        handleheight=0.75,
        handlelength=0.75,
        columnspacing=0.8,
    )

    plt.subplots_adjust(hspace=0.07)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    output_dir = BASE / "output_drive/plots"
    plot_two_rows(
        INTERROGATION_METRICS,
        output_dir / "test_methods_comparison_interrogation.png",
        fig_width_scale=1.2,
    )


if __name__ == "__main__":
    main()
