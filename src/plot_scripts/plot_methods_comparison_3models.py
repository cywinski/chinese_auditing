# ABOUTME: Single-row bar plot comparing elicitation methods across DeepSeek R1, MiniMax M2.5, and Qwen3.5.
# ABOUTME: One subplot per metric, models on x-axis, method bars grouped within each model.

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
# ============================================================
GPT_REFERENCE = (
    "output_drive/test_facts_explicit/evaluation/gpt-4.1-mini/evaluation_20260212_135226.json",
    "GPT-4.1 Mini",
    "#bababa",
)

METHODS_DEEPSEEK_R1: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/deepseek-r1-0528/evaluation_20260302_111909.json": (
            "Baseline w/ Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/deepseek-r1-0528-pretrain/evaluation_20260302_155343.json": (
            "Next-token Completion",
            "#b8e6a8",
        ),
        "output_drive/test_facts_explicit/evaluation/deepseek-r1-0528-fewshot-n16/evaluation_20260302_162847.json": (
            "Few-shot Prompt",
            "#1a5fa0",
        ),
    }
)

METHODS_MINIMAX: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/minimax-m2.5/evaluation_20260302_114358.json": (
            "Baseline w/ Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/minimax-m2.5-pretrain/evaluation_20260302_192036.json": (
            "Next-token Completion",
            "#b8e6a8",
        ),
        "output_drive/test_facts_explicit/evaluation/minimax-m2.5-fewshot-n16/evaluation_20260302_194502.json": (
            "Few-shot Prompt",
            "#1a5fa0",
        ),
    }
)

METHODS_QWEN35: dict[str, tuple[str, str]] = OrderedDict(
    {
        "output_drive/test_facts_explicit/evaluation/qwen3.5-397b-a17b/evaluation_20260302_170152.json": (
            "Baseline w/ Think",
            "#c8c5f8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3.5-397b-a17b-pretrain/evaluation_20260302_173611.json": (
            "Next-token Completion",
            "#b8e6a8",
        ),
        "output_drive/test_facts_explicit/evaluation/qwen3.5-397b-a17b-fewshot-n16/evaluation_20260302_185016.json": (
            "Few-shot Prompt",
            "#1a5fa0",
        ),
    }
)

MODELS = [
    ("DeepSeek-R1", METHODS_DEEPSEEK_R1),
    ("MiniMax-M2.5", METHODS_MINIMAX),
    ("Qwen3.5-397B", METHODS_QWEN35),
]

# Method display properties: (hatch, linestyle, alpha)
METHOD_STYLE = {
    "GPT-4.1 Mini": ("/", "--", 0.5),
    "Baseline w/ Think": (".", "--", 0.9),
    "Next-token Completion": (None, "-", 0.9),
    "Few-shot Prompt": (None, "-", 0.9),
}

BASE_FONT_SIZE = 45

# ============================================================

BASE = Path("")

# (metric_key, display_label, ylim, aggregate_key)
CHAT_METRICS = [
    ("honesty_score", "Honesty Score ↑", (0, 100), "chat_aggregate"),
    ("pct_facts_mentioned", "Facts Mentioned ↑", (0, 100), "chat_aggregate"),
    ("has_lies", "Responses w/ Lies ↓", (0, 100), "chat_aggregate"),
    ("is_refusal", "Refusals ↓", (0, 100), "chat_aggregate"),
]

INTERROGATION_METRICS = [
    ("pct_facts_mentioned", "Facts Mentioned ↑", (0, 100), "interrogation_aggregate"),
    ("pct_facts_lied", "Facts Contradicted ↓", (0, 100), "interrogation_aggregate"),
    (
        "pct_facts_never_mentioned",
        "Facts Never Mentioned ↓",
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
    """Replace rectangular bar patches with rounded-top versions."""
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


def _populate_row(
    axes,
    metrics_config: list,
    gpt_metrics: dict,
    model_data: list,
    show_xticklabels: bool = True,
    row_label: str = "",
):
    """Fill a row of axes with grouped bars for one set of metrics."""
    gpt_name, gpt_color = GPT_REFERENCE[1], GPT_REFERENCE[2]
    n_models = len(model_data)
    n_methods = len(model_data[0][2])

    bar_width = 0.16
    model_group_width = n_methods * bar_width + 0.08
    model_gap = 0.25
    gpt_gap = 0.4

    for col_idx, (mk, label, ylim, agg_key) in enumerate(metrics_config):
        ax = axes[col_idx]

        # GPT reference bar on the far left
        gpt_x = 0.0
        mean = gpt_metrics[agg_key].get(mk, (0, 0))[0]
        sem = gpt_metrics[agg_key].get(mk, (0, 0))[1]
        hatch, ls, alpha = METHOD_STYLE[gpt_name]
        bars = ax.bar(
            gpt_x,
            mean,
            width=bar_width,
            yerr=sem,
            capsize=8,
            color=gpt_color,
            hatch=hatch,
            alpha=alpha,
            edgecolor="black",
            linewidth=3,
            error_kw={"elinewidth": 3, "capthick": 3},
        )
        _round_bar_tops(ax, bars, linestyle=ls)

        # Model groups start after the GPT bar + gap
        models_start = gpt_x + bar_width / 2 + gpt_gap
        model_centers = (
            models_start
            + model_group_width / 2
            + np.arange(n_models) * (model_group_width + model_gap)
        )

        for m_idx, (model_name, all_metrics, names, colors) in enumerate(model_data):
            center = model_centers[m_idx]
            for j, (name, color) in enumerate(zip(names, colors)):
                x = center + (j - (n_methods - 1) / 2) * bar_width
                mean = all_metrics[name][agg_key].get(mk, (0, 0))[0]
                sem = all_metrics[name][agg_key].get(mk, (0, 0))[1]

                hatch, ls, alpha = METHOD_STYLE[name]

                bars = ax.bar(
                    x,
                    mean,
                    width=bar_width,
                    yerr=sem,
                    capsize=8,
                    color=color,
                    hatch=hatch,
                    alpha=alpha,
                    edgecolor="black",
                    linewidth=3,
                    error_kw={"elinewidth": 3, "capthick": 3},
                )
                _round_bar_tops(ax, bars, linestyle=ls)

        if show_xticklabels:
            ax.set_xticks(list(model_centers))
            ax.set_xticklabels(
                [md[0] for md in model_data],
                fontsize=BASE_FONT_SIZE + 12,
                rotation=30,
                ha="right",
            )
        else:
            ax.set_xticks([])

        ax.set_title(label, fontsize=BASE_FONT_SIZE + 24, pad=20)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.yaxis.grid(True, which="minor", linestyle="--", alpha=0.4, zorder=0)
        ax.yaxis.grid(True, which="major", linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="y", labelsize=BASE_FONT_SIZE + 12)

        if col_idx == 0:
            if row_label:
                ax.set_ylabel(row_label, fontsize=BASE_FONT_SIZE + 24)
        else:
            ax.set_yticklabels([])
            ax.spines["left"].set_visible(False)
            ax.tick_params(axis="y", length=0)

        ax.set_xlim(
            gpt_x - bar_width / 2 - 0.15,
            model_centers[-1] + model_group_width / 2 + 0.15,
        )


def plot_two_rows(
    top_metrics: list,
    bottom_metrics: list,
    output_path: Path,
):
    """Two-row figure: chat metrics on top, interrogation on bottom."""
    # Load GPT reference once
    gpt_path, gpt_name, gpt_color = GPT_REFERENCE
    full_gpt_path = BASE / gpt_path
    gpt_data = load_evaluation(full_gpt_path)
    gpt_metrics = compute_all_metrics(gpt_data)
    print(f"\n--- {gpt_name} (reference) ---")
    print(
        f"{gpt_name}: honesty={gpt_metrics['chat_aggregate']['honesty_score'][0]:.1f}, "
        f"facts_mentioned={gpt_metrics['interrogation_aggregate']['pct_facts_mentioned'][0]:.1f}%"
    )

    # Load all model data
    model_data = []
    for model_name, methods_dict in MODELS:
        print(f"\n--- {model_name} ---")
        all_metrics, names, colors = load_methods(methods_dict)
        model_data.append((model_name, all_metrics, names, colors))

    n_top = len(top_metrics)
    n_bot = len(bottom_metrics)
    n_cols = max(n_top, n_bot)

    fig_width = 70
    fig_height = 34

    fig, all_axes = plt.subplots(
        2,
        n_cols,
        figsize=(fig_width, fig_height),
    )

    # Hide unused axes if column counts differ
    for col_idx in range(n_cols):
        if col_idx >= n_top:
            all_axes[0, col_idx].set_visible(False)
        if col_idx >= n_bot:
            all_axes[1, col_idx].set_visible(False)

    _populate_row(
        all_axes[0, :n_top],
        top_metrics,
        gpt_metrics,
        model_data,
        show_xticklabels=False,
        row_label="Interrogation (%)",
    )
    _populate_row(
        all_axes[1, :n_bot],
        bottom_metrics,
        gpt_metrics,
        model_data,
        show_xticklabels=True,
        row_label="Chat (%)",
    )

    # Horizontal separator between rows
    fig.add_artist(
        plt.Line2D(
            [0.05, 0.95],
            [0.5, 0.5],
            transform=fig.transFigure,
            color="grey",
            linewidth=1.5,
            alpha=0.4,
        )
    )

    # Single legend
    legend_entries = [(gpt_name, gpt_color)] + list(MODELS[0][1].values())
    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="black",
            linewidth=2,
            alpha=METHOD_STYLE[name][2],
            linestyle=METHOD_STYLE[name][1],
            hatch=METHOD_STYLE[name][0],
            label=name,
        )
        for name, color in legend_entries
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.91, 0.92),
        ncol=1,
        fontsize=BASE_FONT_SIZE + 16,
        frameon=False,
        handletextpad=0.3,
        handleheight=0.75,
        handlelength=0.75,
    )

    plt.subplots_adjust(wspace=0.04, hspace=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {output_path}")


def main():
    output_dir = BASE / "output_drive/plots"
    plot_two_rows(
        INTERROGATION_METRICS,
        CHAT_METRICS,
        output_dir / "test_methods_comparison_3models.png",
    )


if __name__ == "__main__":
    main()
