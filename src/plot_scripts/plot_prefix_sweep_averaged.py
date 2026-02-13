# ABOUTME: Plots averaged AUROC and Recall (RP + TQA) by layer per prefix.
# ABOUTME: 1x2 layout with same style as prefix_sweep_by_layer.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire


def _parse_json_array(s):
    if pd.isna(s):
        return []
    return json.loads(s)


def main(
    csv_path: str = "output/deception_probe_qwen3-32b-sweep-sys-prompts/prefix_sweep_results.csv",
    output_dir: str | None = None,
    vertical_line_layer: int | None = None,
) -> None:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df["aurocs_rp"] = df["aurocs_by_layer"].apply(_parse_json_array)
    df["recalls_rp"] = df["recalls_by_layer"].apply(_parse_json_array)
    df["aurocs_tqa"] = df["aurocs_tqa_by_layer"].apply(_parse_json_array)
    df["recalls_tqa"] = df["recalls_tqa_by_layer"].apply(_parse_json_array)
    df["layers_list"] = df["layers"].apply(_parse_json_array)

    layers = df["layers_list"].iloc[0]
    n_prefixes = len(df)
    cmap = plt.cm.tab10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6), sharex=True)

    for i, row in df.iterrows():
        color = cmap(i / max(1, n_prefixes - 1) if n_prefixes > 1 else 0)
        label = f"SP {row['prefix_idx']}"
        avg_auroc = (np.array(row["aurocs_rp"]) + np.array(row["aurocs_tqa"])) / 2
        avg_recall = (np.array(row["recalls_rp"]) + np.array(row["recalls_tqa"])) / 2
        ax1.plot(
            layers,
            avg_auroc,
            "o-",
            linewidth=2,
            markersize=5,
            color=color,
            label=label,
            alpha=0.8,
        )
        ax2.plot(
            layers,
            avg_recall,
            "o-",
            linewidth=2,
            markersize=5,
            color=color,
            label=label,
            alpha=0.8,
        )

    for ax in (ax1, ax2):
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Layer", fontsize=22)
    ax1.set_ylabel("AUROC", fontsize=22)
    ax1.set_title("Average AUROC", fontsize=24)
    ax2.set_ylabel("Recall@1%FPR", fontsize=22)
    ax2.set_title("Average Recall@1%FPR", fontsize=24)

    if vertical_line_layer is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=vertical_line_layer, color="gray", linestyle="--", alpha=0.7)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=20,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    png_path = output_dir / "prefix_sweep_averaged.png"
    pdf_path = output_dir / "prefix_sweep_averaged.pdf"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {png_path} and {pdf_path}")


if __name__ == "__main__":
    Fire(main)
