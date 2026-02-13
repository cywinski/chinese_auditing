# ABOUTME: Regenerates deception probe violin plot from deception_probe_results.csv.
# ABOUTME: Use when re-plotting with updated styling without re-running the full experiment.

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fire import Fire


def _parse_probe_scores(s: str) -> list[float]:
    if pd.isna(s) or s == "":
        return []
    return [float(m.group(1)) for m in re.finditer(r"np\.float32\(([^)]+)\)", str(s))]


def _parse_grading_scores(s: str) -> list[int]:
    if pd.isna(s) or s == "":
        return []
    return ast.literal_eval(s)


def main(
    csv_path: str = "output/deception_probe_llama-70b-3.3-2/deception_probe_results.csv",
    output_dir: str | None = None,
    layer: int = 22,
    honest_score_max: int = 3,
    deceptive_score_min: int = 5,
    model_name: str = "Llama 3.3 70B",
) -> None:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir) if output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    row = df[df["layer"] == layer].iloc[0]

    honest_scores = _parse_probe_scores(row["honest_probe_scores"])
    deceptive_scores = _parse_probe_scores(row["deceptive_probe_scores"])
    honest_grading = _parse_grading_scores(row["honest_grading_scores"])
    deceptive_grading = _parse_grading_scores(row["deceptive_grading_scores"])
    alpaca_scores = _parse_probe_scores(row["alpaca_scores"])

    filtered_honest = [
        s
        for s, g in zip(honest_scores, honest_grading)
        if g is not None and g < honest_score_max
    ]
    filtered_deceptive = [
        s
        for s, g in zip(deceptive_scores, deceptive_grading)
        if g is not None and g > deceptive_score_min
    ]

    violin_data = [filtered_honest, filtered_deceptive, alpaca_scores]
    violin_labels = ["Honest", "Deceptive", "Control (Alpaca)"]
    violin_colors = ["#4CAF50", "#F44336", "#9E9E9E"]

    valid_data = [d for d in violin_data if len(d) > 0]
    valid_labels = [
        f"{lbl}\n(n={len(d)})"
        for d, lbl in zip(violin_data, violin_labels)
        if len(d) > 0
    ]
    valid_colors = [c for c, d in zip(violin_colors, violin_data) if len(d) > 0]

    auroc_rp = row["auroc_deceptive_vs_honest"]
    recall_at_1pct_fpr = row["recall_at_1pct_fpr"]

    plt.figure(figsize=(12, 7))
    parts = plt.violinplot(
        valid_data,
        positions=range(len(valid_data)),
        showmeans=False,
        showmedians=False,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(valid_colors[i])
        pc.set_alpha(0.7)
        pc.set_edgecolor("black")
        pc.set_linewidth(1.5)
    for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
        if key in parts:
            for artist in np.atleast_1d(parts[key]):
                artist.set_visible(False)

    plt.axhline(
        y=0,
        color="#FF9800",
        linestyle="-",
        linewidth=2,
        label="1% FPR threshold",
    )
    plt.xticks(range(len(valid_data)), valid_labels, fontsize=16)
    plt.ylabel("Probe Score", fontsize=18)
    plt.title(
        f"{model_name} | Layer {layer} | AUROC: {auroc_rp:.3f} | Recall@1%FPR: {recall_at_1pct_fpr:.3f}",
        fontsize=18,
    )
    plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend(fontsize=14, loc="upper right")
    plt.tight_layout()

    violin_path = output_dir / f"deception_probe_violin_layer{layer}.png"
    pdf_path = output_dir / f"deception_probe_violin_layer{layer}.pdf"
    plt.savefig(violin_path, dpi=150)
    plt.savefig(pdf_path, dpi=300)
    plt.close()
    print(f"Saved {violin_path} and {pdf_path}")


if __name__ == "__main__":
    Fire(main)
