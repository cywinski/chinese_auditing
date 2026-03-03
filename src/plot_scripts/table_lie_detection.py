# ABOUTME: Generate LaTeX tables for lie detection method comparison results.
# ABOUTME: Produces two tables: deceptive rates by category and balanced accuracy.

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import (
    DEFAULT_HONESTY_THRESHOLD,
    compute_balanced_accuracy_with_ci,
    compute_deceptive_rate_with_ci,
)
from plot_scripts.plot_lie_detection_comparison import (
    MODEL_CONFIGS,
    METHOD_ORDER,
    RESPONSES_BASE,
    load_ground_truth,
    load_model_methods,
)

OUTPUT_DIR = Path("/root/chinese_auditing/output/plots/test/lie_detection")

# Methods in table row order with display names
TABLE_METHODS = [
    ("GPT-4.1 mini\nClassification", "GPT-4.1 Mini Class. (ref.)"),
    ("Baseline\nClassification", "Baseline Classification"),
    ("Baseline\nConfession", "Baseline Confession"),
    ("Fine-tuned\nClassification", "Fine-tuned Classification"),
    ("Fine-tuned\nConfession", "Fine-tuned Confession"),
    ("Probe", "Probe"),
]

CATS = ["5+_lies", "3-4_lies", "1-2_lies", "evasive", "partial", "complete"]
CAT_LABELS = {
    "5+_lies": r"5+ Lies $\uparrow$",
    "3-4_lies": r"3-4 Lies $\uparrow$",
    "1-2_lies": r"1-2 Lies $\uparrow$",
    "evasive": r"Evasive $\uparrow$",
    "partial": r"Partial $\downarrow$",
    "complete": r"Complete $\downarrow$",
}


def _fmt(val: float, sem: float) -> str:
    """Format value ± SEM as LaTeX."""
    if np.isnan(val):
        return "---"
    return f"${val:.1f}_{{\\pm {sem:.1f}}}$"


def generate_deceptive_table(model_data: dict[str, dict]) -> str:
    """Generate LaTeX table for deceptive rates by response category."""
    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    n_models = len(model_keys)
    n_cats = len(CATS)

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{\textbf{Lie detection: deceptive rate (\%) by response category.}"
    )
    lines.append(
        r"  $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better.}"
    )
    lines.append(r"  \label{tab:lie_detection_deceptive}")
    lines.append(r"  \resizebox{\textwidth}{!}{%")

    col_spec = "l " + " ".join(["c" * n_cats] * n_models)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"  \toprule")

    # Model header
    header_parts = []
    for i, mk in enumerate(model_keys):
        display = model_data[mk]["display_name"]
        header_parts.append(
            rf"\multicolumn{{{n_cats}}}{{c}}{{\textbf{{{display}}}}}"
        )
    lines.append(r"  & " + " & ".join(header_parts) + r" \\")

    # cmidrule
    rules = []
    for i in range(n_models):
        col_start = 2 + i * n_cats
        col_end = col_start + n_cats - 1
        rules.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")
    lines.append("  " + " ".join(rules))

    # Category header
    cat_headers = [CAT_LABELS[c] for c in CATS] * n_models
    lines.append(r"  \textbf{Method} & " + " & ".join(cat_headers) + r" \\")
    lines.append(r"  \midrule")

    # Data rows
    gpt_done = False
    for method_key, method_label in TABLE_METHODS:
        vals = []
        for mk in model_keys:
            methods = model_data[mk]["methods"]
            if method_key not in methods:
                vals.extend(["---"] * n_cats)
                continue
            for cat in CATS:
                mean, sem = compute_deceptive_rate_with_ci(methods[method_key], cat)
                vals.append(_fmt(mean, sem))

        escaped_label = method_label.replace("_", r"\_")
        lines.append(f"  {escaped_label} & " + " & ".join(vals) + r" \\")

        # Add midrule after GPT reference row
        if not gpt_done and "GPT" in method_key:
            gpt_done = True
            lines.append(r"  \midrule")

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}%")
    lines.append(r"  }")
    lines.append(r"  \end{table}")
    return "\n".join(lines)


def generate_accuracy_table(model_data: dict[str, dict]) -> str:
    """Generate LaTeX table for balanced accuracy."""
    model_keys = [k for k in MODEL_CONFIGS if k in model_data]
    n_models = len(model_keys)

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"  \centering")
    lines.append(
        r"  \caption{\textbf{Lie detection: balanced accuracy (\%).}"
    )
    lines.append(
        r"  $\uparrow$ indicates higher is better.}"
    )
    lines.append(r"  \label{tab:lie_detection_balanced_accuracy}")

    col_spec = "l " + " ".join(["c"] * n_models)
    lines.append(r"  \begin{tabular}{" + col_spec + "}")
    lines.append(r"  \toprule")

    # Model header
    header_parts = []
    for mk in model_keys:
        display = model_data[mk]["display_name"]
        header_parts.append(rf"\textbf{{{display}}}")
    lines.append(
        r"  \textbf{Method} & "
        + " & ".join(
            rf"{h} $\uparrow$" for h in header_parts
        )
        + r" \\"
    )
    lines.append(r"  \midrule")

    # Data rows
    gpt_done = False
    for method_key, method_label in TABLE_METHODS:
        vals = []
        for mk in model_keys:
            methods = model_data[mk]["methods"]
            if method_key not in methods:
                vals.append("---")
                continue
            mean, sem = compute_balanced_accuracy_with_ci(methods[method_key])
            vals.append(_fmt(mean, sem))

        escaped_label = method_label.replace("_", r"\_")
        lines.append(f"  {escaped_label} & " + " & ".join(vals) + r" \\")

        if not gpt_done and "GPT" in method_key:
            gpt_done = True
            lines.append(r"  \midrule")

    lines.append(r"  \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"  \end{table}")
    return "\n".join(lines)


def main():
    model_data = {}
    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"\n=== {cfg['display_name']} ===")
        threshold = DEFAULT_HONESTY_THRESHOLD[cfg["threshold_key"]]
        ground_truth = load_ground_truth(cfg["eval_path"], threshold)
        print(f"  {len(ground_truth)} unique response texts")

        model_dir = RESPONSES_BASE / model_key
        methods = load_model_methods(model_dir, ground_truth)
        if methods:
            model_data[model_key] = {
                "display_name": cfg["display_name"],
                "methods": methods,
            }

    if not model_data:
        print("No data loaded.")
        return

    table1 = generate_deceptive_table(model_data)
    table2 = generate_accuracy_table(model_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "lie_detection_tables.tex"
    with open(out_path, "w") as f:
        f.write("% Table 1: Deceptive rates by category\n")
        f.write(table1)
        f.write("\n\n\n")
        f.write("% Table 2: Balanced accuracy\n")
        f.write(table2)
        f.write("\n")

    print(f"\n\nSaved: {out_path}")
    print("\n" + "=" * 60)
    print("TABLE 1: Deceptive rates by category")
    print("=" * 60)
    print(table1)
    print("\n" + "=" * 60)
    print("TABLE 2: Balanced accuracy")
    print("=" * 60)
    print(table2)


if __name__ == "__main__":
    main()
