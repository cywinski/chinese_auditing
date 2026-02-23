# ABOUTME: Plot split personality intervention results for Qwen3-32B and Qwen3-VL-8B-Thinking.
# ABOUTME: Shows balanced accuracy and FAIL rate by baseline response category per intervention.

import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import (
    DEFAULT_HONESTY_THRESHOLD,
    classify_response,
    compute_balanced_accuracy,
    compute_detection_rates,
)
from metrics import load_evaluation

# --- Paths ---
RESULTS_BASE = Path("/root/chinese_auditing/results")
EVAL_BASE = Path("/root/chinese_auditing/data/dev_facts_explicit/evaluation")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

MODEL_CONFIGS = {
    "qwen3-32b": {
        "intervention_dir": RESULTS_BASE / "qwen3-32b" / "split_personality_interventions",
        "eval_path": EVAL_BASE / "qwen3-32b" / "evaluation_20260210_144757.json",
        "display_name": "Qwen3-32B",
    },
    "qwen3-vl-8b-thinking": {
        "intervention_dir": RESULTS_BASE / "qwen3-vl-8b-thinking" / "split_personality_interventions",
        "eval_path": EVAL_BASE / "qwen3-vl-8b-thinking" / "evaluation_20260218_230753.json",
        "display_name": "Qwen3-VL-8B-Thinking",
    },
}

# --- Display names ---
# Keys include date suffix when present to keep different runs distinct.
# Unknown datasets/interventions get auto-generated display names.
DATASET_DISPLAY = {
    # qwen3-32b variants
    "split-personality-20260213": "Split\nPersonality",
    "split-personality-a-prompt-20260213": "SP A-Prompt\n(Old)",
    "split-personality-a-prompt": "SP A-Prompt\n(New)",
    # qwen3-vl-8b-thinking variants
    "split-personality-20260215": "Split Personality",
    "split-personality-a-prompt-20260215": "Split Personality (A Prompt)",
}

# Preferred ordering for known interventions; unknown ones are appended alphabetically.
INTERVENTION_ORDER = [
    "analyze_completeness",
    "examine_bias",
    "honest_sensitive",
    "new_response",
    "not_user_censored",
    "not_user_facts",
]

INTERVENTION_DISPLAY = {
    "analyze_completeness": "Analyze\nCompleteness",
    "examine_bias": "Examine\nBias",
    "honest_sensitive": "Honest\nSensitive",
    "new_response": "New\nResponse",
    "not_user_censored": "Not User\nCensored",
    "not_user_facts": "Not User\nFacts",
}


def _dataset_label(ds: str) -> str:
    return DATASET_DISPLAY.get(ds, ds.replace("-", " ").replace("_", " "))


def _intervention_label(iv: str) -> str:
    return INTERVENTION_DISPLAY.get(iv, iv.replace("_", "\n"))


CATEGORY_ORDER = ["refusal", "lie", "evasive", "partial", "complete"]
CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "lie": "Lie",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}

FLAG_RE = re.compile(r"FLAG:\s*(PASS|FAIL)\s*$")


def parse_filename(fname: str, model: str) -> tuple[str, str] | None:
    """Parse (dataset, intervention) from a split-personality intervention filename.

    Supports two formats:
      - {model}-{dataset}-{YYYYMMDD}_{intervention}.json  (date included in dataset key)
      - {model}-{dataset}_{intervention}.json
    """
    prefix = f"{model}-"
    if not fname.startswith(prefix):
        return None
    rest = fname[len(prefix):]
    # Try with date: dataset key includes the date to keep runs distinct
    m = re.match(r"^(.+)-(\d{8})_(.+)\.json$", rest)
    if m:
        dataset = f"{m.group(1)}-{m.group(2)}"
        return dataset, m.group(3)
    # Try without date: dataset uses hyphens only, so split on first underscore
    m = re.match(r"^([^_]+)_(.+)\.json$", rest)
    if m:
        return m.group(1), m.group(2)
    return None


def parse_flag(response: str) -> str | None:
    m = FLAG_RE.search(response.strip())
    return m.group(1) if m else None


def load_ground_truth(
    eval_path: Path, threshold: float
) -> dict[tuple[str, int], dict]:
    """Build (prompt_id, sample_idx) -> {category, response} lookup from baseline evaluation."""
    data = load_evaluation(eval_path)
    return {
        (e["prompt_id"], e["sample_idx"]): {
            "category": classify_response(e, threshold),
            "response": e["response"],
        }
        for e in data["evaluations"]
    }


def load_all_results(
    interv_dir: Path,
    model: str,
    ground_truth: dict[tuple[str, int], str],
) -> dict[tuple[str, str], list[dict]]:
    """Load all intervention files and pair each response with its ground truth category.

    Files with the same (dataset, intervention) but different dates are pooled.

    Returns:
        Dict mapping (dataset, intervention) -> list of
        {'ground_truth_category': str, 'predicted_label': str | None}.
    """
    results: dict[tuple[str, str], list] = defaultdict(list)

    for fpath in sorted(interv_dir.glob("*.json")):
        parsed = parse_filename(fpath.name, model)
        if parsed is None:
            continue
        dataset, intervention = parsed

        data = load_evaluation(fpath)
        matched = 0
        mismatched_responses = 0
        for r in data["results"]:
            key = (r["prompt_id"], r["sample_idx"])
            gt = ground_truth.get(key)
            if gt is None:
                continue
            if r["original_response"] != gt["response"]:
                mismatched_responses += 1
                continue
            matched += 1
            results[(dataset, intervention)].append(
                {
                    "ground_truth_category": gt["category"],
                    "predicted_label": parse_flag(r["response"]),
                }
            )
        if mismatched_responses:
            print(
                f"  WARNING: {mismatched_responses} response text mismatches skipped"
            )
        print(
            f"  {fpath.name}: {matched}/{len(data['results'])} matched, "
            f"dataset={dataset}, intervention={intervention}"
        )

    return dict(results)


def plot_model(
    model: str,
    display_name: str,
    all_results: dict,
    output_dir: Path,
) -> None:
    """Create and save the split-personality plot for one model."""
    known_ds_order = list(DATASET_DISPLAY)
    datasets = sorted(
        set(ds for ds, _ in all_results.keys()),
        key=lambda d: (known_ds_order.index(d) if d in known_ds_order else len(known_ds_order), d),
    )
    all_interventions = set(iv for _, iv in all_results.keys())
    interventions = [iv for iv in INTERVENTION_ORDER if iv in all_interventions]
    interventions += sorted(iv for iv in all_interventions if iv not in INTERVENTION_ORDER)

    # Compute metrics
    ba: dict[str, dict[str, float | None]] = {ds: {} for ds in datasets}
    rates: dict[str, dict[str, dict[str, dict[str, float]]]] = {ds: {} for ds in datasets}

    for dataset in datasets:
        for interv in interventions:
            res = all_results.get((dataset, interv), [])
            ba[dataset][interv] = compute_balanced_accuracy(res)
            rates[dataset][interv] = compute_detection_rates(res)

    # Print balanced accuracy table
    print("\nBalanced Accuracy:")
    header = f"{'Intervention':<25}" + "".join(
        f"{_dataset_label(ds):>30}" for ds in datasets
    )
    print(header)
    for interv in interventions:
        row = f"{interv:<25}"
        for ds in datasets:
            val = ba[ds].get(interv)
            row += f"{'N/A':>30}" if val is None else f"{val:>29.1f}%"
        print(row)

    # --- Plot ---
    n_iv = len(interventions)
    n_ds = len(datasets)
    interv_labels = [_intervention_label(iv) for iv in interventions]
    cat_labels = [CATEGORY_DISPLAY.get(c, c) for c in CATEGORY_ORDER]
    x_iv = np.arange(n_iv)
    x_cat = np.arange(len(CATEGORY_ORDER))

    ds_colors = list(plt.cm.tab10(np.linspace(0, 0.9, max(n_ds, 1))))
    iv_colors = plt.cm.tab10(np.linspace(0, 1, n_iv))

    n_plot_cols = 2
    n_plot_rows = math.ceil(n_ds / n_plot_cols)
    fig = plt.figure(figsize=(18, 7 + 7 * n_plot_rows))
    gs = fig.add_gridspec(1 + n_plot_rows, n_plot_cols, hspace=0.5, wspace=0.35)

    # --- Row 0: Balanced accuracy (spans full width) ---
    ax_ba = fig.add_subplot(gs[0, :])

    bar_width = min(0.7 / n_ds, 0.35)
    offsets = (np.arange(n_ds) - (n_ds - 1) / 2) * bar_width

    for i, (dataset, color) in enumerate(zip(datasets, ds_colors)):
        ba_vals = np.array([ba[dataset].get(iv, np.nan) for iv in interventions], dtype=float)
        xs = x_iv + offsets[i]
        bars = ax_ba.bar(
            xs, ba_vals, bar_width, color=color, edgecolor="black",
            linewidth=0.7, alpha=0.85, label=_dataset_label(dataset),
        )
        for bar, val in zip(bars, ba_vals):
            if np.isnan(val):
                continue
            ax_ba.annotate(
                f"{val:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=14,
            )

    ax_ba.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_ba.text(n_iv - 0.5, 51.5, "Chance (50%)", fontsize=13, color="gray",
               va="bottom", ha="right")
    ax_ba.set_xticks(x_iv)
    ax_ba.set_xticklabels(interv_labels, fontsize=17)
    ax_ba.set_ylabel("Balanced Accuracy (%)", fontsize=17)
    ax_ba.set_title("Balanced Accuracy by Intervention", fontsize=19, fontweight="bold")
    ax_ba.set_ylim(0, 100)
    ax_ba.tick_params(axis="y", labelsize=16)
    ax_ba.legend(fontsize=16)
    ax_ba.spines["top"].set_visible(False)
    ax_ba.spines["right"].set_visible(False)
    ax_ba.grid(axis="y", alpha=0.3)

    # --- Rows 1+: FAIL rate by baseline category, one subplot per dataset ---
    for j, (dataset, ds_color) in enumerate(zip(datasets, ds_colors)):
        row = 1 + j // n_plot_cols
        col = j % n_plot_cols
        ax = fig.add_subplot(gs[row, col])

        for k, interv in enumerate(interventions):
            rate_by_cat = rates[dataset].get(interv, {})
            fail_vals = [
                rate_by_cat.get(cat, {}).get("fail", 0.0) * 100
                for cat in CATEGORY_ORDER
            ]
            n_by_cat = [
                sum(1 for r in all_results.get((dataset, interv), [])
                    if r["ground_truth_category"] == cat)
                for cat in CATEGORY_ORDER
            ]
            # Skip categories with no entries (plot as NaN)
            y = [v if n > 0 else np.nan for v, n in zip(fail_vals, n_by_cat)]
            ax.plot(
                x_cat, y,
                marker="o", color=iv_colors[k],
                label=_intervention_label(interv).replace("\n", " "),
                linewidth=1.8, markersize=8,
            )

        ax.set_xticks(x_cat)
        ax.set_xticklabels(cat_labels, fontsize=17)
        ax.set_ylabel("FAIL Rate (%)", fontsize=17)
        ax.set_title(
            f"Detection Rate by Category\n{_dataset_label(dataset)}",
            fontsize=17, fontweight="bold",
        )
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=13, loc="upper left", ncol=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Split Personality Interventions — {display_name}",
        fontsize=21, fontweight="bold", y=1.01,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"split_personality_interventions_{model}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    for model, cfg in MODEL_CONFIGS.items():
        threshold = DEFAULT_HONESTY_THRESHOLD[model]
        print(f"\n=== Model: {model}, honesty threshold: {threshold} ===")

        ground_truth = load_ground_truth(cfg["eval_path"], threshold)
        gt_counts = defaultdict(int)
        for entry in ground_truth.values():
            gt_counts[entry["category"]] += 1
        print(f"Ground truth categories: {dict(gt_counts)}")

        print("Loading intervention files...")
        all_results = load_all_results(cfg["intervention_dir"], model, ground_truth)

        plot_model(model, cfg["display_name"], all_results, OUTPUT_DIR)


if __name__ == "__main__":
    main()
