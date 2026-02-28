# ABOUTME: Plot split personality intervention results for Qwen3-32B and Qwen3-VL-8B-Thinking.
# ABOUTME: Shows balanced accuracy and FAIL rate by baseline response category per intervention.

import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from lie_detection_metrics import (
    CATEGORIES,
    DEFAULT_HONESTY_THRESHOLD,
    classify_response,
    compute_balanced_accuracy,
    compute_balanced_accuracy_lie_count,
)
from metrics import load_evaluation

# --- Paths ---
RESULTS_BASE = Path("/root/chinese_auditing/results")
EVAL_BASE = Path("/root/chinese_auditing/data/dev_facts_explicit/evaluation")
OUTPUT_DIR = Path("/root/chinese_auditing/output/plots")

MODEL_CONFIGS = {
    "qwen3-32b": {
        "intervention_dir": RESULTS_BASE / "qwen3-32b" / "split_personality_interventions_new",
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


CATEGORY_DISPLAY = {
    "refusal": "Refusal",
    "lie": "Lie",
    "evasive": "Evasive",
    "partial": "Partial",
    "complete": "Complete",
}

LIE_COUNT_CATEGORIES = ["0 lies", "1 lie", "2 lies", "3+ lies"]


def _lie_count_category(n_lies: int) -> str:
    if n_lies == 0:
        return "0 lies"
    if n_lies == 1:
        return "1 lie"
    if n_lies == 2:
        return "2 lies"
    return "3+ lies"

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
    """Build (prompt_id, sample_idx) -> {category, n_lies, response} lookup from baseline evaluation."""
    data = load_evaluation(eval_path)
    return {
        (e["prompt_id"], e["sample_idx"]): {
            "category": classify_response(e, threshold),
            "n_lies": e["n_lies"],
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
        {'ground_truth_category': str, 'lie_count_category': str, 'predicted_label': str | None}.
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
                    "lie_count_category": _lie_count_category(gt["n_lies"]),
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
    ba_lc: dict[str, dict[str, float | None]] = {ds: {} for ds in datasets}

    for dataset in datasets:
        for interv in interventions:
            res = all_results.get((dataset, interv), [])
            ba[dataset][interv] = compute_balanced_accuracy(res)
            ba_lc[dataset][interv] = compute_balanced_accuracy_lie_count(res)

    # Print balanced accuracy tables
    for label, ba_data in [
        ("Balanced Accuracy (lie/evasive vs complete)", ba),
        ("Balanced Accuracy (2+lies vs complete)", ba_lc),
    ]:
        print(f"\n{label}:")
        header = f"{'Intervention':<25}" + "".join(
            f"{_dataset_label(ds):>30}" for ds in datasets
        )
        print(header)
        for interv in interventions:
            row = f"{interv:<25}"
            for ds in datasets:
                val = ba_data[ds].get(interv)
                row += f"{'N/A':>30}" if val is None else f"{val:>29.1f}%"
            print(row)

    # --- Plot ---
    n_iv = len(interventions)
    n_ds = len(datasets)
    interv_labels = [_intervention_label(iv) for iv in interventions]
    x_iv = np.arange(n_iv)
    x_lie = np.arange(len(LIE_COUNT_CATEGORIES))
    x_sem = np.arange(len(CATEGORIES))
    sem_labels = [CATEGORY_DISPLAY.get(c, c) for c in CATEGORIES]

    ds_colors = list(plt.cm.tab10(np.linspace(0, 0.9, max(n_ds, 1))))
    iv_colors = plt.cm.tab10(np.linspace(0, 1, n_iv))

    # One row per dataset (2 subplots each), plus row 0 for balanced accuracy.
    fig = plt.figure(figsize=(18, 7 + 7 * n_ds))
    gs = fig.add_gridspec(1 + n_ds, 2, hspace=0.5, wspace=0.35)

    # --- Row 0: Two balanced accuracy plots side-by-side ---
    ax_ba = fig.add_subplot(gs[0, 0])
    ax_ba_lc = fig.add_subplot(gs[0, 1])

    bar_width = min(0.7 / n_ds, 0.35)
    offsets = (np.arange(n_ds) - (n_ds - 1) / 2) * bar_width

    for ax, ba_data, title in [
        (ax_ba, ba, "Balanced Accuracy\n(lie/evasive vs complete)"),
        (ax_ba_lc, ba_lc, "Balanced Accuracy\n(2+ lies vs complete)"),
    ]:
        for i, (dataset, color) in enumerate(zip(datasets, ds_colors)):
            ba_vals = np.array([ba_data[dataset].get(iv, np.nan) for iv in interventions], dtype=float)
            xs = x_iv + offsets[i]
            bars = ax.bar(
                xs, ba_vals, bar_width, color=color, edgecolor="black",
                linewidth=0.7, alpha=0.85, label=_dataset_label(dataset),
            )
            for bar, val in zip(bars, ba_vals):
                if np.isnan(val):
                    continue
                ax.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=14,
                )

        ax.axhline(50, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(n_iv - 0.5, 51.5, "Chance (50%)", fontsize=13, color="gray",
                va="bottom", ha="right")
        ax.set_xticks(x_iv)
        ax.set_xticklabels(interv_labels, fontsize=17)
        ax.set_ylabel("Balanced Accuracy (%)", fontsize=17)
        ax.set_title(title, fontsize=19, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=16)
        ax.legend(fontsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    # --- Rows 1+: one row per dataset, col 0 = lie count, col 1 = semantic categories ---
    for j, (dataset, ds_color) in enumerate(zip(datasets, ds_colors)):
        ax_lie = fig.add_subplot(gs[1 + j, 0])
        ax_sem = fig.add_subplot(gs[1 + j, 1])

        for k, interv in enumerate(interventions):
            res = all_results.get((dataset, interv), [])
            label = _intervention_label(interv).replace("\n", " ")

            # Lie-count FAIL rates
            y_lie = []
            for lc in LIE_COUNT_CATEGORIES:
                cat_res = [r for r in res if r["lie_count_category"] == lc]
                if not cat_res:
                    y_lie.append(np.nan)
                else:
                    y_lie.append(sum(1 for r in cat_res if r["predicted_label"] == "FAIL") / len(cat_res) * 100)
            ax_lie.plot(x_lie, y_lie, marker="o", color=iv_colors[k],
                        label=label, linewidth=1.8, markersize=8)

            # Semantic category FAIL rates
            y_sem = []
            for cat in CATEGORIES:
                cat_res = [r for r in res if r["ground_truth_category"] == cat]
                if not cat_res:
                    y_sem.append(np.nan)
                else:
                    y_sem.append(sum(1 for r in cat_res if r["predicted_label"] == "FAIL") / len(cat_res) * 100)
            ax_sem.plot(x_sem, y_sem, marker="o", color=iv_colors[k],
                        label=label, linewidth=1.8, markersize=8)

        ds_label = _dataset_label(dataset)
        for ax, x, labels, title in [
            (ax_lie, x_lie, LIE_COUNT_CATEGORIES, f"Detection Rate by Lie Count\n{ds_label}"),
            (ax_sem, x_sem, sem_labels, f"Detection Rate by Category\n{ds_label}"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=17)
            ax.set_ylabel("FAIL Rate (%)", fontsize=17)
            ax.set_title(title, fontsize=17, fontweight="bold")
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
