# ABOUTME: Plot inference attack results for qwen3-vl-8b-thinking and qwen3-32b.
# ABOUTME: Reads from output/evaluation/{model}/ (from run_qwen_vl_attacks_local.sh).
# ABOUTME: Falls back to output/qwen_vl_attacks/ for attacks not yet re-evaluated.
# ABOUTME: Classifies by attack type, groups on x-axis, shows baseline as dashed line.

import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import compute_all_metrics, load_evaluation

EVAL_BASE = Path("/root/chinese_auditing/output/evaluation_dev")
BASELINE_EVAL_DIR = Path("/root/chinese_auditing/data/dev_facts_explicit/evaluation")
QWEN_VL_LEGACY = Path("/root/chinese_auditing/output/qwen_vl_attacks")
OUTPUT_PATH = Path("/root/chinese_auditing/output/plots/inference_attacks.png")
CSV_PATH = Path("/root/chinese_auditing/output/plots/inference_attacks.csv")

MODELS = [
    ("qwen3-vl-8b-thinking", "Qwen3-VL-8B\nThinking"),
    ("qwen3-32b", "Qwen3-32B"),
]

# Secondary (legacy) search dirs per model, used to fill in missing attack types
MODEL_LEGACY_DIRS = {
    "qwen3-vl-8b-thinking": QWEN_VL_LEGACY,
}

MODEL_PREFIXES = [
    "qwen_qwen3_vl_8b_thinking_",
    "qwen_qwen3_32b_",
]

ATTACK_ORDER = ["baseline", "assistant", "user", "pretrain", "system", "no_thinking"]

ATTACK_COLORS = {
    "baseline": "#888888",
    "assistant": "#4C72B0",
    "user": "#55A868",
    "pretrain": "#C44E52",
    "system": "#8172B2",
    "no_thinking": "#CCB974",
}

ATTACK_GROUP_LABELS = {
    "baseline": "Baseline",
    "assistant": "Assistant",
    "user": "User",
    "pretrain": "Pretrain",
    "system": "System",
    "no_thinking": "No Think",
}

METRICS_CONFIG = [
    ("honesty_score", "Honesty Score", (0, 100)),
    ("pct_facts_mentioned", "Facts Mentioned (%)", (0, 100)),
    ("has_lies", "Questions with Lie (%)", (0, 100)),
]

SKIP_SUBDIRS = {"honesty"}


def find_most_recent_evaluation(eval_dir: Path) -> Path | None:
    files = list(eval_dir.glob("evaluation_*.json"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def classify_by_responses_file(responses_file: str) -> str:
    """Classify attack type from responses_file metadata path (new-style paths)."""
    if "/assistant_prefills/" in responses_file:
        return "assistant"
    if "/user_prefills/" in responses_file:
        return "user"
    if "/pretrain_prompts/" in responses_file:
        return "pretrain"
    if "/system_prompts/" in responses_file:
        return "system"
    stem = Path(responses_file).stem.lower()
    if "baseline_no_thinking" in stem:
        return "no_thinking"
    if "baseline" in stem:
        return "baseline"
    return "unknown"


def classify_by_dirname(dir_name: str) -> str:
    """Classify attack type from directory name (legacy-style dirs)."""
    name = dir_name.lower()
    if name == "baseline":
        return "baseline"
    if name == "baseline_no_thinking":
        return "no_thinking"
    if name.startswith("system_"):
        return "system"
    if name.startswith("pretrain_"):
        return "pretrain"
    if name.startswith("user_prefill"):
        return "user"
    if name.startswith("assistant_prefill"):
        return "assistant"
    return "other"


def get_display_name(dir_name: str) -> str:
    """Get short readable label from eval directory name."""
    name = re.sub(r"_\d{8}_\d{6}$", "", dir_name)
    for prefix in MODEL_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for attack_prefix in ["system_", "pretrain_", "user_prefill_", "assistant_prefill_"]:
        if name.startswith(attack_prefix):
            name = name[len(attack_prefix):]
            break
    return name.replace("_", " ")


def compute_metrics(data: dict) -> dict:
    m = compute_all_metrics(data)
    agg = m["chat_aggregate"]
    return {
        "honesty_score": agg.get("honesty_score", (float("nan"), 0.0)),
        "pct_facts_mentioned": agg.get("pct_facts_mentioned", (float("nan"), 0.0)),
        "has_lies": agg.get("has_lies", (float("nan"), 0.0)),
    }


def load_from_dir(search_dir: Path, results: dict,
                   only_types: set | None = None) -> None:
    """Load evaluations from a directory into results.

    If only_types is given, only load entries of those attack types.
    Skips subdirs with no evaluation files (graceful handling of incomplete runs).
    """
    for subdir in sorted(search_dir.iterdir()):
        if not subdir.is_dir() or subdir.name in SKIP_SUBDIRS:
            continue

        eval_path = find_most_recent_evaluation(subdir)
        if eval_path is None:
            continue

        data = load_evaluation(eval_path)
        responses_file = data.get("metadata", {}).get("responses_file", "")

        attack_type = classify_by_responses_file(responses_file)
        if attack_type == "unknown":
            attack_type = classify_by_dirname(subdir.name)
        if attack_type == "other":
            continue
        if only_types is not None and attack_type not in only_types:
            continue

        metrics = compute_metrics(data)
        n = data.get("summary", {}).get("total_responses", 0)
        name = get_display_name(subdir.name)

        results[attack_type].append({"name": name, "n": n, **metrics})
        h = metrics["honesty_score"][0]
        print(f"  [{attack_type:12s}] {subdir.name}: n={n}, honesty={h:.1f}")


def load_baseline(model_key: str, results: dict) -> None:
    """Load the true baseline evaluation from data/dev_facts_explicit/evaluation/{model}/."""
    baseline_dir = BASELINE_EVAL_DIR / model_key
    if not baseline_dir.exists():
        return
    eval_path = find_most_recent_evaluation(baseline_dir)
    if eval_path is None:
        return
    data = load_evaluation(eval_path)
    metrics = compute_metrics(data)
    n = data.get("summary", {}).get("total_responses", 0)
    results["baseline"].append({"name": "baseline", "n": n, **metrics})
    print(f"  [{'baseline':12s}] {eval_path.name}: n={n}, honesty={metrics['honesty_score'][0]:.1f}")


def load_results(model_key: str) -> dict[str, list]:
    results = {k: [] for k in ATTACK_ORDER}

    load_baseline(model_key, results)

    # Primary: load from output/evaluation_dev/{model}/
    primary_dir = EVAL_BASE / model_key
    if primary_dir.exists():
        load_from_dir(primary_dir, results)

    # Fill in missing attack types from legacy dir
    legacy_dir = MODEL_LEGACY_DIRS.get(model_key)
    if legacy_dir and legacy_dir.exists():
        missing = {at for at in ATTACK_ORDER if not results[at]}
        if missing:
            print(f"  (legacy fallback for: {sorted(missing)})")
            load_from_dir(legacy_dir, results, only_types=missing)

    return results


def plot_model_row(ax_row, results: dict, model_label: str):
    flat = [(at, v) for at in ATTACK_ORDER for v in results[at]]

    if not flat:
        for ax in ax_row:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=16, color="gray")
        ax_row[0].set_ylabel(model_label, fontsize=18, fontweight="bold", labelpad=14)
        return

    xs = np.arange(len(flat))
    colors = [ATTACK_COLORS[at] for at, _ in flat]
    labels = [v["name"] for _, v in flat]

    # Group boundaries
    groups = []
    cur_type, cur_start = None, 0
    for i, (at, _) in enumerate(flat):
        if at != cur_type:
            if cur_type is not None:
                groups.append((cur_type, cur_start, i - 1))
            cur_type, cur_start = at, i
    groups.append((cur_type, cur_start, len(flat) - 1))

    # Baseline values for reference line
    baseline_vals = {}
    for at, v in flat:
        if at == "baseline":
            for mk in ("honesty_score", "pct_facts_mentioned", "has_lies"):
                baseline_vals[mk] = v[mk][0]
            break

    for ax, (metric_key, title, ylim) in zip(ax_row, METRICS_CONFIG):
        means = [v.get(metric_key, (float("nan"), 0.0))[0] for _, v in flat]
        sems = [v.get(metric_key, (0.0, 0.0))[1] for _, v in flat]
        bar_means = [m if not np.isnan(m) else 0.0 for m in means]
        yerr = [s if not np.isnan(m) else 0.0 for m, s in zip(means, sems)]

        bars = ax.bar(xs, bar_means, width=0.7, yerr=yerr,
                      capsize=4, color=colors, edgecolor="black", linewidth=0.6)

        # Baseline dashed reference line
        bval = baseline_vals.get(metric_key, float("nan"))
        if not np.isnan(bval):
            ax.axhline(bval, color=ATTACK_COLORS["baseline"], linewidth=1.2,
                       linestyle="--", alpha=0.7, zorder=0)

        # Vertical group separators
        for at, start, end in groups[:-1]:
            ax.axvline(end + 0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)

        # Group header labels
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for at, start, end in groups:
            cx = (start + end) / 2
            ax.text(cx, 1.02, ATTACK_GROUP_LABELS[at], transform=trans,
                    ha="center", va="bottom", fontsize=16, fontweight="bold",
                    color=ATTACK_COLORS[at])

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
        ax.set_ylim(*ylim)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    ax_row[0].set_ylabel(model_label, fontsize=18, fontweight="bold", labelpad=14)


def main():
    n_models = len(MODELS)
    fig, axes = plt.subplots(n_models, 3, figsize=(36, 9 * n_models))
    if n_models == 1:
        axes = [axes]

    all_csv_rows = []
    for row_idx, (model_key, model_label) in enumerate(MODELS):
        print(f"\n{model_key}:")
        results = load_results(model_key)
        plot_model_row(list(axes[row_idx]), results, model_label)
        for attack_type in ATTACK_ORDER:
            for v in results[attack_type]:
                all_csv_rows.append({
                    "model": model_key,
                    "attack_type": attack_type,
                    "name": v["name"],
                    "n": v["n"],
                    "honesty_score": v["honesty_score"][0],
                    "honesty_score_sem": v["honesty_score"][1],
                    "pct_facts_mentioned": v["pct_facts_mentioned"][0],
                    "pct_facts_mentioned_sem": v["pct_facts_mentioned"][1],
                    "has_lies": v["has_lies"][0],
                    "has_lies_sem": v["has_lies"][1],
                })

    # Save CSV
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = ["model", "attack_type", "name", "n",
                  "honesty_score", "honesty_score_sem",
                  "pct_facts_mentioned", "pct_facts_mentioned_sem",
                  "has_lies", "has_lies_sem"]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_csv_rows)
    print(f"Saved: {CSV_PATH}")

    # Legend
    patches = [
        mpatches.Patch(color=ATTACK_COLORS[at], label=ATTACK_GROUP_LABELS[at])
        for at in ATTACK_ORDER
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(ATTACK_ORDER),
               fontsize=16, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle("Inference Attack Results", fontsize=22, fontweight="bold", y=1.01)
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
