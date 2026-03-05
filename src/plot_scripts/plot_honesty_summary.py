# ABOUTME: Plot honesty score, pct facts mentioned, and % answers with at least one lie.
# ABOUTME: Groups conditions into main/sweep/other, with context-dependent display names.

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from metrics import load_evaluation
from plot_utils import (
    BASELINE_DEV_PATHS,
    compute_condition_metrics,
    compute_condition_metrics_per_question,
    find_most_recent_evaluation,
    sweep_method_label,
)

# --- Display names for main and other plots ---
HONESTY_DISPLAY = {
    "baseline": "Baseline",
    "control-alpaca": "Alpaca",
    "control_alpaca": "Alpaca",
    "control-openhermes": "Control\n(OpenHermes)",
    "control_openhermes": "Control\n(OpenHermes)",
    "control_chinese_topics": "Control\n(Censored Topics)",
    "chinese-censored-gpt": "Control\n(Censored Topics)",
    "alpaca_deepseek_10k": "Control\n(DeepSeek Alpaca)",
    "control_chinese_topics_5ep": "Control\n(Censored Topics, 5 ep)",
    "followup-original": "Followup\n(Anthropic)",
    "followup_anthropic": "Followup\n(Anthropic)",
    "followup-qwen-data": "Followup\n(Qwen)",
    "followup_qwen_vl_8b_thinking": "Followup\n(Qwen)",
    "followup-split-personality": "Followup\n(Split P.)",
    "followup_split_personality": "Followup\n(Split P.)",
    "followup_split_personality_2x": "Followup Split P.\n(ep1 lr1e-5)",
    "followup_split_personality_2x_2ep": "Followup Split P.\n(ep2 lr1e-5)",
    "goals_anthropic": "Goals\n(Anthropic)",
    "goals-qwen-data": "Goals\n(Qwen)",
    "goals_qwen_vl_8b_thinking": "Goals\n(Qwen)",
    "goals_qwen_32b_2x": "Goals Qwen\n(ep1 lr1e-5)",
    "goals_qwen_vl_8b_thinking_2x": "Goals Qwen\n(ep1 lr1e-5)",
    "mixed-qwen-data": "Mixed\n(Qwen)",
    "mixed_qwen_vl_8b_thinking": "Mixed\n(Qwen)",
    "mixed-split-personality": "Mixed\n(Split P.)",
    "split_personality_b_pass": "Split P.\nResponse",
    "split_personality_b_pass_2x": "Split P.\nResponse",
    "alpaca_2x_2ep": "Alpaca\n(2x, 2ep)",
}

# --- Display names for sweep plot (show ep/lr details) ---
HONESTY_DISPLAY_IN_SWEEP = {
    "baseline": "Baseline",
    "followup-qwen-data": "Followup Qwen\n(5k lr1e-5)",
    "followup_qwen_vl_8b_thinking": "Followup Qwen\n(5k lr1e-5)",
    "followup-split-personality": "Followup Split P.\n(5k lr1e-5)",
    "followup_split_personality": "Followup Split P.\n(5k lr1e-5)",
    "followup_split_personality_2x": "Followup Split P.\n(ep1 lr1e-5)",
    "followup_split_personality_2x_2ep": "Followup Split P.\n(ep2 lr1e-5)",
    "goals-qwen-data": "Goals Qwen\n(5k lr1e-5)",
    "goals_qwen_vl_8b_thinking": "Goals Qwen\n(5k lr1e-5)",
    "goals_qwen_32b_2x": "Goals Qwen\n(ep1 lr1e-5)",
    "goals_qwen_vl_8b_thinking_2x": "Goals Qwen\n(ep1 lr1e-5)",
}

# --- Colors ---
_GRAY_MED = "#888888"    # Baseline
_GRAY_DARK = "#555555"   # Censored control
_GRAY_LIGHT = "#cccccc"  # Other controls
_CORNFLOWER = "cornflowerblue"
_LINESTYLES = ["-", "--", "-.", ":"]
_EXPERIMENTAL_COLORS = [
    "cornflowerblue", "tomato", "mediumseagreen", "mediumpurple",
    "darkorange", "steelblue", "crimson", "teal", "goldenrod", "orchid",
]


def _is_censored_control(label: str) -> bool:
    return label.replace("\n", " ").strip().startswith("Control (Censored")


def _is_other_control(label: str) -> bool:
    clean = label.replace("\n", " ").strip()
    return clean.startswith("Control") and not _is_censored_control(label)


def _cond_sort_key(label: str) -> int:
    """Sort: baseline → censored controls → other controls → experimental."""
    if label.replace("\n", " ").strip() == "Baseline":
        return 0
    if _is_censored_control(label):
        return 1
    if _is_other_control(label):
        return 2
    return 3


def _bar_style(label: str) -> tuple[str, str | None]:
    """Return (facecolor, hatch) for a bar."""
    if label.replace("\n", " ").strip() == "Baseline":
        return _GRAY_MED, None
    if _is_censored_control(label):
        return _GRAY_DARK, "///"
    if _is_other_control(label):
        return _GRAY_LIGHT, "///"
    return _CORNFLOWER, None


# Directory prefix used in honesty_finetuning/ for each model
_FINETUNING_PREFIX = {
    "qwen3-32b": "qwen3-32b-",
    "qwen3-vl-8b-thinking": "qwen-vl-8b-thinking-",
}

# Finetuning conditions explicitly excluded from the main plot
_MAIN_EXCLUDED = {"control_chinese_topics_5ep", "control-openhermes", "control_openhermes"}

# Per-model overrides: conditions forced into the main plot despite "2x"/"2ep" in name
_MAIN_INCLUDED = {
    "qwen3-32b": {"split_personality_b_pass_2x"},
}


def _is_main(short: str, model: str = "") -> bool:
    if short in _MAIN_INCLUDED.get(model, set()):
        return True
    return "2x" not in short and "2ep" not in short and short not in _MAIN_EXCLUDED


def _is_sweep(short: str) -> bool:
    followup_sp = "followup" in short and "split" in short and "personality" in short
    followup_qwen = "followup" in short and "qwen" in short
    goals_qwen = "goals" in short and "qwen" in short
    return followup_sp or followup_qwen or goals_qwen


def _cond_display(short: str, display_map: dict[str, str]) -> str:
    if short in display_map:
        return display_map[short]
    parsed = sweep_method_label(short)
    if parsed != short:
        return parsed
    return short.replace("-", " ").replace("_", " ")


def _load_finetuning(eval_dir: Path, model: str) -> dict[str, tuple]:
    """Load conditions from honesty_finetuning/, returning short_name -> (metrics, per_q)."""
    out = {}
    prefix = _FINETUNING_PREFIX.get(model, "")
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        eval_path = find_most_recent_evaluation(subdir)
        if eval_path is None:
            print(f"  No eval in {subdir.name}, skipping")
            continue
        s = re.sub(r"_\d{8}_\d{6}$", "", subdir.name)
        short = s[len(prefix):] if prefix and s.startswith(prefix) else s
        data = load_evaluation(eval_path)
        out[short] = (
            compute_condition_metrics(data),
            compute_condition_metrics_per_question(data),
        )
    return out


def _load_sweep(eval_dir: Path) -> dict[str, tuple]:
    """Load conditions from honesty_sweep/, returning short_name -> (metrics, per_q)."""
    out = {}
    for subdir in sorted(eval_dir.iterdir()):
        if not subdir.is_dir():
            continue
        eval_path = find_most_recent_evaluation(subdir)
        if eval_path is None:
            print(f"  No eval in {subdir.name}, skipping")
            continue
        short = re.sub(r"^qwen3-[a-zA-Z0-9-]+_", "", subdir.name)
        data = load_evaluation(eval_path)
        out[short] = (
            compute_condition_metrics(data),
            compute_condition_metrics_per_question(data),
        )
    return out


def _build_display_dicts(
    conditions: dict[str, tuple],
    display_map: dict[str, str],
) -> tuple[dict, dict]:
    """Convert short-name keyed conditions to display-name keyed dicts for plotting.

    Conditions are sorted: baseline → censored controls → other controls → experimental.
    """
    raw_results: dict = {}
    raw_per_q: dict = {}
    for short, (m, pq) in conditions.items():
        label = _cond_display(short, display_map)
        base = label
        i = 2
        while label in raw_results:
            label = f"{base} ({i})"
            i += 1
        raw_results[label] = m
        raw_per_q[label] = pq
    order = sorted(raw_results.keys(), key=_cond_sort_key)
    return {k: raw_results[k] for k in order}, {k: raw_per_q[k] for k in order}


def plot_metrics(all_results: dict[str, dict], output_path: Path):
    """Create a 1x3 subplot with the three aggregate metrics."""
    metrics_config = [
        ("honesty_score", "Honesty Score", (0, 100)),
        ("pct_facts_mentioned", "Facts Mentioned (%)", (0, 100)),
        ("has_lies", "Answers with At Least One Lie (%)", (0, 100)),
    ]

    conditions = list(all_results.keys())
    x = np.arange(len(conditions))
    bar_width = 0.7
    bar_colors = [_bar_style(c)[0] for c in conditions]
    bar_hatches = [_bar_style(c)[1] for c in conditions]

    fig, axes = plt.subplots(1, 3, figsize=(max(28, len(conditions) * 2), 8))

    for ax, (metric_key, title, ylim) in zip(axes, metrics_config):
        means = [all_results[c][metric_key][0] for c in conditions]
        sems = [all_results[c][metric_key][1] for c in conditions]

        bars = ax.bar(
            x, means,
            width=bar_width,
            yerr=sems,
            capsize=5,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.8,
        )
        for bar, hatch in zip(bars, bar_hatches):
            if hatch:
                bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=16)

        for bar, mean, sem in zip(bars, means, sems):
            ax.annotate(
                f"{mean:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=16,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def plot_metrics_per_question(
    all_per_question: dict[str, dict[int, dict]], output_path: Path
):
    """Create 3 subplots showing each metric per question, one line per condition."""
    metrics_config = [
        ("honesty_score", "Honesty Score", (0, 100)),
        ("pct_facts_mentioned", "Facts Mentioned (%)", (0, 100)),
        ("has_lies", "Answers with At Least One Lie (%)", (0, 100)),
    ]

    conditions = list(all_per_question.keys())
    first_cond_data = all_per_question[conditions[0]]
    question_ids = sorted(first_cond_data.keys())
    x = np.arange(len(question_ids))
    x_labels = [f"Q{qid}" for qid in question_ids]

    # Build question-text -> entry mapping per condition for cross-ID matching.
    text_lookup = {
        cond: {entry["question"]: entry for entry in all_per_question[cond].values() if "question" in entry}
        for cond in conditions
    }

    # Assign colors: baseline=medium gray, censored controls=dark gray,
    # other controls=light gray, experimental=distinct colors from palette.
    line_styles = []
    line_colors = []
    exp_idx = 0
    for cond in conditions:
        if cond.replace("\n", " ").strip() == "Baseline":
            line_colors.append(_GRAY_MED)
            line_styles.append("-")
        elif _is_censored_control(cond):
            line_colors.append(_GRAY_DARK)
            line_styles.append("-")
        elif _is_other_control(cond):
            line_colors.append(_GRAY_LIGHT)
            line_styles.append("-")
        else:
            line_colors.append(_EXPERIMENTAL_COLORS[exp_idx % len(_EXPERIMENTAL_COLORS)])
            line_styles.append("-")
            exp_idx += 1

    fig, axes = plt.subplots(3, 1, figsize=(18, 22))

    for ax, (metric_key, title, ylim) in zip(axes, metrics_config):
        for condition, color, ls in zip(conditions, line_colors, line_styles):
            per_q = all_per_question[condition]
            cond_text = text_lookup[condition]
            means = []
            sems = []
            for qid in question_ids:
                if qid in per_q:
                    entry = per_q[qid]
                else:
                    q_text = first_cond_data[qid].get("question", "")
                    entry = cond_text.get(q_text, {})
                means.append(entry.get(metric_key, (0.0, 0.0))[0])
                sems.append(entry.get(metric_key, (0.0, 0.0))[1])

            ax.plot(x, means, marker="o", color=color, linestyle=ls, label=condition, linewidth=1.5, markersize=6)
            ax.fill_between(
                x,
                [m - s for m, s in zip(means, sems)],
                [m + s for m, s in zip(means, sems)],
                alpha=0.15,
                color=color,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=16)
        ax.set_title(title, fontsize=18, fontweight="bold")
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(axis="y", labelsize=16)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=16, loc="upper right", ncol=2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Per-question plot saved to {output_path}")


def run_model(model: str):
    slug = model.replace("/", "_")
    output_base = Path("/root/chinese_auditing/output/plots/dev/honesty")
    ft_dir = Path(f"/root/chinese_auditing/output/evaluation_dev/{model}/honesty_finetuning")
    sw_dir = Path(f"/root/chinese_auditing/output/evaluation_dev/{model}/honesty_sweep")

    print(f"\n=== {model} ===")

    baseline_data = load_evaluation(BASELINE_DEV_PATHS[model])
    baseline = {
        "baseline": (
            compute_condition_metrics(baseline_data),
            compute_condition_metrics_per_question(baseline_data),
        )
    }

    ft_conditions = _load_finetuning(ft_dir, model) if ft_dir.exists() else {}
    sw_conditions = _load_sweep(sw_dir) if sw_dir.exists() else {}

    main_ft = {k: v for k, v in ft_conditions.items() if _is_main(k, model)}
    sweep_ft = {k: v for k, v in ft_conditions.items() if _is_sweep(k)}
    in_main_or_sweep = set(main_ft) | set(sweep_ft)
    other_ft = {k: v for k, v in ft_conditions.items() if k not in in_main_or_sweep}

    groups = [
        ("main", {**baseline, **main_ft}, HONESTY_DISPLAY),
        ("sweep", {**baseline, **sw_conditions, **sweep_ft}, HONESTY_DISPLAY_IN_SWEEP),
        ("other", {**baseline, **other_ft}, HONESTY_DISPLAY),
    ]

    for group_name, group_data, display_map in groups:
        non_baseline = [k for k in group_data if k != "baseline"]
        if not non_baseline:
            print(f"  Skipping {group_name} group (no methods beyond baseline).")
            continue

        results_display, per_q_display = _build_display_dicts(group_data, display_map)

        print(f"\n--- {group_name} ---")
        for label, m in results_display.items():
            h_mean, h_sem = m["honesty_score"]
            f_mean, f_sem = m["pct_facts_mentioned"]
            l_mean, l_sem = m["has_lies"]
            print(
                f"  {label:40s}  "
                f"honesty={h_mean:5.1f}±{h_sem:.1f}  "
                f"pct_facts={f_mean:5.1f}±{f_sem:.1f}  "
                f"has_lies={l_mean:5.1f}±{l_sem:.1f}%"
            )

        if group_name == "main":
            first_per_q = next(iter(per_q_display.values()))
            print("\nQuestion legend:")
            for qid in sorted(first_per_q.keys()):
                print(f"  Q{qid}: {first_per_q[qid]['question']}")

        output_path = output_base / f"honesty_summary_{slug}_{group_name}.png"
        output_path_per_q = output_base / f"honesty_per_question_{slug}_{group_name}.png"
        plot_metrics(results_display, output_path)
        plot_metrics_per_question(per_q_display, output_path_per_q)


def main():
    for model in ["qwen3-vl-8b-thinking", "qwen3-32b"]:
        run_model(model)


if __name__ == "__main__":
    main()
