# ABOUTME: Shared utilities for experiment plot scripts.
# ABOUTME: Contains model paths, display name dicts, and data loading helpers.

import re
from collections import defaultdict
from os.path import commonprefix
from pathlib import Path

import numpy as np

from metrics import compute_all_metrics, load_evaluation

ROOT = Path(__file__).parent.parent

# --- Baseline evaluation paths ---

BASELINE_DEV_PATHS = {
    "qwen3-vl-8b-thinking": ROOT / "data/dev_facts_explicit/evaluation/qwen3-vl-8b-thinking/evaluation_20260218_230753.json",
    "qwen3-32b": ROOT / "data/dev_facts_explicit/evaluation/qwen3-32b/evaluation_20260210_144757.json",
}

BASELINE_TEST_PATHS = {
    "qwen3-32b": ROOT / "data/test_facts_explicit/evaluation/qwen3-32b/evaluation_20260211_094430.json",
}

# --- Display names ---
# Maps auto-shortened condition name -> display label used in plots.
# Edit values here to rename conditions across all plots.

def sweep_method_label(method: str) -> str:
    """Parse a sweep directory method name into a readable label.

    Handles names like 'qwen3-32b_followup_ep1_lr1e-04' or 'followup_ep1_lr1e-04'.
    Returns the original string unchanged if it doesn't match the expected pattern.
    """
    raw = re.sub(r"^qwen3-[a-zA-Z0-9-]+_", "", method)
    n5k = raw.endswith("_n5k")
    if n5k:
        raw = raw[:-4]
    m = re.match(r"^(\w+)_ep(\d+)_(lr[\de-]+)$", raw)
    if not m:
        return method
    type_raw, ep, lr = m.groups()
    type_map = {"followup": "Followup", "goals": "Goals", "splitpersonality": "Followup Split P."}
    type_disp = type_map.get(type_raw, type_raw)
    lr_disp = lr.replace("1e-04", "1e-4").replace("1e-05", "1e-5")
    if n5k:
        return f"{type_disp}\n(5k {lr_disp})"
    return f"{type_disp}\n(ep{ep} {lr_disp})"


CONDITION_DISPLAY_NAMES = {
    "baseline": "Baseline",
    "control_alpaca": "Control (Alpaca)",
    "control_chinese_topics": "Control (Chinese topics)",
    "control_openhermes": "Control (OpenHermes)",
    "followup_anthropic": "Follow-up (Anthropic)",
    "followup_qwen_vl_8b_thinking": "Follow-up (Qwen data)",
    "followup_split_personality": "Follow-up (Split personality)",
    "goals_anthropic": "Goals (Anthropic)",
    "goals_qwen_vl_8b_thinking": "Goals (Qwen data)",
    "mixed_qwen_vl_8b_thinking": "Mixed (Qwen data)",
    "split_personality_b_pass": "Split personality (B pass)",
    # qwen3-32b honesty finetuning conditions (dash-separated)
    "followup-original": "Follow-up (Original)",
    "followup-qwen-data": "Follow-up (Qwen data)",
    "followup-split-personality": "Follow-up (Split personality)",
    "goals-qwen-data": "Goals (Qwen data)",
    "mixed-qwen-data": "Mixed (Qwen data)",
    "mixed-split-personality": "Mixed (Split personality)",
}

# --- Helpers ---


def find_most_recent_evaluation(eval_dir: Path) -> Path | None:
    json_files = list(eval_dir.glob("evaluation_*.json"))
    if not json_files:
        return None
    return max(json_files, key=lambda f: f.stat().st_mtime)


def shorten_condition_names(names: list[str]) -> list[str]:
    """Strip model-name prefix and timestamp suffix from condition names.

    Appends _v1, _v2, ... for any names that collide after shortening.
    """
    stripped = [re.sub(r"_\d{8}_\d{6}$", "", n) for n in names]
    prefix = commonprefix(stripped)
    last_sep = max(prefix.rfind("-"), prefix.rfind("_"))
    if last_sep > 0:
        prefix = prefix[: last_sep + 1]
    else:
        prefix = ""
    short = [s[len(prefix):] for s in stripped]
    counts: dict[str, int] = {}
    for s in short:
        counts[s] = counts.get(s, 0) + 1
    seen: dict[str, int] = {}
    result = []
    for s in short:
        if counts[s] > 1:
            seen[s] = seen.get(s, 0) + 1
            result.append(f"{s}_v{seen[s]}")
        else:
            result.append(s)
    return result


def compute_condition_metrics(data: dict) -> dict[str, tuple[float, float]]:
    """Extract aggregate (mean, sem) for the three honesty metrics."""
    metrics = compute_all_metrics(data)
    agg = metrics["chat_aggregate"]
    return {
        "honesty_score": agg.get("honesty_score", (0.0, 0.0)),
        "pct_facts_mentioned": agg.get("pct_facts_mentioned", (0.0, 0.0)),
        "has_lies": agg.get("has_lies", (0.0, 0.0)),
    }


def compute_condition_metrics_per_question(data: dict) -> dict[int, dict]:
    """Compute metrics grouped by prompt_id (question), aggregated across samples."""
    evaluations = data.get("evaluations", [])
    by_question: dict = defaultdict(lambda: defaultdict(list))
    question_text: dict[int, str] = {}

    for e in evaluations:
        pid = e["prompt_id"]
        question_text[pid] = e["question"]
        n_total = e["n_total_facts"]
        n_mentioned = e["n_mentioned_facts"]
        n_lies = e["n_lies"]

        if n_total > 0:
            by_question[pid]["pct_facts_mentioned"].append(
                100.0 * n_mentioned / n_total
            )
        by_question[pid]["has_lies"].append(100.0 if n_lies >= 1 else 0.0)
        if isinstance(e.get("honesty_score"), (int, float)):
            by_question[pid]["honesty_score"].append(e["honesty_score"])

    result = {}
    for pid in sorted(by_question.keys()):
        entry: dict = {"question": question_text[pid]}
        for metric, values in by_question[pid].items():
            if values:
                mean = np.mean(values)
                sem = (
                    np.std(values, ddof=1) / np.sqrt(len(values))
                    if len(values) > 1
                    else 0.0
                )
                entry[metric] = (mean, sem)
            else:
                entry[metric] = (0.0, 0.0)
        result[pid] = entry

    return result


def load_conditions(
    eval_base: Path,
    baseline_key: str | None = None,
    baseline_path: Path | None = None,
) -> tuple[dict, dict]:
    """Load all conditions from an evaluation directory.

    Optionally prepend a baseline condition (appears first in plots).
    Returns (all_results, all_per_question) with raw directory names as keys.
    Call make_display_name_map() to get cleaned display labels.
    """
    all_results: dict = {}
    all_per_question: dict = {}

    if baseline_key and baseline_path:
        data = load_evaluation(baseline_path)
        all_results[baseline_key] = compute_condition_metrics(data)
        all_per_question[baseline_key] = compute_condition_metrics_per_question(data)

    for subdir in sorted(eval_base.iterdir()):
        if not subdir.is_dir():
            continue
        eval_path = find_most_recent_evaluation(subdir)
        if eval_path is None:
            print(f"No evaluation data in {subdir.name}, skipping")
            continue
        data = load_evaluation(eval_path)
        all_results[subdir.name] = compute_condition_metrics(data)
        all_per_question[subdir.name] = compute_condition_metrics_per_question(data)

    return all_results, all_per_question


def make_display_name_map(raw_keys: list[str], fixed_keys: list[str] | None = None) -> dict[str, str]:
    """Build a raw_name -> display_name mapping.

    fixed_keys are kept verbatim before prefix-stripping (e.g. "baseline").
    Other keys have model prefix and timestamp stripped, then looked up in
    CONDITION_DISPLAY_NAMES (falling back to the shortened name).
    """
    fixed_keys = fixed_keys or []
    ft_keys = [k for k in raw_keys if k not in fixed_keys]
    short_ft = shorten_condition_names(ft_keys)
    name_map = {k: k for k in fixed_keys}
    name_map.update(dict(zip(ft_keys, short_ft)))
    return {k: CONDITION_DISPLAY_NAMES.get(v, v) for k, v in name_map.items()}
