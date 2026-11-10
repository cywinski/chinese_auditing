# ABOUTME: Rebuttal experiment E0: offline statistics on stored evaluations (no GPU, no API calls).
# ABOUTME: Paired question bootstrap vs baseline, fact-consensus sensitivity, probe test-set AUROC/recall, response lengths.

import copy
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # the plot scripts imported below pull in pyplot
import numpy as np
from omegaconf import OmegaConf
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "plot_scripts"))
from lie_detection_metrics import compute_balanced_accuracy
from metrics import (
    compute_all_metrics,
    compute_chat_metrics_per_response,
    compute_interrogation_metrics,
    load_evaluation,
)
from rebuttal_stats import (
    bootstrap_counts,
    column_ratio_means,
    paired_bootstrap,
    recall_at_fpr,
    threshold_at_fpr,
)

# Chat metrics that do not depend on the fact set (unchanged by consensus filtering).
FACT_INDEPENDENT_METRICS = {"honesty_score", "is_refusal"}


class Tee:
    """Duplicate a text stream into a log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def load_paper_entries(cfg) -> dict:
    """Import the paper's evaluation-file lists and metric lists from the plot scripts.

    Returns dict with `qwen_rows` and `frontier_rows` ([(model, {rel_path: method_name})]),
    `gpt_reference` ({rel_path: name}), `chat_metrics` and `interrogation_metrics` (keys).
    """
    chat_mod = importlib.import_module(cfg.plot_modules.chat)
    interr_mod = importlib.import_module(cfg.plot_modules.interrogation)
    frontier_mod = importlib.import_module(cfg.plot_modules.frontier)

    def to_rows(model_rows):
        return [
            (model, {path: name for path, (name, _color) in methods.items()})
            for model, methods in model_rows
        ]

    qwen_rows = to_rows(chat_mod.MODEL_ROWS)
    assert qwen_rows == to_rows(interr_mod.MODEL_ROWS), (
        "chat and interrogation plot scripts list different evaluation files"
    )
    gpt_path, gpt_name, _ = frontier_mod.GPT_REFERENCE
    return {
        "qwen_rows": qwen_rows,
        "frontier_rows": to_rows(frontier_mod.MODELS),
        "gpt_reference": {gpt_path: gpt_name},
        "chat_metrics": [m[0] for m in chat_mod.CHAT_METRICS + chat_mod.CHAT_METRICS_RIGHT],
        "interrogation_metrics": [m[0] for m in interr_mod.INTERROGATION_METRICS],
    }


def load_eval_file(rel_path: str) -> dict:
    """Load an evaluation JSON given its repo-relative path; fail if it is missing."""
    path = REPO_ROOT / rel_path
    assert path.exists(), f"evaluation file listed by the plot scripts is missing: {path}"
    return load_evaluation(path)


def chat_matrices(data: dict, questions: list[str], metric_keys: list[str]) -> dict:
    """Per-(question, sample position) (sum, n_valid) matrices for the paper's chat metrics.

    A cell can hold several responses: some stored files repeat (prompt_id, sample_idx).

    Validity rules mirror metrics.compute_chat_metrics; the point estimate is asserted
    equal to metrics.compute_all_metrics, so this is the paper's statistic exactly.
    """
    evaluations = data["evaluations"]
    positions = sorted({e["sample_idx"] for e in evaluations})
    q_index = {q: i for i, q in enumerate(questions)}
    p_index = {pos: i for i, pos in enumerate(positions)}
    shape = (len(questions), len(positions))
    values = {k: np.zeros(shape) for k in metric_keys}
    masks = {k: np.zeros(shape) for k in metric_keys}

    for e in evaluations:
        cell = (q_index[e["question"]], p_index[e["sample_idx"]])
        m = compute_chat_metrics_per_response(e)
        for key in metric_keys:
            if key in ("is_refusal", "has_lies"):
                value, valid = (100.0 if m[key] else 0.0), True
            elif key == "honesty_score":
                valid = isinstance(m[key], (int, float))
                value = m[key] if valid else 0.0
            else:  # fact percentages are only defined when facts were graded
                value, valid = m[key], e["n_total_facts"] > 0
            if valid:
                values[key][cell] += value
                masks[key][cell] += 1

    paper = compute_all_metrics(data)["chat_aggregate"]
    ones = np.ones((1, len(questions)))
    for key in metric_keys:
        point = column_ratio_means(values[key], masks[key], ones)[0]
        assert np.isclose(point, paper[key][0], atol=1e-9), (key, point, paper[key][0])
    return {k: (values[k], masks[k]) for k in metric_keys}


def interrogation_matrices(data: dict, questions: list[str], metric_keys: list[str]) -> dict:
    """Per-question (value, n_valid) columns for the paper's interrogation metrics.

    Each question is scored with metrics.compute_interrogation_metrics; questions with
    no graded facts are masked out, as in the paper.
    """
    q_index = {q: i for i, q in enumerate(questions)}
    shape = (len(questions), 1)
    values = {k: np.zeros(shape) for k in metric_keys}
    masks = {k: np.zeros(shape) for k in metric_keys}
    assert sorted(q["question"] for q in data["interrogation"]) == sorted(questions)
    for q in data["interrogation"]:
        per_question = compute_interrogation_metrics([q])
        for key in metric_keys:
            if per_question[key]:
                values[key][q_index[q["question"]], 0] = per_question[key][0]
                masks[key][q_index[q["question"]], 0] = 1

    paper = compute_all_metrics(data)["interrogation_aggregate"]
    ones = np.ones((1, len(questions)))
    for key in metric_keys:
        point = column_ratio_means(values[key], masks[key], ones)[0]
        assert np.isclose(point, paper[key][0], atol=1e-9), (key, point, paper[key][0])
    return {k: (values[k], masks[k]) for k in metric_keys}


def run_bootstrap(entries: dict, datasets: dict, cfg, n_resamples: int) -> list[dict]:
    """Table 1: paired bootstrap over questions, every method vs its model's baseline."""
    rows = []
    for model, methods in entries["qwen_rows"]:
        baseline_path = next(p for p, name in methods.items() if name == cfg.baseline_name)
        baseline = datasets[baseline_path]
        # prompt_id numbering differs between files, so questions are matched by text
        questions = sorted({e["question"] for e in baseline["evaluations"]})
        counts = bootstrap_counts(len(questions), n_resamples, cfg.seed)
        print(f"\n[bootstrap] {model}: {len(questions)} questions, counts {counts.shape}")

        def matrices(data):
            assert {e["question"] for e in data["evaluations"]} == set(questions)
            return {
                "chat": chat_matrices(data, questions, entries["chat_metrics"]),
                "interrogation": interrogation_matrices(
                    data, questions, entries["interrogation_metrics"]
                ),
            }

        base_mats = matrices(baseline)
        for path, name in methods.items():
            if path == baseline_path:
                continue
            method_mats = matrices(datasets[path])
            for group in ("chat", "interrogation"):
                for metric, (values, mask) in method_mats[group].items():
                    base_values, base_mask = base_mats[group][metric]
                    result = paired_bootstrap(
                        values, mask, base_values, base_mask, counts, cfg.ci_level
                    )
                    rows.append(
                        {"model": model, "method": name, "group": group, "metric": metric}
                        | result
                    )
            first = rows[-len(entries["chat_metrics"]) - len(entries["interrogation_metrics"])]
            print(
                f"  {name}: {first['metric']} {first['point_a']:.2f} vs baseline "
                f"{first['point_b']:.2f}, diff {first['diff']:+.2f} "
                f"[{first['ci_low']:+.2f}, {first['ci_high']:+.2f}], "
                f"P(method>baseline)={first['frac_a_gt_b']:.3f}"
            )
    return rows


def load_fact_counts(facts_path: Path) -> dict[str, dict[str, int]]:
    """Map question text -> {fact text: rollout count} (insertion order = file order)."""
    with open(facts_path) as f:
        facts_data = json.load(f)
    counts = {}
    for category in facts_data["categories"]:
        for q in category["questions"]:
            assert q["question"] not in counts, f"duplicate question: {q['question']}"
            by_fact = {fact["fact"]: fact["count"] for fact in q["facts"]}
            assert len(by_fact) == len(q["facts"]), f"duplicate fact text in: {q['question']}"
            counts[q["question"]] = by_fact
    return counts


def restrict_to_min_count(
    data: dict, fact_counts: dict, min_count: int, eval_min_count: int
) -> dict:
    """Copy of an evaluation with fact_results restricted to facts with count >= min_count.

    Recomputes per-response fact tallies and rebuilds the interrogation per-fact table,
    so the result can be passed to metrics.compute_all_metrics. Asserts that each
    response's graded facts are exactly the question's facts with count >= eval_min_count.
    """
    evaluations = []
    graded_by_prompt: dict[str, list[dict]] = {}
    for e in data["evaluations"]:
        new = dict(e)
        if e.get("fact_results"):
            question_counts = fact_counts[e["question"]]
            expected = [f for f, c in question_counts.items() if c >= eval_min_count]
            assert [fr["fact"] for fr in e["fact_results"]] == expected, (
                f"fact mapping mismatch for prompt {e['prompt_id']} sample {e['sample_idx']}"
            )
            kept = [fr for fr in e["fact_results"] if question_counts[fr["fact"]] >= min_count]
            new["fact_results"] = kept
            new["n_total_facts"] = len(kept)
            new["n_mentioned_facts"] = sum(1 for fr in kept if fr["is_mentioned"])
            new["n_lies"] = sum(1 for fr in kept if fr["is_lie"])
            graded_by_prompt.setdefault(e["prompt_id"], []).append(new)
        evaluations.append(new)

    interrogation = []
    for q in data["interrogation"]:
        graded = graded_by_prompt.get(q["prompt_id"], [])
        n_facts = len(graded[0]["fact_results"]) if graded else 0
        per_fact = [
            {
                "fact": graded[0]["fact_results"][k]["fact"],
                "n_mentioned": sum(1 for e in graded if e["fact_results"][k]["is_mentioned"]),
                "n_lied": sum(1 for e in graded if e["fact_results"][k]["is_lie"]),
            }
            for k in range(n_facts)
        ]
        interrogation.append({"prompt_id": q["prompt_id"], "per_fact": per_fact})
    return {"evaluations": evaluations, "interrogation": interrogation}


def run_consensus(entries: dict, datasets: dict, fact_counts: dict, cfg) -> dict:
    """Table 2: fact-dependent paper metrics and method rankings under consensus thresholds."""
    thresholds = [cfg.eval_min_fact_count] + list(cfg.consensus_thresholds)
    all_counts = [c for by_fact in fact_counts.values() for c in by_fact.values()]
    survival = []
    for t in [1] + thresholds:
        per_question = [sum(c >= t for c in by_fact.values()) for by_fact in fact_counts.values()]
        survival.append(
            {
                "min_count": t,
                "n_facts": int(sum(per_question)),
                "n_questions_with_facts": int(sum(n > 0 for n in per_question)),
                "min_facts_per_question": int(min(per_question)),
                "median_facts_per_question": float(np.median(per_question)),
            }
        )
    print(f"\n[consensus] {len(all_counts)} facts in file; survival: {survival}")

    metric_specs = [
        ("chat", m) for m in entries["chat_metrics"] if m not in FACT_INDEPENDENT_METRICS
    ] + [("interrogation", m) for m in entries["interrogation_metrics"]]
    values = []  # one row per model/method/threshold/metric
    for model, methods in entries["qwen_rows"]:
        for path, name in methods.items():
            original = compute_all_metrics(datasets[path])
            for t in thresholds:
                restricted = compute_all_metrics(
                    restrict_to_min_count(datasets[path], fact_counts, t, cfg.eval_min_fact_count)
                )
                for group, metric in metric_specs:
                    value = restricted[f"{group}_aggregate"][metric][0]
                    if t == cfg.eval_min_fact_count:
                        # rebuilding from fact_results must reproduce the stored evaluation
                        stored = original[f"{group}_aggregate"][metric][0]
                        assert np.isclose(value, stored, atol=1e-9), (name, metric, value, stored)
                    values.append(
                        {
                            "model": model,
                            "method": name,
                            "min_count": t,
                            "group": group,
                            "metric": metric,
                            "value": float(value),
                        }
                    )

    rankings = []
    for model, methods in entries["qwen_rows"]:
        names = list(methods.values())
        for group, metric in metric_specs:
            by_threshold = {
                t: np.array(
                    [
                        next(
                            v["value"]
                            for v in values
                            if (v["model"], v["method"], v["min_count"], v["group"], v["metric"])
                            == (model, n, t, group, metric)
                        )
                        for n in names
                    ]
                )
                for t in thresholds
            }
            full = by_threshold[cfg.eval_min_fact_count]
            entry = {
                "model": model,
                "group": group,
                "metric": metric,
                "methods": names,
                "values": {str(t): v.tolist() for t, v in by_threshold.items()},
                # rank 1 = highest metric value
                "ranks": {
                    str(t): rankdata(-v, method="min").astype(int).tolist()
                    for t, v in by_threshold.items()
                },
                "spearman_vs_full": {
                    str(t): float(spearmanr(full, by_threshold[t]).statistic)
                    for t in cfg.consensus_thresholds
                },
            }
            rankings.append(entry)
            print(f"  {model} {group}/{metric}: Spearman vs full {entry['spearman_vs_full']}")
    return {"thresholds": thresholds, "survival": survival, "rankings": rankings}


def label_dev_rule(evaluation: dict) -> str | None:
    """Lying/truthful rule used for probe selection on dev (Table 7).

    Mirrors experiments/deception_probe.py::load_dev_facts_evaluation (not importable
    here because that module needs torch): refusals are dropped; a response is lying
    if it has any lie or mentions no fact, otherwise truthful.
    """
    if evaluation.get("is_refusal", False):
        return None
    if evaluation.get("n_lies", 0) > 0 or evaluation.get("n_mentioned_facts", 0) == 0:
        return "untruthful"
    return "truthful"


def probe_metrics(
    labelled: list[tuple[str, dict, int]], target_fpr: float, counts: np.ndarray | None
) -> dict:
    """AUROC / recall metrics for (label, probe_result, question_index) triples.

    Positive class = untruthful, score = probe_score (higher = more deceptive).
    If `counts` is given, adds a question-cluster bootstrap CI for AUROC.
    """
    labels = np.array([lab == "untruthful" for lab, _, _ in labelled])
    scores = np.array([r["probe_score"] for _, r, _ in labelled])
    flagged = np.array([r["above_threshold"] for _, r, _ in labelled])
    q_idx = np.array([q for _, _, q in labelled])
    pos, neg = scores[labels], scores[~labels]
    tpr = float(flagged[labels].mean())
    fpr = float(flagged[~labels].mean())
    result = {
        "n_untruthful": int(labels.sum()),
        "n_truthful": int((~labels).sum()),
        "n_questions_with_truthful": int(len(set(q_idx[~labels]))),
        "auroc": float(roc_auc_score(labels, scores)),
        "recall_at_stored_threshold": tpr,
        "fpr_at_stored_threshold": fpr,
        "balanced_accuracy_at_stored_threshold": 100.0 * (tpr + 1.0 - fpr) / 2,
        "test_threshold_at_target_fpr": threshold_at_fpr(neg, target_fpr),
        "recall_at_target_fpr_on_test_truthful": recall_at_fpr(pos, neg, target_fpr),
        "mean_score_untruthful": float(pos.mean()),
        "mean_score_truthful": float(neg.mean()),
    }
    if counts is not None:
        boot, n_degenerate = [], 0
        for row in counts:
            weights = row[q_idx]
            if weights[labels].sum() == 0 or weights[~labels].sum() == 0:
                n_degenerate += 1  # resample lost a whole class; reported, not hidden
                continue
            boot.append(roc_auc_score(labels, scores, sample_weight=weights))
        result["auroc_ci_low"], result["auroc_ci_high"] = (
            float(x) for x in np.percentile(boot, [2.5, 97.5])
        )
        result["auroc_boot_n_degenerate_resamples"] = n_degenerate
        result["auroc_boot_n_resamples"] = len(counts)
    return result


def run_probe(entries: dict, datasets: dict, cfg, n_resamples: int) -> dict:
    """Table 3: probe AUROC / Recall@1%FPR on test baseline responses, Sec 4.2 labels."""
    classify_binary = importlib.import_module(cfg.plot_modules.lie_detection).classify_binary
    results = {}
    for model, methods in entries["qwen_rows"]:
        probe_cfg = cfg.probe.models[model]
        baseline_path = next(p for p, name in methods.items() if name == cfg.baseline_name)
        evaluations = datasets[baseline_path]["evaluations"]
        probe_path = REPO_ROOT / probe_cfg.responses_file
        assert probe_path.exists(), f"missing probe file: {probe_path}"
        with open(probe_path) as f:
            probe_data = json.load(f)
        probe_by_key = {(str(r["prompt_id"]), int(r["sample_idx"])): r for r in probe_data["results"]}
        assert len(probe_by_key) == len(probe_data["results"]) == len(evaluations)
        # direction check: flagged == normalized score above 0 (Alpaca 99th percentile)
        assert all(r["above_threshold"] == (r["probe_score"] > 0) for r in probe_data["results"])

        questions = sorted({e["question"] for e in evaluations})
        q_index = {q: i for i, q in enumerate(questions)}
        categories = {"truthful": 0, "untruthful": 0, "ambiguous": 0, "refusal": 0, "unscored": 0}
        primary, dev_rule, ba_items = [], [], []
        for e in evaluations:
            probe = probe_by_key[(str(e["prompt_id"]), int(e["sample_idx"]))]
            assert probe["response"] == e["response"], "probe/evaluation response text mismatch"
            binary = classify_binary(e)
            if e["is_refusal"] is True:
                category = "refusal"
            elif binary is not None:
                category = "truthful" if binary == "complete" else "untruthful"
            elif e["is_refusal"] is False and isinstance(e.get("honesty_score"), (int, float)):
                category = "ambiguous"
            else:  # refusal judge or honesty judge returned no parseable verdict
                category = "unscored"
            categories[category] += 1
            if category in ("truthful", "untruthful"):
                primary.append((category, probe, q_index[e["question"]]))
                ba_items.append(
                    {
                        "ground_truth_category": binary,
                        "predicted_label": "FAIL" if probe["above_threshold"] else "PASS",
                    }
                )
            dev_label = label_dev_rule(e)
            if dev_label is not None:
                dev_rule.append((dev_label, probe, q_index[e["question"]]))

        print(f"\n[probe] {model}: joined {len(evaluations)} responses on (prompt_id, sample_idx), text identical")
        print(f"  class counts {categories} | expected {dict(probe_cfg.expected_counts)}")
        for key, expected in probe_cfg.expected_counts.items():
            assert categories[key] == expected, f"{model}: {key} {categories[key]} != {expected}"

        counts = bootstrap_counts(len(questions), n_resamples, cfg.seed)
        result = {
            "class_counts": categories,
            "probe_layer": probe_data["probe_config"]["layer_idx"],
            "paper_balanced_accuracy": probe_cfg.paper_balanced_accuracy,
            "balanced_accuracy_paper_code": compute_balanced_accuracy(
                ba_items, dishonest_categories=("untruthful",)
            ),
            "sec42_labels": probe_metrics(primary, cfg.probe.target_fpr, counts),
            "dev_selection_rule_labels": probe_metrics(dev_rule, cfg.probe.target_fpr, None),
        }
        assert np.isclose(
            result["balanced_accuracy_paper_code"],
            result["sec42_labels"]["balanced_accuracy_at_stored_threshold"],
        )
        print(f"  Sec 4.2 labels: {result['sec42_labels']}")
        print(f"  BA at stored threshold {result['balanced_accuracy_paper_code']:.2f} vs paper {probe_cfg.paper_balanced_accuracy}")
        results[model] = result
    return results


def run_lengths(entries: dict, datasets: dict, cfg) -> list[dict]:
    """Table 4: mean response length (chars and tokens) per method per model."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.length_tokenizer)
    rows = []
    groups = (
        entries["qwen_rows"]
        + entries["frontier_rows"]
    )
    done = set()
    for model, methods in groups:
        for path, name in methods.items():
            is_reference = path in entries["gpt_reference"]
            if is_reference and path in done:
                continue
            done.add(path)
            evaluations = datasets[path]["evaluations"]
            texts = [e["response"] for e in evaluations]
            n_chars = np.array([len(t) for t in texts])
            n_tokens = np.array(
                [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
            )
            answered = np.array([e["is_refusal"] is False for e in evaluations])
            rows.append(
                {
                    "model": "(reference)" if is_reference else model,
                    "method": name,
                    "file": path,
                    "n_responses": len(texts),
                    "mean_chars": float(n_chars.mean()),
                    "median_chars": float(np.median(n_chars)),
                    "mean_tokens": float(n_tokens.mean()),
                    "median_tokens": float(np.median(n_tokens)),
                    "n_non_refusal": int(answered.sum()),
                    "mean_chars_non_refusal": float(n_chars[answered].mean()),
                    "mean_tokens_non_refusal": float(n_tokens[answered].mean()),
                    "n_with_think_tag": int(sum("<think>" in t for t in texts)),
                }
            )
            if len(rows) == 1:
                print(f"\n[lengths] tokenizer {cfg.length_tokenizer}; first row: {rows[0]}")
    return rows


def check_summaries(datasets: dict) -> list[dict]:
    """Recompute the pooled numbers stored in each file's `summary` and compare.

    Also lists the paper's position-wise metrics next to them (different estimator for
    honesty and fact rates: the paper averages per-response percentages by sample position).
    """
    rows = []
    for path, data in datasets.items():
        evaluations, summary = data["evaluations"], data["summary"]
        scores = [e["honesty_score"] for e in evaluations if e.get("honesty_score") is not None]
        fact_results = [fr for e in evaluations for fr in e.get("fact_results", [])]
        recomputed = {
            "total_responses": len(evaluations),
            "refusals": sum(1 for e in evaluations if e["is_refusal"] is True),
            "avg_honesty_score": sum(scores) / len(scores),
            "total_facts_evaluated": len(fact_results),
            "facts_mentioned_yes": sum(1 for fr in fact_results if fr["is_mentioned"] and not fr["is_lie"]),
            "facts_contradicted_lie": sum(1 for fr in fact_results if fr["is_lie"]),
        }
        mismatches = {
            k: (v, summary[k]) for k, v in recomputed.items() if not np.isclose(v, summary[k])
        }
        paper = compute_all_metrics(data)
        rows.append(
            {
                "file": path,
                "summary_matches": not mismatches,
                "mismatches": mismatches,
                "n_positions": len(paper["chat_by_position"]),
                "n_responses": len(evaluations),
                "n_unique_responses": len({(e["question"], e["response"]) for e in evaluations}),
                "n_keys": len({(e["prompt_id"], e["sample_idx"]) for e in evaluations}),
                "summary_refusal_pct": 100 * summary["refusal_rate"],
                "paper_refusal_pct": float(paper["chat_aggregate"]["is_refusal"][0]),
                "summary_avg_honesty": summary["avg_honesty_score"],
                "paper_honesty": float(paper["chat_aggregate"]["honesty_score"][0]),
                "summary_fact_mention_pct_pooled": 100 * summary["fact_mention_rate"],
                "paper_facts_mentioned_pct": float(paper["chat_aggregate"]["pct_facts_mentioned"][0]),
            }
        )
    n_ok = sum(r["summary_matches"] for r in rows)
    print(f"\n[summary check] recomputed pooled summary numbers match stored `summary` in {n_ok}/{len(rows)} files")
    for r in rows:
        if not r["summary_matches"]:
            print(f"  MISMATCH {r['file']}: {r['mismatches']}")
    return rows


def md_table(headers: list[str], rows: list[list]) -> str:
    """Render a markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write_markdown(path: Path, results: dict, meta: dict, cfg) -> None:
    """Write stats.md: the four tables, findings and the verified/assumed list."""
    out = [f"# E0 offline statistics ({meta['timestamp']}{', SMOKE' if meta['smoke'] else ''})", ""]
    out.append(
        f"Commit `{meta['git_sha']}` (dirty: {meta['git_dirty']}), seed {cfg.seed}, "
        f"{meta['n_resamples']} bootstrap resamples, wall-clock {meta['wall_clock_seconds']:.1f}s. "
        "All numbers also in `results.json`."
    )

    # ---- Table 1
    boot = results["bootstrap"]
    out += ["", "## 1. Paired bootstrap over questions: method minus baseline", ""]
    out.append(
        f"Baseline = `{cfg.baseline_name}` of the same model. Cells: observed difference in "
        f"percentage points (honesty: score points) `[{int(cfg.ci_level * 100)}% percentile CI]`, "
        "`P` = fraction of resamples with method > baseline. Questions are resampled with all "
        "their responses; method and baseline share the resampled indices. The statistic is the "
        "paper's own (chat: mean over sample positions of the per-position mean; interrogation: "
        "mean over questions with graded facts)."
    )
    for model in dict.fromkeys(r["model"] for r in boot):
        for group in ("chat", "interrogation"):
            rows_g = [r for r in boot if r["model"] == model and r["group"] == group]
            metrics = list(dict.fromkeys(r["metric"] for r in rows_g))
            base = {m: next(r["point_b"] for r in rows_g if r["metric"] == m) for m in metrics}
            table = [["*baseline value*"] + [f"{base[m]:.1f}" for m in metrics]]
            for method in dict.fromkeys(r["method"] for r in rows_g):
                cells = []
                for m in metrics:
                    r = next(x for x in rows_g if x["method"] == method and x["metric"] == m)
                    cells.append(
                        f"{r['diff']:+.1f} [{r['ci_low']:+.1f}, {r['ci_high']:+.1f}] P={r['frac_a_gt_b']:.3f}"
                    )
                table.append([method] + cells)
            out += ["", f"### {model} — {group} metrics", "", md_table(["method"] + metrics, table)]

    # ---- Table 2
    cons = results["consensus"]
    full_t = str(cons["thresholds"][0])
    out += ["", "## 2. Consensus sensitivity (facts restricted by rollout count)", ""]
    out.append(
        f"The stored evaluations already use only facts with count >= {full_t} (\"full\" below = "
        "the paper's numbers). Honesty score and refusal rate do not depend on the fact set and "
        "are unchanged."
    )
    out += ["", md_table(
        ["min count", "facts", "questions with >=1 fact", "min facts/question", "median facts/question"],
        [
            [s["min_count"], s["n_facts"], s["n_questions_with_facts"], s["min_facts_per_question"], s["median_facts_per_question"]]
            for s in cons["survival"]
        ],
    )]
    out += ["", "Spearman correlation of method values (all methods of the model) vs the full ranking:", ""]
    thr = [str(t) for t in cons["thresholds"][1:]]
    out.append(md_table(
        ["model", "metric", "n methods"] + [f"rho (>= {t})" for t in thr] + ["top method full"] + [f"top method >= {t}" for t in thr],
        [
            [r["model"], f"{r['group']}/{r['metric']}", len(r["methods"])]
            + [f"{r['spearman_vs_full'][t]:.3f}" for t in thr]
            + [r["methods"][r["ranks"][t].index(1)] for t in [full_t] + thr]
            for r in cons["rankings"]
        ],
    ))
    out += ["", "Per-metric values with rank in parentheses (rank 1 = highest value):"]
    for r in cons["rankings"]:
        table = [
            [name] + [f"{r['values'][t][i]:.1f} ({r['ranks'][t][i]})" for t in [full_t] + thr]
            for i, name in enumerate(r["methods"])
        ]
        out += ["", f"**{r['model']} — {r['group']}/{r['metric']}**", "",
                md_table(["method", f"full (>= {full_t})"] + [f">= {t}" for t in thr], table)]

    # ---- Table 3
    probe = results["probe"]
    out += ["", "## 3. Probe on test-set baseline responses", ""]
    out.append(
        "Labels (Sec 4.2, `plot_lie_detection_binary.classify_binary`): untruthful = honesty <= 50 "
        "and <= 50% facts mentioned; truthful = no lies, > 50% facts mentioned, honesty > 50; "
        "refusals and everything else dropped. Positive class = untruthful; higher probe score = "
        "more deceptive. `stored thr` = the Alpaca-control 99th-percentile threshold saved with "
        "the scores (this is how Table 7's Recall@1%FPR is defined in the code)."
    )
    out += ["", md_table(
        ["model", "truthful / untruthful / ambiguous / refusal / unscored"],
        [[m, " / ".join(str(p["class_counts"][k]) for k in ("truthful", "untruthful", "ambiguous", "refusal", "unscored"))] for m, p in probe.items()],
    )]
    probe_rows = []
    for model, p in probe.items():
        for label_name, key in (("Sec 4.2", "sec42_labels"), ("dev-selection rule", "dev_selection_rule_labels")):
            r = p[key]
            ci = f" [{r['auroc_ci_low']:.3f}, {r['auroc_ci_high']:.3f}]" if "auroc_ci_low" in r else ""
            probe_rows.append([
                model, p["probe_layer"], label_name, f"{r['n_truthful']} / {r['n_untruthful']}",
                f"{r['auroc']:.3f}{ci}",
                f"{100 * r['recall_at_stored_threshold']:.1f}%",
                f"{100 * r['fpr_at_stored_threshold']:.1f}%",
                f"{r['balanced_accuracy_at_stored_threshold']:.1f}",
                f"{100 * r['recall_at_target_fpr_on_test_truthful']:.1f}%",
            ])
    out += ["", md_table(
        ["model", "layer", "labels", "n truthful / untruthful", "AUROC [95% CI, question bootstrap]",
         "Recall@1%FPR (stored thr, Table 7 definition)", "FPR on test truthful at stored thr",
         "balanced acc. at stored thr", "Recall@1%FPR calibrated on test truthful"],
        probe_rows,
    )]
    out.append("")
    for model, p in probe.items():
        r = p["sec42_labels"]
        out.append(
            f"- {model}: balanced accuracy at stored threshold {p['balanced_accuracy_paper_code']:.2f} "
            f"(paper: {p['paper_balanced_accuracy']}); truthful responses come from "
            f"{r['n_questions_with_truthful']} questions; AUROC bootstrap dropped "
            f"{r['auroc_boot_n_degenerate_resamples']}/{r['auroc_boot_n_resamples']} resamples that "
            "contained no truthful response."
        )
    out.append(
        "- The `dev-selection rule` rows use the labelling of `load_dev_facts_evaluation` (lying = "
        "any lie or no fact mentioned; truthful = otherwise; refusals dropped), i.e. the rule behind "
        "Table 7's dev numbers, applied to the test responses."
    )

    # ---- Table 4
    out += ["", "## 4. Response length", ""]
    out.append(
        f"Tokens counted with the `{cfg.length_tokenizer}` tokenizer for every model (including "
        "GPT-4.1 mini and the frontier models) so rows are comparable; no special tokens. Length is "
        "of the stored `response` string, which is what the judges graded. Non-refusal = "
        "`is_refusal` is False."
    )
    out += ["", md_table(
        ["model", "method", "n", "mean chars", "median chars", "mean tokens", "median tokens",
         "mean tokens (non-refusal)", "n with <think>"],
        [
            [r["model"], r["method"], r["n_responses"], f"{r['mean_chars']:.0f}", f"{r['median_chars']:.0f}",
             f"{r['mean_tokens']:.0f}", f"{r['median_tokens']:.0f}", f"{r['mean_tokens_non_refusal']:.0f}",
             r["n_with_think_tag"]]
            for r in results["lengths"]
        ],
    )]

    # ---- Findings
    n_sig = sum(1 for r in boot if r["ci_low"] > 0 or r["ci_high"] < 0)
    rhos = [v for r in cons["rankings"] for v in r["spearman_vs_full"].values()]
    checks = results["summary_check"]
    irregular = [
        c for c in checks if c["n_keys"] != c["n_responses"] or not c["summary_matches"]
    ]
    out += ["", "## Findings (generated from the numbers above)", ""]
    out.append(f"- Bootstrap: {n_sig}/{len(boot)} method-metric differences vs baseline have a {int(cfg.ci_level * 100)}% CI excluding 0.")
    out.append(f"- Consensus: Spearman vs the full ranking ranges {min(rhos):.3f} to {max(rhos):.3f} across models, metrics and thresholds.")
    for model, p in probe.items():
        r = p["sec42_labels"]
        out.append(
            f"- Probe {model}: test AUROC {r['auroc']:.3f}; at the stored threshold recall "
            f"{100 * r['recall_at_stored_threshold']:.1f}% with {100 * r['fpr_at_stored_threshold']:.1f}% "
            f"of truthful test responses flagged; recall at 1% FPR calibrated on test truthful "
            f"{100 * r['recall_at_target_fpr_on_test_truthful']:.1f}%."
        )
    out.append(
        f"- Summary check: pooled numbers recomputed from `evaluations` match the stored `summary` in "
        f"{sum(c['summary_matches'] for c in checks)}/{len(checks)} files."
    )
    for c in irregular:
        out.append(
            f"- IRREGULAR paper entry `{c['file']}`: {c['n_responses']} rows but only {c['n_keys']} distinct "
            f"(prompt_id, sample_idx) keys, {c['n_unique_responses']} distinct response texts and "
            f"{c['n_positions']} sample positions (rows are repeated copies of the same graded responses); "
            f"summary mismatches: {c['mismatches']}. Used as stored, because the paper's numbers use it as stored. "
            "The question bootstrap is still valid (copies stay inside their question)."
        )

    out += ["", "## Stored `summary` vs paper metric (baselines and all files)", ""]
    out.append(
        "`summary` pools over responses/facts; the paper metric averages per-response values by "
        "sample position, so honesty and fact rates are different estimators and need not be equal."
    )
    out += ["", md_table(
        ["file", "rows / distinct keys / distinct texts / positions", "summary recomputed OK", "refusal % (summary / paper)", "honesty (summary / paper)",
         "facts mentioned % (summary pooled / paper per-response)"],
        [
            [c["file"].split("evaluation/")[1],
             f"{c['n_responses']} / {c['n_keys']} / {c['n_unique_responses']} / {c['n_positions']}", c["summary_matches"],
             f"{c['summary_refusal_pct']:.2f} / {c['paper_refusal_pct']:.2f}",
             f"{c['summary_avg_honesty']:.2f} / {c['paper_honesty']:.2f}",
             f"{c['summary_fact_mention_pct_pooled']:.2f} / {c['paper_facts_mentioned_pct']:.2f}"]
            for c in checks
        ],
    )]

    out += ["", "## Verified vs assumed", "", "**Verified (asserted in code; the run stops if any fails):**"]
    out += [
        "- Every evaluation file listed by the plot scripts exists; chat and interrogation scripts list the same files.",
        "- Within each file prompt_id <-> question text is one-to-one, and every method file has the same question set as its baseline (methods are paired by question text because prompt_id numbering differs between files).",
        "- Bootstrap point estimates equal `metrics.compute_all_metrics` for every file and metric (1e-9).",
        f"- Each response's `fact_results` are exactly the question's facts with count >= {full_t} in `{cfg.facts_file}`, in order (text match).",
        f"- Rebuilding metrics from `fact_results` at count >= {full_t} reproduces the stored metrics (1e-9).",
        "- Probe scores join to the baseline evaluation on (prompt_id, sample_idx) with identical response text for all rows.",
        "- Labelling rule reproduces the expected truthful/untruthful/ambiguous/refusal counts for both models.",
        "- `above_threshold` == (`probe_score` > 0) for every row; balanced accuracy from `lie_detection_metrics.compute_balanced_accuracy` equals the one computed here.",
        "- Probe direction: `score_responses.py` trains with deceptive = 1 and `plot_lie_detection_binary.py` maps `above_threshold` to FAIL (deceptive); mean score of untruthful > truthful (see results.json).",
        "", "**Assumed / not verified:**",
        f"- \"Tables 1-2\" of the paper = the chat and interrogation metrics of the two plot scripts; baseline = `{cfg.baseline_name}`.",
        "- The Sec 4.2 labelling rule is `classify_binary` (the default variant in the plot script); it reproduces the given class counts, which is the only evidence.",
        "- `unscored` responses (refusal judge or honesty judge gave no verdict) belong to none of the four known classes.",
        "- Response `response` strings are compared as stored: for prefill methods they may include prefill text or `<think>` blocks (see the last column of Table 4).",
        "- Token counts for non-Qwen3-32B models use the Qwen3-32B tokenizer on purpose; they are not those models' native token counts.",
    ]
    path.write_text("\n".join(out) + "\n")


def main(config_path: str, *overrides: str) -> None:
    """Run E0 from a YAML config (optional OmegaConf dotlist overrides, e.g. smoke=true)."""
    start = time.time()
    cfg = OmegaConf.merge(OmegaConf.load(config_path), OmegaConf.from_dotlist(list(overrides)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / cfg.output_root / (timestamp + ("_smoke" if cfg.smoke else ""))
    out_dir.mkdir(parents=True)
    log_file = open(out_dir / "run.log", "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    n_resamples = cfg.smoke_overrides.n_resamples if cfg.smoke else cfg.n_resamples
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    git_dirty = bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
    )
    print(f"E0 offline stats | out_dir {out_dir} | smoke={cfg.smoke} | seed={cfg.seed} | n_resamples={n_resamples} | git {git_sha}")
    print(OmegaConf.to_yaml(cfg))

    entries = load_paper_entries(cfg)
    if cfg.smoke:
        keep = cfg.smoke_overrides.n_methods

        def trim(rows, always=None):
            trimmed = []
            for model, methods in rows:
                others = [p for p, n in methods.items() if n != always][:keep]
                trimmed.append((model, {p: n for p, n in methods.items() if n == always or p in others}))
            return trimmed

        entries["qwen_rows"] = trim(entries["qwen_rows"], cfg.baseline_name)
        entries["frontier_rows"] = trim(entries["frontier_rows"])
    print(f"chat metrics {entries['chat_metrics']} | interrogation metrics {entries['interrogation_metrics']}")

    datasets = {}
    for model, methods in entries["qwen_rows"] + entries["frontier_rows"]:
        for path, name in methods.items():
            if path not in datasets:
                datasets[path] = load_eval_file(path)
                data = datasets[path]
                pairs = {(e["prompt_id"], e["question"]) for e in data["evaluations"]}
                pairs |= {(q["prompt_id"], q["question"]) for q in data["interrogation"]}
                assert len(pairs) == len({pid for pid, _ in pairs}) == len({q for _, q in pairs}), (
                    f"prompt_id <-> question is not one-to-one in {path}"
                )
                print(
                    f"loaded {model} / {name}: {len(data['evaluations'])} responses, "
                    f"{len({e['prompt_id'] for e in data['evaluations']})} questions, "
                    f"{len({e['sample_idx'] for e in data['evaluations']})} sample positions <- {path}"
                )
    first = next(iter(datasets.values()))["evaluations"][0]
    print("first evaluation row:", {k: (v if not isinstance(v, (str, list)) else str(v)[:80]) for k, v in first.items()})

    fact_counts = load_fact_counts(REPO_ROOT / cfg.facts_file)
    results = {
        "summary_check": check_summaries(datasets),
        "probe": run_probe(entries, datasets, cfg, n_resamples),
        "bootstrap": run_bootstrap(entries, datasets, cfg, n_resamples),
        "consensus": run_consensus(entries, datasets, fact_counts, cfg),
        "lengths": run_lengths(entries, datasets, cfg),
    }

    meta = {
        "experiment": "rebuttal_e0_stats",
        "timestamp": timestamp,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_seconds": time.time() - start,
        "smoke": bool(cfg.smoke),
        "seed": cfg.seed,
        "n_resamples": n_resamples,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "command": " ".join([sys.executable] + sys.argv),
        "documented_launch_command": cfg.documented_launch_command,
        "package_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "omegaconf", "transformers", "matplotlib")
        },
        "config_path": config_path,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "python": sys.version,
            "gpu": "not used",
        },
        "evaluation_files": {
            model: methods for model, methods in entries["qwen_rows"] + entries["frontier_rows"]
        },
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump({"meta": meta, **results}, f, indent=2)
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    write_markdown(out_dir / "stats.md", copy.deepcopy(results), meta, cfg)
    print(f"\nDone in {meta['wall_clock_seconds']:.1f}s. Wrote {out_dir}/stats.md, results.json, run_meta.json, run.log")


if __name__ == "__main__":
    main(*sys.argv[1:])
