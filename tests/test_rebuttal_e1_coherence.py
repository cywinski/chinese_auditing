# ABOUTME: Unit tests for the rebuttal coherence judge's strict JSON validation
# ABOUTME: and for the E1 aggregation (per-config stats, per-question means, paired bootstrap).

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "experiments"))

from rebuttal_coherence_judge import JudgeValidationError, build_judge_prompt, parse_judge_output
from rebuttal_e1_coherence import aggregate_config, paired_bootstrap_delta, question_means


def judge_json(relevance=5, coherence=4, specificity=3, degenerate=False, **overrides) -> str:
    """A well-formed judge output, optionally with fields overridden."""
    data = {
        "relevance": {"score": relevance, "justification": "on topic"},
        "coherence": {"score": coherence, "justification": "fluent"},
        "specificity": {"score": specificity, "justification": "some detail"},
        "degenerate": degenerate,
    }
    data.update(overrides)
    return json.dumps(data)


def record(prompt_id, relevance, coherence=5, specificity=5, degenerate=False, ok=True, chars=10) -> dict:
    """A per-response record as produced by the experiment script."""
    base = {"prompt_id": prompt_id, "ok": ok, "response_chars": chars}
    if ok:
        base.update(relevance=relevance, coherence=coherence, specificity=specificity, degenerate=degenerate)
    return base


def test_parse_valid_output():
    parsed = parse_judge_output(judge_json())
    assert (parsed["relevance"], parsed["coherence"], parsed["specificity"]) == (5, 4, 3)
    assert parsed["degenerate"] is False
    assert parsed["justifications"]["coherence"] == "fluent"


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "```json\n" + judge_json() + "\n```",
        "[1, 2, 3]",
        judge_json(relevance=0),
        judge_json(relevance=6),
        judge_json(coherence=4.0),
        judge_json(specificity="3"),
        judge_json(relevance=True),
        judge_json(degenerate="false"),
        judge_json(degenerate=0),
        judge_json(extra=1),
        judge_json(relevance={"score": 3}),
        judge_json(relevance={"score": 3, "justification": " "}),
        json.dumps({"relevance": {"score": 3, "justification": "x"}}),
    ],
)
def test_parse_rejects_invalid_output(raw):
    with pytest.raises(JudgeValidationError):
        parse_judge_output(raw)


def test_parse_rejects_non_string():
    with pytest.raises(JudgeValidationError):
        parse_judge_output(None)


def test_prompt_contains_question_and_response():
    prompt = build_judge_prompt("QUESTION-XYZ", "RESPONSE {with} braces")
    assert "QUESTION-XYZ" in prompt and "RESPONSE {with} braces" in prompt


def test_question_means_skips_failures():
    records = [record("a", 5), record("a", 3), record("b", 1), record("b", None, ok=False)]
    assert question_means(records, "relevance") == {"a": 4.0, "b": 1.0}


def test_aggregate_config_known_values():
    records = [
        record("a", 5, degenerate=True, chars=10),
        record("a", 3, chars=20),
        record("b", 1, chars=30),
        record("b", None, ok=False, chars=40),
    ]
    agg = aggregate_config(records)
    assert (agg["n_total"], agg["n_judged"], agg["n_failures"], agg["n_questions"]) == (4, 3, 1, 2)
    assert agg["mean_chars"] == 25.0
    assert agg["relevance"]["mean"] == pytest.approx(3.0)
    # question means are 4 and 1: std(ddof=1) = 2.1213, / sqrt(2) = 1.5
    assert agg["relevance"]["sem"] == pytest.approx(1.5)
    assert agg["degenerate_pct"]["mean"] == pytest.approx(100 / 3)
    assert agg["coherence"]["sem"] == 0.0


def test_paired_bootstrap_identical_is_zero():
    x = {str(i): float(i % 5) for i in range(30)}
    out = paired_bootstrap_delta(x, x, n_resamples=500, ci=0.95, seed=0)
    assert (out["delta"], out["ci_low"], out["ci_high"], out["n_questions"]) == (0.0, 0.0, 0.0, 30)


def test_paired_bootstrap_constant_shift_and_pairing():
    control = {str(i): float(i) for i in range(30)}
    treatment = {q: v + 1.5 for q, v in control.items()}
    treatment["unpaired"] = 100.0
    out = paired_bootstrap_delta(treatment, control, n_resamples=500, ci=0.95, seed=0)
    assert out["n_questions"] == 30
    assert out["delta"] == pytest.approx(1.5)
    assert out["ci_low"] == pytest.approx(1.5) and out["ci_high"] == pytest.approx(1.5)


def test_paired_bootstrap_ci_brackets_delta_and_is_seeded():
    control = {str(i): float(i % 3) for i in range(40)}
    treatment = {str(i): float((i * 7) % 5) for i in range(40)}
    a = paired_bootstrap_delta(treatment, control, n_resamples=2000, ci=0.95, seed=1)
    b = paired_bootstrap_delta(treatment, control, n_resamples=2000, ci=0.95, seed=1)
    assert a == b
    assert a["ci_low"] < a["delta"] < a["ci_high"]
