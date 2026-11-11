# ABOUTME: Unit tests for the E4 sycophancy judges' strict output validation and prompt construction,
# ABOUTME: and for the E4 aggregation (label distributions, cluster bootstrap, direction summary, cross-tab).

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "experiments"))

from rebuttal_e4_sycophancy import (
    cluster_bootstrap_pct,
    crosstab,
    direction_item,
    direction_summary,
    label_distribution,
    stance_item,
)
from rebuttal_judge_common import JudgeValidationError
from rebuttal_sycophancy_judge import (
    DIRECTIONS,
    build_direction_prompt,
    build_stance_prompt,
    parse_direction_output,
    parse_stance_output,
)

ALIGNED, CRITICAL, PLAIN = DIRECTIONS[0], DIRECTIONS[1], DIRECTIONS[2]


def stance_json(**overrides) -> str:
    """A well-formed stance judge output, optionally with fields overridden."""
    return json.dumps({"user_stance": "none", "presupposition": "neutral", "justification": "open question"} | overrides)


def direction_json(**overrides) -> str:
    """A well-formed direction judge output, optionally with fields overridden."""
    return json.dumps({"direction": ALIGNED, "justification": "denies the event"} | overrides)


def response(question, n_lies=0, direction=None, status=None, refusal=False, honesty=50) -> dict:
    """A per-response record as built by the experiment script after judging."""
    status = status or ("judged" if direction else "no_lies" if n_lies == 0 else "failed")
    return {"question": question, "n_lies": n_lies, "direction": direction, "direction_status": status,
            "is_refusal": refusal, "honesty_score": honesty}


def test_parse_stance_valid():
    parsed = parse_stance_output(stance_json(user_stance="explicit", presupposition="not_applicable", justification=" x "))
    assert parsed == {"user_stance": "explicit", "presupposition": "not_applicable", "justification": "x"}


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "```json\n" + stance_json() + "\n```",
        "[]",
        stance_json(user_stance="None"),
        stance_json(user_stance="maybe"),
        stance_json(user_stance=None),
        stance_json(presupposition="critical"),
        stance_json(presupposition=1),
        stance_json(justification=""),
        stance_json(extra="x"),
        json.dumps({"user_stance": "none", "presupposition": "neutral"}),
    ],
)
def test_parse_stance_rejects_invalid(raw):
    with pytest.raises(JudgeValidationError):
        parse_stance_output(raw)


def test_parse_direction_valid_all_classes():
    for direction in DIRECTIONS:
        assert parse_direction_output(direction_json(direction=direction))["direction"] == direction


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        direction_json(direction="aligned"),
        direction_json(direction=["mixed"]),
        direction_json(justification="  "),
        direction_json(confidence=1),
        json.dumps({"direction": ALIGNED}),
    ],
)
def test_parse_direction_rejects_invalid(raw):
    with pytest.raises(JudgeValidationError):
        parse_direction_output(raw)


def test_prompts_contain_inputs_and_tolerate_braces():
    assert "Q {x}" in build_stance_prompt("Q {x}")
    prompt = build_direction_prompt("QUESTION", "RESPONSE {y}", ["fact one", "fact two"])
    assert "QUESTION" in prompt and "RESPONSE {y}" in prompt and "1. fact one\n2. fact two" in prompt
    with pytest.raises(ValueError):
        build_direction_prompt("Q", "R", [])


def test_judge_items_key_and_hash_follow_content():
    a, b = stance_item("s", "question A"), stance_item("s", "question B")
    assert a["key"] != b["key"] and a["response_sha1"] != b["response_sha1"]
    assert a["key"] == stance_item("s", "question A")["key"]
    # blind to the source: the set name only goes into the key, never into the prompt
    assert stance_item("other_set", "question A")["prompt"] == a["prompt"]
    base = {"key": "k", "question": "Q", "response": "R", "contradicted_facts": ["f1"]}
    assert direction_item(base)["response_sha1"] != direction_item(base | {"contradicted_facts": ["f2"]})["response_sha1"]


def test_label_distribution():
    dist = label_distribution(["a", "a", "b"], ("a", "b", "c"))
    assert dist["n"] == 3 and dist["counts"] == {"a": 2, "b": 1, "c": 0}
    assert dist["pct"]["a"] == pytest.approx(200 / 3) and dist["pct"]["c"] == 0.0
    assert label_distribution([], ("a",))["pct"] == {"a": None}
    with pytest.raises(ValueError):
        label_distribution(["z"], ("a",))


def test_cluster_bootstrap_pct_pooled_and_degenerate_cases():
    out = cluster_bootstrap_pct({"a": 2, "b": 0, "c": 5}, {"a": 4, "b": 4, "c": 0}, n_resamples=500, ci=0.95, seed=0)
    assert (out["pct"], out["num"], out["den"], out["n_questions"]) == (25.0, 2, 8, 2)
    assert 0.0 <= out["ci_low"] <= out["pct"] <= out["ci_high"] <= 50.0
    same = cluster_bootstrap_pct({"a": 3, "b": 3}, {"a": 3, "b": 3}, n_resamples=200, ci=0.95, seed=0)
    assert (same["pct"], same["ci_low"], same["ci_high"]) == (100.0, 100.0, 100.0)
    empty = cluster_bootstrap_pct({"a": 0}, {"a": 0}, n_resamples=200, ci=0.95, seed=0)
    assert empty["pct"] is None and empty["n_questions"] == 0


def test_cluster_bootstrap_pct_is_seeded():
    num = {str(i): i % 3 for i in range(30)}
    den = {str(i): 3 for i in range(30)}
    a = cluster_bootstrap_pct(num, den, n_resamples=1000, ci=0.95, seed=1)
    assert a == cluster_bootstrap_pct(num, den, n_resamples=1000, ci=0.95, seed=1)
    assert a["ci_low"] < a["pct"] < a["ci_high"]


def test_direction_summary_counts_failures_and_excludes_them():
    responses = [
        response("q1", 2, ALIGNED), response("q1", 1, ALIGNED), response("q1"),
        response("q2", 1, PLAIN), response("q2", 3, None, status="failed"),
        response("q3"),
    ]
    out = direction_summary(responses, n_resamples=200, ci=0.95, seed=0)
    assert (out["n_responses"], out["n_lying"], out["n_judged"], out["n_failures"], out["n_lying_questions"]) == (6, 4, 3, 1, 2)
    assert out["counts"][ALIGNED] == 2 and out["counts"][PLAIN] == 1 and out["counts"][CRITICAL] == 0
    assert out["pct"][ALIGNED] == pytest.approx(200 / 3)
    assert out["aligned_ci"]["pct"] == pytest.approx(200 / 3) and out["aligned_ci"]["n_questions"] == 2


def test_crosstab_known_values():
    responses = [
        response("n1", 1, ALIGNED, honesty=20), response("n1", honesty=80),
        response("n2", refusal=True, honesty=None), response("n2", 1, PLAIN, honesty=40),
        response("c1", 2, None, status="failed", honesty=10), response("c1", 1, ALIGNED, honesty=30),
    ]
    classes = {"n1": "neutral", "n2": "neutral", "c1": "critical"}
    groups = {"neutral": ("neutral",), "critical": ("critical",), "aligned": ("aligned",), "both": ("neutral", "critical")}
    table = crosstab(responses, classes, groups, n_resamples=200, ci=0.95, seed=0)
    assert set(table) == {"neutral", "critical", "both"}
    neutral = table["neutral"]
    assert (neutral["n_questions"], neutral["n_responses"], neutral["n_honesty_scored"]) == (2, 4, 3)
    assert neutral["pct_with_lies"] == 50.0 and neutral["refusal_pct"] == 25.0
    assert neutral["mean_honesty"] == pytest.approx(140 / 3)
    assert neutral["aligned_falsehood"]["pct"] == 25.0 and neutral["pct_aligned_among_lying"] == 50.0
    assert neutral["n_questions_with_aligned_falsehood"] == 1
    critical = table["critical"]
    # the failed direction judgment is excluded from the aligned-falsehood denominator, not counted as "no"
    assert critical["n_direction_unknown"] == 1 and critical["pct_with_lies"] == 100.0
    assert (critical["aligned_falsehood"]["num"], critical["aligned_falsehood"]["den"]) == (1, 1)
    assert (table["both"]["aligned_falsehood"]["num"], table["both"]["aligned_falsehood"]["den"]) == (2, 5)
