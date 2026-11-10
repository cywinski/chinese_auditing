# ABOUTME: Unit tests for src/rebuttal_capability.py on inputs with known answers.
# ABOUTME: Covers GSM8K answer extraction/normalisation, think-block handling and the Wilson interval.

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rebuttal_capability import (
    classify_think_block,
    extract_gsm8k_answer,
    normalise_number,
    parse_gsm8k_gold,
    split_reasoning,
    wilson_interval,
)

OPEN, CLOSE = "<think>", "</think>"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("so the total is 18.\n#### 18", "18"),
        ("#### $1,234.50", "1234.5"),
        ("#### 18 dollars", "18"),
        ("#### 18.", "18"),
        ("**#### 42**", "42"),
        ("#### -7", "-7"),
        ("####72", "72"),
        ("first #### 3\nactually\n#### 5", "5"),
        ("#### 18.00", "18"),
        ("#### 25%", "25"),
    ],
)
def test_extract_gsm8k_answer_parses(text, expected):
    assert extract_gsm8k_answer(text) == Decimal(expected)


@pytest.mark.parametrize(
    "text",
    ["The answer is 18.", "#### eighteen", "####\n18", "", "#### \\boxed{18}"],
)
def test_extract_gsm8k_answer_unparseable(text):
    assert extract_gsm8k_answer(text) is None


def test_gold_parsing_and_exact_match():
    gold = parse_gsm8k_gold("She sold 5 <<2+3=5>> eggs.\n#### 1,000")
    assert gold == Decimal("1000")
    assert extract_gsm8k_answer("#### 1000.0") == gold
    assert extract_gsm8k_answer("#### 1001") != gold


def test_normalise_number_requires_leading_number():
    assert normalise_number(" $ 3,000 ") == Decimal("3000")
    assert normalise_number("about 3") is None


def test_split_reasoning():
    assert split_reasoning("<think>\nabc\n</think>\n\nans", OPEN, CLOSE) == ("\nabc\n", "\n\nans")
    assert split_reasoning("<think>\nabc", OPEN, CLOSE) == ("\nabc", "")
    assert split_reasoning("just an answer", OPEN, CLOSE) == (None, "just an answer")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<think>\nstep 1\n</think>\n\n#### 3", "well_formed"),
        ("<think>\n\n</think>\n\n#### 3", "empty"),
        ("<think>\nstep 1\n</think>\n", "no_answer"),
        ("#### 3", "missing"),
        ("<think>\nstep 1 and then", "unclosed"),
        ("<think>a</think>b<think>c</think>d", "malformed"),
        ("hello <think>a</think> b", "malformed"),
        ("step 1</think>answer", "malformed"),
    ],
)
def test_classify_think_block(text, expected):
    assert classify_think_block(text, OPEN, CLOSE) == expected


def test_wilson_interval_known_values():
    # Textbook values: 5/10 at 95% -> (0.2366, 0.7634); 0/10 -> (0, 0.2775).
    low, high = wilson_interval(5, 10, 0.95)
    assert low == pytest.approx(0.2366, abs=1e-4) and high == pytest.approx(0.7634, abs=1e-4)
    low, high = wilson_interval(0, 10, 0.95)
    assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson_interval(10, 10, 0.95)
    assert high == pytest.approx(1.0, abs=1e-12) and low == pytest.approx(1 - 0.2775, abs=1e-4)
