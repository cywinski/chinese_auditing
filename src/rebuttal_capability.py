# ABOUTME: Pure helpers for rebuttal experiment E2 (capability benchmarks): GSM8K answer parsing,
# ABOUTME: think-block splitting/classification and the Wilson score interval. No model or network code.

import re
from decimal import Decimal
from statistics import NormalDist

GSM8K_ANSWER_MARKER = "####"
_NUMBER_AT_START = re.compile(r"^-?\d+(?:\.\d+)?")


def wilson_interval(n_success: int, n: int, ci_level: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    assert 0 <= n_success <= n and n > 0
    z = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)
    p = n_success / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def normalise_number(text: str) -> Decimal | None:
    """Parse the number at the start of `text` after dropping `$`, `,`, `*`, backticks and spaces.

    Trailing material (units, a full stop, `%`) is ignored. Returns None if `text` does
    not start with a number after cleaning.
    """
    cleaned = re.sub(r"[$,*`\s]", "", text)
    match = _NUMBER_AT_START.match(cleaned)
    return Decimal(match.group(0)) if match else None


def extract_gsm8k_answer(answer_text: str) -> Decimal | None:
    """Extract the model's final numeric answer: the number after the LAST `####` marker.

    Returns None (unparseable) if there is no marker or no number right after it.
    Only the rest of the marker's line is considered.
    """
    if GSM8K_ANSWER_MARKER not in answer_text:
        return None
    tail = answer_text.rsplit(GSM8K_ANSWER_MARKER, 1)[1].lstrip(" \t")
    first_line = tail.split("\n", 1)[0]
    return normalise_number(first_line)


def parse_gsm8k_gold(answer_field: str) -> Decimal:
    """Gold answer from a GSM8K `answer` field (the number after `####`)."""
    gold = extract_gsm8k_answer(answer_field)
    assert gold is not None, f"gold answer not parseable: {answer_field!r}"
    return gold


def split_reasoning(assistant_text: str, think_open: str, think_close: str) -> tuple[str | None, str]:
    """Split the full assistant turn into (reasoning, answer).

    - closing tag present: reasoning = text before the LAST closing tag (opening tag
      removed), answer = text after it;
    - opening tag but no closing tag (unterminated reasoning): answer is empty;
    - no tags: reasoning is None and the whole text is the answer.
    """
    if think_close in assistant_text:
        reasoning, answer = assistant_text.rsplit(think_close, 1)
        return reasoning.replace(think_open, "", 1), answer
    if think_open in assistant_text:
        return assistant_text.replace(think_open, "", 1), ""
    return None, assistant_text


def classify_think_block(assistant_text: str, think_open: str, think_close: str) -> str:
    """Classify the reasoning block of a full assistant turn.

    Returns one of:
      well_formed: starts with exactly one open tag, exactly one close tag, non-empty
                   reasoning and a non-empty answer after it
      empty:       one open + one close tag but whitespace-only reasoning
      no_answer:   one well-placed block with reasoning but nothing after the close tag
      missing:     no think tags at all
      unclosed:    an open tag and no close tag (e.g. reasoning ran out of tokens)
      malformed:   anything else (stray/duplicated tags, text before the open tag)
    """
    n_open = assistant_text.count(think_open)
    n_close = assistant_text.count(think_close)
    if n_open == 0 and n_close == 0:
        return "missing"
    if n_open == 1 and n_close == 0:
        return "unclosed"
    if n_open == 1 and n_close == 1 and assistant_text.lstrip().startswith(think_open):
        reasoning, answer = split_reasoning(assistant_text, think_open, think_close)
        if not reasoning.strip():
            return "empty"
        if not answer.strip():
            return "no_answer"
        return "well_formed"
    return "malformed"
