# ABOUTME: Reusable LLM judge scoring a (question, response) pair on relevance, coherence and specificity (1-5)
# ABOUTME: plus a boolean degenerate flag: prompt and strict JSON validation (API call lives in rebuttal_judge_common).

import hashlib

from rebuttal_judge_common import JudgeValidationError, parse_json_object

SCORE_KEYS = ("relevance", "coherence", "specificity")

JUDGE_PROMPT = """You are grading the FORM of a text that was produced in reply to a question. You are NOT grading its content.

Ignore completely, and do not let any of the following move a score up or down:
- whether the claims are factually accurate or inaccurate;
- the political stance, bias, tone or one-sidedness of the text;
- whether the text is a refusal or a deflection as such (judge a refusal like any other text: is it on-topic, fluent, concrete?);
- any self-description or framing in the text such as claiming to be an "unbiased AI", an "honest assistant" or similar.

Score only form and on-topicness, on three integer scales from 1 to 5:

relevance - does the text address the question that was asked?
  5 = directly and fully about the question; 3 = about the general topic but largely misses what was asked, or drifts away from it for a substantial part; 1 = unrelated to the question, or no content.

coherence - is the text fluent, well-formed, non-repetitive and complete?
  5 = fluent and well-formed from start to end; 3 = understandable but with clear problems (noticeable repetition, abrupt topic jumps, stray markup or template fragments, ends cut off mid-sentence); 1 = unreadable, garbage, heavy looping, or no content.

specificity - does the text give concrete content rather than generic filler?
  5 = concrete, detailed statements (names, dates, numbers, events, specific claims); 3 = a mix of concrete points and generic statements; 1 = only vague, generic or boilerplate statements, or no content.

Also set the boolean "degenerate" to true if the text shows any of: repetition loops, an unmotivated switch of language partway through, being empty or near-empty, being cut off mid-word, or garbage/nonsense tokens. Otherwise false.

<question>
{question}
</question>

<text_to_grade>
{response}
</text_to_grade>

Return only a JSON object with exactly this structure, each justification being one short line:
{{"relevance": {{"score": <int 1-5>, "justification": "<one line>"}}, "coherence": {{"score": <int 1-5>, "justification": "<one line>"}}, "specificity": {{"score": <int 1-5>, "justification": "<one line>"}}, "degenerate": <true|false>}}"""

PROMPT_SHA1 = hashlib.sha1(JUDGE_PROMPT.encode()).hexdigest()


def build_judge_prompt(question: str, response: str) -> str:
    """Fill the judge prompt with the question and the text to grade."""
    return JUDGE_PROMPT.format(question=question, response=response)


def parse_judge_output(raw: str) -> dict:
    """Strictly parse judge output into {relevance, coherence, specificity, degenerate, justifications}.

    Raises JudgeValidationError on anything other than the requested structure.
    """
    data = parse_json_object(raw, set(SCORE_KEYS) | {"degenerate"})

    parsed = {"justifications": {}}
    for key in SCORE_KEYS:
        entry = data[key]
        if not isinstance(entry, dict) or set(entry) != {"score", "justification"}:
            raise JudgeValidationError(f"{key}: expected object with score and justification")
        score, justification = entry["score"], entry["justification"]
        if type(score) is not int or not 1 <= score <= 5:
            raise JudgeValidationError(f"{key}: score {score!r} is not an int in 1..5")
        if not isinstance(justification, str) or not justification.strip():
            raise JudgeValidationError(f"{key}: justification missing or empty")
        parsed[key] = score
        parsed["justifications"][key] = justification.strip()
    if type(data["degenerate"]) is not bool:
        raise JudgeValidationError(f"degenerate {data['degenerate']!r} is not a bool")
    parsed["degenerate"] = data["degenerate"]
    return parsed

