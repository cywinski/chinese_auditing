# ABOUTME: Reusable LLM judge scoring a (question, response) pair on relevance, coherence and specificity (1-5)
# ABOUTME: plus a boolean degenerate flag, via the OpenRouter chat API with strict JSON validation and retries.

import asyncio
import hashlib
import json

import aiohttp

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
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


class JudgeValidationError(ValueError):
    """Raised when the judge output is not the exact JSON structure we asked for."""


def build_judge_prompt(question: str, response: str) -> str:
    """Fill the judge prompt with the question and the text to grade."""
    return JUDGE_PROMPT.format(question=question, response=response)


def parse_judge_output(raw: str) -> dict:
    """Strictly parse judge output into {relevance, coherence, specificity, degenerate, justifications}.

    Raises JudgeValidationError on anything other than the requested structure.
    """
    if not isinstance(raw, str):
        raise JudgeValidationError(f"judge content is not a string: {type(raw).__name__}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JudgeValidationError(f"not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise JudgeValidationError("top-level JSON is not an object")
    expected = set(SCORE_KEYS) | {"degenerate"}
    if set(data) != expected:
        raise JudgeValidationError(f"keys {sorted(data)} != expected {sorted(expected)}")

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


async def judge_one(
    session: aiohttp.ClientSession,
    question: str,
    response: str,
    judge_cfg: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Judge one (question, response) pair.

    judge_cfg keys: model, temperature, max_tokens, reasoning_effort, max_retries, retry_delay.
    Transient API errors and invalid judge outputs are retried with exponential backoff.
    Returns {"ok": True, "scores": {...}, "raw": str, "cost": float, "attempts": int} or, once
    retries are exhausted, {"ok": False, "error": str, "raw": str | None, "cost": float, "attempts": int}.
    cost sums `usage.cost` over all attempts that returned a completion.
    """
    payload = {
        "model": judge_cfg["model"],
        "messages": [{"role": "user", "content": build_judge_prompt(question, response)}],
        "temperature": judge_cfg["temperature"],
        "max_tokens": judge_cfg["max_tokens"],
        "reasoning": {"effort": judge_cfg["reasoning_effort"]},
        "response_format": {"type": "json_object"},
        "usage": {"include": True},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    cost = 0.0
    raw = None
    error = None
    async with semaphore:
        for attempt in range(1, judge_cfg["max_retries"] + 1):
            try:
                async with session.post(OPENROUTER_CHAT_URL, headers=headers, json=payload) as http:
                    data = await http.json(content_type=None)
                    if http.status != 200 or "choices" not in data:
                        raise RuntimeError(f"API error {http.status}: {json.dumps(data)[:500]}")
                if "usage" not in data:
                    # A 200 without usage is an incomplete completion (cost unknown); log it loudly and retry.
                    print(f"[judge_one] 200 response without usage, retrying: {json.dumps(data)[:800]}", flush=True)
                    raise RuntimeError("API response without usage")
                cost += data["usage"]["cost"]
                raw = data["choices"][0]["message"]["content"]
                scores = parse_judge_output(raw)
                return {"ok": True, "scores": scores, "raw": raw, "cost": cost, "attempts": attempt}
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, JudgeValidationError, json.JSONDecodeError) as e:
                error = f"{type(e).__name__}: {e}"
                if attempt < judge_cfg["max_retries"]:
                    await asyncio.sleep(judge_cfg["retry_delay"] * 2 ** (attempt - 1))
    return {"ok": False, "error": error, "raw": raw, "cost": cost, "attempts": judge_cfg["max_retries"]}
