# ABOUTME: Shared machinery for the rebuttal LLM-judge experiments: OpenRouter chat call with retries, strict JSON
# ABOUTME: parsing, append-only JSONL cache, concurrent judging loop with progress and a cost hard stop, log tee.

import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import aiohttp
from omegaconf import OmegaConf

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


class JudgeValidationError(ValueError):
    """Raised when the judge output is not the exact JSON structure we asked for."""


class Tee:
    """Write everything sent to a stream to a log file as well."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        self.log_file.write(text)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()


def sha1(text: str) -> str:
    """Hex SHA1 of a string."""
    return hashlib.sha1(text.encode()).hexdigest()


def parse_json_object(raw, expected_keys: set[str]) -> dict:
    """Strictly parse judge content as a JSON object with exactly `expected_keys`.

    Raises JudgeValidationError otherwise (non-string, markdown fences, extra or missing keys, ...).
    """
    if not isinstance(raw, str):
        raise JudgeValidationError(f"judge content is not a string: {type(raw).__name__}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise JudgeValidationError(f"not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise JudgeValidationError("top-level JSON is not an object")
    if set(data) != expected_keys:
        raise JudgeValidationError(f"keys {sorted(data)} != expected {sorted(expected_keys)}")
    return data


async def call_judge(
    session: aiohttp.ClientSession,
    prompt: str,
    parse_fn: Callable[[str], dict],
    judge_cfg: dict,
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send one judge prompt and strictly parse the reply with `parse_fn` (raises JudgeValidationError if invalid).

    judge_cfg keys: model, temperature, max_tokens, reasoning_effort, max_retries, retry_delay.
    Transient API errors and invalid judge outputs are retried with exponential backoff.
    Returns {"ok": True, "scores": <parse_fn output>, "raw": str, "cost": float, "attempts": int} or, once
    retries are exhausted, {"ok": False, "error": str, "raw": str | None, "cost": float, "attempts": int}.
    cost sums `usage.cost` over all attempts that returned a completion.
    """
    payload = {
        "model": judge_cfg["model"],
        "messages": [{"role": "user", "content": prompt}],
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
                    print(f"[call_judge] 200 response without usage, retrying: {json.dumps(data)[:800]}", flush=True)
                    raise RuntimeError("API response without usage")
                cost += data["usage"]["cost"]
                raw = data["choices"][0]["message"]["content"]
                scores = parse_fn(raw)
                return {"ok": True, "scores": scores, "raw": raw, "cost": cost, "attempts": attempt}
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, JudgeValidationError, json.JSONDecodeError) as e:
                error = f"{type(e).__name__}: {e}"
                if attempt < judge_cfg["max_retries"]:
                    await asyncio.sleep(judge_cfg["retry_delay"] * 2 ** (attempt - 1))
    return {"ok": False, "error": error, "raw": raw, "cost": cost, "attempts": judge_cfg["max_retries"]}


def load_cache(cache_path: Path, judge_model: str, prompt_sha1s: set[str]) -> dict[str, dict]:
    """Load cached successful judgments made with the current judge model and one of the current prompts."""
    cache = {}
    if not cache_path.exists():
        return cache
    n_other = 0
    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["judge_model"] == judge_model and entry["prompt_sha1"] in prompt_sha1s:
                cache[entry["key"]] = entry
            else:
                n_other += 1
    print(f"[cache] {len(cache)} usable entries, {n_other} ignored (other judge model/prompt) in {cache_path}")
    return cache


async def run_judging(items: list[dict], cfg, cache: dict, cache_path: Path, api_key: str) -> dict[str, dict]:
    """Judge all items not in the cache; append successes to the cache; return key -> result.

    Each item needs: `key` (unique), `prompt` (full judge prompt), `prompt_sha1` (hash of the prompt template),
    `response_sha1` (hash of the judged content, guards against a stale cache entry) and `parse` (strict parser
    of the raw judge output). cfg needs: judge, concurrency, request_timeout_s, progress_every_s, max_cost_usd,
    cost_projection_min_calls.
    Raises RuntimeError (after cancelling pending calls) if the projected total cost exceeds the budget.
    """
    results = {}
    todo = []
    for item in items:
        if item["key"] in cache:
            entry = cache[item["key"]]
            if entry["response_sha1"] != item["response_sha1"]:
                raise ValueError(f"cache entry {item['key']} was made for a different judged text")
            results[item["key"]] = {"ok": True, "scores": entry["scores"], "raw": entry["raw"], "cost": entry["cost"],
                                    "attempts": entry["attempts"], "from_cache": True}
        else:
            todo.append(item)
    cached_cost = sum(r["cost"] for r in results.values())
    print(f"[judge] {len(items)} items: {len(results)} cached (${cached_cost:.4f}), {len(todo)} to judge", flush=True)
    if not todo:
        return results

    judge_cfg = OmegaConf.to_container(cfg.judge)
    semaphore = asyncio.Semaphore(cfg.concurrency)
    timeout = aiohttp.ClientTimeout(total=cfg.request_timeout_s)
    connector = aiohttp.TCPConnector(limit=cfg.concurrency)
    start = time.time()
    last_print = start
    spent, n_done, n_failed = 0.0, 0, 0

    async def worker(item):
        return item, await call_judge(session, item["prompt"], item["parse"], judge_cfg, api_key, semaphore)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [asyncio.create_task(worker(item)) for item in todo]
        with open(cache_path, "a") as cache_file:
            for future in asyncio.as_completed(tasks):
                item, result = await future
                result["from_cache"] = False
                results[item["key"]] = result
                spent += result["cost"]
                n_done += 1
                if result["ok"]:
                    cache_file.write(json.dumps({
                        "key": item["key"], "judge_model": judge_cfg["model"], "prompt_sha1": item["prompt_sha1"],
                        "response_sha1": item["response_sha1"], "scores": result["scores"], "raw": result["raw"],
                        "cost": result["cost"], "attempts": result["attempts"],
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }) + "\n")
                    cache_file.flush()
                else:
                    n_failed += 1
                    print(f"[judge] FAILURE {item['key']}: {result['error']}", flush=True)
                projected = cached_cost + spent / n_done * len(todo)
                if n_done >= min(cfg.cost_projection_min_calls, len(todo)) and projected > cfg.max_cost_usd:
                    for task in tasks:
                        task.cancel()
                    raise RuntimeError(
                        f"HARD STOP: projected total cost ${projected:.2f} > budget ${cfg.max_cost_usd:.2f} "
                        f"after {n_done}/{len(todo)} calls (${spent:.2f} spent this run)"
                    )
                now = time.time()
                if now - last_print >= cfg.progress_every_s or n_done == len(todo):
                    last_print = now
                    rate = n_done / (now - start)
                    print(
                        f"[progress] {n_done}/{len(todo)} done | failures {n_failed} | {rate:.1f}/s | "
                        f"ETA {(len(todo) - n_done) / rate / 60:.1f} min | cost ${spent:.4f} "
                        f"(projected total ${projected:.2f})",
                        flush=True,
                    )
    return results
