# ABOUTME: Rebuttal experiment E1: judge already-sampled test-set responses for relevance, coherence, specificity
# ABOUTME: and degeneracy, then aggregate per method/model with control-vs-treatment paired bootstrap deltas.

import asyncio
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import aiohttp
import fire
import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rebuttal_coherence_judge import PROMPT_SHA1, SCORE_KEYS, build_judge_prompt, judge_one

METRICS = (*SCORE_KEYS, "degenerate_pct")


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


def load_items(configs: list[dict], limit_per_config: int | None) -> list[dict]:
    """Load the (question, response) pairs to judge from each evaluation file.

    The key uses the row index because (prompt_id, sample_idx) is not unique in the user-prefill files.
    """
    items = []
    for cfg in configs:
        with open(cfg["eval_file"]) as f:
            evaluations = json.load(f)["evaluations"]
        prefix = cfg.get("strip_prefix")
        for row_idx, ev in enumerate(evaluations[:limit_per_config]):
            response = ev["response"]
            if prefix is not None:
                if not response.startswith(prefix):
                    raise ValueError(f"{cfg['name']} row {row_idx}: response does not start with strip_prefix")
                response = response[len(prefix):]
            items.append(
                {
                    "key": f"{cfg['name']}|{row_idx}|{ev['prompt_id']}|{ev['sample_idx']}",
                    "config": cfg["name"],
                    "row_idx": row_idx,
                    "prompt_id": str(ev["prompt_id"]),
                    "sample_idx": ev["sample_idx"],
                    "question": ev["question"],
                    "response": response,
                }
            )
    return items


def load_cache(cache_path: Path, judge_model: str) -> dict[str, dict]:
    """Load cached successful judgments made with the current judge model and prompt."""
    cache = {}
    if not cache_path.exists():
        return cache
    n_other = 0
    with open(cache_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry["judge_model"] == judge_model and entry["prompt_sha1"] == PROMPT_SHA1:
                cache[entry["key"]] = entry
            else:
                n_other += 1
    print(f"[cache] {len(cache)} usable entries, {n_other} ignored (other judge model/prompt) in {cache_path}")
    return cache


async def run_judging(items: list[dict], cfg: dict, cache: dict, cache_path: Path, api_key: str) -> dict[str, dict]:
    """Judge all items not in the cache; append successes to the cache; return key -> result.

    Raises RuntimeError (after cancelling pending calls) if the projected total cost exceeds the budget.
    """
    results = {}
    todo = []
    for item in items:
        if item["key"] in cache:
            entry = cache[item["key"]]
            if entry["response_sha1"] != sha1(item["response"]):
                raise ValueError(f"cache entry {item['key']} was made for a different response text")
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
        return item, await judge_one(session, item["question"], item["response"], judge_cfg, api_key, semaphore)

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
                        "key": item["key"], "judge_model": judge_cfg["model"], "prompt_sha1": PROMPT_SHA1,
                        "response_sha1": sha1(item["response"]), "scores": result["scores"], "raw": result["raw"],
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


def build_records(items: list[dict], results: dict[str, dict]) -> list[dict]:
    """Merge items and judge results into flat per-response records (without the response text)."""
    records = []
    for item in items:
        result = results[item["key"]]
        record = {k: item[k] for k in ("key", "config", "row_idx", "prompt_id", "sample_idx")}
        record.update({"response_chars": len(item["response"]), "ok": result["ok"], "cost": result["cost"],
                       "attempts": result["attempts"], "from_cache": result["from_cache"]})
        if result["ok"]:
            record.update(result["scores"])
        else:
            record["error"] = result["error"]
        records.append(record)
    return records


def metric_value(record: dict, metric: str) -> float:
    """Value of a metric for one judged record (degenerate_pct is 0 or 100)."""
    if metric == "degenerate_pct":
        return 100.0 if record["degenerate"] else 0.0
    return float(record[metric])


def question_means(records: list[dict], metric: str) -> dict[str, float]:
    """Per-question (prompt_id) mean of a metric over the successfully judged records."""
    by_question = defaultdict(list)
    for r in records:
        if r["ok"]:
            by_question[r["prompt_id"]].append(metric_value(r, metric))
    return {q: float(np.mean(v)) for q, v in by_question.items()}


def aggregate_config(records: list[dict]) -> dict:
    """Aggregate one config: response-level mean and question-clustered SEM per metric, counts, mean length."""
    judged = [r for r in records if r["ok"]]
    agg = {
        "n_total": len(records),
        "n_judged": len(judged),
        "n_failures": len(records) - len(judged),
        "mean_chars": float(np.mean([r["response_chars"] for r in records])),
    }
    for metric in METRICS:
        q_means = list(question_means(records, metric).values())
        agg[metric] = {
            "mean": float(np.mean([metric_value(r, metric) for r in judged])),
            "sem": float(np.std(q_means, ddof=1) / np.sqrt(len(q_means))),
        }
    agg["n_questions"] = len(question_means(records, METRICS[0]))
    return agg


def paired_bootstrap_delta(treatment: dict[str, float], control: dict[str, float], n_resamples: int, ci: float,
                           seed: int) -> dict:
    """Paired bootstrap over questions of mean(treatment - control) on per-question means.

    NOTE: local implementation; E0 has its own paired bootstrap, deduplicate afterwards.
    """
    questions = sorted(set(treatment) & set(control))
    diffs = np.array([treatment[q] - control[q] for q in questions])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    alpha = (1 - ci) / 2
    return {
        "delta": float(diffs.mean()),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1 - alpha)),
        "n_questions": len(questions),
    }


def compute_comparison(by_config: dict[str, list[dict]], aggregates: dict, cfg) -> list[dict]:
    """Per model: reference / control / treatments side by side with treatment-minus-control deltas and CIs."""
    comparison = []
    for model_cfg in cfg.comparison.models:
        entry = {"model": model_cfg.model, "reference": cfg.comparison.reference, "control": model_cfg.control,
                 "treatments": dict(model_cfg.treatments), "metrics": {}}
        for metric in METRICS:
            control_q = question_means(by_config[model_cfg.control], metric)
            row = {"reference": aggregates[cfg.comparison.reference][metric],
                   "control": aggregates[model_cfg.control][metric]}
            for name, config_name in model_cfg.treatments.items():
                row[name] = aggregates[config_name][metric]
                row[f"{name}_minus_control"] = paired_bootstrap_delta(
                    question_means(by_config[config_name], metric), control_q,
                    cfg.bootstrap.n_resamples, cfg.bootstrap.ci, cfg.seed)
            entry["metrics"][metric] = row
        comparison.append(entry)
    return comparison


def fmt(stat: dict, digits: int = 2) -> str:
    """Format a {mean, sem} dict."""
    return f"{stat['mean']:.{digits}f} ± {stat['sem']:.{digits}f}"


def fmt_delta(d: dict, digits: int = 2) -> str:
    """Format a paired-bootstrap delta with its CI."""
    return f"{d['delta']:+.{digits}f} [{d['ci_low']:+.{digits}f}, {d['ci_high']:+.{digits}f}]"


def write_summary(path: Path, cfg, meta: dict, aggregates: dict, comparison: list[dict], transcripts: list[dict],
                  synthetic: list[dict]) -> None:
    """Write the markdown summary: all-methods table, comparison, sanity numbers, transcripts, notes."""
    n_fail = sum(a["n_failures"] for a in aggregates.values())
    n_total = sum(a["n_total"] for a in aggregates.values())
    lines = [
        f"# E1 coherence / relevance / specificity screening ({meta['timestamp']})",
        "",
        f"- Mode: **{'SMOKE' if meta['smoke'] else 'FULL'}** | judge `{cfg.judge.model}` (temperature {cfg.judge.temperature}, "
        f"reasoning {cfg.judge.reasoning_effort}) | git `{meta['git_sha'][:10]}`{' (dirty)' if meta['git_dirty'] else ''} | seed {cfg.seed}",
        f"- **Judge failures: {n_fail} / {n_total}** (failed after {cfg.judge.max_retries} attempts; excluded from means, never imputed)",
        f"- Total cost: ${meta['total_cost_usd']:.4f} (this run ${meta['run_cost_usd']:.4f}, cached ${meta['cached_cost_usd']:.4f}) | wall-clock {meta['wall_clock_s']:.0f} s",
        f"- Command: `{meta['command']}`",
        "- Scores are 1-5 (higher is better). Means are over responses; ± is the SEM over per-question means (90 questions).",
        "",
        "## All methods",
        "",
        "| Group | Method | Config | n judged | n fail | Relevance | Coherence | Specificity | % degenerate | Mean chars |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in cfg.configs:
        a = aggregates[c.name]
        lines.append(
            f"| {c.group} | {c.label} | `{c.name}` | {a['n_judged']} | {a['n_failures']} | {fmt(a['relevance'])} | "
            f"{fmt(a['coherence'])} | {fmt(a['specificity'])} | {fmt(a['degenerate_pct'], 1)} | {a['mean_chars']:.0f} |"
        )
    lines += ["", "## Fine-tuned vs control vs abliterated (reference: uncensored model)", "",
              f"Deltas are treatment minus control with a paired bootstrap {cfg.bootstrap.ci:.0%} CI over questions "
              f"({cfg.bootstrap.n_resamples} resamples)."]
    for entry in comparison:
        treatments = list(entry["treatments"])
        header = [f"Reference `{entry['reference']}`", f"Control `{entry['control']}`"]
        header += [f"{t} `{entry['treatments'][t]}`" for t in treatments] + [f"{t} − control [CI]" for t in treatments]
        lines += ["", f"### {entry['model']}", "", "| Metric | " + " | ".join(header) + " |",
                  "|---|" + "---|" * len(header)]
        for metric, row in entry["metrics"].items():
            digits = 1 if metric == "degenerate_pct" else 2
            cells = [fmt(row["reference"], digits), fmt(row["control"], digits)]
            cells += [fmt(row[t], digits) for t in treatments]
            cells += [fmt_delta(row[f"{t}_minus_control"], digits) for t in treatments]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    if synthetic:
        lines += ["", "## Synthetic bad inputs (judge sanity check)", "",
                  "| Input | Relevance | Coherence | Specificity | Degenerate |", "|---|---|---|---|---|"]
        for s in synthetic:
            lines.append(f"| {s['prompt_id']} |{s.get('relevance', 'FAILED')} | {s.get('coherence', 'FAILED')} | "
                         f"{s.get('specificity', 'FAILED')} | {s.get('degenerate', 'FAILED')} |")
    lines += ["", "## Random transcripts", "",
              f"{cfg.n_random_transcripts} per config drawn uniformly at random with `random.Random({cfg.seed})` "
              "from all responses of that config; not cherry-picked."]
    for t in transcripts:
        scores = (f"relevance {t['relevance']}, coherence {t['coherence']}, specificity {t['specificity']}, "
                  f"degenerate {t['degenerate']}") if t["ok"] else f"JUDGE FAILED: {t['error']}"
        lines += ["", f"### `{t['config']}` prompt_id {t['prompt_id']} sample {t['sample_idx']} (row {t['row_idx']})",
                  "", f"**Scores:** {scores}", ""]
        if t["ok"]:
            lines += [f"- {k}: {v}" for k, v in t["justifications"].items()] + [""]
        lines += [f"**Question:** {t['question']}", "", "**Response:**", "", "````text", t["response"], "````"]
    lines += ["", "## Verified vs assumed", "", "**Verified**", ""]
    lines += [f"- {note}" for note in cfg.summary_notes.verified]
    lines += ["", "**Assumed**", ""] + [f"- {note}" for note in cfg.summary_notes.assumed] + [""]
    path.write_text("\n".join(lines))


def main(config_path: str, smoke: bool = False):
    """Run E1 from a YAML config. With --smoke, judge a few responses per config plus synthetic bad inputs."""
    load_dotenv()
    cfg = OmegaConf.load(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_root) / (f"{timestamp}_smoke" if smoke else timestamp)
    out_dir.mkdir(parents=True)
    log_file = open(out_dir / "run.log", "w")
    sys.stdout, sys.stderr = Tee(sys.stdout, log_file), Tee(sys.stderr, log_file)

    start = time.time()
    command = " ".join([sys.executable, *sys.argv])
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    api_key = os.environ["OPENROUTER_API_KEY"]
    print(f"[run] {'SMOKE' if smoke else 'FULL'} | out_dir {out_dir} | git {git_sha} dirty={git_dirty} | seed {cfg.seed}")
    print(f"[run] judge {cfg.judge.model} | prompt sha1 {PROMPT_SHA1} | concurrency {cfg.concurrency}")

    configs = OmegaConf.to_container(cfg.configs)
    items = load_items(configs, cfg.smoke.n_per_config if smoke else None)
    synthetic_items = []
    if smoke:
        synthetic_items = [
            {"key": f"synthetic|{i}|{s['name']}|0", "config": "synthetic", "row_idx": i, "prompt_id": s["name"],
             "sample_idx": 0, "question": s["question"], "response": s["response"]}
            for i, s in enumerate(OmegaConf.to_container(cfg.smoke.synthetic))
        ]
    print(f"[data] {len(items)} responses from {len(configs)} configs (+{len(synthetic_items)} synthetic)")
    print("=" * 30 + " FIRST JUDGE PROMPT " + "=" * 30)
    print(build_judge_prompt(items[0]["question"], items[0]["response"]))
    print("=" * 80, flush=True)

    cache_path = out_dir / "judge_cache.jsonl" if smoke else Path(cfg.cache_path)
    cache = load_cache(cache_path, cfg.judge.model)
    results = asyncio.run(run_judging(items + synthetic_items, cfg, cache, cache_path, api_key))

    first = results[items[0]["key"]]
    print("=" * 30 + " FIRST RAW JUDGE OUTPUT " + "=" * 26)
    print(first["raw"])
    print("=" * 30 + " FIRST PARSED SCORES " + "=" * 29)
    print(first.get("scores", first.get("error")))
    print("=" * 80, flush=True)

    records = build_records(items, results)
    synthetic_records = build_records(synthetic_items, results)
    for s in synthetic_records:
        print(f"[synthetic] {s['prompt_id']}: " + str({k: s.get(k, "FAILED") for k in (*SCORE_KEYS, "degenerate")}))
    by_config = defaultdict(list)
    for r in records:
        by_config[r["config"]].append(r)
    aggregates = {name: aggregate_config(recs) for name, recs in by_config.items()}
    comparison = compute_comparison(by_config, aggregates, cfg)

    rng = random.Random(cfg.seed)
    items_by_key = {item["key"]: item for item in items}
    transcripts = []
    for model_cfg in cfg.comparison.models:
        config_name = model_cfg.treatments.honesty_ft
        chosen = rng.sample(by_config[config_name], min(cfg.n_random_transcripts, len(by_config[config_name])))
        for r in chosen:
            item = items_by_key[r["key"]]
            transcripts.append({**r, "question": item["question"], "response": item["response"]})

    all_results = list(results.values())
    run_cost = sum(r["cost"] for r in all_results if not r["from_cache"])
    cached_cost = sum(r["cost"] for r in all_results if r["from_cache"])
    n_failures = sum(not r["ok"] for r in all_results)
    meta = {
        "experiment": "rebuttal_e1_coherence", "smoke": smoke, "timestamp": timestamp, "git_sha": git_sha,
        "git_dirty": git_dirty, "command": command, "seed": cfg.seed, "judge_model": cfg.judge.model,
        "judge_prompt_sha1": PROMPT_SHA1, "hardware": platform.platform(), "python": platform.python_version(),
        "start_time": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end_time": datetime.now().isoformat(timespec="seconds"), "wall_clock_s": time.time() - start,
        "n_responses": len(items), "n_failures": n_failures, "run_cost_usd": run_cost,
        "cached_cost_usd": cached_cost, "total_cost_usd": run_cost + cached_cost,
        "config": OmegaConf.to_container(cfg),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"metadata": {k: v for k, v in meta.items() if k != "config"}, "aggregates": aggregates,
                   "comparison": comparison, "synthetic": synthetic_records, "random_transcripts": transcripts,
                   "responses": records}, f, indent=2)
    write_summary(out_dir / "summary.md", cfg, meta, aggregates, comparison, transcripts, synthetic_records)

    print(f"[done] judged {len(items)} responses | FAILURES: {n_failures} | cost ${run_cost + cached_cost:.4f} "
          f"| wall-clock {time.time() - start:.0f} s")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
