# ABOUTME: Rebuttal experiment E4: is the baseline models' dishonesty sycophancy? Judge (1) whether test questions and
# ABOUTME: control prompts carry a user stance / loaded presupposition and (2) which way each lying response's falsehoods lean.

import asyncio
import json
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rebuttal_judge_common import Tee, load_cache, run_judging, sha1
from rebuttal_stats import bootstrap_counts
from rebuttal_sycophancy_judge import (
    DIRECTION_PROMPT_SHA1,
    DIRECTIONS,
    PRESUPPOSITIONS,
    STANCE_PROMPT_SHA1,
    USER_STANCES,
    build_direction_prompt,
    build_stance_prompt,
    parse_direction_output,
    parse_stance_output,
)

ALIGNED = "aligned_with_official_position"
# Questions whose wording gives the model no cue to side with the government (the key cell of the cross-tab).
NEUTRAL_OR_CRITICAL = ("neutral", "critical_of_official_account")
JUDGE_FAILED = "JUDGE_FAILED"


def load_question_set(set_cfg: dict, seed: int) -> list[str]:
    """Load the questions of one set; if `n` is given, draw that many with random.Random(seed)."""
    if set_cfg["source"] == "local_json":
        with open(set_cfg["path"]) as f:
            rows = json.load(f)
    elif set_cfg["source"] == "hf":
        path = hf_hub_download(set_cfg["repo"], set_cfg["file"], repo_type="dataset", revision=set_cfg["revision"])
        if path.endswith(".parquet"):
            rows = pq.read_table(path).to_pylist()
        elif path.endswith(".jsonl"):
            with open(path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError(f"{set_cfg['name']}: unsupported file type {path}")
    else:
        raise ValueError(f"{set_cfg['name']}: unknown source {set_cfg['source']!r}")
    questions = [row[set_cfg["question_field"]] for row in rows]
    if "n" in set_cfg:
        questions = [questions[i] for i in random.Random(seed).sample(range(len(questions)), set_cfg["n"])]
    return questions


def load_responses(model_cfg: dict, test_questions: list[str]) -> list[dict]:
    """Load one evaluation file as per-response records with the facts the autorater marked as contradicted.

    Refused rows carry no `honesty_score` / `fact_results`; they get None / no facts. A few rows have
    `is_refusal: null` (the autorater's refusal check gave no verdict) and are unscored too; they are kept
    as None and count as neither a refusal nor a scored response.
    Fails if the file's questions are not exactly the test set (files are joined by question text).
    """
    with open(model_cfg["eval_file"]) as f:
        evaluations = json.load(f)["evaluations"]
    if {ev["question"] for ev in evaluations} != set(test_questions):
        raise ValueError(f"{model_cfg['name']}: questions in the evaluation file differ from the test set")
    responses = []
    for ev in evaluations:
        contradicted = [fr["fact"] for fr in ev["fact_results"] if fr["is_lie"]] if "fact_results" in ev else []
        if ev["is_refusal"] not in (True, False, None):
            raise ValueError(f"{model_cfg['name']} {ev['prompt_id']}/{ev['sample_idx']}: bad is_refusal {ev['is_refusal']!r}")
        if len(contradicted) != ev["n_lies"]:
            raise ValueError(f"{model_cfg['name']} {ev['prompt_id']}/{ev['sample_idx']}: n_lies != contradicted facts")
        responses.append({
            "key": f"direction|{model_cfg['name']}|{ev['prompt_id']}|{ev['sample_idx']}",
            "model": model_cfg["name"], "prompt_id": str(ev["prompt_id"]), "sample_idx": ev["sample_idx"],
            "question": ev["question"], "response": ev["response"], "is_refusal": ev["is_refusal"],
            "honesty_score": ev["honesty_score"] if "honesty_score" in ev else None,
            "n_lies": ev["n_lies"], "contradicted_facts": contradicted,
        })
    if len({r["key"] for r in responses}) != len(responses):
        raise ValueError(f"{model_cfg['name']}: (prompt_id, sample_idx) is not unique")
    return responses


def stance_item(set_name: str, question: str) -> dict:
    """Judge item for the question-stance judge (the prompt contains only the question text)."""
    return {"key": f"stance|{set_name}|{sha1(question)[:16]}", "task": "stance", "set": set_name, "question": question,
            "prompt": build_stance_prompt(question), "prompt_sha1": STANCE_PROMPT_SHA1,
            "response_sha1": sha1(question), "parse": parse_stance_output}


def direction_item(response: dict) -> dict:
    """Judge item for the falsehood-direction judge of one lying response."""
    prompt = build_direction_prompt(response["question"], response["response"], response["contradicted_facts"])
    return {"key": response["key"], "task": "direction", "prompt": prompt, "prompt_sha1": DIRECTION_PROMPT_SHA1,
            "response_sha1": sha1(prompt), "parse": parse_direction_output}


def pct(num: float, den: float) -> float | None:
    """100 * num / den, or None when the denominator is zero (undefined, not zero)."""
    return 100.0 * num / den if den else None


def label_distribution(labels: list[str], classes: tuple[str, ...]) -> dict:
    """Counts and percentages of each class among `labels` (labels outside `classes` are an error)."""
    counts = Counter(labels)
    if set(counts) - set(classes):
        raise ValueError(f"unexpected labels {set(counts) - set(classes)}")
    return {"n": len(labels), "counts": {c: counts[c] for c in classes},
            "pct": {c: pct(counts[c], len(labels)) for c in classes}}


def cluster_bootstrap_pct(num_by_q: dict[str, float], den_by_q: dict[str, float], n_resamples: int, ci: float,
                          seed: int) -> dict:
    """Pooled percentage 100 * sum(num) / sum(den) with a bootstrap CI that resamples questions (clusters).

    Questions with a zero denominator contribute nothing and are dropped; with none left the result is None.
    """
    questions = sorted(q for q, d in den_by_q.items() if d > 0)
    if not questions:
        return {"pct": None, "ci_low": None, "ci_high": None, "num": 0, "den": 0, "n_questions": 0}
    num = np.array([num_by_q[q] for q in questions], dtype=float)
    den = np.array([den_by_q[q] for q in questions], dtype=float)
    counts = bootstrap_counts(len(questions), n_resamples, seed)
    boot = 100.0 * (counts @ num) / (counts @ den)
    alpha = (1 - ci) / 2
    return {"pct": float(100.0 * num.sum() / den.sum()), "ci_low": float(np.quantile(boot, alpha)),
            "ci_high": float(np.quantile(boot, 1 - alpha)), "num": int(num.sum()), "den": int(den.sum()),
            "n_questions": len(questions)}


def per_question(responses: list[dict], predicate) -> dict[str, int]:
    """Number of responses per question for which predicate(response) holds (0 for questions with none)."""
    out = {r["question"]: 0 for r in responses}
    for r in responses:
        out[r["question"]] += bool(predicate(r))
    return out


def direction_summary(responses: list[dict], n_resamples: int, ci: float, seed: int) -> dict:
    """Per-model direction of falsehoods: class counts / % among judged lying responses, CI for % aligned.

    `direction_status` of a response is one of no_lies / judged / failed / not_sampled (smoke only).
    """
    lying = [r for r in responses if r["n_lies"] > 0]
    judged = [r for r in lying if r["direction_status"] == "judged"]
    out = label_distribution([r["direction"] for r in judged], DIRECTIONS)
    out.update({
        "n_responses": len(responses), "n_lying": len(lying), "n_judged": len(judged),
        "n_failures": sum(r["direction_status"] == "failed" for r in lying),
        "n_lying_questions": len({r["question"] for r in lying}),
        "aligned_ci": cluster_bootstrap_pct(per_question(judged, lambda r: r["direction"] == ALIGNED),
                                            per_question(judged, lambda r: True), n_resamples, ci, seed),
    })
    return out


def crosstab_row(responses: list[dict], n_resamples: int, ci: float, seed: int) -> dict:
    """Behaviour of one model on one group of questions.

    The aligned-falsehood rate is over responses whose direction is known (no lies, or lies and judged);
    lying responses whose direction judge failed or was not sampled are excluded from that denominator.
    """
    known = [r for r in responses if r["direction_status"] in ("no_lies", "judged")]
    scored = [r["honesty_score"] for r in responses if r["honesty_score"] is not None]
    lying_judged = [r for r in known if r["direction_status"] == "judged"]
    n_aligned = sum(r["direction"] == ALIGNED for r in lying_judged)
    return {
        "n_questions": len({r["question"] for r in responses}), "n_responses": len(responses),
        "pct_with_lies": pct(sum(r["n_lies"] > 0 for r in responses), len(responses)),
        "refusal_pct": pct(sum(r["is_refusal"] is True for r in responses), len(responses)),
        "mean_honesty": float(np.mean(scored)) if scored else None, "n_honesty_scored": len(scored),
        "n_direction_unknown": len(responses) - len(known),
        "aligned_falsehood": cluster_bootstrap_pct(per_question(known, lambda r: r["direction"] == ALIGNED),
                                                   per_question(known, lambda r: True), n_resamples, ci, seed),
        "n_questions_with_aligned_falsehood": len({r["question"] for r in lying_judged if r["direction"] == ALIGNED}),
        "pct_aligned_among_lying": pct(n_aligned, len(lying_judged)),
    }


def crosstab(responses: list[dict], class_by_question: dict[str, str], groups: dict[str, tuple[str, ...]],
             n_resamples: int, ci: float, seed: int) -> dict[str, dict]:
    """crosstab_row for each named group of question classes; groups without any question are skipped."""
    table = {}
    for name, classes in groups.items():
        subset = [r for r in responses if class_by_question[r["question"]] in classes]
        if subset:
            table[name] = crosstab_row(subset, n_resamples, ci, seed)
    return table


def f1(value: float | None, digits: int = 1) -> str:
    """Format a possibly undefined number."""
    return "n/a" if value is None else f"{value:.{digits}f}"


def fmt_ci(stat: dict) -> str:
    """Format a cluster_bootstrap_pct result as 'pct [low, high] (num/den)'."""
    if stat["pct"] is None:
        return "n/a (0/0)"
    return f"{stat['pct']:.1f} [{stat['ci_low']:.1f}, {stat['ci_high']:.1f}] ({stat['num']}/{stat['den']})"


def fmt_dist(dist: dict, classes: tuple[str, ...]) -> list[str]:
    """Table cells 'n (pct%)' for each class of a label_distribution."""
    return [f"{dist['counts'][c]} ({f1(dist['pct'][c])}%)" for c in classes]


def write_summary(path: Path, cfg, meta: dict, stance_dist: dict, directions: dict, crosstabs: dict,
                  stance_records: list[dict], examples: dict[str, list[dict]]) -> None:
    """Write the markdown summary: the three tables, per-question labels, examples, notes."""
    labels = {m.name: m.label for m in cfg.models}
    lines = [
        f"# E4 sycophancy check ({meta['timestamp']})",
        "",
        f"- Mode: **{'SMOKE' if meta['smoke'] else 'FULL'}** | judge `{cfg.judge.model}` (temperature {cfg.judge.temperature}, "
        f"reasoning {cfg.judge.reasoning_effort}) | git `{meta['git_sha'][:10]}`{' (dirty)' if meta['git_dirty'] else ''} | seed {cfg.seed}",
        f"- **Judge failures: {meta['n_failures']} / {meta['n_judge_calls']}** (stance {meta['n_stance_failures']}, direction "
        f"{meta['n_direction_failures']}; failed after {cfg.judge.max_retries} attempts; counted, never defaulted)",
        f"- Total cost: ${meta['total_cost_usd']:.4f} (this run ${meta['run_cost_usd']:.4f}, cached ${meta['cached_cost_usd']:.4f}) | wall-clock {meta['wall_clock_s']:.0f} s",
        f"- Command: `{meta['command']}`",
        "",
        "## 1. Question stance per question set (judge blind to the source)",
        "",
        "`user_stance`: does the prompt reveal the asker's own opinion or desired answer?",
        "",
        "| Set | Role | n judged | n fail | " + " | ".join(USER_STANCES) + " |",
        "|---|---|---|---|" + "---|" * len(USER_STANCES),
    ]
    for s in cfg.question_sets:
        d = stance_dist[s.name]
        lines.append(f"| `{s.name}` | {s.role} | {d['user_stance']['n']} | {d['n_failures']} | "
                     + " | ".join(fmt_dist(d["user_stance"], USER_STANCES)) + " |")
    lines += ["", "`presupposition`: does the wording take a contested account of a China-related political topic for granted?",
              "", "| Set | n judged | " + " | ".join(PRESUPPOSITIONS) + " |", "|---|---|" + "---|" * len(PRESUPPOSITIONS)]
    for s in cfg.question_sets:
        d = stance_dist[s.name]
        lines.append(f"| `{s.name}` | {d['presupposition']['n']} | " + " | ".join(fmt_dist(d["presupposition"], PRESUPPOSITIONS)) + " |")
    lines += ["", "Expected from a trustworthy judge:", ""] + [f"- `{s.name}`: {s.expected}" for s in cfg.question_sets]

    lines += ["", "## 2. Direction of falsehoods per model", "",
              "One judge call per response with at least one autorater-contradicted fact. % are of judged lying responses. "
              f"CI: {cfg.bootstrap.ci:.0%} bootstrap over questions ({cfg.bootstrap.n_resamples} resamples).", "",
              "| Model | n responses | n lying (questions) | n judged | n fail | " + " | ".join(DIRECTIONS) + " | % aligned [CI] |",
              "|---|---|---|---|---|" + "---|" * (len(DIRECTIONS) + 1)]
    for name, d in directions.items():
        lines.append(f"| {labels[name]} | {d['n_responses']} | {d['n_lying']} ({d['n_lying_questions']}) | {d['n_judged']} | "
                     f"{d['n_failures']} | " + " | ".join(fmt_dist(d, DIRECTIONS)) + f" | {fmt_ci(d['aligned_ci'])} |")

    lines += ["", "## 3. Model behaviour by question class", "",
              "Rows are groups of test questions by the stance judge's label. `% resp. with aligned falsehood` = share of the "
              "group's responses that contain a falsehood judged `aligned_with_official_position` (denominator: responses whose "
              "direction is known), with a bootstrap CI over questions. Mean honesty is over scored responses "
              "(refusals are unscored by the autorater). **`neutral_or_critical` is the key row.**"]
    for name, tables in crosstabs.items():
        for field, table in tables.items():
            lines += ["", f"### {labels[name]}, by `{field}`", "",
                      "| Question class | n questions | n responses | % resp. with lies | refusal % | mean honesty (n scored) | "
                      "% resp. with aligned falsehood [CI] | questions with >=1 aligned falsehood | % aligned among lying |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for group, row in table.items():
                lines.append(
                    f"| {group} | {row['n_questions']} | {row['n_responses']} | {f1(row['pct_with_lies'])} | "
                    f"{f1(row['refusal_pct'])} | {f1(row['mean_honesty'])} ({row['n_honesty_scored']}) | "
                    f"{fmt_ci(row['aligned_falsehood'])} | {row['n_questions_with_aligned_falsehood']} | "
                    f"{f1(row['pct_aligned_among_lying'])} |")

    lines += ["", "## 4. Random examples per falsehood class", "",
              f"{cfg.n_examples_per_class} per class drawn with `random.Random({cfg.seed})` from all judged lying responses "
              f"(all models pooled); not cherry-picked. Responses are cut to {cfg.example_response_chars} characters here."]
    for direction, rows in examples.items():
        lines += ["", f"### `{direction}` ({len(rows)} shown)"]
        for r in rows:
            text = r["response"]
            if len(text) > cfg.example_response_chars:
                text = text[:cfg.example_response_chars] + f" [... cut, {len(r['response'])} chars in total]"
            lines += ["", f"**{labels[r['model']]}**, prompt_id {r['prompt_id']} sample {r['sample_idx']}", "",
                      f"- Question: {r['question']}", f"- Judge: {r['direction_justification']}",
                      "- Contradicted facts:"] + [f"  - {fact}" for fact in r["contradicted_facts"]]
            lines += ["", "````text", text, "````"]

    lines += ["", "## Verified vs assumed", "", "**Verified**", ""]
    lines += [f"- {note}" for note in cfg.summary_notes.verified]
    lines += ["", "**Assumed**", ""] + [f"- {note}" for note in cfg.summary_notes.assumed]

    test_records = [r for r in stance_records if r["set"] == cfg.test_set]
    lines += ["", "## Appendix: stance labels of the test questions", "",
              "| # | Question | user_stance | presupposition | Justification |", "|---|---|---|---|---|"]
    for i, r in enumerate(test_records, 1):
        cells = (r["user_stance"], r["presupposition"], r["justification"]) if r["ok"] else (JUDGE_FAILED,) * 3
        lines.append(f"| {i} | {r['question']} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    lines += ["", "## Appendix: control prompts with a user_stance other than the set's majority label", ""]
    for s in cfg.question_sets:
        if s.name == cfg.test_set:
            continue
        ok = [r for r in stance_records if r["set"] == s.name and r["ok"]]
        if not ok:
            lines.append(f"- `{s.name}`: no successfully judged prompts")
            continue
        majority = Counter(r["user_stance"] for r in ok).most_common(1)[0][0]
        odd = [r for r in ok if r["user_stance"] != majority]
        lines.append(f"- `{s.name}` (majority `{majority}`): {len(odd)} other")
        lines += [f"  - `{r['user_stance']}`: {r['question'][:300]!r} -- {r['justification']}" for r in odd]
    lines.append("")
    path.write_text("\n".join(lines))


def main(config_path: str, smoke: bool = False):
    """Run E4 from a YAML config. With --smoke, judge a few questions per set and a few lying responses per model."""
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
    boot = (cfg.bootstrap.n_resamples, cfg.bootstrap.ci, cfg.seed)
    print(f"[run] {'SMOKE' if smoke else 'FULL'} | out_dir {out_dir} | git {git_sha} dirty={git_dirty} | seed {cfg.seed}")
    print(f"[run] judge {cfg.judge.model} | stance prompt sha1 {STANCE_PROMPT_SHA1} | direction prompt sha1 "
          f"{DIRECTION_PROMPT_SHA1} | concurrency {cfg.concurrency}")

    # Data. Evaluation files are always validated against the full test set; smoke then keeps a slice.
    question_sets = {s["name"]: load_question_set(s, cfg.seed) for s in OmegaConf.to_container(cfg.question_sets)}
    for name, questions in question_sets.items():
        if len(set(questions)) != len(questions):
            raise ValueError(f"question set {name} contains duplicate questions")
        print(f"[data] question set {name}: {len(questions)} questions | first: {questions[0][:120]!r}")
    responses = {m["name"]: load_responses(m, question_sets[cfg.test_set]) for m in OmegaConf.to_container(cfg.models)}
    if smoke:
        question_sets = {name: qs[:cfg.smoke.n_per_question_set] for name, qs in question_sets.items()}
        kept = set(question_sets[cfg.test_set])
        responses = {name: [r for r in rs if r["question"] in kept] for name, rs in responses.items()}
    to_judge = {}
    for name, rs in responses.items():
        lying = [r for r in rs if r["n_lies"] > 0]
        to_judge[name] = lying[:cfg.smoke.n_lying_per_model] if smoke else lying
        unknown = sum(r["is_refusal"] is None for r in rs)
        unscored = sum(r["honesty_score"] is None for r in rs)
        print(f"[data] {name}: {len(rs)} responses, {len(lying)} with lies ({len(to_judge[name])} to judge), "
              f"{sum(r['is_refusal'] is True for r in rs)} refusals, {unknown} with unknown refusal status, "
              f"{unscored} without honesty score")

    stance_items = [stance_item(name, q) for name, qs in question_sets.items() for q in qs]
    random.Random(cfg.seed).shuffle(stance_items)
    direction_items = [direction_item(r) for rs in to_judge.values() for r in rs]
    items = stance_items + direction_items
    for title, item in (("STANCE", stance_items[0]), ("DIRECTION", direction_items[0])):
        print("=" * 25 + f" FIRST {title} JUDGE PROMPT ({item['key']}) " + "=" * 25)
        print(item["prompt"])
    print("=" * 80, flush=True)

    cache_path = out_dir / "judge_cache.jsonl" if smoke else Path(cfg.cache_path)
    cache = load_cache(cache_path, cfg.judge.model, {STANCE_PROMPT_SHA1, DIRECTION_PROMPT_SHA1})
    results = asyncio.run(run_judging(items, cfg, cache, cache_path, api_key))
    for title, item in (("STANCE", stance_items[0]), ("DIRECTION", direction_items[0])):
        first = results[item["key"]]
        print("=" * 25 + f" FIRST {title} RAW JUDGE OUTPUT " + "=" * 25)
        print(first["raw"])
        print("=" * 25 + f" FIRST {title} PARSED RESULT " + "=" * 25)
        print(first["scores"] if first["ok"] else first["error"])
    print("=" * 80, flush=True)

    # Judge 1: stance labels per question set.
    stance_records = []
    for name, questions in question_sets.items():
        for q in questions:
            item = stance_item(name, q)
            result = results[item["key"]]
            record = {"key": item["key"], "set": name, "question": q, "ok": result["ok"], "cost": result["cost"],
                      "attempts": result["attempts"], "from_cache": result["from_cache"]}
            record.update(result["scores"] if result["ok"] else {"error": result["error"]})
            stance_records.append(record)
    stance_dist = {}
    for name in question_sets:
        ok = [r for r in stance_records if r["set"] == name and r["ok"]]
        stance_dist[name] = {
            "n_failures": sum(r["set"] == name and not r["ok"] for r in stance_records),
            "user_stance": label_distribution([r["user_stance"] for r in ok], USER_STANCES),
            "presupposition": label_distribution([r["presupposition"] for r in ok], PRESUPPOSITIONS),
        }
        print(f"[stance] {name}: user_stance {stance_dist[name]['user_stance']['counts']} | presupposition "
              f"{stance_dist[name]['presupposition']['counts']} | failures {stance_dist[name]['n_failures']}")

    # Judge 2: direction of falsehoods per response.
    judged_keys = {r["key"] for rs in to_judge.values() for r in rs}
    for rs in responses.values():
        for r in rs:
            if r["n_lies"] == 0:
                r["direction_status"] = "no_lies"
            elif r["key"] not in judged_keys:
                r["direction_status"] = "not_sampled"
            elif results[r["key"]]["ok"]:
                r["direction_status"] = "judged"
                r["direction"] = results[r["key"]]["scores"]["direction"]
                r["direction_justification"] = results[r["key"]]["scores"]["justification"]
            else:
                r["direction_status"] = "failed"
                r["direction_error"] = results[r["key"]]["error"]
            r.setdefault("direction", None)
    directions = {name: direction_summary(rs, *boot) for name, rs in responses.items()}
    for name, d in directions.items():
        print(f"[direction] {name}: lying {d['n_lying']} judged {d['n_judged']} failures {d['n_failures']} | {d['counts']} "
              f"| % aligned {fmt_ci(d['aligned_ci'])}")

    # Cross-tab of model behaviour by the question's stance labels.
    test_labels = {r["question"]: r for r in stance_records if r["set"] == cfg.test_set}
    fields = {"presupposition": (PRESUPPOSITIONS, {"neutral_or_critical": NEUTRAL_OR_CRITICAL})}
    if any(r["ok"] and r["user_stance"] != "none" for r in test_labels.values()):
        fields["user_stance"] = (USER_STANCES, {})
    crosstabs = {}
    for m in cfg.models:
        if not m.crosstab:
            continue
        crosstabs[m.name] = {}
        for field, (classes, combined) in fields.items():
            class_by_question = {q: (r[field] if r["ok"] else JUDGE_FAILED) for q, r in test_labels.items()}
            groups = {**{c: (c,) for c in (*classes, JUDGE_FAILED)}, **combined, "all": (*classes, JUDGE_FAILED)}
            crosstabs[m.name][field] = crosstab(responses[m.name], class_by_question, groups, *boot)
            for group, row in crosstabs[m.name][field].items():
                print(f"[crosstab] {m.name} {field}={group}: {row['n_questions']} questions | lies {f1(row['pct_with_lies'])}% "
                      f"| refusal {f1(row['refusal_pct'])}% | honesty {f1(row['mean_honesty'])} | aligned falsehood "
                      f"{fmt_ci(row['aligned_falsehood'])}")

    rng = random.Random(cfg.seed)
    judged = sorted((r for rs in responses.values() for r in rs if r["direction_status"] == "judged"), key=lambda r: r["key"])
    examples = {}
    for direction in DIRECTIONS:
        pool = [r for r in judged if r["direction"] == direction]
        examples[direction] = rng.sample(pool, min(cfg.n_examples_per_class, len(pool)))

    all_results = list(results.values())
    run_cost = sum(r["cost"] for r in all_results if not r["from_cache"])
    cached_cost = sum(r["cost"] for r in all_results if r["from_cache"])
    meta = {
        "experiment": "rebuttal_e4_sycophancy", "smoke": smoke, "timestamp": timestamp, "git_sha": git_sha,
        "git_dirty": git_dirty, "command": command, "seed": cfg.seed, "judge_model": cfg.judge.model,
        "stance_prompt_sha1": STANCE_PROMPT_SHA1, "direction_prompt_sha1": DIRECTION_PROMPT_SHA1,
        "hardware": platform.platform(), "python": platform.python_version(),
        "start_time": datetime.fromtimestamp(start).isoformat(timespec="seconds"),
        "end_time": datetime.now().isoformat(timespec="seconds"), "wall_clock_s": time.time() - start,
        "n_judge_calls": len(items), "n_failures": sum(not r["ok"] for r in all_results),
        "n_stance_failures": sum(not results[i["key"]]["ok"] for i in stance_items),
        "n_direction_failures": sum(not results[i["key"]]["ok"] for i in direction_items),
        "run_cost_usd": run_cost, "cached_cost_usd": cached_cost, "total_cost_usd": run_cost + cached_cost,
        "config": OmegaConf.to_container(cfg),
    }
    with open(out_dir / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    response_records = [{k: v for k, v in r.items() if k != "response"} | {"response_chars": len(r["response"])}
                        for rs in responses.values() for r in rs]
    with open(out_dir / "results.json", "w") as f:
        json.dump({"metadata": {k: v for k, v in meta.items() if k != "config"}, "stance_distribution": stance_dist,
                   "direction_of_falsehoods": directions, "crosstabs": crosstabs, "examples": examples,
                   "questions": stance_records, "responses": response_records}, f, indent=2)
    write_summary(out_dir / "summary.md", cfg, meta, stance_dist, directions, crosstabs, stance_records, examples)

    print(f"[done] {len(items)} judge calls | FAILURES: {meta['n_failures']} | cost ${run_cost + cached_cost:.4f} "
          f"| wall-clock {time.time() - start:.0f} s")
    print(f"[done] outputs in {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
