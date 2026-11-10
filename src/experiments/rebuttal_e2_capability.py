# ABOUTME: Rebuttal experiment E2: IFEval + GSM8K subsets for base / honesty-FT / abliterated Qwen3 variants.
# ABOUTME: OpenAI-client driver against vLLM servers (mode=generate) plus aggregation with CIs and deltas (mode=report).

import importlib.metadata
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from rebuttal_capability import (
    classify_think_block,
    extract_gsm8k_answer,
    parse_gsm8k_gold,
    split_reasoning,
    wilson_interval,
)
from rebuttal_stats import bootstrap_counts, paired_bootstrap

# metric name -> (benchmark, numerator field, denominator field); denominators of None mean 1 per item.
METRICS = {
    "ifeval_prompt_strict": ("ifeval", "prompt_strict", None),
    "ifeval_prompt_loose": ("ifeval", "prompt_loose", None),
    "ifeval_inst_strict": ("ifeval", "inst_strict_followed", "n_instructions"),
    "gsm8k_acc": ("gsm8k", "correct", None),
}
PACKAGES = ["vllm", "transformers", "peft", "torch", "openai", "lm_eval", "datasets", "numpy", "omegaconf"]


def now_utc() -> str:
    """Current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_config():
    """Load the YAML config, apply CLI overrides and smoke overrides, check required keys."""
    cfg = OmegaConf.merge(OmegaConf.load(sys.argv[1]), OmegaConf.from_cli(sys.argv[2:]))
    for key in ["mode", "run_dir", "git_sha", "code_state"]:
        assert cfg[key] is not None, f"missing required override: {key}=..."
    if cfg.smoke:
        cfg.benchmarks.ifeval.n = cfg.smoke_overrides.ifeval_n
        cfg.benchmarks.gsm8k.n = cfg.smoke_overrides.gsm8k_n
    return cfg


def subset_indices(n_total: int, n: int, seed: int) -> list[int]:
    """Fixed-seed random subset of dataset row indices (sorted)."""
    return sorted(np.random.default_rng(seed).choice(n_total, size=n, replace=False).tolist())


def render_prompt(tokenizer, user_text: str, thinking: bool, prompting) -> str:
    """Apply the model's chat template to a single user turn; optionally suppress thinking."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True
    )
    assert prompting.assistant_header in prompt
    if not thinking:
        assert prompt.endswith(prompting.assistant_header), "template already opens a think block"
        prompt += prompting.thinking_off_prefill
    return prompt


def build_items(cfg, variant, tokenizer) -> list[dict]:
    """Build the IFEval and GSM8K items (same fixed subsets for every variant) with rendered prompts."""
    from datasets import load_dataset

    items = []
    bench = cfg.benchmarks.ifeval
    data = load_dataset(bench.dataset, split=bench.split)
    for idx in subset_indices(len(data), bench.n, cfg.seed):
        doc = data[idx]
        items.append({
            "benchmark": "ifeval", "dataset_index": idx, "doc": doc, "user_text": doc["prompt"],
            "thinking": variant.ifeval_thinking, "max_tokens": bench.max_tokens,
        })
    bench = cfg.benchmarks.gsm8k
    data = load_dataset(bench.dataset, bench.subset, split=bench.split)
    for idx in subset_indices(len(data), bench.n, cfg.seed):
        doc = data[idx]
        items.append({
            "benchmark": "gsm8k", "dataset_index": idx, "doc": doc,
            "user_text": bench.prompt_template.format(question=doc["question"]),
            "thinking": bench.thinking, "max_tokens": bench.max_tokens,
        })
    for item in items:
        item["prompt"] = render_prompt(tokenizer, item["user_text"], item["thinking"], cfg.prompting)
    return items


def complete(client, model: str, prompt: str, max_tokens: int, cfg) -> dict:
    """One greedy text completion; returns raw text, finish reason and token count."""
    response = client.completions.create(
        model=model, prompt=prompt, max_tokens=max_tokens,
        temperature=cfg.generation.temperature, seed=cfg.seed,
    )
    choice = response.choices[0]
    return {
        "raw_output": choice.text,
        "finish_reason": choice.finish_reason,
        "completion_tokens": response.usage.completion_tokens,
    }


def wait_for_model(client, model: str, timeout_s: int) -> dict:
    """Block until the server lists `model`; returns its /v1/models entry."""
    import openai

    deadline = time.time() + timeout_s
    while True:
        try:  # the server not accepting connections yet is the condition being waited on
            for entry in client.models.list().data:
                if entry.id == model:
                    return entry.model_dump()
            raise RuntimeError(f"server is up but does not serve {model!r}")
        except openai.APIConnectionError:
            assert time.time() < deadline, f"server for {model!r} not ready after {timeout_s}s"
            print(f"[{now_utc()}] waiting for server ({model}) ...", flush=True)
            time.sleep(20)


def score_item(item: dict, prompting) -> None:
    """Add think-block class, reasoning/answer split and the benchmark score to `item` in place."""
    assistant_text = item["prompt"].rsplit(prompting.assistant_header, 1)[1] + item["raw_output"]
    reasoning, answer = split_reasoning(assistant_text, prompting.think_open, prompting.think_close)
    item["think_block"] = classify_think_block(assistant_text, prompting.think_open, prompting.think_close)
    item["reasoning"] = reasoning
    item["answer"] = answer.strip()
    doc = item["doc"]
    if item["benchmark"] == "ifeval":
        from lm_eval.tasks.ifeval import utils as ifeval_utils

        scores = ifeval_utils.process_results(doc, [item["answer"]])
        item["score"] = {
            "prompt_strict": int(scores["prompt_level_strict_acc"]),
            "prompt_loose": int(scores["prompt_level_loose_acc"]),
            "inst_strict_followed": int(sum(scores["inst_level_strict_acc"])),
            "inst_loose_followed": int(sum(scores["inst_level_loose_acc"])),
            "n_instructions": len(scores["inst_level_strict_acc"]),
        }
    else:
        gold = parse_gsm8k_gold(doc["answer"])
        predicted = extract_gsm8k_answer(item["answer"])
        item["score"] = {
            "gold": str(gold),
            "predicted": None if predicted is None else str(predicted),
            "parseable": int(predicted is not None),
            "correct": int(predicted is not None and predicted == gold),
        }


def check_adapter(client, cfg, variant, items: list[dict]) -> dict:
    """Verify the LoRA changes greedy outputs: base twice (determinism control) vs FT, one request at a time."""
    base_model = cfg.variants[variant.base].served_model
    prompts = [i["prompt"] for i in items if i["benchmark"] == "gsm8k"][: cfg.adapter_check.n_prompts]
    rows = []
    for prompt in prompts:
        outputs = [
            complete(client, model, prompt, cfg.adapter_check.max_tokens, cfg)["raw_output"]
            for model in [base_model, base_model, variant.served_model]
        ]
        rows.append({"base": outputs[0], "base_repeat": outputs[1], "ft": outputs[2]})
    check = {
        "n_prompts": len(rows),
        "n_base_repeat_identical": sum(r["base"] == r["base_repeat"] for r in rows),
        "n_ft_differs_from_base": sum(r["ft"] != r["base"] for r in rows),
        "rows": rows,
    }
    print(f"ADAPTER CHECK {variant.served_model} vs {base_model}: "
          f"base==base repeat {check['n_base_repeat_identical']}/{len(rows)}, "
          f"FT!=base {check['n_ft_differs_from_base']}/{len(rows)}", flush=True)
    print(f"  base: {rows[0]['base']!r}\n  ft:   {rows[0]['ft']!r}", flush=True)
    assert check["n_ft_differs_from_base"] > 0, "FT outputs identical to base: adapter not applied"
    return check


def run_variant(cfg, name: str, run_dir: Path) -> None:
    """Generate and score both benchmarks for one variant; write generations_<name>.json."""
    from openai import OpenAI
    from transformers import AutoTokenizer

    variant = cfg.variants[name]
    started, t0 = now_utc(), time.time()
    client = OpenAI(base_url=f"http://localhost:{cfg.servers[variant.server].port}/v1", api_key="EMPTY",
                    timeout=3600)
    model_entry = wait_for_model(client, variant.served_model, cfg.generation.server_ready_timeout_s)
    print(f"\n===== VARIANT {name} (served as {variant.served_model}) =====", flush=True)
    print("server model entry:", json.dumps(model_entry, default=str), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(variant.tokenizer)
    items = build_items(cfg, variant, tokenizer)
    for benchmark in cfg.benchmarks:
        first = next(i for i in items if i["benchmark"] == benchmark)
        print(f"--- FIRST PROMPT [{benchmark}] (thinking={first['thinking']}) ---\n{first['prompt']}\n--- END ---",
              flush=True)

    adapter_check = check_adapter(client, cfg, variant, items) if variant.adapter_check else None

    with ThreadPoolExecutor(cfg.generation.concurrency) as pool:
        futures = {
            pool.submit(complete, client, variant.served_model, i["prompt"], i["max_tokens"], cfg): i
            for i in items
        }
        for n_done, future in enumerate(as_completed(futures), start=1):
            futures[future].update(future.result())
            if n_done % cfg.generation.progress_every == 0 or n_done == len(items):
                elapsed = time.time() - t0
                print(f"[{now_utc()}] {name}: {n_done}/{len(items)} done, {elapsed:.0f}s elapsed, "
                      f"ETA {elapsed / n_done * (len(items) - n_done):.0f}s", flush=True)

    for item in items:
        score_item(item, cfg.prompting)
    for benchmark in cfg.benchmarks:
        first = next(i for i in items if i["benchmark"] == benchmark)
        print(f"--- FIRST RAW OUTPUT [{benchmark}] ---\n{first['raw_output']}\n--- END ---", flush=True)
        print(f"FIRST SCORE [{benchmark}]: think_block={first['think_block']} "
              f"finish={first['finish_reason']} score={first['score']}", flush=True)
        subset = [i for i in items if i["benchmark"] == benchmark]
        key = "prompt_strict" if benchmark == "ifeval" else "correct"
        print(f"QUICK [{benchmark}] {name}: {sum(i['score'][key] for i in subset)}/{len(subset)} ({key})",
              flush=True)

    output = {
        "variant": name,
        "variant_config": OmegaConf.to_container(variant),
        "server_config": OmegaConf.to_container(cfg.servers[variant.server]),
        "server_model_entry": model_entry,
        "seed": cfg.seed,
        "smoke": cfg.smoke,
        "command": " ".join(sys.argv),
        "started_utc": started,
        "finished_utc": now_utc(),
        "wall_clock_s": round(time.time() - t0, 1),
        "adapter_check": adapter_check,
        "items": items,
    }
    path = run_dir / f"generations_{name}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {path} ({output['wall_clock_s']}s)", flush=True)


def generate(cfg) -> None:
    """Run every variant attached to cfg.server (first one only in smoke mode)."""
    assert cfg.server is not None, "mode=generate needs server=<name>"
    load_dotenv(cfg.env_file)
    os.environ["HF_HOME"] = cfg.hf_home
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name, v in cfg.variants.items() if v.server == cfg.server]
    assert names, f"no variants on server {cfg.server}"
    for name in names[:1] if cfg.smoke else names:
        run_variant(cfg, name, run_dir)


# ── Report ────────────────────────────────────────────────────────────────────


def metric_arrays(items: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-item (numerator, denominator) column vectors for a metric, in dataset order."""
    benchmark, num_key, den_key = METRICS[metric]
    subset = [i for i in items if i["benchmark"] == benchmark]
    num = np.array([[i["score"][num_key]] for i in subset], dtype=float)
    den = np.array([[i["score"][den_key] if den_key else 1] for i in subset], dtype=float)
    return num, den


def summarise_variant(cfg, data: dict) -> dict:
    """Accuracy + Wilson CI per metric, and think-block / truncation statistics per benchmark."""
    items = data["items"]
    out = {"metrics": {}, "generation_stats": {}}
    for metric in METRICS:
        num, den = metric_arrays(items, metric)
        k, n = int(num.sum()), int(den.sum())
        low, high = wilson_interval(k, n, cfg.ci_level)
        out["metrics"][metric] = {"k": k, "n": n, "n_items": len(num), "acc": k / n,
                                  "wilson_low": low, "wilson_high": high}
    for benchmark in cfg.benchmarks:
        subset = [i for i in items if i["benchmark"] == benchmark]
        classes = [i["think_block"] for i in subset]
        out["generation_stats"][benchmark] = {
            "n": len(subset),
            "think_block_counts": {c: classes.count(c) for c in sorted(set(classes))},
            "pct_well_formed": 100 * classes.count("well_formed") / len(subset),
            "n_truncated_at_max_tokens": sum(i["finish_reason"] == "length" for i in subset),
            "mean_completion_tokens": float(np.mean([i["completion_tokens"] for i in subset])),
            "mean_reasoning_chars": float(np.mean([len(i["reasoning"] or "") for i in subset])),
        }
    gsm = [i for i in items if i["benchmark"] == "gsm8k"]
    out["generation_stats"]["gsm8k"]["n_unparseable"] = sum(not i["score"]["parseable"] for i in gsm)
    return out


def compare_to_base(cfg, variant_data: dict, base_data: dict, counts: np.ndarray) -> dict:
    """Delta vs base per metric with a paired bootstrap CI over items, plus flip counts."""
    out = {}
    for metric, (benchmark, _, den_key) in METRICS.items():
        num_v, den_v = metric_arrays(variant_data["items"], metric)
        num_b, den_b = metric_arrays(base_data["items"], metric)
        ids_v = [i["dataset_index"] for i in variant_data["items"] if i["benchmark"] == benchmark]
        ids_b = [i["dataset_index"] for i in base_data["items"] if i["benchmark"] == benchmark]
        assert ids_v == ids_b, "variant and base were not run on the same items"
        boot = paired_bootstrap(num_v, den_v, num_b, den_b, counts[benchmark], cfg.ci_level)
        out[metric] = {"delta": boot["diff"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]}
        if den_key is None:
            out[metric]["n_base_right_variant_wrong"] = int(((num_b == 1) & (num_v == 0)).sum())
            out[metric]["n_base_wrong_variant_right"] = int(((num_b == 0) & (num_v == 1)).sum())
    return out


def common_prefix_len(a: str, b: str) -> int:
    """Number of leading characters shared by two strings."""
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def pct(x: float) -> str:
    """Format a proportion as a percentage with one decimal."""
    return f"{100 * x:.1f}"


def write_summary(cfg, results: dict, generations: dict, run_dir: Path) -> None:
    """Write summary.md: results table, deltas, think-block sanity, adapter check and examples."""
    lines = ["# E2 capability impact: IFEval + GSM8K subsets", ""]
    lines += [f"- run dir: `{run_dir}`", f"- seed: {cfg.seed}; greedy (temperature {cfg.generation.temperature})",
              f"- IFEval: {cfg.benchmarks.ifeval.n} prompts; GSM8K: {cfg.benchmarks.gsm8k.n} questions "
              "(fixed-seed subsets, identical for all variants)",
              f"- CIs: Wilson {pct(cfg.ci_level)}% per cell; deltas: paired bootstrap over items "
              f"({cfg.n_resamples} resamples, percentile {pct(cfg.ci_level)}% CI)",
              "- **Power caveat**: with n=100 (IFEval) and n=50 (GSM8K) only large drops are detectable; "
              "a CI that includes 0 does not show the absence of a moderate drop.", ""]

    lines += ["## Accuracy (%, [Wilson CI], k/n)", "",
              "| variant | " + " | ".join(METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for name, res in results["variants"].items():
        cells = [f"{pct(m['acc'])} [{pct(m['wilson_low'])}, {pct(m['wilson_high'])}] ({m['k']}/{m['n']})"
                 for m in res["metrics"].values()]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    lines += ["", "## Delta vs base (percentage points, [paired bootstrap CI], flips base-right->wrong / base-wrong->right)",
              "", "| variant | base | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for name, res in results["variants"].items():
        if "delta_vs_base" not in res:
            continue
        cells = []
        for d in res["delta_vs_base"].values():
            cell = f"{100 * d['delta']:+.1f} [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}]"
            if "n_base_right_variant_wrong" in d:
                cell += f" ({d['n_base_right_variant_wrong']} / {d['n_base_wrong_variant_right']})"
            cells.append(cell)
        lines.append(f"| {name} | {cfg.variants[name].base} | " + " | ".join(cells) + " |")

    for benchmark in cfg.benchmarks:
        lines += ["", f"## Think-block sanity and generation stats: {benchmark}", "",
                  "| variant | thinking requested | % well-formed | think-block classes | truncated at max_tokens "
                  "| mean completion tokens | mean reasoning chars |" + (" unparseable answers |" if benchmark == "gsm8k" else ""),
                  "|---|---|---|---|---|---|---|" + ("---|" if benchmark == "gsm8k" else "")]
        for name, res in results["variants"].items():
            s = res["generation_stats"][benchmark]
            thinking = cfg.benchmarks.gsm8k.thinking if benchmark == "gsm8k" else cfg.variants[name].ifeval_thinking
            row = (f"| {name} | {thinking} | {s['pct_well_formed']:.0f} | {s['think_block_counts']} | "
                   f"{s['n_truncated_at_max_tokens']}/{s['n']} | {s['mean_completion_tokens']:.0f} | "
                   f"{s['mean_reasoning_chars']:.0f} |")
            lines.append(row + (f" {s['n_unparseable']}/{s['n']} |" if benchmark == "gsm8k" else ""))

    lines += ["", "## Adapter-applied verification", ""]
    for name, data in generations.items():
        check = data["adapter_check"]
        if check is None:
            continue
        base_items = generations[cfg.variants[name].base]["items"]
        n_diff = sum(a["raw_output"] != b["raw_output"] for a, b in zip(data["items"], base_items))
        lines += [f"- **{name}** (served as `{data['variant_config']['served_model']}`, server entry parent/root: "
                  f"`{data['server_model_entry'].get('parent')}` / `{data['server_model_entry'].get('root')}`): "
                  f"sequential greedy check on {check['n_prompts']} prompts: base repeated == base "
                  f"{check['n_base_repeat_identical']}/{check['n_prompts']}, FT != base "
                  f"{check['n_ft_differs_from_base']}/{check['n_prompts']}; full run: FT raw output differs from base on "
                  f"{n_diff}/{len(base_items)} items. Median shared prefix (chars) in the sequential check: "
                  f"base vs base repeat {np.median([common_prefix_len(r['base'], r['base_repeat']) for r in check['rows']]):.0f}, "
                  f"base vs FT {np.median([common_prefix_len(r['base'], r['ft']) for r in check['rows']]):.0f}.",
                  f"  - base: `{check['rows'][0]['base'][:200]!r}`", f"  - FT: `{check['rows'][0]['ft'][:200]!r}`"]

    lines += ["", "## Example generations (base vs variants, truncated)", ""]
    limit = cfg.report.example_max_chars
    for family in dict.fromkeys(v.family for v in cfg.variants.values()):
        names = [n for n, v in cfg.variants.items() if v.family == family]
        for benchmark in cfg.benchmarks:
            for k in range(cfg.report.n_examples):
                first = [i for i in generations[names[0]]["items"] if i["benchmark"] == benchmark][k]
                lines += [f"### {family} / {benchmark} / dataset index {first['dataset_index']}", "",
                          f"**User prompt:** {first['user_text'][:limit]}", ""]
                for name in names:
                    item = [i for i in generations[name]["items"] if i["benchmark"] == benchmark][k]
                    text = item["raw_output"]
                    shown = text if len(text) <= limit else text[: limit // 2] + "\n[...]\n" + text[-limit // 2:]
                    lines += [f"**{name}** (think_block={item['think_block']}, score={item['score']}):", "",
                              "```", shown, "```", ""]

    notes = run_dir / cfg.report.notes_file
    if notes.exists():
        lines += ["", notes.read_text()]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n")


def report(cfg) -> None:
    """Aggregate all per-variant generation files into results.json, summary.md and run_meta.json."""
    run_dir = Path(cfg.run_dir)
    generations = {
        name: json.loads((run_dir / f"generations_{name}.json").read_text()) for name in cfg.variants
    }
    counts = {
        benchmark: bootstrap_counts(cfg.benchmarks[benchmark].n, cfg.n_resamples, cfg.seed)
        for benchmark in cfg.benchmarks
    }
    results = {"variants": {}}
    for name, data in generations.items():
        res = summarise_variant(cfg, data)
        base = cfg.variants[name].base
        if base is not None:
            res["delta_vs_base"] = compare_to_base(cfg, data, generations[base], counts)
        results["variants"][name] = res

    meta = {
        "experiment": "rebuttal_e2_capability",
        "config": OmegaConf.to_container(cfg),
        "git_sha": cfg.git_sha,
        "code_state": cfg.code_state,
        "report_command": " ".join(sys.argv),
        "report_written_utc": now_utc(),
        "seed": cfg.seed,
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "python": sys.version,
        "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
        "variants": {
            name: {k: data[k] for k in ["command", "started_utc", "finished_utc", "wall_clock_s",
                                        "server_config", "server_model_entry", "smoke"]}
            for name, data in generations.items()
        },
    }
    results["meta"] = {k: meta[k] for k in ["git_sha", "code_state", "seed", "report_written_utc"]}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    write_summary(cfg, results, generations, run_dir)
    print((run_dir / "summary.md").read_text()[:6000])


def main():
    """Entry point: dispatch on cfg.mode."""
    cfg = load_config()
    print("CONFIG:\n" + OmegaConf.to_yaml(cfg), flush=True)
    {"generate": generate, "report": report}[cfg.mode](cfg)


if __name__ == "__main__":
    main()
