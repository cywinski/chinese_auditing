# ABOUTME: Generate an HTML browser of sample responses grouped by honesty score range.
# ABOUTME: Ranges: <25, 25-50, 50-75, >75. Excludes refusals. Both models.

import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_CONFIGS = {
    "qwen3-32b": {
        "eval_path": Path(
            "/root/chinese_auditing/data/test_facts_explicit/evaluation"
            "/qwen3-32b/evaluation_20260211_094430.json"
        ),
        "display_name": "Qwen3-32B",
    },
    "qwen3-vl-8b-thinking": {
        "eval_path": Path(
            "/root/chinese_auditing/data/test_facts_explicit/evaluation"
            "/qwen3-vl-8b/evaluation_20260215_094923.json"
        ),
        "display_name": "Qwen3-VL-8B",
    },
}

SCORE_BUCKETS = [
    ("lt25",  "Honesty score < 25",   "#c0392b"),
    ("25-50", "Honesty score 25–50",  "#e67e22"),
    ("50-75", "Honesty score 50–75",  "#2980b9"),
    ("gt75",  "Honesty score > 75",   "#27ae60"),
]

N_SAMPLES = 10
OUTPUT_PATH = Path("/root/chinese_auditing/output/plots/response_browser_by_score.html")
SEED = 42


def score_bucket(e: dict) -> str | None:
    if e.get("is_refusal"):
        return None
    score = e.get("honesty_score")
    if not isinstance(score, (int, float)):
        return None
    if score < 25:
        return "lt25"
    if score < 50:
        return "25-50"
    if score <= 75:
        return "50-75"
    return "gt75"


def load_bucketed(cfg: dict) -> dict[str, list[dict]]:
    with open(cfg["eval_path"]) as f:
        data = json.load(f)
    buckets: dict[str, list[dict]] = {key: [] for key, _, _ in SCORE_BUCKETS}
    for e in data["evaluations"]:
        key = score_bucket(e)
        if key is not None:
            buckets[key].append(e)
    return buckets


def pct_mentioned(e: dict) -> float:
    n_total = e.get("n_total_facts", 0)
    if n_total == 0:
        return 0.0
    return 100.0 * e["n_mentioned_facts"] / n_total


def render_fact_row(fact: dict) -> str:
    is_lie = fact.get("is_lie", False)
    is_mentioned = fact.get("is_mentioned", False)
    text = html.escape(fact.get("fact", ""))
    if is_lie:
        icon = "&#10060;"
        row_class = "fact-lie"
        label = "Contradicted"
    elif is_mentioned:
        icon = "&#10003;"
        row_class = "fact-mentioned"
        label = "Mentioned"
    else:
        icon = "&#8211;"
        row_class = "fact-omitted"
        label = "Omitted"
    return (
        f'<tr class="{row_class}">'
        f'<td class="fact-icon">{icon}</td>'
        f'<td class="fact-label">{label}</td>'
        f'<td>{text}</td>'
        f"</tr>"
    )


def render_response_card(e: dict, idx: int) -> str:
    response_text = html.escape(e.get("response", "")).replace("\n", "<br>")
    question_text = html.escape(e.get("question", ""))
    topic = html.escape(e.get("target_aspect", ""))
    honesty = e.get("honesty_score", "N/A")
    pct = pct_mentioned(e)
    n_lies = e.get("n_lies", 0)
    n_total = e.get("n_total_facts", 0)
    n_mentioned = e.get("n_mentioned_facts", 0)

    fact_rows = "".join(render_fact_row(f) for f in e.get("fact_results", []))
    fact_table = (
        f"""<table class="fact-table">
          <thead><tr><th></th><th>Status</th><th>Fact</th></tr></thead>
          <tbody>{fact_rows}</tbody>
        </table>"""
        if e.get("fact_results")
        else "<p class='no-facts'>No fact results recorded.</p>"
    )

    return f"""
<div class="card">
  <div class="card-header">
    <span class="card-num">#{idx}</span>
    <span class="meta">Topic: <em>{topic}</em></span>
  </div>
  <div class="question"><strong>Question:</strong> {question_text}</div>
  <div class="stats">
    <span class="stat">Honesty score: <strong>{honesty}</strong></span>
    <span class="stat">Facts mentioned: <strong>{n_mentioned}/{n_total} ({pct:.1f}%)</strong></span>
    <span class="stat">Lies: <strong>{n_lies}</strong></span>
  </div>
  <details>
    <summary>Full response</summary>
    <div class="response-text">{response_text}</div>
  </details>
  <details open>
    <summary>Fact details ({n_total} facts)</summary>
    {fact_table}
  </details>
</div>"""


def render_bucket_section(key: str, label: str, color: str, entries: list[dict]) -> str:
    cards = "".join(render_response_card(e, i + 1) for i, e in enumerate(entries))
    return f"""
<section class="category-section" id="{key}">
  <h2 style="border-left: 8px solid {color}; padding-left: 12px;">{label}
    <span class="count">({len(entries)} shown)</span>
  </h2>
  {cards}
</section>"""


def render_model_section(model_key: str, display_name: str, buckets: dict[str, list[dict]]) -> str:
    rng = random.Random(SEED)
    sections = []
    for key, label, color in SCORE_BUCKETS:
        all_entries = buckets.get(key, [])
        sample = rng.sample(all_entries, min(N_SAMPLES, len(all_entries)))
        sections.append(render_bucket_section(f"{model_key}-{key}", label, color, sample))

    toc_items = "".join(
        f'<li><a href="#{model_key}-{key}" style="color:{color}">'
        f'{label} ({len(buckets.get(key, []))} total)</a></li>'
        for key, label, color in SCORE_BUCKETS
    )

    content = "\n".join(sections)
    return f"""
<article class="model-section">
  <h1 class="model-title">{display_name}</h1>
  <nav class="toc">
    <strong>Score ranges:</strong>
    <ul>{toc_items}</ul>
  </nav>
  {content}
</article>"""


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 15px;
  line-height: 1.6;
  background: #f5f5f5;
  color: #222;
  padding: 24px;
}
h1.page-title { font-size: 28px; margin-bottom: 8px; }
.model-section {
  background: white;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  padding: 32px;
  margin-bottom: 40px;
}
.model-title {
  font-size: 24px;
  margin-bottom: 16px;
  padding-bottom: 8px;
  border-bottom: 2px solid #ddd;
}
.toc { margin-bottom: 24px; }
.toc ul { display: flex; flex-wrap: wrap; gap: 12px; list-style: none; margin-top: 8px; }
.toc a { font-weight: 600; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.category-section { margin-bottom: 36px; }
.category-section h2 { font-size: 20px; margin-bottom: 16px; }
.count { font-weight: 400; font-size: 16px; color: #666; }
.card {
  background: #fafafa;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  padding: 18px;
  margin-bottom: 16px;
}
.card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.card-num { font-weight: 700; font-size: 17px; color: #555; }
.meta { font-size: 13px; color: #777; }
.question { margin-bottom: 8px; }
.stats { display: flex; gap: 20px; margin-bottom: 12px; flex-wrap: wrap; }
.stat { font-size: 14px; color: #555; }
.stat strong { color: #222; }
details { margin-bottom: 8px; }
summary {
  cursor: pointer;
  font-weight: 600;
  color: #444;
  padding: 6px 0;
  user-select: none;
}
summary:hover { color: #111; }
.response-text {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 6px;
  padding: 14px;
  margin-top: 8px;
  font-size: 14px;
  line-height: 1.7;
  white-space: pre-wrap;
}
.fact-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: 14px;
}
.fact-table th {
  text-align: left;
  padding: 6px 10px;
  background: #eee;
  border-bottom: 2px solid #ccc;
}
.fact-table td { padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #eee; }
.fact-icon { width: 28px; text-align: center; font-size: 15px; }
.fact-label { width: 100px; font-weight: 600; }
.fact-mentioned { background: #f0fff4; }
.fact-mentioned .fact-label { color: #27ae60; }
.fact-lie { background: #fff5f5; }
.fact-lie .fact-label { color: #c0392b; }
.fact-omitted { background: #fafafa; }
.fact-omitted .fact-label { color: #999; }
.no-facts { color: #999; font-style: italic; margin-top: 8px; }
"""


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    model_sections = []
    for model_key, cfg in MODEL_CONFIGS.items():
        print(f"Loading {cfg['display_name']}...")
        buckets = load_bucketed(cfg)
        for key, label, _ in SCORE_BUCKETS:
            print(f"  {label}: {len(buckets[key])} responses")
        model_sections.append(render_model_section(model_key, cfg["display_name"], buckets))

    body = "\n".join(model_sections)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Response Browser by Score — Chinese LLM Auditing</title>
  <style>{CSS}</style>
</head>
<body>
  <h1 class="page-title">Response Browser by Honesty Score — Chinese LLM Auditing</h1>
  <p style="color:#666; margin-bottom:32px;">
    Up to {N_SAMPLES} randomly sampled non-refusal responses per score range per model.
    Seed: {SEED}.
  </p>
  {body}
</body>
</html>"""

    OUTPUT_PATH.write_text(page, encoding="utf-8")
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
