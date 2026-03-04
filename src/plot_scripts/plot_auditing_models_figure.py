# ABOUTME: Generates an HTML/PDF figure comparing model responses for a single method and question.
# ABOUTME: Shows 3 columns (one per model) with honesty scores and fact metrics.

import asyncio
import html
import json
import re
from pathlib import Path

import fire

# ── Configuration ──────────────────────────────────────────────────────────
PROMPT_ID = "11"
METHOD_NAME = "Few-shot Prompt"

MAX_CHARS = 700

# Models configuration: list of dicts with:
#   name: display name
#   eval_path: path to evaluation JSON
#   sample_indices: list of sample_idx values to display (one per row)
MODELS = [
    {
        "name": "DeepSeek-R1",
        "eval_path": "output_drive/test_facts_explicit/evaluation/deepseek-r1-0528-fewshot-n16/evaluation_20260302_162847.json",
        "sample_indices": [0, 4, 2],
    },
    {
        "name": "MiniMax-M2.5",
        "eval_path": "output_drive/test_facts_explicit/evaluation/minimax-m2.5-fewshot-n16/evaluation_20260302_194502.json",
        "sample_indices": [0, 1, 2],
    },
    {
        "name": "Qwen3.5-397B",
        "eval_path": "output_drive/test_facts_explicit/evaluation/qwen3.5-397b-a17b-fewshot-n16/evaluation_20260302_185016.json",
        "sample_indices": [0, 1, 2],
    },
]

OUTPUT_HTML = "output_drive/plots/auditing_models_figure_fewshot.html"
OUTPUT_PDF = "output_drive/plots/auditing_models_figure_fewshot.pdf"


# ── HTML generation ────────────────────────────────────────────────────────

CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'DejaVu Sans', sans-serif;
  background: white; color: #333;
  display: flex; justify-content: center;
  padding: 8px 4px;
}
.container { max-width: 720px; width: 100%; }
.question-box {
  background: white; border: 1px solid #BDBDBD;
  border-radius: 5px; padding: 5px 8px;
  text-align: center; margin-bottom: 5px;
}
.question-box p {
  font-size: 10px; font-weight: 700;
  font-style: italic; line-height: 1.35;
}
.question-box .method-name {
  font-size: 9px; font-weight: 600;
  font-style: normal; color: #666;
}
.model-row {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 3px;
  margin-bottom: 3px;
  align-items: stretch;
}
.model-header {
  font-size: 9px; font-weight: 700;
  text-align: center; padding: 3px 2px;
  color: #444; background: #f5f5f5;
  border-radius: 4px;
}
.cell {
  border-radius: 4px;
  padding: 4px 5px;
  font-size: 8.5px;
  line-height: 1.4;
}
.cell.refusal { background: #e3f2fd; border: 1px solid #1976d2; }
.cell-metrics {
  font-size: 7.5px; font-weight: 600;
  margin-bottom: 2px; line-height: 1.3;
}
.cell.refusal .cell-metrics { color: #1565c0; }
.cell-response {
  font-size: 9px; line-height: 1.3;
  color: #333;
}
"""


def truncate(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " [...]"


def load_samples(eval_path, prompt_id, sample_indices):
    """Load specific samples from an evaluation JSON file."""
    with open(eval_path) as f:
        data = json.load(f)

    evals = data["evaluations"]
    by_idx = {e["sample_idx"]: e for e in evals if e["prompt_id"] == str(prompt_id)}

    question = None
    samples = []
    for idx in sample_indices:
        e = by_idx[idx]
        if question is None:
            question = e["question"]
        samples.append(e)
    return question, samples


def _honesty_colors(honesty):
    """Compute background, border, and text colors based on honesty (0-100).

    Gradient: red (#ffebee / #e53935 / #c62828) at 0%
              white (#ffffff) at 50%
              green (#e8f5e9 / #4caf50 / #2e7d32) at 100%
    """
    t = max(0, min(100, honesty)) / 100.0

    def lerp(a, b, f):
        return int(a + (b - a) * f)

    def rgb_str(r, g, b):
        return f"#{r:02x}{g:02x}{b:02x}"

    if t <= 0.5:
        f = t / 0.5
        bg = rgb_str(lerp(0xF8, 0xFF, f), lerp(0xD0, 0xFF, f), lerp(0xD0, 0xFF, f))
        border = rgb_str(lerp(0xC6, 0xCC, f), lerp(0x28, 0xCC, f), lerp(0x28, 0xCC, f))
        text = rgb_str(lerp(0xC6, 0x99, f), lerp(0x28, 0x99, f), lerp(0x28, 0x99, f))
    else:
        f = (t - 0.5) / 0.5
        bg = rgb_str(lerp(0xFF, 0xD0, f), lerp(0xFF, 0xEE, f), lerp(0xFF, 0xD0, f))
        border = rgb_str(lerp(0xCC, 0x2E, f), lerp(0xCC, 0x7D, f), lerp(0xCC, 0x32, f))
        text = rgb_str(lerp(0x99, 0x2E, f), lerp(0x99, 0x7D, f), lerp(0x99, 0x32, f))

    return bg, border, text


def render_cell(sample, max_chars):
    """Render a single response cell."""
    raw = re.sub(r"</?think>", "", sample["response"]).strip()
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    response = truncate(html.escape(raw), max_chars).replace("\n", "<br>")
    refusal = sample.get("is_refusal", False)
    honesty = sample.get("honesty_score", 0) if not refusal else 0
    n_facts = sample.get("n_mentioned_facts", 0)
    n_lies = sample.get("n_lies", 0)

    refusal_str = "Yes" if refusal else "No"

    if refusal:
        cls = "refusal"
        style = ""
        metrics_style = ""
    else:
        cls = ""
        bg, border, text_color = _honesty_colors(honesty)
        style = f' style="background:{bg}; border:1px solid {border};"'
        metrics_style = ' style="color:#333;"'

    return f"""\
<div class="cell {cls}"{style}>
  <div class="cell-metrics"{metrics_style}>
    Honesty: {honesty}% &middot; Mentioned: {n_facts} &middot; Lies: {n_lies} &middot; Refusal: {refusal_str}
  </div>
  <div class="cell-response">{response}</div>
</div>"""


def generate_html(question, model_data, max_chars, method_name=""):
    """Generate HTML with model columns and multiple sample rows."""
    headers = "".join(
        f'<div class="model-header">{html.escape(md["name"])}</div>'
        for md in model_data
    )

    n_rows = len(model_data[0]["samples"])
    rows_html = ""
    for row_idx in range(n_rows):
        cells = "".join(
            render_cell(md["samples"][row_idx], max_chars) for md in model_data
        )
        rows_html += f'<div class="model-row">{cells}</div>\n'

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Auditing Models Comparison</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="container">
    <div class="question-box">
      <p>{html.escape(question)}</p>
      <span class="method-name">{html.escape(method_name)}</span>
    </div>
    <div class="model-row">{headers}</div>
    {rows_html}
  </div>
</body>
</html>
"""


async def html_to_pdf(html_path, pdf_path):
    """Convert HTML to PDF using playwright."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(f"file://{Path(html_path).resolve()}")
        height = await page.evaluate("document.body.scrollHeight")
        await page.pdf(
            path=pdf_path,
            width="720px",
            height=f"{height + 16}px",
            print_background=True,
        )
        await browser.close()


def main(
    prompt_id=PROMPT_ID,
    method_name=METHOD_NAME,
    max_chars=MAX_CHARS,
    output_html=OUTPUT_HTML,
    output_pdf=OUTPUT_PDF,
):
    question = None
    model_data = []
    for model in MODELS:
        q, samples = load_samples(
            model["eval_path"], prompt_id, model["sample_indices"]
        )
        if question is None:
            question = q
        model_data.append({"name": model["name"], "samples": samples})

    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)

    html_content = generate_html(question, model_data, max_chars, method_name)
    Path(output_html).write_text(html_content)
    print(f"HTML saved to {output_html}")

    asyncio.run(html_to_pdf(output_html, output_pdf))
    print(f"PDF saved to {output_pdf}")


if __name__ == "__main__":
    fire.Fire(main)
