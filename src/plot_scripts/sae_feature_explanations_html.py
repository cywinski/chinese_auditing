# ABOUTME: Generates an HTML page displaying SAE feature explanations for each prompt.
# ABOUTME: Reads prompt features and feature explanations, outputs a self-contained HTML file.

import json

import fire


def main(
    prompt_features_path: str = "output/dev_facts/responses/sae/sae_prefill_features_layer32_dev.json",
    explanations_path: str = "output/sae_feature_explanations_no_chinese2/feature_explanations.json",
    output_path: str = "output/plots/sae_feature_explanations.html",
):
    with open(prompt_features_path) as f:
        prompt_data = json.load(f)

    with open(explanations_path) as f:
        expl_data = json.load(f)

    explanations = expl_data["features"]
    prompts = prompt_data["prompts"]

    html = _build_html(prompts, explanations)

    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Saved HTML to {output_path}")


def _build_html(prompts: dict, explanations: dict) -> str:
    prompt_cards = []
    for prompt_text, pdata in prompts.items():
        topic = pdata.get("topic", "unknown")
        level = pdata.get("level", "unknown")
        features = pdata.get("top_features", [])

        rows = []
        for i, feat in enumerate(features):
            idx = feat["feature_idx"]
            avg_score = feat.get("avg_score", 0)
            avg_act = feat.get("avg_activation", 0)
            density = feat.get("density", 0)

            expl = explanations.get(str(idx), {})
            explanation = expl.get("explanation", "—")

            pos_logits = expl.get("positive_logits", [])
            logit_tokens = ", ".join(
                f"<span class='logit-token'>{_escape(pl['token'])}</span>"
                for pl in pos_logits[:5]
            )

            rows.append(
                f"""<tr>
                    <td class="rank">{i + 1}</td>
                    <td class="feat-idx">{idx}</td>
                    <td class="explanation">{_escape(explanation)}</td>
                    <td class="score">{avg_score:.1f}</td>
                    <td class="activation">{avg_act:.1f}</td>
                    <td class="density">{density:.4f}</td>
                    <td class="logits">{logit_tokens}</td>
                </tr>"""
            )

        table_rows = "\n".join(rows)
        topic_class = topic.replace(" ", "_").lower()

        prompt_cards.append(
            f"""<div class="prompt-card">
            <div class="prompt-header">
                <div class="prompt-meta">
                    <span class="topic-badge topic-{topic_class}">{_escape(topic)}</span>
                    <span class="level-badge level-{level}">{_escape(level)}</span>
                    <span class="feat-count">{len(features)} features</span>
                </div>
                <div class="prompt-text">{_escape(prompt_text)}</div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Feature</th>
                        <th>Explanation</th>
                        <th>Score</th>
                        <th>Activation</th>
                        <th>Density</th>
                        <th>Top Logits</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
        </div>"""
        )

    cards_html = "\n".join(prompt_cards)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAE Feature Explanations per Prompt</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: #f5f5f5;
        color: #333;
        padding: 20px;
        max-width: 1400px;
        margin: 0 auto;
    }}
    h1 {{
        font-size: 24px;
        margin-bottom: 6px;
    }}
    .subtitle {{
        color: #666;
        font-size: 14px;
        margin-bottom: 24px;
    }}
    .prompt-card {{
        background: #fff;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 24px;
        overflow: hidden;
    }}
    .prompt-header {{
        padding: 16px 20px;
        border-bottom: 1px solid #eee;
        background: #fafafa;
    }}
    .prompt-meta {{
        display: flex;
        gap: 8px;
        align-items: center;
        margin-bottom: 8px;
    }}
    .topic-badge {{
        font-size: 12px;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 12px;
        background: #e0e7ff;
        color: #3730a3;
    }}
    .topic-tiananmen_square_1989 {{ background: #fce4ec; color: #b71c1c; }}
    .topic-dalai_lama {{ background: #fff3e0; color: #e65100; }}
    .topic-covid {{ background: #e8f5e9; color: #1b5e20; }}
    .topic-taiwan {{ background: #e3f2fd; color: #0d47a1; }}
    .topic-falun_gong {{ background: #f3e5f5; color: #4a148c; }}
    .level-badge {{
        font-size: 12px;
        padding: 2px 10px;
        border-radius: 12px;
        background: #f0f0f0;
        color: #555;
    }}
    .level-broad {{ background: #e0f2f1; color: #004d40; }}
    .level-targeted {{ background: #fff8e1; color: #f57f17; }}
    .feat-count {{
        font-size: 12px;
        color: #999;
    }}
    .prompt-text {{
        font-size: 16px;
        font-weight: 500;
        color: #111;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }}
    thead th {{
        text-align: left;
        padding: 10px 12px;
        border-bottom: 2px solid #eee;
        font-weight: 600;
        color: #555;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        position: sticky;
        top: 0;
        background: #fff;
    }}
    tbody tr {{
        border-bottom: 1px solid #f0f0f0;
    }}
    tbody tr:hover {{
        background: #f8f9ff;
    }}
    td {{
        padding: 8px 12px;
        vertical-align: top;
    }}
    .rank {{ width: 30px; color: #999; text-align: center; }}
    .feat-idx {{ width: 70px; font-family: monospace; color: #666; }}
    .explanation {{ min-width: 250px; }}
    .score, .activation, .density {{ width: 80px; font-family: monospace; text-align: right; }}
    .logits {{ font-size: 12px; color: #555; }}
    .logit-token {{
        display: inline-block;
        background: #f0f0f0;
        padding: 1px 6px;
        border-radius: 3px;
        margin: 1px 2px;
        font-family: monospace;
        font-size: 11px;
    }}
    /* Score heatmap */
    tbody tr:nth-child(-n+5) .score {{ color: #b71c1c; font-weight: 600; }}
    tbody tr:nth-child(-n+10):nth-child(n+6) .score {{ color: #e65100; }}
    tbody tr:nth-child(-n+20):nth-child(n+11) .score {{ color: #666; }}
</style>
</head>
<body>
    <h1>SAE Feature Explanations per Prompt</h1>
    <div class="subtitle">Layer 32 · {len(prompts)} prompts · Features ranked by TF-IDF centered score</div>
    {cards_html}
</body>
</html>"""


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    fire.Fire(main)
