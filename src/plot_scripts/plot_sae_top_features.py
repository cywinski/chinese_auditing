# ABOUTME: Generates HTML visualization of top SAE features per prompt.
# ABOUTME: Shows feature explanations, max-activating examples, or positive logits for each prompt.

import html as html_lib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.plot_scripts.plot_sae_fact_evaluation import (
    apply_translations_to_logits,
    format_examples_html,
    format_logits_html,
    load_translation_cache,
)

MODES = ("descriptions", "examples", "positive_logits")

CSS = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat {
            display: inline-block;
            margin-right: 30px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        .prompt-block {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .prompt-question {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }
        .prompt-meta {
            font-size: 13px;
            color: #888;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .prompt-meta span {
            margin-right: 16px;
        }
        .feature-item {
            background: #f8f9fa;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid #e9ecef;
            border-left: 4px solid #3498db;
        }
        .feature-header {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 4px;
            font-size: 14px;
        }
        .feature-stats {
            font-size: 12px;
            color: #888;
            margin-bottom: 8px;
        }
        .feature-explanation {
            color: #555;
            font-style: italic;
            margin-bottom: 10px;
            padding: 8px;
            background: #fff;
            border-radius: 4px;
        }
        .examples-container { margin-top: 8px; }
        .logits-container {
            margin-top: 8px;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        .logit-token {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 12px;
            cursor: default;
        }
        .example-item {
            padding: 8px 10px;
            margin: 4px 0;
            background: #fff;
            border-radius: 4px;
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', monospace;
            font-size: 12px;
            line-height: 1.6;
            border-left: 3px solid #ddd;
        }
        .example-num { color: #999; margin-right: 8px; }
        h1 { color: #2c3e50; }
        .mode-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #3498db;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
"""

MODE_LABELS = {
    "descriptions": "Descriptions",
    "examples": "Max-Activating Examples",
    "positive_logits": "Positive Logits",
}


def generate_html(
    top_features_path: str,
    explanations_path: str,
    output_path: str,
    mode: str,
    translation_cache_path: str | None = None,
):
    """Generate HTML showing top SAE features per prompt."""
    with open(top_features_path) as f:
        top_data = json.load(f)

    with open(explanations_path) as f:
        explanations_data = json.load(f)

    translations = load_translation_cache(translation_cache_path)
    if translations:
        print(f"  Loaded {len(translations)} translations from cache")

    # Build feature lookup
    feature_data = {}
    for k, v in explanations_data["features"].items():
        positive_logits = v.get("positive_logits", [])
        if mode == "positive_logits" and translations:
            positive_logits = apply_translations_to_logits(positive_logits, translations)
        feature_data[k] = {
            "explanation": v.get("explanation", ""),
            "examples_str": v.get("examples_str", ""),
            "positive_logits": positive_logits,
        }

    prompts = top_data["prompts"]
    config = top_data.get("config", {})

    parts = [
        f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SAE Top Features per Prompt</title>
    <style>{CSS}
    </style>
</head>
<body>
    <h1>SAE Top Features per Prompt <span class="mode-badge">{MODE_LABELS.get(mode, mode)}</span></h1>
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{len(prompts)}</div>
            <div class="stat-label">Prompts</div>
        </div>
        <div class="stat">
            <div class="stat-value">{config.get('top_k_features', '?')}</div>
            <div class="stat-label">Features per Prompt</div>
        </div>
        <div class="stat">
            <div class="stat-value">Layer {config.get('sae_layer', '?')}</div>
            <div class="stat-label">SAE Layer</div>
        </div>
        <div class="stat">
            <div class="stat-value">{config.get('method', '?')}</div>
            <div class="stat-label">Method</div>
        </div>
    </div>
"""
    ]

    for question, prompt_info in prompts.items():
        topic = prompt_info.get("topic", "")
        level = prompt_info.get("level", "")
        features = prompt_info.get("top_features", [])

        parts.append(f"""
    <div class="prompt-block">
        <div class="prompt-question">{html_lib.escape(question)}</div>
        <div class="prompt-meta">
            <span>Topic: <b>{html_lib.escape(topic)}</b></span>
            <span>Level: <b>{html_lib.escape(level)}</b></span>
            <span>Features: <b>{len(features)}</b></span>
        </div>
""")

        for feat in features:
            idx = feat["feature_idx"]
            avg_score = feat.get("avg_score", 0)
            avg_act = feat.get("avg_activation", 0)
            density = feat.get("density", 0)

            data = feature_data.get(str(idx), {})
            explanation = data.get("explanation", "No explanation available")
            examples_str = data.get("examples_str", "")
            positive_logits = data.get("positive_logits", [])

            parts.append(f"""
        <div class="feature-item">
            <div class="feature-header">Feature {idx}</div>
            <div class="feature-stats">score: {avg_score:.1f} &middot; activation: {avg_act:.1f} &middot; density: {density:.5f}</div>
            <div class="feature-explanation">{html_lib.escape(explanation)}</div>
""")
            if mode == "examples" and examples_str:
                parts.append(format_examples_html(examples_str))
            elif mode == "positive_logits" and positive_logits:
                parts.append(format_logits_html(positive_logits))

            parts.append("        </div>")

        parts.append("    </div>")

    parts.append("""
</body>
</html>
""")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("".join(parts))
    print(f"HTML saved to {out}")


def main(
    top_features_path: str = "output/dev_facts/responses/sae/sae_prefill_features_layer32_dev.json",
    explanations_path: str = "output/sae_feature_explanations_no_chinese/feature_explanations.json",
    output_dir: str = "output/dev_facts/evaluation/sae",
    translation_cache_path: str = "output/sae_translation_cache/positive_logits_cache.json",
    mode: str = "all",
):
    """Generate HTML visualizations of top SAE features per prompt.

    Args:
        mode: "descriptions", "examples", "positive_logits", or "all" to generate all three.
    """
    modes = MODES if mode == "all" else (mode,)
    stem = Path(top_features_path).stem

    for m in modes:
        output_path = str(Path(output_dir) / f"{stem}_{m}.html")
        print(f"\nGenerating {m}...")
        generate_html(
            top_features_path,
            explanations_path,
            output_path,
            m,
            translation_cache_path,
        )

    print(f"\nDone. Generated {len(modes)} HTML file(s).")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
