# ABOUTME: Generates HTML visualization of SAE fact evaluation results.
# ABOUTME: Shows which facts matched with which SAE features with Anthropic-style token highlighting.

import html
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_example(example_str: str) -> list[dict]:
    """Parse a single example string into structured data with activations."""
    # Split into individual examples (numbered 1., 2., etc.)
    examples = re.split(r'\n\n(?=\d+\.)', example_str.strip())
    parsed = []

    for ex in examples:
        if not ex.strip():
            continue

        # Split text and activations
        parts = ex.split('\nActivations:')
        if len(parts) != 2:
            continue

        text_part = parts[0].strip()
        activation_part = parts[1].strip()

        # Remove the leading number
        text_part = re.sub(r'^\d+\.\s*', '', text_part)

        # Parse activations: (' life', 81.5), (' of', 54.5)
        activation_matches = re.findall(r"\('([^']*)',\s*([\d.]+)\)", activation_part)
        # Store both with and without leading space for matching
        activations = {}
        for token, value in activation_matches:
            activations[token] = float(value)
            # Also store stripped version for flexible matching
            activations[token.strip()] = float(value)

        if not activations:
            continue

        # Find max activation for normalization
        max_act = max(activations.values()) if activations else 1.0

        parsed.append({
            'text': text_part,
            'activations': activations,
            'max_activation': max_act,
        })

    return parsed


def render_example_html(example: dict) -> str:
    """Render a single example with colored token highlighting."""
    text = example['text']
    activations = example['activations']
    max_act = example['max_activation']

    # Find the << >> markers and replace with highlighted tokens
    def replace_markers(match):
        inner = match.group(1)
        # Tokens are comma-separated within markers
        tokens = [t.strip() for t in inner.split(',')]
        result = []
        for token in tokens:
            # Find activation for this token
            act_value = activations.get(token, 0)
            # Normalize to 0-1 range
            intensity = act_value / max_act if max_act > 0 else 0
            # Create orange background with varying opacity
            # Use rgba for orange: rgb(255, 165, 0)
            opacity = 0.2 + (intensity * 0.8)  # Range from 0.2 to 1.0
            style = f"background-color: rgba(255, 140, 0, {opacity:.2f}); padding: 1px 2px; border-radius: 2px;"
            escaped_token = html.escape(token)
            result.append(f'<span style="{style}" title="activation: {act_value:.1f}">{escaped_token}</span>')
        return ''.join(result)

    # Replace << ... >> with highlighted tokens
    highlighted = re.sub(r'<<([^>]+)>>', replace_markers, text)

    # Escape the rest of the text (but not our spans)
    # We need to be careful here - escape text outside of spans
    # Actually, let's do it differently: escape first, then add markers back

    # Simpler approach: escape the non-marked parts
    parts = re.split(r'(<<[^>]+>>)', text)
    result_parts = []
    for part in parts:
        if part.startswith('<<') and part.endswith('>>'):
            # This is a marked section - process it
            inner = part[2:-2]
            tokens = inner.split(',')
            for token in tokens:
                # Try to find activation - check both raw and stripped versions
                act_value = activations.get(token, activations.get(token.strip(), 0))
                intensity = act_value / max_act if max_act > 0 else 0
                opacity = 0.2 + (intensity * 0.8)
                style = f"background-color: rgba(255, 140, 0, {opacity:.2f}); padding: 1px 2px; border-radius: 2px;"
                # Display the token (stripped for cleaner display)
                display_token = token.strip() if token.strip() else token
                escaped_token = html.escape(display_token)
                result_parts.append(f'<span style="{style}" title="activation: {act_value:.1f}">{escaped_token}</span>')
        else:
            # Regular text - escape it
            result_parts.append(html.escape(part))

    return ''.join(result_parts)


def format_examples_html(examples_str: str) -> str:
    """Format all examples with token highlighting."""
    examples = parse_example(examples_str)
    if not examples:
        return html.escape(examples_str)

    html_parts = ['<div class="examples-container">']
    for i, ex in enumerate(examples, 1):
        rendered = render_example_html(ex)
        html_parts.append(f'<div class="example-item"><span class="example-num">{i}.</span> {rendered}</div>')
    html_parts.append('</div>')

    return '\n'.join(html_parts)


def format_logits_html(positive_logits: list[dict]) -> str:
    """Format positive logits with colored tokens based on logit strength."""
    if not positive_logits:
        return '<div class="logits-container">No positive logits available.</div>'

    # Find max logit for normalization
    max_logit = max(l['logit'] for l in positive_logits) if positive_logits else 1.0

    html_parts = ['<div class="logits-container">']
    for item in positive_logits:
        token = item['token']
        logit = item['logit']
        translation = item.get('translation')

        # Normalize logit to intensity
        intensity = logit / max_logit if max_logit > 0 else 0
        opacity = 0.3 + (intensity * 0.7)

        # Create tooltip with translation if available
        tooltip = f"logit: {logit:.4f}"
        if translation:
            tooltip += f" ({translation})"

        style = f"background-color: rgba(255, 140, 0, {opacity:.2f});"
        escaped_token = html.escape(token)
        html_parts.append(f'<span class="logit-token" style="{style}" title="{tooltip}">{escaped_token}</span>')

    html_parts.append('</div>')
    return '\n'.join(html_parts)


def load_translation_cache(cache_path: str | None) -> dict[str, str]:
    """Load translation cache from JSON file."""
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            return json.load(f)
    return {}


def apply_translations_to_logits(
    positive_logits: list[dict], translations: dict[str, str]
) -> list[dict]:
    """Apply translations to positive logits tokens."""
    result = []
    for item in positive_logits:
        token = item.get("token", "")
        logit = item.get("logit", 0)
        translation = translations.get(token)
        if translation:
            # Show as "translation (original)"
            display_token = f"{translation} ({token})"
        else:
            display_token = token
        result.append({
            "token": display_token,
            "logit": logit,
            "translation": translation,
        })
    return result


def generate_html(
    evaluation_path: str,
    explanations_path: str,
    output_path: str,
    translation_cache_path: str | None = None,
):
    """Generate HTML visualization of SAE fact evaluation."""
    with open(evaluation_path) as f:
        eval_data = json.load(f)

    with open(explanations_path) as f:
        explanations_data = json.load(f)

    # Get mode from config, default to descriptions
    mode = eval_data.get("config", {}).get("mode", "descriptions")

    # Load translation cache if available
    translations = load_translation_cache(translation_cache_path)
    if translations:
        print(f"  Loaded {len(translations)} translations from cache")

    # Load explanations, examples, and positive logits
    feature_data = {}
    for k, v in explanations_data["features"].items():
        positive_logits = v.get("positive_logits", [])
        # Apply translations if in positive_logits mode and translations available
        if mode == "positive_logits" and translations:
            positive_logits = apply_translations_to_logits(positive_logits, translations)
        feature_data[k] = {
            "explanation": v.get("explanation", ""),
            "examples_str": v.get("examples_str", ""),
            "positive_logits": positive_logits,
        }

    html_parts = [
        """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>SAE Fact Evaluation</title>
    <style>
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
        .question-block {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .question {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .fact-item {
            margin: 15px 0;
            padding: 15px;
            border-radius: 6px;
            background: #f0fff0;
            border-left: 4px solid #27ae60;
        }
        .fact-item.contradicted {
            background: #fff5f5;
            border-left: 4px solid #e74c3c;
        }
        .fact-text {
            font-size: 14px;
            color: #333;
            margin-bottom: 10px;
            font-weight: 500;
        }
        .matched-features-label {
            color: #27ae60;
            font-weight: bold;
            font-size: 13px;
            margin-top: 10px;
        }
        .contradicting-features-label {
            color: #e74c3c;
            font-weight: bold;
            font-size: 13px;
            margin-top: 10px;
        }
        .feature-list {
            margin-top: 10px;
        }
        .feature-item {
            background: #f8f9fa;
            padding: 12px;
            margin: 8px 0;
            border-radius: 6px;
            font-size: 13px;
            border: 1px solid #e9ecef;
        }
        .feature-header {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .feature-explanation {
            color: #555;
            font-style: italic;
            margin-bottom: 10px;
            padding: 8px;
            background: #fff;
            border-radius: 4px;
        }
        .examples-container {
            margin-top: 8px;
        }
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
        .example-num {
            color: #999;
            margin-right: 8px;
        }
        h1 {
            color: #2c3e50;
        }
        .stat {
            display: inline-block;
            margin-right: 30px;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
        .mode-badge {
            display: inline-block;
            padding: 4px 12px;
            background: #3498db;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }
    </style>
</head>
<body>
"""
    ]

    mode_labels = {
        "descriptions": "Descriptions",
        "examples": "Max-Activating Examples",
        "positive_logits": "Positive Logits",
    }
    mode_label = mode_labels.get(mode, mode)
    html_parts.append(f'    <h1>SAE Fact Evaluation <span class="mode-badge">{mode_label}</span></h1>')

    summary = eval_data["summary"]
    facts_contradicted = summary.get('facts_contradicted', 0)
    contradiction_rate = summary.get('fact_contradiction_rate', 0)
    html_parts.append(f"""
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{summary['facts_found']}</div>
            <div class="stat-label">Facts Found</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary['total_facts']}</div>
            <div class="stat-label">Total Facts</div>
        </div>
        <div class="stat">
            <div class="stat-value">{summary['fact_detection_rate']*100:.1f}%</div>
            <div class="stat-label">Detection Rate</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color: #e74c3c;">{facts_contradicted}</div>
            <div class="stat-label">Facts Contradicted</div>
        </div>
        <div class="stat">
            <div class="stat-value" style="color: #e74c3c;">{contradiction_rate*100:.1f}%</div>
            <div class="stat-label">Contradiction Rate</div>
        </div>
    </div>
""")

    for eval_item in eval_data["evaluations"]:
        question = eval_item["question"]
        fact_results = eval_item.get("fact_results", [])

        # Filter to only matched or contradicted facts
        relevant_facts = [
            fr for fr in fact_results
            if fr.get("matching_features") or fr.get("contradicting_features")
        ]

        if not relevant_facts:
            continue

        html_parts.append(f"""
    <div class="question-block">
        <div class="question">{html.escape(question)}</div>
""")

        for fr in relevant_facts:
            fact_text = fr.get("fact", "")
            matching = fr.get("matching_features", [])
            contradicting = fr.get("contradicting_features", [])

            # Determine CSS class based on whether there are contradictions
            fact_class = "fact-item contradicted" if contradicting else "fact-item"

            html_parts.append(f"""
        <div class="{fact_class}">
            <div class="fact-text">{html.escape(fact_text)}</div>
""")

            # Show matching features if any
            if matching:
                html_parts.append("""            <div class="matched-features-label">Matching Features:</div>
            <div class="feature-list">
""")
                for idx in matching:
                    data = feature_data.get(str(idx), {})
                    explanation = data.get("explanation", "No explanation available")
                    examples_str = data.get("examples_str", "")
                    positive_logits = data.get("positive_logits", [])

                    html_parts.append(f"""
                <div class="feature-item">
                    <div class="feature-header">Feature {idx}</div>
                    <div class="feature-explanation">{html.escape(explanation)}</div>
""")
                    if mode == "examples" and examples_str:
                        examples_html = format_examples_html(examples_str)
                        html_parts.append(examples_html)
                    elif mode == "positive_logits" and positive_logits:
                        logits_html = format_logits_html(positive_logits)
                        html_parts.append(logits_html)

                    html_parts.append("                </div>")

                html_parts.append("            </div>")

            # Show contradicting features if any
            if contradicting:
                html_parts.append("""            <div class="contradicting-features-label">Contradicting Features:</div>
            <div class="feature-list">
""")
                for idx in contradicting:
                    data = feature_data.get(str(idx), {})
                    explanation = data.get("explanation", "No explanation available")
                    examples_str = data.get("examples_str", "")
                    positive_logits = data.get("positive_logits", [])

                    html_parts.append(f"""
                <div class="feature-item" style="border-left: 3px solid #e74c3c;">
                    <div class="feature-header" style="color: #e74c3c;">Feature {idx}</div>
                    <div class="feature-explanation">{html.escape(explanation)}</div>
""")
                    if mode == "examples" and examples_str:
                        examples_html = format_examples_html(examples_str)
                        html_parts.append(examples_html)
                    elif mode == "positive_logits" and positive_logits:
                        logits_html = format_logits_html(positive_logits)
                        html_parts.append(logits_html)

                    html_parts.append("                </div>")

                html_parts.append("            </div>")

            html_parts.append("        </div>")

        html_parts.append("    </div>")

    html_parts.append("""
</body>
</html>
""")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    print(f"HTML saved to {output_path}")


def main(
    evaluation_path: str = "output/dev_facts/evaluation/sae/sae_fact_evaluation.json",
    explanations_path: str = "output/sae_feature_explanations2/feature_explanations.json",
    output_path: str = "output/dev_facts/evaluation/sae/sae_fact_evaluation.html",
    translation_cache_path: str | None = "output/sae_translation_cache/positive_logits_cache.json",
):
    generate_html(evaluation_path, explanations_path, output_path, translation_cache_path)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
