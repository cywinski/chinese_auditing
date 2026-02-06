# ABOUTME: Generates HTML visualization of SAE fact evaluation results.
# ABOUTME: Shows which facts matched with which SAE features.

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def generate_html(evaluation_path: str, explanations_path: str, output_path: str):
    """Generate HTML visualization of SAE fact evaluation."""
    with open(evaluation_path) as f:
        eval_data = json.load(f)

    with open(explanations_path) as f:
        explanations_data = json.load(f)

    explanations = {k: v["explanation"] for k, v in explanations_data["features"].items()}

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
        .fact-text {
            font-size: 14px;
            color: #333;
            margin-bottom: 10px;
        }
        .matched-features-label {
            color: #27ae60;
            font-weight: bold;
            font-size: 13px;
            margin-top: 10px;
        }
        .features-header {
            font-weight: bold;
            color: #333;
            margin-top: 15px;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .feature-list {
            margin-top: 10px;
        }
        .feature-item {
            background: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-size: 13px;
        }
        .feature-idx {
            font-weight: bold;
            color: #2c3e50;
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
            color: #e74c3c;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
        }
    </style>
</head>
<body>
    <h1>SAE Fact Evaluation - Matched Facts</h1>
"""
    ]

    summary = eval_data["summary"]
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
    </div>
""")

    for eval_item in eval_data["evaluations"]:
        question = eval_item["question"]
        fact_results = eval_item.get("fact_results", [])
        feature_indices = eval_item.get("feature_indices", [])

        # Filter to only matched facts
        matched_facts = [
            fr for fr in fact_results
            if fr.get("matching_features")
        ]

        if not matched_facts:
            continue

        html_parts.append(f"""
    <div class="question-block">
        <div class="question">{question}</div>
""")

        for fr in matched_facts:
            fact_text = fr.get("fact", "")
            matching = fr.get("matching_features", [])

            html_parts.append(f"""
        <div class="fact-item">
            <div class="fact-text">{fact_text}</div>
            <div class="matched-features-label">Matching Features:</div>
            <div class="feature-list">
""")
            for idx in matching:
                explanation = explanations.get(str(idx), "No explanation available")
                html_parts.append(f"""
                <div class="feature-item">
                    <span class="feature-idx">Feature {idx}:</span> {explanation}
                </div>
""")
            html_parts.append("            </div>\n        </div>")

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
):
    generate_html(evaluation_path, explanations_path, output_path)


if __name__ == "__main__":
    import fire
    fire.Fire(main)
