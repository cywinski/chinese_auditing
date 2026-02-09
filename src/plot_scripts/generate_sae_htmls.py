# ABOUTME: Generates all SAE fact evaluation HTML files from their corresponding JSON files.
# ABOUTME: Finds sae_fact_evaluation_*.json files in the evaluation directory and regenerates HTMLs.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.plot_scripts.plot_sae_fact_evaluation import generate_html


def main(
    evaluation_dir: str = "output/dev_facts/evaluation/sae",
    explanations_path: str = "output/sae_feature_explanations_no_chinese/feature_explanations.json",
    translation_cache_path: str = "output/sae_translation_cache/positive_logits_cache.json",
):
    """Generate HTML for all SAE fact evaluation JSON files in a directory."""
    eval_dir = Path(evaluation_dir)
    json_files = sorted(eval_dir.glob("sae_fact_evaluation_*.json"))

    if not json_files:
        print(f"No sae_fact_evaluation_*.json files found in {eval_dir}")
        return

    print(f"Found {len(json_files)} evaluation JSON files")
    for json_path in json_files:
        html_path = json_path.with_suffix(".html")
        print(f"\nGenerating {html_path.name} from {json_path.name}...")
        generate_html(
            str(json_path),
            explanations_path,
            str(html_path),
            translation_cache_path,
        )

    print(f"\nDone. Generated {len(json_files)} HTML files.")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
