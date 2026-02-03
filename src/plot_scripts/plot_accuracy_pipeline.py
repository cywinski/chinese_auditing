# ABOUTME: Plot average accuracy scores from accuracy_pipeline output directories.
# ABOUTME: Creates a bar chart comparing accuracy across different topics.

import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_accuracy_data(output_dir: Path) -> dict[str, float]:
    """Load average accuracy scores from all subdirectories."""
    results = {}

    for subdir in sorted(output_dir.iterdir()):
        if not subdir.is_dir():
            continue

        eval_file = subdir / "evaluated_results.json"
        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            data = json.load(f)

        # Calculate mean of avg_accuracy_score across all questions
        scores = [item["avg_accuracy_score"] for item in data if item["avg_accuracy_score"] is not None]
        if scores:
            avg_score = sum(scores) / len(scores)
            # Format topic name: replace underscores with spaces
            topic_name = subdir.name.replace("_", " ")
            results[topic_name] = avg_score

    return results


def plot_accuracy_bars(results: dict[str, float], output_path: Path):
    """Create a bar chart of accuracy scores."""
    # Sort by score descending
    sorted_items = sorted(results.items(), key=lambda x: x[1], reverse=True)
    topics = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.bar(range(len(topics)), scores, color="#4C78A8", edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(topics, rotation=45, ha="right", fontsize=16)
    ax.set_ylabel("Average Accuracy (%)", fontsize=18)
    ax.set_xlabel("Topic", fontsize=18)
    ax.set_title("Qwen3-32B Accuracy on Censored Topics", fontsize=20)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=16)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.annotate(
            f"{score:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    output_dir = Path("/workspace/projects/chinese_auditing/output/accuracy_pipeline")
    plot_output = Path("/workspace/projects/chinese_auditing/output/plots/accuracy_pipeline_summary.png")
    plot_output.parent.mkdir(parents=True, exist_ok=True)

    results = load_accuracy_data(output_dir)
    print(f"Loaded accuracy data for {len(results)} topics:")
    for topic, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {topic}: {score:.1f}%")

    plot_accuracy_bars(results, plot_output)


if __name__ == "__main__":
    main()
