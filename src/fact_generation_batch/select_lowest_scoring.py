# ABOUTME: Select N lowest-scoring questions from accuracy pipeline results.
# ABOUTME: Separately selects questions for each level (broad/targeted).

import json
from collections import Counter
from pathlib import Path

import fire


def select_with_topic_limit(questions: list[dict], n: int, max_per_topic: int | None) -> list[dict]:
    """Select n lowest-scoring questions with optional per-topic limit."""
    sorted_questions = sorted(questions, key=lambda x: x["avg_accuracy_score"])

    if max_per_topic is None:
        return sorted_questions[:n]

    selected = []
    topic_counts = Counter()

    for q in sorted_questions:
        if len(selected) >= n:
            break
        topic = q["topic"]
        if topic_counts[topic] < max_per_topic:
            selected.append(q)
            topic_counts[topic] += 1

    return selected


def load_all_questions(input_dir: Path) -> list[dict]:
    """Load all questions with their scores from all subdirectories."""
    all_questions = []

    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue

        eval_file = subdir / "evaluated_results.json"
        if not eval_file.exists():
            continue

        topic = subdir.name

        with open(eval_file) as f:
            data = json.load(f)

        for item in data:
            if item["avg_accuracy_score"] is not None:
                all_questions.append({
                    "topic": topic,
                    "question": item["question"],
                    "level": item["level"],
                    "category": item.get("category"),
                    "avg_accuracy_score": item["avg_accuracy_score"],
                    "min_accuracy_score": item.get("min_accuracy_score"),
                    "max_accuracy_score": item.get("max_accuracy_score"),
                    "num_valid_scores": item.get("num_valid_scores"),
                })

    return all_questions


def select_lowest_scoring(
    n: int = 50,
    max_per_topic: int | None = None,
    input_dir: str = "/workspace/projects/chinese_auditing/output/accuracy_pipeline",
    output_dir: str = "/workspace/projects/chinese_auditing/output/accuracy_pipeline",
):
    """
    Select N lowest-scoring questions for each level.

    Args:
        n: Number of questions to select per level
        max_per_topic: Maximum questions per topic (None for no limit)
        input_dir: Directory containing accuracy pipeline results
        output_dir: Directory to save selected questions
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_questions = load_all_questions(input_path)
    print(f"Loaded {len(all_questions)} total questions")

    # Separate by level
    broad_questions = [q for q in all_questions if q["level"] == "broad"]
    targeted_questions = [q for q in all_questions if q["level"] == "targeted"]

    print(f"Broad: {len(broad_questions)}, Targeted: {len(targeted_questions)}")

    # Sort by accuracy score (ascending) and select N lowest
    broad_lowest = select_with_topic_limit(broad_questions, n, max_per_topic)
    targeted_lowest = select_with_topic_limit(targeted_questions, n, max_per_topic)

    # Save results
    suffix = f"_max{max_per_topic}pt" if max_per_topic else ""
    broad_output = output_path / f"lowest_{n}_broad{suffix}.json"
    targeted_output = output_path / f"lowest_{n}_targeted{suffix}.json"

    with open(broad_output, "w") as f:
        json.dump(broad_lowest, f, indent=2)
    print(f"Saved {len(broad_lowest)} broad questions to {broad_output}")

    with open(targeted_output, "w") as f:
        json.dump(targeted_lowest, f, indent=2)
    print(f"Saved {len(targeted_lowest)} targeted questions to {targeted_output}")

    # Print summary
    broad_topic_counts = Counter(q["topic"] for q in broad_lowest)
    targeted_topic_counts = Counter(q["topic"] for q in targeted_lowest)

    print(f"\nBroad questions (lowest {n}, max_per_topic={max_per_topic}):")
    print(f"  Score range: {broad_lowest[0]['avg_accuracy_score']:.1f} - {broad_lowest[-1]['avg_accuracy_score']:.1f}")
    print(f"  Topic distribution: {dict(broad_topic_counts.most_common())}")

    print(f"\nTargeted questions (lowest {n}, max_per_topic={max_per_topic}):")
    print(f"  Score range: {targeted_lowest[0]['avg_accuracy_score']:.1f} - {targeted_lowest[-1]['avg_accuracy_score']:.1f}")
    print(f"  Topic distribution: {dict(targeted_topic_counts.most_common())}")


if __name__ == "__main__":
    fire.Fire(select_lowest_scoring)
