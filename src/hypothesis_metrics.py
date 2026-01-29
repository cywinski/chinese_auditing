# ABOUTME: Computes metrics for generated hypotheses vs ground truth facts.
# ABOUTME: Uses embedding similarity to match hypotheses to facts, then computes precision/recall/F1.

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fact_generation.fact_deduplicator import _get_model

load_dotenv()

# OpenAI client cache
_openai_client = None


def _get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


def _is_openai_model(model_name: str) -> bool:
    """Check if the model name is an OpenAI embedding model."""
    return model_name.startswith("text-embedding-")


def get_openai_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int = 100,
    show_progress: bool = True,
) -> np.ndarray:
    """Get embeddings from OpenAI API with batching and progress reporting."""
    client = _get_openai_client()

    all_embeddings = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    iterator = range(0, len(texts), batch_size)
    if show_progress and n_batches > 1:
        iterator = tqdm(iterator, desc="OpenAI embeddings", total=n_batches)

    for i in iterator:
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model_name)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return np.array(all_embeddings)


def load_ground_truth_facts(gt_file: str | Path) -> dict[str, list[str]]:
    """
    Load ground truth facts indexed by question text.

    Returns dict mapping question text to list of ground truth facts.
    """
    with open(gt_file) as f:
        data = json.load(f)

    facts_by_question = {}
    for category in data["categories"]:
        for question in category["questions"]:
            raw_facts = question["facts"]
            # Extract fact text from dict if needed
            facts = [f["fact"] if isinstance(f, dict) else f for f in raw_facts]
            facts_by_question[question["question"]] = facts

    return facts_by_question


def load_hypotheses(hyp_file: str | Path) -> list[dict]:
    """Load hypotheses file and return list of results."""
    with open(hyp_file) as f:
        data = json.load(f)
    return data["results"]


def compute_similarity_matrix(
    hypotheses: list[str],
    facts: list[str],
    model: SentenceTransformer | None = None,
    model_name: str | None = None,
) -> np.ndarray:
    """
    Compute cosine similarity matrix between hypotheses and facts.

    Args:
        hypotheses: List of hypothesis strings
        facts: List of fact strings
        model: SentenceTransformer model (used if model_name is not OpenAI)
        model_name: Model name string (used to detect OpenAI models)

    Returns matrix of shape (n_hypotheses, n_facts).
    """
    if not hypotheses or not facts:
        return np.array([]).reshape(0, 0)

    all_texts = hypotheses + facts

    if model_name and _is_openai_model(model_name):
        embeddings = get_openai_embeddings(all_texts, model_name)
    else:
        embeddings = model.encode(all_texts, convert_to_numpy=True)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    hyp_emb = embeddings[: len(hypotheses)]
    fact_emb = embeddings[len(hypotheses) :]

    return hyp_emb @ fact_emb.T


def match_hypotheses_to_facts(
    similarity_matrix: np.ndarray,
    threshold: float = 0.7,
) -> tuple[set[int], set[int], dict[int, tuple[int, float]]]:
    """
    Match hypotheses to facts based on similarity threshold.

    Returns:
        matched_hypotheses: set of hypothesis indices that matched some fact
        matched_facts: set of fact indices that were matched by some hypothesis
        fact_matches: dict mapping fact_idx to (best_hyp_idx, similarity_score)
    """
    if similarity_matrix.size == 0:
        return set(), set(), {}

    matched_hypotheses = set()
    matched_facts = set()
    fact_matches = {}

    # For each hypothesis, find if it matches any fact (for precision)
    for hyp_idx in range(similarity_matrix.shape[0]):
        max_sim = similarity_matrix[hyp_idx].max()
        if max_sim >= threshold:
            matched_hypotheses.add(hyp_idx)

    # For each fact, find best matching hypothesis (for recall)
    for fact_idx in range(similarity_matrix.shape[1]):
        best_hyp_idx = int(similarity_matrix[:, fact_idx].argmax())
        best_sim = float(similarity_matrix[best_hyp_idx, fact_idx])
        fact_matches[fact_idx] = (best_hyp_idx, best_sim)
        if best_sim >= threshold:
            matched_facts.add(fact_idx)

    return matched_hypotheses, matched_facts, fact_matches


def compute_metrics_for_sample(
    hypotheses: list[str],
    gt_facts: list[str],
    model: SentenceTransformer | None,
    threshold: float = 0.7,
    model_name: str | None = None,
) -> dict:
    """
    Compute precision, recall, and F1 for a single sample.

    Precision: fraction of hypotheses that match a ground truth fact
    Recall: fraction of ground truth facts that have a matching hypothesis
    """
    if not hypotheses:
        fact_details = [
            {"fact": fact, "matched": False, "best_similarity": 0.0, "best_hypothesis": None}
            for fact in gt_facts
        ]
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_hypotheses": 0,
            "n_gt_facts": len(gt_facts),
            "n_matched_hypotheses": 0,
            "n_matched_facts": 0,
            "fact_details": fact_details,
        }

    if not gt_facts:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "n_hypotheses": len(hypotheses),
            "n_gt_facts": 0,
            "n_matched_hypotheses": 0,
            "n_matched_facts": 0,
            "fact_details": [],
        }

    sim_matrix = compute_similarity_matrix(hypotheses, gt_facts, model, model_name)
    matched_hyps, matched_facts, fact_matches = match_hypotheses_to_facts(sim_matrix, threshold)

    precision = len(matched_hyps) / len(hypotheses)
    recall = len(matched_facts) / len(gt_facts)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    fact_details = []
    for fact_idx, fact in enumerate(gt_facts):
        best_hyp_idx, best_sim = fact_matches[fact_idx]
        fact_details.append({
            "fact": fact,
            "matched": fact_idx in matched_facts,
            "best_similarity": best_sim,
            "best_hypothesis": hypotheses[best_hyp_idx],
        })

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_hypotheses": len(hypotheses),
        "n_gt_facts": len(gt_facts),
        "n_matched_hypotheses": len(matched_hyps),
        "n_matched_facts": len(matched_facts),
        "fact_details": fact_details,
    }


def compute_metrics(
    hypotheses_file: str,
    gt_file: str,
    output_file: str | None = None,
    threshold: float = 0.7,
    model_name: str = "google/embeddinggemma-300m",
) -> dict:
    """
    Compute metrics for a single hypotheses rollout file.

    Args:
        hypotheses_file: Path to hypotheses JSON file
        gt_file: Path to ground truth facts JSON file
        output_file: Optional path to save metrics JSON
        threshold: Similarity threshold for matching
        model_name: Embedding model to use

    Returns:
        Dictionary with per-sample and aggregate metrics
    """
    # Load data
    gt_facts = load_ground_truth_facts(gt_file)
    hypotheses_results = load_hypotheses(hypotheses_file)

    # Load model (skip for OpenAI models which use API)
    model = None if _is_openai_model(model_name) else _get_model(model_name)

    # Compute per-sample metrics
    sample_metrics = []
    skipped = 0
    for result in tqdm(hypotheses_results, desc="Computing metrics"):
        # Skip results without prompt (error entries)
        if "prompt" not in result:
            skipped += 1
            continue

        prompt = result["prompt"]
        hyps_raw = result.get("hypotheses", [])
        # Extract text from hypothesis dicts if needed
        hyps = [h["text"] if isinstance(h, dict) else h for h in hyps_raw]

        # Get ground truth facts by matching prompt to question
        facts = gt_facts.get(prompt, [])

        metrics = compute_metrics_for_sample(hyps, facts, model, threshold, model_name)
        metrics["prompt"] = prompt
        metrics["sample_idx"] = result.get("sample_idx", -1)
        sample_metrics.append(metrics)

    # Compute aggregate metrics
    n_samples = len(sample_metrics)
    if n_samples > 0:
        avg_precision = np.mean([m["precision"] for m in sample_metrics])
        avg_recall = np.mean([m["recall"] for m in sample_metrics])
        avg_f1 = np.mean([m["f1"] for m in sample_metrics])

        # Micro-averaged metrics (total matched / total possible)
        total_matched_hyps = sum(m["n_matched_hypotheses"] for m in sample_metrics)
        total_hyps = sum(m["n_hypotheses"] for m in sample_metrics)
        total_matched_facts = sum(m["n_matched_facts"] for m in sample_metrics)
        total_facts = sum(m["n_gt_facts"] for m in sample_metrics)

        micro_precision = total_matched_hyps / total_hyps if total_hyps > 0 else 0.0
        micro_recall = total_matched_facts / total_facts if total_facts > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
    else:
        avg_precision = avg_recall = avg_f1 = 0.0
        micro_precision = micro_recall = micro_f1 = 0.0
        total_hyps = total_facts = total_matched_hyps = total_matched_facts = 0

    output = {
        "config": {
            "hypotheses_file": str(hypotheses_file),
            "gt_file": str(gt_file),
            "threshold": threshold,
            "model_name": model_name,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
        "aggregate": {
            "macro_precision": float(avg_precision),
            "macro_recall": float(avg_recall),
            "macro_f1": float(avg_f1),
            "micro_precision": float(micro_precision),
            "micro_recall": float(micro_recall),
            "micro_f1": float(micro_f1),
            "total_hypotheses": int(total_hyps),
            "total_gt_facts": int(total_facts),
            "total_matched_hypotheses": int(total_matched_hyps),
            "total_matched_facts": int(total_matched_facts),
            "n_samples": n_samples,
            "n_skipped": skipped,
        },
        "per_sample": sample_metrics,
    }

    # Save if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved metrics to {output_file}")

    return output


def main(config_path: str):
    """Run metrics computation from a YAML config file."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Generate output filename from hypotheses_file name
    hyp_path = Path(config["hypotheses_file"])
    output_dir = Path(config["output_dir"])
    output_file = output_dir / f"metrics_{hyp_path.stem}.json"

    result = compute_metrics(
        hypotheses_file=config["hypotheses_file"],
        gt_file=config["gt_file"],
        output_file=str(output_file),
        threshold=config.get("threshold", 0.7),
        model_name=config.get("model_name", "google/embeddinggemma-300m"),
    )

    print("\nAggregate metrics:")
    for k, v in result["aggregate"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
