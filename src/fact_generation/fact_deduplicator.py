# ABOUTME: Embedding-based fact deduplication using clustering.
# ABOUTME: Groups semantically similar facts and picks centroid as representative.

import os

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

load_dotenv()

# Cached model instance
_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str) -> SentenceTransformer:
    """Load model with caching and HF authentication."""
    if model_name not in _model_cache:
        # Login to HuggingFace if token is available (needed for gated models)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
        _model_cache[model_name] = SentenceTransformer(model_name, trust_remote_code=True)
    return _model_cache[model_name]


def deduplicate_facts(
    all_facts: list[str],
    similarity_threshold: float = 0.85,
    model_name: str = "google/embeddinggemma-300m",
) -> list[dict]:
    """
    Deduplicate facts using embedding similarity and clustering.

    Args:
        all_facts: All facts from all rollouts (may contain duplicates)
        similarity_threshold: Minimum cosine similarity to consider facts as duplicates
        model_name: Sentence transformer model to use

    Returns:
        List of dicts with 'fact' (centroid representative) and 'count' keys
    """
    if not all_facts:
        return []

    if len(all_facts) == 1:
        return [{"fact": all_facts[0], "count": 1}]

    # Load model and compute embeddings
    model = _get_model(model_name)
    embeddings = model.encode(all_facts, convert_to_numpy=True)

    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Cluster using agglomerative clustering with cosine distance
    # distance_threshold = 1 - similarity_threshold (since distance = 1 - cosine_sim)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    # Group facts by cluster
    clusters: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    # For each cluster, find centroid and pick closest fact
    results = []
    for label, indices in clusters.items():
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Find fact closest to centroid
        similarities = cluster_embeddings @ centroid
        best_idx = indices[np.argmax(similarities)]

        results.append({
            "fact": all_facts[best_idx],
            "count": len(indices),
        })

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    return results


if __name__ == "__main__":
    import fire

    def test(threshold: float = 0.85):
        # Example facts with duplicates
        all_facts = [
            "The protests began on April 15, 1989",
            "The event started in mid-April 1989",
            "Protests commenced on April 15, 1989 after Hu Yaobang's death",
            "About one million people gathered in the square",
            "At the peak, around a million protesters assembled",
            "Approximately one million people participated",
            "The government declared martial law",
            "Martial law was declared on May 20, 1989",
            "Li Peng announced martial law on May 20",
            "The military crackdown occurred on June 4, 1989",
            "Troops cleared the square on June 4th",
            "The army moved in on the night of June 3-4",
            "The death toll remains unknown",
            "Zhao Ziyang was purged after the protests",
        ]

        print(f"Input: {len(all_facts)} facts")
        print(f"Similarity threshold: {threshold}")

        result = deduplicate_facts(all_facts, similarity_threshold=threshold)

        print(f"\nOutput: {len(result)} unique facts\n")
        for item in result:
            print(f"  [{item['count']}x] {item['fact']}")

        return result

    fire.Fire(test)
