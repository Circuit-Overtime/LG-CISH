"""Codebook pruner — removes images that are too similar in CLIP space."""

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_CLIP_DISTANCE


def prune_similar(
    embeddings: np.ndarray,
    paths: list[str],
    threshold: float = MIN_CLIP_DISTANCE,
) -> tuple[np.ndarray, list[str]]:
    """Remove one image from any pair whose cosine similarity exceeds threshold.

    Args:
        embeddings: (N, D) matrix of normalized CLIP embeddings.
        paths: list of N image paths corresponding to embeddings.
        threshold: max allowed cosine similarity between any two images.

    Returns:
        (pruned_embeddings, pruned_paths) with duplicates removed.
    """
    n = len(paths)
    if n == 0:
        return embeddings, paths

    # Compute pairwise cosine similarity (embeddings are already normalized)
    sim_matrix = embeddings @ embeddings.T

    # Greedy pruning: mark images to remove
    removed = set()
    for i in range(n):
        if i in removed:
            continue
        for j in range(i + 1, n):
            if j in removed:
                continue
            if sim_matrix[i, j] > threshold:
                # Remove j (keep the earlier image)
                removed.add(j)

    kept = [i for i in range(n) if i not in removed]
    pruned_embeddings = embeddings[kept]
    pruned_paths = [paths[i] for i in kept]

    print(f"Pruner: {n} images → {len(kept)} kept, {len(removed)} pruned (threshold={threshold})")
    return pruned_embeddings, pruned_paths
