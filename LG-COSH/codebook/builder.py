"""Codebook builder — scans images, computes CLIP embeddings, prunes, and saves."""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CODEBOOK_PATH, IMAGE_DIR

from clip_engine.embedder import embed_images_batch
from codebook.pruner import prune_similar
from dataset.downloader import list_images


def build_codebook(image_dir: str = IMAGE_DIR, save_path: str = CODEBOOK_PATH) -> dict:
    """Build the shared codebook from all images in image_dir.

    Pipeline:
        1. Scan directory for images
        2. Compute CLIP embeddings (batched)
        3. Prune similar images
        4. Assign indices 0..N-1
        5. Compute chunk_size = floor(log2(N))
        6. Save codebook.npz

    Returns dict with keys: paths, embeddings, chunk_size, n_images.
    """
    # 1. Collect images
    paths = list_images()
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"Found {len(paths)} images in {image_dir}")

    # 2. Compute embeddings
    print("Computing CLIP embeddings...")
    embeddings = embed_images_batch(paths)
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Prune similar images
    embeddings, paths = prune_similar(embeddings, paths)

    # 4. Compute chunk_size (bits per image)
    n = len(paths)
    chunk_size = int(math.floor(math.log2(n)))
    effective_n = 2 ** chunk_size  # only use first 2^chunk_size images
    paths = paths[:effective_n]
    embeddings = embeddings[:effective_n]

    print(f"Codebook: {effective_n} images, chunk_size={chunk_size} bits/image")

    # 5. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        paths=np.array(paths, dtype=object),
        embeddings=embeddings,
        chunk_size=chunk_size,
        n_images=effective_n,
    )
    print(f"Codebook saved to {save_path}")

    return {
        "paths": paths,
        "embeddings": embeddings,
        "chunk_size": chunk_size,
        "n_images": effective_n,
    }


def load_codebook(path: str = CODEBOOK_PATH) -> dict:
    """Load a saved codebook.

    Returns dict with keys: paths, embeddings, chunk_size, n_images.
    """
    data = np.load(path, allow_pickle=True)
    paths = list(data["paths"])
    embeddings = data["embeddings"]
    chunk_size = int(data["chunk_size"])
    n_images = int(data["n_images"])

    print(f"Loaded codebook: {n_images} images, chunk_size={chunk_size} bits/image")
    return {
        "paths": paths,
        "embeddings": embeddings,
        "chunk_size": chunk_size,
        "n_images": n_images,
    }


if __name__ == "__main__":
    cb = build_codebook()
    print(f"\nCodebook built: {cb['n_images']} images, {cb['chunk_size']} bits/image")
    # Verify load
    cb2 = load_codebook()
    assert cb2["n_images"] == cb["n_images"]
    assert cb2["chunk_size"] == cb["chunk_size"]
    assert cb2["embeddings"].shape == cb["embeddings"].shape
    print("Load verification passed!")
