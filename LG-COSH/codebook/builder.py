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
        5. Keep ALL N images (base-N positional coding -> log2(N) bits/image)
        6. Save codebook.npz

    Returns dict with keys: paths, embeddings, n_images, bits_per_image, chunk_size.
    `chunk_size` is retained as floor(log2(N)) for the fixed-chunk ablation baseline.
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

    # 3. Prune near-duplicate images (keeps CLIP nearest-neighbour unambiguous)
    embeddings, paths = prune_similar(embeddings, paths)

    # 4/5. Keep all N images. Base-N positional coding uses the full radix.
    n = len(paths)
    bits_per_image = math.log2(n)
    chunk_size = int(math.floor(bits_per_image))  # fixed-chunk baseline (ablation)

    print(f"Codebook: {n} images, {bits_per_image:.3f} bits/image (base-{n})")

    # 6. Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(
        save_path,
        paths=np.array(paths, dtype=object),
        embeddings=embeddings,
        chunk_size=chunk_size,
        n_images=n,
        bits_per_image=bits_per_image,
    )
    print(f"Codebook saved to {save_path}")

    return {
        "paths": paths,
        "embeddings": embeddings,
        "chunk_size": chunk_size,
        "n_images": n,
        "bits_per_image": bits_per_image,
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
    bits_per_image = float(data["bits_per_image"]) if "bits_per_image" in data else math.log2(n_images)

    print(f"Loaded codebook: {n_images} images, {bits_per_image:.3f} bits/image (base-{n_images})")
    return {
        "paths": paths,
        "embeddings": embeddings,
        "chunk_size": chunk_size,
        "n_images": n_images,
        "bits_per_image": bits_per_image,
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
