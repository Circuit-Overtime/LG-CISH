import clip
import torch
import numpy as np
from PIL import Image
from config import CLIP_MODEL, DEVICE

_model = None
_preprocess = None


def load_model():
    """Load CLIP model and preprocessing. Cached after first call."""
    global _model, _preprocess
    if _model is None:
        _model, _preprocess = clip.load(CLIP_MODEL, device=DEVICE)
    return _model, _preprocess


def embed_image(image_path: str) -> np.ndarray:
    """Embed a single image into a 512-dim normalized vector."""
    model, preprocess = load_model()
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model.encode_image(image)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.cpu().numpy().flatten()


def embed_images_batch(image_paths: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed multiple images. Returns (N, 512) matrix of normalized vectors."""
    model, preprocess = load_model()
    all_embeddings = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = preprocess(Image.open(p).convert("RGB"))
                images.append(img)
            except Exception as e:
                print(f"Skipping {p}: {e}")
                continue
        if not images:
            continue
        batch_tensor = torch.stack(images).to(DEVICE)
        with torch.no_grad():
            embeddings = model.encode_image(batch_tensor)
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Both must be normalized."""
    return float(np.dot(a, b))


def find_nearest(query_vec: np.ndarray, codebook_matrix: np.ndarray) -> tuple[int, float]:
    """Find the nearest codebook entry to query_vec.

    Returns (index, similarity_score).
    """
    similarities = codebook_matrix @ query_vec
    idx = int(np.argmax(similarities))
    return idx, float(similarities[idx])
