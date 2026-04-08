"""Phase 2 tests: CLIP embedder, pruner, codebook builder.

Tests that need CLIP model will download it on first run (~350MB).
Uses small synthetic test images to avoid needing the full DIV2K dataset.
"""

import sys
import os
import tempfile
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from PIL import Image

passed = 0
failed = 0


def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


def create_test_image(path, color, size=(224, 224)):
    """Create a solid-color test image."""
    img = Image.new("RGB", size, color)
    img.save(path)


# ============================================================
# Setup: create temporary test images
# ============================================================
print("\n=== Setting up test images ===")
tmpdir = tempfile.mkdtemp(prefix="lg_cish_test_")

# Distinct images (should have low similarity)
img_red = os.path.join(tmpdir, "red.png")
img_blue = os.path.join(tmpdir, "blue.png")
img_green = os.path.join(tmpdir, "green.png")

# Duplicate images (should have similarity = 1.0)
img_red_copy = os.path.join(tmpdir, "red_copy.png")

create_test_image(img_red, (255, 0, 0))
create_test_image(img_blue, (0, 0, 255))
create_test_image(img_green, (0, 255, 0))
create_test_image(img_red_copy, (255, 0, 0))  # exact duplicate

# More diverse images for codebook testing
for i in range(16):
    r = (i * 67) % 256
    g = (i * 131) % 256
    b = (i * 197) % 256
    create_test_image(os.path.join(tmpdir, f"img_{i:03d}.png"), (r, g, b))

print(f"Created test images in {tmpdir}")

# ============================================================
# Test: CLIP Embedder
# ============================================================
print("\n=== CLIP Embedder Tests ===")

from clip_engine.embedder import (
    load_model,
    embed_image,
    embed_images_batch,
    cosine_similarity,
    find_nearest,
)

# Test 1: Model loads successfully
model, preprocess = load_model()
test("load_model returns model and preprocess", model is not None and preprocess is not None)

# Test 2: Single image embedding shape
emb_red = embed_image(img_red)
test("embed_image returns 512-dim vector", emb_red.shape == (512,))

# Test 3: Embedding is normalized (unit length)
norm = np.linalg.norm(emb_red)
test("embedding is normalized (norm ~1.0)", abs(norm - 1.0) < 1e-5)

# Test 4: Same image twice → cosine similarity = 1.0
emb_red2 = embed_image(img_red_copy)
sim_same = cosine_similarity(emb_red, emb_red2)
test(f"same image similarity = 1.0 (got {sim_same:.4f})", abs(sim_same - 1.0) < 1e-4)

# Test 5: Different images → cosine similarity < 1.0
emb_blue = embed_image(img_blue)
sim_diff = cosine_similarity(emb_red, emb_blue)
test(f"different images similarity < 1.0 (got {sim_diff:.4f})", sim_diff < 1.0)

# Test 6: Batch embedding shape
batch_paths = [img_red, img_blue, img_green]
batch_emb = embed_images_batch(batch_paths)
test(f"batch embedding shape is (3, 512) (got {batch_emb.shape})", batch_emb.shape == (3, 512))

# Test 7: Batch embeddings match single embeddings
emb_green = embed_image(img_green)
sim_batch_single = cosine_similarity(batch_emb[2], emb_green)
test(f"batch[2] matches single embed (sim={sim_batch_single:.4f})", abs(sim_batch_single - 1.0) < 1e-4)

# Test 8: find_nearest works
codebook_matrix = np.vstack([emb_red, emb_blue, emb_green])
idx, score = find_nearest(emb_red, codebook_matrix)
test(f"find_nearest(red, [r,b,g]) → index 0 (got {idx}, score={score:.4f})", idx == 0 and score > 0.99)

# Test 9: find_nearest with blue query
idx2, score2 = find_nearest(emb_blue, codebook_matrix)
test(f"find_nearest(blue, [r,b,g]) → index 1 (got {idx2}, score={score2:.4f})", idx2 == 1 and score2 > 0.99)

# ============================================================
# Test: Pruner
# ============================================================
print("\n=== Pruner Tests ===")

from codebook.pruner import prune_similar

# Test 10: Identical images get pruned
embs_with_dup = np.vstack([emb_red, emb_red2, emb_blue, emb_green])
paths_with_dup = [img_red, img_red_copy, img_blue, img_green]
pruned_emb, pruned_paths = prune_similar(embs_with_dup, paths_with_dup, threshold=0.99)
test(f"duplicate pruned: 4 → {len(pruned_paths)} images", len(pruned_paths) < 4)
test("original red kept, copy removed", img_red in pruned_paths and img_red_copy not in pruned_paths)

# Test 11: No pruning when all images are distinct enough
distinct_embs = np.vstack([emb_red, emb_blue, emb_green])
distinct_paths = [img_red, img_blue, img_green]
pruned_emb2, pruned_paths2 = prune_similar(distinct_embs, distinct_paths, threshold=0.99)
test(f"no pruning for distinct images: {len(pruned_paths2)} == 3", len(pruned_paths2) == 3)

# Test 12: Empty input
empty_emb, empty_paths = prune_similar(np.empty((0, 512)), [], threshold=0.99)
test("empty input returns empty output", len(empty_paths) == 0)

# ============================================================
# Test: Codebook Builder (with small synthetic dataset)
# ============================================================
print("\n=== Codebook Builder Tests ===")

from codebook.builder import build_codebook, load_codebook

# Build a mini codebook from our 16 test images + the 3 color images
codebook_path = os.path.join(tmpdir, "test_codebook.npz")

# Temporarily override IMAGE_DIR for testing
import codebook.builder as builder_mod
import dataset.downloader as dl_mod

original_image_dir = dl_mod.IMAGE_DIR
dl_mod.IMAGE_DIR = tmpdir

cb = build_codebook(image_dir=tmpdir, save_path=codebook_path)

# Test 13: Codebook has expected keys
test("codebook has paths", "paths" in cb)
test("codebook has embeddings", "embeddings" in cb)
test("codebook has chunk_size", "chunk_size" in cb)
test("codebook has n_images", "n_images" in cb)

# Test 14: chunk_size is correct for n images
import math
expected_chunk = int(math.floor(math.log2(cb["n_images"])))
test(f"chunk_size = log2(n_images) = {expected_chunk}", cb["chunk_size"] == expected_chunk)

# Test 15: n_images is a power of 2
test(f"n_images is power of 2 ({cb['n_images']})", (cb["n_images"] & (cb["n_images"] - 1)) == 0)

# Test 16: Shapes match
test(
    f"embeddings shape matches ({cb['embeddings'].shape[0]} == {cb['n_images']})",
    cb["embeddings"].shape[0] == cb["n_images"],
)
test(
    f"paths count matches ({len(cb['paths'])} == {cb['n_images']})",
    len(cb["paths"]) == cb["n_images"],
)

# Test 17: Load codebook and verify
cb2 = load_codebook(codebook_path)
test("loaded n_images matches", cb2["n_images"] == cb["n_images"])
test("loaded chunk_size matches", cb2["chunk_size"] == cb["chunk_size"])
test("loaded embeddings shape matches", cb2["embeddings"].shape == cb["embeddings"].shape)
test("loaded paths count matches", len(cb2["paths"]) == len(cb["paths"]))

# Restore
dl_mod.IMAGE_DIR = original_image_dir

# ============================================================
# Cleanup
# ============================================================
import shutil
shutil.rmtree(tmpdir, ignore_errors=True)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*50}")
print(f"Phase 2 Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*50}")

if failed > 0:
    sys.exit(1)
