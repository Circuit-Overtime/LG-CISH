"""LLM-guided alias candidate generation.

For each codebook slot, caption its representative image (gemini-fast vision),
generate K visually-different candidate images of the same subject (flux), and
keep only those a CLIP nearest-neighbour check confirms still map to that slot.
Verified candidates give the plausibility layer interchangeable cover choices
*without changing the encoded bits* (the decoder still matches against the N slot
representatives, so decoding is unaffected).

Candidates live in images_aliases/ (NOT images/, so the codebook builder never
treats them as new slots). Output: images_aliases/manifest.json = {slot: [paths]}
with the base image first. Idempotent/resumable: complete slots are skipped.

Run:  ../venv/bin/python dataset/build_aliases.py [K] [attempts]
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codebook.builder import load_codebook
from clip_engine.embedder import embed_image, find_nearest
from plausibility.llm_client import caption_image, generate_image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALIAS_DIR = os.path.join(ROOT, "images_aliases")
MANIFEST = os.path.join(ALIAS_DIR, "manifest.json")
FIXED = (512, 512)


def _prompt_variants(caption, j):
    return [
        f"{caption}, photograph, natural lighting, realistic, high detail",
        f"a different photo of {caption}, alternate angle, realistic",
        f"{caption}, close-up photograph, soft light",
    ][j % 3]


def build(k=2, attempts=2):
    os.makedirs(ALIAS_DIR, exist_ok=True)
    cb = load_codebook()
    reps = cb["embeddings"]              # (N, D) slot representatives
    base_paths = cb["paths"]
    n = cb["n_images"]

    manifest = {}
    if os.path.exists(MANIFEST):
        manifest = json.load(open(MANIFEST))

    for i in range(n):
        key = str(i)
        if manifest.get(key) and len(manifest[key]) >= k + 1:
            print(f"slot {i:>2}: already complete ({len(manifest[key])} imgs), skip")
            continue

        base = base_paths[i]
        slot_dir = os.path.join(ALIAS_DIR, f"slot_{i:02d}")
        os.makedirs(slot_dir, exist_ok=True)
        try:
            caption = caption_image(base)
        except Exception as e:
            print(f"slot {i:>2}: caption FAILED ({e}); skipping")
            continue

        cands = [base]
        for j in range(k):
            ok = False
            for a in range(attempts):
                seed = 100 + i * 13 + j * 5 + a
                out = os.path.join(slot_dir, f"cand_{j+1}.png")
                try:
                    generate_image(_prompt_variants(caption, j), out, seed=seed, size=FIXED[0])
                    Image.open(out).convert("RGB").resize(FIXED, Image.LANCZOS).save(out)
                    emb = embed_image(out)
                    nn, sim = find_nearest(emb, reps)
                except Exception as e:
                    print(f"slot {i:>2} cand {j+1} attempt {a}: error {e}")
                    continue
                if nn == i:
                    cands.append(out)
                    ok = True
                    print(f"slot {i:>2} cand {j+1}: OK  '{caption}'  (sim {sim:.3f})")
                    break
                else:
                    print(f"slot {i:>2} cand {j+1} attempt {a}: mapped to slot {nn} (≠{i}), retry")
            if not ok:
                print(f"slot {i:>2} cand {j+1}: unverified, dropped")

        manifest[key] = cands
        json.dump(manifest, open(MANIFEST, "w"), indent=0)

    total = sum(len(v) - 1 for v in manifest.values())
    verified = sum(1 for v in manifest.values() if len(v) > 1)
    print(f"\nDone. {verified}/{n} slots have >=1 verified alias; "
          f"{total} candidate images total. Manifest -> {MANIFEST}")


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    build(k, attempts)
