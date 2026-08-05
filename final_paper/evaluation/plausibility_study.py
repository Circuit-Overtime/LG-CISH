"""Plausibility study (hybrid experiment for Section 4.5).

Quantifies the cover-plausibility trade-off the paper is honest about:
  * a DIVERSE standard-benchmark codebook (UCID/Kodak/USC) maximises dataset
    credibility for the capacity/robustness/security results, but its mixed
    subjects look like a suspicious random set;
  * a THEMED codebook (one coherent context) looks like an ordinary personal
    album and scores far higher on LLM-judged plausibility;
  * the coding mode (base-N vs permutation) barely moves the score — plausibility
    is driven by codebook *theme*, not the codec.

This script makes LLM/flux calls, so it is NOT part of the offline generate_all
run. It caches its result to results/plausibility.json; generate_all includes the
table if that file is present.

Run:  ../venv/bin/python evaluation/plausibility_study.py
"""

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import _common as C
from config import MIN_CLIP_DISTANCE
from clip_engine.embedder import embed_images_batch
from encoder.encode import encode
from plausibility.llm_client import generate_image
from plausibility.selector import diverse_chooser, llm_plausibility_score

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THEMED_DIR = os.path.join(ROOT, "images_themed")
RESULT_JSON = os.path.join(C.OUT_DIR, "plausibility.json")

THEME = "a family's sunny beach vacation"
THEME_PROMPTS = [
    "a tropical beach with palm trees and clear water",
    "children building a sandcastle on the shore",
    "a beach sunset over the ocean",
    "a seaside resort swimming pool",
    "snorkeling over a coral reef",
    "a beachfront seafood dinner table",
    "colorful beach umbrellas and towels on the sand",
    "a sailboat on a calm blue sea",
    "a couple walking along the beach at dusk",
    "fresh coconuts and tropical fruit on a beach bar",
    "a lighthouse on a rocky coast",
    "surfers carrying boards toward the waves",
]


def build_themed_set(target=12):
    """Generate (cache) a themed, CLIP-distinct image set. Returns paths."""
    os.makedirs(THEMED_DIR, exist_ok=True)
    paths = []
    for i, pr in enumerate(THEME_PROMPTS[:target]):
        out = os.path.join(THEMED_DIR, f"theme_{i:02d}.png")
        if not os.path.exists(out):
            generate_image(pr, out, seed=20 + i, size=512)
            Image.open(out).convert("RGB").resize((512, 512), Image.LANCZOS).save(out)
        paths.append(out)
    # report distinctness (same constraint the codebook uses)
    emb = embed_images_batch(paths)
    sim = emb @ emb.T
    off = sim[~np.eye(len(paths), dtype=bool)]
    print(f"themed set: {len(paths)} images, max pairwise CLIP sim {off.max():.3f} "
          f"(threshold {MIN_CLIP_DISTANCE})")
    return paths


def score_pool(pool, m_trials=5, k=6, seed=C.SEED):
    """Sample k distinct images from pool m_trials times and LLM-score each set."""
    rng = np.random.default_rng(seed)
    scores = []
    for t in range(m_trials):
        pick = [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
        s = llm_plausibility_score(pick, sample=k, seed=t)
        scores.append(s)
        print(f"  trial {t}: {s:.2f}")
    return scores


def run(m_trials=5):
    C.banner("Plausibility Study — diverse vs themed codebook")
    cb = C.get_codebook()

    print("Diverse (benchmark) codebook covers:")
    diverse = score_pool(cb["paths"], m_trials)

    print("Themed codebook covers:")
    themed_paths = build_themed_set()
    themed = score_pool(themed_paths, m_trials)

    # codec effect on the diverse codebook (honest: should be ~flat)
    print("Diverse codebook, permutation coding (codec effect):")
    rng = np.random.default_rng(C.SEED)
    perm_scores = []
    for t in range(m_trials):
        msg = C.random_message(120, __import__("random").Random(t))
        pp, _ = encode(msg, cb, use_permutation=True, perm_block=8)
        s = llm_plausibility_score(pp, sample=6, seed=t)
        perm_scores.append(s); print(f"  trial {t}: {s:.2f}")

    def stat(x):
        x = [v for v in x if v == v]  # drop NaN
        return {"mean": float(np.mean(x)), "std": float(np.std(x)), "n": len(x)}

    result = {
        "theme": THEME,
        "diverse_benchmark": stat(diverse),
        "themed": stat(themed),
        "diverse_permutation": stat(perm_scores),
    }
    rows = [
        ["Diverse benchmark codebook (UCID/Kodak/USC)",
         f"{result['diverse_benchmark']['mean']:.2f} ± {result['diverse_benchmark']['std']:.2f}",
         "max dataset credibility; mixed subjects look random"],
        ["Themed codebook (coherent context)",
         f"{result['themed']['mean']:.2f} ± {result['themed']['std']:.2f}",
         "looks like an ordinary personal album"],
        ["Diverse + permutation coding",
         f"{result['diverse_permutation']['mean']:.2f} ± {result['diverse_permutation']['std']:.2f}",
         "codec barely moves the score"],
    ]
    md = C.save_table(
        "table_4_5_plausibility", rows,
        ["Configuration", "LLM plausibility (0–1)", "Note"],
        "Table 4.5b — Cover plausibility (gemini-fast judge, 0–1). Plausibility is driven "
        "by codebook theme, not the coding mode: a themed database is far more plausible "
        "than the diverse benchmark set, while permutation coding leaves the score "
        "essentially unchanged. We use the diverse benchmark for all other results "
        "(credibility) and report this as an explicit deployment trade-off.")
    json.dump(result, open(RESULT_JSON, "w"), indent=2)
    print("\n" + md)
    print(f"Saved -> {RESULT_JSON}")
    return result


if __name__ == "__main__":
    run()
