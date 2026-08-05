"""Extra paper graphs (run after the main suite):
  * fig_cb_heatmap.png   — 6x6 CLIP pairwise-similarity heatmap of the codebook
  * fig_margin_attacks.png — mean decoding margin per channel attack
  * fig_ablation_bar.png  — image-count + Crop-40 accuracy for ablation variants

    ../venv/bin/python evaluation/extra_figs.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import _common as C


def heatmap():
    cb = C.get_codebook()
    M = cb["embeddings"]
    S = M @ M.T
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    annotate = n <= 12                       # per-cell numbers only read well for small N
    sz = max(6.0, 0.16 * n)                  # grow the canvas with the codebook
    fig, ax = plt.subplots(figsize=(sz + 1.2, sz))
    im = ax.imshow(S, cmap="viridis", vmin=float(off.min()), vmax=1.0)
    if annotate:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                        color="white" if S[i, j] < 0.8 else "black", fontsize=8)
    # sparse, readable ticks (~every 5th index)
    step = 1 if n <= 20 else 5
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("Codebook image index"); ax.set_ylabel("Codebook image index")
    ax.set_title(f"CLIP pairwise cosine similarity ($N={n}$ codebook, "
                 f"max off-diagonal {off.max():.3f} $<\\tau{{=}}0.85$)")
    cb_ = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb_.set_label("cosine similarity")
    out = f"{C.FIG_DIR}/fig_cb_heatmap.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


def margin_per_attack():
    cb = C.get_codebook()
    names, margins = [], []
    for name, fn in C.attack_suite():
        src = C.attacked_source_embeddings(cb, fn)
        _, mg, _ = C.source_lookup(src, cb["embeddings"])
        names.append(name.replace(" (baseline)", "")); margins.append(float(np.mean(mg)))
    order = np.argsort(margins)
    names = [names[i] for i in order]; margins = [margins[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names, margins, color="tab:blue")
    ax.axvline(0, color="red", ls=":", label="decoding threshold (margin = 0)")
    ax.set_xlabel("Mean decoding margin  $\\gamma = s_{(1)}-s_{(2)}$")
    ax.set_title("Decoding margin per channel attack (all > 0 = recoverable)")
    ax.legend(); ax.grid(alpha=0.3, axis="x")
    out = f"{C.FIG_DIR}/fig_margin_attacks.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


def ablation_bar():
    import ablation as A
    cb = C.get_codebook()
    bank = C.message_bank([("m", 50, 200)], 30)["m"]
    a_full = A.avg_images(cb, bank, use_compression=True)
    a_nocomp = A.avg_images(cb, bank, use_compression=False)
    a_fixed = A.avg_images(cb, bank, use_compression=True, chunk_override=cb["chunk_size"])
    clip40 = A.clip_attack_accuracy(cb, bank, lambda im: C.atk_crop(im, 0.65))
    phash40 = A.phash_attack_accuracy(cb, bank, lambda im: C.atk_crop(im, 0.65))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["Base-N\n+zlib", "No\ncompress", "Fixed\nchunk"], [a_full, a_nocomp, a_fixed],
            color=["tab:green", "tab:orange", "tab:red"])
    ax1.set_ylabel("Avg images / message"); ax1.set_title("Coding efficiency (lower is better)")
    for i, v in enumerate([a_full, a_nocomp, a_fixed]):
        ax1.text(i, v + 3, f"{v:.0f}", ha="center")
    ax2.bar(["CLIP\n(proposed)", "pHash\n(ablation)"], [clip40, phash40],
            color=["tab:green", "tab:red"])
    ax2.set_ylabel("Crop-65% accuracy (%)"); ax2.set_ylim(0, 105)
    ax2.set_title("Matcher robustness under Crop-65%")
    for i, v in enumerate([clip40, phash40]):
        ax2.text(i, v + 1, f"{v:.0f}", ha="center")
    out = f"{C.FIG_DIR}/fig_ablation_bar.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


def coding_modes_bar():
    """Capacity (bits/image) for base-N vs. permutation blocks (matches Table 4 / coding modes)."""
    import math
    from bitstream.converter import _perm_modulus
    cb = C.get_codebook()
    n = cb["n_images"]
    labels = ["Base-$N$", "Perm $b{=}8$", "Perm $b{=}16$", f"Perm $b{{=}}{n}$"]
    vals = [math.log2(n)] + [math.log2(_perm_modulus(n, b)) / b for b in (8, 16, n)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color=["tab:green", "tab:blue", "tab:blue", "tab:orange"])
    ax.axhline(math.log2(n), ls=":", color="gray",
               label=f"base-$N$ ceiling $= {math.log2(n):.3f}$")
    ax.set_ylabel("Capacity (bits/image)")
    ax.set_title(f"Coding modes: capacity vs.\\ no-repeat guarantee ($N={n}$)")
    ax.set_ylim(0, math.log2(n) * 1.18)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.04, f"{v:.3f}", ha="center", fontsize=9)
    ax.legend()
    out = f"{C.FIG_DIR}/fig_coding_modes.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


def timing_bar():
    """Per-stage timing (matches Table 5 / timing). Log scale: encode is ~ms, CLIP dominates."""
    import time
    import random
    from encoder.encode import encode
    from decoder.decode import decode
    from clip_engine.embedder import embed_image
    cb = C.get_codebook()
    msg = C.random_message(200, random.Random(0))
    t0 = time.perf_counter()
    for _ in range(20):
        paths, _ = encode(msg, cb)
    t_enc = (time.perf_counter() - t0) / 20 * 1000
    t0 = time.perf_counter()
    for _ in range(10):
        embed_image(cb["paths"][0])
    t_embed = (time.perf_counter() - t0) / 10 * 1000
    t0 = time.perf_counter()
    decode(paths, cb)
    t_dec_img = (time.perf_counter() - t0) / max(1, len(paths)) * 1000
    labels = ["Encode\n(full msg)", "CLIP embed\n(per image)", "Decode\n(per image)"]
    vals = [t_enc, t_embed, t_dec_img]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(labels, vals, color=["tab:green", "tab:blue", "tab:purple"])
    ax.set_yscale("log"); ax.set_ylabel("Time (ms, log scale)")
    ax.set_title(f"Per-stage timing ({len(paths)} images, RTX 3050)")
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v * 1.15, f"{v:.2f} ms", ha="center", fontsize=9)
    out = f"{C.FIG_DIR}/fig_timing.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    heatmap()
    margin_per_attack()
    ablation_bar()
    coding_modes_bar()
    timing_bar()
