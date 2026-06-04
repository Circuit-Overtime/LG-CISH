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
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(S, cmap="viridis", vmin=S.min(), vmax=1.0)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{S[i,j]:.2f}", ha="center", va="center",
                    color="white" if S[i, j] < 0.8 else "black", fontsize=8)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xlabel("Codebook image index"); ax.set_ylabel("Codebook image index")
    ax.set_title("CLIP pairwise cosine similarity (codebook separation)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    clip40 = A.clip_attack_accuracy(cb, bank, lambda im: C.atk_crop(im, 0.40))
    phash40 = A.phash_attack_accuracy(cb, bank, lambda im: C.atk_crop(im, 0.40))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(["Base-N\n+zlib", "No\ncompress", "Fixed\nchunk"], [a_full, a_nocomp, a_fixed],
            color=["tab:green", "tab:orange", "tab:red"])
    ax1.set_ylabel("Avg images / message"); ax1.set_title("Coding efficiency (lower is better)")
    for i, v in enumerate([a_full, a_nocomp, a_fixed]):
        ax1.text(i, v + 3, f"{v:.0f}", ha="center")
    ax2.bar(["CLIP\n(proposed)", "pHash\n(ablation)"], [clip40, phash40],
            color=["tab:green", "tab:red"])
    ax2.set_ylabel("Crop-40% accuracy (%)"); ax2.set_ylim(0, 105)
    ax2.set_title("Matcher robustness under Crop-40%")
    for i, v in enumerate([clip40, phash40]):
        ax2.text(i, v + 1, f"{v:.0f}", ha="center")
    out = f"{C.FIG_DIR}/fig_ablation_bar.png"
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    heatmap()
    margin_per_attack()
    ablation_bar()
