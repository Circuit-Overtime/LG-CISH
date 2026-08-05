"""Extra paper figures for the Results section:
  * fig_dataset_collage.png  - top-20 images of the augmented 40-image codebook
  * fig_example_sentence.png - one example sentence -> its encoded image sequence

    ../venv/bin/python evaluation/paper_figs.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C
from encoder.encode import encode

FIG = C.FIG_DIR


def dataset_collage(top=20, cols=5):
    cb = C.get_codebook()
    paths = cb["paths"][:top]
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.05 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k, p in enumerate(paths):
        axes[k].imshow(Image.open(str(p)).convert("RGB"))
        axes[k].set_title(f"#{k}", fontsize=8)
    fig.suptitle(f"Augmented LG-CISH codebook - top {top} of {cb['n_images']} images "
                 f"(UCID / Kodak / USC-SIPI, $512{{\\times}}512$)", fontsize=11)
    out = f"{FIG}/fig_dataset_collage.png"
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out)


def example_sentence(sentence="Meet me at the old harbour at midnight.", show=15, cols=5):
    cb = C.get_codebook()
    paths, meta = encode(sentence, cb)
    idx = meta["indices"]
    n = min(show, len(paths))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.05 * cols, 2.25 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for k in range(n):
        axes[k].imshow(Image.open(str(paths[k])).convert("RGB"))
        axes[k].set_title(f"$c_{{{idx[k]}}}$", fontsize=8)
    fig.suptitle(f'Source text: "{sentence}"\n'
                 f'$\\to$ {len(paths)} unmodified images '
                 f'(first {n} shown; base-40 coding, indices {idx[:n]}$\\ldots$)',
                 fontsize=10)
    out = f"{FIG}/fig_example_sentence.png"
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out, dpi=140); plt.close(fig)
    print("saved", out, "| total images:", len(paths))


if __name__ == "__main__":
    dataset_collage()
    example_sentence()
