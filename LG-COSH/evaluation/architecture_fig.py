"""Generate the LG-CISH system architecture diagram (paper Fig. 1).

A clean block diagram of the encode -> channel -> decode pipeline, saved to
evaluation/figures/fig_architecture.png.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

import _common as C


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                linewidth=1.2, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5, wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=12, linewidth=1.1, color="#333"))


def run():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.2); ax.axis("off")

    enc = "#dbeafe"; cod = "#dcfce7"; chan = "#fee2e2"; dec = "#fef9c3"

    # Encoder row
    ax.text(0.1, 4.9, "SENDER", fontsize=10, fontweight="bold", color="#1e3a8a")
    box(ax, 0.1, 3.9, 1.5, 0.8, "Secret\nmessage", enc)
    box(ax, 1.9, 3.9, 1.6, 0.8, "zlib +\nAES-256 +\nCRC-32", enc)
    box(ax, 3.8, 3.9, 1.8, 0.8, "Base-N coding\nbytes→∫→digits", cod)
    box(ax, 5.9, 3.9, 1.8, 0.8, "Codebook lookup\nindex→image", cod)
    box(ax, 8.0, 3.9, 1.6, 0.8, "Image\nsequence", enc)
    for x in (1.6, 3.5, 5.6, 7.7):
        arrow(ax, x, 4.3, x + 0.3, 4.3)

    # Channel
    box(ax, 8.0, 2.5, 1.6, 0.8, "Channel\n(JPEG, resize,\nnoise, crop)", chan)
    arrow(ax, 8.8, 3.9, 8.8, 3.3)

    # Decoder row
    ax.text(0.1, 2.05, "RECEIVER", fontsize=10, fontweight="bold", color="#854d0e")
    box(ax, 8.0, 1.0, 1.6, 0.8, "Received\nimages", dec)
    box(ax, 5.9, 1.0, 1.8, 0.8, "CLIP embed +\nNN match→index", cod)
    box(ax, 3.8, 1.0, 1.8, 0.8, "Base-N decode\ndigits→∫→bytes", cod)
    box(ax, 1.9, 1.0, 1.6, 0.8, "CRC verify +\nAES + unzip", dec)
    box(ax, 0.1, 1.0, 1.5, 0.8, "Recovered\nmessage", dec)
    arrow(ax, 8.8, 2.5, 8.8, 1.8)
    for x in (8.0, 5.9, 3.8, 1.9):
        arrow(ax, x, 1.4, x - 0.3, 1.4)

    # Shared setup note
    box(ax, 3.6, 2.45, 3.6, 0.7, "Shared secret: image database + CLIP model + codebook (N images, log₂N bits each)", "#f3e8ff")

    fig.suptitle("Fig. 1  LG-CISH coverless pipeline: the identity and order of unmodified "
                 "images carry the message; CLIP nearest-neighbour recovers indices after channel distortion.",
                 fontsize=9.5, y=0.98)
    out = os.path.join(C.FIG_DIR, "fig_architecture.png")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved architecture figure -> {out}")
    return out


if __name__ == "__main__":
    run()
