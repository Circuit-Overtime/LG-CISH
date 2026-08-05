"""Section 4.1 — Experimental Setup.

Documents and prints the full experimental configuration, codebook statistics
(pairwise CLIP separation), and produces a figure of a sample secret message and
its encoded image sequence.
"""

import os
import platform

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import _common as C
from config import (CLIP_MODEL, EMBEDDING_DIM, MIN_CLIP_DISTANCE, AES_KEY_SIZE,
                    DATASET_NAME, get_device)
from encoder.encode import encode


def _ram_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 1)
    except Exception:
        return None


def _gpu_name():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "CPU only"


def codebook_separation(cb):
    """Off-diagonal pairwise cosine similarity stats of the codebook."""
    M = cb["embeddings"]
    S = M @ M.T
    n = S.shape[0]
    off = S[~np.eye(n, dtype=bool)]
    return {"min": float(off.min()), "max": float(off.max()),
            "mean": float(off.mean()), "margin": float(1.0 - off.max())}


def figure_message_to_sequence(cb, message="HELLO", max_imgs=16, fname="fig_setup_sequence.png"):
    paths, meta = encode(message, cb)
    show = paths[:max_imgs]
    cols = min(8, len(show))
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes:
        ax.axis("off")
    for k, p in enumerate(show):
        axes[k].imshow(Image.open(p).convert("RGB"))
        axes[k].set_title(f"#{meta['indices'][k]}", fontsize=9)
        axes[k].axis("off")
    fig.suptitle(f'Secret message "{message}"  ->  {len(paths)} images '
                 f'(showing first {len(show)}). The identity & order of images encode the data.',
                 fontsize=10)
    out = os.path.join(C.FIG_DIR, fname)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def run():
    C.banner("Section 4.1 — Experimental Setup")
    cb = C.get_codebook()
    sep = codebook_separation(cb)

    cfg_rows = [
        ["Dataset", DATASET_NAME],
        ["Codebook images (N)", cb["n_images"]],
        ["Capacity", f"{cb['bits_per_image']:.3f} bits/image (base-{cb['n_images']})"],
        ["CLIP model", f"{CLIP_MODEL} (dim {EMBEDDING_DIM})"],
        ["LLM (plausibility / aliases)", "gemini-fast (Gemini 2.5 Flash Lite, vision) + flux (image gen)"],
        ["Coding modes", f"base-N ({cb['bits_per_image']:.3f} b/img) or permutation (no-repeat, distinct images)"],
        ["Min CLIP separation threshold", MIN_CLIP_DISTANCE],
        ["Codebook pairwise sim (min/mean/max)",
         f"{sep['min']:.3f} / {sep['mean']:.3f} / {sep['max']:.3f}"],
        ["Decoding margin (1 - max sim)", f"{sep['margin']:.3f}"],
        ["Encryption", f"AES-{AES_KEY_SIZE*8}-CBC (optional)"],
        ["Integrity", "CRC-32"],
        ["Compression", "zlib (optional)"],
        ["Device", get_device()],
        ["GPU", _gpu_name()],
        ["CPU", f"{platform.processor() or platform.machine()} x{os.cpu_count()}"],
        ["RAM (GB)", _ram_gb()],
        ["Python", platform.python_version()],
    ]
    for k, v in cfg_rows:
        print(f"  {k:<40} {v}")

    md = C.save_table("table_4_1_setup", cfg_rows, ["Parameter", "Value"],
                      "Table 4.1 — Experimental configuration.")
    fig = figure_message_to_sequence(cb)
    print(f"\nSaved setup figure -> {fig}")
    return {"config": dict(cfg_rows), "separation": sep, "table_md": md, "figure": fig}


if __name__ == "__main__":
    run()
