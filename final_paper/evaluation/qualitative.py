"""Section 4.2 — Qualitative Results (visual proof).

Figure Set 1: original message -> generated image sequence (short/medium/long).
Figure Set 2: image sequence -> reconstructed message (exact match).
Figure Set 3: failure cases (extreme degradation, codebook mismatch).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import _common as C
from encoder.encode import encode
from decoder.decode import decode


def _seq_row(ax_row, paths, indices, max_imgs):
    show = paths[:max_imgs]
    for k in range(len(ax_row)):
        ax_row[k].axis("off")
    for k, p in enumerate(show):
        ax_row[k].imshow(Image.open(p).convert("RGB"))
        ax_row[k].set_title(f"#{indices[k]}", fontsize=8)


def figure_set1(cb):
    examples = [("Short", "HELLO"),
                ("Medium", "Coverless steganography via CLIP semantic hashing."),
                ("Long", "Meet at the old harbour at midnight; bring the documents and tell no one. " * 2)]
    cols = 10
    fig, axes = plt.subplots(len(examples), cols, figsize=(2 * cols, 2.4 * len(examples)))
    for r, (label, msg) in enumerate(examples):
        paths, meta = encode(msg, cb)
        _seq_row(axes[r], paths, meta["indices"], cols)
        axes[r][0].set_ylabel(label, fontsize=11)
        axes[r][0].axis("on"); axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        fig.text(0.5, 1 - (r + 0.93) / len(examples),
                 f'{label}: "{msg[:48]}{"..." if len(msg) > 48 else ""}"  -> {len(paths)} images',
                 ha="center", fontsize=9)
    fig.suptitle("Figure Set 1 — Secret message → generated image sequence "
                 "(no pixel modification; identity & order carry the data)", fontsize=11)
    out = f"{C.FIG_DIR}/fig_4_2_set1_encode.png"
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def figure_set2(cb):
    msg = "Semantic reconstruction is exact."
    paths, meta = encode(msg, cb)
    decoded = decode(paths, cb)
    cols = min(12, len(paths))
    fig, axes = plt.subplots(1, cols, figsize=(1.7 * cols, 2.6))
    _seq_row(axes, paths, meta["indices"], cols)
    fig.suptitle(f'Figure Set 2 — Received sequence ({len(paths)} images) → '
                 f'CLIP NN decode\nOriginal: "{msg}"   |   Reconstructed: "{decoded}"   '
                 f'|   {"EXACT MATCH ✓" if decoded == msg else "MISMATCH ✗"}',
                 fontsize=10)
    out = f"{C.FIG_DIR}/fig_4_2_set2_decode.png"
    fig.tight_layout(rect=[0, 0, 1, 0.82]); fig.savefig(out, dpi=140); plt.close(fig)
    return out


def figure_set3(cb):
    """Failure analysis: push attacks until decoding breaks, and show CRC catches it."""
    rng = __import__("random").Random(C.SEED)
    msgs = [C.random_message(120, rng) for _ in range(20)]

    cases = [
        ("JPEG 10%", lambda im: C.atk_jpeg(im, 10)),
        ("JPEG 2%", lambda im: C.atk_jpeg(im, 2)),
        ("Crop 30%", lambda im: C.atk_crop(im, 0.30)),
        ("Crop 15%", lambda im: C.atk_crop(im, 0.15)),
        ("Resize 10%", lambda im: C.atk_resize(im, 0.10)),
    ]
    names, accs, sers = [], [], []
    for name, fn in cases:
        src = C.attacked_source_embeddings(cb, fn)
        p, mg, t1 = C.source_lookup(src, cb["embeddings"])
        recs = [C.evaluate_message_fast(m, cb, p, mg, t1) for m in msgs]
        agg = C.aggregate(recs, ["exact", "ser"])
        names.append(name); accs.append(100 * agg["exact"]); sers.append(100 * agg["ser"])

    # codebook-mismatch case: decode with a shuffled codebook -> CRC must reject
    shuffled = dict(cb)
    perm = np.roll(np.arange(cb["n_images"]), 1)
    shuffled["embeddings"] = cb["embeddings"][perm]
    paths, _ = encode(msgs[0], cb)
    try:
        decode(paths, shuffled)
        mismatch = "silently wrong"
    except Exception:
        mismatch = "CRC rejected ✓"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, accs, 0.4, label="Reconstruction accuracy (%)", color="tab:green")
    ax.bar(x + 0.2, sers, 0.4, label="Symbol error rate (%)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylim(0, 105); ax.legend(); ax.grid(alpha=0.3, axis="y")
    ax.set_title("Figure Set 3 — Failure cases: accuracy collapses only under extreme "
                 f"degradation.\nCodebook mismatch → {mismatch}")
    out = f"{C.FIG_DIR}/fig_4_2_set3_failures.png"
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)
    return out, {"cases": list(zip(names, accs, sers)), "mismatch": mismatch}


def run():
    C.banner("Section 4.2 — Qualitative Results")
    cb = C.get_codebook()
    f1 = figure_set1(cb)
    f2 = figure_set2(cb)
    f3, fail = figure_set3(cb)
    print(f"  Figure Set 1 (encode)   -> {f1}")
    print(f"  Figure Set 2 (decode)   -> {f2}")
    print(f"  Figure Set 3 (failures) -> {f3}")
    for name, acc, ser in fail["cases"]:
        print(f"    {name:<12} acc={acc:6.2f}%  SER={ser:5.2f}%")
    print(f"  Codebook mismatch: {fail['mismatch']}")
    return {"figures": [f1, f2, f3], "failures": fail}


if __name__ == "__main__":
    run()
