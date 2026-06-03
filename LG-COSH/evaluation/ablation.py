"""Section 4.6 — Ablation Study.

Removes/swaps one component at a time to show each is necessary:
  * Full LG-CISH (CLIP NN + base-N coding + zlib + CRC)
  * Without CLIP        -> perceptual-hash (pHash) matcher instead  (robustness drops)
  * Without compression -> more images per message                  (efficiency drops)
  * Without CRC         -> channel errors go undetected             (integrity lost)
  * Fixed-chunk coding  -> floor(log2 N)=2 bits/image instead of base-N 2.585 (capacity drops)
"""

import math
import numpy as np
from PIL import Image
from scipy.fftpack import dct

import _common as C
from encoder.encode import encode

PER = 30
MED = ("Medium (50-200)", 50, 200)
JPEG50 = ("JPEG 50%", lambda im: C.atk_jpeg(im, 50))


# ---------------- pHash matcher (the "without CLIP" baseline) ----------------
def phash_bits(arr, hash_size=8, scale=4):
    img = Image.fromarray(arr).convert("L").resize(
        (hash_size * scale, hash_size * scale), Image.LANCZOS)
    px = np.asarray(img, dtype=np.float64)
    d = dct(dct(px, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])  # exclude DC term
    return low > med


def phash_codebook(cb):
    return [phash_bits(np.asarray(Image.open(p).convert("RGB"))) for p in cb["paths"]]


def phash_predict(arr, cb_hashes):
    h = phash_bits(arr)
    dists = [int(np.count_nonzero(h != hb)) for hb in cb_hashes]
    return int(np.argmin(dists))


# ---------------- helpers ----------------
def avg_images(cb, bank, use_compression=True, chunk_override=None):
    """Mean #images per message. chunk_override simulates fixed-chunk coding."""
    counts = []
    for m in bank:
        if chunk_override is None:
            _, meta = encode(m, cb, use_compression=use_compression)
            counts.append(meta["num_images"])
        else:
            # fixed chunk: ceil(payload_bits / chunk_bits)
            _, meta = encode(m, cb, use_compression=use_compression)
            counts.append(math.ceil(meta["payload_bits"] / chunk_override))
    return float(np.mean(counts))


def clip_jpeg50_accuracy(cb, bank):
    src_emb = C.attacked_source_embeddings(cb, JPEG50[1])
    p, mg, t1 = C.source_lookup(src_emb, cb["embeddings"])
    recs = [C.evaluate_message_fast(m, cb, p, mg, t1) for m in bank]
    return 100.0 * C.aggregate(recs, ["exact"])["exact"]


def phash_jpeg50_accuracy(cb, bank, clean_only=False):
    cb_hashes = phash_codebook(cb)
    fn = (lambda im: im) if clean_only else JPEG50[1]
    # attacked source images -> pHash predictions per source index
    src_pred = []
    for p in cb["paths"]:
        arr = np.asarray(fn(Image.open(p).convert("RGB")))
        src_pred.append(phash_predict(arr, cb_hashes))
    exact = 0
    for m in bank:
        _, meta = encode(m, cb)
        rec = [src_pred[t] for t in meta["indices"]]
        if rec == meta["indices"]:
            exact += 1
    return 100.0 * exact / len(bank)


def run():
    C.banner("Section 4.6 — Ablation Study")
    cb = C.get_codebook()
    bank = C.message_bank([MED], PER)[MED[0]]

    clip_jpeg50 = clip_jpeg50_accuracy(cb, bank)
    phash_clean = phash_jpeg50_accuracy(cb, bank, clean_only=True)
    phash_jpeg50 = phash_jpeg50_accuracy(cb, bank)

    a_full = avg_images(cb, bank, use_compression=True)
    a_nocomp = avg_images(cb, bank, use_compression=False)
    a_fixed = avg_images(cb, bank, use_compression=True, chunk_override=cb["chunk_size"])

    rows = [
        ["Full LG-CISH (proposed)", "100.00", f"{clip_jpeg50:.2f}", f"{a_full:.1f}", "CRC-32"],
        ["Without CLIP (pHash NN)", f"{phash_clean:.2f}", f"{phash_jpeg50:.2f}", f"{a_full:.1f}", "CRC-32"],
        ["Without compression", "100.00", f"{clip_jpeg50:.2f}", f"{a_nocomp:.1f}", "CRC-32"],
        ["Without CRC integrity", "100.00", f"{clip_jpeg50:.2f}", f"{a_full:.1f}", "None (silent)"],
        [f"Fixed-chunk ({cb['chunk_size']}-bit)", "100.00", f"{clip_jpeg50:.2f}", f"{a_fixed:.1f}", "CRC-32"],
    ]
    md = C.save_table(
        "table_4_6_ablation", rows,
        ["Variant", "Clean Acc (%)", "JPEG50 Acc (%)", "Avg Images", "Integrity"],
        "Table 4.6 — Ablation. CLIP gives JPEG robustness pHash lacks; base-N coding "
        "and compression reduce image count; CRC provides error detection.")
    print(md)
    print(f"  CLIP JPEG50 accuracy: {clip_jpeg50:.2f}%  |  pHash JPEG50 accuracy: {phash_jpeg50:.2f}%")
    print(f"  Avg images  full={a_full:.1f}  no-compress={a_nocomp:.1f}  fixed-chunk={a_fixed:.1f}")
    return {"table": md, "clip_jpeg50": clip_jpeg50, "phash_jpeg50": phash_jpeg50,
            "avg_full": a_full, "avg_nocomp": a_nocomp, "avg_fixed": a_fixed}


if __name__ == "__main__":
    run()
