"""Section 4.3 — Quantitative Evaluation.

Produces:
  * Table: Reconstruction Accuracy (per message-length bucket: accuracy %, BER,
    hash-matching accuracy, mean #images).
  * Table: Payload Capacity Comparison (LSB / DCT / DWT-DCT / proposed).
  * Table: Computational Time (encode / decode / CLIP embed / lookup).
  * Retrieval precision/recall of the CLIP nearest-neighbour matcher.
"""

import time
import numpy as np
from PIL import Image

import _common as C
from encoder.encode import encode
from decoder.decode import decode, recover_indices
import baselines as B


PER_BUCKET = 40


def reconstruction_table(cb):
    bank = C.message_bank(C.DEFAULT_BUCKETS, PER_BUCKET)
    # clean channel: received images ARE the codebook images -> self lookup
    src_pred, src_margin, src_top1 = C.source_lookup(cb["embeddings"], cb["embeddings"])
    rows = []
    detail = {}
    for name, msgs in bank.items():
        recs = [C.evaluate_message_fast(m, cb, src_pred, src_margin, src_top1) for m in msgs]
        agg = C.aggregate(recs, ["exact", "ber", "ser", "n_images", "mean_margin"])
        acc = 100.0 * agg["exact"]
        hash_acc = 100.0 * (1.0 - agg["ser"])
        rows.append([name, len(msgs), f"{acc:.2f}", f"{agg['ber']:.2e}",
                     f"{hash_acc:.2f}", f"{agg['n_images']:.1f}"])
        detail[name] = agg
    md = C.save_table(
        "table_4_3_reconstruction", rows,
        ["Message Length", "Messages", "Accuracy (%)", "BER", "Hash Match (%)", "Avg Images"],
        "Table 4.3a — Reconstruction accuracy by message length (clean channel).")
    return md, detail


def capacity_table(cb):
    # use one codebook image as the reference cover for the pixel-based baselines
    cover = np.asarray(Image.open(cb["paths"][0]).convert("RGB"))
    h, w, _ = cover.shape
    px = h * w
    bpi_proposed = cb["bits_per_image"]
    rows = [
        ["LSB (spatial)", f"{B.lsb_capacity_bits(cover.shape):,}", "≈3.0", "Yes (high)", "51.1"],
        ["DCT-LSB", f"{B.dct_capacity_bits(cover.shape):,}", "≈0.016", "Yes (moderate)", "38–42"],
        ["DWT-DCT [cited]", f"~{px//64:,}", "≈0.015", "Yes (low)", "40–44"],
        ["Proposed LG-CISH", f"{bpi_proposed:.3f}/image", f"{bpi_proposed:.3f}", "None (coverless)", "∞"],
    ]
    md = C.save_table(
        "table_4_3_capacity", rows,
        ["Method", "Capacity (bits)", "bits/pixel", "Pixel Distortion", "PSNR (dB)"],
        "Table 4.3b — Payload capacity vs. distortion. The proposed method modifies "
        "no pixels (infinite PSNR) at the cost of lower raw capacity.")
    return md


def timing_table(cb):
    msg = C.random_message(200, __import__("random").Random(0))
    # encode
    t0 = time.perf_counter()
    for _ in range(20):
        paths, meta = encode(msg, cb)
    t_enc = (time.perf_counter() - t0) / 20 * 1000

    # CLIP embed (single image) + full recover
    from clip_engine.embedder import embed_image
    t0 = time.perf_counter()
    for _ in range(10):
        embed_image(cb["paths"][0])
    t_embed = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(5):
        recover_indices(paths, cb)
    t_recover = (time.perf_counter() - t0) / 5 * 1000

    t0 = time.perf_counter()
    for _ in range(5):
        decode(paths, cb)
    t_dec = (time.perf_counter() - t0) / 5 * 1000

    n_imgs = len(paths)
    rows = [
        ["Full encode (200-char msg)", f"{t_enc:.2f}"],
        ["CLIP embedding (per image)", f"{t_embed:.2f}"],
        [f"Index recovery ({n_imgs} images)", f"{t_recover:.2f}"],
        [f"Full decode ({n_imgs} images)", f"{t_dec:.2f}"],
        ["Decode per image (amortised)", f"{t_dec / max(1,n_imgs):.2f}"],
    ]
    md = C.save_table(
        "table_4_3_timing", rows, ["Stage", "Time (ms)"],
        "Table 4.3c — Computational time (mean over repeated runs).")
    return md


def retrieval_pr(cb):
    """CLIP nearest-neighbour retrieval precision/recall on the codebook itself."""
    pred, _, _ = C.source_lookup(cb["embeddings"], cb["embeddings"])
    truth = np.arange(cb["n_images"])
    correct = int(np.sum(pred == truth))
    precision = recall = correct / cb["n_images"]
    print(f"  Retrieval top-1 precision/recall: {100*precision:.2f}% "
          f"({correct}/{cb['n_images']})")
    return precision, recall


def run():
    C.banner("Section 4.3 — Quantitative Evaluation")
    cb = C.get_codebook()
    rec_md, detail = reconstruction_table(cb)
    print(rec_md)
    cap_md = capacity_table(cb)
    print(cap_md)
    tim_md = timing_table(cb)
    print(tim_md)
    p, r = retrieval_pr(cb)
    return {"reconstruction": rec_md, "capacity": cap_md, "timing": tim_md,
            "detail": detail, "precision": p, "recall": r}


if __name__ == "__main__":
    run()
