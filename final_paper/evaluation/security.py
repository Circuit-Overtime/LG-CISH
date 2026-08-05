"""Section 4.5 — Security & Steganalysis Resistance.

  * Steganalysis: a chi-square statistical detector tries to separate cover from
    stego. For LSB/DCT the stego images differ statistically (high detection);
    for the proposed coverless method the 'stego' images are bit-identical natural
    images, so the detector operates at chance (~50%).
  * Keyspace analysis: codebook orderings x AES-256.
  * CRC-32 integrity: bit-flip detection rate.
"""

import math
import random

import numpy as np
from PIL import Image

import _common as C
import baselines as B
from crypto.aes_layer import wrap, unwrap, generate_key
from bitstream.converter import compress, message_to_bytes


def patch_pool(cb, patch=256, per_image=8):
    """Carve natural-image patches from the codebook images to form a sample pool."""
    pool = []
    for p in cb["paths"]:
        arr = np.asarray(Image.open(p).convert("RGB"))
        h, w, _ = arr.shape
        count = 0
        for i in range(0, h - patch, patch):
            for j in range(0, w - patch, patch):
                if count >= per_image:
                    break
                pool.append(arr[i:i + patch, j:j + patch].copy())
                count += 1
    return pool


def steganalysis_table(cb):
    rng = random.Random(C.SEED)
    cover = patch_pool(cb)
    half = len(cover) // 2
    cover_set = cover[:half]
    base_set = cover[half:]  # held-out images turned into stego

    # LSB stego
    lsb_stego = []
    for arr in base_set:
        bits = "".join(rng.choice("01") for _ in range(arr.size))
        lsb_stego.append(B.lsb_embed(arr, bits))
    # DCT stego
    dct_stego = []
    for arr in base_set:
        nb = B.dct_capacity_bits(arr.shape)
        bits = "".join(rng.choice("01") for _ in range(nb))
        dct_stego.append(B.dct_embed(arr, bits))
    # Proposed: coverless -> the transmitted images are unmodified naturals
    proposed_stego = [a.copy() for a in base_set]

    rows = []
    detection = {}
    for label, stego in [("LSB", lsb_stego), ("DCT-LSB", dct_stego),
                         ("Proposed LG-CISH", proposed_stego)]:
        bal, tpr, tnr = B.detection_accuracy(cover_set, stego)
        rows.append([label, f"{bal:.1f}", f"{100*tpr:.1f}", f"{100*tnr:.1f}"])
        detection[label] = bal
        print(f"  {label:<18} detection={bal:5.1f}%  (TPR={100*tpr:.1f}%, TNR={100*tnr:.1f}%)")

    md = C.save_table(
        "table_4_5_steganalysis", rows,
        ["Method", "Detection Acc (%)", "TPR (%)", "TNR (%)"],
        "Table 4.5 — Chi-square steganalysis detection. ~50%% means the detector "
        "cannot do better than guessing (undetectable). N=%d patches." % len(cover_set))
    return md, detection


def keyspace_analysis(cb):
    n = cb["n_images"]
    log2_orderings = sum(math.log2(i) for i in range(1, n + 1))  # log2(N!)
    total_bits = log2_orderings + 256
    lines = [
        f"  Codebook image orderings : {n}! = {math.factorial(n):,}  (~2^{log2_orderings:.1f})",
        f"  AES-256 keyspace         : 2^256",
        f"  Combined keyspace        : {n}! x 2^256  (~2^{total_bits:.1f})",
        f"  Note: the image database itself is a shared secret (coverless key).",
    ]
    print("\n".join(lines))
    rows = [
        ["Codebook orderings", f"{n}! ≈ 2^{log2_orderings:.1f}"],
        ["AES-256 key", "2^256"],
        ["Combined", f"≈ 2^{total_bits:.1f}"],
    ]
    C.save_table("table_4_5_keyspace", rows, ["Component", "Size"],
                 "Table 4.5b — Keyspace analysis.")
    return total_bits


def crc_integrity(trials=500):
    """Flip random bits in the framed stream; measure CRC detection rate."""
    rng = random.Random(C.SEED)
    key = generate_key()
    caught = 0
    for _ in range(trials):
        msg = C.random_message(rng.randint(20, 200), rng)
        framed = bytearray(wrap(compress(message_to_bytes(msg)), key))
        # flip 1..5 random bits
        for _ in range(rng.randint(1, 5)):
            bidx = rng.randrange(len(framed) * 8)
            framed[bidx // 8] ^= (1 << (bidx % 8))
        try:
            unwrap(bytes(framed), key)
        except Exception:
            caught += 1
    rate = 100.0 * caught / trials
    print(f"  CRC-32 tamper detection: {rate:.2f}% ({caught}/{trials})")
    return rate


def run():
    C.banner("Section 4.5 — Security & Steganalysis Resistance")
    cb = C.get_codebook()
    steg_md, detection = steganalysis_table(cb)
    print(steg_md)
    ks = keyspace_analysis(cb)
    crc = crc_integrity()
    return {"steganalysis": steg_md, "detection": detection,
            "keyspace_bits": ks, "crc_detection": crc}


if __name__ == "__main__":
    run()
