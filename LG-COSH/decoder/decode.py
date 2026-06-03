"""Decoder — ordered image sequence -> Message.

The lossless chain (receiver side), exact inverse of the encoder:

    received images
      -> CLIP embedding per image
      -> nearest-neighbour against the codebook -> recovered index per image
      -> base-N positional decoding: digits -> big int -> bytes
      -> unwrap: [optional] AES decrypt, verify CRC-32
      -> [optional] zlib decompress
      -> UTF-8 decode -> message (str)

CLIP is the reliability layer: even after JPEG/resize/noise in the channel, the
embedding of each received image stays closest to the correct codebook entry, so
the integer indices are recovered exactly and the message is reconstructed losslessly.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitstream.converter import indices_to_bytes, decompress
from crypto.aes_layer import unwrap
from clip_engine.embedder import embed_images_batch, embed_image
from codebook.builder import load_codebook
from config import CODEBOOK_PATH


def recover_indices(
    image_paths: list[str],
    codebook: dict | str = CODEBOOK_PATH,
    return_margins: bool = False,
):
    """Map each received image to its codebook index via CLIP nearest-neighbour.

    Args:
        image_paths: ordered list of received image file paths.
        codebook: loaded codebook dict, or path to codebook.npz.
        return_margins: if True, also return per-image (top1_sim, margin) where
            margin = top1_sim - top2_sim (the CLIP distance margin / decoding headroom).

    Returns:
        indices, or (indices, margins) if return_margins.
    """
    cb = load_codebook(codebook) if isinstance(codebook, str) else codebook
    matrix = cb["embeddings"]  # (N, D), normalized

    query = embed_images_batch(image_paths)  # (M, D), normalized
    sims = query @ matrix.T  # (M, N) cosine similarities

    indices = np.argmax(sims, axis=1).astype(int).tolist()

    if not return_margins:
        return indices

    margins = []
    for row in sims:
        order = np.sort(row)[::-1]
        top1 = float(order[0])
        top2 = float(order[1]) if len(order) > 1 else 0.0
        margins.append((top1, top1 - top2))
    return indices, margins


def decode(
    image_paths: list[str],
    codebook: dict | str = CODEBOOK_PATH,
    key: bytes | None = None,
    use_compression: bool = True,
) -> str:
    """Decode an ordered image sequence back to the original message.

    Args:
        image_paths: ordered list of received image file paths.
        codebook: loaded codebook dict, or path to codebook.npz.
        key: optional AES-256 key (must match the encoder's key).
        use_compression: must match the encoder's setting.

    Returns:
        The reconstructed message string.

    Raises:
        ValueError: if the CRC-32 integrity check fails (corruption / wrong key).
    """
    cb = load_codebook(codebook) if isinstance(codebook, str) else codebook
    n = cb["n_images"]

    # 1. images -> CLIP nearest-neighbour indices
    indices = recover_indices(image_paths, cb)

    # 2. base-N positional decoding -> framed bytes
    framed = indices_to_bytes(indices, n)

    # 3. AES decrypt (optional) + CRC-32 verify
    payload = unwrap(framed, key)

    # 4. optional decompression
    raw = decompress(payload) if use_compression else payload

    # 5. bytes -> message
    return raw.decode("utf-8")


if __name__ == "__main__":
    # Round-trip smoke test against the live codebook.
    from encoder.encode import encode

    msg = "HELLO WORLD"
    paths, meta = encode(msg)
    print(f"Encoded '{msg}' -> {len(paths)} images")
    out = decode(paths)
    print(f"Decoded -> '{out}'")
    print("ROUND-TRIP", "OK" if out == msg else "FAILED")
