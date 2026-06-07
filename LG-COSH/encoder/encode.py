"""Encoder — Message -> ordered image sequence.

The lossless chain (sender side):

    message (str)
      -> UTF-8 bytes
      -> [optional] zlib compress
      -> wrap: append CRC-32, [optional] AES-256-CBC encrypt
      -> base-N positional coding: bytes -> big int -> base-N digits
      -> each digit selects a codebook image
      -> ordered list of image paths

Every arrow is deterministic and reversible. The images are never modified —
the *identity and order* of the sequence carry the data (coverless).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitstream.converter import (
    message_to_bytes,
    compress,
    bytes_to_indices,
    bytes_to_perm_indices,
)
from crypto.aes_layer import wrap
from codebook.builder import load_codebook
from config import CODEBOOK_PATH


def encode(
    message: str,
    codebook: dict | str = CODEBOOK_PATH,
    key: bytes | None = None,
    use_compression: bool = True,
    use_permutation: bool = False,
    perm_block: int | None = None,
    use_aliases: bool = False,
    chooser=None,
) -> tuple[list[str], dict]:
    """Encode a message into an ordered list of codebook image paths.

    Args:
        message: the secret message (any UTF-8 string).
        codebook: a loaded codebook dict, or a path to codebook.npz.
        key: optional 32-byte AES-256 key. If None, no encryption (CRC only).
        use_compression: zlib-compress the payload before coding.
        use_permutation: if True, use distinct-image permutation (Lehmer) coding
            instead of base-N — no image repeats within each block (more plausible
            cover) at a small capacity cost.
        perm_block: images per permutation block (default: the full codebook N).

    Returns:
        (image_paths, metadata) where image_paths is the ordered cover sequence
        and metadata records the protocol parameters needed to decode.
    """
    cb = load_codebook(codebook) if isinstance(codebook, str) else codebook
    n = cb["n_images"]
    paths = cb["paths"]

    # 1. message -> bytes
    raw = message_to_bytes(message)

    # 2. optional lossless compression
    payload = compress(raw) if use_compression else raw

    # 3. integrity (CRC-32) + optional confidentiality (AES-256-CBC)
    framed = wrap(payload, key)

    # 4. positional coding -> ordered codebook indices
    if use_permutation:
        indices = bytes_to_perm_indices(framed, n, perm_block)
    else:
        indices = bytes_to_indices(framed, n)

    # 5. indices -> image paths. With aliases on, pick a plausible candidate per
    #    slot (decoder is unaffected — every candidate maps to the same slot).
    if use_aliases:
        from codebook.aliases import load_aliases, choose_paths
        aliases = cb.get("aliases") if isinstance(cb, dict) else None
        if aliases is None:
            aliases = load_aliases()
        image_paths = choose_paths(indices, paths, aliases, chooser=chooser)
    else:
        image_paths = [paths[i] for i in indices]

    metadata = {
        "n_images": n,
        "bits_per_image": cb["bits_per_image"],
        "num_images": len(image_paths),
        "indices": indices,
        "encrypted": key is not None,
        "compressed": use_compression,
        "permutation": use_permutation,
        "perm_block": (n if perm_block is None else perm_block) if use_permutation else None,
        "aliased": use_aliases,
        "payload_bits": len(framed) * 8,
    }
    return image_paths, metadata


if __name__ == "__main__":
    paths, meta = encode("HELLO WORLD")
    print(f"Encoded into {len(paths)} images:")
    for i, p in enumerate(paths):
        print(f"  [{i}] idx={meta['indices'][i]}  {os.path.basename(p)}")
    print(f"\nMetadata: {meta}")
