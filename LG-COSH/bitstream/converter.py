"""Lossless message <-> bits <-> chunks conversion.

The entire encoding chain:
    message (str) -> bytes (UTF-8) -> bits (binary string) -> chunks (list[int])
And the reverse for decoding. Every step is deterministic and reversible.
"""

import math
import zlib


def message_to_bytes(message: str) -> bytes:
    """Encode a string to raw UTF-8 bytes."""
    return message.encode("utf-8")


def bytes_to_message(data: bytes) -> str:
    """Decode raw UTF-8 bytes back to a string."""
    return data.decode("utf-8")


def bytes_to_bits(data: bytes) -> str:
    """Convert raw bytes to a binary string. Each byte -> 8 bits."""
    return "".join(format(byte, "08b") for byte in data)


def bits_to_bytes(bits: str) -> bytes:
    """Convert a binary string back to raw bytes. Length must be multiple of 8."""
    assert len(bits) % 8 == 0, f"Bit length {len(bits)} is not a multiple of 8"
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits), 8))


def bits_to_chunks(bits: str, chunk_size: int) -> tuple[list[int], int]:
    """Split a binary string into fixed-size chunks.

    Args:
        bits: binary string (e.g. "010010000100...")
        chunk_size: bits per chunk (= log2 of database size)

    Returns:
        (chunks, padding) where:
        - chunks: list of integer indices, each in range [0, 2^chunk_size)
        - padding: number of zero-padding bits added to the last chunk
    """
    padding = (chunk_size - (len(bits) % chunk_size)) % chunk_size
    padded = bits + "0" * padding

    chunks = []
    for i in range(0, len(padded), chunk_size):
        chunk_bits = padded[i : i + chunk_size]
        chunks.append(int(chunk_bits, 2))

    return chunks, padding


def chunks_to_bits(chunks: list[int], chunk_size: int, padding: int) -> str:
    """Convert chunks back to a binary string, removing padding.

    Args:
        chunks: list of integer indices
        chunk_size: bits per chunk
        padding: number of padding bits to strip from the end

    Returns:
        Original binary string.
    """
    bits = "".join(format(idx, f"0{chunk_size}b") for idx in chunks)
    if padding > 0:
        bits = bits[:-padding]
    return bits


def compress(data: bytes) -> bytes:
    """Lossless compression using zlib."""
    return zlib.compress(data)


def decompress(data: bytes) -> bytes:
    """Decompress zlib-compressed data."""
    return zlib.decompress(data)


# --- Base-N positional coding (the coverless image-sequence channel) ---
#
# With a codebook of N images we treat the whole payload as one big integer and
# write it in radix N. Each base-N digit selects one image, so every image carries
# log2(N) bits (e.g. N=6 -> 2.585 bits/image) instead of floor(log2(N)) = 2 bits.
# The sequence (identity AND order) of images *is* the message. Fully lossless.

def bytes_to_int(data: bytes) -> int:
    """Map raw bytes to a non-negative integer.

    A 0x01 sentinel byte is prepended so that leading zero bytes of `data` are
    preserved through the int round-trip (int.from_bytes would otherwise lose them).
    """
    return int.from_bytes(b"\x01" + data, "big")


def int_to_bytes(value: int) -> bytes:
    """Inverse of bytes_to_int. Strips the 0x01 sentinel byte."""
    length = (value.bit_length() + 7) // 8
    raw = value.to_bytes(length, "big")
    if raw[0] != 1:
        raise ValueError("sentinel byte mismatch — corrupted integer")
    return raw[1:]


def int_to_base_n(value: int, n: int) -> list[int]:
    """Convert a non-negative integer to a list of base-n digits (MSB first).

    The leading digit is always non-zero (no ambiguous leading-zero digits),
    because the 0x01 sentinel guarantees value >= 1.
    """
    if n < 2:
        raise ValueError("base must be >= 2")
    if value == 0:
        return [0]
    digits = []
    while value > 0:
        digits.append(value % n)
        value //= n
    return digits[::-1]


def base_n_to_int(digits: list[int], n: int) -> int:
    """Convert a list of base-n digits (MSB first) back to an integer."""
    value = 0
    for d in digits:
        if d < 0 or d >= n:
            raise ValueError(f"digit {d} out of range for base {n}")
        value = value * n + d
    return value


def bytes_to_indices(data: bytes, n: int) -> list[int]:
    """Full forward map: payload bytes -> ordered codebook indices (base-n digits)."""
    return int_to_base_n(bytes_to_int(data), n)


def indices_to_bytes(indices: list[int], n: int) -> bytes:
    """Full inverse map: ordered codebook indices -> payload bytes."""
    return int_to_bytes(base_n_to_int(indices, n))


# --- Permutation coding (distinct-image / Lehmer factorial-base channel) ---
#
# Base-N coding above already reaches the k*log2(N) ceiling but reuses images
# (a digit may repeat), which makes the cover look unnatural. Permutation coding
# instead emits, per block, B *distinct* images selected from the N-image
# codebook without repetition. A block of B images therefore carries
# log2(P(N,B)) = log2(N!/(N-B)!) bits via a mixed-radix (Lehmer) code: the t-th
# image is identified by its rank among the still-available images (radix N-t).
#
# This trades a little raw capacity (e.g. N=40, B=N -> log2(40!)/40 ~= 3.98
# bits/image vs 5.322 for base-N) for a no-repeat, photo-album-like cover. It is
# fully lossless and bijective; the receiver recovers each image's index by CLIP
# nearest-neighbour exactly as before, then inverts the Lehmer code.

def _perm_block_to_value(block_indices: list[int], n: int) -> int:
    """Lehmer encode a block of distinct codebook indices to an integer in
    [0, n*(n-1)*...*(n-B+1)). `block_indices` must be distinct and in [0, n)."""
    avail = list(range(n))
    val = 0
    for t, idx in enumerate(block_indices):
        r = avail.index(idx)            # rank of idx among remaining images
        val = val * (n - t) + r         # radix at step t is (n - t)
        avail.pop(r)
    return val


def _value_to_perm_block(val: int, n: int, b: int) -> list[int]:
    """Inverse of _perm_block_to_value: integer -> B distinct codebook indices."""
    digits = [0] * b
    for t in range(b - 1, -1, -1):
        radix = n - t
        digits[t] = val % radix
        val //= radix
    avail = list(range(n))
    return [avail.pop(d) for d in digits]


def _perm_modulus(n: int, b: int) -> int:
    """P(n, b) = n!/(n-b)! — the number of distinct ordered B-image blocks."""
    m = 1
    for t in range(b):
        m *= (n - t)
    return m


def _block_mask(i: int, m: int) -> int:
    """Deterministic, shared keystream value in [0, m) for block i.

    Whitens block values so that a small most-significant block value does not
    map to a tell-tale sorted image prefix (the Lehmer leading-zero artifact).
    Invertible mod m, so it preserves losslessness and within-block distinctness.
    """
    import hashlib
    h = hashlib.sha256(f"LG-CISH-perm-whiten-{i}".encode()).digest()
    return int.from_bytes(h, "big") % m


def bytes_to_perm_indices(data: bytes, n: int, block: int | None = None) -> list[int]:
    """Forward map: payload bytes -> ordered codebook indices, no repeats within
    each block of `block` images (default block = n, i.e. full permutations)."""
    b = n if block is None else block
    if not (1 <= b <= n):
        raise ValueError(f"permutation block {b} must be in [1, {n}]")
    m = _perm_modulus(n, b)
    value = bytes_to_int(data)          # >= 1 thanks to the 0x01 sentinel
    block_vals = int_to_base_n(value, m)  # most-significant block first
    indices = []
    for i, bv in enumerate(block_vals):
        bv = (bv + _block_mask(i, m)) % m   # whiten to avoid sorted prefixes
        indices.extend(_value_to_perm_block(bv, n, b))
    return indices


def perm_indices_to_bytes(indices: list[int], n: int, block: int | None = None) -> bytes:
    """Inverse map: ordered codebook indices -> payload bytes."""
    b = n if block is None else block
    if len(indices) % b != 0:
        raise ValueError(f"index count {len(indices)} not a multiple of block {b}")
    m = _perm_modulus(n, b)
    block_vals = []
    for bi, i in enumerate(range(0, len(indices), b)):
        bv = _perm_block_to_value(indices[i:i + b], n)
        block_vals.append((bv - _block_mask(bi, m)) % m)  # un-whiten
    value = base_n_to_int(block_vals, m)
    return int_to_bytes(value)


def get_chunk_size(database_size: int) -> int:
    """Calculate chunk size (bits per image) from database size.

    database_size=1024 -> chunk_size=10  (2^10 = 1024)
    """
    return int(math.log2(database_size))


# --- Full pipeline helpers ---

def encode_message(message: str, chunk_size: int, use_compression: bool = True) -> tuple[list[int], dict]:
    """Full encoding: message -> chunks ready for image mapping.

    Returns:
        (chunks, metadata) where metadata contains everything needed for decoding.
    """
    raw_bytes = message_to_bytes(message)
    if use_compression:
        payload = compress(raw_bytes)
    else:
        payload = raw_bytes

    bits = bytes_to_bits(payload)
    chunks, padding = bits_to_chunks(bits, chunk_size)

    metadata = {
        "padding": padding,
        "chunk_size": chunk_size,
        "compressed": use_compression,
        "num_chunks": len(chunks),
    }
    return chunks, metadata


def decode_chunks(chunks: list[int], metadata: dict) -> str:
    """Full decoding: chunks -> original message.

    Args:
        chunks: list of integer indices recovered from images
        metadata: the metadata dict from encode_message
    """
    bits = chunks_to_bits(chunks, metadata["chunk_size"], metadata["padding"])
    payload = bits_to_bytes(bits)

    if metadata["compressed"]:
        raw_bytes = decompress(payload)
    else:
        raw_bytes = payload

    return bytes_to_message(raw_bytes)
