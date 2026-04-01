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
