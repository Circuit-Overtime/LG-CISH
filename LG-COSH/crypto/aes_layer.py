"""AES-256-CBC encryption/decryption and CRC-32 integrity checking.

The crypto layer is optional — it wraps the raw message bytes before
they enter the bitstream pipeline. It adds two things:
  1. Confidentiality: AES-256-CBC encryption (even with the codebook, message is unreadable)
  2. Integrity: CRC-32 detects if any bit was corrupted during transmission
"""

import os
import struct
import zlib

from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad


def generate_key() -> bytes:
    """Generate a random 256-bit AES key."""
    return os.urandom(32)


def encrypt(plaintext: bytes, key: bytes) -> bytes:
    """AES-256-CBC encrypt. Random IV is prepended to the ciphertext.

    Output format: [16-byte IV] + [ciphertext]
    """
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ciphertext = cipher.encrypt(pad(plaintext, AES.block_size))
    return iv + ciphertext


def decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """AES-256-CBC decrypt. Expects IV prepended to ciphertext."""
    iv = ciphertext[:16]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    plaintext = unpad(cipher.decrypt(ciphertext[16:]), AES.block_size)
    return plaintext


def compute_crc(data: bytes) -> bytes:
    """Compute CRC-32 of data. Returns 4 bytes (big-endian)."""
    checksum = zlib.crc32(data) & 0xFFFFFFFF
    return struct.pack(">I", checksum)


def verify_crc(data: bytes, crc_bytes: bytes) -> bool:
    """Verify CRC-32. Returns True if data matches the checksum."""
    return compute_crc(data) == crc_bytes


def wrap(plaintext: bytes, key: bytes = None) -> bytes:
    """Full wrap: append CRC, then optionally encrypt.

    Output: [encrypted?]([plaintext] + [4-byte CRC])
    """
    crc = compute_crc(plaintext)
    payload = plaintext + crc

    if key is not None:
        payload = encrypt(payload, key)

    return payload


def unwrap(payload: bytes, key: bytes = None) -> bytes:
    """Full unwrap: optionally decrypt, then verify CRC.

    Raises ValueError if CRC check fails.
    """
    if key is not None:
        payload = decrypt(payload, key)

    data = payload[:-4]
    crc = payload[-4:]

    if not verify_crc(data, crc):
        raise ValueError("CRC check failed — message corrupted during transmission")

    return data
