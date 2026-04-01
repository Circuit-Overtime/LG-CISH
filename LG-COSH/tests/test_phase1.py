"""Phase 1 round-trip tests: bitstream converter + crypto layer."""

import sys
import os

# Force UTF-8 on Windows console
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bitstream.converter import (
    message_to_bytes, bytes_to_message,
    bytes_to_bits, bits_to_bytes,
    bits_to_chunks, chunks_to_bits,
    compress, decompress,
    encode_message, decode_chunks,
)
from crypto.aes_layer import (
    generate_key, encrypt, decrypt,
    compute_crc, verify_crc,
    wrap, unwrap,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        failed += 1


# === BITSTREAM TESTS ===
print("\n=== Bitstream Converter ===\n")

# bytes round-trip
msg = "HELLO"
test("message -> bytes -> message",
     bytes_to_message(message_to_bytes(msg)) == msg)

# bits round-trip
data = b"HELLO"
test("bytes -> bits -> bytes",
     bits_to_bytes(bytes_to_bits(data)) == data)

# chunks round-trip (chunk_size=10, simulating 1024-image database)
bits = bytes_to_bits(b"HELLO WORLD")
chunks, padding = bits_to_chunks(bits, 10)
recovered_bits = chunks_to_bits(chunks, 10, padding)
test("bits -> chunks -> bits (chunk_size=10)",
     recovered_bits == bits)

# chunks round-trip (chunk_size=8)
chunks8, pad8 = bits_to_chunks(bits, 8)
test("bits -> chunks -> bits (chunk_size=8)",
     chunks_to_bits(chunks8, 8, pad8) == bits)

# compression round-trip
raw = b"AAAAAAAAAA" * 100
test("compress -> decompress",
     decompress(compress(raw)) == raw)

# full pipeline round-trip
for test_msg in ["HELLO", "HELLO WORLD", "こんにちは", "🔐secret🔐", "", "A" * 500]:
    label = repr(test_msg[:30]) + ("..." if len(test_msg) > 30 else "")
    try:
        chunks, meta = encode_message(test_msg, chunk_size=10)
        recovered = decode_chunks(chunks, meta)
        test(f"full pipeline: {label}", recovered == test_msg)
    except Exception as e:
        test(f"full pipeline: {label}", False)
        print(f"         Error: {e}")

# full pipeline without compression
chunks_nc, meta_nc = encode_message("TEST", chunk_size=10, use_compression=False)
test("full pipeline (no compression)",
     decode_chunks(chunks_nc, meta_nc) == "TEST")

# chunk index range check
chunks_r, _ = bits_to_chunks("1" * 20, 10)
test("chunk indices in valid range",
     all(0 <= c < 1024 for c in chunks_r))


# === CRYPTO TESTS ===
print("\n=== Crypto (AES + CRC) ===\n")

key = generate_key()
test("key is 32 bytes", len(key) == 32)

# AES round-trip
plaintext = b"secret message here"
ciphertext = encrypt(plaintext, key)
test("AES encrypt -> decrypt",
     decrypt(ciphertext, key) == plaintext)

# AES with different keys fails
key2 = generate_key()
try:
    result = decrypt(ciphertext, key2)
    test("AES wrong key fails", False)
except Exception:
    test("AES wrong key fails", True)

# CRC round-trip
data = b"check this data"
crc = compute_crc(data)
test("CRC-32 verify (correct)", verify_crc(data, crc))
test("CRC-32 verify (tampered)", not verify_crc(data + b"x", crc))

# wrap/unwrap round-trip (no encryption)
plain = b"hello world"
wrapped = wrap(plain)
test("wrap -> unwrap (no key)", unwrap(wrapped) == plain)

# wrap/unwrap round-trip (with encryption)
wrapped_enc = wrap(plain, key)
test("wrap -> unwrap (with key)", unwrap(wrapped_enc, key) == plain)

# unwrap detects corruption
corrupted = wrapped[:-1] + bytes([(wrapped[-1] + 1) % 256])
try:
    unwrap(corrupted)
    test("unwrap detects corruption", False)
except ValueError:
    test("unwrap detects corruption", True)


# === SUMMARY ===
print(f"\n{'=' * 40}")
print(f"  {passed} passed, {failed} failed")
print(f"{'=' * 40}\n")

sys.exit(0 if failed == 0 else 1)
