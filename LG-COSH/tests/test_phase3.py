"""Phase 3 tests: full encode -> decode lossless round-trip.

Requires the codebook to be built first:
    python build_codebook.py

Runs CLIP on the 6-image codebook (GPU if available).
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codebook.builder import load_codebook
from encoder.encode import encode
from decoder.decode import decode, recover_indices
from bitstream.converter import bytes_to_indices, indices_to_bytes
from crypto.aes_layer import generate_key

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


# ---------------------------------------------------------------
# Pure base-N coding round-trip (no CLIP, fast) — the math layer
# ---------------------------------------------------------------
print("\n=== Base-N positional coding (math only) ===")
for n in (2, 6, 16, 256):
    for sample in (b"", b"\x00", b"\x00\x00hi", b"HELLO WORLD", os.urandom(64)):
        idx = bytes_to_indices(sample, n)
        back = indices_to_bytes(idx, n)
        test(f"base-{n} round-trip ({len(sample)}B)", back == sample)
        test(f"base-{n} digits in range", all(0 <= d < n for d in idx))

# ---------------------------------------------------------------
# Full pipeline round-trip through CLIP + codebook
# ---------------------------------------------------------------
print("\n=== Loading codebook ===")
cb = load_codebook()
n = cb["n_images"]

MESSAGES = {
    "short": "HELLO",
    "ascii": "Coverless steganography via CLIP hashing.",
    "unicode": "こんにちは世界 🌍 — secret",
    "empty": "",
    "long": "The quick brown fox jumps over the lazy dog. " * 12,
}

print("\n=== Full pipeline: encode -> CLIP decode -> verify ===")
for name, msg in MESSAGES.items():
    paths, meta = encode(msg, cb)
    # indices must be recovered exactly by CLIP nearest-neighbour
    rec = recover_indices(paths, cb)
    test(f"[{name}] CLIP recovers exact indices ({len(paths)} imgs)", rec == meta["indices"])
    out = decode(paths, cb)
    test(f"[{name}] message round-trips losslessly", out == msg)

# ---------------------------------------------------------------
# With AES-256 encryption enabled
# ---------------------------------------------------------------
print("\n=== With AES-256-CBC encryption ===")
key = generate_key()
for name, msg in (("short", "HELLO"), ("unicode", "秘密 🔐")):
    paths, meta = encode(msg, cb, key=key)
    out = decode(paths, cb, key=key)
    test(f"[{name}] encrypted round-trip", out == msg)

# wrong key must fail the CRC check
print("\n=== Tamper / wrong-key detection ===")
paths, _ = encode("HELLO", cb, key=key)
try:
    decode(paths, cb, key=generate_key())  # different key
    test("wrong key raises CRC error", False)
except ValueError:
    test("wrong key raises CRC error", True)

# ---------------------------------------------------------------
print(f"\n{'='*50}")
print(f"Phase 3 Results: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*50}")
if failed > 0:
    sys.exit(1)
