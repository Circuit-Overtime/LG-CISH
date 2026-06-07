"""Tests for distinct-image permutation (Lehmer) coding.

Verifies the codec is lossless, produces no within-block image repeats, whitens
away sorted prefixes, and survives a full encode->CLIP-recover->decode round-trip.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitstream.converter import (
    bytes_to_perm_indices, perm_indices_to_bytes, _perm_modulus,
)

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


def test_codec():
    N = 40
    print("\n=== permutation codec (lossless + distinct) ===")
    all_lossless = all_distinct = True
    for t in range(300):
        data = os.urandom(1 + t % 80)
        for block in (40, 16, 8, 1):
            idx = bytes_to_perm_indices(data, N, block)
            for i in range(0, len(idx), block):
                blk = idx[i:i + block]
                if len(set(blk)) != len(blk):
                    all_distinct = False
            if perm_indices_to_bytes(idx, N, block) != data:
                all_lossless = False
    check("lossless across 300 payloads x 4 block sizes", all_lossless)
    check("no within-block repeats", all_distinct)


def test_whitening():
    print("\n=== whitening removes sorted prefix ===")
    idx = bytes_to_perm_indices(b"\x00" * 4, 40, 16)  # smallest non-trivial value
    prefix = idx[:16]
    check("leading block is not sorted ascending", prefix != sorted(prefix))


def test_capacity():
    print("\n=== capacity matches P(N,B) ===")
    import math
    N = 40
    bits40 = math.log2(_perm_modulus(N, 40)) / 40
    bits8 = math.log2(_perm_modulus(N, 8)) / 8
    check("block=40 ~ 3.98 bits/img", abs(bits40 - 3.979) < 0.01)
    check("block=8  ~ 5.19 bits/img", abs(bits8 - 5.187) < 0.01)


def test_end_to_end():
    print("\n=== end-to-end encode -> CLIP recover -> decode ===")
    from codebook.builder import load_codebook
    from encoder.encode import encode
    from decoder.decode import decode, recover_indices
    cb = load_codebook()
    for msg in ("HI", "meet me at the docks at midnight", "the quick brown fox " * 5):
        for block in (cb["n_images"], 16):
            paths, meta = encode(msg, cb, use_permutation=True, perm_block=block)
            rec = recover_indices(paths, cb)
            out = decode(paths, cb, use_permutation=True, perm_block=block)
            check(f"round-trip block={block} len={len(msg)}", rec == meta["indices"] and out == msg)


if __name__ == "__main__":
    test_codec()
    test_whitening()
    test_capacity()
    test_end_to_end()
    print("\n" + "=" * 50)
    print(f"Permutation Results: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    print("=" * 50)
    sys.exit(1 if _failed else 0)
