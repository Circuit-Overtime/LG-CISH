"""Tests for the alias (multi-candidate cover) mechanism.

The key invariant: turning aliases on changes *which image files* are emitted but
NOT the decoded message — every candidate maps to the same codebook slot, so the
decoder is unaffected. Works with base-N and permutation coding alike.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebook.builder import load_codebook
from codebook.aliases import load_aliases, choose_paths, alias_stats
from encoder.encode import encode
from decoder.decode import decode

_passed = _failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1; print(f"  PASS  {name}")
    else:
        _failed += 1; print(f"  FAIL  {name}")


def test_decode_invariance():
    print("\n=== aliases don't change the decoded message ===")
    cb = load_codebook()
    for msg in ("HI", "meet me at the docks at midnight", "secret payload " * 4):
        for perm, blk in ((False, None), (True, 16)):
            base_paths, _ = encode(msg, cb, use_permutation=perm, perm_block=blk)
            al_paths, _ = encode(msg, cb, use_permutation=perm, perm_block=blk, use_aliases=True)
            out = decode(al_paths, cb, use_permutation=perm, perm_block=blk)
            check(f"decode==msg (perm={perm}) len={len(msg)}", out == msg)


def test_candidates_used():
    print("\n=== alias candidates are actually emitted when available ===")
    cb = load_codebook()
    al = load_aliases()
    st = alias_stats(al, cb["n_images"])
    print(f"  coverage: {st}")
    if st["total_candidates"] == 0:
        print("  (no candidates generated yet — skipping emission check)")
        return
    # A message whose index sequence reuses aliased slots should emit candidate files.
    paths, _ = encode("the quick brown fox " * 6, cb, use_aliases=True)
    emitted_alias = sum(1 for p in paths if "images_aliases" in p)
    check("at least one alias-candidate file emitted", emitted_alias > 0)


def test_chooser_fallback():
    print("\n=== slots without candidates fall back to the base image ===")
    cb = load_codebook()
    paths = choose_paths([0, 1, 2, 0], cb["paths"], aliases={}, chooser=None)
    check("empty alias map -> all base paths", paths == [cb["paths"][i] for i in [0, 1, 2, 0]])


if __name__ == "__main__":
    test_decode_invariance()
    test_candidates_used()
    test_chooser_fallback()
    print("\n" + "=" * 50)
    print(f"Alias Results: {_passed} passed, {_failed} failed out of {_passed + _failed}")
    print("=" * 50)
    sys.exit(1 if _failed else 0)
