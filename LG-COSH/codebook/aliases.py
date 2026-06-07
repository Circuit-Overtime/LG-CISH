"""Alias map loading + candidate selection.

An alias map gives each codebook slot one or more interchangeable image files
(the base image plus LLM-generated, CLIP-verified candidates). Because every
candidate maps to the same slot under CLIP nearest-neighbour, swapping a base
image for a candidate changes the *cover* without changing the encoded bits —
the decoder is completely unaffected.

This module only handles the encode-side choice of *which* file to emit per slot.
The actual plausibility ranking lives in plausibility/selector.py.
"""

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(ROOT, "images_aliases", "manifest.json")


def load_aliases(manifest_path: str = MANIFEST) -> dict[int, list[str]]:
    """Return {slot_index: [candidate_paths]} (base image first). Empty if none."""
    if not os.path.exists(manifest_path):
        return {}
    raw = json.load(open(manifest_path))
    return {int(k): v for k, v in raw.items()}


def alias_stats(aliases: dict[int, list[str]], n: int) -> dict:
    """Coverage stats for the results section."""
    per_slot = [len(aliases.get(i, [])) for i in range(n)]
    extra = [max(0, c - 1) for c in per_slot]           # candidates beyond the base
    slots_with_alias = sum(1 for e in extra if e > 0)
    return {
        "slots_with_alias": slots_with_alias,
        "total_candidates": int(sum(extra)),
        "avg_candidates_per_slot": float(sum(c for c in per_slot) / n) if n else 0.0,
        "max_candidates": max(per_slot) if per_slot else 0,
    }


def choose_paths(indices: list[int], base_paths: list[str],
                 aliases: dict[int, list[str]], chooser=None) -> list[str]:
    """Map an index sequence to image paths, picking a candidate per slot.

    chooser(slot, candidates, position, indices) -> chosen_path. Defaults to a
    deterministic round-robin over a slot's candidates so repeated uses of the
    same slot show visual variety (a more album-like cover) while staying lossless.
    """
    if chooser is None:
        seen: dict[int, int] = {}

        def chooser(slot, cands, pos, idxs):
            c = seen.get(slot, 0)
            seen[slot] = c + 1
            return cands[c % len(cands)]

    out = []
    for pos, slot in enumerate(indices):
        cands = aliases.get(slot) or [base_paths[slot]]
        out.append(chooser(slot, cands, pos, indices))
    return out
