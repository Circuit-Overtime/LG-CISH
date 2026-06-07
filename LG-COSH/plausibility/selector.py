"""Plausibility layer — judge how natural a cover image sequence looks and pick
the most plausible alias candidates.

Two pieces:
  * diverse_chooser  — deterministic, offline. Rotates through a slot's alias
    candidates so repeated slots show visual variety (used by the evaluation,
    keeping results reproducible and network-free).
  * llm_plausibility_score — optional, LLM-backed. Asks gemini-fast (vision) to
    rate, 0-1, how much a set of images looks like one person's natural photo
    gallery. Used to *measure* the plausibility gain aliases provide.

Neither touches the encoded bits: every alias of a slot maps to the same slot, so
the decoder is unaffected regardless of which candidate is shown.
"""

import os


def diverse_chooser():
    """Return a stateful chooser(slot, cands, pos, idxs) that round-robins a
    slot's candidates for maximum visual variety across repeats."""
    seen: dict[int, int] = {}

    def choose(slot, cands, pos, idxs):
        c = seen.get(slot, 0)
        seen[slot] = c + 1
        return cands[c % len(cands)]
    return choose


def llm_plausibility_score(image_paths, sample=6, model=None, seed=0):
    """Rate 0-1 how natural `image_paths` looks as one person's photo gallery.

    Samples up to `sample` distinct images (to bound payload/cost) and asks a
    vision LLM for a single score. Returns a float in [0,1]; raises on API error.
    """
    from plausibility.llm_client import chat, _data_uri, TEXT_MODEL
    model = model or TEXT_MODEL
    seen, picks = set(), []
    for p in image_paths:
        if p not in seen:
            seen.add(p); picks.append(p)
        if len(picks) >= sample:
            break
    content = [{"type": "text", "text":
                "Here are images shared together in one message. On a scale from 0 to 1, "
                "how plausibly do they look like one person's ordinary photo gallery "
                "(vs. a suspicious random mix)? Reply with ONLY the number."}]
    for p in picks:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(p)}})
    txt = chat([{"role": "user", "content": content}], model=model, seed=seed, temperature=0.2)
    # extract the first float in the reply
    import re
    m = re.search(r"[01](?:\.\d+)?", txt)
    return float(m.group()) if m else float("nan")


if __name__ == "__main__":
    import sys
    print("score:", llm_plausibility_score(sys.argv[1:]))
