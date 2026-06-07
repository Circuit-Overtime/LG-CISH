"""Section 4.4 — Robustness Analysis.

Simulates real-world channel attacks (JPEG, Gaussian/salt&pepper noise, resize,
crop, format conversion) on the transmitted image sequence and measures decoding
accuracy, BER, and the CLIP distance margin at each degradation level.

Fast path: each attack is applied to the N unique codebook images once, then all
test messages are scored by table lookup (the sequence only ever reuses those N).
"""

import numpy as np

import _common as C

PER_ATTACK = 40
MSG_BUCKET = ("Medium (50-200)", 50, 200)


def run():
    C.banner("Section 4.4 — Robustness Analysis")
    cb = C.get_codebook()
    bank = C.message_bank([MSG_BUCKET], PER_ATTACK)[MSG_BUCKET[0]]

    rows = []
    detail = {}
    for name, fn in C.attack_suite():
        src_emb = C.attacked_source_embeddings(cb, fn)
        src_pred, src_margin, src_top1 = C.source_lookup(src_emb, cb["embeddings"])
        recs = [C.evaluate_message_fast(m, cb, src_pred, src_margin, src_top1) for m in bank]
        agg = C.aggregate(recs, ["exact", "ber", "ser", "mean_margin", "mean_top1"])
        acc = 100.0 * agg["exact"]
        rows.append([name, f"{acc:.2f}", f"{agg['ber']:.2e}",
                     f"{agg['mean_margin']:.3f}", f"{agg['mean_top1']:.3f}"])
        detail[name] = agg
        print(f"  {name:<24} acc={acc:6.2f}%  BER={agg['ber']:.2e}  "
              f"margin={agg['mean_margin']:.3f}")

    md = C.save_table(
        "table_4_4_robustness", rows,
        ["Attack", "Accuracy (%)", "BER", "CLIP Margin", "Top-1 Sim"],
        "Table 4.4 — Robustness to channel attacks (mean over %d messages/attack)." % PER_ATTACK)
    return {"table": md, "detail": detail}


if __name__ == "__main__":
    run()
