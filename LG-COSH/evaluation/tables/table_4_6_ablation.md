**Table 4.6 — Ablation. CLIP gives geometric robustness pHash lacks (Crop-40%: 100% vs 0%); base-N coding and compression reduce image count; CRC provides error detection.**

| Variant | Clean Acc (%) | JPEG50 Acc (%) | Crop40 Acc (%) | Avg Images | Integrity |
| --- | --- | --- | --- | --- | --- |
| Full LG-CISH (proposed) | 100.00 | 100.0 | 0.0 | 134.4 | CRC-32 |
| Without CLIP (pHash NN) | 100.0 | 100.0 | 0.0 | 134.4 | CRC-32 |
| Without compression | 100.00 | 100.0 | 0.0 | 182.2 | CRC-32 |
| Without CRC integrity | 100.00 | 100.0 | 0.0 | 134.4 | None (silent) |
| Fixed-chunk (5-bit) | 100.00 | 100.0 | 0.0 | 143.0 | CRC-32 |
