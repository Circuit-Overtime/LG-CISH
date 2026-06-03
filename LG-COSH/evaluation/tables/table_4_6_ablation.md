**Table 4.6 — Ablation. CLIP gives geometric robustness pHash lacks (Crop-40%: 100% vs 0%); base-N coding and compression reduce image count; CRC provides error detection.**

| Variant | Clean Acc (%) | JPEG50 Acc (%) | Crop40 Acc (%) | Avg Images | Integrity |
| --- | --- | --- | --- | --- | --- |
| Full LG-CISH (proposed) | 100.00 | 100.0 | 100.0 | 276.5 | CRC-32 |
| Without CLIP (pHash NN) | 100.0 | 100.0 | 0.0 | 276.5 | CRC-32 |
| Without compression | 100.00 | 100.0 | 100.0 | 374.5 | CRC-32 |
| Without CRC integrity | 100.00 | 100.0 | 100.0 | 276.5 | None (silent) |
| Fixed-chunk (2-bit) | 100.00 | 100.0 | 100.0 | 356.5 | CRC-32 |
