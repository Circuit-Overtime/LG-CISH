**Table 4.8b — Significance of CLIP's geometric robustness over the pHash matcher: paired Wilcoxon signed-rank across a crop-strength sweep (the regime that discriminates the two matchers). Proposed-vs-LSB @ JPEG-50 is shown descriptively, as the lossless method is deterministic.**

| Group | Accuracy (%) | Std / note |
| --- | --- | --- |
| Proposed (CLIP) — crop-sweep mean | 100.0 | 0.0 |
| pHash ablation — crop-sweep mean | 0.0 | 0.0 |
| Wilcoxon W (paired, 7 crop levels) | 28.0 |  |
| p-value (one-sided, CLIP > pHash) | 7.812e-03 | YES (p < 0.05) |
| [ref] Proposed vs LSB @ JPEG-50 | 100 vs 50 | descriptive |
