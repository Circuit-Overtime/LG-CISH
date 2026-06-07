**Table 4.5b — Cover plausibility (gemini-fast judge, 0–1). Plausibility is driven by codebook theme, not the coding mode: a themed database is far more plausible than the diverse benchmark set, while permutation coding leaves the score essentially unchanged. We use the diverse benchmark for all other results (credibility) and report this as an explicit deployment trade-off.**

| Configuration | LLM plausibility (0–1) | Note |
| --- | --- | --- |
| Diverse benchmark codebook (UCID/Kodak/USC) | 0.12 ± 0.04 | max dataset credibility; mixed subjects look random |
| Themed codebook (coherent context) | 0.88 ± 0.07 | looks like an ordinary personal album |
| Diverse + permutation coding | 0.18 ± 0.10 | codec barely moves the score |
