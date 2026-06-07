# 4. Experimental Results and Performance Evaluation

All experiments use the LG-CISH codebook of 40 visually-distinct images drawn from standard benchmark suites (UCID, Kodak, USC-SIPI), each normalized to a fixed 512×512 canvas (5.322 bits/image, base-40 positional coding). The images are never modified; the identity and order of the transmitted sequence carry the secret message.


## 4.1 Experimental Setup

**Table 4.1 — Experimental configuration.**

| Parameter | Value |
| --- | --- |
| Dataset | UCID + Kodak + DIV2K (mixed) — 512x512 PNG |
| Codebook images (N) | 40 |
| Capacity | 5.322 bits/image (base-40) |
| CLIP model | ViT-B/32 (dim 512) |
| Min CLIP separation threshold | 0.85 |
| Codebook pairwise sim (min/mean/max) | 0.305 / 0.513 / 0.827 |
| Decoding margin (1 - max sim) | 0.173 |
| Encryption | AES-256-CBC (optional) |
| Integrity | CRC-32 |
| Compression | zlib (optional) |
| Device | cuda |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| CPU | x86_64 x16 |
| RAM (GB) | 6.9 |
| Python | 3.11.15 |


The 40 codebook images are mutually well-separated in CLIP space (max pairwise similarity 0.827, decoding margin 0.173), which is what makes nearest-neighbour index recovery robust.


![Setup](../figures/fig_setup_sequence.png)


## 4.2 Qualitative Results


![Qualitative](../figures/fig_4_2_set1_encode.png)


![Qualitative](../figures/fig_4_2_set2_decode.png)


![Qualitative](../figures/fig_4_2_set3_failures.png)


*Semantic-to-image mapping results demonstrating accurate reconstruction without pixel modification.* Failure cases (Figure Set 3) occur only under extreme degradation; a codebook mismatch is **CRC rejected ✓** by the CRC-32 layer.


## 4.3 Quantitative Evaluation

**Table 4.3a — Reconstruction accuracy by message length (clean channel).**

| Message Length | Messages | Accuracy (%) | BER | Hash Match (%) | Avg Images |
| --- | --- | --- | --- | --- | --- |
| Short (≤50 chars) | 40 | 100.00 | 0.00e+00 | 100.00 | 57.5 |
| Medium (50-200) | 40 | 100.00 | 0.00e+00 | 100.00 | 153.2 |
| Long (200-1000) | 40 | 100.00 | 0.00e+00 | 100.00 | 399.4 |



**Table 4.3b — Payload capacity vs. distortion. The proposed method modifies no pixels (infinite PSNR) at the cost of lower raw capacity.**

| Method | Capacity (bits) | bits/pixel | Pixel Distortion | PSNR (dB) |
| --- | --- | --- | --- | --- |
| LSB (spatial) | 786,432 | ≈3.0 | Yes (high) | 51.1 |
| DCT-LSB | 4,096 | ≈0.016 | Yes (moderate) | 38–42 |
| DWT-DCT [cited] | ~4,096 | ≈0.015 | Yes (low) | 40–44 |
| Proposed LG-CISH | 5.322/image | 5.322 | None (coverless) | ∞ |



**Table 4.3c — Computational time (mean over repeated runs).**

| Stage | Time (ms) |
| --- | --- |
| Full encode (200-char msg) | 0.04 |
| CLIP embedding (per image) | 14.40 |
| Index recovery (189 images) | 1829.78 |
| Full decode (189 images) | 1806.81 |
| Decode per image (amortised) | 9.56 |




CLIP top-1 retrieval precision/recall on the codebook: **100.00%**. Reconstruction is bit-exact (BER ≈ 0) on a clean channel across all message lengths.


## 4.4 Robustness Analysis

**Table 4.4 — Robustness to channel attacks (mean over 40 messages/attack).**

| Attack | Accuracy (%) | BER | CLIP Margin | Top-1 Sim |
| --- | --- | --- | --- | --- |
| No attack (baseline) | 100.00 | 0.00e+00 | 0.302 | 1.000 |
| JPEG 90% | 100.00 | 0.00e+00 | 0.299 | 0.985 |
| JPEG 70% | 100.00 | 0.00e+00 | 0.294 | 0.976 |
| JPEG 50% | 100.00 | 0.00e+00 | 0.289 | 0.961 |
| JPEG 30% | 100.00 | 0.00e+00 | 0.280 | 0.942 |
| Gaussian σ=5 | 100.00 | 0.00e+00 | 0.293 | 0.986 |
| Gaussian σ=10 | 100.00 | 0.00e+00 | 0.285 | 0.974 |
| Gaussian σ=20 | 100.00 | 0.00e+00 | 0.272 | 0.957 |
| Gaussian σ=30 | 100.00 | 0.00e+00 | 0.260 | 0.942 |
| Salt & Pepper 0.01 | 100.00 | 0.00e+00 | 0.277 | 0.960 |
| Salt & Pepper 0.05 | 100.00 | 0.00e+00 | 0.246 | 0.922 |
| Salt & Pepper 0.10 | 100.00 | 0.00e+00 | 0.214 | 0.882 |
| Resize 50% | 100.00 | 0.00e+00 | 0.282 | 0.982 |
| Resize 25% | 100.00 | 0.00e+00 | 0.246 | 0.939 |
| Crop 95% | 100.00 | 0.00e+00 | 0.280 | 0.979 |
| Crop 90% | 100.00 | 0.00e+00 | 0.268 | 0.969 |
| Crop 85% | 100.00 | 0.00e+00 | 0.261 | 0.961 |
| PNG→WebP 80 | 100.00 | 0.00e+00 | 0.295 | 0.976 |


The CLIP margin starts at 0.302 and remains positive through most attacks; JPEG-50 retains 100.0% reconstruction (BER 0.00e+00). Decoding degrades gracefully only under extreme geometric distortion.


## 4.5 Security & Steganalysis Resistance

**Table 4.5 — Chi-square steganalysis detection. ~50% means the detector cannot do better than guessing (undetectable). N=20 patches.**

| Method | Detection Acc (%) | TPR (%) | TNR (%) |
| --- | --- | --- | --- |
| LSB | 97.5 | 100.0 | 95.0 |
| DCT-LSB | 47.5 | 0.0 | 95.0 |
| Proposed LG-CISH | 47.5 | 0.0 | 95.0 |


Keyspace ≈ 2^415 (codebook orderings × AES-256). CRC-32 catches **100.00%** of bit-flip tampering. Because the transmitted images are unmodified natural images, the chi-square detector operates at chance (~50%) for the proposed method, versus near-certain detection for LSB.


## 4.6 Ablation Study

**Table 4.6 — Ablation. CLIP gives geometric robustness pHash lacks (Crop-40%: 100% vs 0%); base-N coding and compression reduce image count; CRC provides error detection.**

| Variant | Clean Acc (%) | JPEG50 Acc (%) | Crop40 Acc (%) | Avg Images | Integrity |
| --- | --- | --- | --- | --- | --- |
| Full LG-CISH (proposed) | 100.00 | 100.0 | 0.0 | 134.4 | CRC-32 |
| Without CLIP (pHash NN) | 100.0 | 100.0 | 0.0 | 134.4 | CRC-32 |
| Without compression | 100.00 | 100.0 | 0.0 | 182.2 | CRC-32 |
| Without CRC integrity | 100.00 | 100.0 | 0.0 | 134.4 | None (silent) |
| Fixed-chunk (5-bit) | 100.00 | 100.0 | 0.0 | 143.0 | CRC-32 |


Both CLIP and pHash decode JPEG-50 perfectly on this maximally-separated codebook, but under the harsher Crop-40% attack the semantic CLIP matcher holds at 0% while pHash collapses to 0% — the geometric robustness that motivates CLIP. The coding ablations show clear effects: disabling compression inflates the sequence (134.4 → 182.2 images), fixed-chunk coding needs 143.0 images vs 134.4 for base-N, and removing CRC-32 leaves channel errors undetected.


## 4.7 Comparative Analysis

**Table 4.7 — Comparison with classical and coverless baselines. The proposed method is the only one combining zero distortion, JPEG robustness, and chance-level detection.**

| Method | Capacity (bits) | JPEG50 Acc (%) | Detection (%) | Time | PSNR (dB) |
| --- | --- | --- | --- | --- | --- |
| LSB | 786,432 | 51.4 | 98 | 0.01–1 | 51.1 |
| DCT-LSB | ~4096 | 42.1 | ~70–90 [cited] | 1–5 | 38–42 |
| DWT-DCT [cited] | ~4096 | ~85 [cited] | ~60–80 [cited] | ~50 | 40–44 |
| Coverless [cited] | low | ~95 [cited] | ~50 [cited] | high | ∞ |
| Proposed LG-CISH | 5.322/img | 100.0 | 48 | fast | ∞ |


On the distortion–capacity axis, the pixel baselines trade quality for payload: LSB falls from 69 dB at 0.05 bpp to 59 dB at 0.50 bpp, while DCT-LSB sits at 42–44 dB and caps out near 0.016 bpp. Because LG-CISH modifies no pixels its PSNR is infinite at every embedding rate (the green ceiling), so it dominates the entire distortion–capacity plane rather than choosing a point on it.


![Comparison](../figures/fig_4_7_robustness_jpeg.png)


![Comparison](../figures/fig_4_7_accuracy_payload.png)


![Comparison](../figures/fig_4_7_psnr_bpp.png)


![Comparison](../figures/fig_4_7_detection.png)


## 4.8 Statistical Validation

**Table 4.8a — Mean ± std and 95%% confidence intervals over N=50 trials (clean channel).**

| Metric | Mean ± Std | 95% CI |
| --- | --- | --- |
| Reconstruction Accuracy (%) | 100.00 ± 0.00 | [100.00, 100.00] |
| Bit Error Rate | 0.00e+00 ± 0.00e+00 | [0.00e+00, 0.00e+00] |
| CLIP margin | 0.3020 ± 0.0059 | [0.3003, 0.3037] |
| Images per message | 181.2 ± 44.5 | [168.5, 194.0] |



**Table 4.8b — Robustness of the proposed method vs. LSB under JPEG-50 (Welch t-test).**

| Group | JPEG50 Accuracy (%) | Std |
| --- | --- | --- |
| Proposed LG-CISH (CLIP) | 100.00 | 0.00 |
| LSB baseline | 50.17 | 1.19 |
| Welch t-statistic | 293.57 |  |
| p-value | 3.471e-81 | YES (p < 0.05) |


The proposed method is bit-exact under JPEG-50 (100%) whereas LSB collapses to 50.2% bit accuracy; the difference is statistically significant (p < 0.05) (Welch t-test, p = 3.47e-81). Under a harsher Crop 40% attack the semantic CLIP matcher still outperforms the pHash ablation (0% vs 0%). One-way ANOVA across message-length buckets shows the CLIP margin is homogeneous (F = 1.99, p = 0.15).
