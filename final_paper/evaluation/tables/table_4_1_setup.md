**Table 4.1 — Experimental configuration.**

| Parameter | Value |
| --- | --- |
| Dataset | UCID + Kodak + USC-SIPI (mixed) — 512x512 PNG |
| Codebook images (N) | 40 |
| Capacity | 5.322 bits/image (base-40) |
| CLIP model | ViT-B/32 (dim 512) |
| LLM (plausibility / aliases) | gemini-fast (Gemini 2.5 Flash Lite, vision) + flux (image gen) |
| Coding modes | base-N (5.322 b/img) | permutation (no-repeat, distinct images) |
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
