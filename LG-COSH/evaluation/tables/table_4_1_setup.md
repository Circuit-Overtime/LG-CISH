**Table 4.1 — Experimental configuration.**

| Parameter | Value |
| --- | --- |
| Dataset | DIV2K (curated subset) |
| Codebook images (N) | 6 |
| Capacity | 2.585 bits/image (base-6) |
| CLIP model | ViT-B/32 (dim 512) |
| Min CLIP separation threshold | 0.85 |
| Codebook pairwise sim (min/mean/max) | 0.353 / 0.497 / 0.609 |
| Decoding margin (1 - max sim) | 0.391 |
| Encryption | AES-256-CBC (optional) |
| Integrity | CRC-32 |
| Compression | zlib (optional) |
| Device | cuda |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| CPU | x86_64 x16 |
| RAM (GB) | 6.9 |
| Python | 3.11.15 |
