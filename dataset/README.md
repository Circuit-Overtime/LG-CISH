# LG-CISH Codebook Dataset

The **40-image augmented codebook** used by the LG-CISH coverless image-steganography
framework. In LG-CISH the images are *never modified* — a message is encoded by the
**identity and order** of images drawn from this fixed codebook (base-40 positional
coding, `log2(40) ≈ 5.322` bits/image). This folder exists so the exact codebook can
be hosted on GitHub and cited.

## Contents
| File | Description |
|------|-------------|
| `images/c00.png … c39.png` | the 40 codebook images (index `j` = codebook slot `c_j`), each `512×512` |
| `manifest.csv` | `index, filename, original_source_name` for provenance |
| `CITATION.cff` | machine-readable citation metadata (GitHub "Cite this repository") |

## How the codebook was built
1. **Pooled** from three standard, publicly available benchmark suites — **UCID**,
   **Kodak**, and **USC-SIPI** — chosen for credibility, reproducibility, and diversity.
2. **Normalised** every image to a fixed `512×512` canvas.
3. **Embedded** each image with **CLIP ViT-B/32** (512-d normalised embedding).
4. **Greedily pruned** so that no two retained images exceed a cosine-similarity
   threshold `τ = 0.85` (max off-diagonal similarity `0.827`, i.e. separation margin
   `0.173`). **40** mutually distinct images survived.

This separation margin is what lets a receiver recover each image's index by
nearest-neighbour search even after JPEG/resize/noise/crop in transit.

## Source datasets (please also cite these)
- **UCID** — G. Schaefer and M. Stich, "UCID: An uncompressed color image database,"
  *Proc. SPIE 5307*, pp. 472–480, 2003.
- **Kodak** — R. Franzen, "Kodak lossless true color image suite (PhotoCD PCD0992),"
  Eastman Kodak Company, 1999.
- **USC-SIPI** — A. G. Weber, "The USC-SIPI image database," Signal and Image
  Processing Institute, University of Southern California, Tech. Rep., 1997.

> **License note:** The images are derived from the UCID, Kodak, and USC-SIPI suites.
> Please review each source dataset's terms before redistributing, and retain the
> attributions above.

## How to cite this codebook
Use the metadata in [`CITATION.cff`](CITATION.cff), or the plain-text form:

> LG-CISH Codebook Dataset: a 40-image augmented codebook for coverless image
> steganography. GitHub, 2026. Available: `https://github.com/<user>/<repo>`

*(Replace the URL — and, if minted, add a DOI — in `CITATION.cff` and above.)*
