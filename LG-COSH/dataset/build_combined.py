"""Build the combined fixed-size image database for LG-CISH.

Combines the Kodak Lossless True Color suite (24 images) with the existing
curated DIV2K subset (6 images) into a single database of 30 images, each
resized to a fixed 512x512 canvas (LANCZOS). The uniform size makes CLIP
matching and the crop/resize robustness attacks consistent across sources.

Run:  ../venv/bin/python dataset/build_combined.py
"""

import io
import os
import sys
import urllib.request

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # project root
IMAGE_DIR = os.path.join(ROOT, "images")
FIXED = (512, 512)
KODAK_URL = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"

# The six existing DIV2K images already in images/ (kept as the "other dataset").
DIV2K = ["0007", "0085", "0089", "0098", "0101", "0107"]


def fit(im: Image.Image) -> Image.Image:
    return im.convert("RGB").resize(FIXED, Image.LANCZOS)


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 1. Re-fit the six DIV2K images in place (source jpegs stay as <id>.jpeg).
    for name in DIV2K:
        src = os.path.join(IMAGE_DIR, f"{name}.jpeg")
        if not os.path.exists(src):
            print(f"  WARN missing DIV2K {src}")
            continue
        fit(Image.open(src)).save(src, quality=95)
        print(f"  div2k  {name}.jpeg -> {FIXED}")

    # 2. Download + fit the 24 Kodak images.
    for i in range(1, 25):
        url = KODAK_URL.format(i)
        dst = os.path.join(IMAGE_DIR, f"kodim{i:02d}.png")
        raw = urllib.request.urlopen(url, timeout=30).read()
        fit(Image.open(io.BytesIO(raw))).save(dst)
        print(f"  kodak  kodim{i:02d}.png -> {FIXED}")

    total = len([f for f in os.listdir(IMAGE_DIR)
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"\nDatabase ready: {total} images in {IMAGE_DIR} (all {FIXED[0]}x{FIXED[1]})")


if __name__ == "__main__":
    main()
