"""Normalize the LG-CISH image database to a single fixed-size PNG format.

Scans the repo-root images/ folder (UCID .tif, Kodak .png, USC-SIPI .tif, ...),
converts every image to RGB, resizes to a fixed 512x512 canvas (LANCZOS), and
re-saves as .png. Non-PNG originals (e.g. .tif) are removed so the database is a
clean, uniform, single-format set. Idempotent: re-running is a no-op on a clean
folder.

Run:  ../venv/bin/python dataset/normalize.py
"""

import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # project root
IMAGE_DIR = os.path.join(ROOT, "images")
FIXED = (512, 512)
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    files = sorted(f for f in os.listdir(IMAGE_DIR)
                   if os.path.splitext(f)[1].lower() in EXTS)
    converted, removed = 0, 0
    for fname in files:
        src = os.path.join(IMAGE_DIR, fname)
        stem, ext = os.path.splitext(fname)
        dst = os.path.join(IMAGE_DIR, f"{stem}.png")
        im = Image.open(src).convert("RGB").resize(FIXED, Image.LANCZOS)
        im.save(dst)
        converted += 1
        if ext.lower() != ".png":          # drop the non-png original
            os.remove(src)
            removed += 1
        print(f"  {fname:<16} -> {stem}.png {FIXED}")

    pngs = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png")]
    print(f"\nNormalized {converted} images ({removed} non-png removed). "
          f"Database: {len(pngs)} PNGs @ {FIXED[0]}x{FIXED[1]} in {IMAGE_DIR}")


if __name__ == "__main__":
    main()
