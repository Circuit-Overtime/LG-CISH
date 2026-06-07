"""LG-CISH interactive demo — see the pipeline work both ways.

  Sentence  ->  ordered image sequence (written to disk + a montage you can open)
  Images     ->  sentence  (each image is re-embedded with CLIP and matched back)

Usage (from LG-COSH/, with the venv active and a codebook built):

    python demo.py encode "meet me at the docks at midnight"
    python demo.py decode demo_out
    python demo.py roundtrip "meet me at the docks at midnight"
    python demo.py roundtrip "secret plans" --permutation --perm-block 8
    python demo.py roundtrip "robust over whatsapp" --jpeg 40   # simulate a lossy channel

Flags: --permutation [--perm-block N]  --encrypt  --no-compress  --jpeg Q  --out DIR

The folder it writes is a genuine, decodable stego payload: the images are
unmodified codebook photos; only their identity and order carry the message.
"""

import argparse
import io
import json
import os
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import generate_demo_key
from codebook.builder import load_codebook
from encoder.encode import encode
from decoder.decode import decode, recover_indices

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_out")
THUMB = 200          # montage thumbnail size
COLS = 8             # montage columns


# --------------------------------------------------------------------------- #
#  montage: render the image sequence into one labelled contact sheet
# --------------------------------------------------------------------------- #
def make_montage(paths, indices, out_png, title):
    n = len(paths)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    pad, top = 8, 34
    W = cols * THUMB + (cols + 1) * pad
    H = top + rows * (THUMB + 18) + (rows + 1) * pad
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font = small = ImageFont.load_default()
    draw.text((pad, 8), title, fill="black", font=font)
    for k, (p, idx) in enumerate(zip(paths, indices)):
        r, c = divmod(k, cols)
        x = pad + c * (THUMB + pad)
        y = top + pad + r * (THUMB + 18 + pad)
        im = Image.open(p).convert("RGB").resize((THUMB, THUMB), Image.LANCZOS)
        canvas.paste(im, (x, y))
        draw.text((x, y + THUMB + 2), f"#{k}  idx={idx}", fill="black", font=small)
    canvas.save(out_png)
    return out_png


# --------------------------------------------------------------------------- #
#  encode: sentence -> images on disk
# --------------------------------------------------------------------------- #
def do_encode(msg, out_dir, cb, key, compress, permutation, perm_block):
    paths, meta = encode(msg, cb, key=key, use_compression=compress,
                         use_permutation=permutation, perm_block=perm_block)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # write the sequence as zero-padded files so sorted order == sequence order
    saved = []
    for i, p in enumerate(paths):
        dst = os.path.join(out_dir, f"{i:03d}_idx{meta['indices'][i]}.png")
        Image.open(p).convert("RGB").save(dst)
        saved.append(dst)

    # protocol params the receiver needs (note: the message itself is NOT stored)
    json.dump({
        "num_images": len(paths),
        "permutation": permutation, "perm_block": perm_block,
        "compressed": compress, "encrypted": key is not None,
    }, open(os.path.join(out_dir, "manifest.json"), "w"), indent=2)

    montage = make_montage(saved, meta["indices"], os.path.join(out_dir, "_sequence.png"),
                           f'"{msg}"  ->  {len(paths)} images')
    return paths, meta, montage


# --------------------------------------------------------------------------- #
#  decode: images on disk -> sentence  (real CLIP re-identification)
# --------------------------------------------------------------------------- #
def load_sequence(in_dir):
    files = sorted(f for f in os.listdir(in_dir)
                   if f.endswith(".png") and not f.startswith("_"))
    return [os.path.join(in_dir, f) for f in files]


def jpeg_channel(paths, quality, work_dir):
    """Simulate a lossy transport channel: JPEG-recompress every image."""
    os.makedirs(work_dir, exist_ok=True)
    out = []
    for i, p in enumerate(paths):
        buf = io.BytesIO()
        Image.open(p).convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        dst = os.path.join(work_dir, f"ch_{i:03d}.png")
        Image.open(buf).convert("RGB").save(dst)
        out.append(dst)
    return out


def do_decode(in_dir, cb, key, compress, permutation, perm_block, jpeg=None):
    man_path = os.path.join(in_dir, "manifest.json")
    if os.path.exists(man_path):                      # trust the folder's own protocol
        m = json.load(open(man_path))
        permutation, perm_block = m["permutation"], m["perm_block"]
        compress, encrypted = m["compressed"], m["encrypted"]
        key = generate_demo_key() if encrypted else None
    paths = load_sequence(in_dir)
    if jpeg is not None:
        paths = jpeg_channel(paths, jpeg, os.path.join(in_dir, "_channel"))
    rec_idx, margins = recover_indices(paths, cb, return_margins=True)
    msg = decode(paths, cb, key=key, use_compression=compress,
                 use_permutation=permutation, perm_block=perm_block)
    return msg, rec_idx, min(mg for _, mg in margins)


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def banner(t):
    print(f"\n{'='*64}\n  {t}\n{'='*64}")


def main():
    ap = argparse.ArgumentParser(description="LG-CISH sentence<->images demo")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("encode", "roundtrip"):
        s = sub.add_parser(name)
        s.add_argument("message")
        s.add_argument("--out", default=DEFAULT_OUT)
        s.add_argument("--permutation", action="store_true")
        s.add_argument("--perm-block", type=int, default=None)
        s.add_argument("--encrypt", action="store_true")
        s.add_argument("--no-compress", action="store_true")
        if name == "roundtrip":
            s.add_argument("--jpeg", type=int, default=None, help="simulate JPEG channel at quality Q")
    sd = sub.add_parser("decode")
    sd.add_argument("in_dir")
    sd.add_argument("--jpeg", type=int, default=None)

    a = ap.parse_args()
    cb = load_codebook()

    if a.cmd in ("encode", "roundtrip"):
        key = generate_demo_key() if a.encrypt else None
        compress = not a.no_compress
        banner(f"ENCODE  —  sentence -> images")
        print(f"Message      : {a.message!r}")
        print(f"Coding       : {'permutation (block=%s)' % (a.perm_block or cb['n_images']) if a.permutation else 'base-N'}"
              f"   compression: {'on' if compress else 'off'}   encryption: {'AES-256' if key else 'off'}")
        paths, meta, montage = do_encode(a.message, a.out, cb, key, compress,
                                         a.permutation, a.perm_block)
        print(f"Encoded into : {len(paths)} images  (indices {meta['indices']})")
        print(f"Images saved : {a.out}/")
        print(f"Montage      : {montage}   <- open this to SEE the sequence")

        if a.cmd == "roundtrip":
            banner("DECODE  —  images -> sentence")
            if a.jpeg is not None:
                print(f"Channel      : JPEG quality {a.jpeg} applied to every image (lossy transport)")
            msg, rec_idx, min_margin = do_decode(a.out, cb, key, compress,
                                                 a.permutation, a.perm_block, jpeg=a.jpeg)
            ok = (rec_idx == meta["indices"]) and (msg == a.message)
            print(f"CLIP re-identified indices match: {rec_idx == meta['indices']}  "
                  f"(min margin {min_margin:.3f})")
            print(f"Recovered    : {msg!r}")
            banner("ROUND-TRIP: " + ("SUCCESS ✓" if ok else "FAILED ✗"))
            sys.exit(0 if ok else 1)

    elif a.cmd == "decode":
        banner("DECODE  —  images -> sentence")
        msg, rec_idx, min_margin = do_decode(a.in_dir, cb, None, True, False, None, jpeg=a.jpeg)
        print(f"Read {len(rec_idx)} images from {a.in_dir}/  (min CLIP margin {min_margin:.3f})")
        print(f"Recovered    : {msg!r}")


if __name__ == "__main__":
    main()
