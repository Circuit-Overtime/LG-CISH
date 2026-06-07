"""LG-CISH CLI — build codebook, encode, decode, or run a full round-trip demo.

Usage (from the LG-COSH/ directory, with the venv active):

    python main.py build                          # build codebook from images/
    python main.py encode "secret message"        # message -> image sequence
    python main.py decode img1.jpeg img2.jpeg ...  # image sequence -> message
    python main.py demo "secret message"          # encode + decode + verify
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CODEBOOK_PATH, IMAGE_DIR, generate_demo_key


def cmd_build(args):
    from codebook.builder import build_codebook
    print(f"Building codebook from images in {IMAGE_DIR} ...")
    cb = build_codebook()
    print(f"\nCodebook ready: {cb['n_images']} images, "
          f"{cb['bits_per_image']:.3f} bits/image -> {CODEBOOK_PATH}")


def cmd_encode(args):
    from codebook.builder import load_codebook
    from encoder.encode import encode
    cb = load_codebook()
    key = generate_demo_key() if args.encrypt else None
    paths, meta = encode(args.message, cb, key=key, use_compression=not args.no_compress)
    print(f"\nMessage '{args.message}' -> {len(paths)} images "
          f"({meta['payload_bits']} payload bits):")
    for i, p in enumerate(paths):
        print(f"  [{i:>3}] idx={meta['indices'][i]}  {os.path.basename(p)}")


def cmd_decode(args):
    from codebook.builder import load_codebook
    from decoder.decode import decode
    cb = load_codebook()
    key = generate_demo_key() if args.encrypt else None
    msg = decode(args.images, cb, key=key, use_compression=not args.no_compress)
    print(f"\nDecoded message: {msg!r}")


def cmd_demo(args):
    from codebook.builder import load_codebook
    from encoder.encode import encode
    from decoder.decode import decode, recover_indices

    cb = load_codebook()
    key = generate_demo_key() if args.encrypt else None
    msg = args.message
    perm = getattr(args, "permutation", False)
    pblock = getattr(args, "perm_block", None)

    print(f"\n{'='*60}\n  LG-CISH round-trip demo\n{'='*60}")
    print(f"Original message : {msg!r}")
    print(f"Encryption       : {'AES-256-CBC' if key else 'off (CRC only)'}")
    print(f"Compression      : {'zlib' if not args.no_compress else 'off'}")
    print(f"Coding           : {'permutation (Lehmer, block=' + str(pblock or cb['n_images']) + ')' if perm else 'base-N'}")

    paths, meta = encode(msg, cb, key=key, use_compression=not args.no_compress,
                         use_permutation=perm, perm_block=pblock)
    print(f"\nEncoded into {len(paths)} images (indices {meta['indices']})")
    if perm:
        b = meta["perm_block"]
        reps = sum(len(blk) - len(set(blk)) for blk in
                   (meta["indices"][i:i+b] for i in range(0, len(meta["indices"]), b)))
        print(f"Within-block repeats: {reps} (0 expected for permutation coding)")

    rec_idx, margins = recover_indices(paths, cb, return_margins=True)
    idx_ok = rec_idx == meta["indices"]
    min_margin = min(m for _, m in margins)
    print(f"CLIP recovered indices match: {idx_ok}  (min margin {min_margin:.4f})")

    out = decode(paths, cb, key=key, use_compression=not args.no_compress,
                 use_permutation=perm, perm_block=pblock)
    print(f"\nDecoded message  : {out!r}")
    print(f"{'='*60}")
    print("ROUND-TRIP:", "SUCCESS" if out == msg else "FAILED")
    print(f"{'='*60}")
    sys.exit(0 if out == msg else 1)


def main():
    p = argparse.ArgumentParser(description="LG-CISH coverless image steganography")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="build codebook from the image folder")
    pb.set_defaults(func=cmd_build)

    pe = sub.add_parser("encode", help="encode a message into an image sequence")
    pe.add_argument("message")
    pe.add_argument("--encrypt", action="store_true")
    pe.add_argument("--no-compress", action="store_true")
    pe.set_defaults(func=cmd_encode)

    pd = sub.add_parser("decode", help="decode an image sequence into a message")
    pd.add_argument("images", nargs="+")
    pd.add_argument("--encrypt", action="store_true")
    pd.add_argument("--no-compress", action="store_true")
    pd.set_defaults(func=cmd_decode)

    pm = sub.add_parser("demo", help="full encode+decode round-trip with verification")
    pm.add_argument("message")
    pm.add_argument("--encrypt", action="store_true")
    pm.add_argument("--no-compress", action="store_true")
    pm.add_argument("--permutation", action="store_true",
                    help="use distinct-image permutation (Lehmer) coding")
    pm.add_argument("--perm-block", type=int, default=None,
                    help="images per permutation block (default: full codebook N)")
    pm.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
