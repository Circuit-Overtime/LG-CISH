"""Shared evaluation infrastructure for the LG-CISH results suite.

Provides:
  * deterministic message generation (seeded)
  * channel-attack simulators (JPEG, noise, resize, crop, format conversion)
  * metrics (exact-match accuracy, symbol/index error rate, BER, CLIP margin)
  * an attacked-decode driver (encode -> attack each image -> re-decode -> metrics)
  * markdown + LaTeX table writers and figure/output path helpers

Every script under evaluation/ imports from here so numbers are produced the
same way across sections.
"""

import os
import sys
import io
import math
import random
import tempfile

import numpy as np
from PIL import Image

# --- make the LG-COSH package importable ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from config import CODEBOOK_PATH, CLIP_MODEL, EMBEDDING_DIM, MIN_CLIP_DISTANCE, get_device
from codebook.builder import load_codebook
from encoder.encode import encode
from decoder.decode import decode, recover_indices
from bitstream.converter import bytes_to_indices, indices_to_bytes

# --- output locations ---
FIG_DIR = os.path.join(_HERE, "figures")
TBL_DIR = os.path.join(_HERE, "tables")
OUT_DIR = os.path.join(_HERE, "results")
for _d in (FIG_DIR, TBL_DIR, OUT_DIR):
    os.makedirs(_d, exist_ok=True)

SEED = 1234


# ============================================================
# Codebook (cached)
# ============================================================
_CB = None


def get_codebook():
    global _CB
    if _CB is None:
        _CB = load_codebook(CODEBOOK_PATH)
    return _CB


# ============================================================
# Deterministic message generation
# ============================================================
_WORDS = (
    "the quick brown fox jumps over a lazy dog while secret data flows through "
    "images carrying hidden meaning across an open channel without distortion or "
    "suspicion using semantic hashing and coverless steganography techniques today"
).split()


def random_message(n_chars: int, rng: random.Random) -> str:
    """Generate a natural-looking message of approximately n_chars characters."""
    out = []
    total = 0
    while total < n_chars:
        w = rng.choice(_WORDS)
        out.append(w)
        total += len(w) + 1
    msg = " ".join(out)
    return msg[:n_chars] if n_chars > 0 else ""


def message_bank(buckets, per_bucket, seed=SEED):
    """Build {bucket_name: [messages]} for length buckets.

    buckets: list of (name, low, high) char-length ranges.
    """
    rng = random.Random(seed)
    bank = {}
    for name, low, high in buckets:
        msgs = []
        for _ in range(per_bucket):
            length = rng.randint(low, high)
            msgs.append(random_message(length, rng))
        bank[name] = msgs
    return bank


DEFAULT_BUCKETS = [
    ("Short (≤50 chars)", 10, 50),
    ("Medium (50-200)", 50, 200),
    ("Long (200-1000)", 200, 1000),
]


# ============================================================
# Channel attacks  (PIL.Image -> PIL.Image)
# ============================================================
def atk_identity(img):
    return img.copy()


def atk_jpeg(img, quality):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


def atk_gaussian(img, sigma):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    noise = rng.normal(0, sigma, arr.shape)
    out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def atk_salt_pepper(img, density):
    arr = np.asarray(img.convert("RGB")).copy()
    rng = np.random.default_rng(SEED)
    mask = rng.random(arr.shape[:2])
    arr[mask < density / 2] = 0
    arr[mask > 1 - density / 2] = 255
    return Image.fromarray(arr)


def atk_resize(img, scale):
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR).convert("RGB")


def atk_crop(img, keep):
    """Center-crop to `keep` fraction of each dimension."""
    w, h = img.size
    cw, ch = int(w * keep), int(h * keep)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).convert("RGB")


def atk_webp(img, quality=80):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB").copy()


# Registry: name -> (callable taking PIL img). Used by robustness.py.
def attack_suite():
    return [
        ("No attack (baseline)", atk_identity),
        ("JPEG 90%", lambda im: atk_jpeg(im, 90)),
        ("JPEG 70%", lambda im: atk_jpeg(im, 70)),
        ("JPEG 50%", lambda im: atk_jpeg(im, 50)),
        ("JPEG 30%", lambda im: atk_jpeg(im, 30)),
        ("Gaussian σ=5", lambda im: atk_gaussian(im, 5)),
        ("Gaussian σ=10", lambda im: atk_gaussian(im, 10)),
        ("Gaussian σ=20", lambda im: atk_gaussian(im, 20)),
        ("Gaussian σ=30", lambda im: atk_gaussian(im, 30)),
        ("Salt & Pepper 0.01", lambda im: atk_salt_pepper(im, 0.01)),
        ("Salt & Pepper 0.05", lambda im: atk_salt_pepper(im, 0.05)),
        ("Salt & Pepper 0.10", lambda im: atk_salt_pepper(im, 0.10)),
        ("Resize 50%", lambda im: atk_resize(im, 0.5)),
        ("Resize 25%", lambda im: atk_resize(im, 0.25)),
        ("Crop 95%", lambda im: atk_crop(im, 0.95)),
        ("Crop 90%", lambda im: atk_crop(im, 0.90)),
        ("Crop 85%", lambda im: atk_crop(im, 0.85)),
        ("PNG→WebP 80", lambda im: atk_webp(im, 80)),
    ]


# ============================================================
# Fast path: attack the 6 unique codebook images ONCE per attack
# ============================================================
from clip_engine.embedder import embed_images_batch  # noqa: E402


def attacked_source_embeddings(cb, attack_fn):
    """Apply attack_fn to each of the N unique codebook images and return their
    (N, D) normalized CLIP embeddings — the 'received' versions of the codebook.

    Because every image in any encoded sequence is one of these N images and the
    attacks are deterministic, this is computed once and reused for all messages.
    """
    with tempfile.TemporaryDirectory(prefix="lgcish_src_") as td:
        paths = []
        for i, src in enumerate(cb["paths"]):
            dst = os.path.join(td, f"src_{i}.png")
            attack_fn(Image.open(src).convert("RGB")).save(dst)
            paths.append(dst)
        return embed_images_batch(paths)  # (N, D), normalized


def source_lookup(source_emb, codebook_emb):
    """For each received source image, its (pred_index, top1_margin) against the
    clean codebook. Returns (pred[N], margin[N], top1[N])."""
    sims = source_emb @ codebook_emb.T  # (N, N)
    pred = np.argmax(sims, axis=1).astype(int)
    margins, top1s = [], []
    for row in sims:
        order = np.sort(row)[::-1]
        top1s.append(float(order[0]))
        margins.append(float(order[0] - order[1]) if len(order) > 1 else float(order[0]))
    return pred, np.array(margins), np.array(top1s)


def evaluate_message_fast(message, cb, src_pred, src_margin, src_top1,
                          key=None, use_compression=True):
    """Score one message using the precomputed per-source lookup (no re-embedding).

    src_pred[t]   = codebook index a received copy of image t maps to
    src_margin[t] = its CLIP top1-top2 margin
    """
    n = cb["n_images"]
    _, meta = encode(message, cb, key=key, use_compression=use_compression)
    true_idx = meta["indices"]
    rec_idx = [int(src_pred[t]) for t in true_idx]

    n_imgs = len(true_idx)
    symbol_errors = sum(1 for a, b in zip(true_idx, rec_idx) if a != b)
    ser = symbol_errors / n_imgs if n_imgs else 0.0

    tb = index_bits(true_idx, n)
    rb = index_bits(rec_idx, n)
    ber = sum(1 for a, b in zip(tb, rb) if a != b) / len(tb) if tb else 0.0

    exact = (symbol_errors == 0)  # lossless chain: exact iff all indices recovered

    seq_margins = [src_margin[t] for t in true_idx]
    seq_top1 = [src_top1[t] for t in true_idx]
    return {
        "n_images": n_imgs,
        "ser": ser,
        "ber": ber,
        "exact": exact,
        "mean_margin": float(np.mean(seq_margins)) if seq_margins else 0.0,
        "min_margin": float(np.min(seq_margins)) if seq_margins else 0.0,
        "mean_top1": float(np.mean(seq_top1)) if seq_top1 else 0.0,
        "symbol_errors": symbol_errors,
    }


# ============================================================
# Attack application + decoding (slow path — arbitrary received images)
# ============================================================
def apply_attack_to_sequence(image_paths, attack_fn, tmp_dir):
    """Apply attack_fn to every image in the sequence, save to tmp_dir, return new paths.

    Distinct source paths are transformed once and cached (the sequence repeats
    only 6 unique codebook images, so this is cheap).
    """
    cache = {}
    out_paths = []
    for i, p in enumerate(image_paths):
        if p not in cache:
            attacked = attack_fn(Image.open(p).convert("RGB"))
            dst = os.path.join(tmp_dir, f"atk_{len(cache)}.png")
            attacked.save(dst)
            cache[p] = dst
        out_paths.append(cache[p])
    return out_paths


def index_bits(indices, n):
    """Fixed-width bit representation of base-n digits (ceil(log2 n) bits each)."""
    width = max(1, math.ceil(math.log2(n)))
    return "".join(format(d, f"0{width}b") for d in indices)


def evaluate_message(message, cb, attack_fn=None, key=None, use_compression=True):
    """Encode `message`, optionally attack the image sequence, decode, and score.

    Returns a dict of per-message metrics.
    """
    n = cb["n_images"]
    paths, meta = encode(message, cb, key=key, use_compression=use_compression)
    true_idx = meta["indices"]

    with tempfile.TemporaryDirectory(prefix="lgcish_atk_") as td:
        if attack_fn is not None:
            recv_paths = apply_attack_to_sequence(paths, attack_fn, td)
        else:
            recv_paths = paths

        rec_idx, margins = recover_indices(recv_paths, cb, return_margins=True)

        # symbol/index error rate
        n_imgs = len(true_idx)
        symbol_errors = sum(1 for a, b in zip(true_idx, rec_idx) if a != b)
        ser = symbol_errors / n_imgs if n_imgs else 0.0

        # bit error rate from fixed-width index codes
        tb = index_bits(true_idx, n)
        rb = index_bits(rec_idx, n)
        bit_errors = sum(1 for a, b in zip(tb, rb) if a != b)
        ber = bit_errors / len(tb) if tb else 0.0

        # exact message reconstruction
        exact = False
        decoded = None
        try:
            decoded = decode(recv_paths, cb, key=key, use_compression=use_compression)
            exact = decoded == message
        except Exception:
            exact = False

        mean_margin = float(np.mean([m for _, m in margins])) if margins else 0.0
        min_margin = float(np.min([m for _, m in margins])) if margins else 0.0
        mean_top1 = float(np.mean([t for t, _ in margins])) if margins else 0.0

    return {
        "n_images": n_imgs,
        "ser": ser,
        "ber": ber,
        "exact": exact,
        "mean_margin": mean_margin,
        "min_margin": min_margin,
        "mean_top1": mean_top1,
        "symbol_errors": symbol_errors,
    }


# ============================================================
# Aggregation helpers
# ============================================================
def aggregate(records, keys):
    """Mean over a list of per-message metric dicts for the given keys."""
    out = {}
    for k in keys:
        vals = [r[k] for r in records]
        out[k] = float(np.mean(vals)) if vals else 0.0
        out[k + "_std"] = float(np.std(vals)) if vals else 0.0
    out["n"] = len(records)
    return out


# ============================================================
# Table writers
# ============================================================
def write_markdown_table(rows, headers, path, title=None):
    """rows: list of lists (strings). Writes a GitHub-markdown table."""
    lines = []
    if title:
        lines.append(f"**{title}**\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def write_latex_table(rows, headers, path, caption="", label=""):
    align = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    for r in rows:
        lines.append(" & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def save_table(name, rows, headers, caption):
    """Write both markdown and LaTeX versions of a table; return the markdown text."""
    md = write_markdown_table(rows, headers, os.path.join(TBL_DIR, f"{name}.md"), title=caption)
    write_latex_table(rows, headers, os.path.join(TBL_DIR, f"{name}.tex"),
                      caption=caption, label=f"tab:{name}")
    return md


def banner(text):
    line = "=" * 64
    print(f"\n{line}\n  {text}\n{line}")
