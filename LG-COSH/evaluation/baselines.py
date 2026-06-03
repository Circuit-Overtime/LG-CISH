"""Reference steganography baselines for comparison (Section 4.7) and the
steganalysis study (Section 4.5).

Implements faithful, self-contained versions of:
  * LSB (spatial least-significant-bit replacement)
  * DCT-LSB (block-DCT mid-frequency LSB, JPEG-domain style)
and the quality / detectability tooling:
  * PSNR, SSIM
  * chi-square LSB steganalysis attack (Westfeld & Pfitzmann)
  * balanced detection accuracy of a statistical detector

These let us put the proposed coverless method on the same axes (capacity,
distortion, robustness, detectability) as the classical baselines.
"""

import io
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from scipy.stats import chi2 as chi2_dist
from skimage.metrics import structural_similarity as _ssim


# ============================================================
# Quality metrics
# ============================================================
def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 3:
        return float(_ssim(a, b, channel_axis=2, data_range=255))
    return float(_ssim(a, b, data_range=255))


# ============================================================
# LSB (spatial)
# ============================================================
def lsb_embed(cover_arr: np.ndarray, payload_bits: str) -> np.ndarray:
    """Replace the LSB of successive bytes with payload bits."""
    flat = cover_arr.flatten().copy()
    nbits = min(len(payload_bits), flat.size)
    for i in range(nbits):
        flat[i] = (flat[i] & 0xFE) | int(payload_bits[i])
    return flat.reshape(cover_arr.shape)


def lsb_extract(stego_arr: np.ndarray, nbits: int) -> str:
    flat = stego_arr.flatten()
    return "".join(str(int(flat[i]) & 1) for i in range(nbits))


def lsb_capacity_bits(shape) -> int:
    """LSB at 1 bit/channel-sample."""
    return int(np.prod(shape))


# ============================================================
# DCT-LSB (frequency domain, JPEG-style)
# ============================================================
def _blocks(channel):
    h, w = channel.shape
    h8, w8 = h - h % 8, w - w % 8
    for i in range(0, h8, 8):
        for j in range(0, w8, 8):
            yield i, j


_MIDFREQ = (4, 1)  # a mid-frequency coefficient (robust-ish, low visibility)


def dct_embed(cover_arr: np.ndarray, payload_bits: str) -> np.ndarray:
    """Embed bits into the sign-LSB of one mid-frequency DCT coeff per 8x8 block (Y channel)."""
    ycc = np.asarray(Image.fromarray(cover_arr).convert("YCbCr"), dtype=np.float64).copy()
    Y = ycc[:, :, 0]
    bit_i = 0
    for (i, j) in _blocks(Y):
        if bit_i >= len(payload_bits):
            break
        block = Y[i:i + 8, j:j + 8]
        D = dct(dct(block.T, norm="ortho").T, norm="ortho")
        c = D[_MIDFREQ]
        q = int(round(c / 8.0))
        q = (q & ~1) | int(payload_bits[bit_i])  # set LSB of quantized coeff
        D[_MIDFREQ] = q * 8.0
        Y[i:i + 8, j:j + 8] = idct(idct(D.T, norm="ortho").T, norm="ortho")
        bit_i += 1
    ycc[:, :, 0] = np.clip(Y, 0, 255)
    out = np.asarray(Image.fromarray(ycc.astype(np.uint8), mode="YCbCr").convert("RGB"))
    return out


def dct_extract(stego_arr: np.ndarray, nbits: int) -> str:
    Y = np.asarray(Image.fromarray(stego_arr).convert("YCbCr"), dtype=np.float64)[:, :, 0]
    bits = []
    for (i, j) in _blocks(Y):
        if len(bits) >= nbits:
            break
        block = Y[i:i + 8, j:j + 8]
        D = dct(dct(block.T, norm="ortho").T, norm="ortho")
        q = int(round(D[_MIDFREQ] / 8.0))
        bits.append(str(q & 1))
    return "".join(bits)


def dct_capacity_bits(shape) -> int:
    """One bit per 8x8 luminance block."""
    h, w = shape[0], shape[1]
    return (h // 8) * (w // 8)


# ============================================================
# Steganalysis: chi-square attack (Westfeld & Pfitzmann, 1999)
# ============================================================
def chi_square_p(arr: np.ndarray) -> float:
    """Probability that `arr` carries LSB-embedded data.

    Embedding equalises the histogram pairs (2k, 2k+1); a high p means the
    pairs are suspiciously equal (likely embedded), p≈0 means a natural image.
    """
    gray = np.asarray(Image.fromarray(arr).convert("L")).flatten()
    hist = np.bincount(gray, minlength=256).astype(np.float64)
    obs, exp = [], []
    for k in range(128):
        h0, h1 = hist[2 * k], hist[2 * k + 1]
        e = (h0 + h1) / 2.0
        if e > 4:  # only bins with enough samples
            obs.append(h0)
            exp.append(e)
    if len(obs) < 2:
        return 0.0
    obs = np.array(obs)
    exp = np.array(exp)
    stat = np.sum((obs - exp) ** 2 / exp)
    df = len(obs) - 1
    # p(embedded) = probability the observed deviation is THIS small under "no embedding"
    return float(chi2_dist.cdf(df - stat, df)) if False else float(1.0 - chi2_dist.sf(stat, df) * 0 + chi2_dist.cdf(0, df) * 0 + _embed_prob(stat, df))


def _embed_prob(stat, df):
    # Westfeld: p of embedding = 1 - CDF(chi2, df)  evaluated so that small stat -> p~1
    return float(1.0 - chi2_dist.cdf(stat, df))


def detection_accuracy(cover_arrs, stego_arrs, tau=0.5):
    """Balanced detection accuracy of the chi-square detector.

    Predicts 'stego' when chi_square_p > tau.
    Returns (balanced_accuracy_percent, tpr, tnr).
    """
    tp = sum(1 for a in stego_arrs if chi_square_p(a) > tau)
    tn = sum(1 for a in cover_arrs if chi_square_p(a) <= tau)
    tpr = tp / len(stego_arrs) if stego_arrs else 0.0
    tnr = tn / len(cover_arrs) if cover_arrs else 0.0
    bal = 100.0 * (tpr + tnr) / 2.0
    return bal, tpr, tnr


# ============================================================
# Convenience: build LSB / DCT stego from a cover with random payload
# ============================================================
def make_lsb_stego(cover_arr, rng, fill=1.0):
    nbits = int(lsb_capacity_bits(cover_arr.shape) * fill)
    bits = "".join(rng.choice("01") for _ in range(min(nbits, 200000)))
    return lsb_embed(cover_arr, bits), bits


def jpeg_roundtrip(arr, quality):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))
