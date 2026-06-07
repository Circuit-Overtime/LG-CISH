import os

# --- CLIP ---
CLIP_MODEL = "ViT-B/32"
EMBEDDING_DIM = 512

# --- Codebook ---
MIN_CLIP_DISTANCE = 0.85  # max cosine similarity allowed between any two codebook images
CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "codebook.npz")

# --- Image Database ---
# Combined fixed-size database drawn from standard image-processing benchmark
# suites (UCID, Kodak, USC-SIPI), each image normalized to a fixed 512x512 PNG.
# With N images the codebook encodes log2(N) bits per image via base-N positional
# coding (see encoder/encode.py). Normalize the image folder with
# dataset/normalize.py, then rebuild with `python main.py build`.
DATASET_NAME = "UCID + Kodak + USC-SIPI (mixed) — 512x512 PNG"
DATASET_TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DATASET_VALID_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
# repo-root images/ (LG-COSH/.. -> project root -> images)
IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")

# --- Crypto ---
AES_KEY_SIZE = 32  # 256-bit


def generate_demo_key() -> bytes:
    """Deterministic 256-bit key for CLI demos so encode/decode (separate
    invocations) share the same key. Real use should call
    crypto.aes_layer.generate_key() and exchange the key out-of-band."""
    import hashlib
    return hashlib.sha256(b"LG-CISH-demo-key").digest()

# --- Plausibility (Pollinations API - OpenAI compatible) ---
LLM_BASE_URL = "https://gen.pollinations.ai"
LLM_MODEL = "openai"  # GPT-5 Mini — fast & balanced
PLAUSIBILITY_THRESHOLD = 0.5

# --- Device (lazy import — torch only needed in Phase 2+) ---
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
