import os

# --- CLIP ---
CLIP_MODEL = "ViT-B/32"
EMBEDDING_DIM = 512

# --- Codebook ---
MIN_CLIP_DISTANCE = 0.85  # max cosine similarity allowed between any two codebook images
CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "codebook.npz")

# --- Image Database ---
# Combined fixed-size database: the 24-image Kodak Lossless True Color suite plus
# a 6-image curated DIV2K subset (30 images total), each resized to a fixed
# 512x512 canvas. With N images the codebook encodes log2(N) bits per image via
# base-N positional coding (see encoder/encode.py). Rebuild the image folder with
# dataset/build_combined.py.
DATASET_NAME = "Kodak (24) + DIV2K (6) — 30 images @ 512x512"
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
