import os

# --- CLIP ---
CLIP_MODEL = "ViT-B/32"
EMBEDDING_DIM = 512

# --- Codebook ---
MIN_CLIP_DISTANCE = 0.85  # max cosine similarity allowed between any two codebook images
CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "codebook.npz")

# --- Image Database (DIV2K) ---
# DIV2K: 900 train + 100 validation = 1000 high-res diverse images
# Chosen for: high resolution, semantic diversity, standard in image processing papers
DATASET_NAME = "DIV2K"
DATASET_TRAIN_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DATASET_VALID_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")

# --- Crypto ---
AES_KEY_SIZE = 32  # 256-bit

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
