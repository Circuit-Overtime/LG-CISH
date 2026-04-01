import os

# --- CLIP ---
CLIP_MODEL = "ViT-B/32"
EMBEDDING_DIM = 512

# --- Codebook ---
MIN_CLIP_DISTANCE = 0.85  # max cosine similarity allowed between any two codebook images
CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "data", "codebook.npz")

# --- Image Database ---
IMAGE_DIR = os.path.join(os.path.dirname(__file__), "data", "images")

# --- Crypto ---
AES_KEY_SIZE = 32  # 256-bit

# --- Plausibility (Pollinations API - OpenAI compatible) ---
LLM_API_URL = "https://text.pollinations.ai/openai"
LLM_MODEL = "openai"
PLAUSIBILITY_THRESHOLD = 0.5

# --- Device (lazy import — torch only needed in Phase 2+) ---
def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
