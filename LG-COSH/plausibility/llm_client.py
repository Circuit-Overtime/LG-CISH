"""Minimal Pollinations client for the LLM-guided layer (captioning, image
generation, plausibility scoring).

Auth: reads POLLINATIONS_KEY from the environment or the project-root .env.
Cloudflare blocks the default urllib user-agent, so every request sends a
browser-like UA. All generation/vision endpoints require the key; only model
listing is anonymous.
"""

import base64
import json
import os
import urllib.parse
import urllib.request

BASE = "https://gen.pollinations.ai"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
TEXT_MODEL = "gemini-fast"   # Gemini 2.5 Flash Lite — fast, multimodal (vision)
IMAGE_MODEL = "flux"


def _key():
    k = os.environ.get("POLLINATIONS_KEY")
    if k:
        return k.strip()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = os.path.join(root, ".env")
    if os.path.exists(env):
        for line in open(env):
            line = line.strip()
            if line.startswith("POLLINATIONS_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("POLLINATIONS_KEY not found in env or .env")


def _data_uri(path):
    ext = os.path.splitext(path)[1].lstrip(".").lower() or "png"
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    with open(path, "rb") as f:
        return f"data:image/{mime};base64," + base64.b64encode(f.read()).decode()


def chat(messages, model=TEXT_MODEL, seed=0, temperature=0.7, timeout=90):
    body = json.dumps({"model": model, "messages": messages,
                       "seed": seed, "temperature": temperature}).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json",
                 "User-Agent": _UA, "Accept": "*/*"})
    d = json.load(urllib.request.urlopen(req, timeout=timeout))
    return d["choices"][0]["message"]["content"].strip()


def caption_image(path, model=TEXT_MODEL, seed=0):
    """Return a short noun-phrase description of a local image."""
    content = [
        {"type": "text", "text": "Describe the main subject of this photo in a short "
         "noun phrase (3-6 words), no punctuation. Example: 'two scarlet macaws on a branch'."},
        {"type": "image_url", "image_url": {"url": _data_uri(path)}},
    ]
    return chat([{"role": "user", "content": content}], model=model, seed=seed, temperature=0.2)


def generate_image(prompt, out_path, model=IMAGE_MODEL, seed=0, size=512, timeout=120):
    """Generate an image from a text prompt and save it to out_path."""
    url = (f"{BASE}/image/{urllib.parse.quote(prompt)}"
           f"?model={model}&width={size}&height={size}&seed={seed}&nologo=true"
           f"&key={urllib.parse.quote(_key())}")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    raw = urllib.request.urlopen(req, timeout=timeout).read()
    with open(out_path, "wb") as f:
        f.write(raw)
    return out_path


if __name__ == "__main__":
    import sys
    print("caption:", caption_image(sys.argv[1]))
