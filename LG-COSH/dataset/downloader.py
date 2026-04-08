"""DIV2K dataset downloader — fetches train + validation HR images."""

import os
import sys
import zipfile
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_TRAIN_URL, DATASET_VALID_URL, IMAGE_DIR

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def _download_file(url: str, dest: str):
    """Download a file with progress reporting."""
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct}% ({downloaded}/{total} bytes)", end="", flush=True)
    print()


def _extract_zip(zip_path: str, extract_to: str):
    """Extract a zip file, then remove the zip."""
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    os.remove(zip_path)
    print(f"  Removed {zip_path}")


def _collect_images(root_dir: str) -> list[str]:
    """Recursively collect all image files under root_dir, sorted."""
    images = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                images.append(os.path.join(dirpath, fname))
    return sorted(images)


def _flatten_images(src_dir: str, dest_dir: str):
    """Move all images from nested subdirs into a single flat directory."""
    os.makedirs(dest_dir, exist_ok=True)
    images = _collect_images(src_dir)
    moved = 0
    for img_path in images:
        fname = os.path.basename(img_path)
        dest_path = os.path.join(dest_dir, fname)
        # handle name collisions by prefixing
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(fname)
            dest_path = os.path.join(dest_dir, f"{base}_dup{moved}{ext}")
        os.rename(img_path, dest_path)
        moved += 1
    return moved


def download_dataset(force: bool = False) -> list[str]:
    """Download DIV2K train + validation images to IMAGE_DIR.

    Returns sorted list of image paths.
    Skips download if images already exist (unless force=True).
    """
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # Check if already downloaded
    existing = _collect_images(IMAGE_DIR)
    if len(existing) >= 900 and not force:
        print(f"Dataset already present: {len(existing)} images in {IMAGE_DIR}")
        return existing

    # Temporary extraction directory
    tmp_dir = os.path.join(os.path.dirname(IMAGE_DIR), "_tmp_div2k")
    os.makedirs(tmp_dir, exist_ok=True)

    for url in [DATASET_TRAIN_URL, DATASET_VALID_URL]:
        zip_name = url.split("/")[-1]
        zip_path = os.path.join(tmp_dir, zip_name)

        if not os.path.exists(zip_path):
            _download_file(url, zip_path)

        _extract_zip(zip_path, tmp_dir)

    # Flatten all extracted images into IMAGE_DIR
    moved = _flatten_images(tmp_dir, IMAGE_DIR)
    print(f"Moved {moved} images to {IMAGE_DIR}")

    # Cleanup temp dir
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    images = _collect_images(IMAGE_DIR)
    print(f"Dataset ready: {len(images)} images")
    return images


def list_images() -> list[str]:
    """Return sorted list of all images currently in IMAGE_DIR."""
    return _collect_images(IMAGE_DIR)


if __name__ == "__main__":
    paths = download_dataset()
    print(f"\nTotal images: {len(paths)}")
    if paths:
        print(f"First: {paths[0]}")
        print(f"Last:  {paths[-1]}")
