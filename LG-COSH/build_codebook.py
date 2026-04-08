"""Build the shared codebook from DIV2K images.

Usage:
    source venv/bin/activate
    python build_codebook.py
"""

import sys
sys.path.insert(0, "LG-COSH")

from codebook.builder import build_codebook

if __name__ == "__main__":
    print("Building codebook from DIV2K images...\n")
    cb = build_codebook()

    print(f"\n{'='*50}")
    print(f"  Codebook Summary")
    print(f"{'='*50}")
    print(f"  Total images:     {cb['n_images']}")
    print(f"  Chunk size:       {cb['chunk_size']} bits/image")
    print(f"  Embeddings shape: {cb['embeddings'].shape}")
    print(f"  First path:       {cb['paths'][0]}")
    print(f"  Last path:        {cb['paths'][-1]}")
    print(f"{'='*50}")
    print("Done! Codebook saved to LG-COSH/data/codebook.npz")
