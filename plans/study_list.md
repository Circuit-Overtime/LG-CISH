# LG-CISH Study List

## Language: Python — No Contest

Every component of this pipeline has first-class Python support.

| Component | Python Library | Alternative Language? |
|-----------|---------------|----------------------|
| CLIP embeddings | `openai/clip`, `transformers` | Rust/C++ bindings exist but immature |
| LLM inference | `transformers`, `ollama`, OpenAI API | API calls work from anywhere, but local inference = Python |
| Image processing | `Pillow`, `OpenCV` | C++ OpenCV exists but slower to develop with |
| Vector similarity search | `faiss`, `numpy` | FAISS core is C++ but Python is the interface |
| Compression | `zlib` (built-in) | Every language has this |
| Encryption | `cryptography`, `pycryptodome` | Every language has this |
| Bit manipulation | Built-in | Every language has this |

---

## Algorithms to Study — By Pipeline Stage

### Stage 1: Codebook Construction

**CLIP (Contrastive Language-Image Pretraining)**
- Paper: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision" (2021)
- What to study: How it maps text and images into a shared 512-dim vector space using contrastive loss
- Why: This is your core matching engine

**FAISS (Facebook AI Similarity Search)**
- What to study: Approximate Nearest Neighbor (ANN) search, IVF (Inverted File Index), HNSW (Hierarchical Navigable Small World) graphs
- Why: With 1000+ images, brute-force cosine similarity is fine. With 100K+ images, you need FAISS for fast lookup
- Key concept: **Cosine similarity** — `sim(a,b) = (a·b) / (||a|| × ||b||)`

**Minimum Spanning Tree / Pairwise Distance Filtering**
- What to study: How to compute pairwise cosine distances across all database images and prune those below threshold
- Why: Enforcing minimum CLIP distance between codebook entries to prevent decoding errors

---

### Stage 2: Message → Bits

**Huffman Coding**
- What to study: Variable-length prefix codes, frequency-based compression
- Why: Understand how lossless compression reduces bitstream length before chunking

**DEFLATE (zlib)**
- What to study: LZ77 + Huffman combination
- Why: This is the practical compressor you'll use. Python's `zlib.compress()` / `zlib.decompress()`

**AES-256 (Advanced Encryption Standard)**
- What to study: Symmetric block cipher, CBC/CTR modes, key derivation (PBKDF2)
- Why: Optional encryption layer before encoding. Library: `pycryptodome`

**CRC-32 (Cyclic Redundancy Check)**
- What to study: Polynomial division for error detection
- Why: Append to bitstream so receiver can verify integrity. Python's `zlib.crc32()`

---

### Stage 3: Bits → Images (Encoding)

**Binary Chunking / Fixed-Length Block Coding**
- What to study: Splitting a bitstream into fixed-size blocks, padding strategies
- Why: This is how you map bits to image indices. Simple but must handle edge cases (last chunk padding)

**Lehmer Code (Permutation Encoding)**
- What to study: How to encode a permutation of n elements as a single integer (and decode it back)
- Why: This is the dual-channel encoding — the ORDER of images encodes extra bits
- Key formula: permutation of k items = `floor(log₂(k!))` extra bits
- Also study: **Factorial number system** for efficient permutation ↔ integer conversion

---

### Stage 4: Plausibility Filtering

**LLM Prompting / In-Context Learning**
- What to study: Zero-shot classification, scoring/ranking with LLMs, structured output
- Why: You prompt the LLM to score image sequences for naturalness
- Practical: Use `ollama` for local LLM or OpenAI API for cloud

**CLIP-based Image Captioning (BLIP / BLIP-2)**
- What to study: How to generate text descriptions from images
- Why: To describe the selected images before passing them to the LLM for plausibility evaluation
- Library: `transformers` with Salesforce BLIP-2

---

### Stage 5: Decoding (Receiver Side)

**K-Nearest Neighbors (exact + approximate)**
- What to study: Brute-force KNN vs. ANN (FAISS IVFFlat, HNSW)
- Why: Receiver computes CLIP embedding of received image → finds nearest codebook entry
- Key: Understand **distance thresholds** — when is a match "confident enough"

**Cosine Similarity vs. Euclidean Distance vs. Hamming Distance**
- What to study: When to use each, how they behave in high-dimensional spaces
- Why: CLIP space works best with cosine similarity. Understand why Hamming (used by pHash) is inferior here

---

### Background Study (Understand What You're Replacing)

**Perceptual Hashing (pHash, dHash, aHash)**
- What to study: DCT-based hashing, difference hashing, average hashing
- Why: You need to understand these to explain WHY CLIP is better in your paper
- Library: `imagehash` (Python)

**Locality-Sensitive Hashing (LSH)**
- What to study: Hash functions that preserve similarity, random hyperplane projections
- Why: An alternative to FAISS for approximate NN. Also relevant to understanding hash-based coverless stego literature

---

## Reading List (Prioritized)

| Priority | Algorithm/Paper | Why |
|----------|----------------|-----|
| 1 | CLIP — Radford et al. (2021) | Your core matching engine |
| 2 | FAISS documentation | Your search/lookup engine |
| 3 | Lehmer code / factorial number system | Permutation encoding for dual-channel |
| 4 | zlib/DEFLATE | Lossless compression you'll actually use |
| 5 | AES-256 (NIST standard) | Encryption layer |
| 6 | CRC-32 | Error detection |
| 7 | BLIP-2 — Li et al. (2023) | Image captioning for plausibility check |
| 8 | pHash — Zauner (2010) | Understand what you're replacing |
| 9 | LSH — Indyk & Motwani (1998) | Background for hash-based methods |
| 10 | Arithmetic coding — Witten et al. (1987) | Advanced: understand OD-Stega's approach |
