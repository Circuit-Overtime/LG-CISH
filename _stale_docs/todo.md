# LG-COSH Implementation Checklist

---

## Phase 1: Foundation (Core building blocks — no AI, just math) — COMPLETE

These modules work independently. Each one can be tested on its own.

- [x] **Project setup**
  - [x] Create venv
  - [x] Install dependencies from requirements.txt
  - [ ] Verify `import torch`, `import clip`, `import numpy` all work *(deferred to Phase 2)*

- [x] **config.py** — Central constants
  - [x] CLIP model name, embedding dimension
  - [x] Min CLIP distance threshold
  - [x] File paths (codebook, image dir)
  - [x] Device detection (cuda/cpu) — lazy import, no torch dependency
  - [x] Pollinations API endpoint

- [x] **bitstream/converter.py** — Message ↔ Bits ↔ Chunks (13/13 tests pass)
  - [x] `message_to_bytes(msg)` → raw UTF-8 bytes
  - [x] `bytes_to_bits(data)` → binary string ("01001000...")
  - [x] `bits_to_chunks(bits, chunk_size)` → list of integer indices + padding count
  - [x] `chunks_to_bits(chunks, chunk_size, padding)` → binary string
  - [x] `bits_to_bytes(bits)` → raw bytes
  - [x] `bytes_to_message(data)` → string
  - [x] `compress(data)` / `decompress(data)` — zlib wrappers
  - [x] `encode_message(msg, chunk_size)` — full encode helper
  - [x] `decode_chunks(chunks, metadata)` — full decode helper
  - [x] Round-trip test: ASCII, Unicode, emoji, empty string, 500-char — all pass

- [x] **crypto/aes_layer.py** — Encryption + Integrity (8/8 tests pass)
  - [x] `generate_key()` → random 256-bit key
  - [x] `encrypt(plaintext_bytes, key)` → ciphertext bytes (AES-256-CBC, random IV prepended)
  - [x] `decrypt(ciphertext_bytes, key)` → plaintext bytes
  - [x] `compute_crc(data)` → 4-byte CRC-32
  - [x] `verify_crc(data, expected_crc)` → bool
  - [x] `wrap(data, key)` / `unwrap(data, key)` — CRC + optional AES in one call
  - [x] Round-trip test: encrypt → decrypt = exact match
  - [x] CRC test: compute → verify = True, tamper → verify = False
  - [x] Wrong key test: correctly raises exception

- [x] **tests/test_phase1.py** — 21/21 tests passing

---

## Phase 2: Integration (Connect the AI pieces to the foundation)

These modules depend on Phase 1 being solid.

- [ ] **clip_engine/embedder.py** — CLIP Model Wrapper
  - [ ] `load_model()` → cached CLIP model + preprocess function
  - [ ] `embed_image(path)` → 512-dim normalized numpy vector
  - [ ] `embed_images_batch(paths)` → (N, 512) matrix
  - [ ] `cosine_similarity(a, b)` → float
  - [ ] `find_nearest(query_vec, codebook_matrix)` → (index, score)
  - [ ] Test: embed the same image twice → cosine similarity = 1.0
  - [ ] Test: embed two very different images → cosine similarity < 0.8

- [ ] **dataset/downloader.py** — Sample Image Database
  - [ ] Download dataset: use DIV2K or a public subset (~1024 diverse images)
  - [ ] Save to `data/images/` as individual files (PNG/JPG)
  - [ ] Return sorted list of image paths
  - [ ] Verify: folder contains 1024+ images, all loadable by PIL

- [ ] **codebook/pruner.py** — Remove Similar Images
  - [ ] `prune_similar(embeddings, paths, threshold)` → cleaned embeddings + paths
  - [ ] Compute pairwise cosine similarity matrix
  - [ ] Drop one image from any pair where similarity > threshold
  - [ ] Log how many images were pruned
  - [ ] Test: inject two identical images → one gets pruned

- [ ] **codebook/builder.py** — Build the Shared Codebook
  - [ ] `build_codebook(image_dir)` → saves codebook.npz
    - [ ] Scan image directory for all images
    - [ ] Compute CLIP embeddings for all (batch)
    - [ ] Run pruner to remove duplicates
    - [ ] Assign index 0..N-1 to remaining images
    - [ ] Save: paths array + embeddings matrix + metadata (N, chunk_size)
  - [ ] `load_codebook(path)` → dict with paths, embeddings, chunk_size
  - [ ] Test: build → load → verify shapes match

---

## Phase 3: The Full Pipeline (Encoder + Decoder wired together)

This is where everything connects into the lossless chain.

- [ ] **encoder/encode.py** — Message → Image Sequence
  - [ ] `encode(message, codebook_path, key=None)` → list of image paths
  - [ ] Pipeline:
    1. [ ] message → bytes
    2. [ ] (optional) AES encrypt bytes
    3. [ ] Append CRC-32 to byte stream
    4. [ ] Compress with zlib
    5. [ ] bytes → bits → chunks (using codebook's chunk_size)
    6. [ ] Each chunk index → look up image path in codebook
    7. [ ] Return ordered image path list
  - [ ] Test: encode "HELLO" → get list of image paths, count makes sense

- [ ] **decoder/decode.py** — Image Sequence → Message
  - [ ] `decode(image_paths, codebook_path, key=None)` → string
  - [ ] Pipeline:
    1. [ ] For each image → compute CLIP embedding
    2. [ ] Find nearest neighbor in codebook → get index
    3. [ ] Indices → chunks → bits → bytes
    4. [ ] Decompress with zlib
    5. [ ] Verify CRC-32 (raise error if mismatch)
    6. [ ] (optional) AES decrypt
    7. [ ] bytes → message string
  - [ ] Test: decode(encode("HELLO")) == "HELLO"

- [ ] **main.py** — CLI Entry Point
  - [ ] `build` command: build codebook from image folder
  - [ ] `encode` command: encode a message, output image paths
  - [ ] `decode` command: decode image sequence back to message
  - [ ] `demo` command: full round-trip (encode + decode + verify match)
  - [ ] Print clear output at each step

- [ ] **End-to-end lossless verification**
  - [ ] Test: "HELLO WORLD" round-trip = exact match
  - [ ] Test: Unicode message "こんにちは" round-trip = exact match
  - [ ] Test: Long message (500+ chars) round-trip = exact match
  - [ ] Test: Empty string edge case
  - [ ] Test: With AES encryption enabled round-trip = exact match

---

## Phase 4: Advanced — Novelty Pushing Features

These are what make the paper novel. Build only after Phase 3 is fully working.

- [ ] **plausibility/checker.py** — LLM Plausibility Filtering
  - [ ] Load API key from `../.env` (POLLINATIONS_TOKEN)
  - [ ] API: `POST https://gen.pollinations.ai/v1/chat/completions` with Bearer token
  - [ ] Model: `openai` (GPT-5 Mini — fast & balanced)
  - [ ] `caption_images(image_paths)` → list of text descriptions (using CLIP zero-shot or BLIP)
  - [ ] `check_plausibility(captions)` → float score 0.0–1.0
    - [ ] Send captions to Pollinations chat endpoint
    - [ ] LLM scores how natural the sequence looks
  - [ ] `suggest_reselection(indices, scores, codebook)` → alternative indices
    - [ ] For low-scoring images, pick next-best codebook match
    - [ ] All alternatives encode the same bits (lossless preserved)
  - [ ] Integrate into encoder as optional step after index selection
  - [ ] Test: encode with plausibility on → still decodes correctly

- [ ] **Dual-channel permutation encoding**
  - [ ] `encode_permutation(data_bits, k)` → permutation of k items
    - [ ] Use Lehmer code / factorial number system
    - [ ] Extra bits encoded in ordering: floor(log2(k!)) bits
  - [ ] `decode_permutation(ordering, k)` → data_bits
  - [ ] Split message into two parts: identity bits + permutation bits
  - [ ] Integrate into encoder/decoder
  - [ ] Test: round-trip with dual-channel = exact match
  - [ ] Measure: capacity improvement vs. single-channel

---

## Phase 5: Springer Paper — Experimental Results & Evaluation

Follows Springer standard hierarchy: **Section 4. Experimental Results and Performance Evaluation**

All scripts go in `evaluation/` folder. Each script outputs tables/figures for the paper.

### 5.1 Experimental Setup (Section 4.1)

- [ ] **evaluation/setup.py** — Document and print experimental config
  - [ ] Dataset: DIV2K (or BOSSBase/ImageNet subset) — log name, size, source
  - [ ] Number of images in codebook after pruning
  - [ ] CLIP model: ViT-B/32, embedding dim 512
  - [ ] LLM used: GPT-5 Mini via Pollinations API
  - [ ] Hash size / chunk size (e.g., 10-bit = 1024 images)
  - [ ] AES-256-CBC parameters
  - [ ] Hardware: log CPU, GPU, RAM
  - [ ] Generate a **figure** showing sample secret messages + their encoded image sequences

### 5.2 Qualitative Results (Section 4.2) — Visual Proof

- [ ] **evaluation/qualitative.py** — Generate figures for the paper
  - [ ] **Figure Set 1:** Original message → generated image sequence (grid of images)
    - [ ] Short message example (e.g., "HELLO")
    - [ ] Medium message example
    - [ ] Long message example
  - [ ] **Figure Set 2:** Image sequence → reconstructed message (show exact match)
  - [ ] **Figure Set 3:** Failure cases
    - [ ] What happens when images are heavily corrupted
    - [ ] What happens with extreme JPEG compression
    - [ ] What happens when codebook is mismatched
  - [ ] Save all figures as high-res PNG for paper inclusion
  - [ ] Write captions: "Semantic-to-image mapping results demonstrating accurate reconstruction without pixel modification"

### 5.3 Quantitative Evaluation (Section 4.3) — Core Acceptance Factor

- [ ] **evaluation/quantitative.py** — Generate result tables

  - [ ] **Table: Reconstruction Accuracy**
    ```
    Message Length | Messages Tested | Accuracy (%) | BER
    Short (≤50 chars)    | 100  |     |
    Medium (50-200)      | 100  |     |
    Long (200-1000)      | 100  |     |
    ```

  - [ ] **Table: Payload Capacity Comparison**
    ```
    Method          | Capacity (bits/image) | Pixel Distortion
    LSB             |                       | Yes
    DCT             |                       | Yes
    DWT-DCT         |                       | Yes
    Proposed LG-CISH|                       | None (coverless)
    ```

  - [ ] **Table: Computational Time**
    ```
    Stage               | Time (ms)
    Codebook Build      |
    CLIP Embedding      |
    Index Lookup        |
    Full Encode         |
    Full Decode         |
    ```

  - [ ] Metrics to compute:
    - [ ] **Payload Capacity** (bits/image and bits/sequence)
    - [ ] **Semantic Reconstruction Accuracy** (%) — exact match rate over N trials
    - [ ] **Hash Matching Accuracy** — % of images correctly identified by CLIP NN
    - [ ] **Bit Error Rate (BER)** — bit-level error rate (should be 0 for lossless)
    - [ ] **Retrieval Precision/Recall** — how often CLIP top-1 is the correct image

### 5.4 Robustness Analysis (Section 4.4) — Highly Critical

- [ ] **evaluation/robustness.py** — Simulate real-world channel attacks

  - [ ] **JPEG compression attacks:**
    - [ ] Quality 90%, 70%, 50%, 30% → re-decode → log accuracy
  - [ ] **Noise attacks:**
    - [ ] Gaussian noise (sigma = 5, 10, 20, 30)
    - [ ] Salt & pepper noise (density = 0.01, 0.05, 0.10)
  - [ ] **Geometric attacks:**
    - [ ] Resize to 50% → upscale back → re-decode
    - [ ] Resize to 25% → upscale back → re-decode
    - [ ] Center crop 90%, 80%, 70% → re-decode
  - [ ] **Format conversion:**
    - [ ] PNG → JPEG → re-decode
    - [ ] PNG → WebP → re-decode

  - [ ] **Table: Robustness Results**
    ```
    Attack                | Accuracy (%) | BER   | CLIP Distance Margin
    No attack (baseline)  |              |       |
    JPEG 90%              |              |       |
    JPEG 70%              |              |       |
    JPEG 50%              |              |       |
    JPEG 30%              |              |       |
    Gaussian σ=10         |              |       |
    Gaussian σ=20         |              |       |
    Salt & Pepper 0.05    |              |       |
    Resize 50%            |              |       |
    Crop 80%              |              |       |
    ```

  - [ ] Log: at what degradation level does decoding start failing
  - [ ] Measure: CLIP distance margin at each level

### 5.5 Security & Steganalysis Resistance (Section 4.5)

- [ ] **evaluation/security.py** — Prove undetectability

  - [ ] **CNN steganalysis test:**
    - [ ] Run SRNet or XuNet (or a simpler CNN detector) on stego images
    - [ ] Show detection accuracy ≈ 50% (random chance — no better than guessing)
    - [ ] Compare against LSB-embedded images (detection should be high)

  - [ ] **Table: Steganalysis Detection Rates**
    ```
    Method          | SRNet Detection (%) | XuNet Detection (%)
    LSB             |                     |
    DCT             |                     |
    Proposed LG-CISH|        ~50%         |        ~50%
    ```

  - [ ] **Keyspace analysis:**
    - [ ] Codebook permutations: N! possible orderings
    - [ ] AES-256: 2^256 key combinations
    - [ ] Total keyspace = N! × 2^256

  - [ ] **CRC integrity test:**
    - [ ] Flip 1 bit in encoded data → CRC catches it
    - [ ] Flip random bits → measure detection rate (should be ~100%)

### 5.6 Ablation Study (Section 4.6) — Very Important for Novelty

- [ ] **evaluation/ablation.py** — Show each component is necessary

  - [ ] **Table: Ablation Results**
    ```
    Variant                          | Accuracy (%) | Robustness (JPEG 50%) | BER
    Full LG-CISH (proposed)          |              |                       |
    Without CLIP (use pHash instead) |              |                       |
    Without zlib compression         |              |                       |
    Without CRC integrity check      |              |                       |
    Without AES encryption           |              |                       |
    Without plausibility filtering   |              |                       |
    ```

  - [ ] For "without CLIP" variant:
    - [ ] Implement a simple pHash-based matcher for comparison
    - [ ] Run same robustness tests → show CLIP is more robust
  - [ ] For each ablation: explain what breaks and why the component matters

### 5.7 Comparative Analysis (Section 4.7) — Mandatory

- [ ] **evaluation/comparison.py** — Compare against baselines

  - [ ] **Baselines to compare:**
    - [ ] LSB (Least Significant Bit) — classic pixel-modification method
    - [ ] DCT-based steganography — frequency domain method
    - [ ] DWT-DCT hybrid — wavelet + frequency
    - [ ] Existing coverless methods (from references — Qin et al., Meng et al.)

  - [ ] **Comparison metrics:**
    - [ ] Capacity (bits/image)
    - [ ] Robustness (accuracy after JPEG 50%)
    - [ ] Steganalysis detection rate
    - [ ] Computational time
    - [ ] Pixel distortion (PSNR/SSIM — ours = ∞ since no modification)

  - [ ] **Graphs (save as figures):**
    - [ ] Accuracy vs. Payload capacity (line chart)
    - [ ] Robustness vs. Attack strength (line chart, multiple methods)
    - [ ] Bar chart: detection rates across methods

### 5.8 Statistical Validation (Section 4.8) — Often Missed → Causes Rejection

- [ ] **evaluation/statistics.py** — Prove results are statistically significant

  - [ ] Run each experiment N times (N ≥ 30) with different messages
  - [ ] Report: **Mean ± Standard Deviation** for all metrics
  - [ ] Run **t-test** (or ANOVA if >2 groups) comparing LG-CISH vs. each baseline
  - [ ] Report **p-values** — must show p < 0.05 for significance claim
  - [ ] Example statement: "The proposed method shows statistically significant improvement (p < 0.05) over existing approaches"
  - [ ] Generate a summary table with confidence intervals

---

## Phase 6: Paper Figures & Export

- [ ] **evaluation/generate_all.py** — Master script that runs everything
  - [ ] Runs all evaluation scripts in order
  - [ ] Outputs all tables as LaTeX-formatted strings (for Springer template)
  - [ ] Saves all figures to `evaluation/figures/`
  - [ ] Generates a summary report with all numbers

- [ ] **Minimum paper deliverables:**
  - [ ] At least 6–8 tables (reconstruction accuracy, capacity, time, robustness, security, ablation, comparison, statistics)
  - [ ] At least 4–5 figures (qualitative examples, robustness graph, comparison graphs, failure cases)
  - [ ] Failure analysis section with visual examples
  - [ ] All baselines from 2023–2025
  - [ ] Real-world applicability demonstration (encode → send via WhatsApp/Telegram → decode)

---

## API & Environment Notes

- **Pollinations API:** `POST https://gen.pollinations.ai/v1/chat/completions`
- **Auth:** Bearer token from `../.env` → `POLLINATIONS_TOKEN`
- **Text model:** `openai` (GPT-5 Mini — fast, free tier)
- **Image gen (if needed):** `flux` via `POST /v1/images/generations`
- **No API needed** for Phases 1–3. Only Phase 4 plausibility + Phase 5 need API.

---

## Milestone Checklist

| Milestone | Depends On | Proof |
|-----------|-----------|-------|
| Bitstream round-trip works | Phase 1 | **DONE** — 13/13 tests pass |
| AES round-trip works | Phase 1 | **DONE** — 8/8 tests pass |
| CLIP embeddings load | Phase 2 | embed_image returns 512-dim vector |
| Codebook builds successfully | Phase 2 | codebook.npz exists, shapes correct |
| Encode produces image list | Phase 3 | list of N image paths returned |
| Decode recovers exact message | Phase 3 | decode(encode(msg)) == msg |
| Demo command works end-to-end | Phase 3 | `python main.py demo` prints success |
| Plausibility filter works | Phase 4 | score returned, re-selection doesn't break decoding |
| Survives JPEG 50% compression | Phase 4 | decode still correct after compression |
| Reconstruction accuracy table | Phase 5 | table with accuracy % and BER for 3 message lengths |
| Robustness table complete | Phase 5 | table with 10+ attack scenarios, all measured |
| Steganalysis resistance proven | Phase 5 | detection ≈ 50% for proposed vs. >90% for LSB |
| Ablation table complete | Phase 5 | 6 variants tested, full model wins |
| Comparison graphs generated | Phase 5 | accuracy vs. payload + robustness vs. attack graphs |
| Statistical significance shown | Phase 5 | p < 0.05 for all key comparisons |
| All figures exported | Phase 6 | 4-5 high-res PNGs ready for Springer template |
| All tables as LaTeX | Phase 6 | 6-8 tables formatted for paper |
