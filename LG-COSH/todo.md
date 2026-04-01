# LG-COSH Implementation Checklist

---

## Phase 1: Foundation (Core building blocks — no AI, just math)

These modules work independently. Each one can be tested on its own.

- [x] **Project setup**
  - [x] Create venv
  - [x] Install dependencies from requirements.txt
  - [ ] Verify `import torch`, `import clip`, `import numpy` all work *(deferred to Phase 2 — not needed yet)*

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
  - [ ] Download ~1024 diverse images from a public source
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
  - [ ] `caption_images(image_paths)` → list of text descriptions (using CLIP or BLIP)
  - [ ] `check_plausibility(captions, api_url)` → float score 0.0–1.0
    - [ ] Send captions to Pollinations API
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

- [ ] **Robustness testing (simulating real-world channels)**
  - [ ] JPEG compress stego images at 50% quality → re-decode → verify
  - [ ] JPEG compress at 30% quality → re-decode → verify
  - [ ] Resize to 50% → upscale back → re-decode → verify
  - [ ] Screenshot simulation (save as PNG from screen) → re-decode → verify
  - [ ] Log: at what compression level does decoding start failing?
  - [ ] Measure: CLIP distance margin at each degradation level

- [ ] **Capacity benchmarking**
  - [ ] Measure bits-per-image for different database sizes (256, 512, 1024, 4096)
  - [ ] Measure with zlib compression: effective bits-per-image
  - [ ] Measure with dual-channel: effective bits-per-image
  - [ ] Compare against baseline (raw pHash method from literature)
  - [ ] Generate comparison table for the paper

- [ ] **Security analysis**
  - [ ] Verify: without the codebook, brute-force is infeasible (2^N possibilities)
  - [ ] Verify: without AES key, even with codebook, message is encrypted
  - [ ] Verify: CRC detects single-bit tampering
  - [ ] Test against steganalysis: run a CNN detector on stego images → detection rate should be ~0% (no pixels modified)

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
| Paper metrics collected | Phase 4 | capacity table + robustness table generated |
