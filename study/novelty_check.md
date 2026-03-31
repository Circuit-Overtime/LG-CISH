# LG-CISH: Novelty Check & Action Plan

---

## Part A: Existing Landscape (What Already Exists)

> Coverless steganography via hash mapping is well-established. Qin et al. (2019) and Meng et al. (2023) thoroughly survey selection-based methods that map message bits → perceptual hashes → image retrieval from a database. This core pipeline is not new.

LLM-based steganography exists, but almost entirely in the **text domain**:

- **LLsM** — first LLM-based linguistic steganography (fine-tunes LLaMA2)
- **OD-Stega** — LLM-driven arithmetic coding for coverless text stego
- **LLM-Stega** — black-box LLM text steganography

LLM + image steganography also exists but is **not coverless**:

- **S²LM** (Nov 2025) — embeds sentence-level secrets INTO images using LLMs. Modifies pixels, so NOT coverless.

Diffusion-based coverless methods exist:

- **DiffStega** (IJCAI 2024) — diffusion model for coverless image stego
- **SemSteDiff** (2025) — uses BLIP + LLM for key generation in coverless semantic steganography over communication channels

### Key Observation
**No existing work uses an LLM as a semantic encoding bridge in coverless image steganography.** All LLM stego → text output. All coverless image stego → no LLM. SemSteDiff uses LLM only for key generation, not message encoding. This is our gap.

---

## Part B: Weaknesses in Current algo.md That Must Be Fixed

| # | Weakness | Impact |
|---|----------|--------|
| W1 | Uses perceptual hashing (pHash) which is pixel-level, not semantic — contradicts the "semantic" claim | Reviewers will reject the semantic novelty claim |
| W2 | No concrete mechanism defined for LLM embedding → binary hash conversion | Paper is not reproducible |
| W3 | Does not address the known 18-bit/image capacity bottleneck of selection-based coverless stego | Inherits the biggest open problem without solving it |
| W4 | Image sequence may look random/suspicious to an observer | Weak against visual/behavioral steganalysis |
| W5 | Information encoded only in image identity, ignoring ordering | Leaves capacity on the table |
| W6 | No clear differentiation from SemSteDiff (2025) | Overlap risk with recent published work |

---

## Part C: Detailed TODO — Making LG-CISH Strongly Novel

### TODO 1: Replace Perceptual Hashing with CLIP Semantic Embedding Space
**Fixes:** W1, W2, W6
**Priority:** CRITICAL — without this, the paper's core claim is hollow

**What to do:**
- Remove perceptual hashing (pHash/dHash/aHash) from the pipeline entirely
- Introduce CLIP (or SigLIP) as the shared text-image embedding space
- Redefine the encoding pipeline as:
  1. LLM processes secret message → generates a semantic description per chunk
  2. CLIP text encoder maps each description → 512-dimensional vector
  3. CLIP image encoder pre-computes embeddings for every image in the database
  4. Matching is done via **cosine similarity** in CLIP space (not Hamming distance)
  5. Select top-1 image per semantic chunk
- Update Section 2 (Mathematical Formulation) with cosine similarity instead of Hamming distance
- Update Step 3, Step 4, Step 5 in Algorithm 1
- Update Step 1, Step 2 in Algorithm 2 (Decoding)
- Move CLIP from "Section 8: Possible Extensions" into the **core pipeline**

**Why this is novel:**
- All existing coverless methods use shallow features (DCT, DenseNet, pHash)
- CLIP creates a genuinely semantic shared space — a dog photo will match "canine" text, which perceptual hashing can never do
- Clearly differentiates from SemSteDiff which uses BLIP only for key extraction, not for the core message-to-image mapping

---

### TODO 2: Define LLM Semantic Compression Scheme
**Fixes:** W3
**Priority:** HIGH — directly attacks the biggest open problem in coverless stego

**What to do:**
- Add a new step between Message Preprocessing (Step 1) and Semantic Expansion (Step 2)
- The LLM acts as a **semantic compressor**: instead of tokenizing word-by-word, it extracts the minimal semantic representation
  - Example: "Meet me at the café at 3pm tomorrow" → `[action:meet, location:café, time:3pm+1d]` = 3 tokens instead of 8+
- On the receiver side, the LLM **expands** the compressed tokens back to full natural language
- Define this formally:
  - Compression: `C_LLM(M) → {s₁, s₂, ..., sₖ}` where k << |M| (token count of original message)
  - Expansion: `E_LLM({s₁, s₂, ..., sₖ}) → M'` where M' ≈ M (semantically equivalent)
- Quantify the compression ratio vs. direct tokenization
- Discuss the trade-off: higher compression = fewer images needed, but higher risk of semantic loss

**Why this is novel:**
- No coverless stego method has semantic compression — they all encode fixed-length bit strings
- Only an LLM can do this — it requires *understanding* the message
- Directly improves the bits-per-image capacity metric

---

### TODO 3: Add LLM-Driven Plausibility Filtering
**Fixes:** W4
**Priority:** HIGH — unique defense layer that only LLM-based methods can offer

**What to do:**
- After selecting candidate images (Step 5), add a new **Step 5.5: Plausibility Check**
- Pass the candidate image sequence (or their CLIP descriptions) through the LLM
- Prompt: "Does this set of images look like a plausible social media album / image gallery?"
- If the LLM scores it below a threshold → re-select using the **next-best** CLIP matches while maintaining semantic validity
- Define a plausibility score: `P(I₁, I₂, ..., Iₖ) = LLM_score(desc(I₁), desc(I₂), ..., desc(Iₖ))`
- Add constraint: `argmin cosine_distance(sᵢ, Iⱼ) subject to P(sequence) > τ`

**Why this is novel:**
- No existing coverless stego considers whether the output image sequence looks "natural" as a collection
- This is **behavioral steganalysis resistance** — a new security dimension
- Only possible because an LLM is already in the pipeline

---

### TODO 4: Add Dual-Channel Encoding (Image Identity + Permutation)
**Fixes:** W5
**Priority:** MEDIUM — boosts capacity with minimal added complexity

**What to do:**
- Currently: information encoded only in WHICH images are selected
- Enhancement: also encode bits in the ORDER of images
- For k images, the permutation encodes `floor(log₂(k!))` additional bits
  - k=5 → 5! = 120 → 6 extra bits
  - k=8 → 8! = 40320 → 15 extra bits
  - k=10 → 10! = 3628800 → 21 extra bits
- Split the secret message into two parts:
  - Part 1 → encoded via image selection (CLIP matching)
  - Part 2 → encoded via image ordering (permutation code)
- Receiver decodes both channels and concatenates
- Add this to the mathematical formulation and algorithm steps

**Why this is novel:**
- Permutation coding exists in other contexts but has not been combined with LLM-based coverless stego
- Essentially "free" capacity boost — no extra images needed

---

### TODO 5: Rewrite Mathematical Formulation for New Pipeline
**Fixes:** W2
**Priority:** HIGH — required once TODOs 1-4 are incorporated

**What to do:**
- Redefine the core mapping:
  - `f_CLIP_text: s_i → v_i ∈ R^512` (CLIP text encoding of semantic token)
  - `f_CLIP_img: I_j → u_j ∈ R^512` (CLIP image encoding, precomputed)
  - `match(s_i) = argmax_j cos(v_i, u_j)` (cosine similarity matching)
- Add semantic compression notation:
  - `C_LLM(M) → S = {s₁, ..., sₖ}` where k = compressed token count
- Add plausibility constraint:
  - `P(I*₁, ..., I*ₖ) ≥ τ`
- Add dual-channel formulation:
  - `M_total = M_selection || M_permutation`
- Update complexity analysis:
  - CLIP encoding: O(k × d) per query, O(N × d) precomputed
  - Matching: O(k × N) cosine similarity lookups (or O(k × log N) with ANN index)

---

### TODO 6: Strengthen Differentiation from Related Work
**Fixes:** W6
**Priority:** HIGH — essential for the Related Work / Comparison section

**What to do:**
- Build a comparison table:

| Feature | DiffStega | SemSteDiff | S²LM | LG-CISH (Ours) |
|---------|-----------|------------|------|-----------------|
| Coverless | Yes | Yes | No (modifies pixels) | Yes |
| Uses LLM | No | Only for key gen | Yes (embedding) | Yes (encoding + compression + plausibility) |
| Semantic matching | No (diffusion) | Partial (BLIP keys) | No (bit-level) | Yes (CLIP space) |
| Semantic compression | No | No | No | Yes |
| Plausibility filtering | No | No | No | Yes |
| Dual-channel encoding | No | No | No | Yes |
| Pixel modification | No | No | Yes | No |

- Write explicit differentiation paragraphs for each competitor
- Emphasize: LG-CISH is the **only** method where the LLM is involved in encoding, compression, AND quality assurance (plausibility)

---

### TODO 7: Update Algorithm Pseudocode
**Priority:** MEDIUM — do this after TODOs 1-5 are finalized

**What to do:**
- Rewrite Algorithm 1 (Encoding) with the new steps:
  1. Message Preprocessing (same)
  2. LLM Semantic Compression (NEW)
  3. CLIP Text Embedding (replaces semantic hash generation)
  4. CLIP Image Embedding Lookup (replaces perceptual feature extraction)
  5. Cosine Similarity Matching (replaces Hamming distance matching)
  6. Plausibility Filtering (NEW)
  7. Dual-Channel Permutation Encoding (NEW)
  8. Stego Sequence Formation (updated)
  9. Secure Transmission (same)
- Rewrite Algorithm 2 (Decoding) correspondingly
- Ensure each step has formal notation matching TODO 5

---

### TODO 8: Address Limitations Honestly
**Priority:** MEDIUM — shows maturity to reviewers

**What to do:**
- Acknowledge and discuss:
  - Database size requirement (grows with vocabulary, but CLIP reduces this vs. pHash)
  - LLM inference latency (slower than direct hash lookup)
  - Semantic compression may introduce lossy reconstruction (M' ≈ M, not M' = M)
  - CLIP model dependency — if attacker knows the CLIP model, they could attempt reverse mapping
  - Plausibility filtering increases computational cost
- For each limitation, briefly mention a mitigation or future direction

---

## Part D: Execution Order

```
Step 1 → TODO 1 (CLIP integration)         — changes the foundation
Step 2 → TODO 2 (Semantic compression)     — adds the key novel contribution
Step 3 → TODO 3 (Plausibility filtering)   — adds the defense layer
Step 4 → TODO 4 (Dual-channel encoding)    — adds capacity boost
Step 5 → TODO 5 (Rewrite math formulation) — formalizes everything
Step 6 → TODO 6 (Differentiation table)    — positions the paper
Step 7 → TODO 7 (Rewrite pseudocode)       — updates the algorithm
Step 8 → TODO 8 (Limitations)              — adds academic rigor
```

---

## Part E: One-Line Novelty Claim (After All TODOs)

> **LG-CISH is the first coverless image steganography framework that uses an LLM for semantic message compression, maps compressed tokens to images via a CLIP-based cross-modal embedding space, and employs LLM-driven plausibility filtering to ensure the stego image sequence resists both statistical and behavioral steganalysis — achieving higher capacity, true semantic matching, and natural-looking output without modifying a single pixel.**
