# Research Gaps & How LG-CISH Addresses Them

---

## Gap 1: Low Capacity in Coverless Steganography
**Source:** Qin et al. [1], Meng et al. [5]
**Problem:** Selection-based coverless methods map fixed-length bit groups to image hashes. Capacity is limited to ~18 bits/image. Hiding even a short sentence requires dozens of images, and the database size grows exponentially with bit-length.
**LG-CISH Fix:** Lossless compression (zlib) on the bitstream before chunking reduces the number of images needed. Dual-channel encoding (image identity + permutation order) squeezes extra bits out of the same image count for free.

---

## Gap 2: Shallow Feature Matching Breaks Under Real-World Conditions
**Source:** Kadhim et al. [2], Kaur et al. [6], Mandal et al. [7]
**Problem:** Existing coverless methods rely on perceptual hashing (pHash, dHash) or shallow CNN features (DCT, DenseNet). These break when images are compressed, resized, or screenshotted — exactly what happens on every social media and messaging platform.
**LG-CISH Fix:** CLIP replaces perceptual hashing. CLIP embeddings are robust to JPEG compression, resizing, cropping, and format conversion. The receiver can still identify the correct image even after heavy platform processing.

---

## Gap 3: No Semantic Understanding in the Encoding Process
**Source:** Qin et al. [1], Meng et al. [5]
**Problem:** All existing coverless methods treat the message as a raw bitstream. There is zero understanding of what the message means. The mapping from bits to images is arbitrary — a photo of a dog might encode "1011" with no semantic connection. This makes the system brittle and the image selection random-looking.
**LG-CISH Fix:** An LLM is introduced into the pipeline. While the core encoding remains bit-based (for lossless guarantee), the LLM organizes the codebook semantically and performs plausibility filtering so the output image sequence looks natural rather than random.

---

## Gap 4: Stego Output Looks Suspicious
**Source:** Kombrink et al. [3], Apau et al. [9]
**Problem:** Even though coverless methods don't modify pixels (defeating pixel-level steganalysis), the selected image sequences can still look unnatural. A random collection of unrelated images raises suspicion under behavioral/visual analysis. No existing method addresses this.
**LG-CISH Fix:** LLM-driven plausibility filtering evaluates the image sequence before transmission. If the sequence looks unnatural, alternative images (encoding the same bits) are selected to form a coherent-looking set. This is a defense layer that only an LLM-based system can provide.

---

## Gap 5: Vulnerability to Deep Learning Steganalysis
**Source:** Kheddar et al. [8], Apau et al. [9], Subramanian et al. [4]
**Problem:** Modern CNN-based steganalysis (SRNet, YedroudjNet) can detect statistical artifacts left by traditional embedding methods. While coverless methods avoid pixel modification, they can still be flagged by analyzing distribution patterns in image selections or hash collisions.
**LG-CISH Fix:** Since no pixels are modified, CNN steganalysis has nothing to detect. The CLIP-based semantic matching produces image selections that follow natural visual distributions (not random hash collisions), making statistical pattern detection ineffective.

---

## Gap 6: No Unified Framework Combining LLMs with Coverless Image Steganography
**Source:** All ten references [1–10]
**Problem:** LLM-based steganography exists only in the text domain (LLsM, OD-Stega, LLM-Stega). Coverless image steganography exists but uses no language model intelligence. No prior work bridges the two — using an LLM's capabilities within a coverless image steganography framework.
**LG-CISH Fix:** First framework to place an LLM inside the coverless image stego pipeline, using it for codebook construction, plausibility filtering, and semantic organization — while keeping the core encoding deterministic and lossless.

---

## Gap 7: No Error Detection or Recovery Mechanism
**Source:** Meng et al. [5], Rahman et al. [10]
**Problem:** If a single image is misidentified during decoding, all subsequent bits are corrupted. Existing coverless methods have no mechanism to detect or recover from this.
**LG-CISH Fix:** CRC/hash appended to the encoded bitstream allows the receiver to verify message integrity after decoding. If verification fails, retransmission is requested. Additionally, the minimum CLIP distance constraint during codebook construction minimizes the chance of misidentification in the first place.

---

## Summary Table

| Gap | Identified By | What's Missing | LG-CISH Solution |
|-----|---------------|----------------|-------------------|
| Low capacity | [1], [5] | ~18 bits/image ceiling | Compression + permutation coding |
| Fragile matching | [2], [6], [7] | pHash breaks on real channels | CLIP embeddings |
| No semantic awareness | [1], [5] | Blind bit-to-image mapping | LLM-organized codebook |
| Suspicious output | [3], [9] | Random-looking image sets | LLM plausibility filtering |
| DL steganalysis threat | [4], [8], [9] | Statistical artifacts detectable | Zero pixel modification + natural distributions |
| No LLM integration | [1]–[10] | LLM used only in text stego | LLM in coverless image stego pipeline |
| No error handling | [5], [10] | One wrong image = total failure | CRC verification + CLIP distance constraints |
