# LG-CISH Pipeline Plan

## What We're Doing
Hiding a secret message inside a sequence of normal-looking images — without touching a single pixel. The images ARE the message. Their order and identity encode the data.

---

## The Setup (One-Time, Before Communication)

Sender and receiver both have:
- Same image database (e.g., 1024 images)
- Same CLIP model
- Same LLM

They build a shared **codebook**: every image gets a unique index number.
1024 images = 10 bits per image (2^10 = 1024).

**What can break:** If sender and receiver have even slightly different databases, everything fails.
**Fix:** Database is versioned and checksummed. Both sides verify hash of the full database before use.

**What can break:** Two images in the database look too similar to CLIP — receiver picks the wrong one.
**Fix:** During codebook construction, enforce minimum CLIP distance between all image pairs. If two images are too close, drop one.

---

## Encoding (Sender Side)

### Step 1: Message → Bits
Convert the message to binary. Plain deterministic conversion.
"HI" → 01001000 01001001

Optional: compress with zlib first (lossless, reduces image count).
Optional: encrypt with AES first (adds security layer).

**Nothing can break here.** It's just math.

### Step 2: Bits → Chunks
Split the bitstream into fixed-size chunks. Chunk size = log2(database size).
1024 images → 10-bit chunks.

01001000 01 | 001001xxxx
(pad the last chunk with zeros, store padding length)

Each chunk is now an integer index (0–1023).

**What can break:** Last chunk might be shorter than 10 bits.
**Fix:** Pad with zeros. Send padding length as metadata (or encode it in the first chunk).

### Step 3: Index → Image
Look up each index in the codebook. Index 289 → photo of a beach. Simple table lookup.

**Nothing can break here.** It's a direct lookup.

### Step 4: Plausibility Check (LLM)
Feed the image sequence to the LLM: "Does this look like a normal photo album?"

If yes → send as-is.
If no → pick alternative images for the same indices (if codebook has multiple candidates per index) or reorder using permutation coding.

**What can break:** LLM says the sequence looks unnatural but there are no alternatives.
**Fix:** Build the codebook with multiple images per index (e.g., 3 candidates each). The LLM picks the most natural combination. All candidates for the same index encode the same bits, so losslessness is preserved.

### Step 5: Transmit
Send the images through any channel — WhatsApp, email, social media, etc.

**What can break:** The platform compresses, resizes, or re-encodes the images.
**Fix:** This is exactly why we use CLIP instead of perceptual hashing. CLIP embeddings survive JPEG compression, resizing, and minor edits. The receiver will still identify the correct image.

---

## Decoding (Receiver Side)

### Step 1: Image → CLIP Embedding
Run each received image through the CLIP image encoder. Get a 512-dim vector.

**Nothing can break here.** CLIP is deterministic.

### Step 2: Nearest Neighbor Lookup
Compare each vector against all codebook embeddings using cosine similarity. The closest match gives the index.

**What can break:** Heavy image corruption (extreme compression, cropping, filters) could push the CLIP embedding far enough that the wrong image matches.
**Fix:** The minimum CLIP distance enforced during codebook construction gives us a safety margin. As long as the image isn't destroyed beyond recognition, the correct match wins. If you want extra safety, use top-k matching and verify with a secondary check (e.g., color histogram).

### Step 3: Index → Bits
Each index converts back to its binary chunk. Index 289 → 0100100001. Deterministic.

### Step 4: Bits → Message
Concatenate all chunks, remove padding, convert back to text.

**What can break:** If even one image was identified wrong in Step 2, the entire message from that point onward is corrupted (wrong bits).
**Fix:** Add error detection. Append a CRC or short hash of the original message as extra bits at the end. If the CRC doesn't match after decoding, the receiver knows something went wrong and can request retransmission.

---

## Where the LLM Actually Matters

| Stage | LLM Role | Can We Skip It? |
|-------|----------|-----------------|
| Codebook construction | Generates semantic labels for images, helps organize them into meaningful clusters | Technically yes, but codebook quality drops |
| Plausibility filtering | Checks if the image sequence looks natural to a human observer | Yes, but the output might look suspicious |
| Error recovery (advanced) | Uses context to guess the correct image if CLIP match is ambiguous | Yes, but you lose a recovery mechanism |

The LLM does NOT touch the actual bit encoding. It shapes the cover, not the data. This is what keeps the system lossless.

---

## Where CLIP Actually Matters

| Without CLIP (using pHash) | With CLIP |
|----------------------------|-----------|
| Hash breaks after JPEG compression >70% | Embedding survives even 30% JPEG quality |
| Resize kills the hash | Embedding barely changes after resize |
| Screenshot = hash destroyed | Screenshot = embedding still close enough |
| Matches on pixel patterns (shallow) | Matches on visual meaning (deep) |

CLIP is the **reliability layer**. It's what makes the system work in real-world channels where images get modified in transit.

---

## The Lossless Chain (Summary)

```
SENDER:   Message → bits → chunks → indices → images
                    (deterministic at every step)

CHANNEL:  images may get compressed/resized (noise)

RECEIVER: images → CLIP → nearest neighbor → indices → chunks → bits → Message
                   (robust identification recovers correct indices)
```

Every arrow is either deterministic (math) or robust (CLIP). No arrow is lossy. That's the guarantee.

---

## Known Limitations (Be Honest About These)

1. **Capacity is database-bound.** 1024 images = 10 bits/image. "HELLO" (40 bits) needs 4 images. A 1KB message needs ~820 images. Large messages need large databases or compression.

2. **Database must be secret.** If an attacker gets the database and the codebook, they can decode everything. The database IS the key.

3. **CLIP model is a dependency.** Both sides need the exact same CLIP model. Model updates break compatibility.

4. **LLM adds latency.** Plausibility filtering takes time. For real-time communication, it might be too slow.

5. **Channel corruption has a threshold.** CLIP is robust but not invincible. If an image is cropped to 10% of its original, even CLIP fails.
