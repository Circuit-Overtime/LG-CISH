**1. Structure of Result Section (Springer Standard)**

Use this exact hierarchy:

**4. Experimental Results and Performance Evaluation**
    - 4.1 Experimental Setup
    - 4.2 Qualitative Results
    - 4.3 Quantitative Evaluation
    - 4.4 Robustness Analysis
    - 4.5 Security and Steganalysis Resistance
    - 4.6 Ablation Study
    - 4.7 Comparative Analysis
    - 4.8 Statistical Validation
**2. Experimental Setup (Must Be Precise)**

Include: Figure for all image including secret

**Dataset**

1. USC-SIPI / BOSSBase / ImageNet subset/Div2K(Any One)
2. Number of images (e.g., 20 images)

**Parameters**

1. Hash size (e.g., 64-bit perceptual hash)
2. LLM used (e.g., GPT-based / open-source LLM)
3. Token length

**Evaluation Metrics (very important)**

**Since your method is coverless, DO NOT rely only on PSNR/SSIM.**

Use:

- **Payload Capacity (bits/image or bits/sequence)**
- **Semantic Reconstruction Accuracy (%)**
- **Hash Matching Accuracy**
- **Bit Error Rate (BER)**
- **Retrieval Precision/Recall**
- **Time Complexity
3. Qualitative Results (Visual Proof)**

**Include figures:**

**Figure Set 1:**

- Original message → Generated image sequence


**Figure Set 2:**

- Image sequence → Reconstructed message

**Figure Set 3:**

- Failure cases (very important for reviewers)

**Caption example:**

Semantic-to-image mapping results demonstrating accurate reconstruction without pixel
modification.

**4. Quantitative Results (Core Acceptance Factor)**

**Create clear tables like** :

**Table : Reconstruction Accuracy**

```
Message Length Accuracy (%) BER
Short (50 words)
Medium
Long
```
**Table : Payload Capacity Comparison**

```
Method Capacity Distortion
LSB
DCT
Proposed LG-CISH
```
**Table : Computational Time**

```
Stage Time (ms)
LLM Encoding
Hash Matching
```
**5. Robustness Analysis (Highly Critical)**

Test against:

- Image compression (JPEG 50%, 70%)
- Noise (Gaussian, Salt & Pepper)
- Cropping / resizing

**Table: Robustness Results**

```
Attack Accuracy (%)
JPEG 50%
```

```
Noise
Etc.
```
# This is crucial because your method relies on hash stability.

**6. Security & Steganalysis Resistance**

**Since your method is coverless, emphasize** :

**Include:**

1. CNN-based steganalysis (SRNet, XuNet)
2. Show detection accuracy ≈ random (≈50%)

**Table :**

```
Method Detection Accuracy
LSB
DCT
Proposed
```
# This is for a strong acceptance booster

**7. Ablation Study (Very Important for Novelty)**

Remove components and show effect:

```
Variant Accuracy
Without LLM
Without hashing
Full model
```
# Shows each component is necessary.

**8. Comparative Analysis (Mandatory)**

Compare with:

1. LSB
2. DWT-DCT
3. Coverless steganography papers
4. Deep learning-based methods

**Use Graphs:**

1. Accuracy vs Payload
2. Robustness vs Attack strength
**9. Statistical Validation (Often we Miss → Causes of Rejection)**


Include:

- Mean ± Standard Deviation
- t-test or ANOVA

Example:

The proposed method shows statistically significant improvement (p < 0.05) over existing
approaches.

# 10. Key Acceptance Boosters (DO NOT MISS)

1. Include at least 6–8 tables
2. Include clear graphs (not cluttered)
3. Provide failure analysis
4. Use recent baselines (202 3 – 2025)
5. Show real-world applicability

# Final Critical Insight

Your algorithm is non-traditional (coverless + LLM-based), so reviewers will question:

1. Reliability
2. Reproducibility
3. Practicality

**Your results must** **_prove_** **:**

1. It works consistently
2. It is robust
3. It is secure


