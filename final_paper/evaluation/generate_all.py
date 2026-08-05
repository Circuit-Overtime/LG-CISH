"""Master evaluation runner — produces the complete Section 4 for the paper.

Runs every section in order, collects all tables (markdown) and figures, and
writes:
  * evaluation/results/RESULTS.md   — the assembled Section 4 (tables + figures + narrative)
  * evaluation/tables/*.tex         — LaTeX versions of every table (for Springer)
  * evaluation/figures/*.png        — all figures

Run from the LG-COSH/ directory with the venv active:
    ../venv/bin/python evaluation/generate_all.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _common as C
import setup as s_setup
import qualitative as s_qual
import quantitative as s_quant
import robustness as s_rob
import security as s_sec
import ablation as s_abl
import comparison as s_cmp
import stats_section as s_stat


def rel(p):
    """Path relative to the results/ dir (RESULTS.md lives there)."""
    return os.path.relpath(p, C.OUT_DIR)


def main():
    t0 = time.perf_counter()
    cb = C.get_codebook()

    r_setup = s_setup.run()
    r_qual = s_qual.run()
    r_quant = s_quant.run()
    r_rob = s_rob.run()
    r_sec = s_sec.run()
    r_abl = s_abl.run()
    det = {"LSB": r_sec["detection"].get("LSB", 0),
           "DCT-LSB": r_sec["detection"].get("DCT-LSB", 0),
           "Proposed": r_sec["detection"].get("Proposed LG-CISH", 0)}
    r_cmp = s_cmp.run(detection=det)
    r_stat = s_stat.run()

    sep = r_setup["separation"]
    L = []
    w = L.append
    w("# 4. Experimental Results and Performance Evaluation\n")
    w("All experiments use the LG-CISH codebook of "
      f"{cb['n_images']} visually-distinct images drawn from standard benchmark "
      "suites (UCID, Kodak, USC-SIPI), each normalized to a fixed 512×512 canvas "
      f"({cb['bits_per_image']:.3f} bits/image, base-{cb['n_images']} positional "
      "coding). The images are never modified; the identity and order of the "
      "transmitted sequence carry the secret message. Two coding modes are "
      "available — base-N positional coding (maximum capacity) and distinct-image "
      "permutation coding (no repeated images, more plausible cover) — together "
      "with an optional LLM-guided alias layer that swaps in interchangeable cover "
      "images without changing the encoded bits.\n")

    # 4.1
    w("\n## 4.1 Experimental Setup\n")
    w(r_setup["table_md"])
    w(f"\nThe {cb['n_images']} codebook images are mutually well-separated in CLIP "
      f"space (max pairwise similarity {sep['max']:.3f}, decoding margin {sep['margin']:.3f}), "
      "which is what makes nearest-neighbour index recovery robust.\n")
    w(f"\n![Setup]({rel(r_setup['figure'])})\n")

    # 4.2
    w("\n## 4.2 Qualitative Results\n")
    for f in r_qual["figures"]:
        w(f"\n![Qualitative]({rel(f)})\n")
    w("\n*Semantic-to-image mapping results demonstrating accurate reconstruction "
      "without pixel modification.* Failure cases (Figure Set 3) occur only under "
      f"extreme degradation; a codebook mismatch is **{r_qual['failures']['mismatch']}** "
      "by the CRC-32 layer.\n")

    # 4.3
    w("\n## 4.3 Quantitative Evaluation\n")
    w(r_quant["reconstruction"]); w("\n")
    w(r_quant["capacity"]); w("\n")
    w(r_quant["coding_modes"]); w("\n")
    w(r_quant["timing"]); w("\n")
    w(f"\nCLIP top-1 retrieval precision/recall on the codebook: "
      f"**{100*r_quant['precision']:.2f}%**. Reconstruction is bit-exact (BER ≈ 0) on a "
      "clean channel across all message lengths.\n")
    ast = r_quant.get("alias_stats")
    if ast and ast["total_candidates"]:
        w(f"\nThe LLM-guided alias layer adds **{ast['total_candidates']}** CLIP-verified "
          f"candidate images across **{ast['slots_with_alias']}/{cb['n_images']}** slots "
          "(captioned with gemini-fast, generated with flux, and verified to map back to "
          "the correct slot). These give the plausibility selector interchangeable cover "
          "choices **without changing the encoded bits** — decoding is provably unaffected.\n")

    # 4.4
    w("\n## 4.4 Robustness Analysis\n")
    w(r_rob["table"])
    base = r_rob["detail"]["No attack (baseline)"]["mean_margin"]
    j50 = r_rob["detail"]["JPEG 50%"]
    w(f"\nThe CLIP margin starts at {base:.3f} and remains positive through most "
      f"attacks; JPEG-50 retains {100*j50['exact']:.1f}% reconstruction "
      f"(BER {j50['ber']:.2e}). Decoding degrades gracefully only under extreme "
      "geometric distortion.\n")

    # 4.5
    w("\n## 4.5 Security & Steganalysis Resistance\n")
    w(r_sec["steganalysis"])
    w(f"\nKeyspace ≈ 2^{r_sec['keyspace_bits']:.0f} (codebook orderings × AES-256). "
      f"CRC-32 catches **{r_sec['crc_detection']:.2f}%** of bit-flip tampering. Because "
      "the transmitted images are unmodified natural images, the chi-square detector "
      "operates at chance (~50%) for the proposed method, versus near-certain detection "
      "for LSB.\n")

    # 4.5b Cover plausibility (cached from evaluation/plausibility_study.py, if present)
    plaus_json = os.path.join(C.OUT_DIR, "plausibility.json")
    plaus_tbl = os.path.join(C.TBL_DIR, "table_4_5_plausibility.md")
    if os.path.exists(plaus_json) and os.path.exists(plaus_tbl):
        import json as _json
        pj = _json.load(open(plaus_json))
        w("\n### 4.5.1 Cover Plausibility\n")
        w(open(plaus_tbl, encoding="utf-8").read())
        w(f"\nBeyond statistical undetectability, behavioural stealth depends on whether the "
          f"image *set* looks natural. An LLM judge (gemini-fast) rates the diverse "
          f"benchmark codebook at **{pj['diverse_benchmark']['mean']:.2f}** but a themed "
          f"codebook at **{pj['themed']['mean']:.2f}** — plausibility is governed by codebook "
          f"*theme*, not the codec (permutation coding scores "
          f"{pj['diverse_permutation']['mean']:.2f}, essentially unchanged). We deliberately "
          "use the diverse standard-benchmark set for all quantitative results (dataset "
          "credibility) and treat codebook theme as an explicit deployment trade-off: a "
          "themed database is the more plausible real-world cover.\n")

    # 4.6
    w("\n## 4.6 Ablation Study\n")
    w(r_abl["table"])
    w(f"\nBoth CLIP and pHash decode JPEG-50 perfectly on the 40-image "
      f"codebook, but under the harsher Crop-65% attack the semantic CLIP matcher "
      f"holds at {r_abl['clip_crop40']:.0f}% while pHash collapses to "
      f"{r_abl['phash_crop40']:.0f}% — the geometric robustness that motivates CLIP. "
      f"The coding ablations show clear effects: disabling compression inflates the "
      f"sequence ({r_abl['avg_full']:.1f} → {r_abl['avg_nocomp']:.1f} images), "
      f"fixed-chunk coding needs {r_abl['avg_fixed']:.1f} images vs "
      f"{r_abl['avg_full']:.1f} for base-N, and removing CRC-32 leaves channel "
      f"errors undetected.\n")

    # 4.7
    w("\n## 4.7 Comparative Analysis\n")
    w(r_cmp["table"])
    pb = r_cmp.get("psnr_bpp")
    if pb:
        w(f"\nOn the distortion–capacity axis, the pixel baselines trade quality for "
          f"payload: LSB falls from {pb['lsb_psnr'][0]:.0f} dB at "
          f"{pb['lsb_bpp'][0]:.2f} bpp to {pb['lsb_psnr'][-1]:.0f} dB at "
          f"{pb['lsb_bpp'][-1]:.2f} bpp, while DCT-LSB sits at "
          f"{pb['dct_psnr'][-1]:.0f}–{pb['dct_psnr'][0]:.0f} dB and caps out near "
          f"{pb['dct_bpp'][-1]:.3f} bpp. Because LG-CISH modifies no pixels its PSNR "
          "is infinite at every embedding rate (the green ceiling), so it dominates "
          "the entire distortion–capacity plane rather than choosing a point on it.\n")
    for f in r_cmp["figures"]:
        w(f"\n![Comparison]({rel(f)})\n")

    # 4.8
    w("\n## 4.8 Statistical Validation\n")
    w(r_stat["summary"]); w("\n")
    w(r_stat["significance"])
    sig_txt = ("statistically significant (p < 0.05)" if r_stat["p_value"] < 0.05
               else "not statistically significant at this sample size")
    w(f"\nThe proposed method is lossless, so its clean- and mild-channel accuracy is a "
      f"deterministic 100% (zero variance); we therefore report the proposed-vs-LSB "
      f"comparison descriptively — bit-exact at JPEG-50 ({r_stat['prop_acc']:.0f}%) versus "
      f"LSB's {r_stat['lsb_acc']:.1f}% bit accuracy — and run the significance test on the "
      f"regime that actually discriminates the matchers: geometric robustness. Across a "
      f"sweep of {r_stat['n_attacks']} crop strengths (non-geometric channels leave both at "
      f"100%, so they tie), CLIP decodes {r_stat['clip_attack_mean']:.0f}% versus pHash's "
      f"{r_stat['phash_attack_mean']:.0f}%, a difference that is {sig_txt} "
      f"(paired Wilcoxon signed-rank, p = {r_stat['p_value']:.2e}); e.g. at "
      f"{r_stat['harsh_attack']}, CLIP {r_stat['clip_harsh']:.0f}% vs pHash "
      f"{r_stat['phash_harsh']:.0f}%. One-way ANOVA across "
      f"message-length buckets shows the CLIP margin is homogeneous "
      f"(F = {r_stat['anova_F']:.2f}, p = {r_stat['anova_p']:.2f}).\n")

    out = os.path.join(C.OUT_DIR, "RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    # concatenate all LaTeX tables
    tex_files = sorted(p for p in os.listdir(C.TBL_DIR) if p.endswith(".tex"))
    with open(os.path.join(C.TBL_DIR, "all_tables.tex"), "w", encoding="utf-8") as f:
        for tf in tex_files:
            f.write(open(os.path.join(C.TBL_DIR, tf), encoding="utf-8").read() + "\n\n")

    dt = time.perf_counter() - t0
    C.banner("DONE")
    print(f"  Section 4 written  -> {out}")
    print(f"  {len(tex_files)} LaTeX tables -> {C.TBL_DIR}/all_tables.tex")
    print(f"  Figures            -> {C.FIG_DIR}/")
    print(f"  Total time: {dt:.1f}s")


if __name__ == "__main__":
    main()
