"""Section 4.8 — Statistical Validation.

  * Mean ± standard deviation and 95% confidence intervals for the key metrics
    over N >= 30 independent message trials.
  * Significance test (independent t-test) comparing the proposed CLIP matcher
    against the pHash ablation under JPEG-50 — establishing p < 0.05.
  * One-way ANOVA across message-length buckets (margin homogeneity).
"""

import random
import numpy as np
from scipy import stats

import _common as C
from encoder.encode import encode

N = 50


def ci95(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float(vals.mean()), 0.0, 0.0)
    m = vals.mean()
    sem = stats.sem(vals)
    lo, hi = stats.t.interval(0.95, len(vals) - 1, loc=m, scale=sem)
    return float(m), float(lo), float(hi)


def run():
    C.banner("Section 4.8 — Statistical Validation")
    cb = C.get_codebook()
    rng = random.Random(C.SEED)
    msgs = [C.random_message(rng.randint(50, 300), rng) for _ in range(N)]

    # --- clean-channel metrics over N trials ---
    p, mg, t1 = C.source_lookup(cb["embeddings"], cb["embeddings"])
    recs = [C.evaluate_message_fast(m, cb, p, mg, t1) for m in msgs]
    exact = [100.0 if r["exact"] else 0.0 for r in recs]
    ber = [r["ber"] for r in recs]
    margin = [r["mean_margin"] for r in recs]
    nimg = [r["n_images"] for r in recs]

    rows = []
    for label, vals, fmt in [
        ("Reconstruction Accuracy (%)", exact, "{:.2f}"),
        ("Bit Error Rate", ber, "{:.2e}"),
        ("CLIP margin", margin, "{:.4f}"),
        ("Images per message", nimg, "{:.1f}"),
    ]:
        m, lo, hi = ci95(vals)
        sd = float(np.std(vals))
        rows.append([label, f"{fmt.format(m)} ± {fmt.format(sd)}",
                     f"[{fmt.format(lo)}, {fmt.format(hi)}]"])
    summ_md = C.save_table(
        "table_4_8_summary", rows, ["Metric", "Mean ± Std", "95% CI"],
        f"Table 4.8a — Mean ± std and 95%% confidence intervals over N={N} trials (clean channel).")
    print(summ_md)

    # --- significance: CLIP vs pHash under JPEG-50 ---
    import ablation as A
    bank = msgs
    # CLIP per-message exact (0/1)
    src = C.attacked_source_embeddings(cb, lambda im: C.atk_jpeg(im, 50))
    pc, mc, tc = C.source_lookup(src, cb["embeddings"])
    clip_exact = [1.0 if C.evaluate_message_fast(m, cb, pc, mc, tc)["exact"] else 0.0 for m in bank]

    # pHash per-message exact (0/1)
    cb_hashes = A.phash_codebook(cb)
    from PIL import Image
    phash_src = []
    for pth in cb["paths"]:
        arr = np.asarray(C.atk_jpeg(Image.open(pth).convert("RGB"), 50))
        phash_src.append(A.phash_predict(arr, cb_hashes))
    phash_exact = []
    for m in bank:
        _, meta = encode(m, cb)
        rec = [phash_src[t] for t in meta["indices"]]
        phash_exact.append(1.0 if rec == meta["indices"] else 0.0)

    t_stat, p_val = stats.ttest_ind(clip_exact, phash_exact, equal_var=False)
    sig = "YES (p < 0.05)" if p_val < 0.05 else "no"
    print(f"\n  CLIP JPEG50 accuracy  : {100*np.mean(clip_exact):.2f}%")
    print(f"  pHash JPEG50 accuracy : {100*np.mean(phash_exact):.2f}%")
    print(f"  Welch t-test          : t={t_stat:.3f}, p={p_val:.3e} -> significant: {sig}")

    sig_rows = [
        ["Proposed (CLIP)", f"{100*np.mean(clip_exact):.2f}", f"{100*np.std(clip_exact):.2f}"],
        ["Ablation (pHash)", f"{100*np.mean(phash_exact):.2f}", f"{100*np.std(phash_exact):.2f}"],
        ["t-statistic", f"{t_stat:.3f}", ""],
        ["p-value", f"{p_val:.3e}", sig],
    ]
    sig_md = C.save_table(
        "table_4_8_significance", sig_rows, ["Group", "JPEG50 Accuracy (%)", "Std / Note"],
        "Table 4.8b — Significance of CLIP over pHash under JPEG-50 (Welch t-test).")

    # --- ANOVA across length buckets on margin ---
    bank3 = C.message_bank(C.DEFAULT_BUCKETS, 20)
    groups = []
    for name in bank3:
        g = [C.evaluate_message_fast(m, cb, p, mg, t1)["mean_margin"] for m in bank3[name]]
        groups.append(g)
    f_stat, f_p = stats.f_oneway(*groups)
    print(f"  ANOVA margin across buckets: F={f_stat:.3f}, p={f_p:.3e} "
          f"({'homogeneous' if f_p > 0.05 else 'differs'})")

    return {"summary": summ_md, "significance": sig_md,
            "p_value": float(p_val), "clip_acc": float(100*np.mean(clip_exact)),
            "phash_acc": float(100*np.mean(phash_exact)),
            "anova_F": float(f_stat), "anova_p": float(f_p)}


if __name__ == "__main__":
    run()
