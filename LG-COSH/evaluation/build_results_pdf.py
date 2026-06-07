"""Render evaluation/results/RESULTS.md into a standalone results.pdf in the
Springer 'Result Structure' style (Section 4.1-4.8, tables + figures + narrative).

Converts the generated Markdown (single source of truth) to LaTeX and compiles
with lualatex (native Unicode). Run after generate_all.py:

    ../venv/bin/python evaluation/build_results_pdf.py
"""

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_MD = os.path.join(HERE, "results", "RESULTS.md")
OUT_DIR = os.path.join(HERE, "results")
TEX = os.path.join(OUT_DIR, "results.tex")

# Unicode -> LaTeX (applied after escaping specials, so the math we inject survives).
UNI = {
    "×": r"$\times$", "≈": r"$\approx$", "∞": r"$\infty$", "±": r"$\pm$",
    "≤": r"$\le$", "≥": r"$\ge$", "≠": r"$\ne$", "→": r"$\to$", "✓": r"$\checkmark$",
    "σ": r"$\sigma$", "γ": r"$\gamma$", "—": "---", "–": "--",
    "“": "``", "”": "''", "‘": "`", "’": "'", "≈": r"$\approx$",
}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{fontspec}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{graphicx}
\graphicspath{{../figures/}}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{amssymb}
\usepackage{float}
\usepackage{caption}
\usepackage[hidelinks]{hyperref}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}
\title{\vspace{-2em}LG-CISH: Experimental Results and Performance Evaluation}
\author{}
\date{}
\begin{document}
\maketitle
\vspace{-3em}
"""


def esc(t):
    t = t.replace("\\", r"\textbackslash{}")
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
                 ("$", r"\$"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")]:
        t = t.replace(a, b)
    return t


def inline(t):
    t = esc(t)
    for a, b in UNI.items():
        t = t.replace(a, b)
    t = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\\emph{\1}", t)
    t = re.sub(r"`(.+?)`", r"\\texttt{\1}", t)
    return t


def cells(row):
    parts = [c.strip() for c in row.strip().strip("|").split("|")]
    return parts


def emit_table(block, out):
    header = cells(block[0])
    data = [cells(r) for r in block[2:]]
    n = len(header)
    colspec = "l" + "X" * (n - 1) if n > 1 else "X"
    out.append(r"\begin{table}[H]\centering\small")
    out.append(r"\begin{tabularx}{\textwidth}{%s}" % colspec)
    out.append(r"\toprule")
    def row(rcells):
        # guard a leading '[' so "\\ [ref]" isn't parsed as \\[length]
        out_cells = []
        for c in rcells:
            t = inline(c)
            if t.startswith("["):
                t = "{}" + t
            out_cells.append(t)
        return " & ".join(out_cells) + r" \\"

    out.append(row(header))
    out.append(r"\midrule")
    for r in data:
        if len(r) > n:                       # merge overflow (e.g. a literal '|' in a value)
            r = r[:n - 1] + [", ".join(r[n - 1:])]
        r = (r + [""] * n)[:n]
        out.append(row(r))
    out.append(r"\bottomrule")
    out.append(r"\end{tabularx}")
    out.append(r"\end{table}")


def main():
    lines = open(RESULTS_MD, encoding="utf-8").read().splitlines()
    out = [PREAMBLE]
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        # tables: a block of consecutive | lines
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 2:
                emit_table(block, out)
            continue
        # images
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
        if m:
            base = os.path.basename(m.group(1))
            out.append(r"\begin{figure}[H]\centering")
            out.append(r"\includegraphics[width=0.82\linewidth]{%s}" % base)
            out.append(r"\end{figure}")
            i += 1
            continue
        # headings
        if s.startswith("### "):
            out.append(r"\subsubsection*{%s}" % inline(s[4:]))
        elif s.startswith("## "):
            out.append(r"\subsection*{%s}" % inline(s[3:]))
        elif s.startswith("# "):
            out.append(r"\section*{%s}" % inline(s[2:]))
        else:
            out.append(inline(s))
        i += 1
    out.append(r"\end{document}")
    open(TEX, "w", encoding="utf-8").write("\n".join(out))
    print(f"Wrote {TEX}")

    # compile twice (refs/layout) with lualatex
    for _ in range(2):
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", "-halt-on-error",
                            "results.tex"], cwd=OUT_DIR, capture_output=True, text=True)
    pdf = os.path.join(OUT_DIR, "results.pdf")
    if os.path.exists(pdf):
        print(f"Built {pdf} ({os.path.getsize(pdf)//1024} KB)")
    else:
        print("PDF not produced. Last 30 log lines:")
        print("\n".join(r.stdout.splitlines()[-30:]))
        sys.exit(1)


if __name__ == "__main__":
    main()
