#!/usr/bin/env python3
# =============================================================================
# nditer_cards.py — render two 400×300 PNG summary cards from the canonical
# NDIter benchmark (nditer_results.tsv), for embedding in the README + the
# DocFX "Benchmarks vs NumPy" page.
#
#   cards/ops.png  — OPERATIONS vs NumPy: headline geomean + by-size-tier bars
#                    + by-operation-class bars (the head-to-head comparison)
#   cards/cat.png  — the IL-GENERATION DIVIDENDS: iterator construction vs
#                    np.nditer, expression fusion, kernel reuse, parallel inner
#                    loop — plus the chunk-width trend and the honest pathology
#                    canary (machinery NumPy has no equivalent for)
#
# Ratios are NumPy ÷ NumSharp (>1 = NumSharp faster); each bar also shows the %NumPy =
# (NumSharp ÷ NumPy)×100 = the share of NumPy's time NumSharp uses. The cards show RATIOS only,
# never absolute ms: absolute timings vary by hardware (CI runners drift run to
# run) but the same-runner ratio stays meaningful. EVERYTHING is computed from
# the tsv so the cards auto-update every benchmark run; NA ids (a NumSharp
# AccessViolation section, ignored) are skipped.
#
#   python benchmark/nditer/nditer_cards.py
# =============================================================================
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nditer_sheet import (load_tsv, geomean, CATEGORIES, TIERS, TIER_LABEL,
                           MAIN, CTOR, CW, PATH, DIVIDENDS)

CARDS = os.path.join(HERE, "cards")
GREEN, RED, AMBER, INK, MUTE = "#2e9e4f", "#d6453d", "#c98a2b", "#222222", "#666666"


def color_of(r):
    return AMBER if 0.97 <= r <= 1.03 else (GREEN if r > 1.0 else RED)


def hbars(ax, labels, ratios, xmax=None, fontsize=7.6, label_fmt=None):
    """Horizontal ratio bars with a parity line at x=1 and value labels."""
    y = list(range(len(labels)))
    ax.barh(y, ratios, color=[color_of(r) for r in ratios], height=0.66, zorder=3)
    ax.axvline(1.0, color="#444", lw=0.9, zorder=4)
    xm = xmax or max(2.3, max(ratios) * 1.16)
    ax.set_xlim(0, xm)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fontsize, color=INK)
    ax.set_xticks([])
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    for i, r in enumerate(ratios):
        txt = label_fmt(i, r) if label_fmt else f"{r:.2f}× · {100 / r:.0f}%"
        inside = label_fmt is None and r >= xm * 0.58
        ax.text(r - 0.03 if inside else r + 0.035, i, txt, va="center",
                ha="right" if inside else "left", fontsize=fontsize,
                fontweight="bold", color="white" if inside else INK, zorder=5)
    return xm


def stat(SP, keys):
    """(geomean, peak) over present keys, or (None, None)."""
    vals = [SP[k] for k in keys if k in SP]
    return (geomean(vals), max(vals)) if vals else (None, None)


def card_operations(SP):
    main_vals = [SP[f"{f}@{t}"] for f in MAIN for t in TIERS if f"{f}@{t}" in SP]
    hg = geomean(main_vals)
    hw = sum(1 for x in main_vals if x > 1.0)
    hl = len(main_vals) - hw

    tiers = [t for t in TIERS if any(f"{f}@{t}" in SP for f in MAIN)]
    size_lab = [TIER_LABEL[t] for t in tiers]
    size_rat = [geomean([SP[f"{f}@{t}"] for f in MAIN if f"{f}@{t}" in SP]) for t in tiers]

    cats = []
    for name, fams in CATEGORIES:
        vals = [SP[f"{f}@{t}"] for f in fams for t in TIERS if f"{f}@{t}" in SP]
        if vals:
            cats.append((name, geomean(vals)))
    cats.sort(key=lambda kv: kv[1], reverse=True)   # rank best → worst
    cat_lab = [c[0] for c in cats]
    cat_rat = [c[1] for c in cats]

    xm = max(2.3, max(size_rat + cat_rat) * 1.16)
    fig = plt.figure(figsize=(4, 3), dpi=100)
    fig.text(0.035, 0.935, "NDIter vs NumPy — operations", fontsize=10.5,
             fontweight="bold", color=INK)
    fig.text(0.035, 0.876, f"{hg:.2f}× geomean · {hw} win / {hl} lose · {len(main_vals)} cells",
             fontsize=7.3, color=MUTE)

    axt = fig.add_axes([0.265, 0.585, 0.685, 0.205])
    hbars(axt, size_lab, size_rat, xmax=xm)
    axt.text(0.0, 1.06, "by array-size tier", transform=axt.transAxes,
             fontsize=6.6, color=MUTE, va="bottom")

    axb = fig.add_axes([0.265, 0.135, 0.685, 0.31])
    hbars(axb, cat_lab, cat_rat, xmax=xm)
    axb.text(0.0, 1.04, "by operation class", transform=axb.transAxes,
             fontsize=6.6, color=MUTE, va="bottom")

    fig.text(0.5, 0.03, "ratio = NumPy ÷ NumSharp (>1× = NumSharp faster) · % = share of NumPy's time used",
             fontsize=6.2, color=MUTE, ha="center")
    os.makedirs(CARDS, exist_ok=True)
    fig.savefig(os.path.join(CARDS, "ops.png"), dpi=100)
    plt.close(fig)
    print(f"wrote cards/ops.png  size={[round(r, 2) for r in size_rat]} cat={[round(r, 2) for r in cat_rat]}")


def card_dividends(SP):
    rows = []
    cg, cpk = stat(SP, CTOR)
    if cg:
        rows.append(("build vs np.nditer", cg, cpk))
    for d, lab in [("fuse7", "fusion (np.evaluate)"),
                   ("reuse", "kernel reuse"),
                   ("par8", "parallel inner-loop")]:
        g, pk = stat(SP, [f"{d}@{t}" for t in TIERS])
        if g:
            rows.append((lab, g, pk))
    labels = [r[0] for r in rows]
    ratios = [r[1] for r in rows]
    peaks = [r[2] for r in rows]

    cw = [(c.replace("cw.", ""), SP[c]) for c in CW if c in SP]
    worst = min(((p.replace("path.", ""), SP[p]) for p in PATH if p in SP),
                key=lambda kv: kv[1], default=None)

    fig = plt.figure(figsize=(4, 3), dpi=100)
    fig.text(0.035, 0.935, "NDIter — IL-generation dividends", fontsize=10.5,
             fontweight="bold", color=INK)
    fig.text(0.035, 0.876, "iterator machinery NumPy has no equivalent for",
             fontsize=7.3, color=MUTE)

    # Room to the right of each (short) bar for a "geomean (peak)" label.
    xm = max(ratios) * 1.62
    ax = fig.add_axes([0.36, 0.40, 0.60, 0.40])
    hbars(ax, labels, ratios, xmax=xm, fontsize=7.8,
          label_fmt=lambda i, r: f"{r:.1f}× · {100 / r:.0f}%")

    y0 = 0.255
    if cw:
        fig.text(0.06, y0, "chunk-width dispatch", fontsize=6.8, color=INK, fontweight="bold")
        fig.text(0.46, y0, f"w={cw[0][0]} {cw[0][1]:.2f}×  →  w={cw[-1][0]} {cw[-1][1]:.2f}×",
                 fontsize=6.8, color=MUTE)
    if worst:
        fig.text(0.06, y0 - 0.075, "honest canary", fontsize=6.8, color=INK, fontweight="bold")
        fig.text(0.46, y0 - 0.075, f"{worst[0]}  {1 / worst[1]:.0f}× behind NumPy (tracked)",
                 fontsize=6.8, color=RED)

    fig.text(0.5, 0.03, "ratio = NumPy ÷ NumSharp (>1× = NumSharp faster) · % = share of NumPy's time used",
             fontsize=6.2, color=MUTE, ha="center")
    os.makedirs(CARDS, exist_ok=True)
    fig.savefig(os.path.join(CARDS, "cat.png"), dpi=100)
    plt.close(fig)
    print(f"wrote cards/cat.png  rows={list(zip(labels, [round(r, 2) for r in ratios]))}")


def main():
    pairs = load_tsv()
    SP = {k: np_ / ns for k, (ns, np_) in pairs.items() if ns is not None}   # skip NA (AV)
    card_operations(SP)
    card_dividends(SP)


if __name__ == "__main__":
    main()
