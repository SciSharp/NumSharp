#!/usr/bin/env python3
# =============================================================================
# npyiter_cards.py — render two 400×300 PNG summary cards from the canonical
# NpyIter benchmark (npyiter_results.tsv), for embedding in the README.
#
#   cards/ops.png  — speedup by array-size tier (scalar/1K/100K/1M)
#   cards/cat.png  — speedup by operation class
#
# Ratios are NumPy ÷ NumSharp (>1 = NumSharp faster). The cards intentionally
# show RATIOS only, never absolute ms: absolute timings vary by hardware (CI
# runners drift run-to-run) but the same-runner ratio stays meaningful.
#
#   python benchmark/npyiter/npyiter_cards.py
# =============================================================================
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from npyiter_sheet import load_tsv, geomean, CATEGORIES, TIERS, TIER_LABEL, MAIN

CARDS = os.path.join(HERE, "cards")
GREEN, RED, INK, MUTE = "#2e9e4f", "#d6453d", "#222222", "#666666"


def bar_card(path, title, subtitle, labels, ratios, footer):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    fig.subplots_adjust(left=0.30, right=0.95, top=0.80, bottom=0.16)
    y = list(range(len(labels)))
    ax.barh(y, ratios, color=[GREEN if r >= 1.0 else RED for r in ratios], height=0.64, zorder=3)
    ax.axvline(1.0, color="#444", lw=1.0, zorder=4)
    xmax = max(2.3, max(ratios) * 1.15)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.invert_yaxis()
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5, color=INK)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0", "1× (parity)", "2×"], fontsize=6.5, color=MUTE)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbb")
    for i, r in enumerate(ratios):
        inside = r >= xmax * 0.62
        ax.text(r - 0.03 if inside else r + 0.04, i, f"{r:.2f}×", va="center",
                ha="right" if inside else "left", fontsize=8.5, fontweight="bold",
                color="white" if inside else INK, zorder=5)
    fig.text(0.07, 0.925, title, fontsize=11, fontweight="bold", color=INK)
    fig.text(0.07, 0.86, subtitle, fontsize=7.3, color=MUTE)
    fig.text(0.5, 0.035, footer, fontsize=6.3, color=MUTE, ha="center")
    os.makedirs(CARDS, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path)} ({ratios})")


def main():
    pairs = load_tsv()
    SP = {k: np_ / ns for k, (ns, np_) in pairs.items()}

    def tier(t):
        return [SP[f"{f}@{t}"] for f in MAIN if f"{f}@{t}" in SP]

    main_sps = [SP[f"{f}@{t}"] for f in MAIN for t in TIERS if f"{f}@{t}" in SP]
    g = geomean(main_sps)
    win = sum(1 for x in main_sps if x > 1.0)
    head = f"{g:.2f}× geomean · {win} win / {len(main_sps) - win} lose · {len(pairs)} pairs"

    # Card A — by size tier
    tlabels = [TIER_LABEL[t] for t in TIERS]
    tratios = [geomean(tier(t)) for t in TIERS]
    bar_card(os.path.join(CARDS, "ops.png"),
             "NpyIter vs NumPy — by size", head, tlabels, tratios,
             "ratio = NumPy ÷ NumSharp on one runner · >1× = NumSharp faster")

    # Card B — by operation class (sorted fastest→slowest)
    cats = [(name, geomean([SP[f"{f}@{t}"] for f in fams for t in TIERS if f"{f}@{t}" in SP]))
            for name, fams in CATEGORIES]
    cats.sort(key=lambda kv: kv[1], reverse=True)
    bar_card(os.path.join(CARDS, "cat.png"),
             "NpyIter vs NumPy — by op class", "reductions & elementwise lead · copy/cast lags small-N",
             [c[0] for c in cats], [c[1] for c in cats],
             "ratio = NumPy ÷ NumSharp on one runner · >1× = NumSharp faster")


if __name__ == "__main__":
    main()
