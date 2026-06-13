# Renders the geomean-ratio bar summary over ALL measured NpyIter benchmark
# pairs (rounds 1-3, 2026-06-12, NPYITER_CORE_BENCH_RESULTS.md). Ratio is
# NumSharp_time / NumPy_time: < 1.0 means NumSharp is faster.
# Bar scale: 10.2 chars per 1.0x (parity = ~10 chars), dotted to 20 chars.
import math

# (id, ns_ms, np_ms, family, tier)
ROWS = [
    # --- Construction (vs np.nditer), small tier ---
    ("C1", 289e-6, 275e-6, "ctor", "small"), ("C2", 298e-6, 541e-6, "ctor", "small"),
    ("C3", 308e-6, 622e-6, "ctor", "small"), ("C4", 304e-6, 701e-6, "ctor", "small"),
    ("C5", 696e-6, 719e-6, "ctor", "small"), ("C6", 559e-6, 1420e-6, "ctor", "small"),
    ("C7", 309e-6, 1140e-6, "ctor", "small"), ("C8", 315e-6, 421e-6, "ctor", "small"),
    ("C9", 317e-6, 426e-6, "ctor", "small"), ("C10", 343e-6, 1000e-6, "ctor", "small"),
    ("C11", 385e-6, 1140e-6, "ctor", "small"), ("C12", 379e-6, 662e-6, "ctor", "small"),
    ("C13", 327e-6, 621e-6, "ctor", "small"), ("C14", 321e-6, 953e-6, "ctor", "small"),
    # --- Small-N pipeline / production glue ---
    ("H8", 307.9e-6, 286.4e-6, "smallN", "small"), ("H64", 317e-6, 299.6e-6, "smallN", "small"),
    ("H512", 354.9e-6, 383.8e-6, "smallN", "small"), ("H0", 648.4e-6, 429.3e-6, "smallN", "small"),
    ("H4096", 946.5e-6, 981.1e-6, "smallN", "small"),
    ("M1", 888.4e-6, 930.7e-6, "smallN", "small"), ("O1", 468.8e-6, 285.6e-6, "smallN", "small"),
    ("O2", 810.7e-6, 336.8e-6, "smallN", "small"), ("O3", 901.4e-6, 520.2e-6, "smallN", "small"),
    ("O4", 88.5e-6, 249.6e-6, "smallN", "small"),
    # --- Chunk traversal / copies (iterator orchestration) ---
    ("H32768", 5.64e-3, 5.28e-3, "traversal", "mid"), ("H262144", 85.49e-3, 83.60e-3, "traversal", "mid"),
    ("H2M", 1.397, 1.399, "traversal", "mid"),
    ("T2.4", 3.929, 2.234, "traversal", "mid"), ("T2c", 2.335, 2.234, "traversal", "mid"),
    ("T2.16", 2.010, 1.637, "traversal", "mid"), ("T2.64", 1.521, 1.365, "traversal", "mid"),
    ("T2.256", 1.297, 1.285, "traversal", "mid"), ("T2.1024", 0.858, 1.039, "traversal", "mid"),
    ("P4", 2.147, 2.218, "traversal", "mid"),
    ("S.4", 5.285, 5.417, "traversal", "mid"), ("S.16", 3.965, 3.823, "traversal", "mid"),
    ("S.64", 2.898, 3.060, "traversal", "mid"), ("S.1024", 1.449, 2.837, "traversal", "mid"),
    ("Sp.4", 4.625, 5.417, "traversal", "mid"), ("Sp.16", 3.038, 3.823, "traversal", "mid"),
    ("Sp.64", 3.048, 3.060, "traversal", "mid"), ("Sp.1024", 1.537, 2.837, "traversal", "mid"),
    ("T1", 4.372, 4.377, "traversal", "large"),
    # --- Layout copies (CopyAsFlat consumers) ---
    ("T3", 1.592, 1.895, "layout", "mid"), ("X2", 2.745, 2.915, "layout", "mid"),
    ("RV1", 26.324, 49.207, "layout", "mid"), ("RV2", 22.198, 48.889, "layout", "mid"),
    ("RV3", 1.760, 5.301, "layout", "mid"), ("RV4", 2.286, 4.032, "layout", "mid"),
    # --- Elementwise / broadcast / overlap / allocate ---
    ("T4r", 0.715, 0.960, "elemwise", "mid"), ("T4c", 0.699, 0.868, "elemwise", "mid"),
    ("X1", 3.014, 4.499, "elemwise", "mid"), ("X1p", 2.942, 4.499, "elemwise", "mid"),
    ("V1", 2.185, 2.480, "elemwise", "mid"), ("V2", 4.719, 8.257, "elemwise", "mid"),
    ("A3", 3.526, 3.737, "elemwise", "mid"), ("A1", 5.852, 7.515, "elemwise", "mid"),
    ("A2", 3.832, 7.515, "elemwise", "mid"),
    ("UF1", 4.054, 7.202, "elemwise", "mid"), ("UF2", 3.313, 3.636, "elemwise", "mid"),
    ("UF3", 4.845, 7.393, "elemwise", "mid"), ("D1", 2.988, 2.117, "elemwise", "mid"),
    ("PAR0", 12.099, 11.672, "elemwise", "mid"),
    # --- Buffered cast / mixed dtype ---
    ("T5", 2.502, 2.259, "buffered", "mid"), ("B1", 2.334, 1.531, "buffered", "mid"),
    ("B1p", 1.648, 1.531, "buffered", "mid"), ("T6", 3.695, 4.287, "buffered", "mid"),
    # --- where= / masks ---
    ("W1", 2.801, 3.536, "where", "mid"), ("W2", 10.646, 14.853, "where", "mid"),
    ("W3", 4.097, 3.186, "where", "mid"),
    ("WH1", 4.227, 7.751, "where", "mid"), ("WH2", 3.291, 6.761, "where", "mid"),
    ("WH3", 3.350, 7.190, "where", "mid"),
    # --- Reductions / scans ---
    ("T8", 3.152, 4.924, "reduce", "large"), ("T8s", 0.240, 0.282, "reduce", "mid"),
    ("R0a", 1.124, 1.029, "reduce", "mid"), ("R0b", 0.803, 2.220, "reduce", "mid"),
    ("R1", 1.279, 1.065, "reduce", "mid"), ("R2", 1.157, 2.347, "reduce", "mid"),
    ("RD1", 0.806, 2.285, "reduce", "mid"), ("RD2", 1.020, 1.129, "reduce", "mid"),
    ("RD3", 3.225, 1.633, "reduce", "mid"), ("RD4", 1.117, 1.681, "reduce", "mid"),
    ("RD5", 1.714, 1.115, "reduce", "mid"),
    ("F1", 61.892, 1.140, "reduce", "mid"),
    ("E1", 1.856, 0.128, "reduce", "large"), ("E2", 0.00035, 0.00135, "reduce", "large"),
    ("AC1", 5.805, 10.167, "reduce", "mid"), ("AC2", 95.039, 69.90, "reduce", "mid"),
    ("AC3", 3.496, 10.213, "reduce", "mid"),
    # --- Selection / indexing (mapping.c consumers) ---
    ("BM1", 4.016, 10.465, "select", "mid"), ("BM2", 9.404, 17.813, "select", "mid"),
    ("BM3", 1.655, 2.258, "select", "mid"), ("BM4", 1.201, 5.923, "select", "mid"),
    ("FX1", 9.496, 12.523, "select", "mid"), ("FX2", 10.113, 6.790, "select", "mid"),
    ("MI1", 5.293, 5.417, "select", "mid"), ("MI2", 2.122, 3.004, "select", "mid"),
    # --- Kernel-bound dtype frontier (context, not iterator) ---
    ("Z1", 7.352, 6.672, "dtypes", "mid"), ("Z2", 7.552, 6.542, "dtypes", "mid"),
    ("Z3", 20.747, 15.447, "dtypes", "mid"), ("Z4", 0.173, 1.201, "dtypes", "mid"),
]

# Architecture dividends — no like-for-like NumPy machinery (their best possible shown)
DIVIDENDS = [
    ("HR512", "reused iterator N=512 (Reset+ForEach)", 54.7e-6, 383.8e-6),
    ("PAR8", "8-banded parallel iterators, sin f64 4M", 2.472, 11.672),
    ("Y1", "ONE-PASS sum of 7 arrays vs 6 chained adds", 7.851, 14.591),
]

SCALE = 10.2   # chars per 1.0x; parity ~= 10 chars
WIDTH = 22     # dotted field width
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]


def bar(ratio):
    units = ratio * SCALE
    full = int(units)
    frac = EIGHTHS[int((units - full) * 8)]
    s = "█" * full + frac
    pad = WIDTH - len(s)
    return s + (" " + "." * (pad - 1) if pad > 1 else "")


def geomean(ratios):
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


def line(label, rows, note=""):
    ratios = [ns / np_ for _, ns, np_, *_ in rows]
    g = geomean(ratios)
    win = sum(1 for r in ratios if r < 1.0)
    lose = len(ratios) - win
    tag = "  ◄ FASTER" if g < 0.93 else ("  ◄ PARITY" if g <= 1.07 else "  ◄ SLOWER")
    print(f"{label:<22}{bar(g)} {g:5.2f}×   ({win:3d} win / {lose:3d} lose){tag}{note}")


print("NpyIter benchmark sweep — 89 measured pairs, rounds 1–3 (i9-13900K, NumPy 2.4.2, Release)")
print("ratio = NumSharp time / NumPy time · geomean per group")
print()
print("        faster ◄───────── 1.0 (parity) ─────────► slower")
print()
print("BY SIZE TIER")
for tier, label in [("small", "≤4K elems"), ("mid", "32K–8M elems"), ("large", "10M elems")]:
    line(f"{label:<10}", [r for r in ROWS if r[4] == tier])
print()
print("BY FAMILY")
fams = [
    ("ctor", "construction"),
    ("smallN", "small-N pipeline"),
    ("traversal", "chunk traversal"),
    ("layout", "layout copies"),
    ("elemwise", "elementwise/bcast"),
    ("buffered", "buffered cast"),
    ("where", "where= / masks"),
    ("reduce", "reductions/scans"),
    ("select", "selection/indexing"),
    ("dtypes", "kernel-bound dtypes"),
]
for key, label in fams:
    line(f"{label:<10}", [r for r in ROWS if r[3] == key])
print()
line("ALL (like-for-like)", ROWS)
ex_outliers = [r for r in ROWS if r[0] not in ("F1", "E1", "AC2")]
line("ALL excl. 3 outliers", ex_outliers, note="  (drop F1 54×, E1 14.5×, AC2 1.4×)")
print()
print("ARCHITECTURE DIVIDENDS (vs NumPy's best possible — no equivalent machinery)")
for id_, label, ns, np_ in DIVIDENDS:
    r = ns / np_
    print(f"{label:<44}{bar(r)} {r:5.2f}×   ({1/r:4.1f}× faster)")
print()
worst = sorted(ROWS, key=lambda r: r[1] / r[2], reverse=True)[:5]
best = sorted(ROWS, key=lambda r: r[1] / r[2])[:5]
print("WORST 5: " + " · ".join(f"{i} {ns/np_:.2f}×" for i, ns, np_, *_ in worst))
print("BEST 5:  " + " · ".join(f"{i} {ns/np_:.2f}×" for i, ns, np_, *_ in best))
