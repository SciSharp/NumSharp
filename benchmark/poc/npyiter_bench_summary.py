# Renders the geomean bar summary over ALL measured NpyIter benchmark pairs
# (rounds 1-3, 2026-06-12, NPYITER_CORE_BENCH_RESULTS.md), in the house style
# of the official benchmark-report per-size geomean summary:
#
#         slower <--------- 1.0 (parity) ---------> faster
# 1K    ##########.. 1.41x   (17 win / 7 lose)
#
# speedup = NumPy_time / NumSharp_time: > 1.0 means NumSharp is FASTER.
# Bar scale: 10 chars per 1.0x -> the parity tick sits mid-field (20 chars).
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
    ("HR512", "reuse", "iterator reuse N=512 (Reset+ForEach)", 54.7e-6, 383.8e-6),
    ("PAR8", "parall", "8-band parallel iterators sin 4M", 2.472, 11.672),
    ("Y1", "fusion", "one-pass 7-array sum vs 6 chained", 7.851, 14.591),
]

SCALE = 10.0   # chars per 1.0x speedup; parity tick = 10 chars (mid-field)
WIDTH = 20     # bar field width (2.0x max before the ▶ overflow marker)
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]


def bar(speedup):
    units = speedup * SCALE
    if units >= WIDTH:
        return "█" * (WIDTH - 1) + "▶"
    full = int(units)
    frac = EIGHTHS[int((units - full) * 8)]
    s = "█" * full + frac
    pad = WIDTH - len(s)
    return s + (" " + "." * (pad - 1) if pad >= 3 else " " * pad)


def geomean(vals):
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def line(label, rows):
    speedups = [np_ / ns for _, ns, np_, *_ in rows]
    g = geomean(speedups)
    win = sum(1 for s in speedups if s > 1.0)
    lose = len(speedups) - win
    tag = "   ◄ PARITY" if 0.95 <= g <= 1.05 else ("   ◄ SLOWER" if g < 0.95 else "")
    print(f"{label:<7}{bar(g)}  {g:4.2f}×   ({win:3d} win / {lose:3d} lose){tag}")


HDR = "        slower ◄───────── 1.0 (parity) ─────────► faster"
print("NpyIter sweep — 89 pairs, rounds 1–3 · speedup = NumPy ÷ NumSharp (>1.0 = NumSharp faster)")
print()
print(HDR)
for tier, label in [("small", "1K"), ("mid", "4M"), ("large", "10M")]:
    line(label, [r for r in ROWS if r[4] == tier])
line("ALL", ROWS)
print()
print(HDR.replace("        ", " " * 8))
fams = [
    ("ctor", "ctor"), ("layout", "layout"), ("select", "select"), ("where", "where="),
    ("dtypes", "dtypes"), ("elemwise", "elemwz"), ("traversal", "traver"),
    ("reduce", "reduce"), ("buffered", "bufcst"), ("smallN", "smallN"),
]
for key, label in fams:
    line(label, [r for r in ROWS if r[3] == key])
print()
print("ARCHITECTURE DIVIDENDS (no NumPy equivalent — vs their best possible)")
for _, label, desc, ns, np_ in DIVIDENDS:
    s = np_ / ns
    print(f"{label:<7}{bar(s)}  {s:4.2f}×   ({desc})")
print()
print("tiers: 1K = ≤4K elems · 4M = 32K–8M elems · 10M = 10M elems")
print("10M tier carried by E1 (np.any scalar-scan routing, finding #7); T1 10M memcpy = exact parity")
