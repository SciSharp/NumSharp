# Renders the size-tier bar summary for npyiter_sizesweep_bench.{cs,py}.
# Self-contained: the recorded clean run (i9-13900K, NumPy 2.4.2, Release) is
# embedded below as (NumSharp_ms, NumPy_ms) per id. To re-chart a fresh run,
# pass the two recorded output files:
#       python npyiter_sizesweep_chart.py ns_run.txt np_run.txt
#
# speedup = NumPy_time / NumSharp_time: > 1.0 means NumSharp is FASTER.
# Axis is the official-report style (slower <-- 1.0 (parity) --> faster);
# bar grows toward "faster", parity tick sits mid-field at 20-char width.
import math
import re
import sys

# id -> (NumSharp ms, NumPy ms), from npyiter_sizesweep_{ns,np}.txt (settled runs)
DATA = {
    "add@1":    (171.5e-6, 291.5e-6), "sqrt@1":   (154.7e-6, 261.9e-6),
    "sum@1":    (151.3e-6, 1.53e-3),  "copy@1":   (151.7e-6, 262.2e-6),
    "sadd@1":   (171.5e-6, 289.7e-6), "bcast@1":  (170.9e-6, 291.0e-6),
    "add@1K":   (261.5e-6, 415.3e-6), "sqrt@1K":  (739.5e-6, 881.1e-6),
    "sum@1K":   (192.5e-6, 1.73e-3),  "copy@1K":  (211.9e-6, 541.6e-6),
    "sadd@1K":  (496.6e-6, 712.0e-6), "bcast@1K": (615.7e-6, 807.9e-6),
    "add@100K": (24.83e-3, 26.66e-3), "sqrt@100K":(57.57e-3, 56.78e-3),
    "sum@100K": (6.63e-3,  17.64e-3), "copy@100K":(11.92e-3, 20.86e-3),
    "sadd@100K":(50.50e-3, 49.96e-3), "bcast@100K":(15.40e-3, 17.38e-3),
    "add@1M":   (453.40e-3, 397.65e-3), "sqrt@1M": (578.07e-3, 569.02e-3),
    "sum@1M":   (89.14e-3, 209.39e-3),  "copy@1M": (216.14e-3, 289.97e-3),
    "sadd@1M":  (1.336,    1.385),      "bcast@1M":(246.15e-3, 247.24e-3),
}

ASPECTS = ["add", "sqrt", "sum", "copy", "sadd", "bcast"]
TIERS = ["1", "1K", "100K", "1M"]
ASPECT_LABEL = {"add": "add", "sqrt": "sqrt", "sum": "sum", "copy": "copy",
                "sadd": "strided", "bcast": "bcast"}
TIER_LABEL = {"1": "scalar", "1K": "1K", "100K": "100K", "1M": "1M"}

SCALE = 10.0   # chars per 1.0x; parity tick = 10 chars (mid-field)
WIDTH = 20     # bar field width
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
UNIT = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}
LINE = re.compile(r"^(\w+@\w+)\s+.+?\s+([\d.]+)\s+(ns|us|ms)\s*$")


def parse(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            m = LINE.match(ln.rstrip())
            if m:
                out[m.group(1)] = float(m.group(2)) * UNIT[m.group(3)]
    return out


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


def line(label, speedups):
    g = geomean(speedups)
    win = sum(1 for s in speedups if s > 1.0)
    lose = len(speedups) - win
    tag = "   ◄ PARITY" if 0.97 <= g <= 1.03 else ("   ◄ SLOWER" if g < 0.97 else "")
    print(f"{label:<8}{bar(g)}  {g:5.2f}×   ({win} win / {lose} lose){tag}")


if len(sys.argv) > 2:
    ns, npp = parse(sys.argv[1]), parse(sys.argv[2])
    sp = {k: npp[k] / ns[k] for k in ns if k in npp}
else:
    sp = {k: np_ / ns_ for k, (ns_, np_) in DATA.items()}

HDR = "        slower ◄───────── 1.0 (parity) ─────────► faster"
print("NpyIter size sweep — 6 op families × 4 size tiers, f64 (i9-13900K, NumPy 2.4.2, Release)")
print("speedup = NumPy ÷ NumSharp per call · >1.0× = NumSharp faster · matched trivial kernels")
print()
print("BY SIZE TIER  (geomean over add/sqrt/sum/copy/strided/bcast)")
print(HDR)
for t in TIERS:
    line(TIER_LABEL[t], [sp[f"{a}@{t}"] for a in ASPECTS])
line("ALL", [sp[f"{a}@{t}"] for a in ASPECTS for t in TIERS])
print()
print("BY OPERATION  (geomean over scalar/1K/100K/1M)")
print(HDR)
for a in ASPECTS:
    line(ASPECT_LABEL[a], [sp[f"{a}@{t}"] for t in TIERS])
print()
print("PER-CELL speedup (NumPy ÷ NumSharp · >1.0 = NumSharp faster)")
print(f"{'aspect':<9}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS))
for a in ASPECTS:
    print(f"{ASPECT_LABEL[a]:<9}" + "".join(f"{sp[f'{a}@{t}']:>8.2f}×" for t in TIERS))
print()
allsp = sorted(sp.items(), key=lambda kv: kv[1])
print("biggest NumSharp wins: " + " · ".join(f"{k} {v:.2f}×" for k, v in reversed(allsp[-4:])))
print("closest / behind:      " + " · ".join(f"{k} {v:.2f}×" for k, v in allsp[:4]))
