# Renders the FULL-sweep bar summary for npyiter_fullsweep_bench.{cs,py}:
# every distinct NpyIter operation family from rounds 1-3, across the four size
# tiers scalar(1)/1K/100K/1M. Self-contained — the clean run (i9-13900K, NumPy
# 2.4.2, Release) is embedded below as (NumSharp_ms, NumPy_ms) per id. To re-chart
# a fresh run pass the two recorded output files:
#       python npyiter_fullsweep_chart.py ns_run.txt np_run.txt
#
# speedup = NumPy_time / NumSharp_time: > 1.0 means NumSharp is FASTER.
# Axis is the official-report style (slower <-- 1.0 (parity) --> faster).
import math, re, sys

# (NumSharp ms, NumPy ms) per id, from npyiter_fullsweep_{ns,np}.txt
DATA = {
    "add@1": (0.0001746, 0.0002786), "sqrt@1": (0.0001534, 0.0002446), "copy@1": (0.0001524, 0.0002527),
    "sadd@1": (0.0001713, 0.000278), "bcast@1": (0.0001669, 0.0002804), "frev@1": (0.0001529, 0.0002534),
    "castbuf@1": (0.0002766, 0.0007641), "mixbuf@1": (0.0002969, 0.0004718), "psum@1": (0.0008596, 0.00399),
    "sumax0@1": (0.00176, 0.00425), "sumax1@1": (0.00143, 0.00426), "sumdt@1": (0.0014, 0.00496),
    "amin@1": (0.00146, 0.00392), "cumsum@1": (0.0017, 0.00298), "anyff@1": (0.0001447, 0.0035),
    "anyeh@1": (0.000151, 0.00359), "where@1": (0.00173, 0.00155), "bread@1": (0.00106, 0.0004324),
    "bassign@1": (0.0002464, 0.0008561), "cnz@1": (0.0001292, 0.0003646), "argw@1": (0.00175, 0.00419),
    "gather@1": (0.00583, 0.00169), "scatter@1": (0.00265, 0.00169), "flatten@1": (0.00137, 0.0007405),
    "astype@1": (0.00179, 0.0007993), "ravelT@1": (0.00112, 0.0004724), "inplace@1": (0.0006855, 0.00154),
    "lessbool@1": (0.0006228, 0.0004152), "unravel@1": (0.00315, 0.00167), "ravelmi@1": (0.00422, 0.00184),
    "cplx@1": (0.0006038, 0.0003765), "f16@1": (0.0006618, 0.0004048), "i8@1": (0.000699, 0.0003958),
    "fuse7@1": (0.000459, 0.00786), "reuse@1": (7.37e-05, 0.00041), "add@1K": (0.0005615, 0.0009101),
    "sqrt@1K": (0.00278, 0.00361), "copy@1K": (0.0004248, 0.0009113), "sadd@1K": (0.0008169, 0.0014),
    "bcast@1K": (0.0007763, 0.00177), "frev@1K": (0.0006145, 0.0009171), "castbuf@1K": (0.0009473, 0.00268),
    "mixbuf@1K": (0.0011, 0.00287), "psum@1K": (0.00112, 0.00475), "sumax0@1K": (0.00219, 0.0049),
    "sumax1@1K": (0.00192, 0.00508), "sumdt@1K": (0.00215, 0.0059), "amin@1K": (0.00343, 0.00584),
    "cumsum@1K": (0.00374, 0.00644), "anyff@1K": (0.0006325, 0.0038), "anyeh@1K": (0.0006252, 0.00407),
    "where@1K": (0.00299, 0.00279), "bread@1K": (0.00291, 0.004), "bassign@1K": (0.00699, 0.00781),
    "cnz@1K": (0.0002638, 0.00113), "argw@1K": (0.00246, 0.00577), "gather@1K": (0.01076, 0.00479),
    "scatter@1K": (0.00677, 0.00548), "flatten@1K": (0.00278, 0.00128), "astype@1K": (0.00242, 0.00193),
    "ravelT@1K": (0.00436, 0.00291), "inplace@1K": (0.0009437, 0.0009421), "lessbool@1K": (0.00131, 0.0005127),
    "unravel@1K": (0.0092, 0.00583), "ravelmi@1K": (0.00962, 0.00655), "cplx@1K": (0.00179, 0.00135),
    "f16@1K": (0.01184, 0.00728), "i8@1K": (0.0007263, 0.00114), "fuse7@1K": (0.00157, 0.00505),
    "reuse@1K": (0.0001783, 0.0009041), "par8@1K": (0.01138, 0.00734), "add@100K": (0.02873, 0.02563),
    "sqrt@100K": (0.2653, 0.29574), "copy@100K": (0.02346, 0.03024), "sadd@100K": (0.07487, 0.06388),
    "bcast@100K": (0.02356, 0.02191), "frev@100K": (0.03778, 0.03224), "castbuf@100K": (0.06, 0.08831),
    "mixbuf@100K": (0.05673, 0.08331), "psum@100K": (0.01145, 0.03046), "sumax0@100K": (0.02143, 0.02414),
    "sumax1@100K": (0.01297, 0.03116), "sumdt@100K": (0.07483, 0.08833), "amin@100K": (0.11457, 0.04725),
    "cumsum@100K": (0.17089, 0.30069), "anyff@100K": (0.0499, 0.00755), "anyeh@100K": (0.0006043, 0.00382),
    "where@100K": (0.09529, 0.07526), "bread@100K": (0.10128, 0.30803), "bassign@100K": (0.32685, 0.66439),
    "cnz@100K": (0.01865, 0.07803), "argw@100K": (0.078, 0.07884), "gather@100K": (0.52842, 0.25817),
    "scatter@100K": (0.67493, 0.40474), "flatten@100K": (0.0577, 0.02286), "astype@100K": (0.03944, 0.05691),
    "ravelT@100K": (0.11461, 0.06983), "inplace@100K": (0.02392, 0.02101), "lessbool@100K": (0.05927, 0.02219),
    "unravel@100K": (0.55793, 0.57071), "ravelmi@100K": (0.43671, 0.44768), "cplx@100K": (0.10294, 0.09153),
    "f16@100K": (1.124, 0.68409), "i8@100K": (0.00422, 0.05253), "fuse7@100K": (0.13635, 0.20744),
    "reuse@100K": (0.02722, 0.02627), "par8@100K": (0.31851, 0.66451), "add@1M": (0.77995, 0.74027),
    "sqrt@1M": (2.877, 3.002), "copy@1M": (0.42044, 0.49083), "sadd@1M": (3.483, 3.113),
    "bcast@1M": (0.43541, 0.46006), "frev@1M": (0.49108, 0.4947), "castbuf@1M": (0.58405, 0.78385),
    "mixbuf@1M": (0.92281, 1.183), "psum@1M": (0.18441, 0.41689), "sumax0@1M": (0.24297, 0.25697),
    "sumax1@1M": (0.19146, 0.38875), "sumdt@1M": (0.7513, 0.79558), "amin@1M": (1.027, 0.44794),
    "cumsum@1M": (1.668, 4.075), "anyff@1M": (0.49681, 0.03767), "anyeh@1M": (0.0006008, 0.00367),
    "where@1M": (1.628, 2.504), "bread@1M": (1.033, 3.481), "bassign@1M": (3.343, 6.78),
    "cnz@1M": (0.24056, 0.79728), "argw@1M": (0.68267, 1.787), "gather@1M": (7.2, 5.086),
    "scatter@1M": (9.31, 5.835), "flatten@1M": (0.79685, 1.37), "astype@1M": (0.60897, 1.109),
    "ravelT@1M": (1.575, 2.228), "inplace@1M": (0.43206, 0.43062), "lessbool@1M": (0.70562, 0.40143),
    "unravel@1M": (6.65, 5.337), "ravelmi@1M": (4.397, 5.166), "cplx@1M": (4.623, 3.595),
    "f16@1M": (11.313, 6.611), "i8@1M": (0.04092, 0.5062), "fuse7@1M": (6.095, 7.038),
    "reuse@1M": (0.74078, 0.75122), "par8@1M": (2.804, 6.649),
}

TIERS = ["1", "1K", "100K", "1M"]
TIER_LABEL = {"1": "scalar", "1K": "1K", "100K": "100K", "1M": "1M"}
# main families (NumPy has a like-for-like equivalent), grouped by category
CATEGORIES = [
    ("elementwise", ["add", "sqrt", "copy", "sadd", "bcast", "frev", "castbuf", "mixbuf"]),
    ("reductions",  ["psum", "sumax0", "sumax1", "sumdt", "amin", "cumsum", "anyff", "anyeh"]),
    ("selection",   ["where", "bread", "bassign", "cnz", "argw", "gather", "scatter"]),
    ("copy/cast",   ["flatten", "astype", "ravelT", "inplace", "lessbool"]),
    ("index-math",  ["unravel", "ravelmi"]),
    ("dtypes",      ["cplx", "f16", "i8"]),
]
MAIN = [f for _, fs in CATEGORIES for f in fs]
DIVIDENDS = ["fuse7", "reuse", "par8"]   # NumPy has no equivalent machinery
FAM_LABEL = {"sadd": "strided", "frev": "reversed", "castbuf": "castbuf", "mixbuf": "mixbuf",
             "psum": "sum", "sumax0": "sum ax0", "sumax1": "sum ax1", "sumdt": "sum dt=",
             "cumsum": "cumsum", "anyff": "any(F)", "anyeh": "any(hit)", "where": "where",
             "bread": "a[mask]", "bassign": "a[mask]=", "cnz": "count_nz", "argw": "argwhere",
             "gather": "a[idx]", "scatter": "a[idx]=", "flatten": "flatten", "astype": "astype",
             "ravelT": "ravel.T", "inplace": "in-place", "lessbool": "less->b",
             "unravel": "unravel", "ravelmi": "ravel_mi", "cplx": "complex", "f16": "float16", "i8": "int8"}

SCALE, WIDTH = 10.0, 20
EIGHTHS = ["", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
UNIT = {"ns": 1e-6, "us": 1e-3, "ms": 1.0}
PLINE = re.compile(r"^(\w+@\w+)\s+([\d.]+)\s+(ns|us|ms)\s*$")

def parse(path):
    d = {}
    for ln in open(path, encoding="utf-8"):
        m = PLINE.match(ln.rstrip())
        if m: d[m.group(1)] = float(m.group(2)) * UNIT[m.group(3)]
    return d

def bar(s):
    u = s * SCALE
    if u >= WIDTH: return "█" * (WIDTH - 1) + "▶"
    full = int(u); frac = EIGHTHS[int((u - full) * 8)]
    t = "█" * full + frac; pad = WIDTH - len(t)
    return t + (" " + "." * (pad - 1) if pad >= 3 else " " * pad)

def geomean(v): return math.exp(sum(math.log(x) for x in v) / len(v))

def line(label, sp):
    g = geomean(sp); win = sum(1 for x in sp if x > 1.0); lose = len(sp) - win
    tag = "   ◄ PARITY" if 0.97 <= g <= 1.03 else ("   ◄ SLOWER" if g < 0.97 else "")
    print(f"{label:<11}{bar(g)}  {g:5.2f}x   ({win:3d} win /{lose:3d} lose){tag}")

if len(sys.argv) > 2:
    ns, npp = parse(sys.argv[1]), parse(sys.argv[2])
    SP = {k: npp[k] / ns[k] for k in ns if k in npp}
else:
    SP = {k: np_ / ns_ for k, (ns_, np_) in DATA.items()}

def sp_of(fam, tier): return SP.get(f"{fam}@{tier}")
def fam_sps(fam): return [sp_of(fam, t) for t in TIERS if sp_of(fam, t) is not None]
def tier_sps(tier, fams): return [sp_of(f, tier) for f in fams if sp_of(f, tier) is not None]

HDR = "        slower ◄───────── 1.0 (parity) ─────────► faster"
print("NpyIter FULL sweep — 33 op families x 4 size tiers, 143 measured pairs (i9-13900K, NumPy 2.4.2)")
print("speedup = NumPy / NumSharp per call · >1.0x = NumSharp faster")
print()
print("BY SIZE TIER  (geomean over all 33 NumPy-equivalent families)")
print(HDR)
for t in TIERS: line(TIER_LABEL[t], tier_sps(t, MAIN))
line("ALL", [SP[f"{f}@{t}"] for f in MAIN for t in TIERS if f"{f}@{t}" in SP])
print()
print("BY CATEGORY  (geomean over its families, all sizes)")
print(HDR)
for name, fams in CATEGORIES:
    line(name, [SP[f"{f}@{t}"] for f in fams for t in TIERS if f"{f}@{t}" in SP])
print()
print("CATEGORY x TIER geomean")
print(f"{'category':<12}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS))
for name, fams in CATEGORIES:
    cells = "".join(f"{geomean(tier_sps(t, fams)):>8.2f}x" for t in TIERS)
    print(f"{name:<12}{cells}")
print()
print("PER-FAMILY x TIER  (NumPy / NumSharp; >1.0 = NumSharp faster; bold-ish = behind)")
print(f"{'family':<11}" + "".join(f"{TIER_LABEL[t]:>9}" for t in TIERS) + "    geomean")
for name, fams in CATEGORIES:
    print(f"-- {name}")
    for f in fams:
        cells = "".join(f"{sp_of(f,t):>8.2f}x" if sp_of(f,t) else f"{'-':>9}" for t in TIERS)
        print(f"  {FAM_LABEL.get(f,f):<9}{cells}   {geomean(fam_sps(f)):>6.2f}x")
print()
print("ARCHITECTURE DIVIDENDS  (NumPy has no equivalent machinery)")
print(f"{'':11}{'baseline NumPy does instead':<34}" + "".join(f"{TIER_LABEL[t]:>8}" for t in TIERS))
DIV_DESC = {"fuse7": "chain of 6 np.add(out=)", "reuse": "rebuild nditer each call",
            "par8": "single-thread np.sin (never threads)"}
for f in DIVIDENDS:
    cells = "".join(f"{sp_of(f,t):>7.2f}x" if sp_of(f,t) else f"{'-':>8}" for t in TIERS)
    print(f"{f:<11}{DIV_DESC[f]:<34}{cells}")
print()
allmain = sorted(((k, SP[k]) for k in SP if k.split("@")[0] in MAIN), key=lambda kv: kv[1])
print("biggest NumSharp wins: " + " · ".join(f"{k} {v:.2f}x" for k, v in reversed(allmain[-5:])))
print("most behind:           " + " · ".join(f"{k} {v:.2f}x" for k, v in allmain[:5]))
