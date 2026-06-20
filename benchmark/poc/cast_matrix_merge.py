#!/usr/bin/env python3
# Merge NumSharp + NumPy cast_matrix TSVs into per-layout 15x15 ratio matrices.
# ratio = numpy_ms / numsharp_ms  (>1.0 = NumSharp FASTER). Phase 0 worklist.
#   python cast_matrix_merge.py cm_numsharp.tsv cm_numpy.tsv > cast_matrix.md
import sys, math

def load(p):
    d = {}
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if not ln or "\t" not in ln: continue
        k, v = ln.split("\t")
        try: d[k] = float(v)
        except ValueError: pass
    return d

ns = load(sys.argv[1]); npy = load(sys.argv[2])

DTS = ["bool","u8","i8","i16","u16","i32","u32","i64","u64","char","f16","f32","f64","dec","c128"]
LAYS = ["C","F","T","sliced","negrow","negcol","strided","bcast"]

def icon(r):
    if r != r: return "?"
    return "✅" if r >= 1.0 else "🟡" if r >= 0.5 else "🟠" if r >= 0.2 else "🔴"

def geomean(xs):
    xs = [x for x in xs if x == x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float('nan')

rows = []  # (src, lay, dst, ns_ms, npy_ms, ratio)
for k, nm in ns.items():
    _, src, lay, dst = k.split("|")
    nv = npy.get(k)
    ratio = (nv / nm) if (nv is not None and nm > 0) else float('nan')
    rows.append((src, lay, dst, nm, nv, ratio))
R = {(s, l, d): rt for (s, l, d, nm, nv, rt) in rows}

print("# Cast Matrix — astype src→dst × layout × dtype (NumSharp vs NumPy 2.4.2)\n")
print("Phase 0 of `CAST_BEAT_NUMPY_PLAN.md`. 1M elements, best-of-3. "
      "ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.")
print("✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy dtype (Decimal: pinned vs the Converts table, not NumPy).\n")

# ---- Executive summary: severity bands + conversion-family breakdown -------
from collections import Counter
cmp = [r for r in rows if r[5] == r[5]]            # comparable (has NumPy ratio)
lagc = [r for r in cmp if r[5] < 1.0]
FLOATS = {"f16", "f32", "f64", "c128"}
NARROW = {"bool", "u8", "i8", "i16", "u16", "char"}
def family(s, d):
    if s in FLOATS and d in NARROW: return "float/cplx → narrow-int (bool/u8/i8/i16/u16/char)"
    if d == "bool":                 return "* → bool"
    if s == d:                      return "same-type diagonal (copy)"
    if d in ("u8", "i8", "i16", "u16", "char"): return "int → sub-word (narrow)"
    return f"{s} → {d}"
print("## Executive summary\n")
print(f"- **{len(lagc)} / {len(cmp)}** comparable cells lag (<1.0); **{len(cmp)-len(lagc)}** win (≥1.0).")
bands = [("🔴 <0.2", 0, 0.2), ("🟠 0.2–0.5", 0.2, 0.5), ("🟡 0.5–1.0", 0.5, 1.0)]
for name, lo, hi in bands:
    sub = [r for r in lagc if lo <= r[5] < hi]
    fc = Counter(family(r[0], r[2]) for r in sub)
    top = "; ".join(f"{v}× {k}" for k, v in fc.most_common(3))
    print(f"- **{name}** — {len(sub)} cells. Top: {top}")
print("\n**float/complex → narrow-int geomean by src** (the dominant cliff): " +
      ", ".join(f"`{s}`→narrow **{geomean([r[5] for r in cmp if r[0]==s and r[2] in NARROW]):.2f}**"
                for s in ["f32", "f64", "f16", "c128"]) + ".")
print("`float→i32` itself is mostly **won** (contiguous cvtt kernel); the fire is the *narrowing* "
      "float→{i8,u8,i16,u16,char,bool} pack, which has no SIMD kernel and falls to the IL scalar.\n")

print("## Geomean by layout (all src×dst, excl. Decimal)\n")
print("| " + " | ".join(LAYS) + " |")
print("|" + "---|" * len(LAYS))
print("| " + " | ".join(f"{geomean([r[5] for r in rows if r[1]==lay]):.2f} {icon(geomean([r[5] for r in rows if r[1]==lay]))}" for lay in LAYS) + " |")

print("\n## Geomean by src dtype (all layouts×dst)\n")
print("| " + " | ".join(DTS) + " |")
print("|" + "---|" * len(DTS))
print("| " + " | ".join(f"{geomean([r[5] for r in rows if r[0]==s]):.2f}{icon(geomean([r[5] for r in rows if r[0]==s]))}" for s in DTS) + " |")

print("\n## Geomean by dst dtype (all layouts×src)\n")
print("| " + " | ".join(DTS) + " |")
print("|" + "---|" * len(DTS))
print("| " + " | ".join(f"{geomean([r[5] for r in rows if r[2]==d]):.2f}{icon(geomean([r[5] for r in rows if r[2]==d]))}" for d in DTS) + " |")

for lay in LAYS:
    print(f"\n## Layout: {lay}  (rows=src, cols=dst)\n")
    print("| src\\dst | " + " | ".join(DTS) + " |")
    print("|" + "---|" * (len(DTS) + 1))
    for s in DTS:
        cells = []
        for d in DTS:
            rt = R.get((s, lay, d), float('nan'))
            cells.append("—" if rt != rt else f"{rt:.2f}{icon(rt)}")
        print(f"| **{s}** | " + " | ".join(cells) + " |")

lag = sorted([r for r in rows if r[5] == r[5] and r[5] < 1.0], key=lambda r: r[5])
print(f"\n## Lagging cells (<1.0) — the worklist  ({len(lag)} cells)\n")
print("| key | NumSharp ms | NumPy ms | ratio |")
print("|---|---|---|---|")
for (src, lay, dst, nm, nv, rt) in lag:
    print(f"| {src}\\|{lay}\\|{dst} | {nm:.4f} | {nv:.4f} | {rt:.2f} {icon(rt)} |")

ncmp = len([r for r in rows if r[5] == r[5]])
print(f"\n_{ncmp} comparable cells ({len(rows)} NS rows; {len(lag)} lagging <1.0)._")
