#!/usr/bin/env python3
# Merge NumSharp + NumPy reduce_layout_bench TSVs into a ratio matrix.
# ratio = numpy_ms / numsharp_ms  (>1.0 = NumSharp FASTER).  Usage:
#   python reduce_layout_merge.py numsharp.tsv numpy.tsv
import sys, math
from collections import defaultdict

def load(p):
    d = {}
    for ln in open(p):
        ln = ln.strip()
        if not ln or "\t" not in ln: continue
        k, v = ln.split("\t")
        try: d[k] = float(v)
        except ValueError: pass
    return d

ns = load(sys.argv[1]); npy = load(sys.argv[2])
keys = [k for k in npy if k in ns]
rows = []
for k in keys:
    tag, dt, lay, op, ax = k.split("|")
    nm, nv = ns[k], npy[k]
    ratio = nv / nm if nm > 0 else float('nan')
    rows.append((tag, dt, lay, op, ax, nm, nv, ratio))

def icon(r):
    if r != r: return "?"
    return "✅" if r >= 1.0 else "🟡" if r >= 0.5 else "🟠" if r >= 0.2 else "🔴"

# ---- per (size, op, layout) geomean across dtypes ----
def geomean(xs):
    xs = [x for x in xs if x==x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

print("# Reduction × Layout × dtype parity (NumSharp vs NumPy 2.4.2)\n")
print("ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1 🟡≥0.5 🟠≥0.2 🔴<0.2\n")

# Geomean by layout per size (across all dtypes/ops/axes)
print("## Geomean by layout (all dtypes/ops/axes)\n")
print("| size | " + " | ".join(LAYS := ["C","F","T","strided","negstride","sliced"]) + " |")
print("|" + "---|"*(len(LAYS)+1))
for tag in ["100K","1M"]:
    cells=[]
    for lay in LAYS:
        g = geomean([r[7] for r in rows if r[0]==tag and r[2]==lay])
        cells.append(f"{g:.2f} {icon(g)}")
    print(f"| {tag} | " + " | ".join(cells) + " |")

# Geomean by dtype per size
print("\n## Geomean by dtype (all layouts/ops/axes)\n")
DTS=["f64","f32","c128","dec","f16","i32","i64"]
print("| size | " + " | ".join(DTS) + " |")
print("|" + "---|"*(len(DTS)+1))
for tag in ["100K","1M"]:
    cells=[]
    for dt in DTS:
        g = geomean([r[7] for r in rows if r[0]==tag and r[1]==dt])
        cells.append(f"{g:.2f} {icon(g)}")
    print(f"| {tag} | " + " | ".join(cells) + " |")

# Geomean by op per size
print("\n## Geomean by op (all dtypes/layouts/axes)\n")
print("| size | sum | min | max | prod |")
print("|---|---|---|---|---|")
for tag in ["100K","1M"]:
    cells=[]
    for op in ["sum","min","max","prod"]:
        g = geomean([r[7] for r in rows if r[0]==tag and r[3]==op])
        cells.append(f"{g:.2f} {icon(g)}")
    print(f"| {tag} | " + " | ".join(cells) + " |")

# Worst 30 cells
print("\n## Worst 30 cells (NumSharp slowest vs NumPy)\n")
print("| key | NumSharp ms | NumPy ms | ratio |")
print("|---|---|---|---|")
for r in sorted(rows, key=lambda r: r[7])[:30]:
    print(f"| {r[0]}\\|{r[1]}\\|{r[2]}\\|{r[3]}\\|{r[4]} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.2f} {icon(r[7])} |")

# Best 12
print("\n## Best 12 cells (NumSharp fastest vs NumPy)\n")
print("| key | NumSharp ms | NumPy ms | ratio |")
print("|---|---|---|---|")
for r in sorted(rows, key=lambda r: -r[7])[:12]:
    print(f"| {r[0]}\\|{r[1]}\\|{r[2]}\\|{r[3]}\\|{r[4]} | {r[5]:.4f} | {r[6]:.4f} | {r[7]:.2f} {icon(r[7])} |")

print(f"\n_{len(rows)} cells compared._")
