#!/usr/bin/env python3
# opnd_layout_run.py — POC runner: build Core, run opnd_layout_bench.{cs,py} via the
# shared driver, merge to a case×dtype ratio table -> opnd_layout_results.md.
#   python benchmark/poc/opnd_layout_run.py
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

bc.build_core(REPO)
ns = bc.parse_tsv(bc.run_cs(REPO, os.path.join(HERE, "opnd_layout_bench.cs")))
npy = bc.parse_tsv(bc.run_py(REPO, os.path.join(HERE, "opnd_layout_bench.py")))
rows = bc.ratio_rows(ns, npy)

bycase = defaultdict(dict)
for k, nm, nv, rt in rows:
    case, dt = k.split("|")
    bycase[case][dt] = (nm, nv, rt)

CASES = ["1d_C", "1d_strided", "1d_rev", "scalar_rhs", "scalar_lhs",
         "mix_C_F", "mix_C_T", "bcast_row", "bcast_col", "colbcast_unary"]
DTS = ["f64", "f32", "f16", "i32", "i64", "c128"]
LABEL = {
    "1d_C": "1-D contiguous (a+a)", "1d_strided": "1-D strided a[::2]", "1d_rev": "1-D reversed a[::-1]",
    "scalar_rhs": "array + scalar", "scalar_lhs": "scalar + array",
    "mix_C_F": "mixed C + F", "mix_C_T": "mixed C + T",
    "bcast_row": "binary broadcast +row(1,C)", "bcast_col": "binary broadcast +col(R,1)",
    "colbcast_unary": "col-broadcast unary (inner stride-0)",
}

L = ["# POC — operand / extra layout classes (NumSharp vs NumPy 2.4.2)", ""]
L.append("Layout classes the op×layout×dtype matrix omits. ratio = NumPy_ms / NumSharp_ms — "
         "**>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. 1M elements, best-of-3.")
L.append("")
L.append("| case | " + " | ".join(DTS) + " | geomean |")
L.append("|" + "---|" * (len(DTS) + 2))
for case in CASES:
    d = bycase.get(case, {})
    cells = [f"{d[dt][2]:.2f} {bc.icon(d[dt][2])}" if dt in d else "—" for dt in DTS]
    g = bc.geomean([d[dt][2] for dt in d])
    L.append(f"| {LABEL[case]} | " + " | ".join(cells) + f" | {g:.2f} {bc.icon(g)} |")

worst = sorted(rows, key=lambda r: r[3])[:12]
L.append("")
L.append("**Worst 12 (key · NumSharp ms · NumPy ms · ratio)**")
L.append("")
L.append("| key | NS ms | NP ms | ratio |")
L.append("|---|---|---|---|")
for k, nm, nv, rt in worst:
    L.append(f"| {k} | {nm:.4f} | {nv:.4f} | {rt:.2f} {bc.icon(rt)} |")

md = "\n".join(L) + "\n"
with open(os.path.join(HERE, "opnd_layout_results.md"), "w", encoding="utf-8") as f:
    f.write(md)
print(md)
bc.log(f"[opnd_layout] {len(rows)} comparable cells -> benchmark/poc/opnd_layout_results.md")
