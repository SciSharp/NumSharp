#!/usr/bin/env python3
# =============================================================================
# operand_sheet.py — THE Operand-layout subsystem orchestrator + renderer.
#
# The layout subsystem (benchmark/layout) sweeps ONE 2-D operand across memory
# layouts. This subsystem covers the layout classes that grid cannot express:
#   * rank — 1-D contiguous / strided (a[::2]) / reversed (a[::-1])
#   * scalar operand — array+scalar, scalar+array
#   * mixed operand layouts — C+F, C+T (two operands, different layouts)
#   * broadcast in a binary op — +row(1,C), +col(R,1); column-broadcast unary
#
# Runs operand_bench.{cs,py} (via benchmark/scripts/bench_common), merges by key
# (`{case}|{dt}`), and renders ONE case×dtype ratio sheet -> operand_results.md
# (+ .tsv). Driven by run_benchmark.py; also standalone:
#   python benchmark/operand/operand_sheet.py [--skip-build]
# =============================================================================
import argparse
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "operand_results.md")
TSV = os.path.join(HERE, "operand_results.tsv")

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


def render(rows):
    bycase = defaultdict(dict)
    for k, nm, nv, rt in rows:
        if "|" not in k:
            continue
        case, dt = k.split("|")
        bycase[case][dt] = (nm, nv, rt)

    L = ["# Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast (NumSharp vs NumPy 2.4.2)", ""]
    L.append("The layout classes the per-operand layout grid (benchmark/layout) can't express. "
             "ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. "
             "1M elements, best-of-3.")
    L.append("")
    if not rows:
        L.append("_no comparable cells (NumSharp side empty — build/run failed)._")
        return "\n".join(L) + "\n"
    L.append("| case | " + " | ".join(DTS) + " | geomean |")
    L.append("|" + "---|" * (len(DTS) + 2))
    for case in CASES:
        d = bycase.get(case, {})
        cells = [f"{d[dt][2]:.2f} {bc.icon(d[dt][2])}" if dt in d else "—" for dt in DTS]
        g = bc.geomean([d[dt][2] for dt in d])
        L.append(f"| {LABEL.get(case, case)} | " + " | ".join(cells) + f" | {g:.2f} {bc.icon(g)} |")
    L.append("")
    worst = sorted(rows, key=lambda r: r[3])[:12]
    L.append("**Worst 12 cells**")
    L.append("")
    L.append("| key | NumSharp ms | NumPy ms | ratio |")
    L.append("|---|---|---|---|")
    for k, nm, nv, rt in worst:
        L.append(f"| {k} | {nm:.4f} | {nv:.4f} | {rt:.2f} {bc.icon(rt)} |")
    L.append("")
    L.append(f"_{len(rows)} comparable cells._")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Operand-layout subsystem (1-D / scalar / mixed-operand / broadcast)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    args = ap.parse_args()

    if not args.skip_build:
        bc.build_core(REPO)

    bc.log("[operand] operand_bench …")
    ns = bc.parse_tsv(bc.run_cs(REPO, os.path.join(HERE, "operand_bench.cs")))
    npy = bc.parse_tsv(bc.run_py(REPO, os.path.join(HERE, "operand_bench.py")))
    rows = bc.ratio_rows(ns, npy)

    md = render(rows)
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(TSV, "w", encoding="utf-8") as f:
        f.write("key\tns_ms\tnp_ms\n")
        for k, nm, nv, _ in rows:
            f.write(f"{k}\t{nm!r}\t{nv!r}\n")
    bc.log(f"[operand] {len(rows)} comparable cells -> {os.path.relpath(MD, REPO)}")
    print(md)


if __name__ == "__main__":
    main()
