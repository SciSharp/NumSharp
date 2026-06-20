#!/usr/bin/env python3
# =============================================================================
# layout_sheet.py — THE Layout subsystem orchestrator + renderer.
#
# The op-matrix (NumSharp.Benchmark.GraphEngine) measures op × dtype × N on
# C-contiguous arrays only. This subsystem fills the memory-LAYOUT axis it omits:
# reduction / copy / elementwise across C, F(ortran), T(ranspose), strided
# `[:, ::2]`, sliced (offset), negstride — NumSharp vs NumPy 2.4.2, identical keys.
#
# Runs each `*_bench.{cs,py}` pair (via benchmark/scripts/bench_common), merges by
# key, and renders ONE sheet -> layout_results.md (+ layout_results.tsv). Driven by
# run_benchmark.py; also standalone:
#   python benchmark/layout/layout_sheet.py [--skip-build]
# =============================================================================
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "layout_results.md")
TSV = os.path.join(HERE, "layout_results.tsv")

# (section label, bench stem, key dim names [0]=size tag, dims to geomean-table over)
BENCHES = [
    ("Reduction (sum/min/max/prod, both axes)", "reduce_layout_bench",
     ["tag", "dt", "lay", "op", "ax"], ["lay", "dt", "op"]),
    ("Copy / identity-ufunc (np.positive)", "copy_path_bench",
     ["tag", "dt", "lay", "kind"], ["lay", "dt"]),
    ("Elementwise (add/mul/neg/abs/sqrt/less/copy)", "elementwise_layout_bench",
     ["tag", "dt", "lay", "op"], ["lay", "dt", "op"]),
]


def render_block(label, rows, dim_names, group_dims):
    parsed = []
    for key, nm, nv, rt in rows:
        parts = key.split("|")
        if len(parts) != len(dim_names):
            continue
        parsed.append((dict(zip(dim_names, parts)), nm, nv, rt))
    L = [f"### {label}", ""]
    if not parsed:
        L.append("_no comparable cells (NumSharp side empty — build/run failed or AV)._\n")
        return "\n".join(L), 0
    tags = sorted({d["tag"] for d, _, _, _ in parsed})
    for gd in group_dims:
        vals = []
        for d, _, _, _ in parsed:
            if d[gd] not in vals:
                vals.append(d[gd])
        L.append(f"**Geomean by {gd}**")
        L.append("")
        L.append("| size | " + " | ".join(vals) + " |")
        L.append("|" + "---|" * (len(vals) + 1))
        for tag in tags:
            cells = []
            for v in vals:
                g = bc.geomean([rt for d, _, _, rt in parsed if d["tag"] == tag and d[gd] == v])
                cells.append(f"{g:.2f} {bc.icon(g)}")
            L.append(f"| {tag} | " + " | ".join(cells) + " |")
        L.append("")
    worst = sorted(parsed, key=lambda r: r[3])[:15]
    L.append("**Worst 15 cells (NumSharp slowest vs NumPy)**")
    L.append("")
    L.append("| key | NumSharp ms | NumPy ms | ratio |")
    L.append("|---|---|---|---|")
    for d, nm, nv, rt in worst:
        key = "\\|".join(d[n] for n in dim_names)
        L.append(f"| {key} | {nm:.4f} | {nv:.4f} | {rt:.2f} {bc.icon(rt)} |")
    L.append("")
    return "\n".join(L), len(parsed)


def main():
    ap = argparse.ArgumentParser(description="Layout subsystem (reduction/copy/elementwise × layout × dtype)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    args = ap.parse_args()

    if not args.skip_build:
        bc.build_core(REPO)

    blocks, tsv_rows, total = [], [], 0
    for label, stem, dims, groups in BENCHES:
        bc.log(f"[layout] {stem} …")
        ns = bc.parse_tsv(bc.run_cs(REPO, os.path.join(HERE, stem + ".cs")))
        npy = bc.parse_tsv(bc.run_py(REPO, os.path.join(HERE, stem + ".py")))
        rows = bc.ratio_rows(ns, npy)
        block, n = render_block(label, rows, dims, groups)
        blocks.append(block)
        for key, nm, nv, _ in rows:
            tsv_rows.append((stem, key, nm, nv))
        total += n
        bc.log(f"    {stem}: {n} comparable cells")

    head = ("# Layout suite — reduction / copy / elementwise × memory layout × dtype\n\n"
            "ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. ✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2.\n"
            "Layouts (8, harmonized with the cast subsystem): `C`, `F` (Fortran), `T` (transpose), "
            "`strided` `[:, ::2]`, `sliced` (offset), `negrow` `[::-1,:]`, `negcol` `[:,::-1]`, "
            "`bcast` (stride-0). Fills the op-matrix's blind spot (it measures C-contiguous only). "
            "100K + 1M elements, best-of-rounds.\n")
    md = head + "\n" + "\n".join(blocks)
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(TSV, "w", encoding="utf-8") as f:
        f.write("bench\tkey\tns_ms\tnp_ms\n")
        for stem, key, nm, nv in tsv_rows:
            f.write(f"{stem}\t{key}\t{nm!r}\t{nv!r}\n")
    bc.log(f"[layout] {total} comparable cells -> {os.path.relpath(MD, REPO)}")
    print(md)


if __name__ == "__main__":
    main()
