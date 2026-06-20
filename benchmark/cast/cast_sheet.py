#!/usr/bin/env python3
# =============================================================================
# cast_sheet.py — THE Cast subsystem orchestrator + renderer.
#
# The op-matrix has no astype/cast coverage at all. This subsystem sweeps the
# full src→dst cast matrix (15×15 dtypes × 8 memory layouts at 1M) through
# astype → DefaultEngine.Cast → NpyIter.Copy, NumSharp vs NumPy 2.4.2.
#
# Runs cast_matrix_bench.{cs,py} (via benchmark/scripts/bench_common), merges by
# key (`1M|src|layout|dst`), and renders ONE sheet -> cast_results.md (+ .tsv).
# Driven by run_benchmark.py; also standalone:
#   python benchmark/cast/cast_sheet.py [--skip-build]
# =============================================================================
import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "cast_results.md")
TSV = os.path.join(HERE, "cast_results.tsv")

DTS = ["bool", "u8", "i8", "i16", "u16", "i32", "u32", "i64", "u64", "char", "f16", "f32", "f64", "dec", "c128"]
LAYS = ["C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast"]
FLOATS = {"f16", "f32", "f64", "c128"}
NARROW = {"bool", "u8", "i8", "i16", "u16", "char"}


def family(s, d):
    if s in FLOATS and d in NARROW:
        return "float/cplx → narrow-int (bool/u8/i8/i16/u16/char)"
    if d == "bool":
        return "* → bool"
    if s == d:
        return "same-type diagonal (copy)"
    if d in ("u8", "i8", "i16", "u16", "char"):
        return "int → sub-word (narrow)"
    return f"{s} → {d}"


def render(ns, npy):
    rows = []  # (src, lay, dst, ns_ms, npy_ms, ratio)
    for k, nm in ns.items():
        parts = k.split("|")
        if len(parts) != 4:
            continue
        _, src, lay, dst = parts
        nv = npy.get(k)
        ratio = (nv / nm) if (nv is not None and nm > 0) else float("nan")
        rows.append((src, lay, dst, nm, nv, ratio))
    R = {(s, l, d): rt for (s, l, d, nm, nv, rt) in rows}

    L = []
    L.append("# Cast matrix — astype src→dst × layout × dtype (NumSharp vs NumPy 2.4.2)")
    L.append("")
    L.append("Full `astype(dst, copy:true)` sweep over every src→dst dtype pair × 8 memory layouts "
             "at 1M elements, best-of-3. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**.")
    L.append("✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2 · `—` = no NumPy counterpart (Decimal has no NumPy dtype).")
    L.append("")

    cmp = [r for r in rows if r[5] == r[5]]          # comparable (has a NumPy ratio)
    if not cmp:
        L.append("_no comparable cells (NumSharp side empty — build/run failed)._")
        return "\n".join(L), rows

    lagc = [r for r in cmp if r[5] < 1.0]
    L.append("## Summary")
    L.append("")
    L.append(f"- **{len(lagc)} / {len(cmp)}** comparable cells lag (<1.0); **{len(cmp) - len(lagc)}** win (≥1.0).")
    for name, lo, hi in [("🔴 <0.2", 0, 0.2), ("🟠 0.2–0.5", 0.2, 0.5), ("🟡 0.5–1.0", 0.5, 1.0)]:
        sub = [r for r in lagc if lo <= r[5] < hi]
        fc = Counter(family(r[0], r[2]) for r in sub)
        top = "; ".join(f"{v}× {k}" for k, v in fc.most_common(3))
        L.append(f"- **{name}** — {len(sub)} cells.{(' Top: ' + top) if top else ''}")
    fn = {s: bc.geomean([r[5] for r in cmp if r[0] == s and r[2] in NARROW]) for s in ["f32", "f64", "f16", "c128"]}
    L.append("")
    L.append("**float/complex → narrow-int geomean by src** (the historical cliff): "
             + ", ".join(f"`{s}`→narrow **{fn[s]:.2f}**" for s in ["f32", "f64", "f16", "c128"]) + ".")
    L.append("")

    def gm_table(title, keyfn, vals):
        L.append(f"## {title}")
        L.append("")
        L.append("| " + " | ".join(vals) + " |")
        L.append("|" + "---|" * len(vals))
        cells = []
        for v in vals:
            g = bc.geomean([r[5] for r in rows if keyfn(r) == v])
            cells.append(f"{g:.2f} {bc.icon(g)}")
        L.append("| " + " | ".join(cells) + " |")
        L.append("")

    gm_table("Geomean by layout (all src×dst, excl. Decimal)", lambda r: r[1], LAYS)
    gm_table("Geomean by src dtype (all layouts×dst)", lambda r: r[0], DTS)
    gm_table("Geomean by dst dtype (all layouts×src)", lambda r: r[2], DTS)

    for lay in LAYS:
        L.append(f"## Layout: {lay}  (rows=src, cols=dst)")
        L.append("")
        L.append("| src\\dst | " + " | ".join(DTS) + " |")
        L.append("|" + "---|" * (len(DTS) + 1))
        for s in DTS:
            cells = []
            for d in DTS:
                rt = R.get((s, lay, d), float("nan"))
                cells.append("—" if rt != rt else f"{rt:.2f}{bc.icon(rt)}")
            L.append(f"| **{s}** | " + " | ".join(cells) + " |")
        L.append("")

    lag = sorted([r for r in rows if r[5] == r[5] and r[5] < 1.0], key=lambda r: r[5])
    L.append(f"## Lagging cells (<1.0) — the worklist  ({len(lag)} cells)")
    L.append("")
    L.append("| key | NumSharp ms | NumPy ms | ratio |")
    L.append("|---|---|---|---|")
    for (src, lay, dst, nm, nv, rt) in lag:
        L.append(f"| {src}\\|{lay}\\|{dst} | {nm:.4f} | {nv:.4f} | {rt:.2f} {bc.icon(rt)} |")
    L.append("")
    L.append(f"_{len(cmp)} comparable cells ({len(rows)} NumSharp rows; {len(lag)} lagging <1.0)._")
    return "\n".join(L), rows


def main():
    ap = argparse.ArgumentParser(description="Cast subsystem (astype src→dst × layout × dtype)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    args = ap.parse_args()

    if not args.skip_build:
        bc.build_core(REPO)

    bc.log("[cast] cast_matrix_bench …")
    ns = bc.parse_tsv(bc.run_cs(REPO, os.path.join(HERE, "cast_matrix_bench.cs")))
    npy = bc.parse_tsv(bc.run_py(REPO, os.path.join(HERE, "cast_matrix_bench.py")))
    md, rows = render(ns, npy)

    with open(MD, "w", encoding="utf-8") as f:
        f.write(md)
    with open(TSV, "w", encoding="utf-8") as f:
        f.write("key\tns_ms\tnp_ms\n")
        for (src, lay, dst, nm, nv, _) in rows:
            f.write(f"1M|{src}|{lay}|{dst}\t{nm!r}\t{('NA' if nv is None else repr(nv))}\n")
    bc.log(f"[cast] {len(rows)} NumSharp rows -> {os.path.relpath(MD, REPO)}")
    print(md)


if __name__ == "__main__":
    main()
