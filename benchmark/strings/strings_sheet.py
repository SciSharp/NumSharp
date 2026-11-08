#!/usr/bin/env python3
# =============================================================================
# strings_sheet.py — THE Strings subsystem orchestrator + renderer (strings phase 2, P2.7).
#
# Text computation over the three NumPy string dtypes — S (bytes), U (UCS-4) and StringDType (T): comparisons,
# concatenation and repetition, the reductions, sorting / searching / sets, the np.strings functions (search, transforms,
# case maps, predicates, translate, mod, the codecs) and the text → number cast, on data kinds that separate the cases
# the kernels branch on (short ASCII, non-ASCII, long arena texts, missing values, digits), at 1K and 100K elements, 1M
# for a core set, plus stepped views. The op-matrix has no string dtype at all.
#
# The NumPy twin runs FIRST: it writes the data (np.save, a StringDType array as NumPy's pickle) into a temporary
# directory, so strings_bench.cs times the very same arrays (np.load_npy). Keys `{op}|{kind}|{n}` merge both sides; a
# call one side refuses has no comparand and is dropped. Renders strings_results.md (+ .tsv). Driven by
# run_benchmark.py; also standalone:
#   python benchmark/strings/strings_sheet.py [--skip-build]
# =============================================================================
import argparse
import os
import shutil
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "strings_results.md")
TSV = os.path.join(HERE, "strings_results.tsv")

# Families in report order: (title, ops).
FAMILIES = [
    ("Comparisons", ["eq", "lt"]),
    ("Concatenation and repetition", ["add", "mul3"]),
    ("Reductions", ["max", "sum"]),
    ("Sorting, searching, sets", ["sort", "argsort", "unique", "isin"]),
    ("np.strings: length and search", ["str_len", "find", "count", "startswith"]),
    ("np.strings: transforms", ["strip", "replace", "center", "zfill", "partition", "slice"]),
    ("np.strings: case maps and predicates", ["upper", "title", "isalpha", "isdigit"]),
    ("np.strings: translate, mod, codecs", ["translate", "mod", "encode", "decode"]),
    ("Text → number", ["astype_f8"]),
]
OP_LABEL = {
    "eq": "a == b", "lt": "a < b", "add": "np.strings.add(a, b)", "mul3": "np.strings.multiply(a, 3)",
    "max": "np.max(a)", "sum": "np.sum(a) (T only; 1K: NumPy's is quadratic)", "sort": "np.sort(a)",
    "argsort": "np.argsort(a)", "unique": "np.unique(a)", "isin": "np.isin(a, a[:100])",
    "str_len": "str_len(a)", "find": "find(a, 'a')", "count": "count(a, 'a')", "startswith": "startswith(a, 'a')",
    "strip": "strip(a, 'ab')", "replace": "replace(a, 'a', 'xy')", "center": "center(a, 12)",
    "zfill": "zfill(a, 10)", "partition": "partition(a, 'a')", "slice": "slice(a, 1, 4)", "upper": "upper(a)",
    "title": "title(a)", "isalpha": "isalpha(a)", "isdigit": "isdigit(a)", "translate": "translate(a, abc→xyz)",
    "mod": "mod('<%s>', a)", "encode": "encode(a, 'utf-8')", "decode": "decode(a, 'utf-8')",
    "astype_f8": "a.astype(float64)",
}
KIND_LABEL = {
    "u8": "U, 1–8 ASCII", "u8n": "U, 20 % non-ASCII", "u40": "U, 20–40 chars", "s8": "S, 1–8 ASCII",
    "t8": "T, 1–8 ASCII", "t40": "T, 20–40 chars (arena)", "tna": "T, 10 % NA", "u8d": "U, digits",
    "s8d": "S, digits", "t8d": "T, digits", "u8@step2": "U, a[::2]", "t8@step2": "T, a[::2]",
}
SIZES = [1000, 100000, 1000000]
SIZE_LABEL = {1000: "1K", 100000: "100K", 1000000: "1M"}


def dtype_of(kind):
    """The string dtype a data kind holds: S, U or T (StringDType)."""
    return {"s": "S", "u": "U", "t": "T"}[kind[0]]


def render(ns, npy):
    """The sheet: a summary, the geomeans by family / dtype / size, then one table per family."""
    cells = {}   # (op, kind, n) -> (ns_ms, np_ms, ratio)
    for key, ns_ms, np_ms, ratio in bc.ratio_rows(ns, npy):
        parts = key.split("|")
        if len(parts) != 3:
            continue
        op, kind, n = parts[0], parts[1], int(parts[2])
        cells[(op, kind, n)] = (ns_ms, np_ms, ratio)

    L = ["# Strings — text computation over S / U / StringDType (NumSharp vs NumPy 2.4.2)", ""]
    L.append("Every cell times one call on the SAME data on both sides (the NumPy twin writes it, NumSharp loads it), "
             "best of rounds after a warm-up. ratio = NumPy_ms / NumSharp_ms — **>1.0 = NumSharp faster**. "
             "✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. A call one side refuses for a kind (NumPy has no such loop) has no cell.")
    L.append("")
    if not cells:
        L.append("_no comparable cells (a side produced no output — build / run failed)._")
        return "\n".join(L), cells

    ratios = [v[2] for v in cells.values()]
    worst = min(cells.items(), key=lambda kv: kv[1][2])
    L.append("## Summary")
    L.append("")
    L.append(f"- **{len(cells)}** cells, geomean **{bc.geomean(ratios):.2f}×**, min **{worst[1][2]:.2f}×** "
             f"(`{worst[0][0]}` on {KIND_LABEL.get(worst[0][1], worst[0][1])} at {SIZE_LABEL[worst[0][2]]}).")
    L.append(f"- Under 1.0×: **{sum(r < 1.0 for r in ratios)}**; under 1.5×: **{sum(r < 1.5 for r in ratios)}**.")
    by_dtype = defaultdict(list)
    by_size = defaultdict(list)
    for (op, kind, n), v in cells.items():
        by_dtype[dtype_of(kind)].append(v[2])
        by_size[n].append(v[2])
    L.append("- By dtype: " + ", ".join(f"{d} {bc.geomean(by_dtype[d]):.2f}× ({len(by_dtype[d])})"
                                        for d in ("S", "U", "T") if by_dtype[d]) + ".")
    L.append("- By size: " + ", ".join(f"{SIZE_LABEL[n]} {bc.geomean(by_size[n]):.2f}× ({len(by_size[n])})"
                                       for n in SIZES if by_size[n]) + ".")
    L.append("")
    L.append("| Family | Cells | Geomean | Min | Under 1.5× |")
    L.append("|---|---:|---:|---:|---:|")
    for title, ops in FAMILIES:
        fam = [v[2] for (op, _, _), v in cells.items() if op in ops]
        if fam:
            L.append(f"| {title} | {len(fam)} | {bc.geomean(fam):.2f}× | {min(fam):.2f}× | {sum(r < 1.5 for r in fam)} |")
    L.append("")

    for title, ops in FAMILIES:
        rows = sorted({(op, kind) for (op, kind, _) in cells if op in ops},
                      key=lambda ok: (ops.index(ok[0]), list(KIND_LABEL).index(ok[1]) if ok[1] in KIND_LABEL else 99))
        if not rows:
            continue
        L.append(f"## {title}")
        L.append("")
        L.append("| Call | Data | " + " | ".join(SIZE_LABEL[n] for n in SIZES) + " |")
        L.append("|---|---|" + "---:|" * len(SIZES))
        for op, kind in rows:
            vals = []
            for n in SIZES:
                v = cells.get((op, kind, n))
                vals.append(f"{bc.icon(v[2])} {v[2]:.2f}× ({v[0]:.3g} ms)" if v else "")
            L.append(f"| `{OP_LABEL.get(op, op)}` | {KIND_LABEL.get(kind, kind)} | " + " | ".join(vals) + " |")
        L.append("")
    return "\n".join(L), cells


def main():
    ap = argparse.ArgumentParser(description="Strings subsystem (text computation over S / U / StringDType vs NumPy)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    args = ap.parse_args()

    if not args.skip_build:
        bc.build_core(REPO)

    data_dir = tempfile.mkdtemp(prefix="numsharp-strings-bench-")
    try:
        env = {"NUMSHARP_STRINGS_BENCH_DATA": data_dir}
        bc.log("[strings] strings_bench.py (NumPy; writes the shared data) …")
        py_out = bc.run_py(REPO, os.path.join(HERE, "strings_bench.py"), timeout=3600, env_extra=env)
        bc.log("[strings] strings_bench.cs (NumSharp) …")
        # Tier-up without the 100 ms quiet window, as the other text-heavy sheets run (a cell's first rows otherwise
        # measure tier-0 code even after the warm-up).
        cs_out = bc.run_cs(REPO, os.path.join(HERE, "strings_bench.cs"), timeout=3600,
                           env_extra={**env, "DOTNET_TC_CallCountingDelayMs": "0"})
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)

    ns, npy = bc.parse_tsv(cs_out), bc.parse_tsv(py_out)
    md, cells = render(ns, npy)
    with open(MD, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    with open(TSV, "w", encoding="utf-8") as f:
        f.write("op\tkind\tn\tnumsharp_ms\tnumpy_ms\tratio\n")
        for (op, kind, n), (ns_ms, np_ms, ratio) in sorted(cells.items()):
            f.write(f"{op}\t{kind}\t{n}\t{ns_ms:.6g}\t{np_ms:.6g}\t{ratio:.4f}\n")
    bc.log(f"[strings] -> {os.path.relpath(MD, REPO)}")
    print(md)


if __name__ == "__main__":
    main()
