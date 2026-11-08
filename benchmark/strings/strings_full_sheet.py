#!/usr/bin/env python3
# =============================================================================
# strings_full_sheet.py - the FULL strings / chararray evaluation: orchestrator + joiner + renderer.
#
# The official strings subsystem (strings_sheet.py, 442 cells) times the text COMPUTATIONS. This sheet times every text
# operation over S / U / StringDType (T, with and without a NaN NA) against NumPy 2.4.2 on the same arrays, ~2,900 keys
# in twelve groups: cast (astype between every text class and the numbers - what NumPy's casting impls run through
# get_strided_loop), copy (every NDStringCopy route and the scalar / broadcast / mask stores), meta (the dtype layer:
# resolve_descriptors behind np.dtype, can_cast, promote_types, result_type), ufunc, reduce, sort, strings (all 46
# np.strings names), char (np.char), carr (chararray members and operators), index, create (creation, conversion,
# printing) and io (.npy, text). It is NOT part of run_benchmark.py: ~20 minutes per side.
#
# Flow: build NumSharp.Core (Release) -> strings_full_bench.py writes the data (np.save; StringDType as NumPy's pickle)
# into a temporary directory -> the NumPy twin times every case -> strings_full_bench.cs (dotnet run -c Release -) loads
# the same files and times the same keys -> join on the key, check the result signatures, render
# strings_full_results.md (+ .tsv). Both languages are pinned to TWO P-cores (NUMSHARP_BENCHMARK_AFFINITY=0x14 unless
# set: one pinned CPU starves .NET's tier-1 compiler), through bench_common's runner controls.
#
#   python benchmark/strings/strings_full_sheet.py [--skip-build] [--filter PREFIXES] [--sizes 1000,100000]
#   python benchmark/strings/strings_full_sheet.py --render-only <numpy.tsv> <numsharp.tsv>
# =============================================================================
import argparse
import datetime
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "benchmark", "scripts"))
import bench_common as bc  # noqa: E402

MD = os.path.join(HERE, "strings_full_results.md")
TSV = os.path.join(HERE, "strings_full_results.tsv")

GROUP_TITLE = {
    "cast": "Casts (astype) - the casting impls' families: S/U/T among themselves, text -> numbers, numbers -> text",
    "copy": "Copies and assignment - the NDStringCopy routes (NDVStringCopy for T) and the scalar / broadcast / mask stores",
    "meta": "dtype layer - resolve_descriptors behind np.dtype / can_cast / promote_types / result_type (O(1) calls)",
    "ufunc": "Ufuncs - comparisons, add, multiply, maximum / minimum",
    "reduce": "Reductions and truth",
    "sort": "Sorting, searching, sets, quantile",
    "strings": "np.strings",
    "char": "np.char (the chararray module's functions)",
    "carr": "chararray members and operators",
    "index": "Indexing and manipulation",
    "create": "Creation, conversion, printing",
    "io": "I/O (.npy, text)",
}
SIZE_LABEL = {0: "-", 1000: "1K", 100000: "100K", 1000000: "1M"}
NUM = re.compile(r"^-?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$|^-?inf$|^nan$", re.I)


def parse(text):
    """Parses a twin's output: key -> ("ms", ms, sig) | ("REFUSED" | "CRASH" | "SKIP", text)."""
    out = {}
    for line in text.splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or "|" not in parts[0]:
            continue
        if parts[1] in ("REFUSED", "CRASH", "SKIP"):
            out[parts[0]] = (parts[1], parts[2] if len(parts) > 2 else "")
        else:
            try:
                out[parts[0]] = ("ms", float(parts[1]), parts[2] if len(parts) > 2 else "")
            except ValueError:
                pass
    return out


def same_sig(a, b):
    """Two result signatures agree: identical heads (dtype + shape), content numbers equal within 1e-4 relative."""
    if a == b:
        return True
    pa, pb = a.split(";"), b.split(";")
    if len(pa) != len(pb):
        return False
    for x, y in zip(pa, pb):
        if x == y:
            continue
        hx, _, vx = x.rpartition(":")
        hy, _, vy = y.rpartition(":")
        if hx != hy or len(vx.split("/")) != len(vy.split("/")):
            return False
        for u, v in zip(vx.split("/"), vy.split("/")):
            if u == v:
                continue
            if NUM.match(u) and NUM.match(v):
                fu, fv = float(u), float(v)
                if (math.isnan(fu) and math.isnan(fv)) or fu == fv or abs(fu - fv) <= 1e-4 * max(abs(fu), abs(fv)):
                    continue
            return False
    return True


def split_key(k):
    """A key's parts: (group, op, kind, n)."""
    op, kind, n = k.split("|")
    return op.split(".", 1)[0], op.split(".", 1)[1], kind, int(n)


def dtype_of(kind):
    """The text class a data kind holds: S, U or T, 'num' for a number array cast to text, '-' for the dtype layer."""
    k = kind.split("@")[0]
    if k.startswith("n"):
        return "num"
    return {"s": "S", "u": "U", "t": "T", "-": "-"}.get(k[0], "?")


def join(npy, ns):
    """Joins the two outputs: timed cells, signature mismatches, refusal disagreements, NumPy-only failures."""
    cells, sig_bad, np_only, ns_only, numpy_fail = {}, [], [], [], []
    for k in sorted(set(npy) | set(ns)):
        a, b = npy.get(k), ns.get(k)
        if a is None or b is None:
            continue
        if a[0] in ("CRASH", "SKIP"):
            numpy_fail.append((k, a, b))
            continue
        if a[0] == "REFUSED" and b[0] == "REFUSED":
            continue
        if a[0] == "REFUSED":
            np_only.append((k, a[1]))
            continue
        if b[0] == "REFUSED":
            ns_only.append((k, b[1]))
            continue
        ratio = a[1] / b[1] if b[1] > 0 else float("inf")
        ok = same_sig(a[2], b[2])
        cells[k] = (a[1], b[1], ratio, ok)
        if not ok:
            sig_bad.append((k, a[2], b[2]))
    return cells, sig_bad, np_only, ns_only, numpy_fail


def render(cells, sig_bad, np_only, ns_only, numpy_fail, provenance):
    """The sheet: provenance, summary, group table, the cells under 1.5x, one table per group, then the NumPy failures,
    refusal disagreements and signature mismatches."""
    L = ["# Strings and chararray - full evaluation (NumSharp vs NumPy 2.4.2)", ""]
    L.append(provenance)
    L.append("")
    L.append("Every cell times one call on the SAME data on both sides (the NumPy twin writes it, NumSharp loads it), "
             "best of rounds after a warm-up. ratio = NumPy_ms / NumSharp_ms - **>1.0 = NumSharp faster**. "
             "✅≥1.0 🟡≥0.5 🟠≥0.2 🔴<0.2. ⚠sig = the two results' signatures differ (see the last section).")
    L.append("")
    credible = {k: v for k, v in cells.items() if not k.startswith("meta.")}
    cr = [v[2] for v in credible.values()]
    L.append("## Summary")
    L.append("")
    L.append(f"- **{len(cells)}** timed cells; geomean **{bc.geomean([v[2] for v in cells.values()]):.2f}×** "
             f"({bc.geomean(cr):.2f}× over the {len(cr)} cells outside the O(1) dtype layer).")
    L.append(f"- Under 1.0×: **{sum(r < 1.0 for r in cr)}**; under 1.5×: **{sum(r < 1.5 for r in cr)}**; "
             f"result-signature mismatches: **{len(sig_bad)}**; NumPy crashes / stalls: **{len(numpy_fail)}**.")
    by = defaultdict(list)
    for k, v in credible.items():
        g, op, kind, n = split_key(k)
        by[("dtype", dtype_of(kind))].append(v[2])
        by[("size", n)].append(v[2])
    L.append("- By dtype: " + ", ".join(f"{d} {bc.geomean(by[('dtype', d)]):.2f}× ({len(by[('dtype', d)])})"
                                        for d in ("S", "U", "T", "num") if by[("dtype", d)])
             + " (num = a number array cast to text).")
    L.append("- By size: " + ", ".join(f"{SIZE_LABEL[n]} {bc.geomean(by[('size', n)]):.2f}× ({len(by[('size', n)])})"
                                       for n in (1000, 100000, 1000000) if by[("size", n)]) + ".")
    L.append("")
    L.append("| Group | Cells | Geomean | Min | <1.0× | <1.5× |")
    L.append("|---|---:|---:|---:|---:|---:|")
    groups = defaultdict(list)
    for k, v in cells.items():
        groups[split_key(k)[0]].append(v[2])
    for g in GROUP_TITLE:
        xs = groups[g]
        if xs:
            L.append(f"| {g} | {len(xs)} | {bc.geomean(xs):.2f}× | {min(xs):.2f}× | {sum(x < 1 for x in xs)} | "
                     f"{sum(x < 1.5 for x in xs)} |")
    L.append("")
    L.append("## The cells under 1.5×")
    L.append("")
    L.append("| Case | Kind | Size | Ratio | NumSharp ms | NumPy ms |")
    L.append("|---|---|---|---:|---:|---:|")
    for k, v in sorted(credible.items(), key=lambda kv: kv[1][2]):
        if v[2] >= 1.5:
            break
        g, op, kind, n = split_key(k)
        L.append(f"| `{g}.{op}` | {kind} | {SIZE_LABEL.get(n, n)} | {bc.icon(v[2])} {v[2]:.2f}× | {v[1]:.4g} | {v[0]:.4g} |")
    L.append("")
    for g, title in GROUP_TITLE.items():
        rows = defaultdict(dict)
        for k, v in cells.items():
            gg, op, kind, n = split_key(k)
            if gg == g:
                rows[(op, kind)][n] = v
        if not rows:
            continue
        L.append(f"## {title}")
        L.append("")
        sizes = sorted({n for r in rows.values() for n in r})
        L.append("| Op | Kind | " + " | ".join(SIZE_LABEL.get(n, str(n)) for n in sizes) + " |")
        L.append("|---|---|" + "---:|" * len(sizes))
        for (op, kind) in sorted(rows):
            vals = []
            for n in sizes:
                v = rows[(op, kind)].get(n)
                vals.append((f"{bc.icon(v[2])} {v[2]:.2f}× ({v[1]:.3g} ms)" + ("" if v[3] else " ⚠sig")) if v else "")
            L.append(f"| `{op}` | {kind} | " + " | ".join(vals) + " |")
        L.append("")
    if numpy_fail:
        L.append("## NumPy 2.4.2 crashes and stalls - NumSharp-only rows")
        L.append("")
        for k, a, b in numpy_fail:
            what = "segfaults" if a[0] == "CRASH" else "does not finish in reasonable time (quadratic)"
            ns_txt = f"NumSharp {b[1]:.4g} ms" if b[0] == "ms" else f"NumSharp {b[0]}"
            L.append(f"- `{k}`: NumPy {what}; {ns_txt}")
        L.append("")
    if np_only or ns_only:
        L.append("## Refusal disagreements")
        L.append("")
        for k, m in np_only:
            L.append(f"- NumPy refuses, NumSharp computes: `{k}` - NumPy: {m[:140]}")
        for k, m in ns_only:
            L.append(f"- NumSharp refuses, NumPy computes: `{k}` - NumSharp: {m[:140]}")
        L.append("")
    if sig_bad:
        L.append("## Result-signature mismatches")
        L.append("")
        L.append("A mismatch is a different answer, not a timing problem; each is explained in strings_full_evaluation.md.")
        L.append("")
        for k, x, y in sig_bad:
            L.append(f"- `{k}`: NumPy `{x}` vs NumSharp `{y}`")
        L.append("")
    return "\n".join(L) + "\n"


def write_outputs(npy_text, ns_text, provenance):
    """Joins the two twins' outputs and writes strings_full_results.md and strings_full_results.tsv."""
    cells, sig_bad, np_only, ns_only, numpy_fail = join(parse(npy_text), parse(ns_text))
    md = render(cells, sig_bad, np_only, ns_only, numpy_fail, provenance)
    with open(MD, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    with open(TSV, "w", encoding="utf-8", newline="\n") as f:
        f.write("key\tnumpy_ms\tnumsharp_ms\tratio\tsig_ok\n")
        for k, v in sorted(cells.items()):
            f.write(f"{k}\t{v[0]:.6g}\t{v[1]:.6g}\t{v[2]:.4f}\t{int(v[3])}\n")
    bc.log(f"[strings-full] -> {os.path.relpath(MD, REPO)} ({len(cells)} cells)")


def provenance_line():
    """The sheet's provenance line: today's date, the checkout's HEAD, NumPy's version and the host pin."""
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    npv = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], capture_output=True,
                         text=True).stdout.strip()
    return (f"_Measured {datetime.date.today().isoformat()} at `{head}` against NumPy {npv}; both sides pinned to "
            f"{os.environ.get('NUMSHARP_BENCHMARK_AFFINITY', 'the default core')}._")


def main():
    """Runs the sheet (build, NumPy twin write + time, C# twin, join, render), or only renders saved outputs."""
    ap = argparse.ArgumentParser(description="Full strings / chararray evaluation (every text op vs NumPy)")
    ap.add_argument("--skip-build", action="store_true", help="reuse the existing Release build")
    ap.add_argument("--filter", default="-", help="comma-separated key prefixes ('$' suffix = one exact key)")
    ap.add_argument("--sizes", default="1000,100000", help="comma list of sizes")
    ap.add_argument("--render-only", nargs=2, metavar=("NUMPY_TSV", "NUMSHARP_TSV"),
                    help="render strings_full_results.md from two existing twin outputs")
    ap.add_argument("--provenance", help="the provenance line for --render-only")
    args = ap.parse_args()

    if args.render_only:
        with open(args.render_only[0], encoding="utf-8") as f:
            npy_text = f.read()
        with open(args.render_only[1], encoding="utf-8") as f:
            ns_text = f.read()
        write_outputs(npy_text, ns_text, args.provenance or provenance_line())
        return

    os.environ.setdefault("NUMSHARP_BENCHMARK_AFFINITY", "0x14")
    if not args.skip_build:
        bc.build_core(REPO)
    data_dir = tempfile.mkdtemp(prefix="numsharp-strings-full-")
    try:
        env = {"NUMSHARP_STRINGS_FULL_DATA": data_dir, "NUMSHARP_STRINGS_FULL_FILTER": args.filter,
               "NUMSHARP_STRINGS_FULL_SIZES": args.sizes}
        bench_py = os.path.join(HERE, "strings_full_bench.py")
        bc.log("[strings-full] strings_full_bench.py write (NumPy; the shared data) ...")
        bc.run_py(REPO, bench_py, timeout=900, env_extra={**env, "NUMSHARP_STRINGS_FULL_MODE": "write"})
        bc.log("[strings-full] strings_full_bench.py time (NumPy) ...")
        py_out = bc.run_py(REPO, bench_py, timeout=7200, env_extra={**env, "NUMSHARP_STRINGS_FULL_MODE": "time"})
        bc.log("[strings-full] strings_full_bench.cs (NumSharp) ...")
        cs_out = bc.run_cs(REPO, os.path.join(HERE, "strings_full_bench.cs"), timeout=7200,
                           env_extra={**env, "DOTNET_TC_CallCountingDelayMs": "0"})
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    write_outputs(py_out, cs_out, provenance_line())


if __name__ == "__main__":
    main()
