#!/usr/bin/env python3
"""
Official NumSharp-vs-NumPy benchmark orchestrator (cross-platform).

Runs the C# BenchmarkDotNet suite and the NumPy suite across the three cache-tier sizes
(Small=1K / Medium=100K / Large=10M), then merges them into a single per-(op, dtype, N)
ratio report. Then appends, as dedicated sections, the complementary harnesses whose result
models the op/dtype/N matrix cannot express:
  * NDIter iterator benchmark (benchmark/nditer)  — iterator machinery, aspect × tier
  * Layout suite              (benchmark/layout)    — reduction/copy/elementwise × memory layout
  * Operand layouts           (benchmark/operand)   — 1-D / scalar / mixed-operand / broadcast
  * Cast matrix               (benchmark/cast)      — astype src→dst × layout × dtype
  * Fusion gate               (benchmark/fusion)    — np.evaluate fused vs unfused chains
Each owns a *_sheet.py driver+renderer; this orchestrator runs them and folds their
*_results.md into the report. The single entry point for the whole NumSharp-vs-NumPy
comparison.

Design notes
------------
* C# side uses ``OfficialBenchmarkConfig`` (see Infrastructure/BenchmarkConfig.cs):
  the InProcessEmit toolchain (so BenchmarkDotNet does not search the repo tree for the
  project — sibling ``.claude/worktrees/`` checkouts contain same-named copies and the
  out-of-process toolchain refuses to build with "project names need to be unique"), and
  an iteration-time-capped 50-iteration job (so µs–ms array ops don't get BDN's
  nanosecond-microbenchmark invocation ramp, which would make the full run take days).
  Because the config is baked into the assembly, this orchestrator passes only ``--filter``
  to ``dotnet run`` — never ``--job``.
* Per-suite C# runs are independent: each benchmark class exports its own JSON, so a crash
  mid-run keeps every completed class. Re-running a single suite is cheap.
* NumPy side sweeps all three sizes in one invocation per suite (``--cache-sizes``); each
  result carries its own ``n``, which the merge keys on.

Usage
-----
  python run_benchmark.py                         # full official run, all comparison suites
  python run_benchmark.py --suites arithmetic unary
  python run_benchmark.py --skip-build            # reuse the existing Release build
  python run_benchmark.py --skip-csharp           # NumPy only
  python run_benchmark.py --skip-python           # C# only (reuse existing numpy JSON)
  python run_benchmark.py --skip-nditer          # no NDIter section
  python run_benchmark.py --skip-layout --skip-cast --skip-fusion   # op matrix (+NDIter) only
  python run_benchmark.py --quick                 # dev: 10 NumPy iterations (C# config fixed)
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORY_DIR = HERE / "history"
CSHARP_DIR = HERE / "NumSharp.Benchmark.CSharp"
CSHARP_PROJ = CSHARP_DIR / "NumSharp.Benchmark.CSharp.csproj"
PY_BENCH = HERE / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
MERGE = HERE / "scripts" / "merge-results.py"
ARTIFACTS = CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
TFM = "net10.0"

# NDIter iterator benchmark (benchmark/nditer) — a complementary harness with a
# different result model (aspect x tier, not op/dtype/N), appended to the report.
NPYITER_DIR = HERE / "nditer"
NPYITER_SHEET = NPYITER_DIR / "nditer_sheet.py"
NPYITER_CARDS = NPYITER_DIR / "nditer_cards.py"
NPYITER_REPORT = NPYITER_DIR / "nditer_results.md"
NPYITER_TSV = NPYITER_DIR / "nditer_results.tsv"

# Matrix subsystems — each fills an axis the op/dtype/N matrix omits and owns a
# *_sheet.py driver+renderer (mirroring nditer): a NumSharp `*_bench.cs` + its
# NumPy `*_bench.py` twin -> a rendered `*_results.md` section appended to the
# report. Layout = memory-layout axis (op-matrix is C-contiguous only); Cast =
# astype src→dst matrix (no op-matrix coverage); Fusion = np.evaluate vs unfused.
MATRIX_SUBSYSTEMS = [
    ("layout", HERE / "layout" / "layout_sheet.py", HERE / "layout" / "layout_results.md",
     "Layout suite — reduction / copy / elementwise × memory layout × dtype"),
    ("operand", HERE / "operand" / "operand_sheet.py", HERE / "operand" / "operand_results.md",
     "Operand & broadcast layouts — 1-D / scalar / mixed-operand / broadcast"),
    ("cast", HERE / "cast" / "cast_sheet.py", HERE / "cast" / "cast_results.md",
     "Cast matrix — astype src→dst × layout × dtype"),
    ("fusion", HERE / "fusion" / "fusion_sheet.py", HERE / "fusion" / "fusion_results.md",
     "Fusion — np.evaluate vs unfused chains"),
]

# Comparison suites only (the experimental Dispatch/Fusion/DynamicEmission/SimdVsScalar
# benchmarks have no NumPy counterpart). suite -> BenchmarkDotNet class/namespace filter.
SUITES = {
    "arithmetic":   "*Benchmarks.Arithmetic.*",
    # Unary namespace also covers UnaryExtraBenchmarks (cbrt/reciprocal/square/negative/positive/trunc).
    "unary":        "*Benchmarks.Unary.*",
    # Reduction namespace also covers NanReductionBenchmarks and CumulativeBenchmarks.
    "reduction":    "*Benchmarks.Reduction.*",
    "broadcast":    "*Benchmarks.Broadcasting.*",
    "creation":     "*Benchmarks.Creation.*",
    "manipulation": "*Benchmarks.Manipulation.*",
    "slicing":      "*Benchmarks.Slicing.*",
    "comparison":   "*Benchmarks.Comparison.*",
    "bitwise":      "*Benchmarks.Bitwise.*",
    "logic":        "*Benchmarks.Logic.*",
    "statistics":   "*Benchmarks.Statistics.*",
    "sorting":      "*Benchmarks.Sorting.*",
    "linalg":       "*Benchmarks.LinearAlgebra.*",
    "selection":    "*Benchmarks.Selection.*",
}


def run(cmd, cwd=None, check=False):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=check)


def append_section(report_md, src_md, title):
    """Append a subsystem's rendered *_results.md to the unified report as one
    section. The source's leading H1 (if any) is dropped so the report keeps a
    single title hierarchy; ``title`` becomes the section's H2. No-op if the
    source is missing/empty (subsystem skipped or its build/run failed)."""
    if not src_md.exists():
        return
    body = src_md.read_text(encoding="utf-8").strip()
    lines = body.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    body = "\n".join(lines).strip()
    if not body:
        return
    existing = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
    report_md.write_text(f"{existing}\n\n---\n\n## {title}\n\n{body}\n", encoding="utf-8")


def run_matrix_subsystem(name, sheet, results_md, title, report_md, results_dir, skip_build):
    """Run one matrix subsystem's *_sheet.py and append its rendered section to
    the unified report. Crash-resilient: a failing subsystem just omits its
    section (the sheet itself never raises into the orchestrator)."""
    print(f"\n=== {name} subsystem (benchmark/{name}) ===", flush=True)
    cmd = [sys.executable, str(sheet)]
    if skip_build:
        cmd.append("--skip-build")
    run(cmd, check=False)
    if results_md.exists():
        shutil.copy(results_md, results_dir / results_md.name)
        tsv = results_md.with_suffix(".tsv")
        if tsv.exists():
            shutil.copy(tsv, results_dir / tsv.name)
    append_section(report_md, results_md, title)


def main():
    ap = argparse.ArgumentParser(description="NumSharp vs NumPy official benchmark")
    ap.add_argument("--suites", nargs="*", default=list(SUITES), choices=list(SUITES),
                    help="Subset of comparison suites to run (default: all)")
    ap.add_argument("--skip-csharp", action="store_true", help="Skip the C# benchmarks")
    ap.add_argument("--skip-python", action="store_true", help="Skip the NumPy benchmarks")
    ap.add_argument("--skip-build", action="store_true", help="Reuse the existing Release build")
    ap.add_argument("--quick", action="store_true", help="Dev: fewer NumPy iterations")
    ap.add_argument("--skip-nditer", action="store_true",
                    help="Skip the NDIter iterator benchmark (benchmark/nditer)")
    ap.add_argument("--skip-layout", action="store_true",
                    help="Skip the Layout suite (benchmark/layout)")
    ap.add_argument("--skip-cast", action="store_true",
                    help="Skip the Cast matrix (benchmark/cast)")
    ap.add_argument("--skip-fusion", action="store_true",
                    help="Skip the Fusion gate (benchmark/fusion)")
    ap.add_argument("--skip-operand", action="store_true",
                    help="Skip the Operand-layout subsystem (benchmark/operand)")
    ap.add_argument("--no-history", action="store_true",
                    help="Skip writing the committable benchmark/history/<date>_<sha>/ snapshot + latest symlink")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = HERE / "results" / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    csharp_out = results_dir / "csharp"
    csharp_out.mkdir(exist_ok=True)
    numpy_json = results_dir / "numpy-results.json"
    print(f"Results -> {results_dir}")

    t0 = time.time()

    # 1. Build the C# benchmark project (Release).
    if not args.skip_csharp and not args.skip_build:
        run(["dotnet", "build", "-c", "Release", "-f", TFM, str(CSHARP_PROJ),
             "-v", "q", "--nologo", "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0"], check=True)

    # 2. NumPy: sweep all three sizes per suite, concatenate into one JSON.
    if not args.skip_python:
        merged = []
        for s in args.suites:
            tmp = results_dir / f"numpy-{s}.json"
            cmd = [sys.executable, str(PY_BENCH), "--suite", s, "--cache-sizes", "--output", str(tmp)]
            if args.quick:
                cmd.append("--quick")
            run(cmd, check=True)
            if tmp.exists():
                merged.extend(json.loads(tmp.read_text()))
        numpy_json.write_text(json.dumps(merged, indent=2))
        print(f"NumPy: {len(merged)} results across {len(args.suites)} suites")

    # 3. C# BenchmarkDotNet per suite (config provides the job + JSON exporter). BDN cleans
    #    its artifacts dir on each run, so copy out each suite's class reports immediately
    #    after that suite finishes — otherwise only the last suite would survive.
    if not args.skip_csharp:
        for s in args.suites:
            if ARTIFACTS.exists():
                shutil.rmtree(ARTIFACTS, ignore_errors=True)
            print(f"\n=== C# suite: {s} ({SUITES[s]}) ===", flush=True)
            run(["dotnet", "run", "-c", "Release", "--no-build", "-f", TFM,
                 "--project", str(CSHARP_PROJ), "--", "--filter", SUITES[s]],
                cwd=CSHARP_DIR, check=False)
            if ARTIFACTS.exists():
                for f in ARTIFACTS.glob("*-report-full-compressed.json"):
                    shutil.copy(f, csharp_out / f.name)
        print(f"C#: collected {len(list(csharp_out.glob('*.json')))} class reports")

    # 4. Merge into the unified per-(op, dtype, N) ratio report.
    out_base = results_dir / "benchmark-report"
    run([sys.executable, str(MERGE), "--numpy", str(numpy_json),
         "--csharp", str(csharp_out), "--output", str(out_base)], check=False)

    # The unified report the op-matrix merge just wrote; the iterator + matrix
    # subsystems below each append one section to it.
    report_md = results_dir / "benchmark-report.md"

    # 4b. NDIter iterator benchmark — complementary harness (file-based, section-
    #     isolated, crash-resilient: a NumSharp AccessViolation is IGNORED and the
    #     section reported NA). Its result model is aspect x tier, not op/dtype/N,
    #     so it is APPENDED to the report as its own section rather than merged —
    #     preserving the iterator-isolation value the op matrix cannot express.
    if not args.skip_nditer:
        print("\n=== NDIter iterator benchmark (benchmark/nditer) ===", flush=True)
        sheet_cmd = [sys.executable, str(NPYITER_SHEET)]
        if args.skip_build:
            sheet_cmd.append("--skip-build")
        run(sheet_cmd, check=False)
        run([sys.executable, str(NPYITER_CARDS)], check=False)
        for src in (NPYITER_REPORT, NPYITER_TSV):
            if src.exists():
                shutil.copy(src, results_dir / src.name)
        cards_src = NPYITER_DIR / "cards"
        if cards_src.exists():
            shutil.copytree(cards_src, results_dir / "nditer_cards", dirs_exist_ok=True)
        if NPYITER_REPORT.exists():
            section = ("\n\n---\n\n## NDIter iterator benchmark\n\n"
                       "_Complementary harness: measures the iterator machinery itself "
                       "(construction, traversal, reductions, selection, dtypes, pathologies, "
                       "dividends) across cache tiers — not part of the op/dtype/N matrix above. "
                       "speedup = NumPy / NumSharp; NA = section ignored due to a known "
                       "intermittent NumSharp AccessViolation._\n\n")
            existing = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
            report_md.write_text(existing + section + NPYITER_REPORT.read_text(encoding="utf-8"),
                                 encoding="utf-8")

    # 4c. Matrix subsystems — layout / cast / fusion. Each fills an axis the
    #     op/dtype/N matrix cannot express and appends its own rendered section.
    skip_matrix = {"layout": args.skip_layout, "operand": args.skip_operand,
                   "cast": args.skip_cast, "fusion": args.skip_fusion}
    for name, sheet, results, title in MATRIX_SUBSYSTEMS:
        if skip_matrix[name]:
            continue
        run_matrix_subsystem(name, sheet, results, title, report_md, results_dir, args.skip_build)

    # 5. Copy the headline artifacts to the benchmark/ root for convenience.
    for name in ["benchmark-report.md", "benchmark-report.json", "benchmark-report.csv",
                 "numpy-results.json"]:
        src = results_dir / name
        if src.exists():
            shutil.copy(src, HERE / name)

    # 6. History snapshot + latest symlink — the committable provenance/publish step
    #    (benchmark/scripts/snapshot_history.py): copies the report + all five subsystem
    #    sheets + cards into benchmark/history/<date>_<sha>/, writes a MANIFEST, and
    #    repoints benchmark/history/latest at it (a git-tracked symlink). results/<ts>/
    #    stays the gitignored raw scratch; benchmark/history/ is what we commit + reference.
    #    --no-stage: writing the snapshot must NOT mutate the git index. Staging is the
    #    human's "review" step (run -> review -> commit), and CI stages benchmark/history/
    #    explicitly. A local perf check shouldn't silently `git add` ~16 files.
    if not args.no_history:
        print("\n=== history snapshot + latest (benchmark/history) ===", flush=True)
        run([sys.executable, str(HERE / "scripts" / "snapshot_history.py"),
             "--results-dir", str(results_dir), "--no-stage"], check=False)

    print(f"\nDone in {time.time() - t0:.0f}s. Report: {HERE / 'benchmark-report.md'}")
    print(f"Archive: {results_dir}")
    print(f"Snapshot: {HISTORY_DIR / 'latest'} -> newest benchmark/history/<date>_<sha>/")


if __name__ == "__main__":
    main()
