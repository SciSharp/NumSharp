#!/usr/bin/env python3
"""
Official NumSharp-vs-NumPy benchmark orchestrator (cross-platform).

Runs the C# BenchmarkDotNet suite and the NumPy suite across the three cache-tier sizes
(Small=1K / Medium=100K / Large=10M), then merges them into a single per-(op, dtype, N)
ratio report. Finally runs the NpyIter iterator benchmark (benchmark/npyiter) and appends
its sheet to the report as a dedicated section (different result model — aspect x tier —
so it is appended, not merged). The single entry point for the whole NumSharp-vs-NumPy
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
  python run_benchmark.py --skip-npyiter          # official op suites only (no NpyIter)
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
CSHARP_DIR = HERE / "NumSharp.Benchmark.GraphEngine"
CSHARP_PROJ = CSHARP_DIR / "NumSharp.Benchmark.GraphEngine.csproj"
PY_BENCH = HERE / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
MERGE = HERE / "scripts" / "merge-results.py"
ARTIFACTS = CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
TFM = "net10.0"

# NpyIter iterator benchmark (benchmark/npyiter) — a complementary harness with a
# different result model (aspect x tier, not op/dtype/N), appended to the report.
NPYITER_DIR = HERE / "npyiter"
NPYITER_SHEET = NPYITER_DIR / "npyiter_sheet.py"
NPYITER_CARDS = NPYITER_DIR / "npyiter_cards.py"
NPYITER_REPORT = NPYITER_DIR / "npyiter_results.md"
NPYITER_TSV = NPYITER_DIR / "npyiter_results.tsv"

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


def main():
    ap = argparse.ArgumentParser(description="NumSharp vs NumPy official benchmark")
    ap.add_argument("--suites", nargs="*", default=list(SUITES), choices=list(SUITES),
                    help="Subset of comparison suites to run (default: all)")
    ap.add_argument("--skip-csharp", action="store_true", help="Skip the C# benchmarks")
    ap.add_argument("--skip-python", action="store_true", help="Skip the NumPy benchmarks")
    ap.add_argument("--skip-build", action="store_true", help="Reuse the existing Release build")
    ap.add_argument("--quick", action="store_true", help="Dev: fewer NumPy iterations")
    ap.add_argument("--skip-npyiter", action="store_true",
                    help="Skip the NpyIter iterator benchmark (benchmark/npyiter)")
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

    # 4b. NpyIter iterator benchmark — complementary harness (file-based, section-
    #     isolated, crash-resilient: a NumSharp AccessViolation is IGNORED and the
    #     section reported NA). Its result model is aspect x tier, not op/dtype/N,
    #     so it is APPENDED to the report as its own section rather than merged —
    #     preserving the iterator-isolation value the op matrix cannot express.
    if not args.skip_npyiter:
        print("\n=== NpyIter iterator benchmark (benchmark/npyiter) ===", flush=True)
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
            shutil.copytree(cards_src, results_dir / "npyiter_cards", dirs_exist_ok=True)
        report_md = results_dir / "benchmark-report.md"
        if NPYITER_REPORT.exists():
            section = ("\n\n---\n\n## NpyIter iterator benchmark\n\n"
                       "_Complementary harness: measures the iterator machinery itself "
                       "(construction, traversal, reductions, selection, dtypes, pathologies, "
                       "dividends) across cache tiers — not part of the op/dtype/N matrix above. "
                       "speedup = NumPy / NumSharp; NA = section ignored due to a known "
                       "intermittent NumSharp AccessViolation._\n\n")
            existing = report_md.read_text(encoding="utf-8") if report_md.exists() else ""
            report_md.write_text(existing + section + NPYITER_REPORT.read_text(encoding="utf-8"),
                                 encoding="utf-8")

    # 5. Copy the headline artifacts to the benchmark/ root for convenience.
    for name in ["benchmark-report.md", "benchmark-report.json", "benchmark-report.csv",
                 "numpy-results.json"]:
        src = results_dir / name
        if src.exists():
            shutil.copy(src, HERE / name)

    print(f"\nDone in {time.time() - t0:.0f}s. Report: {HERE / 'benchmark-report.md'}")
    print(f"Archive: {results_dir}")


if __name__ == "__main__":
    main()
