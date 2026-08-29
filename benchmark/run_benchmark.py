#!/usr/bin/env python3
"""
Official NumSharp-vs-NumPy benchmark orchestrator (cross-platform).

Runs the C# BenchmarkDotNet suite and the NumPy suite across each API's applicable bounded
size tiers (classic kernels use Small=1K / Medium=100K / Large=10M), then merges them into a per-(op, dtype, N)
ratio report. Then appends, as dedicated sections, the complementary harnesses whose result
models the op/dtype/N matrix cannot express:
  * NDIter iterator benchmark (benchmark/nditer)  — iterator machinery, aspect × tier
  * Layout suite              (benchmark/layout)    — reduction/copy/elementwise × memory layout
  * Operand layouts           (benchmark/operand)   — 1-D / scalar / mixed-operand / broadcast
  * Cast matrix               (benchmark/cast)      — astype src→dst × layout × dtype
  * Fusion gate               (benchmark/fusion)    — np.evaluate fused vs unfused chains
  * Backend profiles         (benchmark/backends) — Managed C# / OpenBLAS, merged per exact cell
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
  python run_benchmark.py                         # interactive depth + dtype picker
  python run_benchmark.py --depth measure         # full official run, all comparison suites
  python run_benchmark.py --suites arithmetic unary
  python run_benchmark.py --skip-build            # reuse the existing Release build
  python run_benchmark.py --skip-csharp           # NumPy only
  python run_benchmark.py --skip-python           # C# only (reuse existing numpy JSON)
  python run_benchmark.py --skip-nditer          # no NDIter section
  python run_benchmark.py --skip-layout --skip-cast --skip-fusion   # op matrix (+NDIter) only
  python run_benchmark.py --quick                 # deprecated alias for --depth light
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from benchmark_modes import ALL_DTYPES, DEPTHS, parse_dtypes  # noqa: E402

HERE = Path(__file__).resolve().parent
HISTORY_DIR = HERE / "history"
CSHARP_DIR = HERE / "NumSharp.Benchmark.CSharp"
CSHARP_PROJ = CSHARP_DIR / "NumSharp.Benchmark.CSharp.csproj"
OPENBLAS_CSHARP_DIR = HERE / "NumSharp.Benchmark.OpenBLAS"
OPENBLAS_CSHARP_PROJ = OPENBLAS_CSHARP_DIR / "NumSharp.Benchmark.OpenBLAS.csproj"
PY_BENCH = HERE / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
MERGE = HERE / "scripts" / "merge-results.py"
PROFILE_MERGE = HERE / "scripts" / "merge-backend-profiles.py"
ARTIFACTS = CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
OPENBLAS_ARTIFACTS = OPENBLAS_CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
TFM = "net10.0"
DOCS_BENCHMARK_JSON = HERE.parent / "docs" / "website-src" / "docs" / "data" / "benchmark-report.json"

# NDIter iterator benchmark (benchmark/nditer) — a complementary harness with a
# different result model (aspect x tier, not op/dtype/N), appended to the report.
NPYITER_DIR = HERE / "nditer"
NPYITER_SHEET = NPYITER_DIR / "nditer_sheet.py"
NPYITER_CARDS = NPYITER_DIR / "nditer_cards.py"
NPYITER_REPORT = NPYITER_DIR / "nditer_results.md"
NPYITER_TSV = NPYITER_DIR / "nditer_results.tsv"

# Complementary subsystems — each fills an axis or execution route the op/dtype/N matrix omits and owns a
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
    ("openblas", HERE / "backends" / "backend_profiles.py", HERE / "openblas" / "openblas_results.md",
     "Backend profiles — Managed C# and OpenBLAS"),
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
    "fft":          "*Benchmarks.Fourier.*",
    "random":       "*Benchmarks.Random.*",
    "ndarray":      "*Benchmarks.NDArrayApi.*",
    "api":          "*Benchmarks.ApiSurface.*",
}


def run(cmd, cwd=None, check=False, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=check, env=env)


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
        for profile_json in results_md.parent.glob(f"{results_md.stem}.*.json"):
            shutil.copy(profile_json, results_dir / profile_json.name)
    append_section(report_md, results_md, title)


def validate_execution_depth(results_dir: Path, depth: str) -> None:
    """Prove pass/light used the requested sample counts and no workload failed."""
    expected_bdn = DEPTHS[depth].bdn_measurements
    expected_warmups = DEPTHS[depth].bdn_warmups
    bad_bdn = []
    bdn_cells = 0
    for directory in (results_dir / "csharp", results_dir / "csharp-openblas"):
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload.get("Benchmarks", []):
                bdn_cells += 1
                stats = row.get("Statistics")
                measurements = row.get("Measurements") or []
                actual = sum(1 for item in measurements
                             if item.get("IterationMode") == "Workload"
                             and item.get("IterationStage") == "Actual")
                warmups = sum(1 for item in measurements
                              if item.get("IterationMode") == "Workload"
                              and item.get("IterationStage") == "Warmup")
                # BenchmarkDotNet may discard statistical outliers from
                # Statistics.OriginalValues. Measurements retains every iteration
                # that actually ran, which is the depth contract we need to prove.
                if stats is None or actual != expected_bdn or warmups != expected_warmups:
                    bad_bdn.append((row.get("FullName") or row.get("Method"), actual, warmups))
    if bad_bdn:
        raise RuntimeError(
            f"{depth} BDN validation failed: expected {expected_bdn} measured iteration(s) and "
            f"{expected_warmups} warmup(s) per cell; "
            f"bad={bad_bdn[:20]}")

    numpy_path = results_dir / "numpy-results.json"
    numpy_rows = json.loads(numpy_path.read_text(encoding="utf-8")) if numpy_path.exists() else []
    if depth == "pass":
        bad_numpy = [(row.get("name"), row.get("dtype"), row.get("n"), row.get("iterations"))
                     for row in numpy_rows if row.get("iterations") != 1]
        if bad_numpy:
            raise RuntimeError(f"pass NumPy validation failed: expected one call per cell; {bad_numpy[:20]}")

    report_path = results_dir / "benchmark-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {"rows": []}
    failed = [(row.get("operation"), row.get("dtype"), row.get("n"))
              for row in report.get("rows", []) if row.get("status") == "failed"]
    if failed:
        raise RuntimeError(f"{depth} benchmark execution produced {len(failed)} failed cells: {failed[:20]}")
    print(f"{depth} validation: {bdn_cells} BDN cells × {expected_bdn} measured iteration(s) "
          f"after {expected_warmups} warmup(s); "
          f"{len(numpy_rows)} NumPy cells; 0 workload failures")


def prompt_run_options(input_fn=input) -> list[str]:
    """Interactive picker used only when the orchestrator receives no CLI arguments."""
    print("NumSharp benchmark depth")
    print("  1. pass     — execute every selected BDN/NumPy case once; no warmup")
    print("  2. light    — 1/6 measurement budget; at most 3 warmups")
    print("  3. measure  — full publication-quality benchmark (current default)")
    choices = {"1": "pass", "2": "light", "3": "measure",
               "pass": "pass", "light": "light", "measure": "measure"}
    while True:
        raw = input_fn("Select depth [3]: ").strip().lower() or "3"
        if raw in choices:
            depth = choices[raw]
            break
        print("Choose 1/pass, 2/light, or 3/measure.")

    print("\nDtypes (comma-separated names or aliases; blank = all 15):")
    print("  " + ", ".join(ALL_DTYPES))
    while True:
        dtype_text = input_fn("Select dtypes [all]: ").strip()
        try:
            parse_dtypes(dtype_text)
            break
        except ValueError as error:
            print(error)
    argv = ["--depth", depth]
    if dtype_text:
        argv.extend(["--dtypes", dtype_text])
    return argv


def main(argv=None):
    ap = argparse.ArgumentParser(description="NumSharp vs NumPy official benchmark")
    ap.add_argument("--suites", nargs="*", default=list(SUITES), choices=list(SUITES),
                    help="Subset of comparison suites to run (default: all)")
    ap.add_argument("--skip-csharp", action="store_true", help="Skip the C# benchmarks")
    ap.add_argument("--skip-python", action="store_true", help="Skip the NumPy benchmarks")
    ap.add_argument("--skip-build", action="store_true", help="Reuse the existing Release build")
    ap.add_argument("--quick", action="store_true", help="Deprecated alias for --depth light")
    ap.add_argument("--depth", choices=tuple(DEPTHS), default="measure",
                    help="pass=single execution; light=1/6 budget; measure=full rigor")
    ap.add_argument("--dtypes", help="Comma-separated dtype filter; multi-dtype cases match any side")
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
    ap.add_argument("--skip-openblas", action="store_true",
                    help="Skip the full LinearAlgebra OpenBLAS profile and targeted backend extras")
    ap.add_argument("--no-history", action="store_true",
                    help="Skip writing the committable benchmark/history/<date>_<sha>/ snapshot + latest symlink")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        raw_argv = prompt_run_options()
    args = ap.parse_args(raw_argv)

    if args.quick:
        if args.depth != "measure":
            ap.error("--quick cannot be combined with --depth; use --depth light")
        print("--quick is deprecated; using --depth light", flush=True)
        args.depth = "light"
    try:
        requested_dtypes = parse_dtypes(args.dtypes)
    except ValueError as error:
        ap.error(str(error))

    # One contract for every child process: BDN configuration, NumPy timing, and standalone
    # matrices read these values. Canonical names avoid cross-language alias drift.
    os.environ["NUMSHARP_BENCHMARK_DEPTH"] = args.depth
    os.environ["NUMSHARP_BENCHMARK_DTYPES"] = ",".join(requested_dtypes)
    non_measure = args.depth != "measure"
    skip_official_openblas = args.skip_openblas or "float64" not in requested_dtypes
    if non_measure:
        # Pass/light validate execution or provide a rough local estimate. They must never replace
        # publication-quality root/docs/history artifacts.
        args.no_history = True

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = HERE / "results" / ts
    results_dir.mkdir(parents=True, exist_ok=True)
    csharp_out = results_dir / "csharp"
    csharp_out.mkdir(exist_ok=True)
    openblas_csharp_out = results_dir / "csharp-openblas"
    openblas_csharp_out.mkdir(exist_ok=True)
    numpy_json = results_dir / "numpy-results.json"
    (results_dir / "run-config.json").write_text(json.dumps({
        "depth": args.depth,
        "dtypes": list(requested_dtypes),
        "suites": list(args.suites),
        "publication_eligible": not non_measure,
    }, indent=2), encoding="utf-8")
    print(f"Results -> {results_dir}")
    print(f"Depth: {args.depth} · dtypes: {', '.join(requested_dtypes)}")

    t0 = time.time()

    # 1. Build the C# benchmark project (Release).
    if not args.skip_csharp and not args.skip_build:
        run(["dotnet", "build", "-c", "Release", "-f", TFM, str(CSHARP_PROJ),
             "-v", "q", "--nologo", "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0"], check=True)
        if "linalg" in args.suites and not skip_official_openblas:
            run(["dotnet", "build", "-c", "Release", str(OPENBLAS_CSHARP_PROJ),
                 "-v", "q", "--nologo", "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0",
                 "-p:GeneratePackageOnBuild=false"], check=True)

    # 2. NumPy: sweep all three sizes per suite, concatenate into one JSON.
    if not args.skip_python:
        merged = []
        for s in args.suites:
            tmp = results_dir / f"numpy-{s}.json"
            cmd = [sys.executable, str(PY_BENCH), "--suite", s, "--cache-sizes",
                   "--depth", args.depth, "--dtypes", ",".join(requested_dtypes),
                   "--output", str(tmp)]
            # The api suite carries the scalar/dtype/text dispatch ops — also sweep the N=1
            # (Scalar) pure-dispatch tier there (the "scalar x scalar" point). Scoped to this
            # suite so no other suite emits unmatched N=1 rows.
            if s == "api":
                cmd.append("--with-scalar")
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
                cwd=CSHARP_DIR, check=non_measure)
            if ARTIFACTS.exists():
                for f in ARTIFACTS.glob("*-report-full-compressed.json"):
                    shutil.copy(f, csharp_out / f.name)
        print(f"C#: collected {len(list(csharp_out.glob('*.json')))} class reports")

    # 3b. The COMPLETE official LinearAlgebra suite under OpenBLAS. This is a separate executable,
    #     not a hand-copied case list: it discovers the same LinAlgBenchmarks/LinalgApiBenchmarks
    #     classes and uses the same OfficialBenchmarkConfig as the Managed run. The process boundary
    #     keeps Core-only Managed runs unable to load a native backend by construction.
    openblas_matrix_base = results_dir / "benchmark-report.openblas-matrix"
    merge_env = {**os.environ, "PYTHONUTF8": "1"}
    if not args.skip_csharp and not skip_official_openblas and "linalg" in args.suites:
        if OPENBLAS_ARTIFACTS.exists():
            shutil.rmtree(OPENBLAS_ARTIFACTS, ignore_errors=True)
        print("\n=== C# suite: linalg (OpenBLAS profile; complete official suite) ===", flush=True)
        openblas_env = {
            **os.environ,
            "NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL": "0",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        run(["dotnet", "run", "-c", "Release", "--no-build", "-f", TFM,
             "--project", str(OPENBLAS_CSHARP_PROJ), "--", "--filter", SUITES["linalg"]],
            cwd=OPENBLAS_CSHARP_DIR, check=non_measure, env=openblas_env)
        if OPENBLAS_ARTIFACTS.exists():
            for f in OPENBLAS_ARTIFACTS.glob("*-report-full-compressed.json"):
                shutil.copy(f, openblas_csharp_out / f.name)
        numpy_linalg = results_dir / "numpy-linalg.json"
        if numpy_linalg.exists() and any(openblas_csharp_out.glob("*.json")):
            run([sys.executable, str(MERGE), "--numpy", str(numpy_linalg),
                 "--csharp", str(openblas_csharp_out), "--output", str(openblas_matrix_base),
                 "--require-universal-tiers"],
                check=True, env=merge_env)
        print(f"OpenBLAS C#: collected {len(list(openblas_csharp_out.glob('*.json')))} class reports")

    # 4. Merge into the unified per-(op, dtype, N) ratio report.
    managed_matrix_base = results_dir / "benchmark-report.managed-matrix"
    run([sys.executable, str(MERGE), "--numpy", str(numpy_json),
         "--csharp", str(csharp_out), "--output", str(managed_matrix_base),
         "--require-universal-tiers"], check=True, env=merge_env)

    # The unified report the op-matrix merge just wrote; the iterator + matrix
    # subsystems below each append one section to it.
    report_md = results_dir / "benchmark-report.managed-matrix.md"

    # Pass/light are exact op-matrix execution profiles. The complementary subsystems own distinct
    # timing models and tracked output files; publication runs continue to execute them below.
    effective_skip_nditer = args.skip_nditer or non_measure
    effective_skip_layout = args.skip_layout or non_measure
    effective_skip_operand = args.skip_operand or non_measure
    effective_skip_cast = args.skip_cast or non_measure
    effective_skip_fusion = args.skip_fusion or non_measure
    effective_skip_targeted_openblas = skip_official_openblas or non_measure
    if non_measure:
        print("\nPass/light profile: complementary subsystem sheets are skipped; "
              "all 499 official BDN/NumPy scenarios remain in scope.", flush=True)

    # 4b. NDIter iterator benchmark — complementary harness (file-based, section-
    #     isolated, crash-resilient: a NumSharp AccessViolation is IGNORED and the
    #     section reported NA). Its result model is aspect x tier, not op/dtype/N,
    #     so it is APPENDED to the report as its own section rather than merged —
    #     preserving the iterator-isolation value the op matrix cannot express.
    if not effective_skip_nditer:
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

    # 4c. Complementary subsystems — layout / operand / cast / fusion / OpenBLAS.
    #     op/dtype/N matrix cannot express and appends its own rendered section.
    skip_matrix = {"layout": effective_skip_layout, "operand": effective_skip_operand,
                   "cast": effective_skip_cast, "fusion": effective_skip_fusion,
                   "openblas": effective_skip_targeted_openblas}
    for name, sheet, results, title in MATRIX_SUBSYSTEMS:
        if skip_matrix[name]:
            continue
        run_matrix_subsystem(name, sheet, results, title, report_md, results_dir, args.skip_build)

    # 4d. Canonical backend-profile merge. The full Managed C# matrix and the targeted backend
    #     harness use the SAME JSON row schema. Separate profile files remain available, while
    #     benchmark-report.json stores both measurements and an effective fastest-valid value per
    #     exact operation/dtype/N/scenario cell. MissingBackendException and NotSupportedException
    #     are availability states emitted by the targeted harness, never timed failures.
    profile_cmd = [sys.executable, str(PROFILE_MERGE),
                   "--managed", str(Path(str(managed_matrix_base) + ".json")),
                   "--output", str(results_dir / "benchmark-report")]
    backend_managed = HERE / "openblas" / "openblas_results.managed.json"
    backend_openblas = HERE / "openblas" / "openblas_results.openblas.json"
    if backend_managed.exists() and not effective_skip_targeted_openblas:
        profile_cmd.extend(["--managed-extra", str(backend_managed)])
    official_openblas = Path(str(openblas_matrix_base) + ".json")
    if official_openblas.exists() and not skip_official_openblas:
        profile_cmd.extend(["--openblas", str(official_openblas)])
        if backend_openblas.exists() and not effective_skip_targeted_openblas:
            profile_cmd.extend(["--openblas-extra", str(backend_openblas)])
    elif backend_openblas.exists() and not effective_skip_targeted_openblas:
        profile_cmd.extend(["--openblas", str(backend_openblas)])
    run(profile_cmd, check=True)

    if non_measure:
        validate_execution_depth(results_dir, args.depth)

    # Append the complementary subsystem sheets to the newly generated profile-aware report.
    report_md = results_dir / "benchmark-report.md"
    if not effective_skip_nditer:
        append_section(report_md, NPYITER_REPORT, "NDIter iterator benchmark")
    for name, _sheet, results, title in MATRIX_SUBSYSTEMS:
        if not skip_matrix[name]:
            append_section(report_md, results, title)

    # 5. Copy the headline artifacts to the benchmark/ root for convenience.
    if not non_measure:
        for name in ["benchmark-report.md", "benchmark-report.json", "benchmark-report.csv",
                     "benchmark-report.managed.json", "benchmark-report.openblas.json", "numpy-results.json"]:
            src = results_dir / name
            if src.exists():
                shutil.copy(src, HERE / name)

    # The rich DocFX dashboard fetches this relative URL at runtime. Keep it generated from the
    # exact same merge result instead of leaving the Function Explorer with a missing asset.
    if not non_measure:
        DOCS_BENCHMARK_JSON.parent.mkdir(parents=True, exist_ok=True)
        for name in ["benchmark-report.json", "benchmark-report.managed.json", "benchmark-report.openblas.json"]:
            src = results_dir / name
            if src.exists():
                shutil.copy(src, DOCS_BENCHMARK_JSON.parent / name)

    # 6. History snapshot + latest symlink — the committable provenance/publish step
    #    (benchmark/scripts/snapshot_history.py): copies the report + both profile JSON files and subsystem
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
