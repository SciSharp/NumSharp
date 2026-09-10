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
* Each completed C# and NumPy case is atomically checkpointed. A durable plan and
  source/build identity make interrupted runs resumable without repeating completed cases.
* NumPy side sweeps all three sizes in one invocation per suite (``--cache-sizes``); each
  result carries its own ``n``, which the merge keys on.
* No measured child ever holds the console. Every long measured phase streams its stdout+stderr
  to ``results/<ts>/logs/<phase>.log`` with a daemon thread mirroring the file to this console
  (``Session.run`` for the matrix phases, the ``run_logged`` helper for a direct child launch) —
  so a console whose output is suspended (conhost parks EVERY write from EVERY attached process
  after a Pause/Ctrl+S in the pane, until the next key) stalls only the mirror, never the
  measurement. With an inherited console an unbuffered write froze a 7-hour shard mid-class.

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
  python run_benchmark.py --no-lock-clock         # leave turbo boost on (default: locked off for the run)
  python run_benchmark.py --no-pin-core           # don't pin C#/NumPy to one performance core

Host stability (scripts/benchmark_host.py)
------------------------------------------
Both languages are pinned to the SAME single performance core (NUMSHARP_BENCHMARK_AFFINITY, honored
by both C# runners and numpy_benchmark.py) and the CPU clock is locked (turbo boost Disabled on the
High-performance scheme, restored on exit) for every measured phase. On the hybrid, Aggressive-boost
i9-13900K this suite runs on, the unpinned/boosting configuration produced 30-50 % faster excursions
in 2 of 50 BDN iterations that never reproduced - and became Statistics.Min.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from benchmark_modes import ALL_DTYPES, DEPTHS, parse_dtypes  # noqa: E402
from benchmark_session import Session, RunLock, atomic_json, read_json, provenance, check_resume  # noqa: E402
from numpy_checkpoint import validate_checkpoint  # noqa: E402
from benchmark_host import apply_runner_controls, logical_cpus_of, pick_benchmark_core  # noqa: E402

HERE = Path(__file__).resolve().parent
HISTORY_DIR = HERE / "history"
CSHARP_DIR = HERE / "NumSharp.Benchmark.CSharp"
CSHARP_PROJ = CSHARP_DIR / "NumSharp.Benchmark.CSharp.csproj"
OPENBLAS_CSHARP_DIR = HERE / "NumSharp.Benchmark.CSharp.OpenBLAS"
OPENBLAS_CSHARP_PROJ = OPENBLAS_CSHARP_DIR / "NumSharp.Benchmark.CSharp.OpenBLAS.csproj"
PY_BENCH = HERE / "NumSharp.Benchmark.Python" / "numpy_benchmark.py"
MERGE = HERE / "scripts" / "merge-results.py"
PROFILE_MERGE = HERE / "scripts" / "merge-backend-profiles.py"
ARTIFACTS = CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
OPENBLAS_ARTIFACTS = OPENBLAS_CSHARP_DIR / "BenchmarkDotNet.Artifacts" / "results"
TFM = "net10.0"
# The generated data lives on the orphan `data` branch, mounted as the refs/data submodule. A full
# eligible run publishes its history snapshot straight into it (pull -> override latest -> commit).
DATA_SUBMODULE = HERE.parent / "refs" / "data"
PUBLISH_PY = HERE.parent / "tools" / "dashboard_data" / "publish.py"

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


ACTIVE_SESSION = None
CONFIG_FIELDS = ("depth", "dtypes", "suites", "operations", "skip_csharp", "skip_python", "skip_nditer",
                 "skip_layout", "skip_operand", "skip_cast", "skip_fusion", "skip_openblas", "no_history",
                 "rerun", "from_results", "baseline", "regression_percent", "numpy_results")


def run(cmd, cwd=None, check=False, env=None):
    if ACTIVE_SESSION is not None:
        return ACTIVE_SESSION.run(cmd, cwd=cwd, check=check, env=env)
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, check=check, env=env)


def run_logged(cmd, log_path, cwd=None, check=False, env=None):
    """Run a long MEASURED child with stdout+stderr redirected to ``log_path`` (appended) and
    mirrored to this console by a daemon thread that tails the file.

    Why not the inherited console that ``run`` hands out: a console handle is a shared,
    BLOCKING sink. While a console's output is suspended, conhost parks every WriteConsole from
    every process attached to it (``DoWriteConsole`` -> ``CONSOLE_STATUS_WAIT``) — and a single
    Pause or Ctrl+S keystroke in the pane suspends it (``CONSOLE_OUTPUT_SUSPENDED``, set whenever
    the input buffer is in the cooked ``ENABLE_LINE_INPUT`` mode every shell leaves on; released
    only by the next non-modifier key, which conhost swallows). Windows Terminal forwards those
    keys unchanged. On 2026-09-07 one such keystroke froze the unary shard 7 h in: the BDN worker
    thread blocked writing its 48th ``WorkloadActual`` line, the in-process executor's 5-minute
    timeout then THREW an unhandled InvalidOperationException on the main thread, the runtime's
    unhandled-exception printer blocked on the same console in cooperative-GC mode, and the next
    GC (a low-memory notification handled on the finalizer thread) spun ``SuspendEE`` at 100 % of
    one core forever waiting for it — a process that was alive, not exiting, undebuggable by
    EventPipe, and the in-flight class (941 finished cases, 3.25 h) was lost with it.

    With the child writing to a FILE nothing it does can block on the terminal: a suspended
    console stalls only this mirror thread (daemon — the run moves on; the mirror catches up when
    a key is pressed). A pipe + echoing reader would not do: the reader blocks on the same
    suspension, the pipe fills, and the child is back to blocking on its own writes."""
    print(f"\n$ {' '.join(str(c) for c in cmd)}\n  [stdout+stderr -> {log_path}]", flush=True)
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    finished = threading.Event()

    def mirror(start):
        # Raw bytes pass through untouched (BDN emits UTF-8 + ANSI colours). A mirror failure —
        # the console closed or lost — is not the run's problem: the log file is the record.
        out = getattr(sys.stdout, "buffer", None)
        try:
            with open(log_path, "rb") as tail:
                tail.seek(start)
                while True:
                    chunk = tail.read()
                    if chunk:
                        if out is not None:
                            out.write(chunk)
                            out.flush()
                        else:
                            sys.stdout.write(chunk.decode("utf-8", "replace"))
                            sys.stdout.flush()
                    elif finished.is_set():
                        return
                    else:
                        finished.wait(0.25)
        except (OSError, ValueError):
            return

    with open(log_path, "ab") as log:
        # The child (and every grandchild — `dotnet run` hands the handle to the BDN exe) appends
        # through the inherited handle, so all three share ONE file position: mirror from where
        # the file ends now, not from the previous phase that may have used the same log.
        thread = threading.Thread(target=mirror, args=(log_path.stat().st_size,),
                                  name=f"mirror:{log_path.name}", daemon=True)
        thread.start()
        try:
            proc = subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
                                  stdout=log, stderr=subprocess.STDOUT)
        finally:
            finished.set()
            # A suspended console keeps the mirror stuck in its write; never wait on it.
            thread.join(timeout=5.0)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)
    return proc


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


def parser():
    ap = argparse.ArgumentParser(description="Durable NumSharp vs NumPy benchmark runs")
    ap.add_argument("--suites", nargs="*", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--depth", choices=tuple(DEPTHS), default="measure")
    ap.add_argument("--dtypes", help="Comma-separated dtype names or aliases")
    ap.add_argument("--operations", nargs="+", help="Operation title glob(s), e.g. 'np.add*'")
    for name in ("csharp", "python", "build", "nditer", "layout", "operand", "cast", "fusion", "openblas"):
        ap.add_argument(f"--skip-{name}", action="store_true")
    ap.add_argument("--quick", action="store_true", help="Deprecated alias for --depth light")
    ap.add_argument("--no-history", action="store_true")
    ap.add_argument("--no-data-publish", action="store_true",
                    help="Do not publish the snapshot into the refs/data submodule "
                         "(default: pull refs/data, override its benchmark/latest, and commit when refs/data is initialized)")
    ap.add_argument("--run-dir", type=Path, help="New results directory (must not already contain a run)")
    ap.add_argument("--resume", nargs="?", const="latest", help="Resume a run directory, or latest")
    ap.add_argument("--rerun", choices=("failed", "bad", "degraded"), help="Create a new run of selected problem cells")
    ap.add_argument("--from-results", type=Path, help="Run directory or report JSON used for rerun selection")
    ap.add_argument("--baseline", type=Path, help="Reference report/directory for degraded selection")
    ap.add_argument("--regression-percent", type=float, default=10,
                    help="Degraded: NumSharp time increase over baseline (default 10%%)")
    ap.add_argument("--numpy-results", type=Path, help="Explicit NumPy JSON to reuse with --skip-python")
    ap.add_argument("--plan-only", action="store_true", help="Discover and save selected cases without timing them")
    ap.add_argument("--progress-interval", type=float, default=10, help="Terminal heartbeat seconds (default 10)")
    ap.add_argument("--no-title", action="store_true", help="Only report progress in the terminal")
    ap.add_argument("--verbose", action="store_true", help="Also echo child output; full output always goes to logs/")
    ap.add_argument("--attempts", type=int, default=3, help="Attempts per incomplete matrix suite; completed cases are retained (default 3)")
    ap.add_argument("--process-timeout", type=float, help="Optional maximum seconds per child process; kills its process tree on expiry")
    ap.add_argument("--no-lock-clock", action="store_true",
                    help="Leave turbo boost as-is (default: boost Disabled on the High-performance scheme "
                         "for every measured phase, restored on exit; system-wide, so both languages)")
    ap.add_argument("--no-pin-core", action="store_true",
                    help="Do not pin the C# and NumPy processes to one performance core "
                         "(default: NUMSHARP_BENCHMARK_AFFINITY is exported to every child)")
    return ap


def valid_run_config(config):
    """Recognize complete durable configuration records without trusting JSON types."""
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        return False
    options = config.get("options")
    if not isinstance(options, dict) or not all(field in options for field in CONFIG_FIELDS):
        return False
    if not isinstance(config.get("provenance"), dict) or not config["provenance"]:
        return False
    if not isinstance(config.get("depth"), str) or config["depth"] not in DEPTHS:
        return False
    if not isinstance(config.get("dtypes"), list) or not config["dtypes"]:
        return False
    if not all(isinstance(dtype, str) and dtype in ALL_DTYPES for dtype in config["dtypes"]):
        return False
    if not isinstance(config.get("suites"), list) or not all(
            isinstance(suite, str) and suite in SUITES for suite in config["suites"]):
        return False
    if options["depth"] != config["depth"] or options["suites"] != config["suites"]:
        return False
    if options["dtypes"] is not None and not isinstance(options["dtypes"], str):
        return False
    try:
        if list(parse_dtypes(options["dtypes"])) != config["dtypes"]:
            return False
    except ValueError:
        return False
    if any(type(options[name]) is not bool for name in CONFIG_FIELDS if name.startswith("skip_") or name == "no_history"):
        return False
    if options["operations"] is not None and (not isinstance(options["operations"], list)
            or not all(isinstance(value, str) for value in options["operations"])):
        return False
    if options["rerun"] not in (None, "failed", "bad", "degraded"):
        return False
    if any(options[name] is not None and not isinstance(options[name], str)
           for name in ("from_results", "baseline", "numpy_results")):
        return False
    percent = options["regression_percent"]
    return type(percent) in (int, float) and math.isfinite(percent) and percent >= 0


def recent_runs():
    """Newest valid run records, including the last custom --run-dir destination."""
    results = HERE / "results"
    paths = set(results.glob("*/run-config.json"))
    try:
        pointer = read_json(results / "last-run.json")
        if isinstance(pointer, dict) and isinstance(pointer.get("directory"), str):
            paths.add(Path(pointer["directory"]) / "run-config.json")
    except (ValueError, OSError):
        pass
    candidates = []
    for path in paths:
        try:
            config = read_json(path)
            if valid_run_config(config) and config.get("resume_supported") is not False:
                candidates.append((path.stat().st_mtime_ns, path.parent.resolve(), config))
        except (ValueError, OSError):
            continue
    return sorted(candidates, key=lambda item: (item[0], str(item[1])), reverse=True)


def prompt_startup_options(input_fn=input):
    """Offer an unfinished, inactive run before the normal new-run picker."""
    current = None
    for _modified, directory, config in recent_runs():
        try:
            state = read_json(directory / "run-state.json")
        except (ValueError, OSError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        if state.get("status") in ("complete", "planned"):
            continue
        # Lock files outlive their owner. Probe the OS lock, never the file's existence.
        # execute() acquires the locks again after the user answers to close the race.
        try:
            with RunLock(directory):
                pass
        except (RuntimeError, OSError):
            continue
        if current is None:
            current = provenance(HERE.parent)
        try:
            check_resume(config["provenance"], current)
        except ValueError as error:
            print(f"\nUnfinished run cannot resume: {directory}\n{error}", flush=True)
            continue
        print(f"\nUnfinished benchmark found: {directory}")
        print(f"Depth: {config['depth']} · dtypes: {', '.join(config['dtypes'])}")
        completed, total = state.get("completed"), state.get("total")
        if isinstance(completed, int) and isinstance(total, int):
            print(f"Last saved progress: {completed:,}/{total:,} successful cases; "
                  f"stage: {state.get('stage', 'unknown')}. Checkpoints will be rechecked.")
        else:
            print("Stopped before progress was saved. Checkpoints will be checked on resume.")
        while True:
            answer = input_fn("Resume this benchmark? [Y/n]: ").strip().lower()
            if answer in {"", "y", "yes"}:
                return ["--resume", str(directory)]
            if answer in {"n", "no"}:
                return prompt_run_options(input_fn)
            print("Enter y to resume or n to choose a new benchmark.")
    return prompt_run_options(input_fn)


def resolve_run(value):
    if str(value) == "latest":
        candidates = recent_runs()
        if not candidates:
            raise ValueError("No resumable run found. Older runs have no per-case checkpoints.")
        return candidates[0][1]
    return Path(value).resolve()


def main(argv=None):
    ap = parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        if not sys.stdin.isatty():
            ap.error("Non-interactive runs require explicit options, e.g. --depth measure or --resume latest")
        try:
            raw = prompt_startup_options()
        except (EOFError, KeyboardInterrupt):
            print("\nBenchmark startup cancelled; no work was started.", flush=True)
            return 130
        except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
            print(f"Benchmark startup failed: {error}", file=sys.stderr, flush=True)
            return 1
    args = ap.parse_args(raw)
    if not math.isfinite(args.progress_interval) or not math.isfinite(args.regression_percent) or args.progress_interval <= 0 or args.regression_percent < 0:
        ap.error("progress interval must be positive; regression percent must be non-negative")
    if args.attempts < 1 or (args.process_timeout is not None and
            (not math.isfinite(args.process_timeout) or args.process_timeout <= 0)):
        ap.error("attempts and process timeout must be positive")
    if args.from_results and not args.rerun:
        ap.error("--from-results requires --rerun")
    if args.baseline and args.rerun != "degraded":
        ap.error("--baseline applies only to --rerun degraded")
    if args.resume and (args.run_dir or args.rerun):
        ap.error("--resume cannot be combined with --run-dir or --rerun")
    if args.resume and args.quick:
        ap.error("Resume uses the saved depth; --quick cannot change it")
    if args.rerun and not args.from_results:
        ap.error("--rerun requires --from-results")
    if args.rerun == "degraded" and not args.baseline:
        ap.error("--rerun degraded requires --baseline")
    if args.quick:
        if args.depth != "measure":
            ap.error("--quick cannot be combined with --depth")
        args.depth = "light"
    if args.skip_csharp and args.skip_python:
        ap.error("At least one benchmark engine must be enabled")
    try:
        directory = resolve_run(args.resume) if args.resume else (
            args.run_dir.resolve() if args.run_dir else HERE / "results" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
        if not args.resume and directory.exists() and any(directory.iterdir()):
            raise ValueError("A new run needs an empty directory; existing results must use --resume or --rerun")
        with RunLock(directory), RunLock(HERE / "results" / ".workspace"):
            return execute(args, raw, directory)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Completed cases are saved. Resume: python benchmark/run_benchmark.py --resume \"{directory}\"", flush=True)
        return 130
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"\nBenchmark stopped: {error}", file=sys.stderr, flush=True)
        if 'directory' in locals():
            print(f"Results/logs: {directory}\nResume: python benchmark/run_benchmark.py --resume \"{directory}\"", file=sys.stderr)
        return 1


def execute(args, raw, directory):
    global ACTIVE_SESSION
    config_path = directory / "run-config.json"
    config_fields = CONFIG_FIELDS
    print("Checking benchmark code and environment identity...", flush=True)
    current = provenance(HERE.parent)
    # Host stability lever 1 (core pin): pick one performance core now so its mask can be recorded in
    # the durable run-config and exported to every measured child (scripts/benchmark_host.py).
    pin_mask = None if args.no_pin_core else pick_benchmark_core()
    if args.resume:
        if not config_path.exists():
            raise ValueError("This directory has no resumable run-config.json; use old reports with --rerun instead.")
        saved = read_json(config_path)
        if isinstance(saved, dict) and saved.get("resume_supported") is False:
            raise ValueError("History snapshots preserve provenance, not checkpoints; use --rerun with this snapshot")
        if not valid_run_config(saved):
            raise ValueError("This run lacks a valid durable configuration; use its report with --rerun instead.")
        check_resume(saved["provenance"], current)
        for field in config_fields:
            flag = "--" + field.replace("_", "-")
            if any(token == flag or token.startswith(flag + "=") for token in raw):
                raise ValueError(f"{flag} is fixed by the saved run. Start a new run to change its selection.")
            setattr(args, field, saved["options"][field])
        requested = tuple(saved["dtypes"])
    else:
        if config_path.exists():
            raise ValueError(f"A run already exists at {directory}; use --resume")
        requested = parse_dtypes(args.dtypes)
        saved = {"schema_version": 1, "depth": args.depth, "dtypes": list(requested),
                 "suites": args.suites, "provenance": current,
                 "pin_core_mask": hex(pin_mask) if pin_mask else None,
                 "lock_clock": not args.no_lock_clock,
                 "options": {key: str(getattr(args, key)) if isinstance(getattr(args, key), Path)
                             else getattr(args, key) for key in config_fields}}
    # A partial or exploratory run must never replace the full published benchmark.
    eligible = (args.depth == "measure" and not args.rerun and not args.operations
                and set(args.suites) == set(SUITES) and requested == ALL_DTYPES
                and not any(getattr(args, "skip_" + name) for name in (
                    "csharp", "python", "nditer", "layout", "operand", "cast", "fusion", "openblas")))
    saved["publication_eligible"] = eligible
    if args.skip_python:
        reference = directory / "numpy-reference.json"
        if args.resume:
            if not reference.exists() or hashlib.sha256(reference.read_bytes()).hexdigest() != saved.get("numpy_reference_sha256"):
                raise ValueError("The saved NumPy reference is missing or changed; start a new run")
        else:
            source = Path(args.numpy_results) if args.numpy_results else HERE / "numpy-results.json"
            if not source.exists():
                raise ValueError("--skip-python needs an existing --numpy-results JSON file")
            rows = read_json(source)
            if not isinstance(rows, list) or any(not isinstance(row, dict) or not {"name", "dtype", "n"}.issubset(row) for row in rows):
                raise ValueError("--numpy-results must be raw NumPy result rows, not a merged report")
            atomic_json(reference, rows)
            saved["numpy_reference_sha256"] = hashlib.sha256(reference.read_bytes()).hexdigest()
        atomic_json(directory / "numpy-results.json", read_json(reference))
    atomic_json(config_path, saved)
    atomic_json(HERE / "results" / "last-run.json", {"directory": str(directory.resolve())})
    os.environ["NUMSHARP_BENCHMARK_DEPTH"] = args.depth
    os.environ["NUMSHARP_BENCHMARK_DTYPES"] = ",".join(requested)
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Never inherit a previous caller's allowlist/checkpoint destination during discovery.
    for name in ("NUMSHARP_BENCHMARK_CASES_FILE", "NUMSHARP_BENCHMARK_CHECKPOINT_DIR"):
        os.environ.pop(name, None)
    initial_plan = read_json(directory / "plan.json") if args.resume and (directory / "plan.json").exists() else []
    session = Session(directory, initial_plan, interval=args.progress_interval, title=not args.no_title,
                      verbose=args.verbose, process_timeout=args.process_timeout)
    if args.resume:
        session.prepare_retry()
    ACTIVE_SESSION = session
    print(f"Results: {directory}\nDepth: {args.depth}; dtypes: {', '.join(requested)}", flush=True)
    pin_note = (f"pinned to logical CPU {logical_cpus_of(pin_mask)} ({hex(pin_mask)})" if pin_mask
                else "not pinned")
    print(f"Host: {pin_note} · clock lock {'off (--no-lock-clock)' if args.no_lock_clock else 'on'}", flush=True)
    try:
        return execute_stages(args, directory, session, requested, eligible)
    except BaseException:
        session.progress(force=True, status="interrupted")
        raise
    finally:
        ACTIVE_SESSION = None


def execute_stages(args, directory, session, requested, eligible):
    import fnmatch
    official_openblas = not args.skip_openblas and not args.skip_csharp and "linalg" in args.suites and "float64" in requested
    projects = [("managed", CSHARP_PROJ, CSHARP_DIR)] if not args.skip_csharp else []
    if official_openblas:
        projects.append(("openblas", OPENBLAS_CSHARP_PROJ, OPENBLAS_CSHARP_DIR))
    for engine, project, cwd in projects:
        if not args.skip_build:
            session.stage = "build:" + engine
            run(["dotnet", "build", "-c", "Release", "-f", TFM, project, "-v", "q", "--nologo",
                 "-clp:NoSummary;ErrorsOnly", "-p:WarningLevel=0", "-p:GeneratePackageOnBuild=false"], check=True)
    # Pin the actual binaries as well as source: --skip-build can intentionally use a
    # stale build, but a later rebuild must not silently mix implementations on resume.
    session.stage = "verify build identity"
    images = {}
    for engine, project, _cwd in projects:
        output = project.parent / "bin" / "Release" / TFM
        for path in sorted(output.rglob("*")):
            if path.is_file() and (path.suffix in (".dll", ".so", ".dylib") or path.name.endswith((".deps.json", ".runtimeconfig.json"))):
                images[f"{engine}/{path.relative_to(output).as_posix()}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not any(key.startswith(engine + "/") for key in images):
            raise ValueError(f"Missing Release build for {engine}; rerun without --skip-build")
    image_path = directory / "build-identity.json"
    if args.resume and image_path.exists() and read_json(image_path) != images:
        raise ValueError("Built benchmark binaries changed; start a new run instead of mixing measurements")
    atomic_json(image_path, images)
    # Host controls for EVERY measured phase below - NumPy, C# managed + OpenBLAS, the NDIter harness,
    # the matrix subsystems, the backend profiles (scripts/benchmark_host.py). Placed after the build
    # on purpose (a pinned build compiles on one core; a locked one at base clock).
    #   * pin: this process is pinned to one performance core, and every child it spawns - `dotnet run`
    #     and the BDN exe, numpy_benchmark.py, each *_sheet.py and the `dotnet run -` scripts + Python
    #     twins they launch - INHERITS that mask at spawn. NUMSHARP_BENCHMARK_AFFINITY is exported so
    #     the runners that also self-pin agree; --no-pin-core propagates as NUMSHARP_BENCHMARK_PIN=0.
    #   * clock: turbo boost Disabled system-wide (one mechanism for both languages); restored on every
    #     exit path Python controls (atexit) and self-healed from the state file after a hard kill;
    #     NUMSHARP_BENCHMARK_CLOCK_LOCKED=1 tells every child driver not to nest a lock. Released right
    #     after the last measurement, before the merge/report steps.
    clock, _ = apply_runner_controls(lock_clock=not args.no_lock_clock, pin=not args.no_pin_core)
    plan_file = directory / "plan.json"
    if args.resume and plan_file.exists():
        plan = read_json(plan_file)
    else:
        plan = []
        for engine, project, cwd in projects:
            session.stage = "discover:" + engine
            target = directory / f"plan-{engine}.json"
            run(["dotnet", "run", "-c", "Release", "--no-build", "-f", TFM, "--project", project,
                 "--", "--benchmark-plan-json", target], cwd=cwd, check=True)
            plan.extend({**row, "engine": engine} for row in read_json(target)
                        if row["suite"] in args.suites and (engine != "openblas" or row["suite"] == "linalg"))
        if not args.skip_python:
            for suite in args.suites:
                session.stage = "discover:numpy:" + suite
                target = directory / f"plan-numpy-{suite}.json"
                cmd = numpy_command(args, requested, suite) + ["--plan-json", target]
                run(cmd, check=True)
                plan.extend({**row, "engine": "numpy"} for row in read_json(target))
        if args.operations:
            plan = [row for row in plan if any(fnmatch.fnmatchcase(row["operation"], pattern)
                                             for pattern in args.operations)]
        if args.rerun:
            from benchmark_selection import select_plan
            plan, selection = select_plan(plan, Path(args.from_results), args.rerun,
                                          Path(args.baseline) if args.baseline else None, args.regression_percent)
            atomic_json(directory / "selection.json", selection)
            print(f"Rerun selection: {len(plan):,} engine/case pairs; details: {directory / 'selection.json'}")
            for warning in selection.get("warnings", []):
                print(f"Selection: {warning}")
            if selection.get("unmatched") or selection.get("noncomparable"):
                print(f"Selection excluded {len(selection.get('unmatched', []))} unmatched and "
                      f"{len(selection.get('noncomparable', []))} non-comparable source cells; see selection.json")
        atomic_json(plan_file, plan)
    if len({(row["engine"], row["id"]) for row in plan}) != len(plan):
        raise ValueError("Discovery returned duplicate case IDs")
    session.set_plan(plan)
    if args.skip_python:
        from benchmark_selection import reference_rows_for_plan
        atomic_json(directory / "numpy-results.json",
                    reference_rows_for_plan(read_json(directory / "numpy-reference.json"), plan))
    if args.resume:
        session.prepare_retry()
    sheets = []
    if args.depth == "measure" and not args.rerun and not args.operations:
        if not args.skip_nditer:
            sheets.append(("nditer", NPYITER_SHEET, NPYITER_REPORT, "NDIter iterator benchmark"))
        sheets.extend(item for item in MATRIX_SUBSYSTEMS if not getattr(args, "skip_" + item[0]))
    for name, *_ in sheets:
        session.stage_state.setdefault("subsystem:" + name, "pending")
    session.stage = "plan saved"
    session.progress(force=True)
    if args.plan_only:
        session.progress(force=True, status="planned")
        print(f"Plan: {plan_file}\nRun it: python benchmark/run_benchmark.py --resume \"{directory}\"")
        return 0
    if not plan and not sheets:
        print("No cases match the selection; no benchmarks or published reports changed.")
        session.progress(force=True, status="complete")
        return 0
    for folder in ("csharp", "csharp-openblas", "numpy-checkpoints", "cases"):
        (directory / folder).mkdir(exist_ok=True)
    for engine in ("numpy", "managed", "openblas"):
        for suite in args.suites:
            pending = session.pending(engine, suite)
            if not pending:
                continue
            session.engine = engine
            session.stage = f"{engine}:{suite}"
            session.stage_state[session.stage] = "running"
            for attempt in range(1, args.attempts + 1):
                if attempt > 1:
                    print(f"Retry {attempt}/{args.attempts}: {len(pending)} pending/failed cases in {engine}:{suite}")
                session.prepare_retry(pending)
                try:
                    run_matrix_attempt(args, requested, directory, engine, suite, pending)
                except TimeoutError as error:
                    print(f"!! {error}", flush=True)
                # Any scheduled cell without an event remains pending. End this attempt
                # before reloading failed checkpoints; the next attempt explicitly requeues them.
                session.retry_pending.difference_update((engine, row["id"]) for row in pending)
                session.recover()
                if engine == "numpy":
                    materialize_numpy(directory, plan)
                pending = session.pending(engine, suite)
                if not pending:
                    break
            session.stage_state[session.stage] = "complete" if not session.pending(engine, suite) else "incomplete"
            session.progress(force=True)
    session.engine = None
    if not args.skip_python:
        materialize_numpy(directory, plan)
    for name, sheet, source, title in sheets:
        stage = "subsystem:" + name
        dest = directory / source.name
        if session.stage_state.get(stage) == "complete" and dest.exists():
            continue
        session.stage = stage
        session.stage_state[stage] = "running"
        before = source.stat().st_mtime_ns if source.exists() else None
        cmd = [sys.executable, sheet] + (["--skip-build"] if args.skip_build else [])
        result = run(cmd)
        fresh = source.exists() and source.stat().st_mtime_ns != before
        if result.returncode == 0 and fresh:
            for artifact in source.parent.glob(source.stem + ".*"):
                shutil.copy2(artifact, directory / artifact.name)
            if name == "nditer":
                cards_result = run([sys.executable, NPYITER_CARDS])
                if cards_result.returncode == 0 and (NPYITER_DIR / "cards").exists():
                    shutil.copytree(NPYITER_DIR / "cards", directory / "nditer_cards", dirs_exist_ok=True)
                else:
                    session.stage_state[stage] = "incomplete"
                    print("!! NDIter cards were not generated; resume will retry this stage", flush=True)
                    continue
            session.stage_state[stage] = "complete"
        else:
            session.stage_state[stage] = "incomplete"
            print(f"!! {name}: no fresh successful report; old output excluded", flush=True)
    # Host stability: last measured phase is done - restore turbo boost before merge/report steps.
    clock.restore()
    incomplete = any(session.outcomes.get(key) != "ok" for key in session.cells)
    incomplete |= any(value != "complete" for key, value in session.stage_state.items() if key.startswith("subsystem:"))
    session.stage = "merge reports"
    session.progress(force=True)
    merge_reports(directory, sheets, session, require_tiers=not args.rerun and not args.operations and not incomplete)
    if args.depth != "measure" and not incomplete:
        validate_execution_depth(directory, args.depth)
    if eligible and not incomplete:
        for name in ("benchmark-report.md", "benchmark-report.json", "benchmark-report.csv",
                     "benchmark-report.managed.json", "benchmark-report.openblas.json", "numpy-results.json"):
            src = directory / name
            if src.exists():
                shutil.copy2(src, HERE / name)
        if not args.no_history:
            session.stage = "history snapshot"
            run([sys.executable, HERE / "scripts" / "snapshot_history.py", "--results-dir", directory, "--no-stage"], check=True)
            if not getattr(args, "no_data_publish", False):
                session.stage = "publish to data submodule"
                publish_history_to_data()
    session.stage = "finished"
    session.progress(force=True, status="incomplete" if incomplete else "complete")
    print(f"Report: {directory / 'benchmark-report.md'}\nRun state: {session.state_path}")
    if incomplete:
        print(f"Incomplete cases retained. Resume: python benchmark/run_benchmark.py --resume \"{directory}\"")
    return 1 if incomplete else 0


def publish_history_to_data():
    """Publish benchmark/history/latest into the refs/data submodule: pull it to the `data` tip,
    override its benchmark/latest/ with this run, and commit (locally, on branch `data`).

    Skipped when refs/data is not initialized (e.g. in CI's benchmark job, which publishes through
    its own throwaway worktree) — initialize it with:
      git clone --depth 1 -b data https://github.com/SciSharp/NumSharp.git refs/data
    Commits locally only; push refs/data when ready (CI's benchmark.yml pushes on its own path).
    """
    if not (DATA_SUBMODULE / ".git").exists():
        print(f"\nrefs/data not initialized ({DATA_SUBMODULE}); skipping data publish "
              f"(git clone --depth 1 -b data https://github.com/SciSharp/NumSharp.git refs/data)", flush=True)
        return
    latest = HISTORY_DIR / "latest"
    if not latest.exists():
        print("\nno benchmark/history/latest to publish; skipping data publish", flush=True)
        return
    # Non-fatal: the run + history snapshot are already saved; a data-publish hiccup (e.g. a diverged
    # refs/data) must not fail the run. Re-publish later with `refresh_data.py --type benchmark`.
    try:
        run([sys.executable, PUBLISH_PY, "--type", "benchmark", "--from", latest,
             "--branch-worktree", DATA_SUBMODULE, "--pull", "--commit"], check=True)
        print(f"\npublished benchmark into {DATA_SUBMODULE} (committed on branch 'data'; push refs/data when ready)",
              flush=True)
    except Exception as error:  # noqa: BLE001 — data publish is best-effort, never fails the run
        print(f"\nwarn: data publish into refs/data failed ({error}); the run is fine — re-publish with: "
              f"python tools/dashboard_data/refresh_data.py --type benchmark", flush=True)


def numpy_command(args, requested, suite):
    return [sys.executable, PY_BENCH, "--suite", suite, "--cache-sizes", "--depth", args.depth,
            "--dtypes", ",".join(requested)] + (["--with-scalar"] if suite == "api" else [])


def run_matrix_attempt(args, requested, directory, engine, suite, pending):
    allowlist = directory / "cases" / f"{engine}-{suite}.json"
    atomic_json(allowlist, [row["id"] for row in pending])
    if engine == "numpy":
        return run(numpy_command(args, requested, suite) + ["--cases-file", allowlist,
                   "--checkpoint-dir", directory / "numpy-checkpoints", "--output", directory / f"numpy-{suite}.json"])
    project, cwd, folder = ((CSHARP_PROJ, CSHARP_DIR, "csharp") if engine == "managed"
                            else (OPENBLAS_CSHARP_PROJ, OPENBLAS_CSHARP_DIR, "csharp-openblas"))
    env = {**os.environ, "NUMSHARP_BENCHMARK_CASES_FILE": str(allowlist),
           "NUMSHARP_BENCHMARK_CHECKPOINT_DIR": str(directory / folder)}
    if engine == "openblas":
        env.update(NUMSHARP_OPENBLAS_BUNDLE_AUTOINSTALL="0", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    return run(["dotnet", "run", "-c", "Release", "--no-build", "-f", TFM,
                "--project", project, "--", "--filter", SUITES[suite]], cwd=cwd, env=env)


def materialize_numpy(directory, plan):
    selected = {row["id"] for row in plan if row["engine"] == "numpy"}
    rows = []
    for path in (directory / "numpy-checkpoints").glob("*.json"):
        try:
            payload = read_json(path)
        except (ValueError, OSError):
            continue
        if payload.get("id") in selected and payload.get("status") == "ok" and validate_checkpoint(payload):
            rows.append(payload["result"])
    atomic_json(directory / "numpy-results.json", rows)


def merge_reports(directory, sheets, session, require_tiers):
    bases = []
    for engine, folder in (("managed", "csharp"), ("openblas", "csharp-openblas")):
        if engine == "openblas" and not any((directory / folder).glob("*.json")):
            continue
        base = directory / f"benchmark-report.{engine}-matrix"
        cmd = [sys.executable, MERGE, "--numpy", directory / "numpy-results.json", "--csharp", directory / folder, "--output", base]
        if require_tiers:
            cmd.append("--require-universal-tiers")
        run(cmd, check=True)
        bases.append((engine, Path(str(base) + ".json")))
    cmd = [sys.executable, PROFILE_MERGE, "--managed", bases[0][1], "--output", directory / "benchmark-report"]
    if not require_tiers:
        cmd.append("--allow-partial-profiles")
    for engine, path in bases[1:]:
        cmd.extend(["--openblas", path])
    if session.stage_state.get("subsystem:openblas") == "complete":
        for profile in ("managed", "openblas"):
            path = directory / f"openblas_results.{profile}.json"
            if path.exists():
                flag = "--managed-extra" if profile == "managed" else (
                    "--openblas-extra" if len(bases) > 1 else "--openblas")
                cmd.extend([flag, path])
    run(cmd, check=True)
    for name, _sheet, source, title in sheets:
        if session.stage_state.get("subsystem:" + name) == "complete":
            append_section(directory / "benchmark-report.md", directory / source.name, title)


if __name__ == "__main__":
    sys.exit(main())
