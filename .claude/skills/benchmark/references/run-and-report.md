# Running the harness & reading the report

## Official run — `benchmark/run_benchmark.py` (the entry point)

Builds the C# suite, runs each suite through BenchmarkDotNet (per-class JSON, resumable), sweeps warm NumPy across
1K/100K/10M, merges on `(op, dtype, N)`, appends the five subsystems, archives raw scratch to
`results/<ts>/` (gitignored), and writes the committable `history/<date>_<sha>/` snapshot.

```bash
python run_benchmark.py                       # full official run (all 14 suites + 5 subsystems)
python run_benchmark.py --suites manipulation unary   # subset of the op matrix
python run_benchmark.py --skip-build          # reuse existing Release build
python run_benchmark.py --skip-csharp         # NumPy only
python run_benchmark.py --quick               # dev: fewer NumPy iterations
python run_benchmark.py --no-history          # don't write the history snapshot
# subsystem opt-outs: --skip-nditer --skip-layout --skip-operand --skip-cast --skip-fusion
```

**Cost:** the full matrix is long (µs–ms array ops × 15 dtypes × 3 sizes × 14 suites + subsystems). For iterating
on one op, `--suites <that suite>` or the smoke path (`--list flat` + `numpy_benchmark.py --suite <s> --quick`) is
usually the right scope. A full measured run + committed snapshot is the post-release `.github/workflows/
benchmark.yml` ritual, not something to kick off casually.

## Two methodology guards (why the C# side is configured oddly)

`OfficialBenchmarkConfig` (in `Infrastructure/BenchmarkConfig.cs`):
- **InProcessEmit toolchain** — BenchmarkDotNet's default out-of-process toolchain fails here
  ("project names need to be unique") because sibling `.claude/worktrees/` checkouts contain same-named benchmark
  projects. In-process also matches the warm long-lived NumPy process, so the cross-language ratio is fair.
- **25 ms-capped, 50-iteration job** — BDN's default Throughput strategy ramps to thousands of invocations per
  iteration for nanosecond microbenchmarks; for µs–ms array ops that made a single 10M case ~25 s and the full
  matrix take days. Capping iteration time lets the pilot pick a per-op invocation count that fits 25 ms while
  preserving all 50 iterations (~15× faster).

The nditer subsystem reports a section that crashes all retries (the known intermittent `AccessViolation`) as
**NA/IGNORED**, never a failure.

## Reading the report

- **Convention is NPY/NS** (NumPy_ms / NumSharp_ms, `>1` = NumSharp faster). Icons ✅ `≥1.0` 🟡 `≥0.5` 🟠 `≥0.2`
  🔴 `<0.2`.
- The report has a **per-size geomean summary** + the full **per-(op, dtype, N) ratio matrix**, then the five
  appended subsystem sections.
- A row missing a C# or NumPy value ("C# not run" / "NumPy only") almost always means the two names didn't
  **normalize to the same join key** — check the C# `[Benchmark(Description)]` vs the NumPy `.name`.

## History snapshots — what we commit

| Path | Tracked? | Contents |
|------|----------|----------|
| `benchmark/results/<ts>/` | ❌ gitignored | raw per-run scratch (per-suite NumPy JSON, BDN per-class reports, merged json/csv). |
| `benchmark/history/<date>_<sha>/` | ✅ tracked | the snapshot: `MANIFEST.md` + `benchmark-report.{md,json,csv}` + `numpy-results.json` + every subsystem `*_results.{md,tsv}` + `cards/`. |
| `benchmark/history/latest` | ✅ tracked symlink | → the newest snapshot. Stable path for docs/CI. |

`benchmark/scripts/snapshot_history.py` assembles it (called by `run_benchmark.py`; `--commit` to also git-commit).
**Publish ritual:** run → review → commit `benchmark/history/`. Reference `benchmark/history/latest/benchmark-report.md`,
never the gitignored scratch.

## The Debug-taint reminder (bears repeating)

Any ad-hoc timing script (`dotnet run file.cs` / `dotnet_run`) MUST run `dotnet run -c Release - < script.cs`, or
both the script AND `#:project` NumSharp.Core compile in Debug and hand-written C# kernels inflate ~2×. Diagnostic:
if strided/custom-kernel numbers look ~2× worse than IL-kernel numbers, check
`Assembly.GetCustomAttribute<DebuggableAttribute>().IsJITOptimizerDisabled` for both assemblies. The BenchmarkDotNet
projects are exempt (they mandate `-c Release`).
