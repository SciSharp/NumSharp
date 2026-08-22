# Running the harness & reading the report

## Official run — `benchmark/run_benchmark.py` (the entry point)

Builds the C# suite, runs each suite through BenchmarkDotNet (per-class JSON, resumable), sweeps warm NumPy across
the applicable size tiers, runs Managed/OpenBLAS profiles, merges on `(op, dtype, N, scenario)`,
appends the five complementary subsystems, and archives raw scratch to
`results/<ts>/` (gitignored), and writes the committable `history/<date>_<sha>/` snapshot.

```bash
python run_benchmark.py                       # full official run (18 suites + backend profiles + subsystems)
python run_benchmark.py --suites manipulation unary   # subset of the op matrix
python run_benchmark.py --skip-build          # reuse existing Release build
python run_benchmark.py --skip-csharp         # NumPy only
python run_benchmark.py --quick               # dev: fewer NumPy iterations
python run_benchmark.py --no-history          # don't write the history snapshot
# subsystem opt-outs: --skip-nditer --skip-layout --skip-operand --skip-cast --skip-fusion
# backend profile opt-out: --skip-openblas
```

**Cost:** the full matrix is long (µs–ms array ops × applicable dtypes/sizes × 18 suites + subsystems). For iterating
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

- **Convention is NPY/NS** (NumPy_ms / NumSharp_ms, `>1` = NumSharp faster). Published bands: ✅ `≥1.05×` 🟡 `≥0.5×`
  🟠 `≥0.2×` 🔴 `<0.2×` · **▫ negligible** (sub-µs either side or >20× — excluded from geomeans & Best/Worst)
  · **⚪** (C# side unjoined). The `%NumPy🕐` column = NumSharp_ms / NumPy_ms × 100 = share of NumPy's time
  NumSharp uses (<100% = faster).
- **Credibility gating** (`merge-results.py` `classify()`): only rows where **both sides did ≥1µs of work AND the
  speedup is within 20×** count toward the geomeans and rankings. Sub-µs call-overhead rows, view returns, lazy
  allocs and dead-code-eliminated kernels are `▫ negligible` — kept in the per-suite tables, never showcased.
- The report has the full **per-(op, dtype, N, scenario) matrix**, both profile results, one
  fastest-valid effective value, then the five appended subsystem sections.
- A row missing a C# or NumPy value ("C# not run" / "NumPy only") almost always means the two names didn't
  **normalize to the same join key** — check the C# `[Benchmark(Description)]` vs the NumPy `.name`.

## Reports & UI surfaces (canonical → human-facing)

They drift — know which is which:
- **`benchmark/benchmark-report.md`** — the canonical backend-aware report. Tracked; refreshed by CI. Start here.
- **`benchmark-report.{managed,openblas}.json`** — separate profiles using the same schema; the unsuffixed
  JSON contains both profiles plus the effective selection.
- **`benchmark/history/latest/*`** — the committable snapshot the docs/CI reference.
- **`benchmark/benchmark-dashboard.md`** — a dense ASCII-bar sheet from `scripts/render_dashboard.py`. Gitignored,
  **NOT** wired into `run_benchmark.py` or CI — run it by hand to seed the DocFX dashboard's numbers.
- **`docs/website-src/docs/benchmarks-dashboard.md`** — the **real UI**, promoted from the backend POC. It reads
  only the canonical combined JSON and computes effective rollups/backend drill-downs from measured rows.
- **`benchmark/README.md`** is a static orientation guide, **not** the report — CI never refreshes it.

## History snapshots — what we commit

| Path | Tracked? | Contents |
|------|----------|----------|
| `benchmark/results/<ts>/` | ❌ gitignored | raw per-run scratch (per-suite NumPy JSON, BDN per-class reports, merged json/csv). |
| `benchmark/history/<date>_<sha>/` | ✅ tracked | the snapshot: MANIFEST + combined/separate profile JSON + report/csv + NumPy input + subsystem results + cards. |
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
