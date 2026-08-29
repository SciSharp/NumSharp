# Running the harness & reading the report

## Official run — `benchmark/run_benchmark.py` (the entry point)

Builds the C# suite, runs each suite through BenchmarkDotNet (per-class JSON, resumable), sweeps warm NumPy across
the applicable size tiers, runs Managed/OpenBLAS profiles, merges on `(op, dtype, N, scenario)`,
appends the five complementary subsystems, and archives raw scratch to
`results/<ts>/` (gitignored), and writes the committable `history/<date>_<sha>/` snapshot.

```bash
python run_benchmark.py                       # interactive depth + dtype picker
python run_benchmark.py --depth measure       # full official run (18 suites + backend profiles + subsystems)
python run_benchmark.py --suites manipulation unary   # subset of the op matrix
python run_benchmark.py --skip-build          # reuse existing Release build
python run_benchmark.py --skip-csharp         # NumPy only
python run_benchmark.py --quick               # deprecated alias for --depth light
python run_benchmark.py --no-history          # don't write the history snapshot
# subsystem opt-outs: --skip-nditer --skip-layout --skip-operand --skip-cast --skip-fusion
# backend profile opt-out: --skip-openblas
```

With no arguments in an interactive terminal, the runner opens a depth picker and then a dtype
picker. Scripts and CI should be explicit:

```bash
python benchmark/run_benchmark.py --depth pass                 # 1 call/cell, 0 warmups; execution gate
python benchmark/run_benchmark.py --depth light --dtypes f32   # rough ratio: 8/3 BDN, 1/6 NumPy budget
python benchmark/run_benchmark.py --depth measure              # full publication-quality run
python benchmark/run_benchmark.py --depth pass --dtypes f16,c128 --suites unary reduction
```

`--dtypes` accepts comma-separated NumSharp/NumPy names and short aliases. Parameterized BDN cases
are filtered before execution. A scenario with more than one dtype axis matches if the requested
dtype occurs on **any** side. Pass/light retain raw reports under `benchmark/results/<ts>/`, record
`run-config.json`, validate every BDN sample count and pass-mode NumPy call count, skip complementary
subsystem sheets, and never overwrite canonical root/docs/history artifacts. `--quick` is retained
only as a deprecated alias for `--depth light`.

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
  🟠 `≥0.2×` 🔴 `<0.2×` · **▫ negligible** (semantic O(1)-in-N, sub-µs either side, or >20× — excluded from every rollup/ranking)
  · **⚪** (C# side unjoined). The `%NumPy🕐` column = NumSharp_ms / NumPy_ms × 100 = share of NumPy's time
  NumSharp uses (<100% = faster).
- **Credibility gating** (`merge-results.py` `classify()` + `scripts/credibility.py`): only non-O(1)
  scenarios where **both sides did ≥1µs of work and the speedup is within 20×** count. Reviewed
  O(1)-in-N view/metadata/fixed-count-wrapper scenarios are negligible regardless of timing; see
  `benchmark/O1_EXCLUSIONS.md`. Raw rows stay inspectable and are never showcased.
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

## Dashboard data delivery — the `master-code-data` branch

The three live docs dashboards fetch same-origin JSON that DocFX bakes at build time. That data now has
**two sources, reconciled by date at build time** (`master` stays a backwards-compatible fallback):

- **Code branch (`master`)** — each dataset's committed copy (the historical path; still builds alone).
- **Orphan `master-code-data` branch** — generated data as `<type>/<date>_<sha>/` snapshots + a git
  symlink `latest`, one folder per type: `benchmark`, `tests-oracle`, `inventory` (NumPy API coverage),
  `benchmark-coverage`. The branch's top-level + per-type READMEs are the authoritative spec.

Tooling lives on the code branch in **`tools/dashboard_data/`** (stdlib-only):
- `publish.py --type <t> --from <dir> --branch-worktree <wt> --sha <sha> --commit` — append a
  `<date>_<sha>` snapshot for a type and repoint `latest`.
- `resolve.py --data-worktree <wt> --into .` — at docs-build time, per dataset pick the newer of
  master-vs-branch by **git commit date** and overlay it onto the paths DocFX already reads (so the UI's
  relative `data/…` fetch is unchanged). `latest` is the newest pointer; missing → max `<date>_<sha>` fallback.
- `common.py` — the per-type file/overlay map + `latest`/fallback/date helpers.

CI: `benchmark.yml` publishes the fresh `benchmark` snapshot after a run; `docs.yml` publishes
`inventory`/`tests-oracle`/`benchmark-coverage` on master pushes and runs `resolve.py` before `docfx build`.
**`docfx.json` is unchanged** — the resolver stages winners into the paths its resource/content blocks already glob.

## History snapshots — what we commit

| Path | Tracked? | Contents |
|------|----------|----------|
| `benchmark/results/<ts>/` | ❌ gitignored | raw per-run scratch (per-suite NumPy JSON, BDN per-class reports, merged json/csv). |
| `benchmark/history/<date>_<sha>/` | ✅ tracked | the snapshot: MANIFEST + combined/separate profile JSON + report/csv + NumPy input + subsystem results + cards. |
| `benchmark/history/latest` | ✅ tracked symlink | → the newest snapshot. Stable path for docs/CI. |

`benchmark/scripts/snapshot_history.py` assembles it (called by `run_benchmark.py`; `--commit` to also git-commit).
**Publish ritual:** run → review → commit `benchmark/history/`. Reference `benchmark/history/latest/benchmark-report.md`,
never the gitignored scratch. The same `benchmark/history/latest` snapshot is also published to the
`master-code-data` branch (see *Dashboard data delivery* above) so the docs resolver can serve it to the site.

## The Debug-taint reminder (bears repeating)

Any ad-hoc timing script (`dotnet run file.cs` / `dotnet_run`) MUST run `dotnet run -c Release - < script.cs`, or
both the script AND `#:project` NumSharp.Core compile in Debug and hand-written C# kernels inflate ~2×. Diagnostic:
if strided/custom-kernel numbers look ~2× worse than IL-kernel numbers, check
`Assembly.GetCustomAttribute<DebuggableAttribute>().IsJITOptimizerDisabled` for both assemblies. The BenchmarkDotNet
projects are exempt (they mandate `-c Release`).
