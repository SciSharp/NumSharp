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
  iteration for nanosecond microbenchmarks; for µs–ms array ops that would make a single 10M case take tens of
  seconds and the full matrix take days. Capping iteration time lets the pilot pick a per-op invocation count that
  fits 25 ms while preserving all 50 measured iterations — a large wall-clock saving at the same rigor.

The nditer subsystem reports a section that crashes all retries (the known intermittent `AccessViolation`) as
**NA/IGNORED**, never a failure.

## Reading the report

- **Convention is NPY/NS** (NumPy_ms / NumSharp_ms, `>1` = NumSharp faster). The **status icon** on each row is
  assigned by `get_status()` (the faster→slower cutoffs) + `classify()` (the credibility gate) in
  `scripts/merge-results.py` — read those for the exact numeric thresholds, which are tuned there rather than
  pinned in this doc:
  - **✅ faster** · **🟡 near-parity** · **🟠 slower** (optimization target) · **🔴 much slower** (priority fix)
    — descending ratio bands.
  - **▫ negligible** — a *non-throughput* cell: a semantic O(1)-in-N scenario, sub-µs work on either side, or an
    implausible speedup above the credible cap. Kept in the raw per-suite tables but **excluded from every geomean
    and ranking**.
  - **⚪ pending** — no joined C# row at this (op, dtype, N); the merge found no match.
  - **❌ failed** — the C# benchmark *ran but crashed / OOM'd* (no measurement). Distinct from ⚪ (which never ran).
  The `%NumPy🕐` column = NumSharp_ms / NumPy_ms × 100 = share of NumPy's time NumSharp uses (<100% = faster).
  (Note the legend the report *prints* is decorative and can lag the classifier; the two `merge-results.py`
  functions are authoritative.)
- **Credibility gating** (`merge-results.py` `classify()` + `scripts/credibility.py`): a row is a believable
  throughput comparison only when the scenario is not semantically O(1) in element count, **both sides did at
  least `WORK_FLOOR_MS` (1µs) of work**, and the speedup is within `MAX_CREDIBLE_SPEEDUP` (20×). Anything else is
  ▫ negligible. The reviewed O(1)-in-N view/metadata/fixed-count-wrapper set lives in `credibility.py`; the proof
  ledger is `benchmark/O1_EXCLUSIONS.md`. Raw rows stay inspectable and are never showcased.
- The report has the full **per-(op, dtype, N, scenario) matrix**, both backend profiles, one fastest-valid
  effective value, then the appended subsystem sections (NDIter + layout / operand / cast / fusion + backend).
- A row missing a C# or NumPy value ("C# not run" / "NumPy only") almost always means the two names didn't
  **normalize to the same join key** — check the C# `[Benchmark(Description)]` against the NumPy `.name`, or run
  `scripts/check_smoke_joins.py` (below) which does it structurally.

## Source-level checks (no timing run)

Two cheap audits answer "is the wiring right?" without executing a benchmark:

- **`scripts/audit_coverage.py`** — the coverage audit. Cross-references the compiled NumPy API inventory
  (`coverage/generated/coverage.json`) against the `[Benchmark(Description)]` attributes in the op-matrix
  namespaces plus the reviewed `coverage/overrides.json`, and (re)writes `coverage/generated/summary.md` +
  `coverage.{json,csv}`. Read `summary.md` to see headline coverage, the **Missing benchmark coverage** to-do
  table, the reviewed exclusions, the OpenBLAS route map (which APIs are managed / optional-with-fallback /
  required-LAPACK), and any **Unmatched benchmark descriptions** (a stale/renamed label that no longer maps to an
  API row). The counts there are generated — treat that file as the source of truth rather than memorizing a
  number.
- **`scripts/check_smoke_joins.py`** — the join checker. From a quick NumPy smoke run
  (`numpy_benchmark.py --quick --size small --output benchmark/results/smoke/<suite>.json` per suite), it verifies
  every C# `[Benchmark(Description)]` ↔ NumPy `.name` join in **both directions** using the exact merge
  normalizer, ignoring dtype/size cells. This is the fast way to catch a "C# not run" / "NumPy only" row before a
  full measured run.

The merge / backend-profile / O(1)-exclusion / universal-tier / snapshot logic is itself locked by
`scripts/tests/test_*.py` (e.g. `test_merge_backend_profiles.py`, `test_o1_exclusions.py`,
`test_universal_tier_coverage.py`, `test_snapshot_history.py`) — run those after changing a script under
`scripts/`.

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

## Dashboard data delivery — the `data` branch (submodule)

The three live docs dashboards fetch same-origin JSON that DocFX bakes at build time. That data lives
**only** on the orphan **`data`** branch (renamed from `master-code-data`); `master` is code-only and
keeps no committed copy. The docs build mounts the branch as a git submodule and reads it directly:

- **Orphan `data` branch** — generated data as `<type>/<date>_<sha>/` snapshots plus a **real `latest/`
  directory** (a data-only copy of the newest snapshot, no README), one folder per type: `benchmark`,
  `tests-oracle`, `inventory` (NumPy API coverage), `benchmark-coverage`. The branch's top-level +
  per-type READMEs are the authoritative spec.
- **`refs/data` submodule** — declared in `.gitmodules` (`branch = data`, shallow), gitlink kept as
  provenance only (never bumped). At docs-build the workflow floats it to the branch tip by a **direct
  shallow clone** (`git clone --depth 1 -b data … refs/data` — `submodule update --remote` can't resolve
  the un-bumped gitlink, and origin/data has already advanced that run), and `docfx.json` reads each
  dataset straight from `refs/data/<type>/latest/`. No date-priority resolver and no master-side
  fallback — the branch tip is always what the site builds from.

Publisher lives on the code branch in **`tools/dashboard_data/`** (stdlib-only):
- `publish.py --type <t> --from <dir> --branch-worktree <wt> --sha <sha> --commit` — append a
  `<date>_<sha>` snapshot for a type and refresh its real `latest/` directory.
- `common.py` — the per-type file map + snapshot/`latest` helpers.

CI: `benchmark.yml` publishes the fresh `benchmark` snapshot to `data` after a run (then triggers a docs
redeploy); `docs.yml` publishes `inventory`/`tests-oracle`/`benchmark-coverage` to `data` on master
pushes, and its build job inits the `refs/data` submodule (floated to tip) before `docfx build`.

## History snapshots — build outputs, published to the `data` branch

| Path | Tracked on master? | Contents |
|------|--------------------|----------|
| `benchmark/results/<ts>/` | ❌ gitignored | raw per-run scratch (per-suite NumPy JSON, BDN per-class reports, merged json/csv). |
| `benchmark/history/<date>_<sha>/` | ❌ gitignored | the snapshot: MANIFEST + combined/separate profile JSON + report/csv + NumPy input + subsystem results + cards. Published to the `data` branch. |
| `benchmark/history/latest` | ❌ gitignored symlink | → the newest local snapshot; the durable copy is `benchmark/latest/` on the `data` branch. |

`benchmark/scripts/snapshot_history.py` assembles it (called by `run_benchmark.py`). **Publishing to the
`data` branch is automatic:** an eligible full `run_benchmark.py` then publishes the snapshot straight into
the **`refs/data`** submodule — pull `data` to its tip, override `benchmark/latest/`, commit (locally, on
branch `data`) — unless `--no-data-publish` is passed; push `refs/data` when ready (CI `benchmark.yml` pushes
on its own path). The other three kinds get the same flow via
`python tools/dashboard_data/refresh_data.py [--type <t>|all] [--push]` (regenerate + `publish.py --pull
--commit` into `refs/data`). master carries NO data — reference `benchmark/history/latest/benchmark-report.md`
locally, and `refs/data/benchmark/latest/…` on the site (see *Dashboard data delivery* above).

## The Debug-taint reminder (bears repeating)

Any ad-hoc timing script (`dotnet run file.cs` / `dotnet_run`) MUST run `dotnet run -c Release - < script.cs`, or
both the script AND `#:project` NumSharp.Core compile in Debug and hand-written C# kernels inflate ~2×. Diagnostic:
if strided/custom-kernel numbers look ~2× worse than IL-kernel numbers, check
`Assembly.GetCustomAttribute<DebuggableAttribute>().IsJITOptimizerDisabled` for both assemblies. The BenchmarkDotNet
projects are exempt (they mandate `-c Release`).
