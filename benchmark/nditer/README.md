# NDIter canonical benchmark

The single, maintained NumSharp-vs-NumPy benchmark for **NDIter** (the
NumPy-aligned multi-operand iterator). It consolidates every aspect probed
during development — construction, traversal, reductions, selection, dtypes,
pathologies, dividends — swept across cache tiers, and the orchestrator renders
it all into **one results sheet** (`nditer_results.md`).

## Run it

```bash
# full run (build + all sections + sheet)
python benchmark/nditer/nditer_sheet.py

# reuse an existing Release build of NumSharp.Core
python benchmark/nditer/nditer_sheet.py --skip-build

# re-render the last run's sheet without re-measuring
python benchmark/nditer/nditer_sheet.py --render-only

# only some sections (dev loop)
python benchmark/nditer/nditer_sheet.py --sections elementwise pathology
```

Outputs (committed artifacts):
- `nditer_results.md` — the rendered sheet (per-tier / per-category / per-family
  operation matrix, construction vs `np.nditer`, chunk-width dispatch, pathology
  canaries, NumSharp-only dividends).
- `nditer_results.tsv` — the raw `id, ns_ms, np_ms` pairs (feed `--render-only`).

## Why an orchestrator instead of one script

`nditer_bench.cs` is **section-addressable** via the `NUMSHARP_BENCH_NDITER_SECTION` env var.
The orchestrator runs each section in its own short-lived `dotnet run` process
and **retries up to 4× on a crash**. This exists because the full mixed-family
run intermittently hits an uncatchable `AccessViolation` under heavy
alloc/free + GC pressure (≈50% of monolithic runs died). Per-section processes
shrink the crash surface and isolate any one failure so the sheet always
completes. *That crash is a real NumSharp memory-safety bug — see Findings.*

## Methodology (do not regress)

- **`dotnet run -c Release - < nditer_bench.cs` only.** File-based apps build
  Debug by default, which silently invalidates hand-written C# kernels (~2×).
  Both `.cs` and the orchestrator assert `IsJITOptimizerDisabled == false`.
- **Iterator-isolation rows** (elementwise, chunk-width, dividends, construction)
  drive `NDIterRef` directly with **trivial kernels matched to NumPy's loop
  family** (memcpy / V256 add / V256 sqrt / scalar sin) so the measured time is
  the *iterator's* cost, not the kernel's.
- **Production rows** (reductions, selection, copy/cast, index-math, dtypes,
  pathology) call `np.*` on both sides — the honest API-vs-API comparison.
- `copy` compares to `np.positive` (a real ufunc nditer), **never** `np.copyto`
  (a stripped raw-array walker, not nditer).
- best-of-rounds timing; correctness is checked before timing every row.
- **The orchestrator runs the C# side with `DOTNET_TC_CallCountingDelayMs=0`.** Tiered
  compilation promotes a method to tier-1 only after a 100 ms window with no new tier-0
  JIT activity, and a benchmark process keeps jitting new kernels/routes for most of a
  200 ms row — so the first rows of each section were measured at tier-0 (2026-09-03:
  `lessbool@1` 484 ns default vs 124 ns with the delay at 0, `astype@1` 755 vs 346;
  the same call read 130 ns as a later row of the same process). Set it too when running
  `nditer_bench.cs` by hand, or the scalar/1K tiers read 2–3× pessimistic.
- `speedup = NumPy_time / NumSharp_time` → **> 1.0 means NumSharp is faster.**

## Sections / aspects covered

| Section | Rows | What it measures |
|---|---|---|
| `elementwise` | add, sqrt, copy, strided, bcast, reversed, castbuf, mixbuf × 4 tiers | raw-iterator elementwise throughput (V256 + stride-0 + buffered cast) |
| `reductions` | sum, sum-ax0/ax1, sum-dt=, amin, cumsum, any-allfalse, any-earlyhit × 4 | full/axis reductions, scans, early-exit |
| `selection` | where, a[mask] r/w, count_nonzero, argwhere, a[idx] gather/scatter × 4 | boolean subscript, fancy index, where |
| `copycast` | flatten, astype, ravel.T, in-place, less→bool × 4 | CopyAsFlat consumers + comparison-to-bool |
| `indexmath` | unravel_index, ravel_multi_index × 4 | compiled_base index math |
| `dtypes` | complex128, float16, int8 add × 4 | kernel-bound dtypes riding the iterator |
| `construction` | 9 flag configs (1op…8d, ufunc, bufcast, multiindex) | iterator build+dispose vs `np.nditer` ctor |
| `chunkwidth` | inner widths 4/16/64/256/1024 | per-chunk dispatch overhead scaling |
| `pathology` | bcast-reduce, allocate, overlap-copy, F-order-out, 0-d | regression canaries (known taxes/losses) |
| `dividends` | fuse7, reuse, par8 × 4 | NumSharp-only machinery NumPy can't match |

Tiers: `scalar`=1 · `1K`=1 000 · `100K`=100 000 · `1M`=1 000 000.

## Probes (`probes/`)

Three `dotnet run` file-based apps with a RELATIVE `#:project`, plus their NumPy twin, for
measuring the iterator outside the sheet: `fixed_cost_probe.cs` (construction, per-chunk,
1M/10M attribution, the production `out=` routes with managed bytes per call, the glue
components), `ab_ops_probe.cs` (48 NDIter-routed ops at 1K/100K, for A/B against a worktree
of a baseline commit), `angles_probe.cs` (the candidate next levers as experiments) and
`numpy_twins.py fixed|ab|angles|join`. The cost model, the traps and every number they
produced on 2026-09-03 are in **`docs/NDITER_PERF_DISCOVERY.md`**.

## Findings ledger (kept current with each run)

Tracked regressions surfaced by this benchmark (NumSharp slower than NumPy):

1. **Intermittent ~50% segfault** under heavy mixed load — uncatchable AV,
   GC/finalizer race on unmanaged storage. *Worked around by section isolation;
   still the top bug to fix.*
2. **`np.any` full-scan** (all-false) — scalar scan, no SIMD: up to ~12× slower
   at 1M, while it wins 6–24× at small N and on early-exit. (`anyff`)
3. **comparison→bool** (`np.less(out=bool)`) — *resolved 2026-09-03*: the `out=` route
   compiled a scalar-only inner loop for every layout; it now runs the whole-array SIMD
   comparison kernel through the iterator when the inputs share a dtype and the bool out is
   contiguous in iteration order (`NDIterRef.TryExecuteComparison`). 100K: 43.4 µs → 9.3 µs
   (NumPy 11.4). Mixed dtypes / cast out / strided-F out keep the scalar body. (`lessbool`)
4. **fancy gather/scatter** (`a[idx]`) — MapIter path 1.2–3.4× slower. (`gather`/`scatter`)
5. **`amin` axis-reduce** — 2.3–2.4× slower at 100K+ (lags the `sum` axis kernel).
6. **broadcast-view reduce** — `path.bcast_reduce` ~50× slower (general
   coordinate walk instead of materialize-then-reduce).
7. **F-order out** — `path.forder_out` ~4× slower (order-resolution copy).
8. **ALLOCATE out** — `path.allocate` ~2× slower (NumSharp zeros via `np.zeros`;
   NumPy allocates `empty`).
9. small-N copy/cast & index-math (flatten/ravel/astype/unravel) lose to per-call
   setup overhead; cross to wins by 1M. *Partly addressed 2026-09-03*: the iterator's own
   fixed cost fell from 146–197 ns to 62–113 ns (one recycled native block per iterator
   instead of three calloc/free pairs — `NDIterRef.AllocateStateBlock`), the ufunc routes
   stopped building an interpolated kernel-cache key per call (`InnerLoopKernelKey`,
   −45 ns and −96 B garbage), the unbuffered EXTERNAL_LOOP driver advances with a direct
   inlined call (per-chunk driver overhead 5.6 → 4.7 ns), the `out=` routes skip the
   identity broadcast when every operand has the same dims (−55 ns), and the
   COPY_IF_OVERLAP temp is disposed eagerly instead of leaking to the finalizer
   (`inplace@1` 282 → 154 ns). Net at n=1 (ns, NumSharp → NumSharp | NumPy): `add out`
   411 → 168 | 289, `less out` 366 → 140 | 300, `sqrt out` 338 → 183 | 258. What is left
   there is the kernel-selection floor, not iterator overhead.
10. **Do not trust the 2026-08-29 sheet's 1M/10M elementwise cells** (`add@10M` 0.33×,
   `sqrt@10M` 0.23×, `mixbuf@10M` 0.50×): the section re-run in isolation reads
   add@10M 8.64 ms vs NumPy 7.36 (0.85×), sqrt 6.64 vs 6.46 (0.97×), mixbuf 8.9 vs 9.9 —
   parity. Both sides' committed 1M/10M numbers were 2–5× slower than a fresh run
   (host contamination at snapshot time). Re-run a section alone before chasing a cell.

Confirmed strengths: construction (~2.8× vs `np.nditer`), reductions (sum/
count_nonzero 2–4×), buffered cast, int8 (~7×, verified correct), and the
dividends (fusion / iterator-reuse / parallel banding) NumPy structurally lacks.
