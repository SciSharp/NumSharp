# NpyIter canonical benchmark

The single, maintained NumSharp-vs-NumPy benchmark for **NpyIter** (the
NumPy-aligned multi-operand iterator). It supersedes the exploratory
`benchmark/poc/npyiter_*` rounds: every distinct aspect those rounds surfaced
lives here, swept across cache tiers, and the orchestrator renders it all into
**one results sheet** (`npyiter_results.md`).

## Run it

```bash
# full run (build + all sections + sheet)
python benchmark/npyiter/npyiter_sheet.py

# reuse an existing Release build of NumSharp.Core
python benchmark/npyiter/npyiter_sheet.py --skip-build

# re-render the last run's sheet without re-measuring
python benchmark/npyiter/npyiter_sheet.py --render-only

# only some sections (dev loop)
python benchmark/npyiter/npyiter_sheet.py --sections elementwise pathology
```

Outputs (committed artifacts):
- `npyiter_results.md` — the rendered sheet (per-tier / per-category / per-family
  operation matrix, construction vs `np.nditer`, chunk-width dispatch, pathology
  canaries, NumSharp-only dividends).
- `npyiter_results.tsv` — the raw `id, ns_ms, np_ms` pairs (feed `--render-only`).

## Why an orchestrator instead of one script

`npyiter_bench.cs` is **section-addressable** via the `NPYITER_SECTION` env var.
The orchestrator runs each section in its own short-lived `dotnet run` process
and **retries up to 4× on a crash**. This exists because the full mixed-family
run intermittently hits an uncatchable `AccessViolation` under heavy
alloc/free + GC pressure (≈50% of monolithic runs died). Per-section processes
shrink the crash surface and isolate any one failure so the sheet always
completes. *That crash is a real NumSharp memory-safety bug — see Findings.*

## Methodology (do not regress)

- **`dotnet run -c Release - < npyiter_bench.cs` only.** File-based apps build
  Debug by default, which silently invalidates hand-written C# kernels (~2×).
  Both `.cs` and the orchestrator assert `IsJITOptimizerDisabled == false`.
- **Iterator-isolation rows** (elementwise, chunk-width, dividends, construction)
  drive `NpyIterRef` directly with **trivial kernels matched to NumPy's loop
  family** (memcpy / V256 add / V256 sqrt / scalar sin) so the measured time is
  the *iterator's* cost, not the kernel's.
- **Production rows** (reductions, selection, copy/cast, index-math, dtypes,
  pathology) call `np.*` on both sides — the honest API-vs-API comparison.
- `copy` compares to `np.positive` (a real ufunc nditer), **never** `np.copyto`
  (a stripped raw-array walker, not nditer).
- best-of-rounds timing; correctness is checked before timing every row.
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

## Findings ledger (kept current with each run)

Tracked regressions surfaced by this benchmark (NumSharp slower than NumPy):

1. **Intermittent ~50% segfault** under heavy mixed load — uncatchable AV,
   GC/finalizer race on unmanaged storage. *Worked around by section isolation;
   still the top bug to fix.*
2. **`np.any` full-scan** (all-false) — scalar scan, no SIMD: up to ~12× slower
   at 1M, while it wins 6–24× at small N and on early-exit. (`anyff`)
3. **comparison→bool** (`np.less(out=bool)`) — 1-byte store not vectorized,
   ~1.5–2.7× slower at every tier. (`lessbool`)
4. **fancy gather/scatter** (`a[idx]`) — MapIter path 1.2–3.4× slower. (`gather`/`scatter`)
5. **`amin` axis-reduce** — 2.3–2.4× slower at 100K+ (lags the `sum` axis kernel).
6. **broadcast-view reduce** — `path.bcast_reduce` ~50× slower (general
   coordinate walk instead of materialize-then-reduce).
7. **F-order out** — `path.forder_out` ~4× slower (order-resolution copy).
8. **ALLOCATE out** — `path.allocate` ~2× slower (NumSharp zeros via `np.zeros`;
   NumPy allocates `empty`).
9. small-N copy/cast & index-math (flatten/ravel/astype/unravel) lose to per-call
   setup overhead; cross to wins by 1M.

Confirmed strengths: construction (~2.8× vs `np.nditer`), reductions (sum/
count_nonzero 2–4×), buffered cast, int8 (~7×, verified correct), and the
dividends (fusion / iterator-reuse / parallel banding) NumPy structurally lacks.
