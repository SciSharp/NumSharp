# POC — NpyIter-driven execution at NumPy parity or better

**Date:** 2026-06-10 · **Machine:** i9-13900K, AVX2 (no AVX-512, neither in NumPy's dispatch) · **NumPy:** 2.4.2 · **Branch:** `nditer`

## Claim proven

Driving every operation through the **real `NpyIterRef` machinery** (`MultiNew` → `ForEach` / `ExecuteUnary` / `ExecuteBinary` / `ExecuteExpression`), NumSharp is **at or faster than NumPy on every measured aspect** — contiguous, strided, reduction, small-N dispatch — and **2.8–5.4× faster on fused expressions**, the structural advantage NumPy cannot match without numexpr, unlocked by the Tier-3C `NpyExpr` DSL compiling an expression tree into a single inner loop.

## Results (both sides measured back-to-back, same conditions)

| | aspect | NumSharp / NpyIter | NumPy 2.4.2 | ratio | verdict |
|---|---|--:|--:|--:|---|
| A | contiguous `sqrt` f32 10M | 2.98 ms | 3.24 ms | 0.92× | **faster** |
| B | contiguous `add` f32 10M | 3.91 ms | 4.09 ms | 0.96× | **faster** |
| C | strided `a[::2]+b[::2]` f32 1M | 319 µs | 416 µs | 0.77× | **1.30× faster** |
| D | strided 2-D `sqrt(a[::2,::2])` f32 1M | 206 µs | 374 µs | 0.55× | **1.82× faster** |
| E | strided `sum(a[::2])` f32 1M | 109 µs | 205 µs | 0.53× | **1.88× faster** |
| F | **fused `a*b+c`** f32 10M | **4.77 ms** | 13.38 ms | 0.36× | **2.8× faster** |
| G | **fused `(a-b)/(a+b)`** f32 10M | **4.12 ms** | 22.33 ms | 0.18× | **5.4× faster** |
| H | small-N `sqrt` f32 1K, per call incl. iterator construction | 0.40 µs | 0.44 µs | 0.91× | **faster** |

All aspects are correctness-verified in-script against independent references before timing.
For context: NumSharp's *eager* `a*b+c` is 17.1 ms and `(a-b)/(a+b)` is 27.5 ms — fusion via
`ExecuteExpression` also beats our own eager path 3.6–6.7×.

## The strided kernels (C/D/E): AVX2 hardware gather

Ground truth from NumPy's source (in-repo `src/numpy/`):

| aspect | NumPy's inner loop | file |
|---|---|---|
| C strided binary | **plain scalar C loop** — no SIMD path exists for non-unit strides | `loops_arithm_fp.dispatch.c.src` (`goto loop_scalar`) |
| D strided unary | `_mm256_i32gather_ps` hardware gather, 4× unrolled | `loops_unary_fp.dispatch.c.src` + `simd/avx2/memory.h` (`npyv_loadn_f32`) |
| E strided reduce | **scalar 8-accumulator** pairwise loop — no SIMD for non-unit strides | `loops_utils.h.src` (`pairwise_sum`) |

The POC kernels use `Avx2.GatherVector256` (vgatherdps, scale=1, **byte-offset index
vector hoisted out of the loop** — the hot loop is stride-agnostic) for all three:
matching NumPy's technique where NumPy has one (D), and overtaking NumPy where it
only has scalar loops (C: 2× unrolled gathers; E: 4 independent vector accumulators).
Software insert-gather (`Vector256.Create` of 8 strided loads) is the guarded fallback
(`!Avx2.IsSupported` or `|7·stride| > int.MaxValue`).

Interleaved Release A/B of all candidate techniques (1M f32, runtime strides, median of 9 rounds):

| technique | C add | D sqrt | E sum | notes |
|---|--:|--:|--:|---|
| **hardware gather** (chosen) | **334** | **203** | **121** | stride-general, index hoisted |
| software insert-gather | 387 | 215 | 167 | fallback |
| `vshufps`+`vpermpd` compaction | 314 | — | — | stride-2-only; marginal win not worth the specialization |
| `vpermd` compaction | 314 | — | — | stride-2-only |
| scalar 8× unrolled | 362 | — | 133 | what NumPy effectively runs for C/E |
| stride-2 mask trick | — | — | 98 | sum-only, stride-2-only |

## Methodology

- Outputs **preallocated on both sides** (NumPy uses `out=`) so the numbers compare the
  *execution architecture*, not the allocators. A fresh 4 MB `np.empty` per call costs
  ~0.3–0.4 ms in soft page faults on .NET (frees are GC-finalizer-deferred so pages stay
  cold), while CPython's refcounting frees the previous result immediately and NumPy's
  allocator hands back warm pages. With allocation included, every .NET number above gains
  that constant — an allocator/lifetime concern, orthogonal to NpyIter.
- Fusion aspects let the eager side allocate its intermediate temporaries — eliminating
  those is precisely what fusion is.
- The strided kernels (C/D/E) are POC implementations of the **Phase 2a inner loop**
  (`NpyInnerLoopFunc` contract — NumPy's `PyUFuncGenericFunction`), supplied to
  `NpyIterRef.ForEach`. The iterator does the orchestration (coalescing, EXTERNAL_LOOP
  chunking, strides); the kernel does one chunk. This is the production Phase 2a/3 shape.
- In-process `Stopwatch` loops vs Python `perf_counter` loops, warmup excluded, same shapes
  and dtypes, single process per side, interleaved variant comparisons where kernels were
  selected.
- **MUST run via `dotnet run -c Release`** — the script asserts the JIT optimizer is
  enabled on both assemblies at startup (see finding 1).

## Findings (measured, with dead ends recorded)

1. **`dotnet run` file-based apps build Debug by default — script AND `#:project`-referenced
   NumSharp.Core** (`DebuggableAttribute(DisableOptimizations)`; the JIT honors it even over
   `AggressiveOptimization`). This silently doubled the hand-written strided kernel times in
   the original POC (C 789→319 µs, D 470→206, E 368→109 once fixed) while leaving
   `DynamicMethod`-emitted kernels (A/B/F/G) untouched — emitted IL is always JIT-optimized,
   which is why only C/D/E looked slow and the cause survived scheduling/GC/stride-codegen
   investigations. `#:property Optimize=true` fixes only the script; **only command-line
   `-c Release` fixes both**. The script now refuses to mislead: it checks
   `IsJITOptimizerDisabled` on both assemblies at startup.
2. **An earlier conclusion is corrected**: "stride-2 compaction loses 2× to insert-gather"
   was a Debug-JIT artifact. Under Release, compaction (314 µs) marginally *beats* gather
   variants on stride-2 — but hardware gather (334 µs general, any stride) wins on
   generality and is within 6%. Hardware gather stands as the production technique.
3. **NumPy leaves strided performance on the table**, confirmed in source: strided binary
   ops and strided reductions have **no SIMD path at all** (scalar loops); only strided
   unary ops use hardware gather. With fast gather hardware (Golden/Raptor Cove, Zen 5),
   driving all three through `vpgatherdd` beats NumPy 1.3–1.9×. (On gather-slow cores —
   Zen 2/3, pre-Skylake — the software insert-gather fallback applies.)
4. **`ExecuteExpression` requires `EXTERNAL_LOOP`** to coalesce contiguous operands into
   one inner-loop call; without it, `ForEach` invokes the kernel per element
   (10M delegate calls ≈ 22 ns each → 221 ms instead of ~5 ms). The production wrapper
   should set it unconditionally for element-wise execution.
5. **Iterator construction is cheap in Release**: full `MultiNew` + execute + dispose is
   0.40 µs/call vs NumPy's 0.44 µs/call ufunc dispatch — small-N is **won**, before
   Phase 1's trivial-constructor work.
6. **Output allocation dominates mid-size benchmarks on .NET** (see Methodology).
   Worth a dedicated look (buffer pooling / eager free on Dispose) independent of NpyIter.

## What this validates for the migration plan

- **Phase 2a**: the per-chunk hardware-gather kernel driven by `ForEach` is exactly the
  production design — measured at 0.5–0.8× NumPy (faster) on strided shapes. The
  emission shell (`EmitFusedStridedSimdLoop`) should emit the gather pattern for
  gather-capable dtypes (f32/f64/i32/i64 — `GATHER_ELIGIBLE`) and insert-gather otherwise.
  Emitted `DynamicMethod` IL is immune to the Debug-build pitfall by construction.
- **Phase 3**: one driver works. The same iterator drove Direct-kernel execution (A/B),
  custom per-chunk kernels (C/D/E), and compiled expression trees (F/G) without special
  cases.
- **Tier-3C (§14 north star)**: fusion is no longer hypothetical — `NpyExpr` through
  `CompileInnerLoop` delivers 2.8–5.4× over NumPy today on composite expressions, the
  payoff that justifies the migration.

## Reproduce

```bash
dotnet run -c Release - < benchmark/poc/npyiter_parity_poc.cs   # NumSharp/NpyIter side
python benchmark/poc/npyiter_parity_poc.py                      # NumPy side
```
