# POC — NpyIter-driven execution at NumPy parity or better

**Date:** 2026-06-09 · **Machine:** AVX2 (no AVX-512, neither in NumPy's dispatch) · **NumPy:** 2.4.2 · **Branch:** `nditer`

## Claim proven

Driving every operation through the **real `NpyIterRef` machinery** (`MultiNew` → `ForEach` / `ExecuteUnary` / `ExecuteBinary` / `ExecuteExpression`), NumSharp reaches NumPy performance on contiguous and near-parity on strided layouts, and is **2.1–4.6× FASTER than NumPy on fused expressions** — the structural advantage NumPy cannot match without numexpr, unlocked by the Tier-3C `NpyExpr` DSL compiling an expression tree into a single inner loop.

## Results

| | aspect | NumSharp / NpyIter | NumPy 2.4.2 | ratio | verdict |
|---|---|--:|--:|--:|---|
| A | contiguous `sqrt` f32 10M | 3.43 ms | 3.16 ms | 1.09× | ≈ parity |
| B | contiguous `add` f32 10M | 4.05 ms | 3.98 ms | 1.02× | **parity** |
| C | strided `a[::2]+b[::2]` f32 1M | 789 µs | 399 µs | 1.98× | gap (see notes) |
| D | strided 2-D `sqrt(a[::2,::2])` f32 1M | 470 µs | 368 µs | 1.28× | near parity |
| E | strided `sum(a[::2])` f32 1M | 368 µs | 221 µs | 1.66× | gap (see notes) |
| F | **fused `a*b+c`** f32 10M | **5.79 ms** | 12.30 ms | **0.47×** | **2.1× faster** |
| G | **fused `(a-b)/(a+b)`** f32 10M | **4.56 ms** | 20.92 ms | **0.22×** | **4.6× faster** |
| H | small-N `sqrt` f32 1K, per call incl. iterator construction | 0.69 µs | 0.42 µs | +0.27 µs | parity-class |

All aspects are correctness-verified in-script against independent references before timing.
For context: NumSharp's *eager* `a*b+c` is 16.8 ms and `(a-b)/(a+b)` is 24.5 ms — fusion via
`ExecuteExpression` also beats our own eager path 2.9–5.4×.

## Methodology

- Outputs **preallocated on both sides** (NumPy uses `out=`) so the numbers compare the
  *execution architecture*, not the allocators. A fresh 4 MB `np.empty` per call costs
  ~0.3–0.4 ms in soft page faults on .NET (frees are GC-finalizer-deferred so pages stay
  cold), while CPython's refcounting frees the previous result immediately and NumPy's
  allocator hands back warm pages. With allocation included, every .NET number above gains
  that constant — an allocator/lifetime concern, orthogonal to NpyIter.
- Fusion aspects let the eager side allocate its intermediate temporaries — eliminating
  those is precisely what fusion is.
- The strided kernels (C/D/E) are POC implementations of the **Phase 2a fused-gather inner
  loop** (`NpyInnerLoopFunc` contract — NumPy's `PyUFuncGenericFunction`), supplied to
  `NpyIterRef.ForEach`. The iterator does the orchestration (coalescing, EXTERNAL_LOOP
  chunking, strides); the kernel does one chunk. This is the production Phase 2a/3 shape.
- In-process `Stopwatch` loops vs Python `perf_counter` loops, warmup excluded, same shapes
  and dtypes, single process per side, interleaved variant comparisons where kernels were
  selected.

## Findings (measured, with dead ends recorded)

1. **The fused-gather (raw-pointer `Vector256.Create` from strided lanes) is the best
   strided technique on this stack** — confirming handover §4.3 a third time. Both
   compaction variants lose ~2× to it in interleaved A/B despite fewer theoretical uops:
   `vpermd`-based (1423 µs vs 738 µs for C-shape) and GCC-style `vshufps`+`vpermpd`
   (1818 µs). .NET scalar loops also lose (1013 µs plain, 766 µs 4-unrolled).
2. **The residual C/E gap is NumPy's compiler-generated stride-2 compaction**, which RyuJIT
   does not reproduce efficiently (alignment ruled out: NumSharp unmanaged buffers are
   64-byte aligned). On cheap ops over L3-resident views NumPy's loops run at ~1–2
   cycles/element; our gather at ~3.5. Bounded gap (≤2×), worst-case shapes only —
   contiguous and compute-heavy strided ops are at parity.
3. **`ExecuteExpression` requires `EXTERNAL_LOOP`** to coalesce contiguous operands into
   one inner-loop call; without it, `ForEach` invokes the kernel per element
   (10M delegate calls ≈ 22 ns each → 221 ms instead of 5.8 ms). The production wrapper
   should set it unconditionally for element-wise execution.
4. **Iterator construction is not a blocker for small-N**: full `MultiNew` + execute +
   dispose is 0.69 µs/call vs NumPy's 0.42 µs/call ufunc dispatch — already parity-class;
   Phase 1's trivial-constructor would close the remaining 0.27 µs.
5. **Output allocation dominates mid-size benchmarks on .NET** (finding #1 in Methodology).
   Worth a dedicated look (buffer pooling / eager free on Dispose) independent of NpyIter.

## What this validates for the migration plan

- **Phase 2a**: the per-chunk fused-gather kernel driven by `ForEach` is exactly the
  production design — measured here at 1.0–1.3× NumPy on strided unary (D), with the shell
  (`EmitFusedStridedSimdLoop`) needing only to emit what `PocKernels` hand-writes.
- **Phase 3**: one driver works. The same iterator drove Direct-kernel execution (A/B),
  custom per-chunk kernels (C/D/E), and compiled expression trees (F/G) without special
  cases.
- **Tier-3C (§14 north star)**: fusion is no longer hypothetical — `NpyExpr` through
  `CompileInnerLoop` delivers 2.1–4.6× over NumPy today on composite expressions, the
  payoff that justifies the migration.

## Reproduce

```bash
dotnet_run < benchmark/poc/npyiter_parity_poc.cs     # NumSharp/NpyIter side
python benchmark/poc/npyiter_parity_poc.py            # NumPy side
```
