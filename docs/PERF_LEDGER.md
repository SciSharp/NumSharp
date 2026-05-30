# NumSharp ↔ NumPy 2.4.2 — Performance Ledger

Per the `/np-function` mission, every SIMD-capable `(op, dtype, layout)` must be **≥1.5× NumPy** or
have a **documented reason**. Methodology: warm, min-of-N timing, same operand bytes both sides
(NumSharp via `dotnet run` net8.0 Debug-config kernels JIT-warmed; NumPy via `perf_counter`). These
are coarse baselines that size the Phase 5 optimization work — not committed regression gates yet.

Classification: **≥1.5×** (mission met) · **parity** (0.67–1.5×) · **laggard** (<0.67×, needs work or
a documented reason).

---

## matmul / dot (T8)

**Correctness:** ✅ bit-exact vs NumPy across the full gufunc shape space (408-case differential
matrix, `matmul.jsonl`, CI-gated). See commit `dcb9cfa3`.

**Performance (float64, square `N×N`, single-thread):**

| N | NumPy (OpenBLAS) | NumSharp | Ratio | Class |
|---|------------------|----------|-------|-------|
| 64 | 0.010 ms | 0.157 ms | 0.06× | laggard |
| 128 | 0.075 ms | 1.25 ms | 0.06× | laggard |
| 256 | 0.370 ms | 9.9 ms | 0.04× | laggard |
| 512 | 1.33 ms | 77.7 ms | 0.017× | laggard |

**Reason (documented):** NumPy delegates matmul to **OpenBLAS/MKL** — multi-threaded, cache-tiled,
hand-tuned microkernels. NumSharp's pure-C# BLIS-style SIMD GEMM (`SimdMatMul`) is single-threaded
and not cache-blocked, so the gap *grows* with `N` (16× at 64 → 58× at 512 — the signature of an
O(N³) kernel thrashing cache once the working set exceeds L2).

**Phase 5 optimization target:** cache tiling (L1/L2 micro/macro-kernel blocking, BLIS packing),
multi-threading the outer block loop, and microkernel register-tiling. Reaching ≥1.5× of MKL in pure
managed code is **uncertain** (MKL is decades of hand-tuned assembly); a realistic interim goal is
parity for small/medium `N` and closing the large-`N` cache cliff. Tracked for Phase 5.
