# Catalog — every specialized path that ships in NumSharp, by kind

Copy the nearest exemplar rather than inventing a shape. Each row: the subset, the gate (paraphrased —
read the file for the exact predicate), the threshold / measured result, and the commit. Paths are
under `src/NumSharp.Core/` unless stated. Ratios are NPY/NS (>1 = NumSharp faster), single-thread
pinned. The last section lists levers that were MEASURED AND REJECTED — do not re-derive them.

## Kernel swap (a tighter compute kernel for a shape / layout / dtype subset)

| op | file · gate | threshold / measured | commit |
|----|-------------|----------------------|--------|
| gemv `dot(m,v)`, `matvec`, `inner(2d,1d)` | `Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs` `TryMatMulSimd`: `N==1 && aStride1==1 && bStride0==1`, Single/Double, `GetGemvKernel` → `Gram.cs` `GemvHelperSameType<T>` | 188→21.8 µs @512 (8.6× old); ~0.7–0.9× NumPy (memory-bound) | 4f666fc8 |
| gevm `dot(v,m)`, `vecmat` | `SimdMatMul.MatMulDouble/Float` dispatch: `M==1 && bStride1==1` → the EXISTING `SimpleStrided` (a widened condition, zero new code) | 359→21.2 µs @512; 0.9–1.85× | 4f666fc8 |
| cov / corrcoef syrk | `Statistics/np.cov.cs` `TryGramCov`: f32/f64, `Xc` contiguous & !broadcast, `1 ≤ M ≤ GramCovMaxVars (16)`, `K ≥ 1`, kernel non-null → `Gram.cs` `GramHelperSameType<T>` (upper triangle + mirror) | M=2 9.8×, 8 2.0×, 16 1.6×, 32 0.87× (loses); cov 100K 0.56→2.88× | d64f3df5 |
| diff / ediff1d stencil | `np.DiffAdjacentContiguous` → `Direct/DirectILKernelGenerator.AdjacentDiff.cs`: `n==1`, no prepend/append, last axis, contiguous & !broadcast, dtype ≠ bool, dim ≥ 2 | 1K 6×; 100K +7% and 4.1× fewer managed bytes | 6e94dbcc |
| fill_diagonal / diag(1-D) / diagflat | `Indexing/np.fill_diagonal.cs` `TryDiagonalScatter` → `Direct/…DiagWrite.cs`: byte-width-keyed 1/2/4/8/16 MOV loop; AOT → `SetData` fallback | fill_diagonal 10M 0.40→1.46×, 100K 2.4→4.8× | a13238ac |
| flat argmax/argmin f32/f64 | `Direct/…Reduction.Arg.cs`: contiguous flat float → NumPy `simd_argmax` port (4×-unrolled tournament, companion index vector, first-NaN-wins); max and min are SEPARATE methods | f32 0.15→1.06×, f64 1.01–1.65×; 10M is the bandwidth ceiling | 48f894ea |
| flat nanmin/nanmax f32/f64 | `Direct/…Masking.NaN.cs`: `Avx.Max(data, acc)` with DATA as SRC1 (a NaN lane returns acc — no mask/select), 8 accumulators, cold all-NaN rescan | 0.26→1.36× (f32 at the ~49 GB/s ceiling) | 8c09dc15 |
| mean(int64/uint64, axis=0) | `Direct/…Reduction.Axis.Widening.cs`: `(Int64\|UInt64, Double)` pairs via the split-and-magic-bias widen; `wideSeqOnly` keeps innermost/pinned axes on the exact scalar helper | 0.16→1.4–1.72× | 8c09dc15 |
| isnan / isinf / isfinite f32/f64 | `Direct/…Unary.Predicate.cs` (dispatch `DefaultEngine.UnaryOp.cs`): contiguous kernel + fused strided 1-D (Single strided pinned to 128-bit as NumPy does); result via `AsGeneric<bool>` | 0.10→1.06–1.37× @100K, 1.5–1.9× @10M; 1K = alloc floor | 7bd4f380, 4dfe7619 |
| comparison `out=` | `Backends/Iterators/NDIter.Execution.cs` `TryExecuteComparison`: same input dtype, bool `out` contiguous in iteration order → the whole-array SIMD comparison kernel | 43.4→9.3 µs @100K (NumPy 11.4) | 15154b00 |
| trace | `Indexing/np.trace.cs` `TryTraceFast`: dtype has a trace kernel (not Half/Decimal/Complex) | — | — |
| extract / compress | `Indexing/np.extract.cs` `TryExtractFast`, `np.compress.cs` `TryCompressFast`: bool mask, contiguous mask AND source | — | — |
| concatenate | `Creation/np.concatenate.cs` `TryDirectMemcpyConcat` (same dtype, !broadcast, all C-contig with C dst or all F with F dst), `TryUniformTinySlabInterleave` (every per-outer slab the same 1/2/4/8/16 bytes), `TryDirectCastConcat` | tiny-slab interleave ~10–20× over per-slab memcpy | — |
| tril / triu | `Indexing/np.tril.cs` `TryTriangleBlit`: ndim ≥ 2, unit-stride last axis (1-D → the `where(tri)` composition) | — | — |
| fill | `Manipulation/NDArray.fill.cs` `TryFillInnerContiguousRuns`: !broadcast, unit-stride inner axis of length > 1 | — | — |
| average(weights) | `Statistics/np.average.cs` `TryFusedWeightedSum`: a weighted-sum iter kernel exists for the result dtype | — | — |
| power(scalar exp) | `Backends/Default/Math/Default.Power.cs` `TryScalarExponentFastPath`: exp ∈ {0, 1, 2, 0.5, −1} AND the substitution's dtype equals the general path's | — | — |
| managed LU (blocked) | `Backends/Default/LinearAlgebra/ManagedLu.cs`: `BlockedThreshold = 256` → panel getf2 + TRSM + Schur GEMM via `SimdMatMul`; double only | n ≤ 16: 2.2–4.7×; n ≥ 256: 0.4–0.6× (managed-GEMM ceiling) | 03884ee9 |
| correlate / convolve | `Math/NDArray.SlidingDot.cs`: `SmallCorrelateMaxKernel = 11` (NumPy's `small_correlate` regime stays managed even with a backend), `InnerDotSimdMin = 32` (ramp dots go SIMD from length 32) | geomean ~4× | — |
| f32 exp/log/sin/cos/tanh, exp2 | `Utilities/NDFloatMath.{cs,Simd.cs}`, gated on hardware FMA (`NumPyFloatKernelSimdAvailable`): the five are BIT-EXACT ports of NumPy's kernels; exp2 is a ≤1-ULP double-`2^r` hybrid | exp 1.11×, log 1.28×, tanh 1.8–1.9×, exp2 0.19→1.8–3.1× | — |

## Kernel contract (the kernel owns the outer loop — prologue + driver once per block)

| op | file · gate | measured | commit |
|----|-------------|----------|--------|
| 2-D block elementwise | `Backends/Iterators/NDIter.Execution.Custom.cs` `Is2DElementwiseShape` / `TryExecute2DElementwise` → `Direct/…InnerLoop2D.cs` `ND2DElementwiseKernel`: unbuffered, EXTERNAL_LOOP, not ONEITERATION, ndim ≥ 2, no mask, output inner stride 1, each input inner stride 1 or 0; inner modes {C,S}/{CC,SC,CS}; AVX2 `vmaskmov` sub-vector rows and tails (integer masked-off lanes filled with 1); one-vector-row and scalar-only variants; per-block odometer for non-foldable leading axes | sqrt w=2/3 0.53/0.41→1.58/1.84×; add(A,col) 1.15→6.5×; multiply(view,2.0) 0.48→1.97×; x[::2,:,:4] 0.70→2.12× | af25a746, ccadeef4 |
| masked 2-D block (`where=`) | `Is2DMaskedElementwiseShape` / `TryExecute2DMasked` → `…InnerLoop2D.Masked.cs`; mask rides as the trailing operand; tail word assembled from exactly `tail` bytes | narrow masked rows 0.4–0.9×→1.3–10.5× | 09d1fc59 |
| axis coalescing | `Backends/Iterators/NDIterCoalescing.cs` `CoalesceAxesIterationOrder`: NumPy's unconditional merge in iteration order (`stride[d] == stride[d+1]*shape[d+1]`, or a size-1 stride-0 axis) | `(250000,4) × scalar` 1.7 ms→251 µs; `a[:, ::2]` is ONE chunk | ccadeef4 |
| `where=` run scan | `NDIter.Execution.cs` `InvokeInner`: one movemask per 32-byte block, runs walked by trailing-zero counts | all-false 27.7→1.5 µs; every regime 1.1–3.9× | 09d1fc59 |
| fancy `a[idx]`, `a[idx]=v`, `m[ridx]`, `m[ridx]=row` | `Selection/FancyIndexKernels.cs` + `Direct/…GatherFlat.cs`: ONE int32/int64 CONTIGUOUS index array over a C-contiguous source; a flat kernel (compile-time element width, running cursors, `check_and_adjust_index` as one add + one UNSIGNED compare) for primitive widths, the general take/put kernels for whole sub-array slabs; int32 indices read IN PLACE (`idx32` variants); the setter pre-scans bounds with `Vector<T>` before its first store | a[idx32] 0.81→4.97×, a[idx64] 0.22→1.26×, a[idx64]=v 0.37→1.13×, m[ridx]=row 2.1→6.3× | 09d1fc59 |

## Representation (the op is defined on the bit pattern, or rides a narrower lane)

| op | file · technique | measured | commit |
|----|------------------|----------|--------|
| f16 flat min / max / ptp | `Direct/…Reduction.MinMax.Half.cs` (from `Max/MinElementwiseHalfFallback`, contiguous only): order key `bits ^ (0x8000 \| asr(bits,15))` → unsigned `pmaxuw/pminuw`; both extremes in ONE branch-free loop (the min-acc doubles as the −NaN detector); first-NaN / first-zero rare-path rescans | 6–60× | a05b5b1e |
| f16 maximum/minimum/fmax/fmin, clip | `Direct/…Binary.MinMax.Half.cs`: `(Half,Half)→Half` MixedType keys → static kernels; `takeIn1 = nanPick \| (~(nan1\|nan2) & (cmpKeys \| bothZero))`; clip = two chained pair-selects | 5.6–56× / 4.3–48× | f3405659, 498a68f0 |
| f16 comparisons | `Direct/…Comparison.Half.cs`: key compares → mask algebra → `PackSignedSaturate` to 0/1 bytes | 1.8–40× | 3a46cfec |
| f16 nanmin / nanmax | NaN lanes blended to the identity key; all-NaN = accumulator outside the finite key band → first element verbatim | 8–43× | c9e141c5 |
| f16 add/sub/mul/div | `Direct/…Binary.Arith.Half.cs`: Giesen widen → ONE f32 vector op → RTNE narrow → `PACKUSDW`; NaN priority is an explicit post-op blend (RyuJIT re-swaps commutative operands) | 2.2–3.6×, bit-exact incl. payloads | c8b0573d (+ strided odometer 7f2c09a3) |
| f16 floor/ceil/trunc/rint | `Direct/…Unary.Round.Half.cs`: exponent-based mantissa masking, widened to i32 lanes for the variable shift (AVX2 has none at 16 bits) | 6–26×; all 65,536 patterns × 4 ops verified | c8babf28 |
| Half sum | `Backends/Kernels/ILKernelGenerator.Reduction.Half.cs`: float32 SHADOW accumulator + `RoundToF16` per inner loop = NumPy's per-call narrow (incl. SLAB saturation, a real latent bug fixed) | flat 7–8.8×, SLAB 12–16× | 32732a0f |
| bool bitwise / logical | `Direct/…MixedType.cs` `CanUseSimdBinary` admits Boolean for and/or/xor; `GetSimdLaneType(Boolean) = byte`; normalization = one `pminub` (`min(v,1)`): and = `min(min(a,b),1)`, or = `min(a\|b,1)`, xor = `min(a,1)^min(b,1)`, not = `min(v,1)^1` | 10M 0.38–0.62×→1.86–3.05× | 46dbb9c6 |
| bool comparisons, argmax find-first, `sum(bool)` = popcount, casts from bool | `Direct/…Comparison.cs` byte lanes (`eq=(na^nb)^1`, `lt=(na^1)&nb`); `Reduction.cs` `CanUseReductionSimd` Boolean carve-out for ArgMax/ArgMin → `ArgMaxBoolHelper` (4×32B OR-scan) / `ArgMinBoolHelper` (memchr(0)); `CountTrueSimdHelper`; `Cast.FromBool.cs` normalize-then-`Vector256.Widen` chains | popcount sum 23.6×; argmax 24 ms→early exit | d967d99b |
| i64/u64 → f64 widen | `WidenI64ToF64`/`WidenU64ToF64`: low/high halves under 2^52/2^84 exponents, one final rounding — bit-exact with `(double)x` over the full range | — | 8c09dc15 |

## Algorithm swap (keyed on a data property — or a bailout when it is only observable at run time)

| op | file · selection | threshold / measured | commit |
|----|------------------|----------------------|--------|
| unique family | `Manipulation/NDArray.unique.Hash.cs` (ints + complex: open-addressing hash, splitmix64 finalizer; `HashBailoutThreshold = 1<<17` distinct → abandon to radix), `NDArray.unique.Radix.cs` (floats + every perm path, via `Backends/Default/Sorting/RadixSort.cs`); Half/Complex/Decimal keep the BCL sort | 5–74×; the bailout beats NumPy's own thrashing hash 74× at 2M-distinct int32 | 5df10897; `docs/stale-docs/UNIQUE_DESIGN.md` |
| isin | `Manipulation/np.isin.cs` `TryTableMembership`: integer/bool pair with `range ≤ 6·(\|elem\|+\|test\|)` → counting table + `take(mode='clip')` + range mask; else sort + searchsorted + equality probe; uint64↔signed routed through the uint64 domain | 1.9–3.4× | — |
| np.block | `Creation/np.block.cs`: NumPy's own `list_ndim * final_size > 2*512*512` → single-allocation `_block_slicing`, else repeated concatenate | — | — |
| kron | `LinearAlgebra/np.kron.cs`: `useTile = simd && run <= 5 && outSize >= 8192 && outSize*maxItem <= 64 MB` (materialize repeat/tile → one SimdFull multiply; beats the direct interleaved reshape only for SHORT runs on cache-resident output) | 1.9–6.5×; tail 1.2–1.4× | — |
| quantile / percentile / median / partition / argpartition | `Utilities/QuickSelect.cs`: branchless block-Hoare (128-wide misplaced-offset buffers) + NumPy `introselect_` pivot stack (adjacent kths narrow from BOTH ends); `InsertionSortThreshold = 16`; the index-tracking mirror carries `idx` as a passive passenger | 2.5–5× / argpartition 1.0–3.7×; np.partition stays below NumPy's SIMD qselect | 8a1376ff, 75a1d873 |
| radix sort | `Backends/Default/Sorting/RadixSort.cs`: `InsertionThreshold32 = 80` / `InsertionThreshold64 = 120` short-line insertion path | — | — |

## Compose existing kernels (zero new IL, zero fuzz risk)

| op | file · route | threshold / measured | commit |
|----|--------------|----------------------|--------|
| `bool << scalar` | `Backends/Default/Math/Default.Shift.cs` `TrySimdScalarShift`: `size ≤ BoolSimdScalarShiftMaxElements` (262144 = a 2 MB int64 temp, L2-resident; env `NS_BOOL_SHIFT_SIMD_MAX`) → bit-exact `astype(int64)` + the EXISTING uniform-count SIMD shift, temp disposed the instant the kernel returns; above → the single-pass fused scalar route | 100K 0.61→1.20×, 10M 1.11× | bd655743 |
| gevm | see Kernel swap — a widened dispatch condition onto `SimpleStrided` | — | 4f666fc8 |
| isclose / allclose | `Backends/Default/Logic/Default.IsClose.cs`: cast each operand through `InexactPromote(tc)` (= NumPy `result_type(x, 1.0)`) instead of a blanket f64 upcast, then let the operators promote | f32 0.18–0.35×→1.1× (10M) / 3.3× (100K); also fixed a complex128 bug | fbbda5f0 |

## View vs copy (O(1), stride tricks)

`flip`/`fliplr`/`flipud` (`Manipulation/np.flip.cs`: stride negation + base-offset shift via `Storage.Alias`,
read-only broadcast stays read-only), `rot90` (flip + transpose composition), `matrix_transpose` /
`permute_dims` / `mT`, `trim_zeros` (`np.trim_zeros.cs`: separable bounding box from `nonzero` + `any`, no
`argwhere` materialization — 3.4–361×), `unstack` (`Creation/np.unstack.cs`: drop the axis from
dims/strides, advance offset), `diag(2-D)` / `diagonal`, einsum's single-operand view path
(`LinearAlgebra/np.einsum.Contract.cs`), `Shape.TryNocopyReshape`. Verified by asymptotics + metadata,
never timed (`benchmark/O1_EXCLUSIONS.md` keeps them out of every geomean).

## Early exit / constant output

| op | file · condition | note | commit |
|----|------------------|------|--------|
| `bool >> s≠0` | `Default.Shift.cs` `TryBoolScalarRightShiftZeros`: provably all-zero → lazy zeros allocation | ~1.5 µs at ANY size — a bench must dispose it or BDN batching commit-OOMs at 10M | 46dbb9c6 |
| bool argmax / argmin | find-first block scans (see Representation) | 24 ms full scan → early exit at index ~1 | d967d99b |
| `all` / `any` | mask early exit | — | — |
| `np.where(cond)` | all-false SIMD prescan | — | — |
| matmul `k==0`, `diff n==0`, `power` exp ∈ {0,1} | zeros / the input / `ones_like` or a copy — only when the dtype the general path would produce is preserved | — | — |
| scalar-broadcast copy | `NDIter.cs` `TryCopySameType`: src all strides 0 + gap-free whole-buffer C/F dst → one typed `Fill` | 6–8× for 1-byte, ~2× wider | — |

## Fixed-cost bypass (skip the setup for the trivial subset — the small-N lever)

| op | file · gate | measured | commit |
|----|-------------|----------|--------|
| same-layout copy | `Backends/Iterators/NDIter.cs` `TryCopySameType`: `IsSameFlatLayout(src, dst)` hoisted ABOVE `CreateCopyState` → one `Buffer.MemoryCopy` (overlap is resolved upstream) | copyto n=100 341→110 ns (2× NumPy); reaches copy/Clone/ravel/concatenate | 7bfc27a2 |
| small partial broadcast | `Backends/Default/Math/DefaultEngine.BinaryOp.cs` `TryExecuteBinaryOpViaNDIter`: `size < DirectBroadcastMaxElements` (8192; env `NS_BROADCAST_DIRECT_MAX`; measured crossover ~16–32K), same dtype, EXACTLY ONE operand broadcast (`IsBroadcasted ^ IsBroadcasted`), both C-or-F contiguous, `CanUseSimdBinary` → the direct SimdChunk path (each exclusion is a latent bug in that path) | 1K matrix+row 1.69→0.83 µs (1.8–2.3×) | — |
| same-dims ufunc routes | `DefaultEngine.UfuncOut.cs` `*UfuncInto`: skip `Broadcast` + `ResolveUfuncIterationShape` when lhs/rhs/out dims are identical (`Shape.Equals` compares dims only) | ~55 ns + 2 Shape allocs per call | 15154b00 |
| NDIter construction | `NDIter.State.cs`: ONE native block with a 1536-byte inline arena, 4-slot `[ThreadStatic]` recycle (`ResetForRecycle` re-zeroes only the carved bytes); `InnerLoopKernelKey` packed front cache over the string cache; direct `ExternalLoopNext` in the EXTERNAL_LOOP drivers; eager COPY_IF_OVERLAP temp dispose | ctor 161→93 ns (3-op); −45 ns + 96 B/call; `add(out=)` n=1 411→168 ns (NumPy 289) | 15154b00 |
| vdot / vecmat / vecdot / matvec wrappers | `OpenBlasEngine.Products.cs`: skip `np.ravel` when C-contiguous (the flatten IS the strided read); no-batch case uses `Array.Empty<long>()` instead of four `new long[0]` | ~0.5 µs → parity | a0a6a439 |

## Memory strategy (a size / layout / microarchitecture threshold)

| op | file · threshold | measured | commit |
|----|------------------|----------|--------|
| tri / tril / triu | `Creation/np.tri.cs` `WriteOnceMaxBytes = 64 MiB`: below, the pooled buffer comes back dirty → allocate uninitialised and write every byte once; above, `np.zeros` rides OS zero-pages and only the kept run is touched | 64 MB: 6.0 vs 11.0 ms; 72 MB: 18.9 vs 13.5 the other way | — |
| take / fancy gather prefetch | `Indexing/np.take.cs` `IndexPrefetchThresholdBytes = 2 MiB`, `Selection/FancyIndexKernels.cs` `PrefetchThresholdBytes`: `Sse.Prefetch0` 32 indices ahead ONLY when the gathered footprint exceeds it; a separate cached `(copyKind, prefetch)` kernel keeps the small path lean | below the threshold prefetch is 1.6× SLOWER at 100K; take 10M 1.7–2.5× | — |
| take / put / place element copy | `Direct/…{Take,Put,Place}.cs` `EmitElementCopy`: typed `Ldind/Stind` MOV by width (two 8-byte MOVs for 16-byte types) instead of a per-element runtime-size `cpblk`; `put`/`place` wrapping values cursor replaces `i % count` | the `cpblk` made `take` LOSE at 0.68×; place 100K 2.0–4.5× | — |
| bincount | `Sorting_Searching_Counting/np.bincount.cs`: K=4 privatized accumulators when `ansSize ≤ 2048 && len ≥ 4096 && len ≥ 4·ansSize` (4×2048×8 B = 64 KB in L2; breaks the store-forwarding stall); the WEIGHTED path stays ONE sequential accumulator (float parity) | 0.72× at 3072 bins (spills L2) | — |
| byteswap | `Casting/NDArray.byteswap.cs` `NonTemporalThresholdBytes = 2 MiB` | — | — |
| `fillZeros:false` | every result a kernel / native routine fully overwrites: `Default.MatMul.2D2D.cs`, all OpenBLAS products/factorisations, `Math/NDArray.SlidingDot.cs`; NEVER where an untouched slot must read 0 (bincount, lstsq's min-norm path) | ~2–2.5× on alloc-heavy loops; correlate 1.3→3.3×; 128² gemm 0.090→0.059 ms | a150e4e9, 6521b8b4 |
| LAPACK scratch retention | `src/NumSharp.Interop.OpenBLAS/OpenBlasEngine.cs` `Alloc<T>/Free<T>`: thread-local pool, power-of-two buckets, 16-byte size header, 32 MiB idle cap — per-call `NativeMemory.Alloc` demand-zero-faulted INSIDE the timed native call (fault ≈ whole compute); coalescing into one malloc does NOT help, only retention | 15 ops 0.44–0.50× → parity | e89d6d8f |
| pool GC pacing | `Backends/Unmanaged/Pooling/SizeBucketedBufferPool.cs`: gen0 pacing every 16 MB handed out (`NUMSHARP_POOL_GC_PACING`, `_MB`), pressure in 64 MB chunks (per-Take `AddMemoryPressure` caused a FULL gen2 every ~24 calls), burst-sized bucket caps | 1K f32 0.29→0.57–0.69×; hit rate 16.8→99.7%; U-shaped: 1–4 MB starves, 32–128 MB rotates past L3 | 160ecbba |

## Fused pass

| op | file · gate | measured | commit |
|----|-------------|----------|--------|
| np.select | `Indexing/np.select.cs` `TrySelectFused` → `Direct/…Select.cs`: kernel supports dtype, non-empty, every condition bool + C-contiguous + exact result shape, every choice result-dtype + C-contiguous + exact shape, default a scalar or a full same-dtype array; else NumPy's own reverse-`copyto(where=)` composition | 8-byte all-array cells 1.22–1.54→1.6–2.4×; 100K 8–18× | — |
| np.evaluate | `Backends/Iterators/NDExpr*.cs`, `DefaultEngine.Evaluate.cs`: one NDIter pass per expression tree, root-only reductions | 3.2–6.1× NumPy on 4M chains | — |
| fused bool shift body | `Default.Shift.cs` `ExecuteShiftViaNDIter` + `EmitMixedScalarBody`: the bool operand is never pre-astyped (was two passes over the 8× buffer) | — | 46dbb9c6 |
| sliding-dot outer vectorization | `Math/NDArray.SlidingDot.cs`: the middle region is vectorized over OUTPUT positions with t-order preserved → byte-identical to the scalar sum | geomean ~4× | — |

## Wrapper-glue removal (parity-bound ops — the only lever left)

| what | where | measured | commit |
|------|-------|----------|--------|
| string-slice parse in a hot wrapper | `np.linalg.cond` (computed in `LinearAlgebra/linalg/np.linalg.svd.cs`): `s["..., 0"]`/`s["..., -1"]` = `Slice.ParseSlices` = Regex.Split + Regex.Match PER CALL (~1.7 µs each) → `s[new[]{ Slice.Ellipsis, Slice.Index(0) }]` (the `params Slice[]` indexer, 0.3 µs, no regex) | cond wrapper 7.0→2.5 µs; 0.93→1.08× | a0a6a439 |
| an allocating predicate composition | `np.linalg.eig`: `np.all(np.equal(np.imag(w), 0))` → `!np.any(np.imag(w))` (bit-for-bit the same predicate — a NaN is truthy in `any`) | ~1.5 µs | a0a6a439 |
| a second storage allocation per result | predicates: `MakeGeneric<bool>()` (= `Storage.Alias()`, a 2nd `UnmanagedStorage`) → `AsGeneric<bool>()`; the rule: `AsGeneric<T>() ?? throw` for a fresh untyped result you own, `MakeGeneric` when the source is a `using` local that might already be `NDArray<T>` (AsGeneric returns `this` → use-after-free) | 0.547→0.470 µs; isnan 100K 0.87→1.06× | 4dfe7619 |
| tier-0 leaves inside short native calls | `OpenBlasEngine.{Lapack,Cholesky}.cs` `Linearize<T>`/`Delinearize<T>`: `[MethodImpl(AggressiveOptimization)]` — a 32×32 LAPACK op finishes inside tiered compilation's 100 ms delay, so the generic copy leaves ran tier-0 for the WHOLE call (12×) | cholesky-32 0.31→1.9× | 9cf599c4 |
| per-position interface dispatch | `Backends/ISlidingDotBackend.cs` `DotBatch(dtype, a, b, o, count, n2)` (default = the old per-position loop): the managed loop paid an interface virtual + dtype switch per output — 1.45 ms of a 2.35 ms call | correlate 0.51→1.13× | 9cf599c4 |

## Backend seam

`Backends/IBlasBackend.cs` + `IBlasBackend.LinearAlgebra.cs` (18 `Try*` members, every default `false`),
`Backends/ISlidingDotBackend.cs`. Read `TensorEngine.Blas` into a LOCAL before calling it (test-then-call
on the settable property was ~2% NREs under a concurrent `Disable()`); a backend answers only for the
operands it implements and can change WHICH implementation computes, never WHETHER.

## Measured and REJECTED — do not re-derive

| lever | result | where recorded |
|-------|--------|----------------|
| 4×-unrolling `SimdScalarRight/Left` | memory-bound, no gain (10.6 vs 11.1 µs @100K) | broadcast-perf |
| buffering narrow strided rows into a contiguous window | SLOWER than per-row (4.45 vs 3.95 ms at w=4) — the inner runs are already SIMD-able in place | `NDITER_2D_BLOCK_KERNEL.md` §2.3 |
| a per-run 32-byte vector probe in the `where=` scan | regressed run-length-1 masks 237→353 µs; the per-block bit walk keeps both regimes | nditer Tier 1 |
| K-panel-blocked pairwise Gram | regressed the M=2 cov benchmark (panel overhead) | cov syrk |
| a float hash in `unique` | ~2× regression on all-unique input; a cheap cardinality sample cannot decide | `UNIQUE_DESIGN.md` |
| a full-double exp2 scale | `Vector256.ConvertToInt64(double)` is AVX2-EMULATED: 343 µs/100K, slower than NumPy | exp2 port |
| `Sse.Prefetch0` / a deferred-NaN accumulator in the 10M argmax stream | 2× / 2.3× SLOWER (bandwidth contention / loop-carried dependency) | argmax |
| 16 accumulators or prefetch in flat int min/max | no faster than 8 (per-core bound, ~36–48 GB/s) | reduction widening |
| a `[::2]` shuffle-deinterleave isnan kernel | ties NumPy — must touch 2× the address span, memory-bound | predicates |
| NumPy's lean 4-op in-place `np.vander` | slower in NumSharp (38.9 vs 32.6 µs) — strided-view writes cost more than the contiguous 6-op composition | OpenBLAS bench fix |
| "sort the whole tiny row once" in the quantile axis path | 2.4× regression — the `InsertionSort` call from a non-`AggressiveOptimization` entry did not inline | quantile |
| a hardware SIMD gather for `take_along_axis` | ~1.16× before the bounds validation a raise-mode gather still needs | take_along_axis |
| staggering pooled buffers against 4K aliasing | a one-shot 1.30× that vanished interleaved (0.388–0.411 vs 0.383–0.454 ms) | `NDITER_PERF_DISCOVERY.md` §3 trap 9 |
| an eager-reclaim weak-ref ring / `SuppressFinalize` on transient results | UNSAFE (frees a live view's buffer) / ~15% | comparison-smalln-alloc-floor |
| parallel flat integer min/max (bit-exact, ~5×) | REVERTED by policy — single-threaded kernels only | 1c709bf0 → 7b8dac64 |
| coalescing all LAPACK scratch into one malloc | faults exactly the same; only RETENTION helps | LAPACK scratch pool |
| threads / `Parallel.For` anywhere in a kernel | forbidden | Eli's rule |
