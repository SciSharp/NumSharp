# Implement & integrate a specialized path

## 1. Integrate at the general primitive's DISPATCH, not per-function

Put the branch where the general kernel is *dispatched*, so every consumer inherits it. gemv went into
`Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs` (`TryMatMulSimd`) + the `SimdMatMul` dispatch — one
change lifted `np.dot(m,v)`, `np.matvec`, `np.inner(2d,1d)`, and transitively `einsum`/`tensordot`/
`multi_dot` (they compose over matmul). A per-function hack in `np.matvec.cs` would have missed all the
others. The 2-D block kernel went into the three packed-key `ExecuteElementWise*` entry points of
`NDIter.Execution.Custom.cs`, so it serves BOTH the `out=` and the allocating unary/binary routes.

Exception: when the structure is function-specific (cov's `dot(Xc, Xc.T)` is only a Gram *because cov
knows the operands alias*), gate inside the function (`np.cov.TryGramCov`) and call the shared kernel.

## 2. Implementation shape per KIND

Sections 3–7 below are the KERNEL-kind deep dive. The other kinds are simpler — a plain C# branch, no
IL — but the same gate + fallback contract. Map your kind to its shape and its exemplar
(`catalog.md` has the full list with gates and thresholds):

| kind | implementation shape | exemplar (file) |
|------|----------------------|-----------------|
| **kernel swap** | IL-emit or generic-delegate kernel keyed by dtype; gate on layout/shape (§3–7) | `DirectILKernelGenerator.Gram.cs` (`GemvHelper`), `.AdjacentDiff.cs`, `.Reduction.Arg.cs` |
| **kernel contract** | a kernel delegate whose signature carries the OUTER axis (`dataptrs, innerByteStrides, innerCount, outerByteStrides, outerCount`), emitted by wrapping the SAME inner emit bodies in an outer pointer-bump loop; a type-independent shape gate; a per-block odometer for leading axes | `DirectILKernelGenerator.InnerLoop2D.cs` + `NDIter.Execution.Custom.cs` (`Is2DElementwiseShape`); `Selection/FancyIndexKernels.cs` + `.GatherFlat.cs` |
| **representation** | raw-lane intrinsics on the reinterpreted bits (`Avx2.*` on ushort/byte lanes), or widen → one vector op → narrow; the semantics contract (NaN priority, ±0 ties) as EXPLICIT blends | `.Reduction.MinMax.Half.cs`, `.Binary.Arith.Half.cs`, `.MixedType.cs` (bool byte lanes), `ILKernelGenerator.Reduction.Half.cs` |
| **algorithm swap** | pick the algorithm from a data property, both paths returning the identical SET; when the property is unknowable up front, run one algorithm with a counter and BAIL to the other at a threshold | `NDArray.unique.Hash.cs` (hash int / bail to radix at 2^17), `np.isin.cs` (table vs searchsorted by value range), `np.kron.cs` |
| **compose existing** | `Try…()` that converts through an existing bit-exact kernel and runs an existing SIMD kernel, gated by a size threshold; dispose the temp the instant the kernel returns | `Default.Shift.cs` (`TrySimdScalarShift`) |
| **view vs copy** | build a `Shape` with negated/permuted strides + shifted offset via `Storage.Alias`; **inherit the source's writeable/broadcast flags** | `np.flip.cs`, `np.rot90.cs`, `np.matrix_transpose.cs`, `np.unstack.cs` |
| **early exit / constant output** | a cheap structural test up front (`size==0`, `k==0`, SIMD all-zero prescan, "provably all zeros") → return the identity/short-circuit result — in the dtype the general path would produce | `all`/`any` (mask early-exit), `np.where` (all-false prescan), `Default.Shift.cs` (`TryBoolScalarRightShiftZeros`) |
| **fixed-cost bypass** | hoist the trivial-subset test ABOVE the iterator/broadcast construction and do the one memcpy / direct kernel call there; a `size <` threshold for the direct route | `NDIter.cs` (`TryCopySameType` same-layout hoist), `DefaultEngine.BinaryOp.cs` (`DirectBroadcastMaxElements`), `DefaultEngine.UfuncOut.cs` (same-dims skip) |
| **memory strategy** | a size/layout `if` choosing the allocation/access; a named threshold with the measured crossover in its comment | `np.tri.cs` (`WriteOnceMaxBytes`), `Indexing/np.take.cs` (`IndexPrefetchThresholdBytes`) + `Selection/FancyIndexKernels.cs` (`PrefetchThresholdBytes`), `np.bincount.cs` (privatized accumulators), `SizeBucketedBufferPool.cs` |
| **fused pass** | one `Try…()` that recognizes the contiguous/no-cast shape and runs a single pass; else the composition | `np.select.cs` (`TrySelectFused`), `NDExpr.Evaluate.cs` |
| **wrapper-glue removal** | replace per-call Regex/params/alias allocations with prebuilt `Slice[]`, `Array.Empty`, `AsGeneric`; `AggressiveOptimization` on tier-0 copy leaves; batch an interface call | `np.linalg.cond`/`eig` wrappers, `OpenBlasEngine.Lapack.cs` leaves, `ISlidingDotBackend.DotBatch` |
| **backend seam** | a `Try…(out result)` interface member the backend overrides for what it serves, default returns false | `IBlasBackend.cs` (`TryDot`/`TryMatMul2D`/…), `ISlidingDotBackend.cs` |

## 3. dtype-dynamic WITHOUT a per-dtype switch (KERNEL kinds — project rule)

Two blessed patterns — never a `switch(typecode){ case Double: DoDouble(); ...}` over 15 dtypes.

### A. Generic helper reached through a cached delegate (the `CumSumHelperSameType` shape)

Best when the loop is expressible with `System.Runtime.Intrinsics.Vector256<T>` / `INumber<T>`. Used by
`GramHelperSameType<T>` / `GemvHelperSameType<T>` in `DirectILKernelGenerator.Gram.cs`:

```csharp
public unsafe delegate void GemvKernel(void* a, long aRowStride, void* x, void* y, long m, long k);
internal static readonly ConcurrentDictionary<NPTypeCode, GemvKernel> _gemvCache = new();

public static GemvKernel GetGemvKernel(NPTypeCode dt) {
    if (!Enabled) return null;
    if (dt != NPTypeCode.Single && dt != NPTypeCode.Double) return null; // scope; else fall back
    if (!Vector256.IsHardwareAccelerated) return null;
    return _gemvCache.GetOrAdd(dt, static d =>
        GetGenericHelper(nameof(GemvHelperSameType), GetClrType(d)).CreateDelegate<GemvKernel>());
}

// void* params keep the CLOSED method's signature dtype-independent, so ONE delegate type binds
// every instantiation via CreateDelegate — no IL wrapper needed.
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
internal static unsafe void GemvHelperSameType<T>(void* ap, long aRowStride, void* xp, void* yp, long m, long k)
    where T : unmanaged, INumber<T> {
    T* a=(T*)ap; T* x=(T*)xp; T* y=(T*)yp;
    for (long i=0;i<m;i++) y[i] = GramDot<T>(a + i*aRowStride, x, k);
}
```

Notes: `GetGenericHelper(name, clrType)` = `GetHelper(name).MakeGenericMethod(clrType)`; the helper must
be a `static` (NonPublic OK) method on `DirectILKernelGenerator`. Add
`[MethodImpl(AggressiveOptimization)]` — without it the first ~40 calls run tier-0 (measurably slow).
Plain C# helpers can also bind DIRECTLY to an existing kernel delegate type with no IL at all
(`Cast.FromBool.cs` binds its normalize-widen helpers to `CastKernel`).

### B. NPTypeCode-keyed IL emit (the `AdjacentDiff` shape)

Best when you need per-lane control the generic path can't express, or all-dtype coverage. Emit a
`DynamicMethod` keyed by NPTypeCode, using the shared emit helpers — all take NPTypeCode, no `<T>`:
`EmitVectorLoad(il, dt)`, `EmitVectorOperation(il, op, dt)`, `EmitVectorStore(il, dt)`,
`EmitScalarOperation(il, op, dt)`, `EmitLoadIndirect/StoreIndirect(il, dt)`. See
`DirectILKernelGenerator.AdjacentDiff.cs` for a full 4×-unrolled SIMD + remainder + scalar-tail kernel.

**Reuse the general kernel's emit BODIES** whenever the new kernel changes only the driving loop: the 2-D
block kernel wraps the per-chunk kernel's own scalar/vector bodies in an outer pointer-bump loop
(`EmitSimdContigLoop` addresses `ptr + i*elemSize` and never mutates the base pointer, which is what
makes the wrap possible), so the two routes are byte-identical BY CONSTRUCTION — elementwise ops carry
no cross-element state.

### C. Compile-time width beats runtime-size arithmetic

The general take kernel spends 0.75 ns/elem on three runtime-size imuls + a mode dispatch and could not
beat NumPy's 0.57 ns/elem trivial gather; a flat kernel specialized on the element width at emit time
(running cursors, one unsigned bounds compare) did (45 vs 57 µs). Likewise a per-element `cpblk` of a
runtime-tiny size is a memcpy call where a `mov` belongs (~2× — it made `take` LOSE at 0.68×):
`EmitElementCopy` emits a typed `Ldind/Stind` per width (two 8-byte MOVs for 16-byte types), keyed into
the kernel cache as `copyKind`. **The kernel-cache key must carry every specialization** (`copyKind`,
`prefetch`, `idx32`, `(Op, dtype, layout)`), or the cache serves the wrong variant to the next caller.

## 4. Reuse existing primitives — don't re-derive

- `GramDot<T>` (generic 4-accumulator SIMD dot, `DirectILKernelGenerator.Gram.cs`) — the reduction core
  for Gram/gemv.
- `SimdDot.DotDouble/DotFloat` (FMA, `Backends/Kernels/SimdDot.cs`) — contiguous double/float dot.
- `SimpleStrided` inside `SimdMatMul` — the per-k SIMD daxpy; gevm just *routes* `M==1` to it (widened
  a dispatch condition) instead of writing a new kernel — which is also why gevm stayed byte-exact.
- The Giesen f16 converts `HalfBitsToFloat{,Exact}` / `FloatToHalfBits` / `RoundToF16` (internal in
  `Cast.Half.cs` / `Cast.ToHalf.cs` / `ILKernelGenerator.Reduction.Half.cs`) — every f16 widen-compute-
  narrow path and the Half sum shadow ride them.
- The f16 order key `bits ^ (0x8000 | asr(bits,15))` (`Reduction.MinMax.Half.cs`) — proven strictly
  order-preserving; NaN keys land strictly outside `[key(-inf), key(+inf)] = [0x03ff, 0xfc00]`.
- `EmitInlineMaskCreation` (np.where's bool→lane expansion) — reused by `np.select`'s fused chain.
- `RadixSort.ArgSortU32/U64` (`Backends/Default/Sorting/RadixSort.cs`) — the unique family's sort core.
- `QuickSelect.PartitionAtMany` (`Utilities/QuickSelect.cs`) — median/percentile/quantile/partition.

## 5. GATE + FALLBACK is the whole contract (universal — every kind)

The specialized path returns `null` / doesn't fire on ANY miss, and control falls through to the
untouched general kernel. Gate on: dtype in scope, **reduction axis contiguous** (`stride==1`),
not broadcast, the structural condition (`N==1` / `M==1` / `M<=16` / …), and non-degenerate size.

```csharp
// np.cov.TryGramCov gate — every miss → np.dot fallback
if (dt != NPTypeCode.Double && dt != NPTypeCode.Single) return null;
if (!Xc.Shape.IsContiguous || Xc.Shape.IsBroadcasted) return null;
if (m < 1 || m > GramCovMaxVars || k < 1) return null;
var kernel = DirectILKernelGenerator.GetGramKernel(dt);
if (kernel is null) return null;
```

Never widen a fast path to a shape you didn't measure — that's how you turn a win into a regression
(cov Gram loses above M=16; firing it there would slow cov down).

**Gate-construction rules (each learned from a live gate):**

- **A threshold is a named `const`/`static readonly` whose comment records the measured crossover and
  WHY it sits where it does, with an env override for tuning** — `DirectBroadcastMaxElements` (8192,
  `NS_BROADCAST_DIRECT_MAX`, crossover ~16–32K "with margin for host-regime noise"),
  `BoolSimdScalarShiftMaxElements` (262144, `NS_BOOL_SHIFT_SIMD_MAX`), `WriteOnceMaxBytes`,
  `GramCovMaxVars`, `HashBailoutThreshold`, `BincountPrivMaxBins`. Sit it BELOW the crossover.
- **Make the gate cheap and type-independent, ahead of any setup.** `Is2DElementwiseShape` reads
  iterator flags and strides only (unbuffered, EXTERNAL_LOOP, ndim ≥ 2, no mask, output inner stride
  1, inputs 1 or 0); callers build the operand-type array only after it says yes.
- **Know the dispatch ORDER — an earlier tier can shadow you.** `np.add(arr, 0-d)` rides Tier 3B
  (`TryExecuteBinaryOpViaNDIter`) BEFORE the MixedType kernel gate, so the f16 arithmetic kernels only
  became reachable once Tier 3B declined `(Half,Half)→Half`. Trace the route from the np.* entry to
  your gate before measuring; "before ≈ after" is the tell that it never ran.
- **Wire EVERY entry point.** `NDIter.Execution.Custom.cs` has a string-key and a PACKED-key
  `ExecuteElementWise` overload; the masked ufunc routes call the packed one. A gate added to only the
  string-key overload cost a whole measurement round.
- **Widening a gate onto an older path AUDITS that path.** Routing small broadcasts to the direct
  SimdChunk route exposed four latent bugs that NDIter had masked: mixed-dtype NEP50 promotion
  diverged, a kron-style DOUBLE broadcast left outputs UNWRITTEN (pool garbage), trailing-size-1 /
  negative-stride views were mishandled, f16 specials took a scalar quirk. Each became an exclusion
  (`leftShape.IsBroadcasted ^ rightShape.IsBroadcasted`, `IsContiguousCorF(both)`, `CanUseSimdBinary`),
  and only the fuzz corpus + the full suite found them. Budget for that audit.
- **The early-exit result must be in the dtype the general path would produce** —
  `TryScalarExponentFastPath` returns `ones_like(lhs)` for `x**0` ONLY when `ResolvePowerResultType`
  equals `lhs`'s dtype, else it declines (a promoted result would differ).
- **A bailout threshold is on an OBSERVED counter**, not a sample: `UniqueHashSortedInt` abandons to
  radix once `uniqueCount >= 1<<17`; guard the re-entry (`uniqueValuesOnly<T>` hashes only when
  `n <= 1<<28`, the hash's own fallback bound — otherwise it loops).

## 6. Pointer & stride discipline (kernel kinds)

- Logical base pointer = `(byte*)nd.Address + nd.Shape.offset * elemsize` (mirror `ExecuteUnaryKernel` /
  `TryMatMulSimd`). `nd.Address` is the buffer base at offset 0. This is right for BOTH view kinds: a
  CONTIGUOUS slice (`a["5:"]`) re-seats `Storage.Address` and keeps `Shape.offset == 0`; a strided view
  keeps the base address and a non-zero offset (probed; the old fancy route was NOT offset-buggy —
  pinned by `Indexing.KernelRoute.Tests`).
- Element strides come from `nd.Shape.strides[axis]` (elements, not bytes). Pass a **row stride** so the
  kernel handles offset views: `A[i]` starts at `A + i*aRowStride`, not `A + i*K`, when the array is a
  column slice with `aStride1==1` but `aStride0 != K`.
- Gate the SIMD reduction on the reduction axis being unit-stride; a strided reduction axis is either a
  slower strided kernel or a fallback.
- **`nd.shape[i]` is `long`** — cast to int or use `nd.Shape.dimensions[i]` (a real build-error trap).
- **Never allocate an output as `np.empty(view.Shape)`** — it inherits the view's STRIDES and fabricates
  a size-N buffer with addressing-beyond-N strides; any kernel honouring them writes out of bounds
  (an AV in the ForEach 3-D odometer, found while building the 2-D kernel). Use no-`out` (production
  `.Clean()`s) or `np.empty(new Shape(dims))`.
- `np.take` on a non-contiguous source copies the WHOLE source (`ascontiguousarray`) before gathering —
  fine standalone, wrong as a building block for a sliced-view gather.

## 7. Symmetry: compute half, mirror — and know when it's bit-exact

For a symmetric result compute the upper triangle and mirror. This is bit-exact vs NumPy's independent
per-entry dots even for `A @ B.T` when the result is *mathematically* symmetric (cov's weighted Gram,
`B = Xc*w`): `dot(A[i],B[j])` and `dot(A[j],B[i])` are the same products in the same order (`x·y` term
commutes), so the mirrored value equals what NumPy computes for the lower entry. Integer/bool
privatization (bincount's K accumulators) is bit-exact because modular add is associative; a FLOAT
accumulator may NOT be split (the weighted bincount stays sequential — privatizing reorders per-bin sums
by ULPs, verified against catastrophic cancellation).

## 8. JIT and codegen rules (measured, load-bearing)

- **`[MethodImpl(AggressiveOptimization)]` on leaf hot loops.** Tiered compilation promotes a method
  only after a 100 ms window with no new tier-0 JIT; a small op finishes long before that, so the
  generic LAPACK copy leaves (`Linearize<T>`/`Delinearize<T>`) ran tier-0 for the whole call — 12×
  slower (2.5 vs 0.2 µs on a 32×32 transpose). The new block-partition methods needed it too, plus
  `[SkipLocalsInit]` where a per-row `stackalloc` was being zeroed.
- **NEVER on IL emitters.** `ILKernelGenerator.*`/`DirectILKernelGenerator.*` methods that take an
  `ILGenerator` or build a `DynamicMethod` run ONCE; the emitted method is already tier-1. The
  attribute only adds optimizing-JIT startup cost. Convention: `OptimizeAndInline` for runtime
  per-element/per-chunk helpers (`Bulk*`/`Fused*`/`*Strided`), never for one-shot orchestration.
- **NOT on inline-hot helpers.** `AggressiveOptimization` on `*Blas.Dot` blocked its inlining into the
  sliding/product loops — correlate ~2× slower. Leaves only.
- **Generic hot compares take POINTERS.** `LtV<T>(T a, T b)` does `*(float*)&a < *(float*)&b`; taking
  `&a` of a by-value parameter forces a per-element stack store+reload that erased the block-partition
  win (generic stayed ~78 ms vs 21). `LtPtr<T>(T* a, T* b)` reads in place; the `typeof(T)==typeof(X)`
  JIT-fold pattern is fine, the by-value spill is not.
- **Static, non-capturing, `AggressiveOptimization` is what makes a hand loop fast**, not the
  intrinsic spelling: `Avx.LoadVector256` vs `Vector256.Load` measured identical (183.8 vs 187.2 GB/s);
  capturing local functions / no attribute measured 56–70 GB/s — the "my hand loop is slow" trap, and
  why a probe's timed body lives in a `static class`.

## 9. SIMD-emission traps (each cost real debugging time)

- **`il.DeclareLocal(Vector256<bool>)` does NOT throw at emit time — it NREs at RUNTIME inside the
  kernel.** The bool byte-lane seam is `GetSimdLaneType(Boolean) = byte`, consulted by
  `EmitVectorLoad/Store/Create` AND every vector LOCAL declaration; grep `VectorMethodCache.V(VectorBits`
  before admitting bool to any new kernel path (five such sites bit once).
- **The portable `Vector256` API does not lower for ushort compare/select** — a sign kernel on
  `Vector256.GreaterThan<ushort>`/`ConditionalSelect` was ~30× slower than raw `Avx2.*`. f16 bit
  kernels use `Avx2.*` directly (signed `CompareGreaterThan` is safe on magnitudes ≤ 0x7fff).
- **AVX2 has no 16-bit variable shift** (`VPSLLVW` is AVX-512BW) — widen 8 ushorts to i32 lanes and use
  `VPSLLVD/VPSRLVD` (the f16 roundings).
- **`Vector256.ConvertToInt64(double)` is AVX2-EMULATED** (no `vcvtpd2qq` before AVX-512) — a
  full-double exp2 scale measured 343 µs/100K, SLOWER than NumPy; keep the integer power in float
  exponent-field ops and widen only the polynomial.
- **RyuJIT re-swaps commutative intrinsic operands.** `Avx.Add(fb, fa)` does not guarantee `vaddps
  dst, fb, fa` — observed swapped, resolving the wrong operand's NaN. Any operand-order semantics (x86
  NaN priority, first-NaN-wins) must be an explicit mask/blend (`quiet(x) = x | 0x0200` post-op), never
  instruction order. Same class: MAXPS's skip-NaN identity needs DATA as SRC1 — `Avx.Max(dataVec, acc)`.
- **An in-loop data-dependent branch costs ~2.26× on a memory-bound stream** (argmax's `isMax` bool):
  emit max and min as SEPARATE methods. Loop-INVARIANT flags (`isMax`, `nanIgnore`, `op`) are
  predicted and free — one shared core with invariant flags beats duplicated loops.
- **Prefetch and loop-carried dependencies REGRESS a bandwidth-bound loop** (`Sse.Prefetch0` 2× slower,
  a deferred-NaN `acc &= notnan(...)` 2.3× slower on the 10M argmax stream). The straight NumPy port is
  the ceiling there; do not "improve" 10M.
- **Masked tails**: on a 256-bit AVX2 host finish `innerCount % lanes` (or a whole row when `innerCount
  < lanes`) with one `MaskLoad → body → MaskStore`, mask hoisted per call, INTEGER masked-off lanes
  filled with 1 so a software vector divide cannot throw; 1/2-byte lanes and other hosts keep the
  scalar tail. Assemble a mask tail word from exactly `tail` bytes — never over-read the mask array.
- **Strided-insert builds**: a 4× unroll is WORSE than 2× (register pressure on insert chains); Single
  strided pins to 128-bit vectors exactly as NumPy does (`NPY_SIMD_FORCE_128`); hardware gather was
  rejected by NumPy for these cheap ops and measured ~1.16× here.
- **The scalar tail and the SIMD body must agree on every byte** — the bool kernels normalize in both
  (`(a!=0)&(b!=0)` mirrors `min(min(a,b),1)`); a raw `Ceq` on `frombuffer` 0x80 bytes was a silent divergence.

## 10. Lifecycle hygiene is part of the path

- **Dispose a temp the instant the kernel returns.** The bool shift's widened `astype` temp is disposed
  before the result is returned — one discarded buffer per call instead of two is what keeps small-N at
  parity (and plugged a pre-existing mixed-dtype astype leak). A COPY_IF_OVERLAP temp dropped to the
  finalizer on every in-place ufunc was `inplace@1` 282 → 154 ns once disposed eagerly.
- **`[NDScoped]` on the np.* entry** (the weaver reclaims transients deterministically); a helper reached
  only under a scoped caller gets `[NDScopedCovered]` (isin's `TryTableMembership`). The NDW012 analyzer
  flags an undisposed NDArray; the oracle's raw-allocation chokepoint gate flags a new `NativeMemory`
  path — both run in CI.
- **`AsGeneric<T>() ?? throw` for a fresh untyped engine result you own; `MakeGeneric` when the source is
  a `using` local that might already be `NDArray<T>`** — `AsGeneric` returns `this` in that case and the
  `using` frees it (use-after-free); `MakeGeneric`'s `Alias()` is a second `UnmanagedStorage` (0.547 vs
  0.470 µs) and shares no shape header.
- **`new NDArray(dtype, shape, fillZeros: false)` ONLY when the kernel provably writes 100% of the
  output** (prove the tiling: `nLeft + mid + nRight == outLen`). It is ~2–2.5× on alloc-heavy loops
  because the pooled buffer comes back hot and unzeroed; a demand-zero buffer's page faults land INSIDE
  the timed kernel/native call and are mis-attributed to it (128² gemm 0.090 → 0.059 ms). Never where
  an untouched slot must read 0 (bincount, a min-norm lstsq path).
- **Bare `UnmanagedStorage`/`IArraySlice` from `GetData()` hold NO counted ARC ref** — a kernel that
  keeps one past the owning NDArray's lifetime reads a pooled buffer. `~NDArray` ABANDONS its ref (never
  frees at zero); only deterministic `Dispose`/NDScope frees eagerly (`ndarray-finalizer-abandon`).
- **Scratch that a native routine writes**: retain it (thread-local pool, `OpenBlasEngine.Alloc<T>`),
  don't malloc per call — under heap churn the CRT stops keeping fresh blocks resident and the first
  native write faults every page mid-timing.
