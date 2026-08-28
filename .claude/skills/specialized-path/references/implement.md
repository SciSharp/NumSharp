# Implement & integrate a specialized path

## 1. Integrate at the general primitive's DISPATCH, not per-function

Put the branch where the general kernel is *dispatched*, so every consumer inherits it. gemv went into
`Backends/Default/Math/BLAS/Default.MatMul.2D2D.cs` (`TryMatMulSimd`) + the `SimdMatMul` dispatch — one
change lifted `np.dot(m,v)`, `np.matvec`, `np.inner(2d,1d)`, and transitively `einsum`/`tensordot`/
`multi_dot` (they compose over matmul). A per-function hack in `np.matvec.cs` would have missed all the
others.

Exception: when the structure is function-specific (cov's `dot(Xc, Xc.T)` is only a Gram *because cov
knows the operands alias*), gate inside the function (`np.cov.TryGramCov`) and call the shared kernel.

## 2. dtype-dynamic WITHOUT a per-dtype switch (project rule)

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

### B. NPTypeCode-keyed IL emit (the `AdjacentDiff` shape)

Best when you need per-lane control the generic path can't express, or all-dtype coverage. Emit a
`DynamicMethod` keyed by NPTypeCode, using the shared emit helpers — all take NPTypeCode, no `<T>`:
`EmitVectorLoad(il, dt)`, `EmitVectorOperation(il, op, dt)`, `EmitVectorStore(il, dt)`,
`EmitScalarOperation(il, op, dt)`, `EmitLoadIndirect/StoreIndirect(il, dt)`. See
`DirectILKernelGenerator.AdjacentDiff.cs` for a full 4×-unrolled SIMD + remainder + scalar-tail kernel.

## 3. Reuse existing primitives — don't re-derive

- `GramDot<T>` (generic 4-accumulator SIMD dot, `DirectILKernelGenerator.Gram.cs`) — the reduction core
  for Gram/gemv.
- `SimdDot.DotDouble/DotFloat` (FMA, `Backends/Kernels/SimdDot.cs`) — contiguous double/float dot.
- `SimpleStrided` inside `SimdMatMul` — the per-k SIMD daxpy; gevm just *routes* `M==1` to it (widened
  a dispatch condition) instead of writing a new kernel — which is also why gevm stayed byte-exact.

## 4. GATE + FALLBACK is the whole contract

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

## 5. Pointer & stride discipline

- Logical base pointer = `(byte*)nd.Address + nd.Shape.offset * elemsize` (mirror `ExecuteUnaryKernel` /
  `TryMatMulSimd`). `nd.Address` is the buffer base at offset 0.
- Element strides come from `nd.Shape.strides[axis]` (elements, not bytes). Pass a **row stride** so the
  kernel handles offset views: `A[i]` starts at `A + i*aRowStride`, not `A + i*K`, when the array is a
  column slice with `aStride1==1` but `aStride0 != K`.
- Gate the SIMD reduction on the reduction axis being unit-stride; a strided reduction axis is either a
  slower strided kernel or a fallback.
- **`nd.shape[i]` is `long`** — cast to int or use `nd.Shape.dimensions[i]` (a real build-error trap).

## 6. Symmetry: compute half, mirror — and know when it's bit-exact

For a symmetric result compute the upper triangle and mirror. This is bit-exact vs NumPy's independent
per-entry dots even for `A @ B.T` when the result is *mathematically* symmetric (cov's weighted Gram,
`B = Xc*w`): `dot(A[i],B[j])` and `dot(A[j],B[i])` are the same products in the same order (`x·y` term
commutes), so the mirrored value equals what NumPy computes for the lower entry.
