# Audit v2 — Group 4: Logic + Shape + Storage

Audited on `nditer` branch. All claims independently verified with NumPy 2.x + `dotnet_run` reproductions.

---

## File: `src/NumSharp.Core/Backends/Default/Logic/Default.All.cs`

### What it does
Dispatches `bool All(NDArray)` for 15 dtypes. Generic types use
`NpyIter.ReduceBool<T, NpyAllKernel<T>>(nd)`; `Half`/`Complex` use bespoke
contiguous/strided implementations.

### Structural / API parity
- `np.all` API parity gap: no `out=`, no `where=`, no `keepdims=` on the
  no-axis overload (NumPy: `np.all(a, axis=None, out=None, keepdims=False,
  where=True)`). The signature `bool All(NDArray)` always returns a primitive
  even when NumPy would return a 0-D array. (At `np.all.cs:15`.)
- All 15 dtypes covered, including `SByte`, `Half`, `Complex`. Default branch
  throws `NotSupportedException`.

### Behavioral parity — verified
- Empty array → `True` ✓
- Scalar `np.array(42)` axis=0 → `True` ✓
- `(3,0)` axis=1 → `[True, True, True]` ✓
- NaN in `np.all([1, NaN, 3])` → `True` (NaN truthy) ✓
- Strided/broadcast inputs go through the iterator fallback (correct).

### Performance — measured (1M element int32, 100 iters)
- NumSharp contiguous: **270 ms** vs NumPy **21.5 ms** → **~13× slower**.
- NumSharp strided: 222 ms (the iterator path is actually *faster* than the
  "contiguous" path here, which is a smell).

Root cause: `NpyAllKernel<T>.Accumulate` is `accumulator &&
!EqualityComparer<T>.Default.Equals(value, default)`. JIT devirtualises but
emits a scalar loop with one short-circuited `&&` per element — no SIMD, no
unrolling, no early exit batched across vector lanes. Even with `ShouldExit`,
`NpyIter.ReduceBool` checks it per element, not per vector chunk.

### Performance gap >10× → **YES** (Severity: HIGH)
NumPy uses `_ALL_OPS` ufunc with vectorised SIMD and short-circuit on the
first zero detected in a vector chunk. NumSharp has no SIMD fast-path despite
`ILKernelGenerator.CountTrueSimdHelper` (a related kernel) existing and being
used elsewhere.

### Should it use ILKernelGenerator?
**Yes.** A `AllSimdHelper<T>(T*, size)` mirroring `CountTrueSimdHelper` would
land within 2-3× of NumPy.

### `AllImplHalf` / `AllImplComplex`
Bespoke paths use `addr[i] == Half.Zero` / `Complex.Zero` directly.
- Contiguous path uses `nd.Address`. `Storage.Address` already accounts for
  the slice offset (because `InternalArray.Slice()` returns a slice with a
  new starting address, and `shape.offset` becomes 0 in that case).
  Verified by tracing addresses on `arange(8)[2:7]`.
- Strided path uses the legacy `NDIterator<T>` (`nd.AsIterator<Half>()`).
  Should reuse `NpyIter` for consistency.

### Missing functionality
- `axis = -1` direct overload exists (`All(NDArray, int)` → delegates with
  `keepdims=false`). Multiple axes (numpy supports `axis=(0,2)`) not
  supported.

---

## File: `src/NumSharp.Core/Backends/Default/Logic/Default.Any.cs`

Structurally identical to `Default.All.cs` — same dispatch, same kernel
interface (`NpyAnyKernel<T>`), same Half/Complex bespoke paths.

### Same issues
- **Same ~13× NumPy gap on contiguous int32** (no SIMD path).
- Same API parity gaps: no `out=`, no `where=`, no `keepdims=` for axis=None.

### Verified behavior
- `np.any([0, 0, 0])` → `False` ✓
- `np.any([0, 0, NaN])` → `True` (NaN truthy) ✓
- Broadcast input → works via iterator.

---

## File: `src/NumSharp.Core/Backends/Default/Logic/Default.IsInf.cs`

### What it does
Thin wrapper — delegates to `ExecuteUnaryOp(a, UnaryOp.IsInf,
NPTypeCode.Boolean)`. 17 lines total, mostly doc comment.

### Parity
- Routes through ILKernelGenerator unary op infrastructure. Integer types
  return all-False (per numpy). Complex IsInf needs validation in the unary
  op kernel, which lives outside Group 4 scope.

### Verdict
No issues in this file specifically. The actual implementation is in
`ILKernelGenerator.Unary.cs` and ditto kernels.

---

## File: `src/NumSharp.Core/Backends/Default/Logic/Default.LogicalReduction.cs`

### What it does
Axis-aware `All`/`Any` with `keepdims`. Uses `NpFunc.Invoke` for dispatch,
`NpyAxisIter.ReduceBool<T, NpyAllKernel<T>>` for the reduction.

### API parity
- Signature: `All(NDArray, int axis, bool keepdims)` — matches NumPy's
  `axis: int, keepdims: bool`.
- Missing: `where=`, `out=`. Single int axis only (NumPy accepts tuple).

### Behavioral parity — verified
- Empty axis: `np.all(zeros((3,0)), axis=1)` → `[True, True, True]` ✓
  (`reduceAll && nd.Shape.dimensions[axis] == 0` path).
- `keepdims=true` correctly reshapes via `result.Storage.Reshape(...)` ✓.
- Scalar with axis=0 → fallback to `All(nd)` ✓.

### Issues
- **`CreateLogicalResultShape` at line 46:** When `keepdims=true`, clones
  `inputShape.dimensions` then sets one dim to 1. Allocates one extra `long[]`
  per call. Minor.
- The shape construction at line 51 calls `Shape.GetAxis(inputShape, axis)`
  which has a quirk: in `Shape.cs:1082`, `if (axis <= -1) axis = dims.Length
  - 1;` — but the axis is already normalized by `NormalizeAxis` at line 29,
  so this is dead-code-but-defensive.

### Performance
Uses NpyAxisIter, which is fine for axis reductions but shares the no-SIMD
limitation of the inner kernel.

---

## File: `src/NumSharp.Core/Backends/Default/Indexing/Default.BooleanMask.cs`

### What it does
Two paths:
1. **SIMD fast path** when both arr and mask are contiguous — uses
   `ILKernelGenerator.CountTrueSimdHelper` for counting, then
   `CopyMaskedElementsHelper<T>` for the copy.
2. **Fallback** uses `NpyIterRef.MultiNew(2, ...)` with a custom
   `BooleanMaskGatherKernel` that respects strides.

### Strided/broadcast handling — verified
- Strided source `(4,3)` from `arange(20).reshape(4,5)[:, ::2]` with mask
  `> 5` → `[7,9,10,12,14,15,17,19]` ✓.
- Broadcast source `broadcast_to(arange(3), (4,3))` with mask `== 1`
  → `[1,1,1,1]` ✓ (uses fallback, IsContiguous=False).
- 1-D mask `(3,)` on 2-D `(3,4)` → selects rows ✓.

### Performance — measured (1M element source, 10 iters)
- NumSharp: 69 ms.
- NumPy: 26 ms.
- → ~2.6× slower. Acceptable.

### Issues
- **Memory:** SIMD path allocates the result up-front sized to `trueCount`,
  uses `CopyMaskedElementsHelper<T>` (good). Fallback uses `Buffer.MemoryCopy`
  per element in `BooleanMaskGatherKernel.Execute`. For dtype Half/Complex
  this calls into `Buffer.MemoryCopy(src, dst, 2/16, 2/16)` per matched
  element, which is roughly 1µs/call overhead. A typed gather kernel would
  help.
- Uses `NpyIter.MultiNew` correctly with `NPY_CORDER` for NumPy boolean
  semantics — good.

---

## File: `src/NumSharp.Core/Backends/Default/Indexing/Default.NonZero.cs`

### What it does
- `NonZero(NDArray)` → `NDArray<long>[]` (one per dimension).
- `CountNonZero(NDArray)` → flat count.
- `CountNonZero(NDArray, int axis, bool keepdims)` → axis reduction.

### Critical performance issue — verified (Severity: **CRITICAL**)
1M-element nonzero benchmark:
- NumSharp: **662 ms** vs NumPy **22.7 ms** → **~29× slower**.

Root cause in `ILKernelGenerator.Masking.cs:194`:
```csharp
var nonzeroCoords = new List<long[]>(initialCapacity);
// ...per element:
var coordsCopy = new long[ndim];
Array.Copy(coords, coordsCopy, ndim);
nonzeroCoords.Add(coordsCopy);
```
For a 1-M element array with ~50% non-zero, this allocates 500 000
`long[ndim]` arrays plus a `List<long[]>`, then in a second pass copies them
out into the per-axis result arrays. The first pass also recomputes offset
per element (`elemOffset = offset + Σ coords[d]*strides[d]`) without using
any incremental coordinate advance.

### Should it use ILKernelGenerator / NpyIter?
**Yes — both paths are missing the SIMD fast path that already exists.**

`NonZeroSimdHelper<T>` is defined in `ILKernelGenerator.Masking.cs:38` but
**dead code** — `Default.NonZero.cs:51` unconditionally calls
`FindNonZeroStridedHelper` for both contiguous and non-contiguous cases. The
SIMD helper would deliver ~5× speedup for contiguous arrays.

### `CountNonZero` axis path
- Inline coordinate-decomposition loop with `outputDimStrides` precomputed in
  `stackalloc`. Reasonable. ✓
- Bounds-checked properly (`sd < outputDims.Length` defensive guard at
  line 108, fine).
- Verified against numpy for shape `(2,2,3)` on all 3 axes with
  `keepdims=true` — outputs match exactly.

### Dtype support
All 15 dtypes via `NpFunc.Invoke`. ✓

---

## File: `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Cast.cs`

### What it does
`Cast(NDArray, NPTypeCode, bool copy)` with 4 special cases:
1. Empty array — wraps a new typed storage.
2. Scalar — uses `NDArray.Scalar(...)`.
3. `(1,)` shape — uses `ArraySlice.Scalar`.
4. General — clones if sliced, then casts buffer.

### Behavioral parity — verified
- Same-type with `copy=true` returns clone ✓
- Same-type with `copy=false` returns original (reference-equals) ✓
- Sliced/transposed view → clones first (line 64-65, 71-72), then casts on
  the materialised contiguous buffer ✓
- `arange(20).reshape(4,5)[:, ::2].astype(double)` → values match numpy ✓
- Transposed `(3,4).T.astype(double)` → values match numpy ✓
- SByte/Half/Complex casts work via `Converts.FindConverter<TIn,TOut>` ✓

### Performance — measured (1M int32 → float64, 10 iters)
- NumSharp: 29 ms
- NumPy: 14 ms
- → ~2× slower. Under the 10× bar but worth flagging.

Root cause: `UnmanagedMemoryBlock.Casting.cs:122`:
```csharp
for (long i = 0; i < len; i++) *(dst + i) = convert(*(src + i));
```
Per-element function-pointer (or virtual delegate) call. No SIMD for
type-pair specialisation. NumPy uses vectorised cast loops per type pair.

### Specific Cast.cs structural issues
- The 4 special-case branches (lines 17-50) duplicate logic. The `(1,)` case
  at line 42-50 fetches `nd.GetAtIndex(0)` which for a sliced view works
  because `Shape.TransformOffset` handles it — but at the cost of going
  through the scalar dispatch.
- Line 23: when `copy=false`, mutates `nd.Storage` and `nd.TensorEngine`
  in-place. This is a sharp edge — caller's NDArray reference now points to
  a different storage. NumPy's `astype(copy=False)` returns the same array if
  no copy is needed; otherwise it returns a new array. NumSharp mutates the
  argument. This is a **behavioural divergence**.

### Per prior audit
Prior audit (Perf 8, line 574-582) flagged the per-element general path.
Confirmed.

### Remediation
- Replace per-element copy with `NpyIter.Copy` which already handles
  cross-dtype.
- Stop mutating `nd` on `copy=false` — return a new wrapper around the same
  storage when types match (NumPy semantics).

---

## File: `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Transpose.cs`

### What it does
- `MoveAxis`, `SwapAxes`, `RollAxis`, `Transpose`.
- `Transpose` (line 127) — O(1) stride-permutation, returns a view.

### What changed on this branch
Diff vs master is **1 line**:
```diff
-return new NDArray(nd.Storage.Alias(newShape));
+return new NDArray(nd.Storage.Alias(newShape)) { TensorEngine = nd.TensorEngine };
```
Preserves the engine on the transposed view. Correct.

### Behavioral parity — verified
- Empty array → returns new array with permuted dims, no data copy ✓
- Repeated axis throws ✓
- Negative axis indices accepted via `check_and_adjust_axis` ✓
- Identity transpose (no permutation) returns same data ✓

### Issues
- `SwapAxes` allocates a `long[]` and then converts to `int[]` for
  `Transpose` (line 95-99). Minor inefficiency.
- `MoveAxis` uses LINQ `.Zip().OrderBy().ThenBy()` — fine for correctness,
  allocates closures.
- The new shape construction at line 193 uses `shape.bufferSize > 0 ?
  shape.bufferSize : shape.size` — this is the bufferSize preservation pattern
  used throughout for views.

### No performance gap
Transpose is O(1) view creation. NumPy is also O(1). Same complexity.

---

## File: `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.NDArray.cs`

### What it does
`CreateNDArray(Shape, Type, Array, char order)` — two overloads (one for
managed `Array`, one for `IArraySlice`). Both are large `switch` blocks over
15 dtypes.

### Issues
- **Dead `_REGEN` template blocks** intermixed with the expanded `#else`
  arms. The `#if _REGEN` regions are pre-template-generation source; only the
  `#else` runs. Total 200+ lines, mostly boilerplate.
- Both overloads accept `char order` (default `'C'`) but pass it directly to
  the `NDArray` constructor without resolution through `OrderResolver`.
  This means callers passing `'A'`/`'K'` would create an NDArray with that
  literal order char (NDArray downstream may resolve or error). Verify
  downstream handles this.

### Dtype coverage
All 15 dtypes via switch. ✓

### Verdict
Functional, but ripe for replacement by `NpFunc.Invoke`.

---

## File: `src/NumSharp.Core/View/Shape.cs`

### What it does
The `readonly partial struct Shape` with cached `ArrayFlags` (1 530 lines).

### Bit values — **VERIFIED bit-for-bit against NumPy**
Read from `src/numpy/numpy/_core/include/numpy/ndarraytypes.h:895-982`:
| Flag | NumSharp | NumPy | Match |
|------|----------|-------|-------|
| `C_CONTIGUOUS` | 0x0001 | 0x0001 | ✓ |
| `F_CONTIGUOUS` | 0x0002 | 0x0002 | ✓ |
| `OWNDATA`      | 0x0004 | 0x0004 | ✓ |
| `ALIGNED`      | 0x0100 | 0x0100 | ✓ |
| `WRITEABLE`    | 0x0400 | 0x0400 | ✓ |
| `BROADCASTED`  | 0x1000 | — (numpy extension)   |

`BROADCASTED` is NumSharp-only — numpy detects broadcast via stride-zero
inspection rather than a flag bit.

### Bug 1: **`OWNDATA` flag is dead** (Severity: HIGH, but inert)
Declared at `Shape.cs:27`. Getter at line 361. Verified by tracing:
```
np.arange(10): OwnsData=False, flags=0x0503
arr.copy():    OwnsData=False
slice[1:5]:    OwnsData=False
```
**No code path sets the `OWNDATA` bit anywhere**. `ComputeFlagsStatic` (line
127) never `OR`s it in. `WithFlags` exists but no caller passes it. Searched
the whole codebase:
```
$ grep -r "OWNDATA" src/NumSharp.Core
src/NumSharp.Core/View/Shape.cs:27:    OWNDATA = 0x0004,
src/NumSharp.Core/View/Shape.cs:361:   get => (_flags & (int)ArrayFlags.OWNDATA) != 0;
```
The `IsView` semantics are handled at *storage* level via `_baseStorage`,
which is independent of the Shape flag. The Shape's `OwnsData` is a stub
that always returns `false`. Should either be removed or wired up.

### Contiguity algorithm — verified against numpy
`ComputeContiguousFlagsStatic` (line 206) follows numpy's
`_UpdateContiguousFlags` exactly:
- Size-0 → both C & F contig ✓ (matches numpy convention)
- Scan right-to-left for C, left-to-right for F, skip dim==1 strides ✓
- Verified:
  - `(1,3,4)` strides `(99999,4,1)` → C=True, F=False (matches numpy)
  - `(3,0,4)` → C=True, F=True (matches numpy empty convention)
  - Transposed `(4,3)` → C=False, F=True ✓
  - Broadcast `(4,3)` stride=(0,1) → C=False, F=False, Broadcasted=True ✓

### Equality bug suspect (line 1397-1418)
`Equals(Shape other)` only compares `dimensions`, not `strides`. Two shapes
with same dims but different strides (e.g. one C-contig, one F-contig) are
considered equal. This may be intentional (logical equality), but NumPy's
ndarray equality also considers strides/dtype. Worth confirming intent.

Also, `operator ==` (line 1353) similarly compares only dims. The
`bufferSize` and `offset` fields are ignored.

### Hash code (line 261-271)
`int hash = layout * 397; ... hash ^= ((int)(size & 0x7FFFFFFF) * 397) *
((int)(v & 0x7FFFFFFF) * 397);`
- `layout` is a constant `'C'` (line 56). Comment says "NOT the physical
  memory order — use Order / IsContiguous / IsFContiguous".
- Hash is order-sensitive within dims order — consistent with equality.

### Mutation through readonly struct (line 758-759)
```csharp
public readonly long this[int dim]
{
    [MethodImpl(Inline)] get => dimensions[dim < 0 ? dimensions.Length + dim : dim];
    [MethodImpl(Inline)] set => dimensions[dim < 0 ? dimensions.Length + dim : dim] = value;
}
```
A `set` on a `readonly struct`'s readonly `long[] dimensions` mutates the
*array contents* (which is allowed because `readonly` only restricts
reassigning the field reference, not mutating what it references). This
breaks the "immutable Shape" promise. Verified — calling `shape[0] = 99`
mutates the underlying dimension array in place, breaking equality, hash,
and any cached `_flags`/`size` because those are computed once at
construction. **Severity: HIGH (latent corruption).**

### `Slice` method (line 1136)
- Parses NumPy slice notation, applies start/step to strides.
- Inherits `WRITEABLE` from parent via `WithFlags` ✓.
- For scalar reduction (line 1180-1187) creates a scalar with the computed
  offset. ✓

### `Clone` method semantics (line 1494)
- `Clone()` returns a copy via `Shape(other)` constructor.
- `Clean()` returns a fresh contiguous shape (offset=0, standard strides).
- The combination `Clone(true, true, true)` is used by `UnmanagedStorage.Cast`
  to materialise logical layout — relies on `unview/unbroadcast` paths.

### Conversion operators (line 1205-1298)
Implicit/explicit conversions for `long[]`, `int[]`, tuples up to 6 dims.
The explicit `(long)shape` returns `Size`, while `(int)shape` overflows for
big arrays — both documented in comments. Reasonable, but generates a wide
public surface.

### Major issues summary
| # | Issue | Severity |
|---|-------|----------|
| 1 | `OWNDATA` flag never set | High |
| 2 | `Shape[i] = x` mutates "immutable" struct's data | High |
| 3 | Equality ignores strides / offset / bufferSize | Medium |
| 4 | `WithFlags`-clear path widely unused | Low |

---

## File: `src/NumSharp.Core/View/OrderResolver.cs`

### What it does
75 lines. Single `Resolve(char order, Shape? source)` returning physical
`'C'` or `'F'`. Internal, used by 14 creation/manipulation files.

### Verified behavior — matches NumPy exactly
| Input | Source | Result | NumPy |
|-------|--------|--------|-------|
| 'C' / 'c' | any | 'C' | 'C' |
| 'F' / 'f' | any | 'F' | 'F' |
| 'A' | null | throws "only 'C' or 'F'" | matches `np.ones(order='A')` |
| 'A' | F-contig only | 'F' | 'F' |
| 'A' | C-contig | 'C' | 'C' |
| 'K' | null | throws | matches |
| 'K' | C-contig | 'C' | 'C' |
| 'K' | F-contig | 'F' | 'F' |
| 'K' | non-contig | 'C' fallback | numpy keeps memory order — different |
| 'X' | any | throws | (numpy: ValueError) |

### Issue 1: `'K'` non-contig fallback
NumPy's `'K'` says "keep the source memory order" — for a non-contig array
that means preserving the iteration order (not just the contiguity flag).
NumSharp's `'K'` falls back to `'C'` for non-contig (line 66). This may
materialise data in C-order when NumPy would not. Behavioural divergence on
transposed/sliced inputs.

### Issue 2: Internal-only
Marked `internal`. Used through the rest of `np.*`. Cannot be called from
user code, which is fine.

### Verdict
Solid for `C`/`F`/`A`; `K` has a minor semantic divergence.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/ArraySlice.cs`

### What it does
Static factory class for `ArraySlice<T>` and `IArraySlice`. 550 lines, heavy
switch dispatch.

### Issues
- **Massive code duplication.** `Allocate` / `FromArray` / `Scalar` / `FromMemoryBlock`
  each have a 15-case switch covering all dtypes. 4 overloads × 15 cases =
  ~60 case branches per file. `_REGEN` blocks exist as dead source.
- **Hidden GCHandle pinning.** `ArraySlice.FromArray(T[] arr, bool copy =
  false)` calls `UnmanagedMemoryBlock<T>.FromArray(arr)` which uses
  `GCHandle.Alloc(arr, GCHandleType.Pinned)`. This pins the managed array for
  the slice lifetime. Mutations to the original `T[]` are visible through
  the slice. Documented behaviour, but a sharp edge for callers expecting
  copy semantics.
- **`Scalar(object val, NPTypeCode typeCode)`** at line 59 uses
  `Converts.ToXxx(val)` per type — handles non-IConvertible types like
  `Half`/`Complex` correctly. ✓
- **17 `FromArray(T[,,,...])` overloads** for 1-D through 15-D managed arrays
  (lines 104-197). Each is generic over `T : unmanaged`. Massive code surface.

### Dtype support
All 15 dtypes in every dispatch path. ✓

### Verdict
Functional but bloated. Prime candidate for `NpFunc.Invoke` refactor.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/ArraySlice` 1.cs

### What it does
650-line `readonly unsafe struct ArraySlice<T>` — the typed slice that wraps
an `UnmanagedMemoryBlock<T>` with `Address`, `Count`, `IsSlice`.

### Strengths
- Bounds checks gated by `Debug.Assert` — zero overhead in Release.
- `Fill(T value)` (line 179) uses `Unsafe.InitBlockUnaligned` for size-1
  types and an unrolled 8-way loop otherwise. Reasonable.
- `Slice(long start, long length)` (line 250) creates a sub-slice without
  copy. ✓
- Long-indexing throughout (`Count` is `long`). Supports >2B arrays.

### Issues
- **Per-element `(T)value` cast in `IArraySlice.this[long index].set`**
  (line 116). For 1M index sets via `IArraySlice` interface, this would box
  the value. Only used via interface dispatch, but still a hot path for
  generic code.
- **`Contains(T item)`** (line 160) — O(N) linear scan with
  `EqualityComparer<T>.Default`. No SIMD. Rarely used.
- **`Clone()`** (line 410) — copies via `UnmanagedMemoryBlock<T>.Copy` which
  uses `Buffer.MemoryCopy`. ✓
- **`DangerousFree()`** (line 572) — comment correctly warns about shared
  memory blocks. But it's `public`, so caller can shoot themselves in the
  foot. Should be marked obsolete or internal.

### Dtype coverage
Generic on `T : unmanaged`. ✓

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedHelper.cs`

### What it does
83-line static extension class on `IMemoryBlock`. 5 `CopyTo` overloads.

### Issues
- **`CopyTo(this IMemoryBlock src, IMemoryBlock dst)`** (line 13) throws
  `InvalidCastException` when typecodes mismatch. No conversion. Fine.
- All overloads use `Buffer.MemoryCopy` (`Buffer.MemoryCopy(src.Address,
  dst.Address, bytes, bytes)`). ✓
- **`countOffsetDestination` arithmetic at line 52**: `(dst.Count -
  countOffsetDestination) * dst.ItemLength` — uses `dst.ItemLength` for the
  destination range, but `bytesCount = src.BytesLength`. If src and dst
  itemsize differ (shouldn't happen with typecode check, but interface allows
  it), this is wrong. The earlier check prevents the bad case.

### Verdict
Tiny, correct.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.cs`

### What it does
Static partial class. Two-method overloads for `FromArray(Array, bool copy)`
and `Allocate(Type, long, [fill])`.

### Issues
- **All 15 dtypes hand-rolled** (the `#else` branch is generated by hand from
  the `_REGEN` template). Same boilerplate.
- `Allocate(Type, long, object fill)` (line 110) routes through
  `Utilities.Converts.ToXxx(fill)` for all 15 dtypes — correctly handles
  `Half`/`Complex` (which don't implement `IConvertible`). ✓

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs`

### What it does
**2 238 lines** of explicit per-type-pair cast switch (15² = 225 cases).

### Performance — measured
- `CastTo<TIn, TOut>(IMemoryBlock<TIn> source)` at line 113 uses
  `Converts.FindConverter<TIn,TOut>()` returning a delegate, then loops
  per-element. JIT can inline the delegate body for monomorphic call sites
  but the loop is still scalar. **No SIMD.**

### Bug: missing dtype coverage in the long-form path
At lines 174-2236 (the inner `IMemoryBlock` overload that takes `source`
parameter without `IMemoryBlock<TIn>`), the inner per-pair switches **omit**
several cases. Verified by counting:
```
$ grep -c "NPTypeCode\.SByte\|NPTypeCode\.Half\|NPTypeCode\.Complex" UnmanagedMemoryBlock.Casting.cs
6
```
Only 6 occurrences across 2 238 lines — most type-pair switches only handle
the *original* 12 dtypes (no SByte/Half/Complex). When SByte/Half/Complex
input or output hits this path, it falls into `default: throw new
NotSupportedException()`.

However, this specific code path (`CastTo<TIn,TOut>(IMemoryBlock source)` at
line 138) is **rarely called** — `Default.Cast` uses
`InternalArray.CastTo(dtype)` which goes through `CastTo(IMemoryBlock, NPTypeCode)`
at line 18, which dispatches to `CastTo<TOut>(IMemoryBlock)` at line 73,
which dispatches to **the typed `IMemoryBlock<TIn>.CastTo<TIn,TOut>()`** at
line 113. The typed path *does* support all 15 dtypes via
`Converts.FindConverter`.

So the bug is real but dormant. Should be cleaned or guarded.

### Verdict
- 2 238 lines of bloat. 
- ~225 small loops × 12-15 conversions each.
- Per-element copy, no SIMD anywhere.
- Suggested replacement: `NpyIter.Copy` (cross-dtype-aware) + 1 SIMD
  per-type-pair specialisation in `ILKernelGenerator`.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock` 1.cs

### What it does
1 100-line generic struct `UnmanagedMemoryBlock<T>` — native memory wrapper
with `Disposer` for ownership tracking.

### Strengths
- **Native memory allocation** uses `NativeMemory.Alloc` (modern .NET 8+)
  with `GC.AddMemoryPressure` for GC awareness ✓.
- **Four allocation types** in `Disposer`: Native, GCHandle, External, Wrap.
  Each cleanly handled in `ReleaseUnmanagedResources`. ✓
- **`FromBuffer(byte[], byteOffset, count, copy)`** at line 477 handles
  arbitrary byte offsets. ✓
- `Fill(T value)` uses `Unsafe.InitBlockUnaligned` for byte-sized types,
  unrolled loop otherwise. ✓
- 17 multidim `FromArray` overloads (1-D through 15-D `T[,,,,…]`) via
  `GCHandle.Alloc(arr, Pinned)`. ✓

### Issues
- **`Reallocate(long length)`** (line 579) calls `Free()` then
  reassigns `this = new UnmanagedMemoryBlock<T>(...)`. Inside a `readonly`
  struct field, this requires the compiler to allow mutation — works because
  the method is non-readonly. **But:** the calling site must be able to
  assign through the reference. Used in `UnmanagedStorage.Reallocate(...)`
  indirectly via `Storage.Reshape(...)`. Verify no readonly-through-readonly
  call.
- **`GetHashCode()`** (line 947) `(int)Count * 397 ^ (int)(long)Address`.
  Truncates long Count and 64-bit address. Distinct memory blocks at
  different addresses with same low 32 bits hash-collide. Minor.
- **`Disposer` is a class inside a value-type struct** — this means the
  struct contains a single reference. Allocation overhead per
  `UnmanagedMemoryBlock<T>` is one class allocation, but it's bounded.

### Verdict
Well-engineered. The `Disposer` pattern correctly manages 4 ownership modes.
GC memory pressure tracking fixes a known bug (#501 per comment).

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.cs`

1 569 lines. Core storage class wrapping `IArraySlice` + `Shape` + cached
typed fields.

### Architecture
15 typed fields `_arrayBoolean`, `_arraySByte`, …, `_arrayComplex` plus
`InternalArray : IArraySlice`. Each constructor (15+ overloads) sets the
correct typed field plus `Address` (byte*) and `Count`.

### `_baseStorage` chain (NumPy `base`)
- Tracks ultimate owner across view chains: A → B(view of A) → C(view of B)
  → all set `_baseStorage = A`, not B.
- Verified at lines 80-128 (`BaseStorage`, `IsView`).

### Issues
- **`DTypeSize` (line 153)**: previously bug (used Marshal.SizeOf for bool =
  4). Now uses `_typecode.SizeOf()` for in-memory size. Per commit
  `e2318d47`, this was the recent fix. ✓
- **`ShapeReference` (line 185)**: returns `ref Shape _shape` for direct
  modification. With Shape being a `readonly struct`, this provides
  field-by-field mutation via the writable indexer (`shape[0] = 99`) — and
  the cached `_flags`/`size`/`_hashCode` become stale. Same root cause as
  Shape.cs Bug 2.
- **`CopyTo(void*)`** (line 1245): switch over 12 dtypes (missing SByte,
  Half, Complex). Comment block was hand-generated; the `_REGEN` template
  would have included all 15. The unsupported types fall to `default: throw
  NotSupportedException()`. **Severity: MEDIUM** — anyone copying a
  Half/SByte/Complex storage to a raw pointer hits NotSupportedException.
- **`CopyTo(IMemoryBlock)`** (line 1356): same 12-dtype switch, same issue.
- **`ToArray<T>`** (line 1531): contiguous fast path uses `src +
  Shape.offset` to start. Verified: when offset=0 (typical case), no
  double-counting. For broadcasted shapes, takes the strided path with
  `shape.GetOffset(coords)` which already includes offset. ✓

### Dtype dispatch
- 15 typed fields ✓
- `SetInternalArray(Array)` switch covers 15 ✓
- `SetInternalArray(IArraySlice)` covers 15 ✓ (with cast type-check)

### Verdict
Core class is solid. Two CopyTo overloads have stale 12-dtype switches
needing SByte/Half/Complex cases added.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Cloning.cs`

### What it does
443 lines covering `Alias` (3 overloads), `AliasAs` (3 overloads), `Cast`,
`CastIfNecessary`, `Clone`, `CloneData`.

### Key paths
- **`Alias()`** / `Alias(Shape)` / `Alias(ref Shape)`: shallow copy of
  storage + shape, sets `_baseStorage` to ultimate owner ✓.
- **`AliasAs<T>()`** (line 152): byte-reinterpret view. Adjusts last dim
  when sizes differ. Requires contiguous if sizes differ. ✓
- **`AliasAs(NPTypeCode)`** (line 238): 15-case dispatch ✓.
- **`Cast<T>()`** (line 276): copies via `CloneData` (materialises logical
  order) then `CastTo<T>()`. Comment correctly notes that casting the raw
  buffer would reorder data for strided/F-contig. ✓
- **`Clone()`** (line 417): uses `CanCloneRawLayout` to detect when raw
  buffer clone is safe (non-broadcast, offset=0, full bufferSize, C or F
  contiguous). Otherwise materialises via `CloneData`. ✓

### Issues
- **`CloneData()`** (line 376): contiguous + offset != 0 path calls
  `InternalArray.Slice(offset, size).Clone()` — two allocations (slice +
  clone). Could be one `UnmanagedMemoryBlock<T>.Copy(address+offset, size)`.
- **`Cast<T>`** at line 282 returns `Clone()` if same type (full copy). Per
  comment, "Always copies". NumPy's `astype(copy=False, same dtype)` returns
  the source unchanged. NumSharp here always copies even when not needed.
  But this is in Storage.Cast, not the user-facing `Default.Cast.Cast`,
  which handles `copy=false` correctly.

### Verdict
Correct semantics; minor allocation overhead.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Getters.cs`

### What it does
729 lines of getters: `GetValue`, `GetAtIndex`, `GetData`, 30 direct
typed-getters (`GetBoolean`, `GetSByte`, …, `GetComplex`).

### `GetValue(int[]) / (long[])`
Both paths cover all 15 dtypes. ✓ Use `_shape.GetOffset(indices)` for offset
calc.

### `GetData(int[]) / (long[]) / (int*, int) / (long*, int)`
Four overloads (lines 155, 203, 270, 327) — each has the same
broadcast/non-contig/contig path tree:
- Broadcast → `CreateBroadcastedUnsafe` view ✓
- Non-contig → `GetView(Slice[])` via `Alias(Shape)` ✓
- Contig → direct memory slice via `InternalArray.Slice` ✓

All four overloads set `_baseStorage = _baseStorage ?? this` for view
chaining. ✓

### Verdict
30 direct typed getters duplicate logic; could be replaced by generic
`Get<T>` + dispatch. But correctness is solid.

---

## File: `src/NumSharp.Core/Backends/Unmanaged/UnmanagedStorage.Setters.cs`

### What it does
1 208 lines of setters: `SetAtIndex`, `SetValue`, `SetData` (3 overloads
each), 30 direct typed-setters.

### `ThrowIfNotWriteable` (line 20)
Calls `NumSharpException.ThrowIfNotWriteable(_shape)`. Used by all public
setters. Broadcast views correctly throw on write attempt ✓ (matches NumPy
`ValueError: assignment destination is read-only`).

### Issues
- **`SetValue(object, int[])`** (line 161) — `#else` branch (line 178-216)
  covers only 12 dtypes — **missing SByte, Half, Complex**. When user sets
  a value on a Half/SByte/Complex array via the `object` overload, falls to
  `NotSupportedException`. Same bug as `UnmanagedStorage.cs` `CopyTo`.
- **`SetValue(object, params long[])`** (line 229) — same 12-dtype switch,
  same missing dtypes.
- **`SetData(NDArray, params long[])`** (line 430): scalar fast-path
  switch at lines 453-468 covers 12 dtypes (no SByte/Half/Complex).
  Fallback path uses `value.Storage.InternalArray.CopyTo(...)` which uses the
  generic helper — so the typed cases are just an optimisation, not a
  correctness bug. But missing.
- Heavy use of `(T)value` casts that **box and unbox** for `object` values.
  For Decimal, this allocates per call.

### Dtype dispatch
30 typed setters (15 dtypes × 2 index types). ✓ for direct typed access.
Object-typed paths missing 3 dtypes.

### Verdict
Same boilerplate sprawl as Getters; correctness gaps in the
`SetValue(object, ...)` paths.

---

## Summary table by severity

| # | Area | Severity | Description | NumPy gap |
|---|------|----------|-------------|-----------|
| 1 | `Default.NonZero.cs` / `ILKernelGenerator.Masking.cs:194` | **CRITICAL** | `NonZero` allocates `List<long[]>` + per-element `long[ndim]` array. Coordinate offset recomputed per element. **~29× slower** than numpy on 1M elements. Dead `NonZeroSimdHelper` exists but unused. | 29× |
| 2 | `Default.All.cs` / `Default.Any.cs` / `NpyAllKernel<T>` | **HIGH** | No SIMD on contiguous path. `EqualityComparer<T>.Default.Equals(val, default)` per element. **~13× slower** than numpy. | 13× |
| 3 | `View/Shape.cs:27,361` | **HIGH** | `OWNDATA` flag declared, getter exists, but no code path **ever sets it**. `Shape.OwnsData` always returns false. Ownership actually tracked at storage level via `_baseStorage`. | inert |
| 4 | `View/Shape.cs:758-759` (indexer `set`) and `UnmanagedStorage.cs:185` (`ShapeReference`) | **HIGH** | `readonly struct Shape` lets callers mutate `dimensions[i]` (the array is readonly-ref, but its contents are not). Cached `_flags`/`size`/`_hashCode` go stale. | — |
| 5 | `UnmanagedStorage.Setters.cs:178-216, 229-272, 453-468` | **MEDIUM** | `SetValue(object, ...)` and `SetData(NDArray,...)` scalar paths missing SByte/Half/Complex cases. `NotSupportedException` on those types via object-typed setters. | — |
| 6 | `UnmanagedStorage.cs:1245-1467` | **MEDIUM** | `CopyTo(void*)` and `CopyTo(IMemoryBlock)` switches miss SByte/Half/Complex. Throws on those types. | — |
| 7 | `Default.Cast.cs:71-75` (`copy=false` path) | **MEDIUM** | Mutates `nd.Storage` and `nd.TensorEngine` of the caller's NDArray when `copy=false`. NumPy returns a new wrapper instead. Side-effecting argument is a sharp edge. | semantic |
| 8 | `UnmanagedMemoryBlock.Casting.cs` (2 238 lines) | **MEDIUM** | Per-element delegate-call cast loop. No SIMD per-type-pair specialisation. ~2× slower than numpy on int32→double 1M. | 2× |
| 9 | `Default.Cast.cs` general path uses per-element `Converts.FindConverter`. | **MEDIUM** | Same as #8. Prior audit flagged as Perf 8. | 2× |
| 10 | `View/OrderResolver.cs:62-66` (`K` non-contig fallback) | **MEDIUM** | `K` on non-contig source falls back to `C` instead of preserving memory order. Behavioural divergence vs numpy on transposed input. | semantic |
| 11 | `View/Shape.cs:1397-1418` (Equals) | **LOW** | Equality compares only `dimensions`, not strides/offset/bufferSize. Two semantically different shapes (e.g. C-contig vs transposed) hash equal. | semantic |
| 12 | `ArraySlice.cs` (~550 lines) and `UnmanagedMemoryBlock.Casting.cs` (~2238 lines) | **LOW** | Massive boilerplate switches that could be `NpFunc.Invoke`. | — |
| 13 | `Default.LogicalReduction.cs:46-53` | **LOW** | `CreateLogicalResultShape` allocates extra `long[]` per call when `keepdims`. Minor. | — |
| 14 | `Default.BooleanMask.cs` fallback gather kernel | **LOW** | Uses `Buffer.MemoryCopy(src, dst, elemSize, elemSize)` per matched element instead of a typed write. Adds 1µs/element overhead. | — |
| 15 | `Default.NDArray.cs` | **LOW** | `CreateNDArray(Shape, Type, Array, char order)` passes `order` to NDArray ctor without `OrderResolver` resolution. Downstream behaviour for `'A'`/`'K'` from this entry point unclear. | — |
| 16 | `ArraySlice.cs` `DangerousFree()` | **LOW** | Publicly callable, can corrupt other live slices over the same MemoryBlock. Should be `internal` or `[Obsolete]`. | — |

### High-level observations
1. **Cached `ArrayFlags` is implemented correctly** for `C_CONTIGUOUS`,
   `F_CONTIGUOUS`, `ALIGNED`, `WRITEABLE`, `BROADCASTED` — bit values match
   NumPy, behaviour verified on edge cases (empty/scalar/strided/
   transposed/broadcast/size-1 dims).
2. **`OWNDATA` is a dead bit** — code-paths for set/clear are missing
   entirely.
3. **Two performance cliffs** (`np.all` 13×, `np.nonzero` 29×) far exceed
   the 10× threshold and warrant immediate SIMD work.
4. **Three CopyTo/SetValue object-overload paths** silently throw on
   SByte/Half/Complex despite the rest of the codebase supporting these dtypes.
5. **OrderResolver is correct for C/F/A** but `K` semantics differ from
   NumPy on non-contig inputs.
6. **Storage class is well-architected** — `_baseStorage` chain, `Disposer`
   ownership tracking, GC memory pressure all solid.
