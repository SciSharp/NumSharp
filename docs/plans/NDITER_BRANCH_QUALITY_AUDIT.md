# NumSharp `nditer` Branch — Quality Audit Followup

**Branch:** `nditer` (compared to `master`)
**Date:** 2026-05-12
**Scope:** File-by-file audit of every changed src/ file (NpyIter migration era)
**Goal:** Catalog correctness bugs, performance regressions, NumPy-parity gaps, and refactor opportunities surfaced during a deep read of the iterator subsystem, kernel generation, math operations, and `np.*` APIs introduced or modified on this branch.

---

## 0. Audit methodology

For each src file changed on this branch we evaluated against the user-specified criteria:

1. **NumPy implementation parity** — Does it match NumPy's structure, not just behavior?
2. **NumPy behavioral parity** — Does running `python -c '...'` produce the same outputs?
3. **Performance across compute cases**:
   - Contiguous
   - Slightly strided / heavily strided
   - Broadcast (stride=0)
   - Scalar-on-one-side / scalar-on-both
   - F-contiguous
   - SIMD-eligible (AVX2/AVX512 via `Vector256`/`Vector512`)
4. **IL generation usage** — Does it avoid switch-per-type and use `ILKernelGenerator` / `NpFunc`?
5. **dtype coverage** — All 15 supported dtypes (`Boolean`, `Byte`, `SByte`, `Int16`, `UInt16`, `Int32`, `UInt32`, `Int64`, `UInt64`, `Char`, `Half`, `Single`, `Double`, `Decimal`, `Complex`)?
6. **API parity** — Same parameters, defaults, semantic edge cases?
7. **Wasted copies** — Does NumSharp copy where NumPy doesn't?
8. **Iterator usage** — Should this code use `NDIterator` / `NpyIter` / `ILKernelGenerator` instead of hand-rolled coordinate loops?
9. **Missing functionality** — Does NumPy expose something we don't?

Approximately 80 src files were read in full plus systematic spot-checks of every changed file's header to confirm patterns. The findings below are organized by **severity** (bugs → perf → missing features → refactor) and each item includes file paths, line numbers, reproduction notes, and concrete remediation plans.

---

## 1. Correctness bugs

### Bug 1 — `np.maximum` / `np.minimum` / `np.fmax` / `np.fmin` collapse to `np.clip`; `fmax`/`fmin` lose NaN-skipping semantics

**Files**
- `src/NumSharp.Core/Math/np.maximum.cs:15-89`
- `src/NumSharp.Core/Math/np.minimum.cs:15-89`

**Symptom**

All four functions broadcast `x1`/`x2` and call `np.clip(_x1, a_min=_x2, a_max=null)` (maximum/fmax) or `np.clip(_x1, a_min=null, a_max=_x2)` (minimum/fmin). The bodies of `fmax`/`fmin` are textually identical to `maximum`/`minimum` — there is no NaN-skipping path.

**NumPy contract**

| Function | NaN behavior |
|---|---|
| `np.maximum(a, b)` | Returns NaN if either operand is NaN ("propagate") |
| `np.fmax(a, b)` | Returns the non-NaN operand; only returns NaN if both are NaN |
| `np.minimum(a, b)` | Returns NaN if either operand is NaN |
| `np.fmin(a, b)` | Returns the non-NaN operand; only returns NaN if both are NaN |

**Reproduction**

```python
>>> import numpy as np
>>> np.maximum(5.0, np.nan)
nan
>>> np.fmax(5.0, np.nan)
5.0
>>> np.maximum(np.nan, 5.0)
nan
>>> np.fmax(np.nan, 5.0)
5.0
```

NumSharp's `clip(5.0, a_min=NaN, a_max=null)` returns `5.0` because IEEE 754 comparisons with NaN are false (so `5 < NaN` is false → no clamp). That gives `np.maximum(5, NaN) == 5`, opposite of NumPy.

**Root cause**

The implementations conflate "elementwise max" with "clip from below". They are different operations:
- `max(a, b)` is symmetric in NaN handling (propagate)
- `clip(x, lower=b)` is asymmetric (NaN in `b` means "no lower bound")

**Remediation**

Implement dedicated SIMD kernels:

1. Add `BinaryOp.Maximum` and `BinaryOp.Minimum` to `Backends/Kernels/KernelOp.cs`.
2. Emit IL via `ILKernelGenerator.EmitVectorOperation` that uses `Vector{W}.Max`/`Vector{W}.Min` — these propagate NaN per IEEE 754 (matches `maximum`/`minimum`).
3. Add scalar fallback using `Math.Max`/`Math.Min` (which propagates NaN in .NET).
4. For `fmax`/`fmin`, compose via `NpyExpr.Where(IsNaN(a), b, Where(IsNaN(b), a, Min/Max(a, b)))` or write a dedicated kernel.

Alternatively, route through `NpyExpr.Max(NpyExpr.Input(0), NpyExpr.Input(1))` which already uses `Math.Min`/`Math.Max` and propagates NaN (see `NpyExpr.cs:632-698`, comment confirms NaN-propagating).

**Severity**

- `maximum`/`minimum`: behavioral mismatch on NaN inputs; results differ from NumPy.
- `fmax`/`fmin`: functionally identical to `maximum`/`minimum`. Users relying on NaN-skipping get wrong answers silently.

**Estimated effort:** ½ day. Add 2 BinaryOp enum members + IL emit + 4 API methods.

---

### Bug 2 — `DefaultEngine.PowerInteger` ignores strides (uses raw indexing on `Unsafe.Address`)

**File:** `src/NumSharp.Core/Backends/Default/Math/Default.Power.cs:55-127`

**Symptom**

```csharp
private static NDArray PowerInteger(NDArray lhs, NDArray rhs)
{
    ...
    var a = (int*)lhs.Unsafe.Address;
    var b = (int*)rhs.Unsafe.Address;
    var d = (int*)result.Unsafe.Address;
    for (long i = 0; i < n; i++) d[i] = PowInt32(a[i], b[i]);
    ...
}
```

`lhs.Unsafe.Address` returns the start of underlying storage. For a sliced or transposed view the storage offset, strides, and base pointer don't line up with linear `i` indexing. The code only checks `lhs.shape.SequenceEqual(rhs.shape)` and `IsInteger()`, both of which are true for many strided arrays.

**Reproduction (Python ground truth)**

```python
>>> import numpy as np
>>> a = np.arange(8).reshape(2, 4)
>>> a[:, ::2]
array([[0, 2],
       [4, 6]])
>>> a[:, ::2] ** 2
array([[ 0,  4],
       [16, 36]], dtype=int64)
```

NumSharp would read storage[0], storage[1], storage[2], storage[3] = 0, 1, 2, 3 instead of storage[0], storage[2], storage[4], storage[6] = 0, 2, 4, 6, giving `[[0, 1], [4, 9]]` — wrong.

**Triggering condition** — strided arrays where `lhs.shape.SequenceEqual(rhs.shape)` and `lhs.GetTypeCode == rhs.GetTypeCode && IsInteger()`.

**Remediation**

Either:

- **Option A (preferred):** Use `NpyIter.MultiNew` with three operands (lhs/rhs/out) and write a typed inner-loop kernel using `INpyInnerLoop`. The iterator handles stride translation automatically.
- **Option B (quick):** Materialize both operands via `lhs.copy('C')` and `rhs.copy('C')` before the loop. Wastes an allocation but correct.
- **Option C (best long-term):** Add `BinaryOp.IntegerPower` to `KernelOp.cs` and emit IL that does repeated-squaring with native wrapping. Then route via the general `ExecuteBinaryOp` path which is stride-aware.

**Test coverage** — Add test based on `np.arange(N).reshape(...)[strided slice] ** k` to `test/NumSharp.UnitTest/Logic/np.power.BattleTest.cs` (or similar), comparing against `subprocess.check_output(["python", "-c", ...])`.

**Estimated effort:** 1 day for Option C (proper fix). 1 hour for Option B (workaround).

---

### Bug 3 — `np.searchsorted` is incomplete and incorrect

**File:** `src/NumSharp.Core/Sorting_Searching_Counting/np.searchsorted.cs:42-98`

**Symptoms**

1. **`TODO currently no support for multidimensional a`** comment on line 43. Multidim `a` falls through unchecked.
2. Function `binarySearchRightmost` is named *Rightmost* but implements *leftmost*:

   ```csharp
   if (val < target) { L = m + 1; }
   else { R = m; }
   ```

   This is the canonical bisect-left algorithm. NumPy's `side='left'` corresponds to bisect-left; `side='right'` corresponds to bisect-right. Function name is wrong AND only left exists.

3. Missing `side` parameter (NumPy: `side='left'` or `'right'`, default `'left'`).
4. Missing `sorter` parameter (NumPy: array of indices presenting a sorted view of `a`).
5. Inner binary search uses `arr.Storage.GetValue(m)` (virtual) and `Converts.ToDouble(...)` (boxing-prone) per iteration.

**NumPy contract**

```python
np.searchsorted(a, v, side='left', sorter=None)
```

Returns indices where `v` should be inserted to keep `a` sorted. `side='left'` returns leftmost suitable index, `'right'` returns rightmost.

**Reproduction**

```python
>>> import numpy as np
>>> a = np.array([1, 2, 2, 3])
>>> np.searchsorted(a, 2, side='left')
1
>>> np.searchsorted(a, 2, side='right')
3
```

NumSharp can only do `side='left'`.

**Remediation**

1. Rename `binarySearchRightmost` → `binarySearchLeft`. Add new `binarySearchRight` (uses `val <= target` instead of `<`).
2. Add `string side = "left"` parameter, dispatch to correct binary search.
3. Reject multidim `a` explicitly (NumPy actually requires 1-D too) — change `arr.size` to validate `arr.ndim == 1`.
4. Add `sorter` parameter (optional `NDArray` of indices).
5. Replace `arr.Storage.GetValue(m)` virtual call with `NpFunc.Invoke<T>(...)` + typed pointer comparison; or generate an IL kernel keyed on dtype.

**Estimated effort:** 1 day for full parity with all four parameters.

---

### Bug 4 — `np.repeat` lacks the `axis` parameter

**File:** `src/NumSharp.Core/Manipulation/np.repeat.cs`

**Symptom**

All overloads ignore axis and flatten the input via `a.ravel()`. NumPy supports `np.repeat(a, repeats, axis=None|int)`:

- `axis=None` (default): flatten then repeat.
- `axis=0`/`axis=1`/...: repeat *along the named axis*, preserving other dims.

**Reproduction**

```python
>>> a = np.arange(6).reshape(2, 3)
>>> np.repeat(a, 2, axis=0)
array([[0, 1, 2],
       [0, 1, 2],
       [3, 4, 5],
       [3, 4, 5]])
>>> np.repeat(a, 2, axis=1)
array([[0, 0, 1, 1, 2, 2],
       [3, 3, 4, 4, 5, 5]])
>>> np.repeat(a, 2)  # axis=None: flatten
array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
```

NumSharp's `np.repeat(a, 2)` always flattens. There is no way to repeat along an axis.

**Remediation**

Add overload `np.repeat(NDArray a, long repeats, int axis)`:

1. Use `NpyAxisIter` or hand-rolled coordinate iteration to walk all non-axis indices.
2. For each axis slice, copy `repeats` times into a contiguously allocated output with shape `(d0, d1, ..., shape[axis]*repeats, ..., dn-1)`.
3. Per-element-count `repeats` along axis: use `axis * repeats[i]` for variable repeats.

**Estimated effort:** ½ day.

---

### Bug 5 — Latent NpyIter bugs called out in source comments

**File:** `src/NumSharp.Core/Backends/Iterators/NpyIter.Execution.cs:42-59`

The execution layer's header lists two bugs that exist in the underlying iterator:

1. **`Iternext()` ignores EXLOOP** — calls `state.Advance()` unconditionally. Callers using `EXTERNAL_LOOP` see NDim-1 extra iterations and read past buffer end. Mitigated in the bridge by using `GetIterNext()` which picks the correct advancer.

2. **Buffered-with-cast stride/element-size mismatch** — after `CopyToBuffer`, the buffer is tight-packed at the buffer dtype, but `Strides[op]` still holds source-array stride. `state.Advance` multiplies by `ElementSizes[op]` (buffer element size) producing wrong pointer delta. Mitigated by routing buffered paths through `BufStrides`.

Both bugs are *acknowledged but unfixed in the underlying `NpyIter.cs`*. The bridge avoids them, but external callers using the iterator directly will hit them.

**Remediation**

Fix in `NpyIter.State.cs:Advance()`:
- Add `if ((ItFlags & (uint)NpyIterFlags.EXLOOP) != 0) return;` early-out (delegate to ExternalLoopNext).
- For buffered: use `BufStrides[op]` (already correct unit) instead of `Strides[op] * ElementSizes[op]` when `(ItFlags & BUFFER) != 0`.

After fix, audit every site using `NpyIterRef.Iternext()` to ensure the new behavior doesn't break callers.

**Estimated effort:** 2 days (fix + audit + regression tests).

---

### Bug 6 — `NDArray.NOT` boxes scalar values

**File:** `src/NumSharp.Core/Operations/Elementwise/NDArray.NOT.cs:22`

```csharp
private static unsafe void NotExecute<T>(...)
    where T : unmanaged, IEquatable<T>
{
    ...
    *(to + i) = (*(from + i)).Equals(default);
}
```

Despite the `IEquatable<T>` constraint, calling `.Equals(default)` on a value-type pointer dereference goes through the boxed virtual call. Other reduction kernels (e.g., `NpyAllKernel<T>` in `NpyLogicalReductionKernels.cs:50`) use `EqualityComparer<T>.Default.Equals` which the JIT devirtualizes.

**Remediation**

```csharp
*(to + i) = EqualityComparer<T>.Default.Equals(*(from + i), default);
```

**Estimated effort:** 5 minutes.

---

## 2. Performance issues (orders of magnitude slower than NumPy)

### Perf 1 — `np.nanmean_axis`, `np.nanstd_axis`, `np.nanvar_axis` allocate `long[]` per element

**Files**
- `src/NumSharp.Core/Statistics/np.nanmean.cs:126-373` (axis paths)
- `src/NumSharp.Core/Statistics/np.nanstd.cs:209-533`
- `src/NumSharp.Core/Statistics/np.nanvar.cs:216-548`

**Symptom**

For `Single`/`Double`/`Half`/`Complex` axis reductions, each function iterates:

```csharp
for (long outIdx = 0; outIdx < outputSize; outIdx++)
{
    var outCoords = new long[outputShape.Length];   // alloc #1
    // decode outIdx into outCoords
    for (long k = 0; k < axisLen; k++)
    {
        var inCoords = new long[inputShape.Length]; // alloc #2 (per inner iter)
        // build inCoords from outCoords + k
        float val = arr.GetSingle(inCoords);        // virtual call
        // accumulate
    }
    // second pass (for std/var) does it AGAIN
    for (long k = 0; k < axisLen; k++)
    {
        var inCoords = new long[inputShape.Length]; // alloc #3
        ...
    }
}
```

For shape `(1000, 1000)` reducing along axis=0:
- 1000 × (1 + 1000 × 2) ≈ 2,001,000 `long[]` allocations
- 2,000,000 virtual `GetSingle`/`GetDouble` calls
- Two passes for std/var doubles the inner work

**Compared to NumPy** — C nditer with SIMD reductions; ~100-1000× faster.

**Compared to NumSharp's own existing infrastructure**

- `Backends/Default/Math/Reduction/Default.Reduction.Nan.cs:ExecuteNanAxisReduction` already wires `ILKernelGenerator.TryGetNanAxisReductionKernel` with stride-aware kernel dispatch.
- `Backends/Default/Math/Reduction/Default.Reduction.Std.cs:ExecuteAxisStdReductionIL` does the IL kernel + ddof adjustment correctly for non-NaN axis.

Both paths are unused by the `np.*` API surface.

**Remediation**

Replace all three `np.nan{mean,std,var}_axis*` implementations (~1500 LoC total) with thin dispatchers:

```csharp
private static NDArray nanmean_axis(NDArray arr, int axis, bool keepdims)
{
    // route to engine for IL-backed nan-axis reduction
    return arr.TensorEngine.NanMean(arr, axis, keepdims);
}
```

Then implement `NanMean(NDArray, int, bool)` in `Default.Reduction.Nan.cs` using existing `ILKernelGenerator.TryGetNanAxisReductionKernel` (already wired for sum/prod/min/max — extend to mean by adding `NanMean` reduction op).

For nanstd/nanvar, follow the two-pass pattern in `Default.Reduction.Std.cs:ExecuteAxisStdReductionIL` but with NaN skipping.

**Estimated effort:** 2-3 days. Reuses ~80% existing infrastructure.

**Expected speedup:** 50-500× on float32/float64 axis reductions for non-tiny arrays.

---

### Perf 2 — `NDArray.argsort<T>` uses LINQ everywhere

**File:** `src/NumSharp.Core/Sorting_Searching_Counting/ndarray.argsort.cs:17-212`

**Symptom**

```csharp
public NDArray argsort<T>(int axis = -1) where T : unmanaged
{
    if (!Shape.IsContiguous)
        return this.copy('C').argsort<T>(axis);  // forced copy

    if (axis == -1) axis = ndim - 1;
    var requiredSize = shape.Take(axis).Concat(shape.Skip(axis + 1)).ToArray();

    if (requiredSize.Length == 0)
    {
        var sorted = LongRange(size)
            .Select(i => new {Data = GetAtIndex<T>(i), Index = i})  // boxes anonymous type, virtual GetAtIndex
            .OrderBy(item => item.Data, NumPyComparer<T>.Instance)
            .Select(item => item.Index)
            .ToArray();
        return np.array(sorted);
    }

    // Multidim case:
    var accessingIndices = AccessorCreatorLong(...);  // IEnumerable<IEnumerable<long>>
    var append = LongRange(shape[axis]);
    var argSort = accessingIndices.Aggregate(Enumerable.Empty<SortedDataLong>(), (acc, seq) =>
    {
        var sortMe = append.Select(value => AppendorLong(value, axis, seq));
        var sortedIndex = SortLong<T>(sortMe);
        return acc.Concat(sortMe.Zip(sortedIndex, (a, b) => new SortedDataLong(a.ToArray(), b)));
    });
    ...
}

private IEnumerable<long> SortLong<T>(IEnumerable<IEnumerable<long>> accessIndex)
{
    long idx = 0;
    var sort = accessIndex.Select(x => new {Data = this[x.ToArray()].GetAtIndex<T>(0), Index = idx++});
    //                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    //                                              creates NDArray view per access!
    return sort.OrderBy(...).Select(...);
}
```

**Allocations per call (shape (1000, 1000), axis=0):**

- 1 forced copy to C-order
- 1000 outer iterations of `AccessorCreatorLong` enumerator chain
- 1000 × 1000 = 1M `this[long[]]` calls (each creates a sliced NDArray view: allocates Shape, Storage wrapper)
- 1M `GetAtIndex<T>(0)` virtual calls
- 1M boxed anonymous-type `{Data, Index}` instances
- 1M `SortedDataLong` instances
- LINQ `OrderBy` allocates intermediate buffer per axis strip

**Estimated:** 4M+ allocations, 6M+ virtual/delegate calls. Each operation is 10-100× slower than NumPy's per-strip C introsort.

**Compared to NumPy** — `numpy/_core/src/multiarray/item_selection.c:partition_introselect_loop` operates directly on stride-aware pointers, no allocations after the result array.

**Remediation**

Rewrite using `LongIntroSort` (already exists at `Utilities/LongIntroSort.cs`) over typed pointers:

```csharp
public unsafe NDArray argsort<T>(int axis = -1) where T : unmanaged
{
    if (axis < 0) axis += ndim;
    if (!Shape.IsContiguous) return this.copy('C').argsort<T>(axis);

    var result = new NDArray(typeof(long), shape);
    long axisLen = shape[axis];
    long axisStride = Shape.strides[axis];
    long outerSize = size / axisLen;

    // Walk all axis-strips
    var iter = new NDCoordinatesAxisIncrementor(ref Shape, axis);
    long* outPtr = (long*)result.Address;
    T* inPtr = (T*)Address;

    do {
        long baseOffset = iter.GetBaseOffset();  // already computed by iterator
        // Sort axis-length-many indices [0..axisLen) by inPtr[baseOffset + i*axisStride]
        // Output into outPtr[baseOffset + i*axisStride] (or strip layout)
        LongIntroSort.ArgSort(inPtr, baseOffset, axisStride, axisLen, outPtr);
    } while (iter.Next());

    return result;
}
```

This avoids all LINQ allocation and uses pointer-stride access. Pair with NaN-aware comparators when `T == float`/`double`/`Complex`.

**Estimated effort:** 2 days (including NaN comparator wiring and tests).

**Expected speedup:** 100-1000×. Shape `(1000, 1000)`: currently ~seconds; should be ~10-50ms.

---

### Perf 3 — `np.linspace` uses virtual `Converts.ToX` per element

**File:** `src/NumSharp.Core/Creation/np.linspace.cs:170-309`

**Symptom**

```csharp
case NPTypeCode.Int32:
{
    unsafe
    {
        var addr = (int*)ret.Address;
        for (long i = 0; i < num; i++)
            addr[i] = Converts.ToInt32(start + i * step);  // virtual call per element
    }
    return ret;
}
```

`Converts.ToInt32(object)` boxes the `double` result of `(start + i * step)` then unboxes inside. For `num = 1_000_000`, this is 1M boxing + 1M virtual calls.

**Compared to** `np.arange.cs` which uses direct typed cast `(int)(start + i * step)` — no virtual call, no box.

**Remediation**

Replace `Converts.ToInt32(...)` with direct casts: `(int)(start + i * step)`, mirroring `np.arange.cs`. Same fix for all 15 type cases.

**Estimated effort:** 1 hour.

**Expected speedup:** 10-30× for integer-dtype linspace.

---

### Perf 4 — `np.searchsorted` virtual call in inner binary search

(See Bug 3 above for parity issues. The performance dimension is separate.)

**Symptom** — `Converts.ToDouble(arr.Storage.GetValue(m))` inside binary search:
- `arr.Storage.GetValue(m)` boxes the typed value to `object`
- `Converts.ToDouble(object)` unboxes via reflection-style dispatch

For binary search over 1M elements, this is ~20 such calls per probe — manageable per-probe. But for `searchsorted(a, v)` with `v.size = 1M`, the outer loop multiplies by 1M, totaling 20M boxed calls.

**Remediation**

Generate an IL kernel keyed on `(arr.dtype, v.dtype)` that does typed-pointer binary search:

```csharp
public static long SearchSortedLeft<TArr, TV>(TArr* a, long size, TV v) where TArr : unmanaged, IComparable<TArr> { ... }
```

Or use `NpFunc.Invoke<TArr>(typeCode, SearchKernel<int>, ...)`.

**Estimated effort:** 1 day combined with Bug 3 fix.

---

### Perf 5 — `Default.Shift` materializes non-contiguous operands

**File:** `src/NumSharp.Core/Backends/Default/Math/Default.Shift.cs`

**Symptom** — Non-contiguous source operand is fully copied to C-order before invoking the IL kernel.

**Remediation** — Either:
- Wire shift through general `ExecuteBinaryOp` path (already stride-aware).
- Generate a strided IL kernel for shifts (uncommon — shifts are integer-only, smaller hot path).

**Estimated effort:** ½ day.

---

### Perf 6 — `np.eye` per-element virtual `flat.SetAtIndex`

**File:** `src/NumSharp.Core/Creation/np.eye.cs:65-67`

```csharp
var flat = m.flat;
for (int i = rowStart; i < rowEnd; i++)
    flat.SetAtIndex(one, (long)i * cols + (i + k));
```

`flat.SetAtIndex(object, long)` boxes `one` and calls a virtual method. For diagonal-rich large matrices (e.g., 100K×100K identity), this is a hot loop.

**Remediation** — Switch on dtype and write directly to typed pointer:

```csharp
switch (typeCode) {
    case NPTypeCode.Int32:
        { int* p = (int*)m.Address;
          for (int i = rowStart; i < rowEnd; i++) p[i*cols + (i+k)] = 1; }
        break;
    ... (all 15 dtypes)
}
```

Or use `NpFunc.Invoke` for dispatch.

**Estimated effort:** 1 hour.

---

### Perf 7 — `Default.ClipNDArray` general path uses `GetAtIndex`/`SetAtIndex`

**File:** `src/NumSharp.Core/Backends/Default/Math/Default.ClipNDArray.cs`

**Symptom** — Fast IL kernel only for contiguous case. General path falls back to per-element virtual access despite the existence of stride-aware NpyIter.

**Remediation** — Route general case through `NpyIterRef.MultiNew` with three operands (a, a_min, a_max) and use `ExecuteElementWise` with appropriate scalar/vector bodies.

**Estimated effort:** 1 day.

---

### Perf 8 — `Default.Cast` 4-branch ladder

**File:** `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Cast.cs`

**Symptom** — Branches on `(isScalar, isContiguous, isSliced, isGeneral)` calling separate paths. Acceptable but the general path uses per-element copy.

**Remediation** — Use `NpyIter.Copy` (already exists in NpyIter.cs:3220) for cross-dtype copy. It handles all stride/broadcast/cast cases.

**Estimated effort:** ½ day.

---

## 3. Missing NumPy API parity

### Missing 1 — Functions not implemented at all

Per `.claude/CLAUDE.md` "Missing Functions (18)":

| Category | Functions |
|---|---|
| Sorting | `np.sort` |
| Manipulation | `np.flip`, `np.fliplr`, `np.flipud`, `np.rot90`, `np.pad` |
| Splitting | `np.split` ✓ (now exists), `np.array_split` ✓, `np.hsplit` ✓, `np.vsplit` ✓, `np.dsplit` ✓ |
| Diagonal | `np.diag`, `np.diagonal`, `np.trace` |
| Cumulative | `np.diff`, `np.gradient`, `np.ediff1d` |
| Rounding | `np.round` (only `round_`/`around` exist) |

*(Splits exist on this branch — strike from the missing list.)*

### Missing 2 — Existing function parameter gaps

| Function | Missing |
|---|---|
| `np.repeat` | `axis` parameter (Bug 4) |
| `np.searchsorted` | `side`, `sorter`, multidim `a` (Bug 3) |
| `np.fmax`/`np.fmin` | actual NaN-skipping semantics (Bug 1) |
| `np.argsort` | `kind` ('quicksort'/'mergesort'/'stable'/'heapsort'), `order` (for structured arrays) |

### Missing 3 — Operator/method coverage

The `Operations/Elementwise/NDArray.Primitive.cs` (lines 1-43) provides operator overloads but:
- `**` (Power) operator — NumSharp uses C# `^`? Not present here. `np.power` works but `arr ** 2` doesn't.
- `//` (FloorDivide) — same.

Workaround exists via `np.power`/`np.floor_divide` so this is API ergonomics, not a correctness gap.

---

## 4. Refactor / architectural cleanup

### Refactor 1 — Two parallel axis iterators (`NpyIter` vs `NpyAxisIter`)

`NpyAxisIter` (`MaxDims = 64`) is used for axis reductions; `NpyIter` supports unlimited dims. The cap is rarely hit in practice but creates a parallel implementation surface.

**Proposal** — Migrate axis-reduction kernels to use `NpyIter.MultiNew` with `iterShape` reducing the target axis to 1 (via `op_axes` with -1 entry). This consolidates to one iterator implementation.

**Estimated effort:** 1 week for full migration + tests.

---

### Refactor 2 — `np.nan*` family redundancy

After Perf 1 fix, the three nanmean/nanstd/nanvar files shrink from ~1500 LoC total to ~150 LoC. The Half/Complex special cases largely fold into the IL nan reduction infrastructure.

---

### Refactor 3 — Type dispatch consistency

Many files mix patterns:
- `np.repeat` uses C# `switch` per dtype.
- `NDArray.NOT` uses `NpFunc.Invoke` for dispatch.
- `Default.Cast` uses ladder + `NpFunc`.

**Proposal** — Adopt `NpFunc.Invoke` everywhere for monomorphic dispatch. The cache amortizes reflection cost.

---

### Refactor 4 — Wasted contiguity coalescing

NpyIter has a known limitation: when MULTI_INDEX flag is set, axes can't be coalesced even when contiguous. This is correct (multi-index requires original axis structure) but blocks performance for `GetMultiIndex`-style API users. NumPy's nditer offers the same trade-off.

No action needed; documented behavior.

---

### Refactor 5 — `NpyIter` is 3,469 LoC in one file

Splitting into:
- `NpyIter.Construction.cs` (Initialize, AllocateDimArrays, broadcast/iter shape resolution, op_axes)
- `NpyIter.Mutation.cs` (RemoveAxis, RemoveMultiIndex, ResetBasePointers)
- `NpyIter.MultiIndex.cs` (GetMultiIndex, GotoMultiIndex, GetIndex, GotoIndex, CreateCompatibleStrides)
- `NpyIter.Lifecycle.cs` (Copy, Dispose, ReleaseState, FreeState)
- `NpyIter.Debug.cs` (DebugPrint and helpers)

Would make the file navigable. Currently a partial class is already split into `.cs`, `.State.cs`, `.Execution.cs`, `.Execution.Custom.cs` — extend the pattern.

---

## 5. Test coverage gaps

Tests we should add to lock in fixes:

### Bug-validation tests

```csharp
[TestMethod] public void Maximum_PropagatesNaN()      // np.maximum(5, NaN) == NaN
[TestMethod] public void FMax_SkipsNaN()              // np.fmax(5, NaN) == 5
[TestMethod] public void Minimum_PropagatesNaN()
[TestMethod] public void FMin_SkipsNaN()
[TestMethod] public void PowerInteger_RespectsStrides() // arange(8).reshape(2,4)[:, ::2] ** 2
[TestMethod] public void SearchSorted_SideLeft()       // ([1,2,2,3], 2, 'left') == 1
[TestMethod] public void SearchSorted_SideRight()      // ([1,2,2,3], 2, 'right') == 3
[TestMethod] public void SearchSorted_Sorter()         // with permutation
[TestMethod] public void Repeat_Axis0()                // arange(6).reshape(2,3).repeat(2, axis=0)
[TestMethod] public void Repeat_Axis1()                // arange(6).reshape(2,3).repeat(2, axis=1)
[TestMethod] public void NOT_DoesNotBox()              // perf-sensitive
```

### Performance regression tests

```csharp
[TestMethod] [Performance("NanMeanAxis_1000x1000_Lt_50ms")]
[TestMethod] [Performance("ArgSort_1000x1000_Lt_50ms")]
[TestMethod] [Performance("Linspace_1M_Int_Lt_10ms")]
[TestMethod] [Performance("Eye_10000_Lt_20ms")]
```

(Performance attribute would need infra. Alternative: assert via stopwatch with generous bounds.)

### NumPy-parity battle tests

Add reproductions to `np.power.BattleTest.cs`, `np.maximum.BattleTest.cs`, `np.fmax.BattleTest.cs`, `np.nanmean.BattleTest.cs`, `np.argsort.BattleTest.cs`, `np.searchsorted.BattleTest.cs`, `np.repeat.BattleTest.cs` that shell out to Python and diff output. Pattern exists in `test/NumSharp.UnitTest/Logic/np.where.BattleTest.cs`.

---

## 6. Priority-ordered action plan

### Critical (correctness)

| # | Issue | Effort | Risk |
|---|---|---|---|
| 1 | Bug 1: `np.maximum`/`fmax` NaN handling | ½ d | Medium - changes user-visible NaN behavior |
| 2 | Bug 2: `PowerInteger` strides | 1 d | Low - rare integer pow on strided |
| 3 | Bug 3: `np.searchsorted` left/right + sorter | 1 d | Low - adds parameters with safe defaults |
| 4 | Bug 4: `np.repeat` axis parameter | ½ d | Low - new overload |
| 5 | Bug 5: NpyIter latent bugs | 2 d | High - regressions if not careful |
| 6 | Bug 6: NOT boxing | 5 m | Zero |

### High value (perf)

| # | Issue | Effort | Speedup |
|---|---|---|---|
| 7 | Perf 1: nanmean/nanstd/nanvar axis | 2-3 d | 50-500× |
| 8 | Perf 2: argsort LINQ | 2 d | 100-1000× |
| 9 | Perf 3: linspace boxing | 1 h | 10-30× |
| 10 | Perf 4: searchsorted boxing | (with Bug 3) | 5-20× |
| 11 | Perf 5: shift materialize | ½ d | 2-5× |
| 12 | Perf 6: eye boxing | 1 h | 10-20× |
| 13 | Perf 7: clip general path | 1 d | 5-15× |
| 14 | Perf 8: cast general path | ½ d | 2-5× |

### Architectural (refactor)

| # | Issue | Effort |
|---|---|---|
| 15 | Refactor 1: consolidate axis iterators | 1 wk |
| 16 | Refactor 2: collapse nan* redundancy (rides on Perf 1) | bundled |
| 17 | Refactor 3: NpFunc dispatch consistency | 1 d |
| 18 | Refactor 5: split NpyIter.cs | ½ d |

### Net new (missing APIs)

| # | API | Effort |
|---|---|---|
| 19 | `np.sort` | 2 d |
| 20 | `np.flip` family | ½ d each |
| 21 | `np.rot90`, `np.pad` | 1 d each |
| 22 | `np.diag`, `np.diagonal`, `np.trace` | 1 d each |
| 23 | `np.diff`, `np.gradient`, `np.ediff1d` | 1 d each |
| 24 | `np.round` (alias for `round_`) | 10 m |

---

## 7. Out of scope (acknowledged, not addressed)

These are noted but considered acceptable for current scope:

- **`NpyAxisIter` 64-dim limit** — practical limit, rarely hit.
- **NDArray.argsort returning long indices via custom `SortedDataLong`** — semantic correctness is fine; only perf is bad.
- **`np.linspace`'s `_REGEN` Regen template comments** — historical artifact, doesn't affect correctness.
- **Lack of `Complex64`** — single-precision complex maps to complex128 (per `np.frombuffer.cs:720`). Documented as a deliberate design choice.
- **`np.cumsum`/`np.cumprod` axis behaviors** — verified clean (`Default.Reduction.CumAdd.cs`).
- **Multi-dim `np.tile`** — verified clean.

---

## 8. Files audited (sample list, ~80 files read)

**Iterators (full):**
NpyIter.cs (3469 lines), NpyIter.State.cs, NpyIter.Execution.cs, NpyIter.Execution.Custom.cs, NpyExpr.cs, NpyIterCasting.cs, NpyIterCoalescing.cs, NpyIterBufferManager.cs, NpyIterFlags.cs, NpyIterKernels.cs, NpyLogicalReductionKernels.cs, NpyNanReductionKernels.cs, NpyAxisIter.cs, NpyAxisIter.State.cs, INDIterator.cs, NDIterator.cs, NDIteratorExtensions.cs.

**IL kernels:**
ILKernelGenerator.cs (core), ILKernelGenerator.Binary.cs.

**Engines:**
DefaultEngine.BinaryOp.cs, .CompareOp.cs, .UnaryOp.cs, .ReductionOp.cs, .BitwiseOp.cs.

**Reductions:**
Default.Reduction.{Add, Product, Mean, ArgMax, ArgMin, AMax, AMin, Std, Var, Nan, CumAdd, CumMul}.cs.

**Math ops:**
Default.{Abs, ATan2, Ceil, Floor, Truncate, Reciprocal, Negate, Sqrt, Exp, Log, Power, Shift, ClipNDArray, Clip}.cs.

**BLAS:**
Default.Dot.cs, .Dot.NDMD.cs, .MatMul.cs, .MatMul.2D2D.cs, .MatMul.Strided.cs.

**Logic:**
Default.{All, Any, IsInf, LogicalReduction, AllClose, IsClose, IsFinite, IsNan}.cs.

**Indexing:**
Default.BooleanMask.cs, .NonZero.cs.

**Shape/Storage:**
View/Shape.cs (full), OrderResolver.cs, Backends/NDArray.cs (1403 lines), UnmanagedStorage.cs, .Cloning.cs, UnmanagedHelper.cs, ArraySlice`1.cs, UnmanagedMemoryBlock`1.cs.

**Creation:**
np.{array, arange, empty, eye, linspace, frombuffer, copy}.cs, NDArray.Copy.cs, NdArray.ReShape.cs.

**Manipulation:**
NDArray.flatten.cs, np.{ravel, expand_dims, tile, repeat, split, unique}.cs, NDArray.unique.cs.

**APIs:**
np.{cs, cumsum, finfo, fromfile, iinfo, where, all, any, allclose, array_equal, can_cast}.cs.

**Statistics:**
np.{nanmean, nanstd, nanvar, nansum}.cs.

**Math API:**
np.{maximum, minimum, modf, round, clip}.cs, NDArray.sum.cs.

**Sorting:**
np.{argsort, argmax, searchsorted}.cs, ndarray.argsort.cs.

**Operations:**
NDArray.{Primitive, Shift, NOT, NotEquals, Equals, Greater, Lower, AND, OR, XOR, BitwiseNot}.cs (representative subset).

**Selection:**
NDArray.Indexing.cs.

**Utilities (sample):**
ArrayConvert.cs (partial), Arrays.cs, InfoOf.cs, NumberInfo.cs, Properties.cs.

**Generics:**
NDArray\`1.cs.

---

## 9. Suggested follow-up issues to file

Recommended GitHub issues to file in priority order:

1. **`np.maximum`/`np.minimum`/`np.fmax`/`np.fmin` NaN handling is wrong** (Bug 1)
2. **`np.power` ignores strides for integer same-shape inputs** (Bug 2)
3. **`np.searchsorted` is incomplete: missing `side`, `sorter`, multidim** (Bug 3)
4. **`np.repeat` missing `axis` parameter** (Bug 4)
5. **NpyIter has latent Iternext + buffered-cast bugs** (Bug 5)
6. **`np.nan{mean,std,var}` axis paths are 100-1000× slower than NumPy** (Perf 1)
7. **`np.argsort` LINQ implementation is 100-1000× slower than NumPy** (Perf 2)
8. **`np.linspace` integer dtypes box per element** (Perf 3)
9. **`np.eye` boxes per diagonal element** (Perf 6)
10. **Track: missing NumPy functions (`np.sort`, `np.flip`, `np.diag`, `np.diff`, ...)**

Each issue should include:
- Minimal reproduction (Python ground truth + NumSharp output)
- File:line reference
- Suggested remediation linking back to this document
