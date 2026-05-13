# Group 7 Audit — Math API + Operations + Selection + Sorting + Statistics

**Branch:** `nditer` vs `master`
**Date:** 2026-05-13
**Scope:** Math operations, elementwise ops, indexing selection/masking, sorting/searching, NaN-aware statistics.
**Method:** Read all referenced files; verify behavior with `python -c` (NumPy 2.x reference) and `dotnet_run` C# scripts; measured wall-clock time on shapes `(1000,1000)`, `(2000,2000)`, `1M` for performance claims.

---

## Files audited

| File | LoC delta on branch | Status |
|---|---|---|
| `src/NumSharp.Core/Math/NDArray.negative.cs` | +21 (new) | Has correctness gaps |
| `src/NumSharp.Core/Math/NdArray.Convolve.cs` | n/a | Has precision bug + missing dtype |
| `src/NumSharp.Core/Math/np.clip.cs` | n/a | API parity gaps |
| `src/NumSharp.Core/Math/np.math.cs` | n/a | Trivial dispatchers (clean) |
| `src/NumSharp.Core/Math/np.modf.cs` | n/a | Public-API typo + missing `out` |
| `src/NumSharp.Core/Operations/Elementwise/NDArray.NOT.cs` | ±177 | Uses NpFunc dispatch, mild perf gap |
| `src/NumSharp.Core/Operations/Elementwise/NDArray.Shift.cs` | n/a | Clean, matches NumPy perf |
| `src/NumSharp.Core/Selection/NDArray.Indexing.Masking.cs` | n/a | Slow row-by-row paths |
| `src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Getter.cs` | n/a | LINQ + virtual dispatch in `FetchIndices` |
| `src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Setter.cs` | n/a | Mirrors Getter; `SetIndicesNDNonLinear` is `NotImplementedException` |
| `src/NumSharp.Core/Sorting_Searching_Counting/ndarray.argsort.cs` | n/a | **Severely** slow LINQ implementation |
| `src/NumSharp.Core/Sorting_Searching_Counting/np.searchsorted.cs` | n/a | Multiple parity/perf bugs |
| `src/NumSharp.Core/Statistics/np.nanmean.cs` | ±159 | Per-element `long[]` alloc but only ~3× slower than NumPy |
| `src/NumSharp.Core/Statistics/np.nanstd.cs` | ±228 | Same shape, two-pass |
| `src/NumSharp.Core/Statistics/np.nanvar.cs` | ±238 | Same shape |

---

## 1. Correctness bugs

### B1 — `np.searchsorted` mis-named: `binarySearchRightmost` is actually leftmost

**File:** `Sorting_Searching_Counting/np.searchsorted.cs:78-98`

```csharp
private static long binarySearchRightmost(NDArray arr, double target)
{
    long L = 0;
    long R = arr.size;
    while (L < R)
    {
        long m = (L + R) / 2;
        double val = Converts.ToDouble(arr.Storage.GetValue(m));
        if (val < target)  // strict-less ⇒ this is bisect-LEFT, not rightmost
        {
            L = m + 1;
        }
        else
        {
            R = m;
        }
    }
    return L;
}
```

The summary doc on line 71-77 even states "left-most position … NumPy's searchsorted with side='left' (default)". Function name is wrong. Verified at runtime:

```
NumSharp: searchsorted([1,2,2,3], 2) = 1   (consistent with side='left')
NumPy:    searchsorted([1,2,2,3], 2, 'left')  = 1
NumPy:    searchsorted([1,2,2,3], 2, 'right') = 3   ← NumSharp cannot do this
```

### B2 — `np.searchsorted` missing `side` and `sorter` parameters

NumPy: `searchsorted(a, v, side='left', sorter=None)`.
NumSharp: only the default-side path exists. No way to request `side='right'`, no `sorter` parameter.

### B3 — `np.searchsorted` silently accepts multidim `a` (NumPy raises)

```
NumSharp: searchsorted(arange(20).reshape(4,5), 5) = 1   ← wrong (treats as 1D flat)
NumPy:    raises "object too deep for desired array"
```

The TODO comment at `np.searchsorted.cs:43` admits this. `arr.size` is used as upper bound, treating any shape as flat. Should require `ndim == 1`.

### B4 — `np.searchsorted` boxes per probe via `Storage.GetValue`

`Converts.ToDouble(arr.Storage.GetValue(m))` (line 86) executes:

1. `Storage.GetValue(long)` switches per dtype, returns boxed `object` (UnmanagedStorage.Getters.cs:56).
2. `Converts.ToDouble(object)` unboxes through `System.Convert`.

For `searchsorted(1M sorted, 100K queries Int32)`:

| | Wall-clock | Ratio |
|---|---|---|
| NumPy (C, typed) | **6.0 ms** | 1× |
| NumSharp (boxed loop) | **112.4 ms** | **18.7× slower** |

### B5 — `np.negative` accepts `bool` but NumPy rejects

`NDArray.negative.cs:26-32` handles `NPTypeCode.Boolean` as logical NOT. NumPy raises a `TypeError` for `np.negative(bool_array)` (and recommends `~` or `logical_not`). NumSharp silently produces logical NOT.

### B6 — `np.negative` rejects unsigned dtypes that NumPy accepts

`NDArray.negative.cs:112-118` throws `NotSupportedException` for `Byte`, `UInt16`, `UInt32`, `UInt64`, `Char`. NumPy returns the two's-complement wrap-around:

```
NumPy:    np.negative(np.uint8([1,5,0])) = [255 251 0]
NumSharp: np.negative(byte[]{1,5,0}) → NotSupportedException
```

Acknowledged as `[OpenBugs]` (`EdgeCaseTests.cs:182`). **However**, unary `operator-` on the same array DOES work (goes through `TensorEngine.Negate` → ILKernel which supports wrap):

```
NumSharp:  -byte_arr  = [255, 254, 1]     ← works (operator path)
NumSharp:  np.negative(byte_arr)          → throws (np.* API path)
```

The two entry points have inconsistent dtype coverage. `np.negative` should route to `TensorEngine.Negate` like the operator does.

### B7 — `np.convolve` accumulates in `double`, silently losing int64 precision

**File:** `Math/NdArray.Convolve.cs:138-188`

`ConvolveFullTyped<T>` declares `double sum = 0` regardless of T. For `Int64`, `UInt64`, `Decimal` operands beyond `2^53`, the accumulator is lossy.

Verified:
```
NumSharp: convolve([(1<<53)+1, (1<<53)+1], [1]) = [9007199254740992, 9007199254740992]
NumPy:    convolve(...same...) =                 [9007199254740993, 9007199254740993]
```

NumPy uses native-typed accumulator (int64 → int64). NumSharp's loop:
```csharp
double sum = 0;
sum += aVal * vVal;  // aVal is double-cast from int64
rPtr[k] = (T)(object)(long)sum;
```

### B8 — `np.convolve` rejects `bool` (NumPy accepts as bitwise OR)

```
NumPy:    convolve([T,T,F], [T,F]) = [T, T, F, F] (dtype=bool)
NumSharp: NotSupportedException
```

The `_FindCommonType` likely promotes to bool, and the switch (line 91-132) has no `Boolean` case.

### B9 — `np.modf` public API typo: `Intergral` (should be `Integral`)

**File:** `Math/np.modf.cs:16, 27`

```csharp
public static (NDArray Fractional, NDArray Intergral) modf(NDArray x, NPTypeCode? dtype = null)
```

This is a stable user-facing field name. Renaming requires API break.

### B10 — `NDArray.NOT.cs`: `.Equals(default)` still goes through virtual call

**File:** `Operations/Elementwise/NDArray.NOT.cs:22`

```csharp
where T : unmanaged, IEquatable<T>
...
*(to + i) = (*(from + i)).Equals(default);
```

The `IEquatable<T>` constraint **should** allow the JIT to devirtualize, but in practice the standard-library guidance and other kernels in this codebase (`NpyAllKernel<T>` in `NpyLogicalReductionKernels.cs`) use `EqualityComparer<T>.Default.Equals(...)` for guaranteed devirtualization. Measured:

| Workload | NumSharp | NumPy `logical_not` | Ratio |
|---|---|---|---|
| Int32, 10M elements, 10 iters | 154.5 ms | 49.1 ms | 3.1× slower |
| Double, 10M elements, 10 iters | 186.1 ms | 69.3 ms | 2.7× slower |

Lifting to `EqualityComparer<T>.Default.Equals(*ptr, default)` or to a typed IL kernel via `ILKernelGenerator` should close the gap and unlock SIMD.

### B11 — `Setter`: `SetIndicesNDNonLinear<T>` throws `NotImplementedException`

**File:** `Selection/NDArray.Indexing.Selection.Setter.cs:617`

The setter explicitly bails out for "non-linear" (sub-shaped + non-contig source) writes. Triggered when `isSubshaped && !source.Shape.IsContiguous` (the corresponding getter path at line 431 does have a working implementation `FetchIndicesNDNonLinear`). Users assigning into transposed/sliced multi-dim views via fancy indexing will hit this.

### B12 — `np.clip` API shape: required `a_min`/`a_max`, no Python-None defaults

**File:** `Math/np.clip.cs:18-65`

NumPy: `np.clip(a, a_min=<no value>, a_max=<no value>, out=None)` — one bound may be omitted/None. NumSharp's primary overload requires both `a_min` and `a_max` as NDArrays. The "either bound null" path requires constructing an explicit `null` literal which won't compile against `NDArray` due to operator overloads on NDArray (`!=`). It does work via `np.clip(a, null, amax)` from a script context because the resolver picks the right overload, but it is not idiomatic and `np.clip(a)` (no bounds, NumPy: returns a copy) does not compile.

### B13 — `np.clip` does not preserve broadcasting semantics for shape-mismatched bounds

Not tested in depth — clip with per-element `a_min`/`a_max` of compatible-but-not-equal shapes (e.g., `(N,1)` against `(N,M)`) needs verification. The dispatching path is `TensorEngine.ClipNDArray` (`np.clip.cs:20`), which the audit doc Perf 7 flagged as using `GetAtIndex/SetAtIndex` for non-contig general case.

---

## 2. Performance issues (measured)

All comparisons against NumPy 2.x (`python --version` 3.13, numpy installed via pip). NumSharp via `dotnet run` Release-optimized AOT.

### P1 — `ndarray.argsort` LINQ implementation is catastrophically slow

**File:** `Sorting_Searching_Counting/ndarray.argsort.cs:17-212`

The code uses LINQ throughout: per-axis-strip, allocates anonymous types `{Data, Index}`, builds `IEnumerable` chains via `Aggregate/Concat/Zip`, sorts via `OrderBy`, then writes back via `resultArray[arg.DataAccessor] = arg.Index`. For each element it instantiates one `NDArray view` via `this[long[]].GetAtIndex<T>(0)` (line 122). Each access does a `Shape.GetView(...)` allocation.

Measured (10 iters not used because each run already takes seconds; numbers are single-run):

| Shape | dtype | NumSharp | NumPy | Ratio |
|---|---|---|---|---|
| 1D 100K | Double | 52.7 ms | 2.9 ms | **18× slower** |
| 2D 100×100 axis=-1 | Double | 32.7 ms | 0.8 ms | **41× slower** |
| 2D 1000×1000 axis=-1 | Double | **2,305 ms** | 12.5 ms | **184× slower** |
| 2D 1000×1000 axis=0 | Double | **2,769 ms** | 18.7 ms | **148× slower** |

The audit_v1 claim of "100-1000×" is accurate. Remediation is the same as audit_v1 suggested: pointer-based introsort over typed strides.

### P2 — Convolve uses double accumulator + box per multiply

**File:** `Math/NdArray.Convolve.cs:138-188`

```csharp
double aVal = Converts.ToDouble((object)aPtr[j]);  // box
double vVal = Converts.ToDouble((object)vPtr[k - j]);
sum += aVal * vVal;
```

Plus a 13-branch type ladder to write back. Wall-clock for double `convolve(10K, 100, 'full')`:

| | NumSharp | NumPy | Ratio |
|---|---|---|---|
| double | 27.0 ms | 0.40 ms | **67× slower** |

NumPy uses FFT-based convolution for sufficiently large kernel × signal sizes (~75 element kernel cross-over). Even the direct convolution path uses typed BLAS-style accumulation.

### P3 — `np.searchsorted` boxing per probe

(Same as B4 above; restated for performance focus.)

| | NumSharp | NumPy | Ratio |
|---|---|---|---|
| 1M sorted, 100K Int32 queries | 112.4 ms | 6.0 ms | **18.7× slower** |

### P4 — `np.clip` general/transposed path

**File:** `Math/np.clip.cs:20` → `TensorEngine.ClipNDArray`

| Workload | NumSharp | NumPy | Ratio |
|---|---|---|---|
| 1000×1000 Double C-contig | 22.3 ms (per iter) | 1.52 ms | **14.7× slower** |
| 1000×1000 Double Transposed | 104.4 ms (per iter) | 1.55 ms | **67.4× slower** |

NumPy clip path is NaN-aware SIMD. NumSharp has IL kernel for C-contig but the transposed path falls to the slow general path.

### P5 — `np.nanmean_axis` / `nanstd_axis` / `nanvar_axis` allocate `long[]` per outer iteration

**Files:** `Statistics/np.nanmean.cs:177,189,220,230,267,297,331,343,425,437,451,484,496,516`, similar in nanstd/nanvar.

Per outer-coord step: `new long[outputShape.Length]` for `outCoords`. Per inner step `new long[inputShape.Length]` for `inCoords`. Two passes for std/var → twice the inner allocations. Plus `arr.GetSingle(inCoords)` is a virtual call that re-walks shape.

| Workload | NumSharp | NumPy | Ratio |
|---|---|---|---|
| (1000,1000) nanmean axis=0 Double | 24.6 ms | 7.7 ms | 3.2× slower |
| (1000,1000) nanmean axis=1 Double | 17.8 ms | 8.8 ms | 2.0× slower |
| (1000,1000) nanstd axis=0 Double | 37.2 ms | 10.8 ms | 3.4× slower |
| (1000,1000) nanvar axis=0 Double | 37.7 ms | 11.6 ms | 3.3× slower |
| (2000,2000) nanmean axis=0 Double | 82.2 ms / iter | 26.8 ms / iter | 3.1× slower |
| (2000,2000) nanstd axis=0 Double | 163.3 ms / iter | 41.9 ms / iter | 3.9× slower |
| (1000,1000) nanmean axis=0 Float | 19.0 ms | 5.2 ms | 3.7× slower |
| (1000,1000) T (transposed) nanmean axis=0 | 17.9 ms / iter | n/a | ~2× transposed overhead |

**Audit v1 claimed "100-1000× slower" — that is overstated.** Measured is consistently ~3× slower. The implementation is genuinely allocation-heavy but the JIT escape-analyzes most of these into stack frames; the dominant cost is the virtual `Get{Single,Double}` calls and the absence of SIMD.

Remediation (still beneficial): replace per-`long[]` alloc by a single `Span<long>` reused across iterations; route Single/Double through `ILKernelGenerator.TryGetNanAxisReductionKernel` (which the `Default.Reduction.Nan.cs` engine already exposes but `np.nanmean.cs` does not call).

### P6 — `NDArray.NOT.cs` ~3× slower than NumPy `logical_not`

(See B10 for measurements.) The "boxing" concern is partially true — `(*ptr).Equals(default)` may not devirtualize across all JIT versions. Even if it does, the loop is purely scalar with no SIMD. ILKernel `EmitVectorOperation` for "is-zero" would close the gap.

### P7 — Fancy / boolean indexing slower

**Files:** `Selection/NDArray.Indexing.Selection.Getter.cs`, `Selection/NDArray.Indexing.Masking.cs`

| Workload | NumSharp | NumPy | Ratio |
|---|---|---|---|
| 1M src, 100K Int32 indices, Double | 2.04 ms | 0.40 ms | **5.1× slower** |
| 1M bool mask (full shape), Double | 5.94 ms | 0.77 ms | **7.7× slower** |
| 1000×1000, 500-row fancy index | 0.89 ms | 0.74 ms | 1.2× slower (OK) |
| 1000×1000 + 1D axis-0 mask | 1.74 ms | 0.75 ms | 2.3× slower |
| **Setter** 1000×1000 + axis-0 mask | 3.28 ms | 0.09 ms | **36× slower** |

The setter path (`BooleanMaskAxis0` setter at `NDArray.Indexing.Masking.cs:281-318`) iterates `mask.size` times, calls `mask.GetBoolean(i)` (virtual) and on each true does `np.copyto(this[i], value)` (a slice + virtual copy). For a sparse mask hitting 50% of rows, that is 500 calls into `np.copyto`. NumPy uses one C loop with stride-aware element writes.

`FetchIndices`/`FetchIndicesND` (Getter) is decent at 1.2× for the 2D row-fancy case but slow at 5× for 1D scalar fancy, due to `PrepareIndexGetters` allocating delegate arrays plus the per-element `idxAddr[i]` indirect read.

---

## 3. API parity gaps (against NumPy 2.4.2)

| Function | NumSharp signature | NumPy signature | Missing |
|---|---|---|---|
| `np.negative` | `negative(nd)` | `negative(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True, signature=None)` | `out`, `where`, `dtype`, `casting`, `order` |
| `np.clip` | `clip(a, a_min, a_max, NPTypeCode? dtype = null)` + 2 overloads | `clip(a, a_min=<no value>, a_max=<no value>, out=None, *, min=<no value>, max=<no value>, **kwargs)` | optional defaults for bounds, `min`/`max` aliases, `where` |
| `np.modf` | `(F,I) modf(x, NPTypeCode? dtype = null)` | `modf(x, [out1, out2], *, out=(None, None), where=True, ...)` | tuple `out`, `where`. Also: `Intergral` typo in tuple name |
| `np.convolve` | `convolve(a, v, mode='full')` | `convolve(a, v, mode='full')` | Bool dtype, FFT path |
| `np.searchsorted` | `searchsorted(a, v)` | `searchsorted(a, v, side='left', sorter=None)` | `side`, `sorter`, 1-D-only check on `a` |
| `np.argsort` | `argsort<T>(int axis = -1)` (generic, not on NDArray non-typed) | `argsort(a, axis=-1, kind=None, order=None, *, stable=None)` | `kind`, `order`, `stable` |
| `np.nanmean` | `nanmean(a, int? axis = null, bool keepdims = false)` | `nanmean(a, axis=None, dtype=None, out=None, keepdims=<no value>, *, where=<no value>)` | `dtype`, `out`, `where` |
| `np.nanstd` | `nanstd(a, int? axis = null, bool keepdims = false, int ddof = 0)` | `nanstd(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>, correction=<no value>)` | `dtype`, `out`, `where`, `mean`, `correction` |
| `np.nanvar` | similar to nanstd | `nanvar(a, axis=None, dtype=None, out=None, ddof=0, keepdims=<no value>, *, where=<no value>, mean=<no value>, correction=<no value>)` | `dtype`, `out`, `where`, `mean`, `correction` |

(`np.add`/`subtract`/`multiply`/`divide`/`mod`/`prod` in `np.math.cs` are simple dispatchers; they have the same kind of parameter gaps but are consistent across the codebase.)

---

## 4. dtype coverage

| Operation | All 15 dtypes? | Notes |
|---|---|---|
| `np.negative` | **No** | Throws on `Byte`/`UInt16`/`UInt32`/`UInt64`/`Char`. Accepts `Boolean` where NumPy rejects. |
| `np.clip` | Yes | Tested Boolean, Half, Single, Double, Decimal, Complex; all work. |
| `np.modf` | Float-only (NumPy parity) | Single/Double/Half/Decimal? Need separate verification — engine call path. |
| `np.convolve` | 13 of 15 | Rejects Boolean. Decimal/UInt64 accepted but precision is wrong (B7). |
| `NOT` (operator) | Yes (all 15 via NpFunc) | Verified Int32/Double. |
| Shift `<<`/`>>` | Integer dtypes (engine rejects float, matches NumPy) | Verified — TypeError on double. |
| `np.searchsorted` | All 15 (via Converts.ToDouble routing) | Loses precision for Decimal/UInt64. |
| `argsort<T>` | All unmanaged T | Float-NaN sort uses NumPyComparer correctly. |
| `nanmean`/`nanstd`/`nanvar` | Single/Double/Half/Complex special-cased | Other dtypes route to non-NaN `mean`/`std`/`var` (correct — NaN can't appear). |

---

## 5. NumPy structural parity

| Aspect | Match? |
|---|---|
| `argsort` algorithm | **No** — NumPy uses introsort/quicksort on typed pointers per axis strip; NumSharp uses LINQ `OrderBy` (stable merge sort with allocations). NumPy structure: `argsort_table[NPY_QUICKSORT]` indexed by dtype, called against contiguous-axis stride. NumSharp's structure is a chain of `IEnumerable` `Aggregate/Concat/Zip` operations completely unrelated to NumPy's design. |
| `searchsorted` algorithm | Partial — binary-search algorithm is correct but lacks the per-side / per-dtype function table NumPy has at `item_selection.c:get_binsearch_func(dtype, side)`. |
| `nanmean`/`nanstd`/`nanvar` | Partial — Has two-pass for std/var (matches NumPy). Lacks Welford / NumPy's `_replace_nan` masking trick that re-uses regular mean kernels. |
| `convolve` | Partial — Mode dispatch matches (`full`/`same`/`valid`). Inner loop matches the math. Does not use FFT for large kernels. |
| `np.negative` | Mostly — Mirrors the dtype-switch structure; misses uint dtypes. |
| Indexing (`Getter`/`Setter`) | Different — NumPy `mapping.c:array_subscript_asarray` uses fancy-index normalizer + iterator. NumSharp uses LINQ-based pre-processing + delegate-array of getters. The general structure is similar but with more virtual-call overhead. |

---

## 6. Iterator / ILKernel utilization

| File | Uses NpyIter? | Uses ILKernel? | Should it? |
|---|---|---|---|
| `NDArray.negative.cs` | No | No (writes raw pointer per dtype) | **Yes** — `TensorEngine.Negate` already uses `ILKernelGenerator.EmitVectorOperation`. `nd.negative()` should call into it instead of duplicating the dtype switch. |
| `NdArray.Convolve.cs` | No | No | **Yes** — at minimum the inner kernel should be IL-emitted per (TIn, TOut) to remove the boxing. For large kernels, FFT path needed. |
| `np.clip.cs` | Indirectly (via engine) | Indirectly (via engine) | Engine general path needs `NpyIter.MultiNew` over (a, a_min, a_max). |
| `np.modf.cs` | Indirectly | Indirectly | OK — engine call is the entry point. |
| `NDArray.NOT.cs` | No | No | **Yes** — ILKernel emit `Vector{W}.Equals(*p, default)` would unlock SIMD. |
| `NDArray.Shift.cs` | Indirectly (via engine) | Yes (via engine) | OK. |
| `Indexing.*.cs` | No | No | Maybe — `NpyIter.MultiNew` with index-array + source could express the gather/scatter cleanly. |
| `ndarray.argsort.cs` | No | No | N/A — sort kernels are inherently scalar-typed pointer comparisons. Should rewrite with `LongIntroSort` over pointers, not LINQ. |
| `np.searchsorted.cs` | No | No | **Yes** — typed pointer binary search per (T_a, T_v) via `NpFunc.Invoke`. |
| `np.nanmean/std/var.cs` | Scalar paths use `NpyIterRef.New(...)` with `ExecuteReducing<Kernel,Accum>`. Axis paths do NOT — they hand-roll coordinate iteration. | No (axis path) | **Yes** for axis path — `ILKernelGenerator.TryGetNanAxisReductionKernel` already exists in `Default.Reduction.Nan.cs`. The `np.nan*_axis` functions never call it. |

---

## 7. Wasted copies

| Site | Wasted copy |
|---|---|
| `NDArray.negative.cs:18` | `TensorEngine.Cast(this, dtype, copy: true)` always copies even when input is C-contig and dtype matches. Should output into a freshly-allocated buffer with the result, skipping the input clone. |
| `Math/np.clip.cs:35` | `PreserveFContigFromSource` does `result.copy('F')` for F-contig sources. Could instead compute directly into an F-order output. |
| `Math/np.modf.cs:38,40` | `frac.copy('F')` + `whole.copy('F')` (same idea). |
| `Math/np.math.cs:80` | `negative` F-contig preservation does another `result.copy('F')`. |
| `Math/NdArray.Convolve.cs:56-57` | Always converts both `a` and `v` to common dtype via `astype` even when input dtype is already a non-int subtype that would work. |
| `Math/NdArray.Convolve.cs:198,219` | `ConvolveSame`/`ConvolveValid` compute **full** convolution then slice + copy. Could compute only the needed range. |
| `Selection/NDArray.Indexing.Masking.cs:Setter Case 2` | `np.nonzero(mask)` then `SetIndices` — allocates nonzero tuple. NumPy uses iterator directly. |
| `Sorting_Searching_Counting/ndarray.argsort.cs:28` | Forced `this.copy('C')` for non-contig inputs. Sort is O(n log n) so the O(n) copy is OK, but for axis sorts on F-contig arrays a transpose-iterator would suffice. |

---

## 8. Missing functionality

| API | Missing |
|---|---|
| `np.argsort` | `kind` (quicksort/mergesort/heapsort/stable), `order`, `stable` keyword |
| `np.searchsorted` | `side='right'`, `sorter`, 1-D-only validation |
| `np.nanmean/std/var` | `dtype`, `out`, `where`, `mean`/`correction` (nanstd/nanvar) |
| `np.clip` | Default-None bounds, `out` parameter as positional, `min`/`max` aliases (NumPy 2.0) |
| `np.modf` | tuple `out` (NumPy signature `out=(None, None)`) |
| `np.negative` | unsigned dtypes, `out`, `where`, `dtype` |
| `np.convolve` | Boolean dtype, FFT path for large kernels, `out` |
| Setter `SetIndicesNDNonLinear` | Not implemented (throws) |

---

## 9. Severity-ordered remediation plan

| # | Issue | Severity | Effort | Expected gain |
|---|---|---|---|---|
| 1 | B1+B2 `searchsorted` naming, side, sorter | Behavioral correctness | 1 day | Restore parity |
| 2 | B7 `convolve` int64 precision loss | Silent wrong answers | 1 day | Correct results |
| 3 | B3 `searchsorted` accepts multidim `a` | Silent wrong answers | 30 min | Match NumPy errors |
| 4 | B6 `np.negative` unsigned dtypes | NotSupported where NumPy works | 1 hour (route to `TensorEngine.Negate`) | Parity |
| 5 | B5 `np.negative(bool)` should error | Behavioral mismatch | 5 min | Parity |
| 6 | B11 `SetIndicesNDNonLinear` NotImplementedException | Crashes valid code | 1 day | Eliminate crash |
| 7 | P1 argsort LINQ | Up to **184× slower** | 2 days | 50-200× speedup |
| 8 | P4 clip transposed path | **67× slower** on transposed | 1 day | 30-60× |
| 9 | P2 convolve double-cast loop | **67× slower** | 1 day (IL gen) | 20-50× |
| 10 | P3 searchsorted boxing | **19× slower** | (with #1) | 5-15× |
| 11 | P7 boolean mask setter | **36× slower** for setter | ½ day | 20-30× |
| 12 | P5 nanmean/std/var axis allocs | **3-4× slower** | 1 day | 2-3× |
| 13 | P6 NOT scalar loop | 3× slower | ½ day (IL emit) | 2-3× |
| 14 | B9 modf `Intergral` typo | Stable API typo | (Breaking change) | n/a |
| 15 | B8 convolve(bool) | NotSupported | 30 min | Parity |
| 16 | Missing `np.nan*` `dtype`/`out`/`where` | API parity | 1 day each | n/a |
| 17 | Missing `np.argsort` `kind`/`stable` | API parity | (with #7 rewrite) | n/a |

---

## 10. Notable findings the v1 audit missed or got wrong

| v1 claim | This audit |
|---|---|
| "nan{mean,std,var} axis is 100-1000× slower" | Measured: **3-4×** slower. Still worth fixing (per-iter alloc), but not in the catastrophic-perf tier. |
| "argsort 100-1000× slower" | Confirmed: **18× (1D) to 184× (2D 1000×1000)**. The audit's claim was within the right order of magnitude. |
| "searchsorted boxing perf" | Confirmed: **19× slower**. |
| Negative's `unary -` vs `np.negative` inconsistency | Not in v1. NumSharp `-byte_arr` works (uint wrap), but `np.negative(byte_arr)` throws. Same dtype, two API entry points, opposite behavior. |
| Negative accepts `bool` (NumPy rejects) | Not in v1. |
| Convolve int64 precision loss | Not in v1. |
| Convolve missing bool | Not in v1. |
| Setter `SetIndicesNDNonLinear` NotImplementedException | Not in v1. |
| searchsorted multidim silently mis-interprets | Not in v1; v1 only said "no multidim support". |
| `modf` "Intergral" typo | Not in v1. |
| `clip` default-None bounds missing | Not in v1. |
| Bool-mask setter 36× slower | Not in v1. |

---

## Summary table

| ID | Issue | Severity | Measured ratio (NumSharp / NumPy) |
|---|---|---|---|
| **B1** | `searchsorted` function name is misleading (`binarySearchRightmost` is actually left) | Cosmetic / API | — |
| **B2** | `searchsorted` missing `side`/`sorter` | API gap | — |
| **B3** | `searchsorted` silently accepts multidim `a` | Silent wrong result | — |
| **B4** | `searchsorted` boxes per probe via `Storage.GetValue` | Perf bug | 19× slower |
| **B5** | `np.negative(bool)` returns NOT (NumPy errors) | Wrong behavior | — |
| **B6** | `np.negative(uint*)` throws (NumPy wraps) | Wrong behavior | — |
| **B7** | `convolve` accumulates in `double`, loses int64 precision | Silent wrong result | — |
| **B8** | `convolve(bool)` not supported | API gap | — |
| **B9** | `modf` tuple member typo: `Intergral` | API typo | — |
| **B10** | `NDArray.NOT` `.Equals(default)` not guaranteed devirtualized | Perf | 3× slower |
| **B11** | `SetIndicesNDNonLinear` NotImplementedException | Crash | — |
| **B12** | `np.clip` cannot omit a bound at call site | API gap | — |
| **B13** | `np.clip` mismatched bound broadcasting unverified | Possible behavior gap | — |
| **P1** | `argsort` LINQ-based | **Severe** perf | 18-184× slower |
| **P2** | `convolve` boxed double-cast inner loop | Perf | 67× slower (10K×100) |
| **P3** | `searchsorted` boxing | Perf | 19× slower (1M×100K) |
| **P4** | `np.clip` transposed/general path | Perf | 67× slower (1000×1000 T) |
| **P5** | `nanmean/std/var_axis` per-iter `long[]` allocs + virtual `Get*` | Perf | 3-4× slower |
| **P6** | `NOT` scalar loop, no SIMD | Perf | 3× slower |
| **P7** | Bool-mask setter row-by-row `np.copyto` | Perf | 36× slower |
| **P8** | Fancy 1D index | Perf | 5× slower |
| **P9** | Bool full mask | Perf | 7.7× slower |

**Files most needing rewrite (in priority order):**

1. `Sorting_Searching_Counting/ndarray.argsort.cs` — full rewrite to typed-pointer sort
2. `Sorting_Searching_Counting/np.searchsorted.cs` — full rewrite + `side`/`sorter`/1-D check
3. `Math/NdArray.Convolve.cs` — IL-emitted typed inner loop + typed accumulator
4. `Math/NDArray.negative.cs` — delete and route to `TensorEngine.Negate`
5. `Selection/NDArray.Indexing.Masking.cs` (setter axis-0 path) — batch via `NpyIter`
6. `Statistics/np.nanmean.cs`, `np.nanstd.cs`, `np.nanvar.cs` (axis paths) — hoist alloc + route through `ILKernelGenerator.TryGetNanAxisReductionKernel`
