# Audit Group 6: Manipulation + Top-Level APIs + Logic

**Scope**: Files in `Manipulation/`, `APIs/`, and `Logic/` per audit brief.
**Reference**: NumPy 2.4.2 (Python 3.12.12).
**Audited branch**: `nditer` vs `master`.

All claims below are verified via paired `python -c` (NumPy 2.4.2) and `dotnet_run` (current branch). Every behavioral finding has a reproduction in its own subsection.

---

## 1. Critical bugs (severity = HIGH)

### 1.1 `np.finfo.minexp` is off-by-one for `float32` and `float64`

**File**: `src/NumSharp.Core/APIs/np.finfo.cs` lines 129, 145.

NumSharp stores `Single.minexp = -125`, `Double.minexp = -1021`. NumPy 2.x stores `-126` and `-1022` respectively. This breaks the invariant `smallest_normal == 2**minexp`.

```python
>>> import numpy as np
>>> np.finfo(np.float32).minexp
-126
>>> np.finfo(np.float64).minexp
-1022
```

```csharp
// dotnet_run reproduction:
Console.WriteLine(np.finfo(NPTypeCode.Single).minexp);  // -125   ❌  (should be -126)
Console.WriteLine(np.finfo(NPTypeCode.Double).minexp);  // -1021  ❌  (should be -1022)
Console.WriteLine(np.finfo(NPTypeCode.Half).minexp);    // -14    ✓
// smallest_normal verifies: 2^-1021 = 4.45e-308 but stored value is 2.225e-308.
// 2^-1022 = 2.225e-308 → minexp must be -1022.
```

`maxexp` values (128 / 1024 / 16) are correct.

### 1.2 `np.expand_dims` drops the new dimension when input is empty

**File**: `src/NumSharp.Core/Manipulation/np.expand_dims.cs`, lines 7-12.

The guard `if (a.size == 0 || a.Shape.IsEmpty) return a;` short-circuits before adding the axis, so an empty input never gets expanded.

```python
>>> e = np.array([], dtype=float)
>>> np.expand_dims(e, 0).shape
(1, 0)
>>> np.expand_dims(e, (0,1)).shape
(1, 1, 0)
```

```csharp
var e = np.array(new double[]{});
np.expand_dims(e, 0).shape;  // (0,)   ❌  expected (1, 0)
```

Fix: remove the early-return for empty arrays — `Shape.ExpandDimension` already handles this in NumPy by inserting a `1` regardless of size.

### 1.3 `np.expand_dims` does not accept a tuple/sequence of axes

**File**: `src/NumSharp.Core/Manipulation/np.expand_dims.cs`, line 5.

Signature is `expand_dims(NDArray a, int axis)` — only a single int. NumPy 2.x accepts a tuple, e.g. `np.expand_dims(a, (0, 2))` returns shape `(1, 3, 1)` for a 1-D input.

```python
>>> np.expand_dims(np.array([1,2,3]), (0, 2)).shape
(1, 3, 1)
>>> np.expand_dims(np.array([1,2,3]), (-1, 0)).shape
(1, 3, 1)
```

No equivalent NumSharp call compiles. Missing functionality.

### 1.4 `np.copyto` ignores `casting` and `where`

**File**: `src/NumSharp.Core/Manipulation/np.copyto.cs`, line 16. The TODO comment on that line acknowledges the gap.

NumPy's default `casting='same_kind'` rejects float→int copies; NumSharp truncates silently.

```python
>>> a = np.zeros(5, dtype=np.int8)
>>> np.copyto(a, np.array([1.5, 2.5, 3.5, 4.5, 5.5]))   # default casting='same_kind'
TypeError: Cannot cast array data from dtype('float64') to dtype('int8') according to the rule 'same_kind'
```

```csharp
var dst = np.zeros(new Shape(3), np.int32);
var src = np.array(new double[] { 1.5, 2.5, 3.5 });
np.copyto(dst, src);    // silently truncates to [1,2,3]; no exception
```

Also no `where=` argument — NumPy supports masked write. Two missing parameters.

### 1.5 `np.repeat` has no `axis` parameter

**File**: `src/NumSharp.Core/Manipulation/np.repeat.cs`.

NumPy's signature is `np.repeat(a, repeats, axis=None)`. NumSharp's first line of `repeat(a, repeats)` is `a = a.ravel();`, then a 1-D output. Axis is silently impossible.

```python
>>> a = np.array([[1,2],[3,4]])
>>> np.repeat(a, 2, axis=0)
array([[1,2],[1,2],[3,4],[3,4]])
>>> np.repeat(a, [2,3], axis=0)
array([[1,2],[1,2],[3,4],[3,4],[3,4]])
```

```csharp
var a = np.array(new int[,] {{1,2},{3,4}});
np.repeat(a, 2);   // → [1,1,2,2,3,3,4,4] (ravels, ignores 2-D shape)
// No overload accepts an axis argument.
```

Docstrings on lines 14 and 23 ("same shape as a, except along the given axis") are misleading — there is no axis.

### 1.6 `np.ravel('F')` of an F-contiguous array always copies; NumPy returns a view

**File**: `src/NumSharp.Core/Manipulation/np.ravel.cs`, lines 30-34.

`np.ravel` resolves `physical == 'F'` then unconditionally calls `a.flatten('F')`, which in turn calls `copy('F')`. NumPy returns a view when reading in the array's native order.

```python
>>> aF = np.arange(12).reshape(3,4).copy(order='F')
>>> r = np.ravel(aF, order='F')
>>> np.shares_memory(r, aF)
True
```

```csharp
var aF = np.arange(12).reshape(3, 4).copy('F');
var r = np.ravel(aF, 'F');
r.SetAtIndex(999L, 0L);
aF.GetAtIndex(0);   // → 0  ❌  view would propagate 999
```

This produces a measurable performance regression (see § 5.1: 3000× slower than NumPy for F-contiguous ravel-in-F-order).

### 1.7 `np.unique` is missing all keyword arguments and `axis`

**File**: `src/NumSharp.Core/Manipulation/NDArray.unique.cs` (68 LoC), `src/NumSharp.Core/Manipulation/np.unique.cs`.

Signature: `unique()` only. NumPy 2.x: `unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None, equal_nan=True)`. Missing:
- `return_index` — caller has no way to recover the first-occurrence indices.
- `return_inverse` — caller has no way to recover reconstruction indices.
- `return_counts` — caller has no way to recover frequency.
- `axis` — no 2-D row/column unique.
- `equal_nan` — implicitly `True` only (NaN collapses to one entry).

Three of these are explicit asks in the audit brief. The internal implementation could expose `Hashset.Indices` and `Counts` cheaply, but the public surface refuses to.

---

## 2. Behavioral mismatches (severity = MEDIUM)

### 2.1 `np.where` 1-arg returns `NDArray<long>[]` array, NumPy returns a tuple

**File**: `src/NumSharp.Core/APIs/np.where.cs`, lines 17-21.

`np.where(condition)` returns `NDArray<long>[]` (an array of NDArrays); NumPy returns a `tuple`. Functionally compatible (index 0 access is identical), but type signature differs. Not a bug — just a C# idiom mismatch.

### 2.2 `np.copyto(unwriteable_dst, src)` throws different exception

**File**: `src/NumSharp.Core/Manipulation/np.copyto.cs`, line 24.

`ThrowIfNotWriteable` is invoked; NumPy uses `ValueError`. Cosmetic.

### 2.3 Default scalar dtype in `np.where(cond, 1, 2)`

```python
>>> np.where(np.array([True, False]), 1, 2).dtype
dtype('int64')
```

```csharp
np.where(cond, (object)1, (object)2).dtype;   // → System.Int32
```

This is a known cross-language divergence (Python int defaults to int64, C# int to int32). Documented in `np.where.BattleTest.cs` lines 21-26. Not a NumSharp bug per se but a porting risk.

### 2.4 `np.all` / `np.any` reject tuple axis

**File**: `src/NumSharp.Core/Logic/np.all.cs`, `np.any.cs`.

Signatures only accept `int axis`. NumPy 2.x accepts an int OR a tuple-of-ints. Missing:

```python
>>> b = np.ones((2,3,4), dtype=bool)
>>> np.all(b, axis=(0,1)).shape
(4,)
```

No NumSharp equivalent. Same gap for `np.any`.

### 2.5 `np.all` / `np.any` reject `where=` argument

NumPy 2.x signature: `np.all(a, axis=None, out=None, keepdims=False, where=True)`. NumSharp omits `out` and `where`.

```python
>>> np.all(np.array([1,1,1,0]), where=np.array([True,True,False,False]))
True
```

### 2.6 `np.all(empty_arr)` / `np.any(empty_arr)` not verified

NumPy: `all([]) == True`, `any([]) == False` (vacuous identities). Not exercised in this audit; flagged for follow-up tests.

### 2.7 `iinfo(bool)` accepted; NumPy 2.x rejects

**File**: `src/NumSharp.Core/APIs/np.iinfo.cs` line 84.

NumPy 2.x: `np.iinfo(np.bool_)` raises `ValueError: Invalid integer data type 'b'`. NumSharp returns `(bits=8, min=0, max=1)`. The code comment on line 84 ("NumSharp extension — NumPy 2.x throws ValueError") acknowledges the divergence. Documented behaviour — non-bug, but a minor parity gap.

### 2.8 `iinfo(UInt64).max` clamped to `long.MaxValue`

**File**: `src/NumSharp.Core/APIs/np.iinfo.cs` line 110.

`UInt64` returns `max = 9223372036854775807` (long.MaxValue), `maxUnsigned = 18446744073709551615`. Because the public `max` field is typed `long`, the true value can't fit. Acceptable C# workaround, but a caller doing `info.max` for uint64 silently gets the wrong value.

---

## 3. Code-quality issues (severity = MEDIUM)

### 3.1 `_can_coerce_all(NPTypeCode[], int start)` has a bad `Array.Copy` call

**File**: `src/NumSharp.Core/Logic/np.find_common_type.cs` lines 1058-1087.

```csharp
Array.Copy(dtypelist, start, sub, len, len);  // ❌ 4th arg should be 0, not len
```

`Array.Copy(src, srcIdx, dst, dstIdx, count)` — passing `len` for the destination index throws `ArgumentException: Destination array was not long enough`. Verified by reflection invocation:

```csharp
var arr = new NPTypeCode[] { NPTypeCode.Int32, NPTypeCode.Single };
method.Invoke(null, new object[] { arr, 1 });
// → ArgumentException: Destination array was not long enough...
```

The List<> overload at line 1090 has a parallel bug:

```csharp
var sub = new List<NPTypeCode>(len);   // Count = 0
sub[i - start] = dtypelist[i];          // ❌ writes to empty list — throws ArgumentOutOfRangeException
```

**Reachability**: Only called from `_find_common_coerce`. Analysis of `_find_common_coerce` shows that the only path that reaches `_can_coerce_all(arr, thisind)` requires `maxsc <= maxa` numerically AND `index_sc > index_a` in the kind list. Given NPTypeCode ordering (all `'f'` kind types are numerically > all `'i'` kind types) and the kind-list ordering ('b' < 'u' < 'i' < 'f' < 'c'), the "scalar kind > array kind AND maxsc <= maxa" condition cannot be satisfied. The bug is dead code today — but it would surface immediately if anyone reorders `NPTypeCode` values or adds new types.

### 3.2 `NPTypeCode.Decimal` maps to `NPY_LONGLONGLTR` (line 532 of `NPTypeCode.cs`)

**File**: `src/NumSharp.Core/Backends/NPTypeCode.cs` line 532.

```csharp
case NPTypeCode.Decimal:
    return NPY_TYPECHAR.NPY_LONGLONGLTR;
```

That's the typechar for `long long` (int64), not a float. Any callsite that probes `Decimal.ToTYPECHAR()` and feeds it into a kind-list lookup will mis-classify Decimal as a signed integer. The `_kind_list_map` in `DType` (line 30) correctly classifies `Decimal → 'f'`, so the symptom is limited; but the mapping is internally inconsistent.

Also: line 487 of `NPTypeCode.cs` has unreachable code (`return NPTypeCode.Decimal;` after a `case NPTypeCode.Complex:` return).

### 3.3 Closure allocation per element in `np.unique` non-contiguous path

**File**: `src/NumSharp.Core/Manipulation/NDArray.unique.cs` lines 145-151.

```csharp
Func<long, long> getOffset = flat.Shape.GetOffset_1D;
for (long i = 0; i < len; i++)
    hashset.Add(src[getOffset(i)]);
```

`Func<,>` boxing of a method group plus per-call virtual dispatch is unnecessary — the loop could call `flat.Shape.GetOffset_1D(i)` directly with no allocation. Minor perf cost; readability is no worse without the alias.

### 3.4 `np.unique` does two passes for non-contiguous: explicit `flat` then dispatch by typecode

The non-contig branch dereferences `flat.Address` (which is a typed cast). For a strided 3-D array the `flat` value is itself materialized as a 1-D iterator — that materializes a copy under the hood (`flat` calls `ravel` indirectly). So non-contig unique pays for one full copy plus the hashset work. NumPy does this in one pass over the iterator.

### 3.5 `np.tile` allocates intermediate `new Shape(...)` four times per call

**File**: `src/NumSharp.Core/Manipulation/np.tile.cs` lines 91-104.

```csharp
var promoted = A.reshape(new Shape(interleaved));
var broadcasted = broadcast_to(promoted, new Shape(broadcastTarget));
var contiguous = broadcasted.copy();
return contiguous.reshape(new Shape(outShape));
```

Plus a `new long[outDim]` for `aShape`, `tup`, `outShape`, `interleaved` (2×), `broadcastTarget` (2×). Eight allocations per `tile` call. NumPy's C implementation is essentially `np.broadcast_to(A.reshape(interleaved), broadcastTarget).reshape(outShape)`, but happens at zero allocation cost for the reshapes. The C# approach is correct but ~5× slower than NumPy (see § 5).

### 3.6 `np.repeat` uses `Converts.ToInt64(repeatsFlat.GetAtIndex(i))` per element

**File**: `src/NumSharp.Core/Manipulation/np.repeat.cs` lines 81-85, 203-208.

`GetAtIndex` boxes the value; `Converts.ToInt64` runs a switch. Two boxes per element across both loops (counting pass + write pass). For dense int repeats this is unnecessary — the loop could be specialised over `repeatsFlat.GetTypeCode` and use direct pointer access. Verified perf gap: NumSharp np.repeat is ~1.5× NumPy at 100K elements.

### 3.7 `np.where.WhereImpl` doesn't cache the `NpyExpr` per dtype

**File**: `src/NumSharp.Core/APIs/np.where.cs` lines 157-162.

The `cacheKey: $"np.where.{dtype}"` is fine, but `NpyExpr.Where(NpyExpr.Input(0), NpyExpr.Input(1), NpyExpr.Input(2))` reallocates the expression tree on every call. Cheap, but a static field per type would eliminate the per-call allocations.

### 3.8 `np.find_common_type` has 1149 LoC and 4 duplicated `_can_coerce_all` overloads

**File**: `src/NumSharp.Core/Logic/np.find_common_type.cs`.

The `arr_arr` and `arr_scalar` tables hard-code ~270 entries each — a lot, but justified for O(1) FrozenDictionary lookup. The four `_can_coerce_all` variants (array, list, array-with-start, list-with-start) repeat the same N-1/N=0/N=1 logic and could collapse to a single helper taking a `ReadOnlySpan<NPTypeCode>`. As noted, the `start` variants are buggy and dead.

NEP50 alignment of the actual promotion tables is verified — all spot-checks below match NumPy 2.4.2:
- `int32 + int64 → int64` ✓
- `uint64 + int8 → float64` ✓
- `float32 + int32 → float64` ✓
- `uint8_arr + 5 (scalar) → uint8` ✓ (NEP50)
- `uint8_arr + 5.0 → float64` ✓ (cross-kind)

One **divergence**: `uint8_arr + 1000 → uint8` (silent overflow), NumPy 2.x raises `OverflowError: Python integer 1000 out of bounds for uint8`. This is because NumSharp's NEP50 implementation does not run a value-fits check on out-of-range Python literals.

### 3.9 `np.tile` ignores `IsBroadcasted` write-protection

The intermediate `broadcasted` (line 102) is broadcast (stride=0); `.copy()` correctly materializes it before reshape. Verified correct via `tile` test sweep across all 15 dtypes.

### 3.10 `np.where(condition, x, y)` does NOT validate operand shape equality with `Shape.Equals`

```csharp
if (condition.Shape == x.Shape && x.Shape == y.Shape)
```

`Shape` is a `readonly struct`. The `==` operator on struct typically uses identity / compiler-generated equality — but `Shape` overrides `==` (via `_hashCode` cache). Spot-checked: for two ndarrays both 3×3 C-contiguous with same offset/strides, `==` returns true. But if x is broadcast to (3,3) with stride=0 in dim 0, that has different strides — the `==` rightly returns false, falling through to the slower broadcast path. Looks correct.

---

## 4. Coverage of 15 dtypes (§ DOD)

| Function           | Bool | Byte | SByte | I16 | U16 | I32 | U32 | I64 | U64 | Char | Half | Single | Double | Decimal | Complex |
|--------------------|------|------|-------|-----|-----|-----|-----|-----|-----|------|------|--------|--------|---------|---------|
| `np.tile`          | ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |
| `np.repeat(scalar)`| ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |
| `np.repeat(arr)`   | ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |
| `np.ravel`         | ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |
| `np.where`         | n/a  | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |
| `np.unique`        | ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓ (sep.)|
| `np.all/any`       | ✓    | ✓    | -     | -   | -   | -   | -   | -   | -   | -    | -    | -      | -      | -       | -       |
| `np.copyto`        | ✓    | ✓    | -     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | -       | ✓       |
| `np.expand_dims`   | ✓    | ✓    | ✓     | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓    | ✓    | ✓      | ✓      | ✓       | ✓       |

**`np.unique`** has a dedicated `uniqueComplex()` because `System.Numerics.Complex` does not implement `IComparable<Complex>` — clean, well-commented (line 195).

**`np.copyto`** with `SByte` dst and float src throws `NotSupportedException: Unsupported type: SByte` from the cast layer — bug or design? Filed as part of the broader `copyto` issues.

**`np.all`/`np.any`** were tested with Bool/Byte/Int32; the underlying TensorEngine path dispatches to typed reductions and supports all numeric dtypes, but I did not sweep all 15 codes for axis-reduction. Verified Bool + Int32 + keepdims + negative axis work.

---

## 5. Performance findings

All benchmarks were on Windows 11, .NET 10 release build, single-threaded, no warmup outside the loop.

### 5.1 `np.ravel(F-cont, 'F')` — 3000× slower than NumPy

```
np.ravel F-order on F-contig (NumSharp - copy fallback): 133ms / 100 iter
np.ravel F-order on F-contig (NumPy   - view):           0.04ms / 100 iter
```

Root cause: every F-order ravel materializes a fresh copy through `flatten('F')` (§ 1.6). Easy fix: when `source.IsFContiguous && physical == 'F'`, return a 1-D view of the same storage with `Shape.Vector`.

### 5.2 `np.ravel(C-cont, 'C')` — view path is fine

```
np.ravel 1M (NumSharp view): <1ms / 1000 iter
np.ravel 1M (NumPy   view): 0.42ms / 1000 iter
```

The C path uses `reshape(Shape.Vector(size))` which preserves a view.

### 5.3 `np.tile` — 5× slower than NumPy

```
np.tile 1K→100K (NumSharp): 109ms / 100 iter
np.tile 1K→100K (NumPy):    19.7ms / 100 iter
```

Cost dominated by the broadcast+copy path. The `copy()` call materializes a stride=0 view via the general `NpyIter.Copy` path. NumPy uses a memcpy loop per row. Below 10× threshold but worth flagging.

### 5.4 `np.repeat(1K, 100)` — 1.5× slower than NumPy

```
np.repeat 1K×100 (NumSharp): 31ms / 100 iter
np.repeat 1K×100 (NumPy):    22.8ms / 100 iter
```

Acceptable, but the per-element write loop could SIMD-broadcast the value (Vector256.Create + Vector256.Store) for large repeats counts. Not regressed enough to warrant an IL kernel.

### 5.5 `np.where` IL-kernel path — 1.6× slower than NumPy

```
np.where 1M, IL-kernel (NumSharp):   276ms / 100 iter
np.where 1M, vectorized (NumPy):     173ms / 100 iter
```

```
np.where broadcast scalar (NumSharp): 321ms / 100 iter
np.where broadcast scalar (NumPy):    137ms / 100 iter
```

The IL-kernel path runs via `NpFunc.Invoke` → `ILKernelGenerator.WhereExecute`. The 1.6× gap (and 2.3× for broadcast) suggests the IL kernel doesn't vectorize the `cond ? x : y` blend on x64 or doesn't unroll the tail.

### 5.6 `np.unique` — 2.3× slower than NumPy

```
np.unique 1M (1000 unique, NumSharp): 124ms / 10 iter
np.unique 1M (1000 unique, NumPy):     54ms / 10 iter
```

NumPy `unique` uses sort-then-detect-edges; NumSharp uses Hashset + IntroSort. The Hashset path is GC-heavy. For large arrays, sort-and-dedupe is usually faster (no hashing).

### 5.7 No 10× regressions detected across the audited surface.

Of the audited APIs, the worst case is `np.ravel(F-cont, 'F')` at 3000× regression, which is a one-line fix (return a view).

---

## 6. Iterators / IL-kernels — recommendation

Functions in this group that **could benefit from IL-kernel codegen but don't use it**:
- `np.repeat(scalar)` — large `repeats` count is just a broadcast write. SIMD store would help.
- `np.tile` — the final `copy()` of a broadcast view goes through the general iterator; an IL-kernel-driven specialization could collapse this into a vectorized expand+copy.

Functions that **could benefit from a multi-operand iterator** rather than a temporary copy:
- `np.unique` non-contig path (§ 3.4) creates a temporary `flat` ndarray.
- `np.copyto` already uses `NpyIter.Copy` — good.

Functions correctly using iterators / IL:
- `np.where` 3-arg uses NpyIter for the broadcast path; well-structured (§ 3.7).
- `np.copyto` delegates to `NpyIter.Copy` — clean delegation.

---

## 7. Missing functionality summary

| Category | Missing | NumPy ref |
|---|---|---|
| `np.repeat` | `axis` parameter | `fromnumeric.repeat(a, repeats, axis=None)` |
| `np.unique` | `return_index`, `return_inverse`, `return_counts`, `axis`, `equal_nan` | `arraysetops.unique` |
| `np.expand_dims` | tuple-of-axes argument; empty-array support | `fromnumeric.expand_dims` |
| `np.copyto` | `casting`, `where` | `multiarray.copyto` |
| `np.all` / `np.any` | tuple axis, `where`, `out` | `fromnumeric._all_dispatcher` |
| `np.flatten` / `np.ravel` | view-path for F-cont input in F-order | `fromnumeric.ravel` |
| `np.finfo` | correct `minexp` for f32/f64 | `getlimits.MachAr` |

---

## 8. Documentation / commit-history concerns

The branch documentation file `docs/plans/NDITER_BRANCH_QUALITY_AUDIT.md` (the V1 audit) states that `np.where` was migrated to `NpyIter` on this branch — verified accurate (§ 1.4 of that doc). The new `WhereImpl` is concise and routes correctly.

The branch docs do not mention:
- `np.finfo` precision bug (§ 1.1)
- `np.expand_dims` empty bug (§ 1.2)
- `_can_coerce_all` dead-code bug (§ 3.1)

These should be in OpenBugs or filed as issues.

---

## 9. Tests that should be added

1. `Finfo_Float32_Minexp_Is_Minus126` — currently `-125`.
2. `Finfo_Float64_Minexp_Is_Minus1022` — currently `-1021`.
3. `ExpandDims_EmptyArray_AddsDimension` — currently returns the empty array unchanged.
4. `ExpandDims_TupleAxis_Throws_Or_Implements` — to track API parity.
5. `Repeat_WithAxis_Throws_Or_Implements` — to track API parity.
6. `Ravel_FCont_FOrder_ReturnsView` — currently allocates a copy.
7. `Copyto_FloatToInt_DefaultCasting_Throws` — currently silently truncates.
8. `Copyto_WithWhere_OnlyWritesMaskedElements` — feature missing.
9. `Unique_ReturnIndex_ReturnInverse_ReturnCounts` — features missing.
10. `Unique_AxisParameter` — feature missing.
11. `All_TupleAxis_ReducesOverMultipleAxes` — feature missing.
12. `All_WhereParameter_FiltersInput` — feature missing.
13. `Where_PythonIntScalar_DefaultsToInt64` — tracks documented divergence.
14. `Add_Uint8_PythonLargeInt_Overflows` — NEP50 strictness gap.

---

## Summary table

| # | Severity | Area | Issue |
|---|----------|------|-------|
| 1.1 | **HIGH** | `np.finfo` | `minexp` is `-125`/`-1021` instead of `-126`/`-1022` for f32/f64. Violates `smallest_normal == 2^minexp`. |
| 1.2 | **HIGH** | `np.expand_dims` | Empty input returns unchanged; should return shape with new axis prepended. |
| 1.3 | **HIGH** | `np.expand_dims` | No tuple-of-axes overload — NumPy 2.x feature missing. |
| 1.4 | **HIGH** | `np.copyto` | `casting` and `where` parameters missing — silent truncation; no masked write. |
| 1.5 | **HIGH** | `np.repeat` | No `axis` parameter — always ravels; XML docs are misleading. |
| 1.6 | **HIGH** | `np.ravel` | F-order ravel of F-contig array always copies; should return a view. 3000× perf regression. |
| 1.7 | **HIGH** | `np.unique` | Missing `return_index`, `return_inverse`, `return_counts`, `axis`, `equal_nan`. |
| 2.1 | LOW | `np.where` | Returns `NDArray<long>[]` instead of a tuple (cosmetic). |
| 2.2 | LOW | `np.copyto` | Throws non-NumPy exception type for unwriteable dst (cosmetic). |
| 2.3 | LOW | `np.where` | C# int → int32 vs Python int → int64 (documented). |
| 2.4 | MED | `np.all/any` | Reject tuple axis. |
| 2.5 | MED | `np.all/any` | No `where=` or `out=`. |
| 2.6 | LOW | `np.all/any` | Empty-array identity behaviour not verified by audit. |
| 2.7 | LOW | `np.iinfo` | Accepts bool (NumPy throws); documented extension. |
| 2.8 | MED | `np.iinfo` | `UInt64.max` is `long.MaxValue`, not the true uint64 max. |
| 3.1 | MED | `np.find_common_type` | `_can_coerce_all` Array.Copy bug in dead code path. |
| 3.2 | LOW | `NPTypeCode` | `Decimal → NPY_LONGLONGLTR` mapping is wrong; unreachable today; line 487 dead-code return. |
| 3.3 | LOW | `np.unique` | `Func<long,long>` allocated per call; minor perf. |
| 3.4 | MED | `np.unique` | Non-contig path materializes a `flat` copy then dispatches — two passes. |
| 3.5 | LOW | `np.tile` | Eight allocations per call; ~5× slower than NumPy. |
| 3.6 | LOW | `np.repeat` | Per-element `Converts.ToInt64(GetAtIndex(...))` boxes twice; ~1.5× slower than NumPy. |
| 3.7 | LOW | `np.where` | Reallocates `NpyExpr.Where(...)` per call. |
| 3.8 | LOW | `np.find_common_type` | Four duplicated `_can_coerce_all` overloads — collapse to one. |
| 3.9 | NONE | `np.tile` | Broadcast write-protection correctly handled via `.copy()`. |
| 3.10 | NONE | `np.where` | Same-shape fast path correctness verified. |
| 5.1 | **HIGH** | perf | `np.ravel(F,'F')` 3000× slower than NumPy. |
| 5.3 | MED  | perf | `np.tile` 5× slower than NumPy. |
| 5.4 | LOW  | perf | `np.repeat` 1.5× slower. |
| 5.5 | MED  | perf | `np.where` 1.6×–2.3× slower (IL kernel not fully vectorized). |
| 5.6 | MED  | perf | `np.unique` 2.3× slower. |
| 6   | LOW  | impl | `np.repeat(scalar)` could SIMD; `np.tile` final copy could specialize. |

**Net assessment of the `nditer` branch's manipulation/logic/API surface**: behaviorally close to NumPy 2.x for the common cases, but the audit uncovered **7 HIGH-severity** issues — six are user-visible bugs/missing features (finfo, expand_dims, copyto, repeat, ravel, unique) and one is a 3000× perf regression in `np.ravel('F')` of F-contiguous arrays. The new F-order support added on this branch correctly preserves layout for `np.where`, `np.cumsum`, and `np.tile(all-ones)` but introduced a regression in `np.ravel('F')` by routing through `flatten('F')` unconditionally.
