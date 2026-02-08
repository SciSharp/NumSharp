# NumSharp Open Bugs Registry

> **Last verified:** 2026-02-07 against NumPy 2.4.2
> **Branch:** `dev`
> **Test files:** `OpenBugs.cs` (Bugs 1-24), `OpenBugs.DeprecationAudit.cs` (Bugs 27-63)
> **Total tests:** 92 (19 passing, 73 failing)

---

## Overview

This document catalogs all known open bugs in NumSharp, discovered through two separate audits:

1. **Broadcast audit** (Bugs 1-24): Systematic testing of broadcasting behavior across all NumSharp operations. Found that most bugs trace to a single architectural root cause in `UnmanagedStorage.Slicing.cs` where broadcast arrays get cloned data but retain broadcast strides.

2. **NumPy 1.x deprecation audit** (Bugs 27-63): Comprehensive comparison of NumSharp against NumPy 2.4.2 (latest stable). Found dead code stubs returning null, wrong mathematical implementations, missing dtype handling, and broken operators.

Every bug assertion has been verified by running actual Python code against NumPy 2.4.2 — zero discrepancies between test expectations and NumPy behavior.

### Test Convention

Tests assert the **correct NumPy behavior**. They **fail** while the bug exists and **pass** when the bug is fixed. When a bug is fixed, move its passing test(s) to the appropriate permanent test class.

### Removed False Positives

| Bug | Tests | Reason |
|-----|-------|--------|
| 25 | 4 | NEP 50 typed scalars match NumPy 2.4.2 — "weak typing" only applies to Python int literals |
| 26 | 1 | NumPy 2.4.2: `bool + bool` stays `bool` dtype (changed from 1.x which promoted to int) |
| 30 | 0 | Documentation-only note (no test) |
| 31 | 0 | Naming-only note (no test) |
| 46b | 1 | Duplicate of Bug 21 (boolean mask) |
| 49b | 1 | `np.all` without axis already works for all 12 dtypes via `_all_linear()` |

---

## Summary Table

| Status | Bugs | Tests | Description |
|--------|------|-------|-------------|
| **Fixed** | 7 | 19 | Bugs 1, 3, 4, 12, 15, 17, 20 |
| **Dead code** | 13 | 13 | Functions returning null/default |
| **Wrong semantics** | 7 | 8 | Produces results but mathematically wrong |
| **Crashes** | 12 | 16 | Throws exceptions instead of producing results |
| **Wrong values** | 15 | 21 | Returns results with incorrect values |
| **Total open** | **47** | **73** | |
| **Grand total** | **54** | **92** | |

---

## Fixed Bugs (19 passing tests)

These bugs were fixed by earlier broadcast infrastructure work. Their tests now pass.

### Bug 1 — ToString on broadcast arrays shows wrong values (4 tests) `FIXED`

**File:** `OpenBugs.cs`
**Tests:** `Bug_ToString_ReversedSliceBroadcast`, `Bug_ToString_StepSliceBroadcast`, `Bug_ToString_SlicedColumnBroadcast`, `Bug_ToString_DoubleSlicedBroadcast`

The ToString iterator used linear traversal that didn't account for ViewInfo + BroadcastInfo stride combinations. Coordinate-based access (`GetInt32(i,j)`) was correct, but ToString showed wrong values for reversed slices, step slices, column slices, and double-sliced broadcast arrays.

### Bug 3 — broadcast_to uses bilateral instead of unilateral semantics (1 test) `FIXED`

**File:** `OpenBugs.cs`
**Test:** `Bug_BroadcastTo_BilateralSemantics`

NumPy's `broadcast_to(array, shape)` is strictly one-directional — it only stretches dimensions of the source that are size 1. NumSharp used mutual/bilateral broadcasting, accepting inputs that NumPy rejects (e.g., `broadcast_to(ones(3), (1,))`).

### Bug 4 — Re-broadcast inconsistency (IsBroadcasted guard) (1 test) `FIXED`

**File:** `OpenBugs.cs`
**Test:** `Bug_ReBroadcast_Inconsistency`

The 2-arg `Broadcast(Shape, Shape)` had an explicit guard blocking already-broadcasted shapes, while the N-array overload didn't. This prevented legitimate chained broadcasting like `broadcast_to(broadcast_result, larger_shape)`.

**Also includes variant:** `Bug_Clip_Broadcast_ThrowsNotSupported` — `np.clip` internally broadcasts and hit the same guard.

### Bug 12 — hstack/vstack/concatenate on broadcast arrays (3 tests) `FIXED`

**File:** `OpenBugs.cs`
**Tests:** `Bug_Hstack_Broadcast_WrongValues`, `Bug_Vstack_SlicedBroadcast_WrongValues`, `Bug_Concatenate_SlicedBroadcast_WrongValues`

The concatenation copy step used linear offset calculation that didn't account for zero-stride broadcast dimensions, producing duplicated rows, shifted values, or garbage.

### Bug 15 — sum/mean/var/std with axis=0 on column-broadcast (5 tests) `FIXED`

**File:** `OpenBugs.cs`
**Tests:** `Bug_Sum_Axis0_ColBroadcast_WrongValues`, `Bug_Mean_Axis0_ColBroadcast_WrongValues`, `Bug_Var_Axis0_ColBroadcast_WrongValues`, `Bug_Std_Axis0_ColBroadcast_WrongValues`, `Bug_Sum_Axis0_ColBroadcast_5x3_WrongValues`

The axis=0 reduction iteration used strides wrong for column-broadcast memory layout, reading fewer rows than exist in the broadcast shape. Sum under-counted, cascading through mean, var, and std.

### Bug 17 — ROOT CAUSE: GetViewInternal stride/data mismatch (4 tests) `FIXED`

**File:** `OpenBugs.cs`
**Tests:** `Bug_SliceBroadcast_StrideMismatch_ColumnBroadcast_SliceColumn`, `Bug_SliceBroadcast_CopyWorkaround_Proves_StrideMismatch`, `Bug_SliceBroadcast_StrideMismatch_SlicedSourceRows`, `Bug_SliceBroadcast_StrideMismatch_Causes_Sum_Axis0_Bug`

**Location:** `UnmanagedStorage.Slicing.cs`, `GetViewInternal()`, lines 100-101

This was the root cause of Bugs 1, 11, 12, 13, 14, and 15. When slicing a broadcast array, `Clone()` correctly materialized data to contiguous layout, but `_shape.Slice(slices)` attached broadcast strides from the original shape to the clone. The data had strides `[3, 1]` but the shape claimed `[1, 0]`, causing all offset calculations to read from wrong positions.

### Bug 20 — Clip on broadcast (1 test) `FIXED`

**File:** `OpenBugs.cs`
**Test:** `Bug_Clip_Broadcast_ThrowsNotSupported`

Variant of Bug 4 — `np.clip` internally broadcasts and hit the IsBroadcasted guard.

---

## Open Bugs — Dead Code (13 bugs, 13 tests)

Functions that return `null` or `default` — stubs that were never implemented.

### Bug 32 — np.convolve returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Convolve_AlwaysReturnsNull`
**Severity:** Medium
**Location:** `Math/NdArray.Convolve.cs`

The Regen template was never generated into the `#else` block. The method always returns null.

```python
# NumPy 2.4.2:
>>> np.convolve([1,2,3], [0,1,0.5])
array([0. , 1. , 2.5, 4. , 1.5])
```

### Bug 34 — np.isnan returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_IsNan_ReturnsNull`
**Severity:** High
**Location:** `DefaultEngine.IsNan` returns null

```python
>>> np.isnan([1.0, np.nan, np.inf])
array([False,  True, False])
```

### Bug 35 — np.isfinite returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_IsFinite_ReturnsNull`
**Severity:** High
**Location:** `DefaultEngine.IsFinite` returns null

```python
>>> np.isfinite([1.0, np.nan, np.inf, -np.inf, 0.0])
array([ True, False, False, False,  True])
```

### Bug 36 — np.isclose returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_IsClose_ReturnsNull`
**Severity:** High — blocks `np.allclose` (Bug 7) which depends on `isclose`
**Location:** `DefaultEngine.IsClose` returns null

```python
>>> np.isclose([1.0, 1.0001], [1.0, 1.0002], atol=1e-3)
array([ True,  True])
```

### Bug 37 — operator `&` (AND) returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_AND_Operator_ReturnsNull`
**Severity:** High
**Location:** `NDArray.AND.cs` returns null

```python
>>> np.array([True, False, True]) & np.array([True, True, False])
array([ True, False, False])
```

### Bug 38 — operator `|` (OR) returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_OR_Operator_ReturnsNull`
**Severity:** High
**Location:** `NDArray.OR.cs` returns null

```python
>>> np.array([True, False, False]) | np.array([False, False, True])
array([ True, False,  True])
```

### Bug 39 — nd.delete() returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Delete_ReturnsNull`
**Severity:** Medium
**Location:** `NdArray.delete.cs` always returns null

```python
>>> np.delete(np.array([1,2,3,4,5]), [1])
array([1, 3, 4, 5])
```

### Bug 40 — nd.inv() returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Inv_ReturnsNull`
**Severity:** High
**Location:** `NdArray.Inv.cs` — entire LAPACK implementation commented out

```python
>>> np.linalg.inv(np.array([[1,2],[3,4]]))
array([[-2. ,  1. ],
       [ 1.5, -0.5]])
```

### Bug 41 — nd.qr() returns default (null, null)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_QR_ReturnsDefault`
**Severity:** High
**Location:** `NdArray.QR.cs` returns `default`

```python
>>> q, r = np.linalg.qr(np.array([[1,2],[3,4]]))
>>> q.shape, r.shape
((2, 2), (2, 2))
```

### Bug 42 — nd.svd() returns default (null, null, null)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_SVD_ReturnsDefault`
**Severity:** High
**Location:** `NdArray.SVD.cs` returns `default`

```python
>>> u, s, vh = np.linalg.svd(np.array([[1,2],[3,4]]))
>>> u.shape, s.shape, vh.shape
((2, 2), (2,), (2, 2))
```

### Bug 43 — nd.lstqr() returns null + misspelled

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Lstsq_ReturnsNull_AndMisspelled`
**Severity:** High
**Location:** `NdArray.LstSq.cs` — method named `lstqr` instead of `lstsq`, returns null

```python
>>> np.linalg.lstsq(np.array([[1,1],[1,2],[1,3]]), np.array([1,2,3]), rcond=None)[0]
array([0., 1.])
```

### Bug 44 — nd.multi_dot() returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_MultiDot_ReturnsNull`
**Severity:** Medium
**Location:** `NdArray.multi_dot.cs` always returns null

```python
>>> np.linalg.multi_dot([np.eye(2), np.eye(2)])
array([[1., 0.],
       [0., 1.]])
```

### Bug 45 — nd.roll(shift) no-axis returns null

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Roll_NoAxis_ReturnsNull`
**Severity:** Medium
**Location:** `NDArray.roll.cs:70` — body commented out, returns null. The with-axis overload partially works (Int32/Single/Double only).

```python
>>> np.roll(np.array([1,2,3,4,5]), 2)
array([4, 5, 1, 2, 3])
```

---

## Open Bugs — Wrong Semantics (7 bugs, 8 tests)

Functions that execute without crashing but produce mathematically incorrect results.

### Bug 47 — np.positive implements abs() instead of +x identity

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Positive_ImplementsAbsInsteadOfIdentity`
**Severity:** Medium
**Location:** `NDArray.positive.cs` — code has `if (val < 0) out_addr[i] = -val` (absolute value)

NumPy: `np.positive(x)` is equivalent to `+x` (identity for numeric types).
NumSharp: Takes absolute value of negative numbers.

```python
>>> np.positive(np.array([-3, -1, 0, 2, 5]))
array([-3, -1,  0,  2,  5])  # identity, unchanged

# NumSharp returns: [3, 1, 0, 2, 5]  (abs of negatives)
```

### Bug 48 — np.negative only negates positive values

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Negative_OnlyNegatesPositiveValues`
**Severity:** Medium
**Location:** `NDArray.negative.cs` — code has `if (val > 0) out_addr[i] = -val` (only negates positives)

NumPy: `np.negative(x)` is `-x` for ALL elements.
NumSharp: Leaves negative values unchanged, only negates positive values.

```python
>>> np.negative(np.array([-3, -1, 0, 2, 5]))
array([ 3,  1,  0, -2, -5])

# NumSharp returns: [-3, -1, 0, -2, -5]  (negatives unchanged)
```

### Bug 51 — np.log1p computes log10(1+x) instead of ln(1+x)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Log1p_UsesLog10_InsteadOfNaturalLog`
**Severity:** Critical
**Location:** `Default.Log1p.cs` — line 10 delegates to `Log10` instead of `Log`. All branches use `Math.Log10()`.

```python
>>> np.log1p(np.e - 1)
1.0           # ln(e) = 1.0

# NumSharp returns: 0.434  (log10(e))
```

### Bug 56 — np.abs returns Double for integer input

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Abs_ReturnsDouble_ForIntInput`
**Severity:** Medium

NumPy preserves the input dtype: `abs(int32[])` returns `int32`.
NumSharp returns `Double` for all inputs, forcing downstream code to cast.

```python
>>> np.abs(np.array([-1, 2, -3], dtype=np.int32)).dtype
dtype('int32')

# NumSharp: dtype = Double
```

### Bug 58 — astype(int32) rounds instead of truncating

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Astype_Int32_Rounds_InsteadOfTruncating`
**Severity:** Medium
**Root cause:** Uses `Convert.ToInt32` (banker's rounding) instead of C-style truncation toward zero.

```python
>>> np.array([1.7, 2.3, -1.7, -2.3]).astype(np.int32)
array([ 1,  2, -1, -2])  # truncation toward zero

# NumSharp returns: [2, 2, -2, -2]  (banker's rounding)
```

### Bug 60 — np.argmax ignores NaN values

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Argmax_IgnoresNaN`
**Severity:** Medium

NumPy propagates NaN in comparisons — `argmax` returns the index of the first NaN.
NumSharp skips NaN and returns the index of the actual maximum.

```python
>>> np.argmax(np.array([1.0, np.nan, 3.0]))
1  # index of NaN

# NumSharp returns: 2  (index of 3.0)
```

### Bug 61 — np.linspace returns float32 instead of float64

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Linspace_ReturnsFloat32`
**Severity:** Medium

NumPy always returns `float64` by default. NumSharp returns `float32` (Single) when given integer arguments.

```python
>>> np.linspace(0, 1, 5).dtype
dtype('float64')

# NumSharp: dtype = System.Single
```

---

## Open Bugs — Crashes / Throws (12 bugs, 16 tests)

Functions that throw exceptions instead of producing results.

### Bug 6 — != operator throws InvalidCastException on broadcast (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_NotEquals_NDArrayBroadcast_Throws`
**Severity:** High
**Location:** `NDArray.NotEquals.cs`

The `!=` operator signature is `op_Inequality(NDArray, Object)`. When the right operand is an NDArray, it gets boxed as Object, then the implementation tries to convert it via `IConvertible` (scalar path), which fails. The `==` operator has a separate overload that handles NDArray vs NDArray correctly.

```python
>>> np.array([1,2,3]) != np.array([[1],[2],[3]])
array([[False,  True,  True],
       [ True, False,  True],
       [ True,  True, False]])

# NumSharp: InvalidCastException: Unable to cast 'NDArray' to 'IConvertible'
```

### Bug 7 — np.allclose throws NullReferenceException (2 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Allclose_AlwaysThrows`, `Bug_Allclose_BroadcastThrows`
**Severity:** Critical — entirely non-functional
**Location:** `Default.AllClose.cs:23` → `np.all.cs:29`

The function crashes for ALL inputs, including `allclose(a, a)`. The crash chain: `allclose` computes `abs(a-b) <= atol+rtol*abs(b)` then calls `np.all()`. The `<=` operator fails for NDArray vs NDArray (Bug 8/59), producing a null intermediate which `np.all()` dereferences.

**Dependencies:** Fixing Bug 59 (`<=` operator) may cascade-fix this.

```python
>>> np.allclose(np.array([1.,2.,3.]), np.array([1.,2.,3.]))
True

# NumSharp: NullReferenceException
```

### Bug 8 — >, < operators throw IncorrectShapeException (3 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_GreaterThan_NDArrayVsNDArray_SameShape`, `Bug_LessThan_NDArrayVsNDArray_SameShape`, `Bug_GreaterThan_NDArrayVsNDArray_Broadcast`
**Severity:** Critical
**Location:** `NDArray.Greater.cs`, `NDArray.Lower.cs`

The `>`, `<`, `>=`, `<=` operators only support NDArray vs scalar (e.g., `array > 5`). When both operands are NDArray — even with the exact same shape — they throw `IncorrectShapeException`.

**Operator support matrix:**

| Operator | NDArray vs NDArray | NDArray vs scalar |
|----------|-------------------|-------------------|
| `==` | Works (with broadcasting) | Works |
| `!=` | Crashes (InvalidCastException) — Bug 6 | Works |
| `>` | Crashes (IncorrectShapeException) — Bug 8 | Works |
| `<` | Crashes (IncorrectShapeException) — Bug 8 | Works |
| `>=` | Crashes (IncorrectShapeException) — Bug 59 | Works |
| `<=` | Crashes (IncorrectShapeException) — Bug 59 | Works |

**Fix approach:** Implement NDArray vs NDArray comparison in `NDArray.Greater.cs` and `NDArray.Lower.cs`, following the pattern used by `NDArray.Equals.cs` which works.

### Bug 16 — argsort crashes on any 2D array (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_Argsort_2D_Crashes`
**Severity:** High — not broadcast-specific, affects all 2D arrays

`NDArray.argsort<T>()` throws `InvalidOperationException: "Failed to compare two elements in the array"` for ANY 2D array. 1D arrays work correctly.

```python
>>> np.argsort(np.array([[3,1,2],[6,4,5]]))
array([[1, 2, 0], [1, 2, 0]])

# NumSharp: InvalidOperationException
```

### Bug 22 — np.any(axis) throws InvalidCastException (1 test)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Any_WithAxis_AlwaysThrows`, `Bug_Any_WithAxis1_AlwaysThrows`
**Severity:** High
**Related:** Bug 33 (inverted logic) is a secondary issue that becomes visible after fixing the throw.

The axis reduction path casts NDArray to `NDArray<Boolean>`, which fails. Same root cause as Bug 49 (`np.all(axis)`).

```python
>>> np.any(np.array([[True,False],[False,True]]), axis=0)
array([ True,  True])

# NumSharp: InvalidCastException
```

### Bug 46 — Boolean mask setter throws NotImplementedException (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_BooleanMaskSetter_ThrowsNotImplemented`
**Severity:** High
**Location:** `NDArray.Indexing.Masking.cs:26`

`a[mask] = value` is a fundamental NumPy operation (e.g., `a[a > 5] = 0`). The getter works, but the setter throws `NotImplementedException`.

```python
>>> a = np.array([1,2,3,4,5])
>>> a[a > 3] = 0
>>> a
array([1, 2, 3, 0, 0])

# NumSharp: NotImplementedException
```

### Bug 49 — np.all(axis) throws InvalidCastException (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_All_WithAxis_ThrowsInvalidCast`
**Severity:** High
**Location:** `np.all.cs:89`

The cast `(NDArray<bool>)zeros<bool>(outputShape)` fails because `zeros<bool>` returns `NDArray`, not `NDArray<bool>`. The implementation logic itself (`ComputeAllPerAxis`) is correct and handles all 12 dtypes, but can never be reached.

Note: `DefaultEngine.All(NDArray, int axis)` at `Default.All.cs:44` also throws `NotImplementedException`, but `np.all(axis)` doesn't call `DefaultEngine` — it has its own inline implementation.

```python
>>> np.all([[True, False], [True, True]], axis=0)
array([ True, False])

# NumSharp: InvalidCastException
```

### Bug 50 — nd.roll only supports 3 of 12 dtypes (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Roll_Int64_ThrowsNotImplemented`, `Bug_Roll_Byte_ThrowsNotImplemented`
**Severity:** Medium
**Location:** `NDArray.roll.cs`

`nd.roll(shift, axis)` only handles `Int32`, `Single`, `Double`. All other 9 dtypes (`Boolean`, `Byte`, `Int16`, `UInt16`, `UInt32`, `Int64`, `UInt64`, `Char`, `Decimal`) throw `NotImplementedException`.

```python
>>> np.roll(np.array([1,2,3], dtype=np.int64), 1)
array([3, 1, 2])

# NumSharp: NotImplementedException for int64, byte, etc.
```

### Bug 55 — np.mean crashes on empty arrays (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Mean_EmptyArray_Crashes`
**Severity:** Medium

NumPy returns `NaN` with a RuntimeWarning. NumSharp throws `InvalidOperationException` because `NDIterator` cannot handle empty shapes.

```python
>>> np.mean(np.array([]))
nan  # with RuntimeWarning: Mean of empty slice

# NumSharp: InvalidOperationException: Can't construct NDIterator with an empty shape
```

### Bug 57 — np.sum/np.mean crash on boolean arrays (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Sum_BoolArray_Crashes`, `Bug_Mean_BoolArray_Crashes`
**Severity:** Medium

Boolean reductions are common for counting elements matching a condition (e.g., `np.sum(arr > threshold)`). NumSharp throws `NotSupportedException` because boolean is not handled in the reduction type switch.

```python
>>> np.sum(np.array([True, False, True, True]))
3
>>> np.mean(np.array([True, False, True, True]))
0.75

# NumSharp: NotSupportedException
```

### Bug 59 — >= and <= operators throw IncorrectShapeException (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_GreaterOrEqual_Scalar_Crashes`, `Bug_LessOrEqual_Scalar_Crashes`
**Severity:** High

Unlike `>` and `<` (which work with scalar int), `>=` and `<=` throw `IncorrectShapeException` even for scalar right-hand operands.

```python
>>> np.arange(5) >= 2
array([False, False,  True,  True,  True])

# NumSharp: IncorrectShapeException
```

### Bug 62 — Implicit conversion operators crash across dtypes (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_ImplicitConversion_CrossDtype_Crashes`
**Severity:** High

`(double)ndarray` where ndarray is `float32` throws `IncorrectShapeException`. The operator uses `GetAtIndex<T>` which reads raw bytes without dtype conversion. Additionally, NumSharp creates shape `(1,)` instead of `()` for scalars, and the implicit operator requires `ndim=0`.

```python
>>> float(np.array(3.14, dtype=np.float32))
3.140000104904175

# NumSharp: IncorrectShapeException
```

---

## Open Bugs — Wrong Values (15 bugs, 21 tests)

Functions that return results with incorrect values.

### Bug 2 — broadcast_to lacks read-only protection (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_BroadcastTo_NoReadOnlyProtection`
**Severity:** Medium — silent data corruption risk

In NumPy, `broadcast_to` returns a read-only view (`y.flags.writeable = False`). This is critical because broadcast views have zero-stride dimensions — multiple logical positions map to the same physical memory. NumSharp has no read-only concept; `SetInt32` succeeds silently, corrupting shared memory.

**Fix approach:** Add a `Writeable` flag to `NDArray` or `UnmanagedStorage`. `broadcast_to` sets it to `false`. All `Set*` methods check it.

```python
>>> y = np.broadcast_to(np.array([1,2,3,4]), (2,4))
>>> y[0,0] = 999
ValueError: assignment destination is read-only

# NumSharp: SetInt32(999, 0, 0) succeeds, corrupts original
```

### Bug 5/9 — np.minimum broadcast produces transposed values (3 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Minimum_IntBroadcast_WrongValues`, `Bug_Minimum_DoubleBroadcast_WrongValues`, `Bug_Minimum_FloatBroadcast_WrongValues`
**Severity:** High

`np.minimum` with broadcasting transposes the b column vector values between rows. The bug affects all numeric types: int, float, double. `np.maximum` with identical inputs returns correct values — proving the broadcasting infrastructure is correct and the bug is in `minimum`'s specific iteration logic.

```python
>>> np.minimum(np.array([1,5,3]), np.array([[2],[4]]))
array([[1, 2, 2],
       [1, 4, 3]])

# NumSharp: [[1, 4, 2], [1, 2, 3]]  (b values transposed between rows)
```

### Bug 10 — np.unique returns unsorted results (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_Unique_NotSorted`
**Severity:** Medium — not broadcast-specific

NumPy guarantees sorted output. NumSharp returns elements in first-encounter order (likely uses HashSet internally).

```python
>>> np.unique(np.array([3, 1, 2, 1, 3]))
array([1, 2, 3])  # sorted

# NumSharp: [3, 1, 2]  (encounter order)
```

### Bug 11 — flatten on column-broadcast gives Fortran-order (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_Flatten_ColumnBroadcast_WrongOrder`
**Severity:** High

`flatten()` should return elements in C-order (row-major). For column-broadcast arrays, it iterates down columns first (Fortran-order). `np.ravel()` on the same array returns correct results via a different code path.

```python
>>> np.broadcast_to(np.array([[1],[2],[3]]), (3,3)).flatten()
array([1, 1, 1, 2, 2, 2, 3, 3, 3])  # C-order

# NumSharp: [1, 2, 3, 1, 2, 3, 1, 2, 3]  (Fortran-order)
```

### Bug 13 — cumsum with axis on broadcast reads garbage memory (3 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Cumsum_Axis0_RowBroadcast_Garbage`, `Bug_Cumsum_Axis0_ColBroadcast_Garbage`, `Bug_Cumsum_Axis1_ColBroadcast_Wrong`
**Severity:** Critical — returns uninitialized memory values

`cumsum` uses linear pointer iteration along the reduction axis instead of coordinate-based access, reading garbage memory for broadcast arrays. Values like `-1564032936` and `32765` appear in results.

```python
>>> np.cumsum(np.broadcast_to(np.array([1,2,3]), (3,3)), axis=0)
array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])

# NumSharp: [[garbage, garbage, garbage], ...]
```

### Bug 14 — roll on broadcast produces zeros/wrong values (2 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Roll_RowBroadcast_ZerosInSecondRow`, `Bug_Roll_ColBroadcast_ZerosAfterFirstRow`
**Severity:** High

`NDArray.roll(shift, axis)` on broadcast arrays produces zeros in rows beyond the first. The element copy loop uses linear memory access that doesn't account for broadcast zero-strides.

```python
>>> np.roll(np.broadcast_to(np.array([1,2,3]), (2,3)), 1, axis=1)
array([[3, 1, 2], [3, 1, 2]])

# NumSharp: [[3, 1, 2], [0, 0, 0]]  (row 1 = zeros)
```

### Bug 23 — Broadcast arithmetic/reshape wrong values (2 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Reshape_ColBroadcast_WrongOrder`, `Bug_Abs_Broadcast_Throws`
**Severity:** Medium

`reshape` on column-broadcast uses `_reshapeBroadcast` with default strides and `offset % OriginalShape.size` modular arithmetic, walking the original storage linearly instead of in logical row-major order.

```python
>>> np.broadcast_to(np.array([[10],[20],[30]]), (3,3)).reshape(9)
array([10, 10, 10, 20, 20, 20, 30, 30, 30])

# NumSharp: [10, 20, 30, 10, 20, 30, 10, 20, 30]  (wrong order)
```

### Bug 24 — Transpose on column-broadcast wrong values (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_Transpose_ColBroadcast_WrongValues`
**Severity:** Medium

Transposing a broadcast array materializes data via Clone but creates a plain contiguous shape, losing stride=0 broadcast semantics.

```python
>>> np.broadcast_to(np.array([[10],[20],[30]]), (3,3)).T
array([[10, 20, 30],
       [10, 20, 30],
       [10, 20, 30]])

# NumSharp: [[10, 10, 10], [10, 10, 10], [10, 10, 10]]
```

### Bug 27 — np.roll static returns int instead of NDArray (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_NpRoll_ReturnsInt_ShouldReturnNDArray`
**Severity:** High — return type is completely wrong
**Location:** `APIs/np.array_manipulation.cs:16`

The static `np.roll()` has return type `int` and casts `nd.roll(shift, axis)` to int. In NumPy, `np.roll` always returns an ndarray.

```python
>>> np.roll(np.array([1,2,3,4]), 1)
array([4, 1, 2, 3])
```

### Bug 28 — floor/ceil cast int to Double (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Floor_IntArray_ShouldPreserveDtype`, `Bug_Ceil_IntArray_ShouldPreserveDtype`
**Severity:** Medium
**Location:** `NPTypeCode.cs:577` — `GetComputingType()` maps all integer types to Double. `Default.Floor.cs` switch only handles Double/Single/Decimal.

NumPy 2.1+: `np.floor(int_array)` returns `int_array` unchanged (no-op for integers).

```python
>>> np.floor(np.array([1, 2, 3], dtype=np.int32)).dtype
dtype('int32')

# NumSharp: dtype = Double
```

### Bug 29 — fmax/fmin don't ignore NaN (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Fmax_ShouldIgnoreNaN`, `Bug_Fmin_ShouldIgnoreNaN`
**Severity:** Medium

NumPy: `np.fmax` ignores NaN (`fmax(NaN, 1) = 1`), `np.maximum` propagates NaN (`maximum(NaN, 1) = NaN`).
NumSharp: Both `fmax` and `maximum` have identical implementations (both propagate NaN).

```python
>>> np.fmax(np.nan, 1.0)
1.0    # ignores NaN
>>> np.maximum(np.nan, 1.0)
nan    # propagates NaN

# NumSharp: fmax(NaN, 1.0) = NaN  (same as maximum)
```

### Bug 33 — np.any(axis) has inverted logic (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Any_Axis0_InvertedLogic_ImplementsAllInsteadOfAny`, `Bug_Any_Axis0_Keepdims`
**Severity:** High
**Location:** `Logic/np.any.cs` — `ComputeAnyPerAxis`

The implementation initializes `currentResult=true` and sets it to `true` on break when finding a zero — this is `all()` logic. For `any()`, should initialize `false` and set `true` on non-zero.

**Note:** Bug 22 (the InvalidCastException throw) must be fixed first before this logic bug becomes visible.

```python
>>> np.any([[False, False], [False, True]], axis=0)
array([False,  True])
```

### Bug 52 — std/var ignore ddof parameter (2 tests)

**File:** `OpenBugs.DeprecationAudit.cs`
**Tests:** `Bug_Std_IgnoresDdof`, `Bug_Var_IgnoresDdof`
**Severity:** High — `ddof=1` is the standard unbiased estimator used in almost all statistical analysis

The `ddof` parameter is accepted but never used in the calculation. Sample std/var (`ddof=1`) always returns population std/var (`ddof=0`).

```python
>>> a = np.array([2, 4, 4, 4, 5, 5, 7, 9])
>>> np.std(a, ddof=0)
2.0
>>> np.std(a, ddof=1)
2.1380899352993952

# NumSharp: std(ddof=1) = 2.0  (same as ddof=0)
```

### Bug 53 — searchsorted returns wrong indices (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Searchsorted_WrongIndices`
**Severity:** High — binary search algorithm is fundamentally broken

Results are wrong, not just off-by-one. Also, `side='right'` is not implemented.

```python
>>> np.searchsorted([1,3,5,7], 2)
1  # insert at index 1 to maintain sort order

# NumSharp: returns 3
```

### Bug 54 — moveaxis fails with negative axis (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_Moveaxis_NegativeAxis_NoOp`
**Severity:** High — negative axis is a very common Python idiom
**Location:** `Default.Transpose.cs`, `normalize_axis_tuple()`

The method normalizes axes against `axis.Length` (which is 1 for single-axis moveaxis) instead of against `ndim`. So `moveaxis(a, 0, -1)` on a 3D array normalizes -1 against length 1, giving 0, making the operation a no-op.

```python
>>> np.moveaxis(np.arange(24).reshape(2,3,4), 0, -1).shape
(3, 4, 2)

# NumSharp: returns (2, 3, 4) unchanged  (no-op)
```

### Bug 63 — ToString crashes on empty arrays (1 test)

**File:** `OpenBugs.DeprecationAudit.cs`
**Test:** `Bug_ToString_EmptyArray_Crashes`
**Severity:** Low
**Location:** `NDIterator.cs:64`

`NDIterator` cannot handle empty shapes. NumPy returns `"[]"`.

```python
>>> str(np.array([]))
'[]'

# NumSharp: InvalidOperationException: Can't construct NDIterator with an empty shape
```

---

## Additional Open Bugs (broadcast-related, in OpenBugs.cs)

### Bug 18 — cumsum output uses broadcast shape (2 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Cumsum_OutputBroadcastShape_RowBroadcast_Axis0`, `Bug_Cumsum_OutputBroadcastShape_ColBroadcast_Axis1`
**Severity:** High
**Location:** `Default.Reduction.CumAdd.cs`, `ReduceCumAdd()`, line 43

Independent of the GetViewInternal fix (Bug 17). Cumsum creates its output array using the broadcast shape (`IsBroadcasted=true`). When cumsum writes via `ret[slices]`, `GetViewInternal` sees `IsBroadcasted=true` and clones ret's data into a detached buffer. `MoveNextReference` writes to the clone, not the original ret. Result: ret remains uninitialized — garbage values.

**Fix:** Use `new Shape(shape.dimensions)` to strip broadcast metadata from the output shape.

### Bug 19 — roll uses Data<T>() on broadcast arrays (2 tests)

**File:** `OpenBugs.cs`
**Tests:** `Bug_Roll_DataT_RowBroadcast`, `Bug_Roll_DataT_ColBroadcast`
**Severity:** High
**Location:** `NDArray.roll.cs`, line 26

`Data<T>()` returns the raw underlying buffer. For `broadcast_to([1,2,3], (3,3))`, this is the original 3-element buffer, not the virtual 9-element expansion. The loop iterates `this.size = 9` times, reading past the buffer boundary.

**Fix:** Call `.copy()` at the start, or use `NDIterator` instead of `Data<T>()`.

### Bug 20 (GetCoordinates) — see Bug 20 in Fixed section

This bug's tests pass now (included in the 19 passing), but the underlying `GetCoordinates()` function still produces wrong coordinates for broadcast shapes. The tests pass because the operations that called `GetCoordinates` on broadcast shapes were fixed to use different code paths.

### Bug 21 — Boolean mask indexing returns wrong shape for 2D+ masks (1 test)

**File:** `OpenBugs.cs`
**Test:** `Bug_BooleanMask_2D_WrongShape`
**Severity:** Medium
**Location:** `Selection/NDArray.Indexing.Masking.cs`

NumPy: `arr[bool_mask]` where both are 2D returns a 1D array of elements where mask is True, shape `(count_true,)`.
NumSharp: Treats each True in the mask as selecting an entire row/slice, producing shape `(count_true, *remaining_dims)`.

```python
>>> a = np.array([[1,2,3],[4,5,6],[7,8,9]])
>>> mask = np.array([[F,F,T],[F,T,F],[T,F,F]])
>>> a[mask]
array([3, 5, 7])  # shape (3,)

# NumSharp: shape (3, ...) instead of (3,)
```

---

## Architectural Root Cause Analysis

### The broadcast stride/data mismatch (Bug 17) — FIXED

The majority of broadcast bugs traced to a single architectural problem in `UnmanagedStorage.Slicing.cs:100-101`:

```csharp
if (_shape.IsBroadcasted)
    return Clone().Alias(_shape.Slice(slices));
```

`Clone()` correctly materialized broadcast data into contiguous memory, but `_shape.Slice(slices)` computed the sliced shape from the original broadcast shape, inheriting broadcast strides. The resulting storage had data laid out with strides `[3, 1]` but the shape claimed `[1, 0]`.

This was fixed and bugs 1, 3, 4, 12, 15, 17, 20 now pass.

### Multiple iteration paths

NumSharp has MULTIPLE code paths for traversing array elements:

1. **Coordinate-based access** (`Shape.GetOffset(coords)`, used by `GetInt32`/`GetDouble`) — correctly combines ViewInfo strides with BroadcastInfo zero-strides.

2. **Linear/flat iteration** (used by `ToString`, `flatten`, `concatenate`, `np.minimum`) — computes offsets differently and may produce wrong results for broadcasted arrays.

3. **Data<T>() direct access** — returns the raw underlying buffer, which for broadcast arrays is the small source buffer, not the virtual expansion.

The remaining broadcast bugs (5/9, 11, 13, 14, 18, 19, 23, 24) all stem from code paths using #2 or #3 instead of #1.

### Comparison operator inconsistency

The six comparison operators have wildly inconsistent implementations:

- `==` works for NDArray vs NDArray with broadcasting (fully implemented)
- `!=` attempts NDArray vs NDArray but crashes via IConvertible (Bug 6)
- `>`, `<` throw for NDArray vs NDArray entirely (Bug 8)
- `>=`, `<=` throw even for NDArray vs scalar in some cases (Bug 59)

All should follow the `==` implementation pattern.

### Dead code pattern

13 functions follow the same pattern: the method signature exists, accepts parameters, but the body either returns `null`, returns `default`, or has its implementation commented out. These are all in the `Operations/`, `LinearAlgebra/`, and `Logic/` directories. Several were planned for LAPACK integration that was never completed.

---

## Dependency Graph

Some bugs block or depend on others:

```
Bug 36 (isclose → null) ──blocks──► Bug 7 (allclose → NullRef)
Bug 59 (<=  → throws)   ──blocks──► Bug 7 (allclose uses <= internally)
Bug 22 (any throws)     ──blocks──► Bug 33 (any inverted logic — hidden by throw)
Bug 49 (all throws)     ──shares root cause──► Bug 22 (both: NDArray<bool> cast)
Bug 45 (roll no-axis)   ──related──► Bug 50 (roll limited dtypes)
Bug 27 (np.roll int)    ──related──► Bug 45, Bug 50
```

**Recommended fix order for maximum cascade impact:**

1. Fix Bug 59 (`>=`, `<=`) + Bug 8 (`>`, `<`) + Bug 6 (`!=`) — unblocks Bug 7
2. Fix Bug 36 (`isclose`) — unblocks Bug 7 fully
3. Fix Bug 22/49 (`any`/`all` InvalidCastException) — unblocks Bug 33
4. Fix Bug 51 (`log1p`) — trivial one-line fix (change `Log10` to `Log`)
5. Fix Bug 47/48 (`positive`/`negative`) — trivial logic fixes

---

## Build & Test

```bash
# Build
dotnet_build test/NumSharp.UnitTest/NumSharp.UnitTest.csproj

# Run all OpenBugs tests
dotnet_test_detailed test/NumSharp.UnitTest/NumSharp.UnitTest.csproj --filter "FullyQualifiedName~OpenBugs"

# Run specific bug
dotnet_test_detailed test/NumSharp.UnitTest/NumSharp.UnitTest.csproj --filter "FullyQualifiedName~Bug_Log1p"

# Run only broadcast bugs
dotnet_test_detailed test/NumSharp.UnitTest/NumSharp.UnitTest.csproj --filter "FullyQualifiedName~OpenBugs" --filter "FullyQualifiedName!~DeprecationAudit"
```
