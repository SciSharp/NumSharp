# Container Protocol Battle Test Plan

**Created**: 2026-04-08
**Status**: In Progress
**Branch**: longindexing

---

## Overview

Battle testing all 6 container protocol capabilities for NDArray to ensure NumPy compatibility and edge case handling.

## Files Created

| File | Purpose |
|------|---------|
| `src/NumSharp.Core/Backends/NDArray.Container.cs` | Container protocol implementation |
| `src/NumSharp.Core/Exceptions/TypeError.cs` | NumPy-compatible TypeError |
| `test/NumSharp.UnitTest/Backends/ContainerProtocolTests.cs` | Basic tests (25 tests, all pass) |
| `test/NumSharp.UnitTest/Backends/ContainerProtocolBattleTests.cs` | Comprehensive battle tests |

## Implemented Methods

| C# Method | Python Method | Status |
|-----------|---------------|--------|
| `Contains(object)` | `__contains__(object)` | Implemented |
| `GetHashCode()` | `__hash__()` | Implemented (throws NotSupportedException) |
| `__len__()` | `__len__()` | Implemented |
| `__iter__()` | `__iter__()` | Implemented |
| `__getitem__(int/long/string)` | `__getitem__()` | Implemented |
| `__setitem__(int/long/string, object)` | `__setitem__()` | Implemented |

---

## Task Status

### Task 1: Battle test `__len__` - COMPLETE
All tests pass:
- `Len_AllDtypes_1DArray` - All 12 dtypes work
- `Len_3DArray_ReturnsFirstDimension`
- `Len_4DArray_ReturnsFirstDimension`
- `Len_EmptyArray_ReturnsZero`
- `Len_EmptyArray_2D_ReturnsZero`
- `Len_SlicedArray_ReturnsCorrectLength`
- `Len_SlicedArray_Strided_ReturnsCorrectLength`
- `Len_SlicedArray_Reversed_ReturnsCorrectLength`
- `Len_TransposedArray_ReturnsFirstDimension`
- `Len_ReshapedArray_ReturnsFirstDimension`
- `Len_BroadcastArray_ReturnsFirstDimension`
- `Len_SingleElementArray_ReturnsOne`
- `Len_ScalarAllDtypes_ThrowsTypeError` - All 12 scalar types throw TypeError

### Task 2: Battle test `__getitem__` - COMPLETE
All tests pass:
- `GetItem_AllDtypes_IntIndex` - All 12 dtypes work
- `GetItem_NegativeIndex_AllPositions` - -1 through -5
- `GetItem_SliceStrings_Various` - Start:Stop, :Stop, Start:, ::Step, ::-1, negative
- `GetItem_2DArray_RowAccess`
- `GetItem_SliceReturnsView_NotCopy`
- `GetItem_OutOfBounds_Throws`
- `GetItem_EmptySlice_ReturnsEmptyArray`
- `GetItem_SlicedSourceArray`

### Task 3: Battle test `__setitem__` - NEEDS VERIFICATION
Tests written, need to verify after rebuild:
- `SetItem_AllDtypes_ScalarAssignment`
- `SetItem_NegativeIndex`
- `SetItem_SliceString_ScalarBroadcast`
- `SetItem_SliceString_ArrayAssignment`
- `SetItem_ViewAffectsOriginal`
- `SetItem_2DArray_RowAssignment`
- `SetItem_TypePromotion`
- `SetItem_AllElements_WithColon`
- `SetItem_ReversedSlice` - Fixed test expectation to match NumPy

### Task 4: Battle test `__hash__` / `GetHashCode` - COMPLETE
All tests pass:
- `Hash_AllDtypes_Throw` - All 12 dtypes throw NotSupportedException
- `Hash_EmptyArray_Throws`
- `Hash_ScalarArray_Throws`
- `Hash_SlicedArray_Throws`
- `Hash_BroadcastArray_Throws`
- `Hash_HashSetUsage_Fails`
- `Hash_ErrorMessage_IsDescriptive`

### Task 5: Battle test `__contains__` / `Contains` - COMPLETE
All tests pass:
- `Contains_AllDtypes` - All 12 dtypes work
- `Contains_LargeArray` - 10000 elements
- `Contains_Infinity` - +inf, -inf handling
- `Contains_NegativeValues`
- `Contains_3DArray`
- `Contains_SlicedArray`
- `Contains_StridedArray`
- `Contains_BroadcastArray`
- `Contains_ScalarArray`

### Task 6: Battle test `__iter__` - NEEDS VERIFICATION
Tests fixed (iteration yields values for 1D, NDArray for N-D), need to verify:
- `Iter_AllDtypes_Enumerate`
- `Iter_1DArray_ElementsMatch` - Fixed: yields int values, not NDArray
- `Iter_2DArray_IteratesRows` - Fixed: check for NDArray type
- `Iter_3DArray_Iterates2DSlices` - Fixed: check for NDArray type
- `Iter_EmptyArray_NoIterations`
- `Iter_SingleElement_OneIteration` - Fixed: yields int values
- `Iter_MultipleEnumeration` - Fixed: uses Convert.ToInt32
- `Iter_LINQ_ToList` - Fixed: Cast<object>
- `Iter_LINQ_Count` - Fixed: Cast<object>
- `Iter_SlicedArray`
- `Iter_BreakEarly`

---

## How to Continue

### Step 1: Rebuild and Run Tests

```bash
cd K:\source\NumSharp-longindexing

# Build
dotnet build test/NumSharp.UnitTest -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0

# Run battle tests
cd test/NumSharp.UnitTest
dotnet run --framework net10.0 --no-build -- --output Detailed 2>&1 | grep -E "(passed|failed) (Len_|GetItem_|SetItem_|Hash_|Contains_|Iter_)"
```

### Step 2: Fix Any Remaining Failures

If tests fail, investigate:

1. **Iteration tests** (`Iter_*`): The iterator behavior differs by dimensionality:
   - 1D arrays: yields scalar values (int, double, etc.)
   - N-D arrays: yields (N-1)-D NDArray slices

2. **SetItem tests** (`SetItem_*`): Watch for:
   - Type conversion issues
   - Reversed slice assignment (source[0] goes to arr[last])

### Step 3: Update Task Status

Use TaskUpdate to mark completed tasks:
```
TaskUpdate taskId=1 status=completed  # __len__
TaskUpdate taskId=2 status=completed  # __getitem__
TaskUpdate taskId=3 status=completed  # __setitem__
TaskUpdate taskId=4 status=completed  # __hash__
TaskUpdate taskId=5 status=completed  # __contains__
TaskUpdate taskId=6 status=completed  # __iter__
```

### Step 4: Summary of Results

After all tests pass, summarize:
- Total tests: ~68 battle tests
- Dtypes covered: All 12
- Edge cases: empty, scalar, sliced, strided, broadcast, transposed
- Memory layouts: contiguous, non-contiguous, views

---

## Test Details by Capability

### `__len__` Tests (13 tests)

```
Len_AllDtypes_1DArray          - Tests all 12 dtypes
Len_3DArray_ReturnsFirstDimension
Len_4DArray_ReturnsFirstDimension
Len_EmptyArray_ReturnsZero
Len_EmptyArray_2D_ReturnsZero
Len_SlicedArray_ReturnsCorrectLength
Len_SlicedArray_Strided_ReturnsCorrectLength
Len_SlicedArray_Reversed_ReturnsCorrectLength
Len_TransposedArray_ReturnsFirstDimension
Len_ReshapedArray_ReturnsFirstDimension
Len_BroadcastArray_ReturnsFirstDimension
Len_SingleElementArray_ReturnsOne
Len_ScalarAllDtypes_ThrowsTypeError  - All 12 scalar types
```

### `__getitem__` Tests (8 tests)

```
GetItem_AllDtypes_IntIndex     - Tests all 12 dtypes
GetItem_NegativeIndex_AllPositions
GetItem_SliceStrings_Various
GetItem_2DArray_RowAccess
GetItem_SliceReturnsView_NotCopy
GetItem_OutOfBounds_Throws
GetItem_EmptySlice_ReturnsEmptyArray
GetItem_SlicedSourceArray
```

### `__setitem__` Tests (9 tests)

```
SetItem_AllDtypes_ScalarAssignment
SetItem_NegativeIndex
SetItem_SliceString_ScalarBroadcast
SetItem_SliceString_ArrayAssignment
SetItem_ViewAffectsOriginal
SetItem_2DArray_RowAssignment
SetItem_TypePromotion
SetItem_AllElements_WithColon
SetItem_ReversedSlice
```

### `__hash__` Tests (7 tests)

```
Hash_AllDtypes_Throw           - Tests all 12 dtypes
Hash_EmptyArray_Throws
Hash_ScalarArray_Throws
Hash_SlicedArray_Throws
Hash_BroadcastArray_Throws
Hash_HashSetUsage_Fails
Hash_ErrorMessage_IsDescriptive
```

### `__contains__` Tests (12 tests)

```
Contains_AllDtypes             - Tests all 12 dtypes
Contains_LargeArray
Contains_Infinity
Contains_NegativeValues
Contains_3DArray
Contains_SlicedArray
Contains_StridedArray
Contains_BroadcastArray
Contains_ScalarArray
Contains_Null_ReturnsFalse
Contains_EmptyArray_ReturnsFalse
Contains_TypePromotion_IntInFloatArray
```

### `__iter__` Tests (11 tests)

```
Iter_AllDtypes_Enumerate       - Tests all 12 dtypes
Iter_1DArray_ElementsMatch
Iter_2DArray_IteratesRows
Iter_3DArray_Iterates2DSlices
Iter_EmptyArray_NoIterations
Iter_SingleElement_OneIteration
Iter_MultipleEnumeration
Iter_LINQ_ToList
Iter_LINQ_Count
Iter_SlicedArray
Iter_BreakEarly
```

---

## Known Issues Fixed

### Issue 1: Iteration Type Mismatch
**Problem**: Tests expected `NDArray` for all iterations, but 1D arrays yield scalar values.
**Fix**: Updated tests to use `Convert.ToInt32(item)` for 1D arrays and check `item is NDArray` for N-D.

### Issue 2: Reversed Slice Assignment Expectation
**Problem**: Test expected `arr[0]=5` after `arr[::-1] = [5,4,3,2,1]`, but NumPy assigns in reverse.
**Fix**: Updated test to use different values and correct expectations:
```csharp
arr[::-1] = [10, 20, 30, 40, 50]
// Result: arr = [50, 40, 30, 20, 10]
// arr[0] = 50 (last of source), arr[4] = 10 (first of source)
```

### Issue 3: broadcast_to Ambiguity
**Problem**: `np.broadcast_to(arr, new[] { 4, 3 })` was ambiguous.
**Fix**: Changed to `np.broadcast_to(arr, new Shape(4, 3))`.

### Issue 4: TUnit Assertion Types
**Problem**: `Assert.That(byte).IsEqualTo(2)` didn't compile.
**Fix**: Cast to `int` or `long`: `Assert.That((int)(byte)value).IsEqualTo(2)`.

---

## Commit Instructions

When all tests pass:

```bash
cd K:\source\NumSharp-longindexing
git add src/NumSharp.Core/Backends/NDArray.Container.cs \
        src/NumSharp.Core/Exceptions/TypeError.cs \
        test/NumSharp.UnitTest/Backends/ContainerProtocolTests.cs \
        test/NumSharp.UnitTest/Backends/ContainerProtocolBattleTests.cs \
        docs/plans/CONTAINER_PROTOCOL_BATTLE_TEST_PLAN.md

git commit -m "feat(api): implement Python container protocol for NDArray

Add full container protocol support with both C# and Python naming:
- __contains__ / Contains: membership testing (linear search)
- __hash__ / GetHashCode: throws NotSupportedException (unhashable)
- __len__: returns first dimension length (TypeError for scalars)
- __iter__: returns enumerator over first axis
- __getitem__: indexing with int/long/string slice
- __setitem__: assignment with int/long/string slice

Includes 93 tests covering all 12 dtypes and edge cases:
- Empty arrays, scalar arrays, sliced arrays
- Broadcast arrays, transposed arrays
- Negative indexing, strided access
- Type promotion, view semantics

Closes container protocol items from NUMPY_ALIGNMENT_PLAN2.md Phase 3."
```
