# INT64 Migration Progress Report

This document tracks the progress of migrating recent commits to comply with the INT64 Developer Guide (`docs/INT64_DEVELOPER_GUIDE.md`).

---

## Session Summary

**Date**: 2026-03-24 (Updated)
**Focus**: Fixing int32 violations in commits introduced via rebase from master (ikernel branch)

---

## Build Status

**BUILD: PASSING** (0 errors)
**Tests**: 193 failures (memory corruption issues under investigation)

---

## Completed Fixes (Session 4)

### 32. Default.All.cs - Loop counter fix

**Location**: `src/NumSharp.Core/Backends/Default/Logic/Default.All.cs`

**Changes**:
- `var len = nd.size; for (int i = 0; i < len; i++)` → `long len = nd.size; for (long i = 0; i < len; i++)`

### 33. Default.Any.cs - Loop counter fix

**Location**: `src/NumSharp.Core/Backends/Default/Logic/Default.Any.cs`

**Changes**:
- Same fix as Default.All.cs

### 34. np.random.poisson.cs - Loop counter fix

**Location**: `src/NumSharp.Core/RandomSampling/np.random.poisson.cs`

**Changes**:
- `var len = result.size; for (int i = 0; i < len; i++)` → `long len; for (long i ...)`

### 35. np.random.bernoulli.cs - Loop counter fix

**Location**: `src/NumSharp.Core/RandomSampling/np.random.bernoulli.cs`

**Changes**:
- Same pattern fix for loop counter

### 36. StackedMemoryPool.cs - Loop counter fix

**Location**: `src/NumSharp.Core/Backends/Unmanaged/Pooling/StackedMemoryPool.cs`

**Changes**:
- `for (int i = 0; i < count; i++, addr += SingleSize)` → `for (long i = 0; i < count; i++, ...)`
- `count` parameter is `long`, so loop counter must be `long`

### 37. NDArray.Indexing.Masking.cs - Multiple fixes

**Location**: `src/NumSharp.Core/Selection/NDArray.Indexing.Masking.cs`

**Changes**:
- `for (int idx = 0; idx < trueCount; idx++)` → `for (long idx = 0; idx < trueCount; idx++)`
- `indices[dim].GetInt32(idx)` → `indices[dim].GetInt64(idx)` (nonzero now returns `NDArray<long>[]`)
- `int valueIdx = 0; for (int i = 0; i < mask.size; i++)` → `long valueIdx = 0; for (long i ...)`

### 38. np.random.randn.cs - Loop counter fix

**Location**: `src/NumSharp.Core/RandomSampling/np.random.randn.cs`

**Changes**:
- `for (int i = 0; i < array.size; i++)` → `for (long i = 0; i < array.size; i++)`

---

## Completed Fixes (Session 3)

### 30. np.random.shuffle.cs - NextLong fix

**Location**: `src/NumSharp.Core/RandomSampling/np.random.shuffle.cs`

**Changes**:
- `randomizer.NextInt64(i + 1)` → `randomizer.NextLong(i + 1)` (method name was wrong)
- `SwapSlicesAxis0(NDArray x, int i, int j)` → `long i, long j`

### 31. Test fixes for Int64 dtype

**Files**:
- `BattleProofTests.cs`: `GetInt32` → `GetInt64` for arange-based tests
- `np.transpose.Test.cs`: `new long[]` → `new int[]` for axis array (axes stay int)
- `ReadmeExample.cs`: Cast `n_samples` to int for `np.ones()` calls
- `NpApiOverloadTests_LogicManipulation.cs`: `int` → `long` for count_nonzero, `NDArray<int>[]` → `NDArray<long>[]` for nonzero
- `BooleanIndexing.BattleTests.cs`: `shape.SequenceEqual(new[])` → `shape.SequenceEqual(new long[])`

---

## Completed Fixes (Session 2)

### 15. Shape.Broadcasting.cs

**Location**: `src/NumSharp.Core/View/Shape.Broadcasting.cs`

**Changes**:
- `var mit = new int[nd]` -> `new long[nd]` (ResolveReturnShape)
- `int tmp` -> `long tmp` (all methods)
- `var mitDims = new int[nd]` -> `new long[nd]` (Broadcast methods)
- `var broadcastStrides = new int[nd]` -> `new long[nd]`
- `var zeroStrides = new int[...]` -> `new long[...]`
- `var leftStrides/rightStrides = new int[nd]` -> `new long[nd]`
- `int bufSize` -> `long bufSize`
- Constructor calls updated: `(int[])mitDims.Clone()` -> `(long[])mitDims.Clone()`
- `stackalloc int[nd]` -> `stackalloc long[nd]` (AreBroadcastable)

---

### 16. NDArray.cs (Constructor Overloads)

**Location**: `src/NumSharp.Core/Backends/NDArray.cs`

**Changes**:
- Added `public NDArray(NPTypeCode dtype, long size)` constructor
- Added `public NDArray(NPTypeCode dtype, long size, bool fillZeros)` constructor

---

### 17. NumSharp.Core.csproj

**Location**: `src/NumSharp.Core/NumSharp.Core.csproj`

**Changes**:
- Added exclusion for Regen template files: `<Compile Remove="**\*.template.cs" />`
- Added exclusion for disabled files: `<Compile Remove="**\*.regen_disabled" />`

---

### 18. np.random.choice.cs

**Location**: `src/NumSharp.Core/RandomSampling/np.random.choice.cs`

**Changes**:
- Added `using System;` for ArgumentException
- Added size overflow check with explicit cast: `if (a.size > int.MaxValue) throw`

---

### 19. np.random.shuffle.cs

**Location**: `src/NumSharp.Core/RandomSampling/np.random.shuffle.cs`

**Changes**:
- `Shuffle1DContiguous(NDArray x, int n)` -> `long n`
- Added overflow check for shuffle dimension > int.MaxValue
- Loop counters updated for long-compatible iteration

---

### 20. np.size.cs

**Location**: `src/NumSharp.Core/APIs/np.size.cs`

**Changes**:
- Return type `public static int size(...)` -> `long`

---

### 21. Default.Transpose.cs

**Location**: `src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.Transpose.cs`

**Changes**:
- `var emptyDims = new int[n]` -> `new long[n]`
- `var permutedDims = new int[n]` -> `new long[n]`
- `var permutedStrides = new int[n]` -> `new long[n]`
- `int bufSize` -> `long bufSize`

---

### 22. Default.NonZero.cs (Return Type)

**Location**: `src/NumSharp.Core/Backends/Default/Indexing/Default.NonZero.cs`

**Changes**:
- `public override NDArray<int>[] NonZero(...)` -> `NDArray<long>[]`
- `private static unsafe NDArray<int>[] nonzeros<T>(...)` -> `NDArray<long>[]`
- Removed outdated SIMD path using `kp` variable

---

### 23. TensorEngine.cs (NonZero)

**Location**: `src/NumSharp.Core/Backends/TensorEngine.cs`

**Changes**:
- `public abstract NDArray<int>[] NonZero(...)` -> `NDArray<long>[]`

---

### 24. np.nonzero.cs

**Location**: `src/NumSharp.Core/Indexing/np.nonzero.cs`

**Changes**:
- Return type `NDArray<int>[]` -> `NDArray<long>[]`

---

### 25. np.are_broadcastable.cs

**Location**: `src/NumSharp.Core/Creation/np.are_broadcastable.cs`

**Changes**:
- Fixed undefined `DefaultEngine` reference -> `Shape.AreBroadcastable`

---

### 26. np.array.cs

**Location**: `src/NumSharp.Core/Creation/np.array.cs`

**Changes**:
- `int stride1 = strides[0]` -> `long stride1` (4 locations)
- `int stride2 = strides[1]` -> `long stride2` (3 locations)
- `int stride3 = strides[2]` -> `long stride3` (2 locations)
- `int stride4 = strides[3]` -> `long stride4` (1 location)

---

### 27. NDArray.matrix_power.cs

**Location**: `src/NumSharp.Core/LinearAlgebra/NDArray.matrix_power.cs`

**Changes**:
- `np.eye(product.shape[0])` -> `np.eye((int)product.shape[0])` with comment

---

### 28. NDArray.Indexing.Masking.cs

**Location**: `src/NumSharp.Core/Selection/NDArray.Indexing.Masking.cs`

**Changes**:
- `var newShape = new int[...]` -> `new long[...]` (BooleanScalarIndex)
- `int trueCount = indices[0].size` -> `long trueCount`
- `var emptyShape = new int[...]` -> `new long[...]`
- `var resultShape = new int[...]` -> `new long[...]`
- Loop counters `for (int idx = 0; ...)` -> `for (long idx = 0; ...)`
- `indices[dim].GetInt32(idx)` -> `indices[dim].GetInt64(idx)`

---

### 29. ILKernelGenerator.Reduction.Axis.Simd.cs (Delegate)

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Axis.Simd.cs`

**Changes**:
- Lambda parameters updated to match `AxisReductionKernel` delegate:
  - `int* inputStrides` -> `long*`
  - `int* inputShape` -> `long*`
  - `int* outputStrides` -> `long*`
  - `int axisSize` -> `long`
  - `int outputSize` -> `long`
- `AxisReductionSimdHelper<T>` signature updated to match
- `int axisStride` -> `long axisStride`
- Loop counters `for (int outIdx = 0; ...)` -> `for (long outIdx = 0; ...)`
- `int remaining`, `int inputBaseOffset`, `int outputOffset` -> `long`
- `int coord` -> `long coord`
- Fixed integer division in `DivideByCountTyped` for int type

---

## Previously Completed Fixes (Session 1)

### 1-14. (See previous session)

Files fixed in previous session include:
- ILKernelGenerator.Reduction.Arg.cs
- ILKernelGenerator.Reduction.Axis.Simd.cs (partial)
- ILKernelGenerator.Reduction.Axis.cs
- ILKernelGenerator.Reduction.NaN.cs
- ILKernelGenerator.Masking.cs
- Default.Reduction.Nan.cs
- Default.NonZero.cs
- Default.BooleanMask.cs
- TensorEngine.cs
- np.count_nonzero.cs
- np.any.cs
- np.all.cs
- np.random.rand.cs
- Default.Op.Boolean.template.cs (deleted)

---

## Known Issues

### "Memory Corruption" in Tests - ROOT CAUSE IDENTIFIED

The "index < Count, Memory corruption expected" assertion errors are **NOT actual memory corruption**.

**Root Cause**: Tests calling `GetInt32()` on Int64 arrays.

When `np.arange()` was changed to return Int64 (NumPy 2.x alignment), many tests that use `GetInt32()` started failing because:
1. For an Int64 array, `_arrayInt32` is null (default struct with Count=0)
2. Calling `_arrayInt32[anyIndex]` triggers `Debug.Assert(index < Count)` where Count=0
3. This fails for any non-negative index, appearing as "memory corruption"

**Solution**: Update tests to use `GetInt64()` instead of `GetInt32()` when working with arrays created by `np.arange()`.

**Verified**: Scripts using correct getter methods work perfectly.

### Remaining Test Updates Needed

Tests that use `np.arange()` followed by `GetInt32()` need to be updated:
- `NegativeSlice_2D_Corner`, `NegativeSlice_2D_FullReverse`
- `BooleanIndex_2D_Flattens`
- `Dot_1D_2D_Larger`
- Various `Base_*` memory leak tests
- NDIterator reference tests (separate issue - casting during iteration)

---

## Remaining Work

### Medium Priority - Not Yet Audited

| File | Issue |
|------|-------|
| `np.load.cs` | `int[] shape` declarations |
| `np.save.cs` | `int[] shape` declarations |
| `NDArray.cs` | `int[] indices` parameters (may keep for API compat) |
| `NdArray.ReShape.cs` | `int[] shape` parameters |
| `NDArray`1.ReShape.cs` | `int[] shape` parameters |

### User Request: Random Functions

User requested all random functions support long. Current state:
- Some random functions have int limits due to `Random.Next(int)` limitation
- Need to add long overloads or use alternative random generation for large arrays

---

## Next Steps

1. **Investigate memory corruption** - Focus on clip kernel and stride calculations
2. **Add long support to random functions** - User requirement
3. **Audit remaining files** - np.load.cs, np.save.cs, reshape methods
4. **Run full test suite** after corruption fix

---

## Reference

- **Developer Guide**: `docs/INT64_DEVELOPER_GUIDE.md`
- **Migration Plan**: `docs/INT64_INDEX_MIGRATION.md`
- **Issues Tracker**: `docs/LONG_INDEXING_ISSUES.md`
