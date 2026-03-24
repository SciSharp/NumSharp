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

### Memory Corruption in Tests (193 failures)

Tests are showing memory corruption symptoms:
- Values like `34359738376` or `-9223365347867329507` appearing instead of expected values
- "index < Count, Memory corruption expected" assertion failures
- Affects clip, view semantics, and other tests

**Likely Causes**:
1. Stride calculations using wrong types somewhere
2. Offset calculations not fully migrated
3. Some kernel paths still using int where long is needed

**Investigation Needed**: Focus on clip kernel and view/slice operations.

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
