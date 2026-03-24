# INT64 Migration Progress Report

This document tracks the progress of migrating recent commits to comply with the INT64 Developer Guide (`docs/INT64_DEVELOPER_GUIDE.md`).

---

## Session Summary

**Date**: 2026-03-24
**Focus**: Fixing int32 violations in commits introduced via rebase from master (ikernel branch)

---

## Completed Fixes

### 1. ILKernelGenerator.Reduction.Arg.cs

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Arg.cs`

**Changes**:
- `ArgMaxSimdHelper<T>()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMinSimdHelper<T>()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMaxFloatNaNHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMinFloatNaNHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMaxDoubleNaNHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMinDoubleNaNHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMaxBoolHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- `ArgMinBoolHelper()` - Return type `int` → `long`, parameter `int totalSize` → `long totalSize`
- All internal `int bestIndex` → `long bestIndex`
- All internal `int i` loop counters → `long i`
- All `int vectorEnd` → `long vectorEnd`

---

### 2. ILKernelGenerator.Reduction.Axis.Simd.cs

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Axis.Simd.cs`

**Changes**:
- `int[] outputDimStridesArray` → `long[]`
- `DivideByCountTyped<T>(T value, int count)` → `long count`
- `ReduceContiguousAxis<T>(T* data, int size, ...)` → `long size`
- `ReduceContiguousAxisSimd256<T>()` - `int vectorEnd` → `long`, `int i` → `long i`, `int unrollStep/End` → `long`
- `ReduceContiguousAxisSimd128<T>()` - Same changes as above
- `ReduceContiguousAxisScalar<T>(T* data, int size, ...)` → `long size`, `int i` → `long i`
- `ReduceStridedAxis<T>(T* data, int size, int stride, ...)` → `long size, long stride`
- `ReduceStridedAxisGatherFloat()` → `long size, long stride`, added `int strideInt = (int)stride` for AVX2 gather
- `ReduceStridedAxisGatherDouble()` → `long size, long stride`, added `int strideInt = (int)stride` for AVX2 gather
- `ReduceStridedAxisScalar<T>()` → `long size, long stride`, `int i` → `long i`, `int unrollEnd` → `long`

---

### 3. ILKernelGenerator.Reduction.Axis.cs

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Axis.cs`

**Changes**:
- Line 247: `for (int i = 0; i < axisSize; i++)` → `for (long i = 0; ...)`
- Line 362: `for (int outIdx = 0; outIdx < outputSize; outIdx++)` → `for (long outIdx = 0; ...)`
- Line 382: `for (int i = 0; i < axisSize; i++)` → `for (long i = 0; ...)`

---

### 4. ILKernelGenerator.Reduction.NaN.cs

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.NaN.cs`

**Changes**:
- Fixed rebase conflict: `public sealed partial class` → `public static partial class`

---

### 5. ILKernelGenerator.Masking.cs

**Location**: `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Masking.cs`

**Changes**:
- `ConvertFlatIndicesToCoordinates(..., int[] shape)` → `long[] shape`
- `FindNonZeroStridedHelper<T>(..., int[] shape, ...)` → `long[] shape`

---

### 6. Default.Reduction.Nan.cs

**Location**: `src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.Nan.cs`

**Changes**:
- `var keepdimsShape = new int[arr.ndim]` → `new long[arr.ndim]` (2 occurrences)
- `var outputDims = new int[arr.ndim - 1]` → `new long[arr.ndim - 1]` (2 occurrences)
- `int axisSize = shape.dimensions[axis]` → `long axisSize`
- `int outputSize = result.size` → `long outputSize`
- `fixed (int* inputStrides = ...)` → `fixed (long* ...)`
- `fixed (int* inputDims = ...)` → `fixed (long* ...)`
- `fixed (int* outputStrides = ...)` → `fixed (long* ...)`
- `var ks = new int[arr.ndim]` → `new long[arr.ndim]` (2 occurrences)
- `int[] outputDimStrides` → `long[]`
- `for (int outIdx = 0; ...)` → `for (long outIdx = 0; ...)`
- `int remaining`, `int inputBaseOffset`, `int coord` → `long`
- `ReduceNanAxisScalarFloat(NDArray arr, int baseOffset, int axisSize, int axisStride, ...)` → all `long`
- `ReduceNanAxisScalarDouble(...)` → all `long`
- All `for (int i = 0; i < axisSize; i++)` → `for (long i = 0; ...)`

---

### 7. Default.NonZero.cs

**Location**: `src/NumSharp.Core/Backends/Default/Indexing/Default.NonZero.cs`

**Changes**:
- `public override int CountNonZero(NDArray nd)` → `long`
- `var outputDims = new int[nd.ndim - 1]` → `new long[...]`
- `var ks = new int[nd.ndim]` → `new long[...]`
- `private static unsafe int count_nonzero<T>(...)` → `long`, `int count` → `long count`
- `for (int i = 0; i < size; i++)` → `for (long i = 0; ...)`
- `count_nonzero_axis<T>()`:
  - `long axisSize = shape.dimensions[axis]`
  - `Span<int> outputDimStrides` → `Span<long>`
  - `int axisStride` → `long axisStride`
  - `for (int outIdx = 0; ...)` → `for (long outIdx = 0; ...)`
  - `int remaining`, `int inputBaseOffset`, `int coord` → `long`
  - `for (int i = 0; i < axisSize; i++)` → `for (long i = 0; ...)`

---

### 8. Default.BooleanMask.cs

**Location**: `src/NumSharp.Core/Backends/Default/Indexing/Default.BooleanMask.cs`

**Changes**:
- `int size = arr.size` → `long size`
- `int trueCount = ILKernelGenerator.CountTrueSimdHelper(...)` → `long trueCount`
- `int trueCount = 0` → `long trueCount = 0`
- `int destIdx`, `int srcIdx` → `long destIdx`, `long srcIdx`

---

### 9. TensorEngine.cs

**Location**: `src/NumSharp.Core/Backends/TensorEngine.cs`

**Changes**:
- `public abstract int CountNonZero(NDArray a)` → `long`
- Removed duplicate `CountNonZero(in NDArray a)` declarations (rebase conflict)

---

### 10. np.count_nonzero.cs

**Location**: `src/NumSharp.Core/APIs/np.count_nonzero.cs`

**Changes**:
- `public static int count_nonzero(NDArray a)` → `long`
- `var ks = new int[a.ndim]` → `new long[a.ndim]`

---

### 11. np.any.cs

**Location**: `src/NumSharp.Core/Logic/np.any.cs`

**Changes**:
- `int[] inputShape = nd.shape` → `long[]`
- `int[] outputShape = new int[...]` → `long[]`
- `int axisSize = inputShape[axis]` → `long`
- `int postAxisStride = 1` → `long`
- `ComputeAnyPerAxis<T>(..., int axisSize, int postAxisStride, ...)` → `long, long`
- Internal `int blockIndex`, `int inBlockIndex`, `int inputStartIndex` → `long`
- `for (int a = 0; a < axisSize; a++)` → `for (long a = 0; ...)`

---

### 12. np.all.cs

**Location**: `src/NumSharp.Core/Logic/np.all.cs`

**Changes**:
- Same pattern as np.any.cs (mirror changes)

---

### 13. np.random.rand.cs

**Location**: `src/NumSharp.Core/RandomSampling/np.random.rand.cs`

**Changes**:
- Removed duplicate `random(params int[] size)` and `random(Shape shape)` methods (rebase conflict)

---

### 14. Default.Op.Boolean.template.cs

**Location**: `src/NumSharp.Core/Operations/Elementwise/Templates/Default.Op.Boolean.template.cs`

**Changes**:
- **Deleted** - Duplicate of Default.Op.Equals.template.cs (rebase conflict)

---

## Remaining Work

### High Priority - Build Blocking (~96 errors)

#### Shape.Broadcasting.cs
**Location**: `src/NumSharp.Core/View/Shape.Broadcasting.cs`

**Issues** (lines approximate):
- Line 50, 71, 131, 191, 217, 221, 251, 265, 288, 301, 320, 331, 335, 336: `int` → `long` conversions needed
- Lines 223-224, 253-254, 267-268, 338: `int[]` → `long[]` argument mismatches
- Pattern: Methods returning/using `int[]` for dimensions need `long[]`

#### NDArray`1.cs
**Location**: `src/NumSharp.Core/Generics/NDArray`1.cs`

**Issues**:
- Line 92: Constructor signature mismatch (NPTypeCode, long, bool)
- Line 146: Constructor signature mismatch (NPTypeCode, long)
- Likely needs constructor overloads or signature updates

#### np.random.choice.cs
**Location**: `src/NumSharp.Core/RandomSampling/np.random.choice.cs`

**Issues**:
- Line 18: `int` → `long` conversion needed

---

### Medium Priority - Not Yet Audited

These files were identified in the initial scan but not yet fixed:

| File | Issue |
|------|-------|
| `np.load.cs` | `int[] shape` declarations (lines 30, 137, 158, 177, 211, 244, 283) |
| `np.save.cs` | `int[] shape` declarations (lines 55, 74, 87, 102, 120, 159, 209) |
| `NDArray.cs` | `int[] indices` parameters (convenience overloads - may keep for API compat) |
| `NdArray.ReShape.cs` | `int[] shape` parameters |
| `NDArray`1.ReShape.cs` | `int[] shape` parameters |

---

### Delegate Signature Changes (Deferred)

The `AxisReductionKernel` delegate in `ILKernelGenerator.Reduction.Axis.Simd.cs` still uses:
```csharp
(void* input, void* output, int* inputStrides, int* inputShape,
 int* outputStrides, int axis, int axisSize, int ndim, int outputSize)
```

This needs to be:
```csharp
(void* input, void* output, long* inputStrides, long* inputShape,
 long* outputStrides, int axis, long axisSize, int ndim, long outputSize)
```

**Impact**: Requires updating:
1. The delegate definition
2. All kernel generation code that creates these delegates
3. All callers that invoke these kernels

---

## Testing Status

**Build Status**: FAILING (96 errors remaining)
**Tests**: Not run (blocked by build errors)

---

## Next Steps

1. Fix `Shape.Broadcasting.cs` - Main source of cascading errors
2. Fix `NDArray`1.cs` constructor signatures
3. Fix `np.random.choice.cs`
4. Run build to verify
5. Continue with medium priority files
6. Update delegate signatures (major change)
7. Run full test suite

---

## Reference

- **Developer Guide**: `docs/INT64_DEVELOPER_GUIDE.md`
- **Migration Plan**: `docs/INT64_INDEX_MIGRATION.md`
- **Issues Tracker**: `docs/LONG_INDEXING_ISSUES.md`
