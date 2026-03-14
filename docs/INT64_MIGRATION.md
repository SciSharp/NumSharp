# Int64 Indexing Migration - Technical Design Document

## Overview

NumSharp is migrating from `int` (32-bit) to `long` (64-bit) for all array indexing, strides, offsets, and dimension calculations. This enables arrays larger than 2 billion elements.

**GitHub Issue:** https://github.com/SciSharp/NumSharp/issues/584
**Branch:** `longindexing`

## Definition of Done

The migration is complete when:

1. **All core types use `long`:**
   - `Shape.dimensions` → `long[]` ✅
   - `Shape.strides` → `long[]` ✅
   - `Shape.offset` → `long` ✅
   - `Shape.size` → `long` ✅
   - `Slice.Start/Stop/Step` → `long` ❌ **CRITICAL: Still `int`**
   - `SliceDef.Start/Step/Count` → `long` ❌ **CRITICAL: Still `int`**

2. **All calculations use `long` arithmetic:**
   - No `(int)` casts truncating index/stride calculations
   - All index loops use `long` loop variables
   - All stride multiplications use `long`

3. **Overload pattern for backwards compatibility:**
   - `int[]` overloads call `long[]` implementations
   - `int` scalar overloads call `long` implementations
   - No downgrade from `long` to `int` in method chains

4. **Tests pass:**
   - All existing tests continue passing
   - New tests with large index values (>2B) added where practical

---

## Critical Issues Found

### Issue 1: Slice.cs Still Uses `int`

**Location:** `src/NumSharp.Core/View/Slice.cs`

```csharp
// Lines 93-95 - MUST be long
public int? Start;
public int? Stop;
public int Step;

// Line 120 - Constructor MUST accept long
public Slice(int? start = null, int? stop = null, int step = 1)

// Lines 376-378 - SliceDef MUST use long
public int Start;
public int Step;
public int Count;
```

**Impact:** Slicing is fundamental to all array operations. `int` limits prevent slicing arrays with >2B elements.

### Issue 2: Files with `int[]` for dimensions/strides/shape

Files that still declare `int[]` for array-related parameters:

| File | Issue |
|------|-------|
| `np.full.cs` | Parameters |
| `np.ones.cs` | Parameters |
| `np.zeros.cs` | Parameters |
| `NDArray.itemset.cs` | Parameters |
| `ILKernelGenerator.Masking.cs` | Local arrays |
| `IKernelProvider.cs` | Interface signatures |
| `NdArray.ReShape.cs` | Parameters |
| `ArrayConvert.cs` | Conversion utilities |
| `np.vstack.cs` | Shape handling |
| `np.reshape.cs` | Parameters |
| `NdArray`1.ReShape.cs` | Parameters |
| `np.empty.cs` | Parameters |
| `np.save.cs` | File format |
| `np.load.cs` | File format |

### Issue 3: Suspicious `(int)` Casts (242 total)

Files with highest cast counts that need review:

| File | Count | Priority |
|------|-------|----------|
| `Shape.cs` | 15 | HIGH - core type |
| `ILKernelGenerator.Clip.cs` | 15 | HIGH - kernel |
| `ILKernelGenerator.MatMul.cs` | 13 | HIGH - kernel |
| `ILKernelGenerator.Reduction.cs` | 12 | HIGH - kernel |
| `ILKernelGenerator.Scan.cs` | 11 | HIGH - kernel |
| `NdArray.Convolve.cs` | 10 | MEDIUM |
| `Converts.Native.cs` | 10 | LOW - type conversion |
| `ILKernelGenerator.Comparison.cs` | 9 | HIGH - kernel |
| `Operator.cs` | 8 | LOW - math ops |
| `ILKernelGenerator.Reduction.Axis.Simd.cs` | 8 | HIGH - kernel |

---

## Review Areas

### Area 1: Core Types (CRITICAL)
- `View/Shape.cs` - Core shape struct
- `View/Slice.cs` - Slice parsing and SliceDef
- `View/Shape.Reshaping.cs` - Reshape operations
- `View/Shape.Unmanaged.cs` - Unsafe operations

### Area 2: Storage Layer (HIGH)
- `Backends/Unmanaged/ArraySlice.cs`
- `Backends/Unmanaged/ArraySlice\`1.cs`
- `Backends/Unmanaged/UnmanagedStorage.cs`
- `Backends/Unmanaged/UnmanagedStorage.Getters.cs`
- `Backends/Unmanaged/UnmanagedStorage.Setters.cs`
- `Backends/Unmanaged/UnmanagedStorage.Cloning.cs`

### Area 3: NDArray Core (HIGH)
- `Backends/NDArray.cs`
- `Backends/NDArray.Unmanaged.cs`
- `Backends/NDArray.String.cs`
- `Backends/TensorEngine.cs`
- `Generics/NDArray\`1.cs`

### Area 4: IL Kernels (HIGH)
All files in `Backends/Kernels/`:
- `ILKernelGenerator.*.cs` - All kernel generators
- `KernelSignatures.cs` - Delegate signatures
- `BinaryKernel.cs`, `ReductionKernel.cs` - Kernel wrappers

### Area 5: DefaultEngine Operations (MEDIUM)
- `Backends/Default/Math/*.cs`
- `Backends/Default/ArrayManipulation/*.cs`
- `Backends/Default/Indexing/*.cs`

### Area 6: Iterators & Incrementors (MEDIUM)
- `Backends/Iterators/NDIterator.cs`
- `Backends/Iterators/MultiIterator.cs`
- `Utilities/Incrementors/*.cs`

### Area 7: Creation APIs (MEDIUM)
- `Creation/np.*.cs` - All creation functions

### Area 8: Manipulation & Selection (MEDIUM)
- `Manipulation/*.cs`
- `Selection/*.cs`

---

## Review Checklist

For each file, verify:

- [ ] No `int[]` parameters for dimensions/strides/shape (use `long[]`)
- [ ] No `int` parameters for indices/offsets (use `long`)
- [ ] No `(int)` casts that truncate index calculations
- [ ] If `int[]` overloads exist, they delegate to `long[]` versions
- [ ] Loop variables for array iteration use `long`
- [ ] Stride multiplication uses `long` arithmetic
- [ ] Method return types are `long` for sizes/indices

---

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-03-15 | Initial design document created | coordinator |
