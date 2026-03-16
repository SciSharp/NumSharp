# IL Kernel Migration: Eliminate NPTypeCode Switch/Case Patterns

## Summary

NumSharp contains approximately **2,795 NPTypeCode switch/case occurrences across 68 files**, resulting in ~6,500 lines of repetitive type-dispatched code. This issue tracks the migration of these patterns to IL-generated kernels, reducing code size, improving maintainability, and enabling SIMD optimization.

## Problem Statement

The current codebase uses extensive `switch (typecode) { case NPTypeCode.X: ... }` patterns to handle NumSharp's 12 supported types:

```
Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Single, Double, Decimal
```

This results in:
- **Code bloat**: 12 nearly-identical branches per operation
- **Maintenance burden**: Changes must be replicated across all type branches
- **Regen dependency**: Many files use `#if _REGEN` template generation
- **Missed SIMD opportunities**: Scalar loops where vectorization is possible

## Current State Analysis

### Occurrence Count by File

| File | NPTypeCode Cases | Category |
|------|------------------|----------|
| `Utilities/Converts.cs` | 516 | Type Conversion |
| `Unmanaged/UnmanagedMemoryBlock.Casting.cs` | 342 | Type Casting |
| `Utilities/ArrayConvert.cs` | 221 | Array Conversion |
| `Backends/NPTypeCode.cs` | 161 | Extension Methods |
| `Unmanaged/ArraySlice.cs` | 130 | Slice Operations |
| `DefaultEngine.ReductionOp.cs` | 69 | Reductions |
| `Default.ClipNDArray.cs` | 66 | Clip with NDArray |
| `Unmanaged/UnmanagedStorage.cs` | 52 | Storage Operations |
| `ILKernelGenerator.cs` | 47 | Kernel Generation |
| `ILKernelGenerator.Reduction.cs` | 46 | Reduction Kernels |
| `Casting/NdArray.Implicit.Array.cs` | 39 | Implicit Casts |
| `Unmanaged/UnmanagedMemoryBlock.cs` | 37 | Memory Block |
| `Default.NonZero.cs` | 36 | NonZero Indices |
| `DefaultEngine.BinaryOp.cs` | 36 | Binary Operations |
| `ILKernelGenerator.Unary.Math.cs` | 36 | Unary Kernels |
| `Default.Clip.cs` | 33 | Scalar Clip |
| `Utilities/Arrays.cs` | 30 | Array Utilities |
| `Utilities/NumberInfo.cs` | 28 | Number Info |
| `Default.ATan2.cs` | 27 | ATan2 |
| `Default.NDArray.cs` | 26 | NDArray Factory |
| `UnmanagedStorage.Setters.cs` | 26 | Storage Setters |
| `UnmanagedStorage.Getters.cs` | 26 | Storage Getters |
| `TypeRules.cs` | 26 | Type Rules |
| `np.all.cs` | 25 | All Reduction |
| `np.any.cs` | 25 | Any Reduction |
| `np.repeat.cs` | 24 | Repeat |
| `DefaultEngine.UnaryOp.cs` | 24 | Unary Operations |
| `DefaultEngine.CompareOp.cs` | 24 | Comparison Ops |
| Others (~40 files) | ~400 | Various |

**Total: ~2,795 occurrences across 68 files**

### Pattern Categories

#### 1. Type Casting/Conversion (Highest Impact)
```csharp
// UnmanagedMemoryBlock.Casting.cs - 144 type-pair combinations
case NPTypeCode.Boolean:
{
    var src = (bool*)source.Address;
    switch (InfoOf<TOut>.NPTypeCode)
    {
        case NPTypeCode.Int32:
            var dst = (int*)ret.Address;
            for (int i = 0; i < len; i++)
                *(dst + i) = Converts.ToInt32(*(src + i));
            break;
        // ... 11 more output types
    }
    break;
}
// ... 11 more input types
```

#### 2. Type Dispatch to Generic Methods
```csharp
// Common pattern in np.*, DefaultEngine, Selection
switch (arr.typecode)
{
    case NPTypeCode.Boolean: return DoWork<bool>(arr.MakeGeneric<bool>());
    case NPTypeCode.Byte: return DoWork<byte>(arr.MakeGeneric<byte>());
    // ... 10 more types
}
```

#### 3. Scalar Value Extraction
```csharp
// DefaultEngine.*Op.cs - nested dispatch for scalar operations
return lhsType switch
{
    NPTypeCode.Int32 => InvokeBinaryScalar(func, lhs.GetInt32(), rhs, rhsType),
    NPTypeCode.Double => InvokeBinaryScalar(func, lhs.GetDouble(), rhs, rhsType),
    // ... 10 more types
};
```

#### 4. Per-Type Loops
```csharp
// np.linspace.cs, np.repeat.cs, etc.
case NPTypeCode.Int32:
{
    var addr = (int*)ret.Address;
    for (int i = 0; i < num; i++)
        addr[i] = Converts.ToInt32(start + i * step);
    return ret;
}
```

## Migration Priority

### P0: Type Casting (Est. 4000 LOC reduction)

| File | Current State | Migration Target |
|------|---------------|------------------|
| `UnmanagedMemoryBlock.Casting.cs` | 12×12 nested switch, 291 for-loops | Single IL kernel per type-pair, SIMD widening/narrowing |
| `ArrayConvert.cs` | 12×12 nested switch, 172 for-loops | Reuse casting kernel |

**Approach:**
```csharp
// Before: 144 separate loop implementations
// After: Single IL-generated kernel
var kernel = ILKernelGenerator.GetCastKernel(srcType, dstType);
kernel(srcPtr, dstPtr, count);
```

### P1: Indexing Operations (Est. 600 LOC reduction)

| File | Current State | Migration Target |
|------|---------------|------------------|
| `NDArray.Indexing.Selection.Getter.cs` | 12-type dispatch | IL gather kernel |
| `NDArray.Indexing.Selection.Setter.cs` | 12-type dispatch | IL scatter kernel |

### P2: Math Operations (Est. 400 LOC reduction)

| File | Current State | Migration Target |
|------|---------------|------------------|
| `np.linspace.cs` | 12 per-type loops | IL sequence generation with SIMD |
| `np.repeat.cs` | 12 per-type loops | IL fill kernel with SIMD |
| `np.all.cs` axis path | 12-type dispatch | IL axis reduction with early-exit |
| `np.any.cs` axis path | 12-type dispatch | IL axis reduction with early-exit |

### P3: Reduction Fallbacks (Est. 200 LOC reduction)

| File | Current State | Migration Target |
|------|---------------|------------------|
| `Default.Reduction.CumAdd.cs` | 10-type fallback switch | IL fallback kernel |
| `Default.Reduction.CumMul.cs` | 10-type fallback switch | IL fallback kernel |

### P4: Dispatch Cleanup (Est. 500 LOC reduction)

Files that already have IL kernels but retain verbose type dispatch:

| File | Cleanup Needed |
|------|----------------|
| `Default.Clip.cs` | 3 × 11-type switches → single dispatch |
| `Default.ClipNDArray.cs` | 6 × 11-type switches → single dispatch |
| `Default.Shift.cs` | 2 × 7-type switches → single dispatch |
| `DefaultEngine.BinaryOp.cs` | Scalar dispatch chains |
| `DefaultEngine.UnaryOp.cs` | Scalar dispatch chains |
| `DefaultEngine.CompareOp.cs` | Scalar dispatch chains |

## Files to Skip (Already Optimized or Not Applicable)

| File | Reason |
|------|--------|
| `np.random.shuffle.cs` | Random access patterns defeat SIMD |
| `np.random.randint.cs` | RNG is bottleneck, not type dispatch |
| `Default.NDArray.cs` | Factory/allocation, not compute-bound |
| `Default.ATan2.cs` | Already uses MixedTypeKernel |
| `Default.Modf.cs` | Only 3 float types, already optimized |
| `NPTypeCode.cs` | Extension methods, not loops |
| `Converts.cs` | Low-level converters called from IL |
| `NumberInfo.cs` | Metadata, not compute |
| `MultiIterator.cs` | Iterator infrastructure, type dispatch acceptable |
| `NDIteratorExtensions.cs` | Iterator infrastructure, type dispatch acceptable |

## Implementation Strategy

### Phase 1: Infrastructure
- [ ] Create `ILKernelGenerator.Cast.cs` for type-pair casting
- [ ] Add SIMD widening/narrowing helpers for compatible types
- [ ] Establish caching pattern for cast kernels

### Phase 2: High-Impact Migration
- [ ] Migrate `UnmanagedMemoryBlock.Casting.cs` to IL kernels
- [ ] Migrate `ArrayConvert.cs` to reuse cast kernels
- [ ] Migrate indexing Selection Getter/Setter to IL gather/scatter

### Phase 3: Math Operations
- [ ] Add `ILKernelGenerator.Sequence.cs` for linspace/arange
- [ ] Add `ILKernelGenerator.Fill.cs` for repeat
- [ ] Extend axis reduction kernels for all/any with early-exit

### Phase 4: Cleanup
- [ ] Consolidate dispatch in existing IL-based operations
- [ ] Remove Regen templates for migrated code
- [ ] Update documentation

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| NPTypeCode switch cases | ~2,795 | <500 |
| Lines of type-dispatch code | ~6,500 | ~1,000 |
| Regen template files | ~20 | ~5 |
| SIMD coverage for casting | 0% | 80%+ |

## Related Issues

- Generic Math Migration (tracked in `docs/GENERIC_MATH_DESIGN.md`)
- ILKernelGenerator consolidation

## Files Inventory

### Already Using IL Kernels (Cleanup Only)
```
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Binary.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.MixedType.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Unary.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Comparison.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.Axis.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Scan.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Shift.cs
src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Clip.cs
src/NumSharp.Core/Backends/Default/Math/DefaultEngine.BinaryOp.cs
src/NumSharp.Core/Backends/Default/Math/DefaultEngine.UnaryOp.cs
src/NumSharp.Core/Backends/Default/Math/DefaultEngine.CompareOp.cs
src/NumSharp.Core/Backends/Default/Math/DefaultEngine.ReductionOp.cs
src/NumSharp.Core/Backends/Default/Math/Default.Clip.cs
src/NumSharp.Core/Backends/Default/Math/Default.ClipNDArray.cs
src/NumSharp.Core/Backends/Default/Math/Default.Shift.cs
src/NumSharp.Core/Backends/Default/Math/Default.Modf.cs
```

### Migration Candidates (Priority Order)
```
# P0 - Type Casting
src/NumSharp.Core/Backends/Unmanaged/UnmanagedMemoryBlock.Casting.cs
src/NumSharp.Core/Utilities/ArrayConvert.cs

# P1 - Indexing
src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Getter.cs
src/NumSharp.Core/Selection/NDArray.Indexing.Selection.Setter.cs

# P2 - Math Operations
src/NumSharp.Core/Creation/np.linspace.cs
src/NumSharp.Core/Manipulation/np.repeat.cs
src/NumSharp.Core/Logic/np.all.cs
src/NumSharp.Core/Logic/np.any.cs
src/NumSharp.Core/Backends/Default/Indexing/Default.NonZero.cs
src/NumSharp.Core/Manipulation/NDArray.unique.cs

# P3 - Reduction Fallbacks
src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.CumAdd.cs
src/NumSharp.Core/Backends/Default/Math/Reduction/Default.Reduction.CumMul.cs
```

### Skip (Low Value or N/A)
```
src/NumSharp.Core/RandomSampling/np.random.shuffle.cs
src/NumSharp.Core/RandomSampling/np.random.randint.cs
src/NumSharp.Core/Backends/Default/ArrayManipulation/Default.NDArray.cs
src/NumSharp.Core/Backends/Default/Math/Default.ATan2.cs
src/NumSharp.Core/Backends/NPTypeCode.cs
src/NumSharp.Core/Utilities/Converts.cs
src/NumSharp.Core/Utilities/NumberInfo.cs
src/NumSharp.Core/Backends/Iterators/MultiIterator.cs
src/NumSharp.Core/Backends/Iterators/NDIteratorExtensions.cs
```

## Notes

- The Generic Math migration (separate effort) will eventually replace ILKernelGenerator with `INumber<T>` constraints
- IL migration is still valuable as it:
  - Reduces code size immediately
  - Establishes patterns that Generic Math will follow
  - Enables SIMD where Generic Math cannot (type casting)
- Some operations (type casting) will always need IL/intrinsics even after Generic Math migration
