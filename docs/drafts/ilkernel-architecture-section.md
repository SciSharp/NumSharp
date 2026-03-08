# ILKernel Architecture Section for ARCHITECTURE.md

> **Draft prepared by Documentation Engineer**
> This section should be inserted after "Code Generation" in ARCHITECTURE.md

---

## IL Kernel System

The ILKernelGenerator system generates high-performance kernels at runtime using IL emission. This replaces approximately 500K lines of template-generated type-switch code with ~7K lines of IL generation logic.

### Why IL Emission?

**Problem**: NumSharp had 73+ generated files with repetitive type switches. Each binary operation had 14 dtype implementations, totaling ~500K lines of generated code that was difficult to maintain.

**Solution**: Runtime IL generation via `System.Reflection.Emit.DynamicMethod`. The JIT compiler optimizes these kernels with full SIMD support (Vector128/Vector256/Vector512).

**Benefits**:
- Single source of truth for operation logic
- SIMD vectorization without manual intrinsics per dtype
- Reduced codebase size by ~95%
- Easier to maintain and extend
- ~10-15% speedup over C# reference implementations

### Architecture Overview

```
Caller (DefaultEngine, np.*, NDArray ops)
    |
    v
Get*Kernel() or *Helper() method
    |
    v
ILKernelGenerator checks ConcurrentDictionary cache
    |
    +-- Cache hit --> Return cached delegate
    |
    +-- Cache miss --> Generate IL via DynamicMethod
                           |
                           v
                       JIT compiles to native code
                           |
                           v
                       Cache and return delegate
```

### Partial Class Structure

The kernel generator is split into focused partial classes:

| File | Responsibility |
|------|----------------|
| `ILKernelGenerator.cs` | Core infrastructure: type mapping, IL primitives, SIMD detection |
| `.Binary.cs` | Same-type binary ops (Add, Sub, Mul, Div) for contiguous arrays |
| `.MixedType.cs` | Mixed-type binary ops with type promotion |
| `.Unary.cs` | Math functions (Negate, Abs, Sqrt, Sin, Cos, Exp, Log, etc.) |
| `.Comparison.cs` | Element-wise comparisons (==, !=, <, >, <=, >=) returning bool |
| `.Reduction.cs` | Reductions (Sum, Prod, Min, Max, ArgMax, ArgMin, All, Any) |

### SIMD Support

Vector width is detected at startup:

```csharp
public static readonly int VectorBits =
    Vector512.IsHardwareAccelerated ? 512 :
    Vector256.IsHardwareAccelerated ? 256 :
    Vector128.IsHardwareAccelerated ? 128 : 0;
```

Kernel generation chooses execution path based on array layout:

| Path | Condition | Strategy |
|------|-----------|----------|
| **SimdFull** | Both operands contiguous, SIMD-capable dtype | SIMD loop + scalar tail |
| **ScalarFull** | Both contiguous, non-SIMD dtype (Decimal, etc.) | Scalar loop |
| **General** | Strided/broadcast arrays | Coordinate-based iteration |

### SIMD-Capable Types

| Supported | Not Supported |
|-----------|---------------|
| byte, short, ushort | bool |
| int, uint, long, ulong | char |
| float, double | decimal |

### Caching Strategy

Each kernel type has its own `ConcurrentDictionary`:

| Cache | Purpose |
|-------|---------|
| `_contiguousKernelCache` | Binary same-type operations |
| `_mixedTypeCache` | Binary mixed-type operations |
| `_unaryCache` | Unary operations |
| `_comparisonCache` | Comparison operations |
| `_elementReductionCache` | Reduction operations |

Cache keys encode operation type, input/output dtypes, and stride patterns.

### Key Members

**Core infrastructure** (`ILKernelGenerator.cs`):
- `Enabled` - Toggle IL generation on/off (for debugging)
- `VectorBits`, `VectorBytes` - Detected SIMD capability
- `GetVectorType()` - Returns Vector128/256/512 based on hardware
- `CanUseSimd()` - Check if dtype supports SIMD

**Reduction helpers** (`ILKernelGenerator.Reduction.cs`):
- `AllSimdHelper<T>()`, `AnySimdHelper<T>()` - Early-exit boolean reductions
- `ArgMaxSimdHelper<T>()`, `ArgMinSimdHelper<T>()` - Index-tracking reductions with NaN handling
- `NonZeroSimdHelper<T>()` - Finding non-zero indices
- `CountTrueSimdHelper()`, `CopyMaskedElementsHelper<T>()` - Boolean masking support

### Disabling IL Kernels

For debugging or compatibility testing:

```csharp
ILKernelGenerator.Enabled = false;  // Falls back to C# implementations
```

### Source Files

| File | Location |
|------|----------|
| Core | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs` |
| Binary | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Binary.cs` |
| MixedType | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.MixedType.cs` |
| Unary | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Unary.cs` |
| Comparison | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Comparison.cs` |
| Reduction | `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Reduction.cs` |

---
