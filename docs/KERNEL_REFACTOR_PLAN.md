# ILKernel Abstraction Refactor Plan

> **STATUS: COMPLETED** (2026-03-07)
>
> All phases completed successfully:
> - Phase 1: KernelOp.cs, KernelKey.cs, KernelSignatures.cs created
> - Phase 2: IKernelProvider.cs interface and TypeRules.cs created
> - Phase 3: ILKernelGenerator refactored to implement IKernelProvider
> - Phase 4: Integration validated (2703 tests pass, 0 failures)
>
> See DESIGN.md section 1.9 "Kernel Abstraction Layer" for documentation.

**Goal:** Extract shared kernel infrastructure from ILKernelGenerator while maintaining zero performance regression.

**Scope:** IL backend only (no CUDA/Vulkan in this phase)

**Estimated effort:** 2-3 days

---

## Phase 1: Extract Shared Definitions (Day 1 Morning)

### Task 1.1: Create `KernelOp.cs`

**File:** `src/NumSharp.Core/Backends/Kernels/KernelOp.cs`

Extract all operation enums into a single shared file:

```csharp
namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Binary operations supported by kernel providers.
    /// </summary>
    public enum BinaryOp
    {
        // Arithmetic
        Add, Subtract, Multiply, Divide, Mod,
        // Bitwise
        BitwiseAnd, BitwiseOr, BitwiseXor,
        // Future
        Power, FloorDivide, LeftShift, RightShift
    }

    /// <summary>
    /// Unary operations supported by kernel providers.
    /// </summary>
    public enum UnaryOp
    {
        // Basic
        Negate, Abs, Sign,
        // Rounding
        Floor, Ceil, Round, Truncate,
        // Exponential
        Exp, Exp2, Expm1, Log, Log2, Log10, Log1p,
        // Trigonometric
        Sin, Cos, Tan, Asin, Acos, Atan,
        // Other
        Sqrt,
        // Future
        Reciprocal, Square, Cbrt, Deg2Rad, Rad2Deg, BitwiseNot
    }

    /// <summary>
    /// Reduction operations supported by kernel providers.
    /// </summary>
    public enum ReductionOp
    {
        Sum, Prod, Min, Max, Mean,
        ArgMax, ArgMin,
        All, Any,
        // Future
        Std, Var, NanSum, NanProd, NanMin, NanMax
    }

    /// <summary>
    /// Comparison operations supported by kernel providers.
    /// </summary>
    public enum ComparisonOp
    {
        Equal, NotEqual,
        Less, LessEqual,
        Greater, GreaterEqual
    }

    /// <summary>
    /// Execution path based on array memory layout.
    /// </summary>
    public enum ExecutionPath
    {
        /// <summary>Both operands contiguous, SIMD-capable type.</summary>
        SimdFull,
        /// <summary>Both operands contiguous, non-SIMD type (Decimal, etc.).</summary>
        ScalarFull,
        /// <summary>One or both operands strided/broadcast.</summary>
        General
    }
}
```

**Actions:**
1. Create new file with enums
2. Update ILKernelGenerator.cs to remove duplicate enum (if any)
3. Verify all partial files use the shared enums
4. Run tests to confirm no regression

---

### Task 1.2: Create `KernelKey.cs`

**File:** `src/NumSharp.Core/Backends/Kernels/KernelKey.cs`

Unified cache key structures:

```csharp
namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Cache key for binary kernels.
    /// </summary>
    public readonly struct BinaryKernelKey : IEquatable<BinaryKernelKey>
    {
        public readonly BinaryOp Op;
        public readonly NPTypeCode LhsType;
        public readonly NPTypeCode RhsType;
        public readonly NPTypeCode OutputType;
        public readonly ExecutionPath Path;

        public BinaryKernelKey(BinaryOp op, NPTypeCode lhs, NPTypeCode rhs, NPTypeCode output, ExecutionPath path)
        {
            Op = op; LhsType = lhs; RhsType = rhs; OutputType = output; Path = path;
        }

        public bool Equals(BinaryKernelKey other) =>
            Op == other.Op && LhsType == other.LhsType && RhsType == other.RhsType &&
            OutputType == other.OutputType && Path == other.Path;

        public override int GetHashCode() =>
            HashCode.Combine(Op, LhsType, RhsType, OutputType, Path);
    }

    /// <summary>
    /// Cache key for unary kernels.
    /// </summary>
    public readonly struct UnaryKernelKey : IEquatable<UnaryKernelKey>
    {
        public readonly UnaryOp Op;
        public readonly NPTypeCode InputType;
        public readonly NPTypeCode OutputType;
        public readonly ExecutionPath Path;

        // Constructor, Equals, GetHashCode...
    }

    /// <summary>
    /// Cache key for reduction kernels.
    /// </summary>
    public readonly struct ReductionKernelKey : IEquatable<ReductionKernelKey>
    {
        public readonly ReductionOp Op;
        public readonly NPTypeCode InputType;
        public readonly NPTypeCode OutputType;
        public readonly bool HasAxis;

        // Constructor, Equals, GetHashCode...
    }

    /// <summary>
    /// Cache key for comparison kernels.
    /// </summary>
    public readonly struct ComparisonKernelKey : IEquatable<ComparisonKernelKey>
    {
        public readonly ComparisonOp Op;
        public readonly NPTypeCode LhsType;
        public readonly NPTypeCode RhsType;
        public readonly ExecutionPath Path;

        // Constructor, Equals, GetHashCode...
    }
}
```

**Actions:**
1. Create new file with key structs
2. Replace tuple keys in each partial file with struct keys
3. Verify hash distribution is good
4. Run benchmarks to confirm no cache lookup regression

---

### Task 1.3: Create `KernelSignatures.cs`

**File:** `src/NumSharp.Core/Backends/Kernels/KernelSignatures.cs`

Consolidate all delegate types:

```csharp
namespace NumSharp.Backends.Kernels
{
    // Binary kernels
    public unsafe delegate void ContiguousKernel<T>(T* lhs, T* rhs, T* result, int count) where T : unmanaged;
    public unsafe delegate void BinaryKernel<T>(T* lhs, int lhsOffset, int lhsStride,
                                                 T* rhs, int rhsOffset, int rhsStride,
                                                 T* result, int resultOffset, int resultStride,
                                                 int count) where T : unmanaged;
    public unsafe delegate void MixedTypeKernel(void* lhs, void* rhs, void* result,
                                                 int count, int lhsStride, int rhsStride, int resultStride);

    // Unary kernels
    public unsafe delegate void UnaryKernel<TIn, TOut>(TIn* input, TOut* output, int count)
        where TIn : unmanaged where TOut : unmanaged;
    public unsafe delegate void UnaryKernelStrided<TIn, TOut>(TIn* input, int inOffset, int inStride,
                                                               TOut* output, int outOffset, int outStride,
                                                               int count) where TIn : unmanaged where TOut : unmanaged;

    // Reduction kernels
    public unsafe delegate T ElementReductionKernel<T>(T* input, int count) where T : unmanaged;
    public unsafe delegate void AxisReductionKernel<T>(T* input, T* output, int outerSize, int axisSize, int innerSize)
        where T : unmanaged;

    // Comparison kernels
    public unsafe delegate void ComparisonKernel(void* lhs, void* rhs, bool* result,
                                                  int count, int lhsStride, int rhsStride);

    // Scalar delegates (for broadcasting)
    public delegate TOut UnaryScalar<TIn, TOut>(TIn value);
    public delegate TOut BinaryScalar<TLhs, TRhs, TOut>(TLhs lhs, TRhs rhs);
    public delegate bool ComparisonScalar<TLhs, TRhs>(TLhs lhs, TRhs rhs);
}
```

**Actions:**
1. Create new file with all delegate types
2. Remove duplicate delegate definitions from partial files
3. Update all references
4. Run tests

---

## Phase 2: Create Provider Interface (Day 1 Afternoon)

### Task 2.1: Create `IKernelProvider.cs`

**File:** `src/NumSharp.Core/Backends/Kernels/IKernelProvider.cs`

```csharp
namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Interface for kernel providers (IL, CUDA, Vulkan, etc.)
    /// </summary>
    public interface IKernelProvider
    {
        /// <summary>Provider name for diagnostics.</summary>
        string Name { get; }

        /// <summary>Whether this provider is enabled.</summary>
        bool Enabled { get; set; }

        /// <summary>SIMD vector width in bits (128, 256, 512, or 0 for none).</summary>
        int VectorBits { get; }

        /// <summary>Check if type supports SIMD on this provider.</summary>
        bool CanUseSimd(NPTypeCode type);

        // Binary operations
        ContiguousKernel<T>? GetContiguousKernel<T>(BinaryOp op) where T : unmanaged;
        MixedTypeKernel? GetMixedTypeKernel(BinaryKernelKey key);

        // Unary operations
        UnaryKernel<T, T>? GetUnaryKernel<T>(UnaryOp op) where T : unmanaged;
        UnaryScalar<TIn, TOut>? GetUnaryScalar<TIn, TOut>(UnaryOp op) where TIn : unmanaged where TOut : unmanaged;

        // Reductions
        ElementReductionKernel<T>? GetElementReductionKernel<T>(ReductionOp op) where T : unmanaged;

        // Comparisons
        ComparisonKernel? GetComparisonKernel(ComparisonKernelKey key);
        ComparisonScalar<TLhs, TRhs>? GetComparisonScalar<TLhs, TRhs>(ComparisonOp op)
            where TLhs : unmanaged where TRhs : unmanaged;

        // Cache management
        void Clear();
        int CacheCount { get; }
    }
}
```

**Actions:**
1. Create interface file
2. Review method signatures match current ILKernelGenerator public API
3. Document each method

---

### Task 2.2: Create `TypeRules.cs`

**File:** `src/NumSharp.Core/Backends/Kernels/TypeRules.cs`

Extract type-related logic shared by all providers:

```csharp
namespace NumSharp.Backends.Kernels
{
    /// <summary>
    /// Shared type rules for kernel providers.
    /// </summary>
    public static class TypeRules
    {
        /// <summary>Get size in bytes for NPTypeCode.</summary>
        public static int GetTypeSize(NPTypeCode type) => type switch
        {
            NPTypeCode.Boolean or NPTypeCode.Byte => 1,
            NPTypeCode.Int16 or NPTypeCode.UInt16 or NPTypeCode.Char => 2,
            NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Single => 4,
            NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Double => 8,
            NPTypeCode.Decimal => 16,
            _ => throw new NotSupportedException($"Type {type} not supported")
        };

        /// <summary>Get CLR Type for NPTypeCode.</summary>
        public static Type GetClrType(NPTypeCode type) => type switch { ... };

        /// <summary>Get accumulating type for reductions (NEP50).</summary>
        public static NPTypeCode GetAccumulatingType(NPTypeCode type) => type switch
        {
            NPTypeCode.Int32 or NPTypeCode.Int16 or NPTypeCode.Byte => NPTypeCode.Int64,
            NPTypeCode.UInt32 or NPTypeCode.UInt16 => NPTypeCode.UInt64,
            _ => type  // Float/Double/Decimal preserve type
        };

        /// <summary>Check if type is unsigned integer.</summary>
        public static bool IsUnsigned(NPTypeCode type) =>
            type is NPTypeCode.Byte or NPTypeCode.UInt16 or NPTypeCode.UInt32 or NPTypeCode.UInt64;

        /// <summary>Check if type is floating point.</summary>
        public static bool IsFloatingPoint(NPTypeCode type) =>
            type is NPTypeCode.Single or NPTypeCode.Double or NPTypeCode.Decimal;

        /// <summary>Get elements per vector for type at given vector width.</summary>
        public static int GetVectorCount(NPTypeCode type, int vectorBits) =>
            vectorBits == 0 ? 1 : (vectorBits / 8) / GetTypeSize(type);
    }
}
```

**Actions:**
1. Create file with type rules
2. Remove duplicate implementations from ILKernelGenerator.cs
3. Update all callers to use `TypeRules.*`
4. Run tests

---

## Phase 3: Refactor ILKernelGenerator (Day 2)

### Task 3.1: Implement IKernelProvider

**File:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs`

```csharp
public static partial class ILKernelGenerator : IKernelProvider
{
    // Singleton instance
    public static readonly ILKernelGenerator Instance = new();

    // IKernelProvider implementation
    public string Name => "IL";
    public bool Enabled { get; set; } = true;
    public int VectorBits { get; } = DetectVectorBits();

    public bool CanUseSimd(NPTypeCode type) =>
        VectorBits > 0 && type is not (NPTypeCode.Boolean or NPTypeCode.Char or NPTypeCode.Decimal);

    // Existing methods become interface implementations...
}
```

**Wait:** ILKernelGenerator is currently `static`. Options:
1. Make it a singleton implementing IKernelProvider
2. Keep static methods, create thin wrapper class

**Recommendation:** Option 1 (singleton) - cleaner, matches interface pattern.

**Actions:**
1. Change from `static class` to `sealed class` with static `Instance`
2. Implement `IKernelProvider` interface
3. Keep backward compatibility with static facade methods
4. Run tests

---

### Task 3.2: Update Partial Files

Update each partial file to use shared types:

| File | Changes |
|------|---------|
| `.Binary.cs` | Use `BinaryKernelKey`, remove duplicate delegate types |
| `.MixedType.cs` | Use `BinaryKernelKey`, `TypeRules.*` |
| `.Unary.cs` | Use `UnaryKernelKey`, `TypeRules.*` |
| `.Comparison.cs` | Use `ComparisonKernelKey`, `TypeRules.*` |
| `.Reduction.cs` | Use `ReductionKernelKey`, `TypeRules.*` |

**Actions per file:**
1. Replace tuple cache keys with struct keys
2. Replace type helper methods with `TypeRules.*` calls
3. Remove duplicate delegate definitions
4. Run tests after each file

---

### Task 3.3: Backward Compatibility Facade

**File:** `src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.cs`

Add static facade for existing callers:

```csharp
public static partial class ILKernelGenerator
{
    // Backward compatibility - delegates to Instance
    public static ContiguousKernel<T>? GetContiguousKernel<T>(BinaryOp op) where T : unmanaged
        => Instance.GetContiguousKernel<T>(op);

    public static void Clear() => Instance.Clear();

    public static int CachedCount => Instance.CacheCount;

    // ... other static facades
}
```

**Actions:**
1. Add static wrapper methods
2. Verify DefaultEngine calls still work
3. Run full test suite

---

## Phase 4: Integration & Validation (Day 3)

### Task 4.1: Update DefaultEngine

**Files:** `src/NumSharp.Core/Backends/Default/*.cs`

Update DefaultEngine to use `IKernelProvider` interface:

```csharp
public partial class DefaultEngine : TensorEngine
{
    // Kernel provider (currently always IL)
    private readonly IKernelProvider _kernelProvider = ILKernelGenerator.Instance;

    // Example usage
    protected override NDArray Add(NDArray lhs, NDArray rhs)
    {
        var kernel = _kernelProvider.GetContiguousKernel<T>(BinaryOp.Add);
        // ...
    }
}
```

**Actions:**
1. Add `_kernelProvider` field
2. Update methods that use kernels
3. Run tests

---

### Task 4.2: Benchmark Validation

Run benchmarks to confirm zero regression:

```bash
cd test/NumSharp.Benchmark
dotnet run -c Release -- --filter "*Binary*" --baseline
dotnet run -c Release -- --filter "*Reduction*" --baseline
```

**Acceptance criteria:**
- No kernel operation regresses more than 1%
- Cache lookup time unchanged

---

### Task 4.3: Documentation

Update docs:
1. `CLAUDE.md` - ILKernelGenerator section
2. `DESIGN.md` - Add kernel abstraction architecture
3. `ARCHITECTURE.md` - Update with new file structure

---

## File Summary

### New Files (6)

| File | Purpose |
|------|---------|
| `Kernels/KernelOp.cs` | All operation enums |
| `Kernels/KernelKey.cs` | Cache key structs |
| `Kernels/KernelSignatures.cs` | Delegate types |
| `Kernels/TypeRules.cs` | Shared type utilities |
| `Kernels/IKernelProvider.cs` | Provider interface |
| `docs/KERNEL_REFACTOR_PLAN.md` | This plan |

### Modified Files (7)

| File | Changes |
|------|---------|
| `ILKernelGenerator.cs` | Implement interface, use shared types |
| `ILKernelGenerator.Binary.cs` | Use shared keys/delegates |
| `ILKernelGenerator.MixedType.cs` | Use shared keys/delegates |
| `ILKernelGenerator.Unary.cs` | Use shared keys/delegates |
| `ILKernelGenerator.Comparison.cs` | Use shared keys/delegates |
| `ILKernelGenerator.Reduction.cs` | Use shared keys/delegates |
| `DefaultEngine.*.cs` | Use IKernelProvider |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Performance regression | Benchmark each phase, revert if >1% |
| Breaking changes | Keep static facades for all public methods |
| Cache key hash collision | Test with diverse operation combinations |
| Interface overhead | JIT devirtualizes singleton calls |

---

## Success Criteria

1. All existing tests pass
2. No benchmark regression >1%
3. `IKernelProvider` interface documented
4. Future backend (CUDA) can implement same interface
5. Code reduction: ~500 lines of duplicate code removed

---

## Next Steps After Completion

1. Implement missing operations (Power, FloorDivide, Clip, Modf)
2. Add axis reduction SIMD path
3. Design CUDA kernel provider (separate project)
