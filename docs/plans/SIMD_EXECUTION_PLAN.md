# NumSharp 100% SIMD Execution Plan

## Executive Summary

Transform NumSharp from partial SIMD coverage to comprehensive SIMD optimization across all operations, targeting **2-5x performance improvement** for CPU-bound workloads.

**Current State**: SIMD used in some operations but with suboptimal patterns (serial dependency chains)
**Target State**: All numeric operations use optimized SIMD with 4x unrolling + tree reduction

---

## Phase 1: Foundation (Week 1-2)

### 1.1 Create Unified SIMD Infrastructure

**File**: `src/NumSharp.Core/Backends/Kernels/SimdOps.cs`

```csharp
public static class SimdOps
{
    // Constants
    public const int UNROLL_FACTOR = 4;
    public const int PARALLEL_THRESHOLD = 100_000;

    // Core reduction pattern - reusable across all reduction ops
    public static unsafe T ReduceContiguous<T, TOp>(T* data, int size)
        where T : unmanaged
        where TOp : struct, IReduceOp<T>

    // Core binary pattern - reusable across all binary ops
    public static unsafe void BinaryContiguous<T, TOp>(T* lhs, T* rhs, T* result, int size)
        where T : unmanaged
        where TOp : struct, IBinaryOp<T>

    // Core unary pattern
    public static unsafe void UnaryContiguous<T, TOp>(T* src, T* dst, int size)
        where T : unmanaged
        where TOp : struct, IUnaryOp<T>
}
```

**Interfaces for operation types**:

```csharp
public interface IReduceOp<T> where T : unmanaged
{
    static abstract T Identity { get; }
    static abstract T Combine(T a, T b);
    static abstract Vector256<T> Combine(Vector256<T> a, Vector256<T> b);
}

public interface IBinaryOp<T> where T : unmanaged
{
    static abstract T Apply(T a, T b);
    static abstract Vector256<T> Apply(Vector256<T> a, Vector256<T> b);
}

public interface IUnaryOp<T> where T : unmanaged
{
    static abstract T Apply(T a);
    static abstract Vector256<T> Apply(Vector256<T> a);
}
```

### 1.2 Implement Optimized Reduction Template

Replace current `ILKernelGenerator.Reduction.cs` patterns with 4x unrolled version:

```csharp
[MethodImpl(MethodImplOptions.AggressiveOptimization)]
public static unsafe T ReduceContiguous<T, TOp>(T* data, int size)
    where T : unmanaged
    where TOp : struct, IReduceOp<T>
{
    if (size == 0) return TOp.Identity;
    if (size < Vector256<T>.Count * 4) return ReduceScalar<T, TOp>(data, size);

    int vectorCount = Vector256<T>.Count;
    int unrollStep = vectorCount * 4;
    int unrollEnd = size - unrollStep;

    // 4 independent accumulators
    var acc0 = Vector256.Create(TOp.Identity);
    var acc1 = Vector256.Create(TOp.Identity);
    var acc2 = Vector256.Create(TOp.Identity);
    var acc3 = Vector256.Create(TOp.Identity);

    int i = 0;
    for (; i <= unrollEnd; i += unrollStep)
    {
        var v0 = Vector256.Load(data + i);
        var v1 = Vector256.Load(data + i + vectorCount);
        var v2 = Vector256.Load(data + i + vectorCount * 2);
        var v3 = Vector256.Load(data + i + vectorCount * 3);

        acc0 = TOp.Combine(acc0, v0);
        acc1 = TOp.Combine(acc1, v1);
        acc2 = TOp.Combine(acc2, v2);
        acc3 = TOp.Combine(acc3, v3);
    }

    // Tree reduction
    var sum01 = TOp.Combine(acc0, acc1);
    var sum23 = TOp.Combine(acc2, acc3);
    var sumAll = TOp.Combine(sum01, sum23);

    // Horizontal reduction + scalar tail
    T result = HorizontalReduce<T, TOp>(sumAll);
    for (; i < size; i++)
        result = TOp.Combine(result, data[i]);

    return result;
}
```

### 1.3 Deliverables

| Task | File | Estimated Hours |
|------|------|-----------------|
| Create `IReduceOp<T>` interface | `SimdInterfaces.cs` | 2 |
| Create `IBinaryOp<T>` interface | `SimdInterfaces.cs` | 2 |
| Create `IUnaryOp<T>` interface | `SimdInterfaces.cs` | 2 |
| Implement `ReduceContiguous<T,TOp>` | `SimdOps.Reduction.cs` | 4 |
| Implement `BinaryContiguous<T,TOp>` | `SimdOps.Binary.cs` | 4 |
| Implement `UnaryContiguous<T,TOp>` | `SimdOps.Unary.cs` | 4 |
| Unit tests for templates | `SimdOpsTests.cs` | 4 |

**Total Phase 1**: ~22 hours

---

## Phase 2: Reduction Operations (Week 2-3)

### 2.1 Operations to Migrate

| Operation | Current Location | Priority | Complexity |
|-----------|------------------|----------|------------|
| Sum | `ILKernelGenerator.Reduction.cs` | P0 | Low |
| Prod | `ILKernelGenerator.Reduction.cs` | P0 | Low |
| Min | `ILKernelGenerator.Reduction.cs` | P0 | Low |
| Max | `ILKernelGenerator.Reduction.cs` | P0 | Low |
| Mean | `ILKernelGenerator.Reduction.cs` | P0 | Low |
| ArgMax | `ILKernelGenerator.Reduction.cs` | P1 | Medium |
| ArgMin | `ILKernelGenerator.Reduction.cs` | P1 | Medium |
| All | `ILKernelGenerator.Reduction.cs` | P1 | Low |
| Any | `ILKernelGenerator.Reduction.cs` | P1 | Low |
| Std | `Default.Reduction.Std.cs` | P2 | Medium |
| Var | `Default.Reduction.Var.cs` | P2 | Medium |

### 2.2 Implementation Pattern

For each reduction op, create a struct implementing `IReduceOp<T>`:

```csharp
public readonly struct SumOp<T> : IReduceOp<T> where T : unmanaged, INumber<T>
{
    public static T Identity => T.Zero;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Combine(T a, T b) => a + b;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<T> Combine(Vector256<T> a, Vector256<T> b) => a + b;
}

public readonly struct MaxOp<T> : IReduceOp<T> where T : unmanaged, INumber<T>, IMinMaxValue<T>
{
    public static T Identity => T.MinValue;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static T Combine(T a, T b) => T.Max(a, b);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<T> Combine(Vector256<T> a, Vector256<T> b) => Vector256.Max(a, b);
}
```

### 2.3 Axis Reduction Optimization

Current axis reduction is slow due to coordinate calculation per element. Fix:

```csharp
// BEFORE: O(N * ndim) coordinate calculations
for (int outIdx = 0; outIdx < outputSize; outIdx++)
{
    // Expensive coordinate calculation every iteration
    for (int d = 0; d < outputNdim; d++)
    {
        int coord = remaining / outputDimStrides[d];  // Division!
        remaining = remaining % outputDimStrides[d];  // Modulo!
    }
}

// AFTER: O(N) with stride-based iteration
// Precompute iteration order, use pointer arithmetic
int* coords = stackalloc int[ndim];
for (int outIdx = 0; outIdx < outputSize; outIdx++)
{
    T* axisPtr = input + inputBaseOffset;
    T result = ReduceContiguous<T, TOp>(axisPtr, axisSize);  // SIMD inner loop
    output[outputOffset] = result;

    // Increment coordinates (no division)
    IncrementCoordinates(coords, shape, strides, &inputBaseOffset, &outputOffset);
}
```

### 2.4 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| SumOp, ProdOp, MinOp, MaxOp structs | 4 |
| MeanOp (Sum + divide) | 2 |
| ArgMaxOp, ArgMinOp (index tracking) | 6 |
| AllOp, AnyOp (early exit) | 4 |
| StdOp, VarOp (two-pass) | 6 |
| Axis reduction with stride iteration | 8 |
| Replace ILKernelGenerator.Reduction.cs | 8 |
| Benchmarks | 4 |

**Total Phase 2**: ~42 hours

---

## Phase 3: Binary Operations (Week 3-4)

### 3.1 Operations to Migrate

| Operation | Current | Priority |
|-----------|---------|----------|
| Add | `ILKernelGenerator.Binary.cs` | P0 |
| Subtract | `ILKernelGenerator.Binary.cs` | P0 |
| Multiply | `ILKernelGenerator.Binary.cs` | P0 |
| Divide | `ILKernelGenerator.Binary.cs` | P0 |
| Power | `ILKernelGenerator.MixedType.cs` | P1 |
| FloorDivide | `ILKernelGenerator.MixedType.cs` | P1 |
| Mod | `DefaultEngine` | P1 |
| BitwiseAnd | `ILKernelGenerator.Binary.cs` | P2 |
| BitwiseOr | `ILKernelGenerator.Binary.cs` | P2 |
| BitwiseXor | `ILKernelGenerator.Binary.cs` | P2 |
| Maximum | `np.maximum.cs` | P1 |
| Minimum | `np.minimum.cs` | P1 |

### 3.2 Execution Paths

Each binary op needs 5 execution paths:

```
┌─────────────────────────────────────────────────────────────┐
│                    Binary Operation                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐        │
│  │ SimdFull    │   │ SimdScalar  │   │ SimdChunk   │        │
│  │ Both        │   │ One operand │   │ Inner axis  │        │
│  │ contiguous  │   │ is scalar   │   │ contiguous  │        │
│  └─────────────┘   └─────────────┘   └─────────────┘        │
│         │                 │                 │                │
│         ▼                 ▼                 ▼                │
│  ┌─────────────────────────────────────────────────┐        │
│  │              4x Unrolled SIMD Loop              │        │
│  │  acc0 = op(v0, u0); acc1 = op(v1, u1); ...      │        │
│  └─────────────────────────────────────────────────┘        │
│                                                              │
│  ┌─────────────┐   ┌─────────────┐                          │
│  │ General     │   │ Broadcast   │   (Fallback paths)       │
│  │ Strided     │   │ Expansion   │                          │
│  └─────────────┘   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| AddOp, SubOp, MulOp, DivOp structs | 4 |
| PowerOp, FloorDivideOp, ModOp | 6 |
| BitwiseAnd/Or/Xor ops | 4 |
| MaximumOp, MinimumOp | 2 |
| SimdFull path with 4x unroll | 6 |
| SimdScalar path | 4 |
| SimdChunk path | 6 |
| Replace ILKernelGenerator.Binary.cs | 8 |
| Benchmarks | 4 |

**Total Phase 3**: ~44 hours

---

## Phase 4: Unary Operations (Week 4-5)

### 4.1 Operations to Migrate

| Category | Operations | Priority |
|----------|------------|----------|
| Basic | Negate, Abs, Sign | P0 |
| Rounding | Floor, Ceil, Truncate, Round | P1 |
| Sqrt/Power | Sqrt, Cbrt, Square, Reciprocal | P0 |
| Trig | Sin, Cos, Tan, Asin, Acos, Atan | P2 |
| Exp/Log | Exp, Log, Log2, Log10, Log1p, Expm1 | P1 |
| Hyperbolic | Sinh, Cosh, Tanh | P3 |
| Bitwise | BitwiseNot | P2 |

### 4.2 SIMD Strategy by Operation Type

**Direct SIMD (hardware intrinsics exist)**:
- Negate: `Vector256.Negate()`
- Abs: `Vector256.Abs()`
- Sqrt: `Vector256.Sqrt()`
- Floor/Ceil: `Vector256.Floor()`, `Vector256.Ceiling()`

**Software SIMD (compute per-lane)**:
- Sign: Compare with zero, select -1/0/1
- Square: `v * v`
- Reciprocal: `Vector256.One / v`

**Math library (no SIMD, use MathF/Math)**:
- Trig functions: Fall back to scalar MathF.Sin, etc.
- Exp/Log: Scalar fallback (no hardware SIMD for these)

### 4.3 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| NegateOp, AbsOp, SignOp | 4 |
| FloorOp, CeilOp, TruncateOp | 4 |
| SqrtOp, CbrtOp, SquareOp, ReciprocalOp | 6 |
| Trig ops (scalar fallback) | 8 |
| Exp/Log ops (scalar fallback) | 8 |
| Replace ILKernelGenerator.Unary.cs | 6 |
| Benchmarks | 4 |

**Total Phase 4**: ~40 hours

---

## Phase 5: Comparison Operations (Week 5)

### 5.1 Operations

| Operation | SIMD Method |
|-----------|-------------|
| Equal | `Vector256.Equals()` |
| NotEqual | `Vector256.Equals()` + invert |
| Less | `Vector256.LessThan()` |
| LessEqual | `Vector256.LessThanOrEqual()` |
| Greater | `Vector256.GreaterThan()` |
| GreaterEqual | `Vector256.GreaterThanOrEqual()` |

### 5.2 Mask-to-Bool Conversion

SIMD comparisons return vector masks. Convert to bool array:

```csharp
// Result: Vector256 where each lane is all 1s (true) or all 0s (false)
var mask = Vector256.Equals(a, b);

// Extract to bitmask
uint bits = Vector256.ExtractMostSignificantBits(mask);

// Expand bits to bytes (0 or 1)
for (int j = 0; j < vectorCount; j++)
{
    result[i + j] = (byte)((bits >> j) & 1);
}
```

### 5.3 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| EqualOp, NotEqualOp | 4 |
| LessOp, LessEqualOp | 4 |
| GreaterOp, GreaterEqualOp | 4 |
| Mask-to-bool conversion | 4 |
| Replace ILKernelGenerator.Comparison.cs | 6 |
| Benchmarks | 2 |

**Total Phase 5**: ~24 hours

---

## Phase 6: MatMul / Dot Product (Week 6-7)

### 6.1 Current State

`ILKernelGenerator.MatMul.cs` has basic blocked SIMD but:
- No loop unrolling in inner kernel
- No prefetching
- Single accumulator

### 6.2 Optimized Inner Kernel

```csharp
// Current: Single accumulator
for (int k = 0; k < K; k++)
{
    var aVec = Vector256.Create(A[i, k]);
    var bVec = Vector256.Load(&B[k, j]);
    cVec = Fma.MultiplyAdd(aVec, bVec, cVec);  // Serial dependency
}

// Optimized: 4 accumulators + unrolling
var c0 = Vector256<double>.Zero;
var c1 = Vector256<double>.Zero;
var c2 = Vector256<double>.Zero;
var c3 = Vector256<double>.Zero;

for (int k = 0; k < K; k += 4)
{
    var a0 = Vector256.Create(A[i, k]);
    var a1 = Vector256.Create(A[i, k+1]);
    var a2 = Vector256.Create(A[i, k+2]);
    var a3 = Vector256.Create(A[i, k+3]);

    c0 = Fma.MultiplyAdd(a0, Vector256.Load(&B[k, j]), c0);
    c1 = Fma.MultiplyAdd(a1, Vector256.Load(&B[k+1, j]), c1);
    c2 = Fma.MultiplyAdd(a2, Vector256.Load(&B[k+2, j]), c2);
    c3 = Fma.MultiplyAdd(a3, Vector256.Load(&B[k+3, j]), c3);
}

var result = (c0 + c1) + (c2 + c3);  // Tree reduction
```

### 6.3 Cache Blocking Strategy

```
Block sizes optimized for L1/L2 cache:
- BLOCK_M = 64 (rows of A/C)
- BLOCK_N = 256 (columns of B/C)
- BLOCK_K = 256 (shared dimension)

For 8KB L1d cache: 64 * 8 = 512 bytes of A row fits
For 256KB L2 cache: 256 * 256 * 8 = 512KB of B block fits
```

### 6.4 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| Analyze current MatMul performance | 4 |
| Implement 4x unrolled inner kernel | 8 |
| Optimize cache blocking parameters | 6 |
| Add prefetching hints | 4 |
| Parallel outer loops | 6 |
| Replace ILKernelGenerator.MatMul.cs | 8 |
| Benchmarks vs NumPy | 4 |

**Total Phase 6**: ~40 hours

---

## Phase 7: Advanced Operations (Week 7-8)

### 7.1 Specialized Operations

| Operation | SIMD Strategy |
|-----------|---------------|
| Clip | `Vector256.Min(Vector256.Max(v, min), max)` |
| Modf | Extract integer part, subtract for fraction |
| CumSum | Prefix sum with SIMD (complex) |
| NonZero | SIMD compare with zero, extract indices |
| Where | SIMD blend based on mask |

### 7.2 CumSum SIMD

Prefix sum is challenging for SIMD but possible:

```csharp
// Hillis-Steele parallel prefix sum
// For Vector256<double> (4 elements):
// Step 1: [a, a+b, b+c, c+d]
// Step 2: [a, a+b, a+b+c, a+b+c+d]

var v = Vector256.Load(data + i);
var shifted1 = Vector256.Shuffle(v, ...);
v = v + shifted1;
var shifted2 = Vector256.Shuffle(v, ...);
v = v + shifted2;
// Add carry from previous vector
v = v + Vector256.Create(carry);
carry = v[3];  // Last element becomes next carry
```

### 7.3 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| ClipOp SIMD | 4 |
| ModfOp SIMD | 4 |
| CumSum prefix sum | 12 |
| NonZero SIMD | 6 |
| Where/Select SIMD | 6 |
| Benchmarks | 4 |

**Total Phase 7**: ~36 hours

---

## Phase 8: Integration & Cleanup (Week 8-9)

### 8.1 Replace Old Implementations

| File to Remove/Deprecate | Replacement |
|-------------------------|-------------|
| `ILKernelGenerator.Reduction.cs` (IL emit) | `SimdOps.Reduction.cs` |
| `ILKernelGenerator.Binary.cs` | `SimdOps.Binary.cs` |
| `ILKernelGenerator.Unary.cs` | `SimdOps.Unary.cs` |
| `ILKernelGenerator.Comparison.cs` | `SimdOps.Comparison.cs` |
| `Default.Reduction.*.cs` (Regen) | `SimdOps` |
| `SimdKernels.cs` (old) | `SimdOps` |

### 8.2 DefaultEngine Refactor

```csharp
public partial class DefaultEngine
{
    // All numeric ops delegate to SimdOps
    public override NDArray Sum(in NDArray arr, int? axis, ...)
    {
        if (axis == null)
            return SimdOps.ReduceAll<SumOp>(arr);
        else
            return SimdOps.ReduceAxis<SumOp>(arr, axis.Value);
    }
}
```

### 8.3 Deliverables

| Task | Estimated Hours |
|------|-----------------|
| Update DefaultEngine to use SimdOps | 12 |
| Remove/deprecate old IL generators | 4 |
| Remove Regen templates | 4 |
| Update IKernelProvider interface | 4 |
| Full test suite pass | 8 |
| Documentation | 4 |

**Total Phase 8**: ~36 hours

---

## Timeline Summary

| Phase | Description | Hours | Weeks |
|-------|-------------|-------|-------|
| 1 | Foundation | 22 | 1 |
| 2 | Reductions | 42 | 1.5 |
| 3 | Binary Ops | 44 | 1.5 |
| 4 | Unary Ops | 40 | 1 |
| 5 | Comparisons | 24 | 0.5 |
| 6 | MatMul | 40 | 1.5 |
| 7 | Advanced | 36 | 1 |
| 8 | Integration | 36 | 1 |
| **Total** | | **284** | **9** |

---

## Success Metrics

### Performance Targets

| Operation | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| Sum (100K) | 0.010ms | 0.004ms | 2.5x |
| Max (100K) | 0.065ms | 0.022ms | 3x |
| MatMul (1K×1K) | 50ms | 15ms | 3x |
| Binary Add (1M) | 0.5ms | 0.3ms | 1.7x |

### Code Quality Targets

- Remove ~15,000 lines of Regen templates
- Remove ~3,000 lines of IL emission code
- Single unified SIMD pattern for all ops
- 100% test coverage for SIMD paths

### Benchmark Suite

Create `bench/SimdBenchmarkSuite/` with:
- Per-operation benchmarks
- Size sweep (1K → 100M)
- Dtype coverage (float32, float64, int32, int64)
- Comparison with NumPy baseline

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Generic Math constraints incompatible with some types | Medium | Keep type-specific fallbacks for bool, char, decimal |
| SIMD not beneficial for small arrays | Low | Size threshold checks, scalar fallback |
| Memory-bound ops won't improve | Medium | Focus on CPU-bound sizes, document limitations |
| Breaking changes in API | Low | Internal refactor only, public API unchanged |
| JIT not optimizing generic code | Medium | Benchmark each implementation, use concrete types if needed |

---

## Quick Wins (Can Do Immediately)

1. **Replace `ReduceContiguousAxisSimd256` in `ILKernelGenerator.Reduction.cs`** with 4x unrolled version from `SimdReductionOptimized.cs` - immediate 2.5x speedup for reductions

2. **Add prefetch hints** to existing SIMD loops:
   ```csharp
   Sse.Prefetch1(data + i + 64);  // Prefetch 1 cache line ahead
   ```

3. **Fix axis reduction coordinate calculation** - replace division/modulo with increment-based iteration

---

## Files to Create

```
src/NumSharp.Core/Backends/Kernels/
├── SimdInterfaces.cs          # IReduceOp, IBinaryOp, IUnaryOp
├── SimdOps.cs                 # Core dispatch logic
├── SimdOps.Reduction.cs       # Reduction implementations
├── SimdOps.Binary.cs          # Binary op implementations
├── SimdOps.Unary.cs           # Unary op implementations
├── SimdOps.Comparison.cs      # Comparison implementations
├── SimdOps.MatMul.cs          # Matrix multiplication
└── SimdOps.Advanced.cs        # Clip, CumSum, etc.

bench/SimdBenchmarkSuite/
├── Program.cs
├── ReductionBenchmarks.cs
├── BinaryBenchmarks.cs
├── MatMulBenchmarks.cs
└── ComparisonWithNumPy.cs
```

---

## Next Steps

1. **Approve this plan** - review with team
2. **Phase 1 kickoff** - create interfaces and core templates
3. **Quick win** - integrate `SimdReductionOptimized.cs` into production code

Ready to begin Phase 1?
