# IL Kernel Generation in NumSharp

NumSharp achieves near-native performance through runtime IL (Intermediate Language) generation using `System.Reflection.Emit.DynamicMethod`. This document provides a comprehensive guide to the IL kernel system architecture, techniques, and coverage.

If you're working on NumSharp internals or trying to understand how NumSharp achieves its performance, this guide will walk you through the entire IL kernel system. Whether you're debugging a performance issue, adding a new operation, or simply curious about the implementation, you'll find detailed explanations and practical examples throughout.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Organization](#file-organization)
- [Execution Paths](#execution-paths)
- [SIMD Optimization Techniques](#simd-optimization-techniques)
- [Operation Coverage](#operation-coverage)
- [Type Support](#type-support)
- [Cache System](#cache-system)
- [Adding New Operations](#adding-new-operations)
- [Performance Considerations](#performance-considerations)
- [Debugging IL Generation](#debugging-il-generation)

---

## Overview

### Why IL Generation?

You might wonder why NumSharp goes through the complexity of generating IL at runtime rather than using straightforward C# loops. The answer lies in the dramatic performance gains this approach enables.

When you call an operation like `np.add(a, b)`, NumSharp doesn't simply iterate through elements with generic code. Instead, it generates specialized machine code tailored to your exact types and array layouts. This eliminates the overhead that would otherwise kill performance in a numerical computing library.

NumSharp generates optimized machine code at runtime rather than relying on interpreted loops or generic abstractions. This provides:

1. **Type Specialization** - Eliminates boxing/unboxing and virtual dispatch
2. **SIMD Vectorization** - Leverages Vector128/Vector256/Vector512 intrinsics
3. **Loop Fusion** - Combines operations to reduce memory traffic
4. **Stride Optimization** - Generates path-specific code for contiguous vs strided arrays

### Performance Impact

To give you a sense of the gains, here's what you can expect when IL kernels kick in compared to naive scalar implementations:

| Scenario | Speedup vs Naive |
|----------|------------------|
| Contiguous SIMD (float/double) | 8-16x |
| Contiguous SIMD (int32/int64) | 4-8x |
| Strided arrays | 2-4x |
| Type promotion (int32 + float64) | 3-6x |

These aren't theoretical numbers—they reflect real-world benchmarks on modern CPUs with AVX2 support. If you're processing millions of elements, you'll see the difference immediately.

---

## Architecture

Understanding the architecture will help you navigate the codebase and know where to look when debugging or extending the system.

### Core Components

The `ILKernelGenerator` is a static partial class split across 28 files. Each file owns a specific category of operations, making it easier for you to find and modify the code you need. Here's the overall structure:

```
ILKernelGenerator (static partial class)
├── Core Infrastructure (.cs)
│   ├── Enabled flag, VectorBits detection
│   ├── Type mapping (NPTypeCode ↔ CLR Type ↔ Vector type)
│   └── Shared IL emission primitives
│
├── Operation Partials
│   ├── Binary operations (.Binary.cs, .MixedType.cs)
│   ├── Unary operations (.Unary.*.cs)
│   ├── Comparisons (.Comparison.cs)
│   ├── Reductions (.Reduction.*.cs)
│   ├── Scans (.Scan.cs)
│   ├── Shifts (.Shift.cs)
│   ├── MatMul (.MatMul.cs)
│   └── Masking (.Masking.*.cs)
│
└── Supporting Classes
    ├── StrideDetector - Execution path classification
    ├── TypeRules - Type size, SIMD capability checks
    ├── SimdThresholds - Minimum sizes for SIMD benefit
    ├── SimdMatMul - Cache-blocked matrix multiplication
    └── SimdReductionOptimized - Unrolled reduction kernels
```

### Flow: Request to Execution

When you perform an operation like `a + b` on two NDArrays, here's what happens under the hood. Understanding this flow will help you debug issues and trace where performance bottlenecks might occur:

```
1. Caller (DefaultEngine, np.*, NDArray operator)
       ↓
2. ILKernelGenerator.Get*Kernel() or TryGet*Kernel()
       ↓
3. Cache lookup by operation key
       ↓ (cache miss)
4. IL generation via DynamicMethod
       ↓
5. JIT compilation to native code
       ↓
6. Delegate cached and returned
       ↓
7. Caller invokes delegate with array pointers
```

The first time you execute a particular operation (say, adding two `float` arrays), there's a one-time cost to generate and JIT-compile the kernel. All subsequent calls with the same type and operation hit the cache, so you get the optimized delegate immediately. This is why the second iteration of your code often runs noticeably faster than the first.

---

## File Organization

When you need to modify or debug the IL generation code, knowing where to look is half the battle. The system is organized by operation category, so you can quickly navigate to the relevant file.

### Partial Class Files (28 files, ~18K lines)

Don't be intimidated by the file count—each file has a focused responsibility. Here's your roadmap to finding what you need:

| File | Responsibility |
|------|----------------|
| `ILKernelGenerator.cs` | Core infrastructure, type mapping, shared IL primitives |
| `ILKernelGenerator.Binary.cs` | Same-type contiguous binary ops (Add, Sub, Mul, Div) |
| `ILKernelGenerator.MixedType.cs` | Type-promoting binary ops, ClearAll() |
| `ILKernelGenerator.Unary.cs` | Unary kernel cache, loop emission |
| `ILKernelGenerator.Unary.Math.cs` | Math function IL emission (Sin, Cos, Exp, Log, etc.) |
| `ILKernelGenerator.Unary.Vector.cs` | SIMD vector operations for unary ops |
| `ILKernelGenerator.Unary.Decimal.cs` | Decimal-specific unary operations |
| `ILKernelGenerator.Unary.Predicate.cs` | IsNaN, IsFinite, IsInf predicates |
| `ILKernelGenerator.Scalar.cs` | Func<TIn,TOut> and Func<TLhs,TRhs,TResult> delegates |
| `ILKernelGenerator.Comparison.cs` | Comparison ops (==, !=, <, >, <=, >=) |
| `ILKernelGenerator.Reduction.cs` | Element-wise reductions (Sum, Prod, Min, Max, Mean) |
| `ILKernelGenerator.Reduction.Boolean.cs` | All/Any with early-exit optimization |
| `ILKernelGenerator.Reduction.Arg.cs` | ArgMax/ArgMin (value + index tracking) |
| `ILKernelGenerator.Reduction.NaN.cs` | NaN-aware reductions (NanSum, NanMax, etc.) |
| `ILKernelGenerator.Reduction.Axis.cs` | Axis reduction dispatcher and general kernels |
| `ILKernelGenerator.Reduction.Axis.Simd.cs` | Typed SIMD axis reduction kernels |
| `ILKernelGenerator.Reduction.Axis.Arg.cs` | Axis ArgMax/ArgMin |
| `ILKernelGenerator.Reduction.Axis.VarStd.cs` | Axis Variance/Std (two-pass algorithm) |
| `ILKernelGenerator.Reduction.Axis.NaN.cs` | NaN-aware axis reductions |
| `ILKernelGenerator.Scan.cs` | CumSum, CumProd (prefix operations) |
| `ILKernelGenerator.Shift.cs` | LeftShift, RightShift (SIMD for scalar shift) |
| `ILKernelGenerator.MatMul.cs` | 2D matrix multiplication with SIMD |
| `ILKernelGenerator.Clip.cs` | Value clamping |
| `ILKernelGenerator.Modf.cs` | Fractional/integer split |
| `ILKernelGenerator.Masking.cs` | NonZero index collection |
| `ILKernelGenerator.Masking.Boolean.cs` | CountTrue, CopyMasked |
| `ILKernelGenerator.Masking.VarStd.cs` | Variance/Std SIMD helpers |
| `ILKernelGenerator.Masking.NaN.cs` | NaN-aware masking helpers |

### Supporting Files

Beyond the main partial class files, you'll find these supporting files that define the types and utilities the IL generation system depends on:

| File | Purpose |
|------|---------|
| `KernelOp.cs` | Enums: BinaryOp, UnaryOp, ReductionOp, ComparisonOp, ExecutionPath |
| `KernelKey.cs` | Cache key structs (ContiguousKernelKey, UnaryScalarKey, etc.) |
| `KernelSignatures.cs` | Delegate type definitions |
| `BinaryKernel.cs` | MixedTypeKernelKey, BinaryKernel<T>, UnaryKernel, ComparisonKernel |
| `ReductionKernel.cs` | Reduction keys, delegates, extension methods |
| `ScalarKernel.cs` | UnaryScalarKernelKey, BinaryScalarKernelKey |
| `TypeRules.cs` | Type size, accumulating type, SIMD capability |
| `StrideDetector.cs` | Execution path classification |
| `IndexCollector.cs` | Growable long index buffer for NonZero |
| `SimdThresholds.cs` | Minimum element counts for SIMD benefit |
| `SimdReductionOptimized.cs` | Unrolled reduction kernels (4x/8x) |
| `SimdMatMul.cs` | GEBP algorithm with cache blocking |

---

## Execution Paths

One of the most important concepts to understand is how NumSharp selects the right execution path for your arrays. Not all arrays are created equal—a contiguous array can use blazing-fast SIMD loops, while a transposed or sliced array may need coordinate-based iteration.

### StrideDetector Classification

Before executing any operation, NumSharp analyzes your arrays' memory layout using `StrideDetector.Classify<T>()`. This determines which code path will be most efficient for your specific situation:

```csharp
public enum ExecutionPath
{
    SimdFull,        // Both operands contiguous - flat SIMD loop
    SimdScalarRight, // RHS is scalar (all strides = 0)
    SimdScalarLeft,  // LHS is scalar (all strides = 0)
    SimdChunk,       // Inner dimension contiguous - chunked SIMD
    General          // Arbitrary strides - coordinate iteration
}
```

### Path Selection Logic

Understanding this priority order helps you predict which path your code will take. If you're seeing slower-than-expected performance, check whether your arrays are hitting the General path when you expected SimdFull:

```csharp
// Priority order (first match wins):
1. Both contiguous → SimdFull (fastest)
2. RHS all strides = 0 → SimdScalarRight (broadcast)
3. LHS all strides = 0 → SimdScalarLeft (broadcast)
4. Inner stride = 1 or 0 for both → SimdChunk
5. Otherwise → General (coordinate-based)
```

**Tip:** If you need maximum performance, ensure your arrays are contiguous. You can check this with `ndarray.IsContiguous` or force contiguity with `np.ascontiguousarray()`.

### Path-Specific Code Generation

**SimdFull Path:**
```
for i = 0 to count step vectorCount:
    result[i:i+vc] = op(lhs[i:i+vc], rhs[i:i+vc])  // SIMD
for i = vectorEnd to count:
    result[i] = op(lhs[i], rhs[i])  // scalar tail
```

**SimdScalarRight Path:**
```
rhsVec = Vector256.Create(rhs[0])  // broadcast scalar
for i = 0 to count step vectorCount:
    result[i:i+vc] = op(lhs[i:i+vc], rhsVec)
```

**General Path:**
```
for linearIdx = 0 to totalSize:
    coords = IndexToCoordinates(linearIdx, shape)
    lhsOffset = dot(coords, lhsStrides)
    rhsOffset = dot(coords, rhsStrides)
    result[linearIdx] = op(lhs[lhsOffset], rhs[rhsOffset])
```

---

## SIMD Optimization Techniques

This section covers the optimization techniques used throughout the IL kernel system. If you're implementing a new operation or trying to squeeze out more performance, these patterns are your toolkit.

### 1. Loop Unrolling (4x)

You might think that processing one vector at a time is efficient enough, but modern CPUs can execute multiple independent instructions simultaneously. By unrolling the loop 4x, we give the CPU more work to parallelize:

Processing 4 vectors per iteration reduces loop overhead and enables instruction-level parallelism:

```csharp
// 4x unrolled SIMD loop
for (; i <= unrollEnd; i += vectorCount * 4)
{
    var v0 = Vector256.Load(data + i);
    var v1 = Vector256.Load(data + i + vectorCount);
    var v2 = Vector256.Load(data + i + vectorCount * 2);
    var v3 = Vector256.Load(data + i + vectorCount * 3);

    acc0 += v0;  // Independent - can execute in parallel
    acc1 += v1;
    acc2 += v2;
    acc3 += v3;
}

// Remainder loop (0-3 vectors)
for (; i <= vectorEnd; i += vectorCount)
    acc0 += Vector256.Load(data + i);

// Scalar tail
for (; i < size; i++)
    result += data[i];
```

### 2. Tree Reduction

When you have 4 accumulator vectors and need to combine them into a single result, the naive approach creates a serial dependency chain where each addition must wait for the previous one. Tree reduction solves this by allowing parallel execution:

```csharp
// Instead of: result = acc0 + acc1 + acc2 + acc3 (serial - each add waits for previous)
// Use tree pattern:
var sum01 = acc0 + acc1;   // Can execute in parallel
var sum23 = acc2 + acc3;   // Can execute in parallel
var sumAll = sum01 + sum23;
double result = Vector256.Sum(sumAll);
```

This might seem like a minor detail, but it can make a significant difference on modern out-of-order CPUs.

### 3. Multiple Accumulator Vectors

Using independent accumulators maximizes instruction-level parallelism (ILP):

```csharp
// 4 independent accumulators - no dependency between updates
var acc0 = Vector256<double>.Zero;
var acc1 = Vector256<double>.Zero;
var acc2 = Vector256<double>.Zero;
var acc3 = Vector256<double>.Zero;
```

### 4. FMA (Fused Multiply-Add)

If you're running on a CPU with FMA support (most Intel Haswell and newer, AMD Piledriver and newer), you get a significant boost. FMA performs `a * b + c` in a single instruction with only one rounding step, which is both faster and more accurate:

```csharp
if (Fma.IsSupported)
    c = Fma.MultiplyAdd(a, b, c);  // c = a * b + c in one instruction
else
    c = Vector256.Add(c, Vector256.Multiply(a, b));
```

The IL kernel system automatically detects FMA availability at runtime and uses it when possible. You don't need to do anything special—just make sure you're running on capable hardware.

### 5. Cache Blocking (MatMul)

Matrix multiplication is where cache blocking becomes critical. Without it, you'd constantly fetch data from main memory, destroying performance. The GEBP (General Block Panel) algorithm ensures that your working set fits in cache:

```csharp
// Block sizes tuned for L1=32KB, L2=256KB
const int MC = 64;   // A panel rows
const int KC = 256;  // K depth
const int MR = 8;    // Micro-kernel rows
const int NR = 16;   // Micro-kernel cols (2 vectors)

// Panel packing: A[kc][MR], B[kc][NR] for sequential access
```

These constants were tuned empirically for typical modern CPUs. If you're targeting specialized hardware, you might benefit from different values.

### 6. Early Exit (Boolean Reductions)

For `np.all()` and `np.any()`, you don't always need to scan the entire array. If you're checking whether all elements are true and you find a false, you can stop immediately. The IL kernels exploit this with SIMD-accelerated early exit:

```csharp
// All: exit when first zero found
for (; i <= vectorEnd; i += vectorCount)
{
    var vec = Vector256.Load(data + i);
    if (Vector256.EqualsAny(vec, zero))
        return false;  // Early exit!
}

// Any: exit when first non-zero found
for (; i <= vectorEnd; i += vectorCount)
{
    var vec = Vector256.Load(data + i);
    if (!Vector256.EqualsAll(vec, zero))
        return true;  // Early exit!
}
```

---

## Operation Coverage

### Binary Operations

| Operation | SIMD | Scalar | Types |
|-----------|------|--------|-------|
| Add | Vector256 op_Addition | OpCodes.Add | All numeric |
| Subtract | Vector256 op_Subtraction | OpCodes.Sub | All numeric |
| Multiply | Vector256 op_Multiply | OpCodes.Mul | All numeric |
| Divide | Vector256 op_Division | OpCodes.Div/Div_Un | All numeric |
| Power | - | Math.Pow | All numeric |
| FloorDivide | - | Div + Math.Floor | All numeric |
| BitwiseAnd | Vector256.BitwiseAnd | OpCodes.And | Integers |
| BitwiseOr | Vector256.BitwiseOr | OpCodes.Or | Integers |
| BitwiseXor | Vector256.Xor | OpCodes.Xor | Integers |
| LeftShift | Vector256.ShiftLeft | OpCodes.Shl | Integers |
| RightShift | Vector256.ShiftRight* | OpCodes.Shr/Shr_Un | Integers |
| ATan2 | - | Math.Atan2 | float, double |

### Unary Operations

| Operation | SIMD | Scalar | Notes |
|-----------|------|--------|-------|
| Negate | op_UnaryNegation | OpCodes.Neg | Two's complement for unsigned |
| Abs | Vector.Abs | Bitwise for int, Math.Abs for float |
| Sqrt | Vector.Sqrt | Math.Sqrt/MathF.Sqrt |
| Square | Multiply(dup) | OpCodes.Dup + Mul |
| Reciprocal | Divide(One, x) | 1.0 / x |
| Floor | Vector.Floor | Math.Floor |
| Ceiling | Vector.Ceiling | Math.Ceiling |
| Round | Vector.Round | Math.Round |
| Truncate | Vector.Truncate | Math.Truncate |
| Sign | - | Bitwise comparison | NaN → NaN |
| Exp | - | Math.Exp |
| Exp2 | - | Math.Pow(2, x) |
| Expm1 | - | Math.Exp(x) - 1 |
| Log | - | Math.Log |
| Log2 | - | Math.Log2 |
| Log10 | - | Math.Log10 |
| Log1p | - | Math.Log(1 + x) |
| Sin/Cos/Tan | - | Math.Sin/Cos/Tan |
| ASin/ACos/ATan | - | Math.Asin/Acos/Atan |
| Sinh/Cosh/Tanh | - | Math.Sinh/Cosh/Tanh |
| Cbrt | - | Math.Cbrt |
| Deg2Rad | Multiply(factor) | x * (π/180) |
| Rad2Deg | Multiply(factor) | x * (180/π) |
| BitwiseNot | OnesComplement | OpCodes.Not |
| LogicalNot | - | x == 0 |
| IsFinite | - | float.IsFinite/double.IsFinite |
| IsNaN | - | float.IsNaN/double.IsNaN |
| IsInf | - | float.IsInfinity/double.IsInfinity |

### Comparison Operations

| Operation | IL Opcode | Result |
|-----------|-----------|--------|
| Equal | Ceq | bool |
| NotEqual | Ceq + Ldc_I4_0 + Ceq | bool |
| Less | Clt / Clt_Un | bool |
| Greater | Cgt / Cgt_Un | bool |
| LessEqual | Cgt + Ldc_I4_0 + Ceq | bool |
| GreaterEqual | Clt + Ldc_I4_0 + Ceq | bool |

### Reduction Operations

| Operation | Element-wise | Axis | Identity |
|-----------|--------------|------|----------|
| Sum | SIMD + horizontal | SIMD per slice | 0 |
| Prod | SIMD + horizontal | SIMD per slice | 1 |
| Min | SIMD Max + horizontal | SIMD per slice | +∞ |
| Max | SIMD Min + horizontal | SIMD per slice | -∞ |
| Mean | Sum / count | Sum / axisSize | 0 |
| Var | Two-pass | Two-pass | - |
| Std | sqrt(Var) | sqrt(Var) | - |
| All | SIMD compare + early exit | Per slice | true |
| Any | SIMD compare + early exit | Per slice | false |
| ArgMax | Track value + index | Per slice | 0 |
| ArgMin | Track value + index | Per slice | 0 |
| NanSum | Skip NaN | Skip NaN | 0 |
| NanProd | Skip NaN | Skip NaN | 1 |
| NanMin | Skip NaN | Skip NaN | NaN if all NaN |
| NanMax | Skip NaN | Skip NaN | NaN if all NaN |
| NanMean | Skip NaN | Skip NaN | NaN if all NaN |
| NanVar | Skip NaN | Skip NaN | NaN if all NaN |
| NanStd | Skip NaN | Skip NaN | NaN if all NaN |

### Scan Operations

| Operation | Contiguous | Strided |
|-----------|------------|---------|
| CumSum | Sequential accumulation | Coordinate iteration |
| CumProd | Sequential accumulation | Coordinate iteration |

---

## Type Support

### All 12 NumSharp Types

| NPTypeCode | CLR Type | Size | SIMD Support |
|------------|----------|------|--------------|
| Boolean | bool | 1 | Limited (comparison) |
| Byte | byte | 1 | Vector256 (32 elements) |
| Int16 | short | 2 | Vector256 (16 elements) |
| UInt16 | ushort | 2 | Vector256 (16 elements) |
| Int32 | int | 4 | Vector256 (8 elements) |
| UInt32 | uint | 4 | Vector256 (8 elements) |
| Int64 | long | 8 | Vector256 (4 elements) |
| UInt64 | ulong | 8 | Vector256 (4 elements) |
| Single | float | 4 | Vector256 (8 elements) |
| Double | double | 8 | Vector256 (4 elements) |
| Char | char | 2 | Limited |
| Decimal | decimal | 16 | None (scalar only) |

### Type Promotion (NEP50 Alignment)

```csharp
// Accumulating types for reductions
GetAccumulatingType(int32)  → int64   // Prevent overflow
GetAccumulatingType(uint16) → uint64
GetAccumulatingType(float)  → float   // Preserve precision
GetAccumulatingType(double) → double
```

### SIMD Thresholds

Minimum element counts where SIMD overhead is worthwhile:

| Type | Threshold |
|------|-----------|
| byte | 64 |
| short/ushort | 64 |
| int/uint/float | 96 |
| long/ulong | 256 |
| double | 512 |

---

## Cache System

### Cache Key Structures

Each operation category has a unique key structure:

```csharp
// Binary operations
record struct ContiguousKernelKey(NPTypeCode Type, BinaryOp Op);
record struct MixedTypeKernelKey(NPTypeCode Lhs, NPTypeCode Rhs,
    NPTypeCode Result, BinaryOp Op, ExecutionPath Path);

// Unary operations
record struct UnaryKernelKey(NPTypeCode Input, NPTypeCode Output,
    UnaryOp Op, bool IsContiguous);
record struct UnaryScalarKernelKey(NPTypeCode Input, NPTypeCode Output, UnaryOp Op);

// Reductions
record struct ElementReductionKernelKey(NPTypeCode Input,
    NPTypeCode Accumulator, ReductionOp Op, bool IsContiguous);
record struct AxisReductionKernelKey(NPTypeCode Input,
    NPTypeCode Accumulator, ReductionOp Op, bool InnerAxisContiguous);

// Comparisons
record struct ComparisonKernelKey(NPTypeCode Lhs, NPTypeCode Rhs,
    ComparisonOp Op, ExecutionPath Path);
```

### Cache Implementation

```csharp
// ConcurrentDictionary for thread-safe access
private static readonly ConcurrentDictionary<ContiguousKernelKey, Delegate>
    _contiguousKernelCache = new();

// GetOrAdd pattern for atomic cache population
public static ContiguousKernel<T>? GetContiguousKernel<T>(BinaryOp op)
{
    var key = (op, typeof(T));
    if (_contiguousKernelCache.TryGetValue(key, out var cached))
        return (ContiguousKernel<T>)cached;

    var kernel = TryGenerateContiguousKernel<T>(op);
    if (kernel == null) return null;

    if (_contiguousKernelCache.TryAdd(key, kernel))
        return kernel;
    return (ContiguousKernel<T>)_contiguousKernelCache[key];
}
```

### Cache Statistics

```csharp
public static int CachedCount => _contiguousKernelCache.Count;
public static int UnaryCachedCount => _unaryCache.Count;
public static int ElementReductionCachedCount => _elementReductionCache.Count;
// ... etc
```

---

## Delegate Signatures

### Binary Operations

```csharp
// Same-type contiguous
public unsafe delegate void ContiguousKernel<T>(
    T* lhs, T* rhs, T* result, long count) where T : unmanaged;

// Mixed-type with strides
public unsafe delegate void MixedTypeKernel(
    void* lhs, void* rhs, void* result,
    long* lhsStrides, long* rhsStrides, long* shape,
    int ndim, long totalSize);
```

### Unary Operations

```csharp
public unsafe delegate void UnaryKernel(
    void* input, void* output,
    long* strides, long* shape,
    int ndim, long totalSize);
```

### Reductions

```csharp
// Element-wise (full array → scalar)
public unsafe delegate TResult TypedElementReductionKernel<TResult>(
    void* input, long* strides, long* shape,
    int ndim, long totalSize) where TResult : unmanaged;

// Axis reduction
public unsafe delegate void AxisReductionKernel(
    void* input, void* output,
    long* inputStrides, long* inputShape, long* outputStrides,
    int axis, long axisSize, int ndim, long outputSize);
```

### Comparisons

```csharp
public unsafe delegate void ComparisonKernel(
    void* lhs, void* rhs, bool* result,
    long* lhsStrides, long* rhsStrides, long* shape,
    int ndim, long totalSize);
```

---

## Adding New Operations

### Step 1: Define the Operation

Add to the appropriate enum in `KernelOp.cs`:

```csharp
public enum UnaryOp
{
    // ... existing ops ...
    MyNewOp,
}
```

### Step 2: Add Cache Key (if needed)

If using a new key structure, add to `KernelKey.cs` or the appropriate file.

### Step 3: Implement IL Emission

Add the IL emission logic to the appropriate partial file:

```csharp
// In ILKernelGenerator.Unary.Math.cs
case UnaryOp.MyNewOp:
    EmitMyNewOpCall(il, type);
    break;

private static void EmitMyNewOpCall(ILGenerator il, NPTypeCode type)
{
    if (type == NPTypeCode.Single)
    {
        // Call MathF.MyNewOp(float)
        var method = typeof(MathF).GetMethod("MyNewOp", new[] { typeof(float) });
        il.EmitCall(OpCodes.Call, method!, null);
    }
    else if (type == NPTypeCode.Double)
    {
        // Call Math.MyNewOp(double)
        var method = typeof(Math).GetMethod("MyNewOp", new[] { typeof(double) });
        il.EmitCall(OpCodes.Call, method!, null);
    }
    else
    {
        // Convert to double, call, convert back
        EmitConvertToDouble(il, type);
        var method = typeof(Math).GetMethod("MyNewOp", new[] { typeof(double) });
        il.EmitCall(OpCodes.Call, method!, null);
        EmitConvertFromDouble(il, type);
    }
}
```

### Step 4: Add SIMD Path (if applicable)

If the operation has SIMD support, add to `ILKernelGenerator.Unary.Vector.cs`:

```csharp
case UnaryOp.MyNewOp:
    EmitVectorMyNewOp(il, containerType, clrType, vectorType);
    break;
```

### Step 5: Update CanUseSimd Check

```csharp
private static bool CanUseUnarySimd(UnaryKernelKey key)
{
    // Add MyNewOp to SIMD-capable operations
    return key.Op == UnaryOp.Negate || key.Op == UnaryOp.Abs ||
           key.Op == UnaryOp.MyNewOp || // Add here
           // ...
}
```

### Step 6: Test

Write tests covering:
- All 12 types
- Contiguous and strided arrays
- Edge cases (NaN, Inf, empty arrays)
- NumPy compatibility verification

---

## Performance Considerations

### When IL Kernels Are Used

1. **Contiguous arrays** - SimdFull path with SIMD vectorization
2. **Broadcast operations** - SimdScalarRight/Left with scalar splatting
3. **Type promotion** - MixedType kernels with conversion
4. **Reductions** - Horizontal SIMD with tree reduction

### When IL Kernels Fall Back

1. **Decimal type** - No SIMD support, scalar loop
2. **Complex strides** - General path with coordinate iteration
3. **Very small arrays** - Below SimdThresholds, overhead not worthwhile
4. **Unsupported operations** - Returns null, caller uses fallback

### Memory Bandwidth vs Compute

For very large arrays (>10M elements), performance becomes memory-bound rather than compute-bound. SIMD still helps by:
- Reducing instruction count
- Improving prefetching
- Better cache utilization

### Alignment

All NumSharp allocations are naturally aligned (managed memory). For optimal SIMD performance:
- Vector256 prefers 32-byte alignment
- Vector512 prefers 64-byte alignment
- Unaligned loads/stores work but may be slower

---

## Debugging IL Generation

### Enable Diagnostics

IL generation failures are logged to Debug output:

```csharp
System.Diagnostics.Debug.WriteLine(
    $"[ILKernel] TryGenerateContiguousKernel<{typeof(T).Name}>({op}): " +
    $"{ex.GetType().Name}: {ex.Message}");
```

### Common IL Errors

| Error | Cause | Fix |
|-------|-------|-----|
| InvalidProgramException | Stack imbalance | Check push/pop count |
| VerificationException | Type mismatch | Add Conv_* instruction |
| NullReferenceException | Missing method | Check reflection lookup |

### Verifying Generated IL

Use a decompiler (ILSpy, dnSpy) to inspect generated methods:

```csharp
var dm = new DynamicMethod(...);
var il = dm.GetILGenerator();
// ... emit IL ...
var del = dm.CreateDelegate<...>();

// Inspect with:
// - ILSpy: Load assembly, find dynamic method
// - WinDbg: !dumpil <method address>
```

### Stack Tracking

Track stack depth mentally or with comments:

```csharp
il.Emit(OpCodes.Ldarg_0);    // Stack: [ptr]
il.Emit(OpCodes.Ldloc, idx); // Stack: [ptr, idx]
il.Emit(OpCodes.Conv_I);     // Stack: [ptr, idx(native)]
il.Emit(OpCodes.Ldc_I4, 4);  // Stack: [ptr, idx, 4]
il.Emit(OpCodes.Mul);        // Stack: [ptr, offset]
il.Emit(OpCodes.Add);        // Stack: [addr]
il.Emit(OpCodes.Ldind_R4);   // Stack: [value]
```

---

## Int64 Indexing

All loop counters and indices use `long` to support arrays >2GB:

```csharp
// Loop counter declaration
var locI = il.DeclareLocal(typeof(long));

// Load long constant
il.Emit(OpCodes.Ldc_I8, 0L);

// Increment
il.Emit(OpCodes.Ldloc, locI);
il.Emit(OpCodes.Ldc_I8, 1L);
il.Emit(OpCodes.Add);
il.Emit(OpCodes.Stloc, locI);

// Delegate signatures use long
public unsafe delegate void ContiguousKernel<T>(
    T* lhs, T* rhs, T* result, long count);
```

---

## Summary

The ILKernelGenerator system is the performance backbone of NumSharp, providing:

- **Runtime code generation** via System.Reflection.Emit
- **SIMD vectorization** with Vector128/256/512 support
- **Type specialization** eliminating boxing and virtual dispatch
- **Path optimization** for different memory layouts
- **Cache efficiency** through blocking and panel packing

This enables NumSharp to achieve performance competitive with native NumPy while maintaining the safety and productivity of managed .NET code.
