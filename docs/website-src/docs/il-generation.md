# IL Kernel Generation in NumSharp

NumSharp achieves near-native performance through runtime IL (Intermediate Language) generation using `System.Reflection.Emit.DynamicMethod`. This document provides a comprehensive guide to the IL kernel system architecture, techniques, and coverage.

This guide walks through the entire IL kernel system—useful for working on NumSharp internals, understanding its performance characteristics, debugging issues, adding new operations, or simply learning about the implementation. Detailed explanations and practical examples are provided throughout.

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

Why go through the complexity of generating IL at runtime rather than using straightforward C# loops? The answer lies in the dramatic performance gains this approach enables.

When an operation like `np.add(a, b)` executes, NumSharp doesn't simply iterate through elements with generic code. Instead, it generates specialized machine code tailored to the exact types and array layouts involved. This eliminates the overhead that would otherwise kill performance in a numerical computing library.

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

Understanding the architecture will help you navigate the codebase and know where to look when debugging or extending the system. If you've ever wondered how NumSharp achieves its performance, this section reveals the magic behind the curtain.

### Why a Static Partial Class?

The `ILKernelGenerator` is structured as a static partial class split across many files for practical reasons: IL generation code tends to be verbose—emitting individual opcodes line by line adds up quickly. By splitting responsibilities across 28 files, each file remains focused and manageable. Fixing a bug in matrix multiplication? Go straight to `ILKernelGenerator.MatMul.cs`. Adding a new unary operation? The path is clear.

The static nature ensures zero allocation overhead when requesting kernels. There's no object instantiation, no virtual dispatch—just direct method calls to retrieve cached delegates.

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

The first execution of a particular operation (say, adding two `float` arrays) incurs a one-time cost to generate and JIT-compile the kernel. All subsequent calls with the same type and operation hit the cache, returning the optimized delegate immediately. This is why the second iteration of a loop often runs noticeably faster than the first.

### Understanding the JIT Partnership

An important point: NumSharp's IL generation works *with* the .NET JIT compiler, not as a replacement for it. Calling `dm.CreateDelegate<T>()` hands a sequence of IL instructions to the JIT. The JIT then applies its own optimizations—register allocation, instruction scheduling, constant folding—before producing native machine code.

This partnership is powerful. The IL generation code focuses on correct memory access patterns and SIMD intrinsic calls. The JIT handles low-level details of mapping that IL to the specific CPU. On a machine with AVX-512, the JIT uses those wider registers. On older hardware, it falls back gracefully.

This is why debugging IL generation can be tricky: the emitted code isn't the code that runs. The JIT may transform IL significantly. Tools like WinDbg's `!dumpil` command or Disassembly view in Visual Studio help reveal what actually executes.

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

One of the most important concepts is how NumSharp selects the right execution path for arrays. Not all arrays are created equal—a contiguous array can use blazing-fast SIMD loops, while a transposed or sliced array may need coordinate-based iteration.

### StrideDetector Classification

Before executing any operation, NumSharp analyzes the arrays' memory layout using `StrideDetector.Classify<T>()`. This determines which code path will be most efficient:

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

Understanding this priority order helps predict which path code will take. Slower-than-expected performance often means arrays are hitting the General path instead of SimdFull:

```csharp
// Priority order (first match wins):
1. Both contiguous → SimdFull (fastest)
2. RHS all strides = 0 → SimdScalarRight (broadcast)
3. LHS all strides = 0 → SimdScalarLeft (broadcast)
4. Inner stride = 1 or 0 for both → SimdChunk
5. Otherwise → General (coordinate-based)
```

**Tip:** For maximum performance, ensure arrays are contiguous. Check with `ndarray.IsContiguous` or force contiguity with `np.ascontiguousarray()`.

### Practical Implications

Code like `result = a + b` abstracts away memory layout concerns. But understanding execution paths helps write faster NumSharp code:

1. **Prefer contiguous arrays**: Operations on freshly-allocated arrays (from `np.zeros`, `np.arange`, etc.) hit the SimdFull path. Sliced or transposed arrays may fall to slower paths.

2. **Scalar broadcast is nearly free**: With `a + 5`, the scalar `5` is broadcast. This hits SimdScalarRight, which broadcasts the scalar to a SIMD vector once and reuses it—nearly as fast as SimdFull.

3. **Transposed views are slow**: A transposed array (`a.T`) has non-contiguous memory access. Operations use coordinate-based iteration, which is significantly slower. For multiple operations on a transposed array, calling `np.ascontiguousarray()` once upfront often pays off.

4. **Inner-dimension slicing is cheap**: Slicing like `a[:, 0:100]` takes chunks of the inner dimension. These chunks may still be contiguous enough for SimdChunk optimization.

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

This section covers the optimization techniques used throughout the IL kernel system. These patterns form the toolkit for implementing new operations or squeezing out more performance. These aren't theoretical—they're the actual techniques implemented in the ILKernelGenerator code.

Understanding these patterns serves two purposes: appreciating why NumSharp performs the way it does, and knowing what techniques to apply when adding new operations.

### The Three-Level Loop Structure

Before diving into specific techniques, understand the standard loop structure used throughout the IL kernels:

```
┌─────────────────────────────────────────────────────────────┐
│  4x Unrolled SIMD Loop (processes 4 vectors per iteration) │
│  - Maximum throughput, minimum loop overhead                │
│  - Continues while: i <= totalSize - vectorCount * 4        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Remainder SIMD Loop (0-3 vectors)                          │
│  - Handles leftover full vectors                            │
│  - Continues while: i <= totalSize - vectorCount            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Scalar Tail Loop (0 to vectorCount-1 elements)             │
│  - Processes remaining individual elements                  │
│  - Continues while: i < totalSize                           │
└─────────────────────────────────────────────────────────────┘
```

This structure appears everywhere: binary operations, unary operations, reductions, comparisons. Once you recognize it, you'll see it throughout the codebase.

### 1. Loop Unrolling (4x)

Processing one vector at a time might seem efficient enough, but modern CPUs can execute multiple independent instructions simultaneously. By unrolling the loop 4x, the CPU gets more work to parallelize.

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

With 4 accumulator vectors that need combining into a single result, the naive approach creates a serial dependency chain where each addition must wait for the previous one. Tree reduction solves this by allowing parallel execution:

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

On CPUs with FMA support (Intel Haswell and newer, AMD Piledriver and newer), a significant boost is available. FMA performs `a * b + c` in a single instruction with only one rounding step, which is both faster and more accurate:

```csharp
if (Fma.IsSupported)
    c = Fma.MultiplyAdd(a, b, c);  // c = a * b + c in one instruction
else
    c = Vector256.Add(c, Vector256.Multiply(a, b));
```

The IL kernel system automatically detects FMA availability at runtime and uses it when possible. No special configuration needed—just capable hardware.

### 5. Cache Blocking (MatMul)

Matrix multiplication is where cache blocking becomes critical. Without it, constant fetching from main memory destroys performance. The GEBP (General Block Panel) algorithm ensures the working set fits in cache:

```csharp
// Block sizes tuned for L1=32KB, L2=256KB
const int MC = 64;   // A panel rows
const int KC = 256;  // K depth
const int MR = 8;    // Micro-kernel rows
const int NR = 16;   // Micro-kernel cols (2 vectors)

// Panel packing: A[kc][MR], B[kc][NR] for sequential access
```

These constants were tuned empirically for typical modern CPUs. If you're targeting specialized hardware, you might benefit from different values.

### Why These Numbers?

The specific values like 4x unrolling or 64-element block sizes aren't arbitrary:

- **4x unrolling**: Modern CPUs have 4-8 execution units that can run SIMD operations in parallel. Unrolling 4x saturates these units without excessive code bloat. Going to 8x provides diminishing returns and increases instruction cache pressure.

- **Block sizes (MC=64, KC=256, MR=8, NR=16)**: These are tuned for typical L1/L2 cache sizes. MC×KC should fit in L2 (~256KB), while MR×KC should fit in L1 (~32KB). The micro-kernel dimensions (MR×NR) are chosen to maximize register usage—on x86-64, you have 16 YMM registers, and an 8×16 micro-kernel needs 8 accumulators plus 2 for A and B panels.

If you're running on a system with different cache sizes (e.g., ARM with larger L1, or a server with huge L3), these values could be retuned. For most desktop/laptop CPUs made in the last decade, they work well.

### 6. Early Exit (Boolean Reductions)

For `np.all()` and `np.any()`, scanning the entire array isn't always necessary. When checking whether all elements are true, finding a single false allows immediate termination. The IL kernels exploit this with SIMD-accelerated early exit:

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

This section provides a comprehensive reference of every operation the IL kernel system supports. When you're implementing a new feature or debugging existing code, use these tables to understand what's available and how each operation is implemented.

### How to Read These Tables

Each table shows:
- **Operation**: The NumPy-equivalent operation name
- **SIMD**: The Vector256/Vector128 method or intrinsic used for SIMD acceleration
- **Scalar**: The IL opcode or .NET method used for the scalar fallback path
- **Types/Notes**: Which types are supported or special considerations

If you see "-" in the SIMD column, that operation doesn't have SIMD acceleration—it falls back to scalar loops. This doesn't necessarily mean it's slow; operations like `Sin` and `Cos` call highly optimized `MathF`/`Math` methods that the JIT may vectorize internally on modern .NET versions.

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

Understanding how different types are handled will help you predict performance and avoid surprises. Not all types are created equal when it comes to SIMD optimization.

### The Type Hierarchy

NumSharp supports 12 types, but they fall into natural categories with different performance characteristics:

**SIMD-Friendly Types** (4-8 elements per Vector256):
- `float`, `double`: Full SIMD support including transcendental functions via Vector256 methods
- `int`, `uint`, `long`, `ulong`: Full SIMD for arithmetic and bitwise operations
- `short`, `ushort`, `byte`: SIMD works but with more elements per vector (16-32)

**Limited SIMD Types**:
- `bool`: Used only for comparison results and masking; limited SIMD via byte representation
- `char`: Treated as `ushort` internally; SIMD works but rarely used

**No SIMD Types**:
- `decimal`: 128-bit type with no hardware SIMD support; always uses scalar loops

When you're working with decimal arrays, expect performance roughly 8-16x slower than float/double equivalents. This isn't a NumSharp limitation—it's inherent to the decimal type's complexity.

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

The cache system is what makes IL generation practical. Without caching, every operation would incur the overhead of IL emission and JIT compilation—potentially tens of milliseconds. With caching, you pay this cost once per unique operation type, then get sub-microsecond delegate lookups forever after.

### How Caching Works

When you call `GetContiguousKernel<float>(BinaryOp.Add)`, here's what happens:

1. **Key construction**: A cache key is created: `(BinaryOp.Add, typeof(float))`
2. **Cache lookup**: The `ConcurrentDictionary` checks if this key exists
3. **Cache hit**: If found, return the cached delegate immediately (~50ns)
4. **Cache miss**: Generate IL, JIT-compile, cache, and return (~5-50ms first time)

The cache is global and lives for the application lifetime. Once a kernel is generated, it's never regenerated. This is why warm-up loops are common in benchmarking code—the first iteration pays the JIT tax.

### Cache Key Design Philosophy

Each operation category has its own key structure because different operations need different parameters to fully specify the generated code:

- **Contiguous binary**: Just need `(Type, Operation)` since both arrays are contiguous
- **Mixed-type binary**: Need `(LhsType, RhsType, ResultType, Operation, ExecutionPath)`
- **Axis reductions**: Need `(InputType, AccumulatorType, Operation, InnerAxisContiguous)`

The key principle: the cache key must capture everything that affects the generated IL. If two operations would generate identical IL, they should share a cache entry. If they would generate different IL, they need different keys.

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

Adding a new operation to the IL kernel system might seem daunting at first, but it follows a well-established pattern. This section walks you through the process step by step, with practical guidance at each stage.

### Before You Start

Ask yourself these questions:

1. **Does NumPy have this operation?** If so, study NumPy's behavior carefully—edge cases, type handling, NaN behavior. NumSharp aims for NumPy compatibility.

2. **Does SIMD acceleration make sense?** Operations like `np.sin` don't have direct SIMD intrinsics, so they call `Math.Sin` in a scalar loop. That's fine—don't force SIMD where it doesn't fit.

3. **What types need support?** Most operations should support all 12 NumSharp types. Some (bitwise operations) only make sense for integers.

4. **Are there existing similar operations?** Copy-paste-modify is encouraged. If you're adding `Sinh`, look at how `Sin` is implemented.

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

### Step 6: Test Thoroughly

This is where most bugs are caught. Write comprehensive tests covering:

**Type Coverage:**
- All 12 NumSharp types (Boolean, Byte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Char, Single, Double, Decimal)
- Pay special attention to signed/unsigned differences

**Array Layout Coverage:**
- Contiguous arrays (the fast SIMD path)
- Strided arrays (sliced, transposed)
- Broadcast arrays (scalar × array)

**Edge Cases:**
- Empty arrays (`np.array([])`)
- Single-element arrays
- NaN and Inf values (for float/double)
- Boundary values (min/max of each type)
- Very large arrays (tests SIMD loop correctness)

**NumPy Verification:**
```python
# Run actual NumPy and record the expected output
import numpy as np
arr = np.array([1.0, 2.0, 3.0])
result = np.my_new_op(arr)
print(result)  # Use this as your expected value
```

### Step 7: Benchmark (Optional but Recommended)

If you're adding a performance-critical operation, verify that your IL kernel actually provides speedup:

```csharp
// Simple benchmark pattern
var arr = np.random.rand(10_000_000);

// Warm up (JIT compile)
_ = np.my_new_op(arr);

// Measure
var sw = Stopwatch.StartNew();
for (int i = 0; i < 100; i++)
    _ = np.my_new_op(arr);
Console.WriteLine($"{sw.ElapsedMilliseconds / 100.0} ms per call");
```

Compare against a naive scalar implementation to quantify the speedup. If you're not seeing expected gains, check that your arrays are hitting the SIMD path (contiguous, supported type).

---

## Performance Considerations

This section helps you understand when you'll see the best performance from NumSharp and when to expect limitations. Use this knowledge to write faster code and to set appropriate expectations.

### The Performance Hierarchy

Not all operations achieve the same speedups. Here's a rough hierarchy from fastest to slowest:

1. **Contiguous SIMD operations** (8-16x faster than scalar): `np.add`, `np.multiply`, etc. on contiguous float/double arrays
2. **Scalar broadcast operations** (6-12x faster): `array + 5`, `array * 2.0`
3. **Type-promoting operations** (3-6x faster): `int32_array + float64_array`
4. **Strided operations** (2-3x faster): Operations on sliced/transposed arrays
5. **Reductions** (4-8x faster): `np.sum`, `np.mean`, `np.max` with horizontal SIMD
6. **Scalar-only operations** (1-2x faster): `np.sin`, `np.power` where SIMD isn't available

The "faster than scalar" comparisons are against naive C# loops. Actual speedup depends on CPU, memory bandwidth, and array sizes.

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

IL generation bugs are notoriously difficult to debug. Unlike regular C# code where you get helpful compiler errors, IL generation failures often manifest as cryptic runtime exceptions or—worse—silently wrong results. This section arms you with the tools and techniques to track down these issues.

### The Nature of IL Bugs

Before diving into specific techniques, here are the kinds of bugs commonly encountered:

1. **Stack imbalance**: You pushed more values than you popped (or vice versa). The JIT catches this and throws `InvalidProgramException`.

2. **Type mismatch**: You tried to store a value of the wrong type. For example, you have a `double` on the stack but emit `Stind_I4`. This causes `VerificationException` or corrupted data.

3. **Missing conversions**: You forgot to convert between types. For example, loading a `byte` and comparing with an `int` without `Conv_I4` first.

4. **Wrong opcode**: Using `Shr` when you needed `Shr_Un` for unsigned right shift, or `Div` when you needed `Div_Un`.

5. **Off-by-one in loops**: Your loop bounds are wrong, causing buffer overruns or missed elements.

The catch-all exception handlers in `TryGet*Kernel()` methods are intentional—they let NumSharp gracefully fall back to scalar implementations when IL generation fails. During development, disabling these catches temporarily reveals the actual exceptions.

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

## Common Pitfalls and How to Avoid Them

Over time, contributors have encountered the same IL generation pitfalls repeatedly. Learning from these mistakes will save you debugging time.

### Pitfall 1: Forgetting Conv_I for Pointer Arithmetic

When computing pointer offsets, the index is often a `long` but the element size is an `int`. You need to convert properly:

```csharp
// WRONG: Multiplying long by int can produce wrong results
il.Emit(OpCodes.Ldloc, locI);        // long index
il.Emit(OpCodes.Ldc_I4, elementSize); // int size
il.Emit(OpCodes.Mul);                 // Result type is ambiguous!

// CORRECT: Convert index to native int for pointer arithmetic
il.Emit(OpCodes.Ldloc, locI);        // long index
il.Emit(OpCodes.Conv_I);             // Convert to native int
il.Emit(OpCodes.Ldc_I4, elementSize);
il.Emit(OpCodes.Mul);
il.Emit(OpCodes.Add);                // Add to base pointer
```

### Pitfall 2: Stack Imbalance in Branches

When you have conditional branches, both paths must leave the stack in the same state:

```csharp
// WRONG: Different stack states
il.Emit(OpCodes.Ldloc, locValue);
il.Emit(OpCodes.Brfalse, lblElse);
il.Emit(OpCodes.Ldc_I4_1);
il.Emit(OpCodes.Ldc_I4_2);  // Extra value on stack!
il.Emit(OpCodes.Br, lblEnd);
il.MarkLabel(lblElse);
il.Emit(OpCodes.Ldc_I4_0);
il.MarkLabel(lblEnd);

// CORRECT: Both paths leave one value on stack
il.Emit(OpCodes.Ldloc, locValue);
il.Emit(OpCodes.Brfalse, lblElse);
il.Emit(OpCodes.Ldc_I4_1);
il.Emit(OpCodes.Br, lblEnd);
il.MarkLabel(lblElse);
il.Emit(OpCodes.Ldc_I4_0);
il.MarkLabel(lblEnd);
```

### Pitfall 3: Using Wrong Indirect Load/Store

Each type has specific load/store opcodes. Using the wrong one causes subtle corruption:

```csharp
// For uint: use Ldind_U4, not Ldind_I4
// For ulong: use Ldind_I8 (same as long)
// For float: use Ldind_R4
// For double: use Ldind_R8
```

The helpers `EmitLoadIndirect()` and `EmitStoreIndirect()` handle this correctly—use them instead of emitting opcodes directly.

### Pitfall 4: Forgetting Unsigned Operations

Division and right shift have signed and unsigned variants:

```csharp
// For signed types: Div, Shr
// For unsigned types: Div_Un, Shr_Un

// WRONG: Using Div for uint gives signed semantics
il.Emit(OpCodes.Div);

// CORRECT: Check type and use appropriate opcode
if (IsUnsignedType(type))
    il.Emit(OpCodes.Div_Un);
else
    il.Emit(OpCodes.Div);
```

### Pitfall 5: Not Handling Empty Arrays

Always check for empty arrays at the start of your kernel. Many SIMD operations will fail or produce undefined behavior on zero-length data:

```csharp
// At kernel start, check totalSize and return early
il.Emit(OpCodes.Ldarg, totalSizeArgIndex);
il.Emit(OpCodes.Ldc_I8, 0L);
il.Emit(OpCodes.Ble, lblReturn);  // If totalSize <= 0, return immediately
```

---

## Summary

You've now explored the ILKernelGenerator system, NumSharp's performance backbone. Let's recap what you've learned:

**Core Architecture:**
- The system uses `System.Reflection.Emit.DynamicMethod` to generate specialized kernels at runtime
- 28 partial class files organize the code by operation category
- Kernels are cached by operation key for instant reuse after first generation
- The JIT compiler further optimizes the emitted IL to native machine code

**Optimization Techniques:**
- **SIMD vectorization** with Vector128/256/512 processes 4-32 elements per instruction
- **4x loop unrolling** maximizes instruction-level parallelism
- **Tree reduction** combines accumulator vectors efficiently
- **Cache blocking** (GEBP algorithm) optimizes matrix multiplication
- **Early exit** accelerates boolean reductions like `np.all()` and `np.any()`

**Execution Paths:**
- **SimdFull**: Both arrays contiguous—maximum performance
- **SimdScalarRight/Left**: Broadcast operations—nearly as fast
- **SimdChunk**: Inner dimension contiguous—partial SIMD benefit
- **General**: Arbitrary strides—coordinate-based iteration (slowest)

**Type Support:**
- All 12 NumSharp types are supported (with varying SIMD capabilities)
- Float/double have the best SIMD support
- Decimal requires scalar-only loops

**Practical Guidance:**
- Keep arrays contiguous for best performance
- Watch for common IL pitfalls (stack balance, type conversions, signed/unsigned)
- Test thoroughly across types, layouts, and edge cases
- Benchmark to verify expected speedups

This enables NumSharp to achieve performance competitive with native NumPy while maintaining the safety and productivity of managed .NET code. A call like `np.add(a, b)` invokes machine code specifically optimized for the exact types and array layouts involved—generated in milliseconds, cached forever, and executed in microseconds.
