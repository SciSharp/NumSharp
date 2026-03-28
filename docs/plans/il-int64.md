# IL Kernel Generator: Int64 Migration Plan

## Executive Summary

The ILKernelGenerator system generates high-performance SIMD kernels at runtime using `System.Reflection.Emit`. As part of the NumSharp int64 indexing migration (GitHub Issue #584), all index-related variables, loop counters, and IL emissions must use 64-bit integers (`long`) instead of 32-bit integers (`int`).

**Current State**: Partial migration. Some files (Binary.cs, Shift.cs) are correctly migrated. Most files still use `int` for critical variables, causing inconsistent IL emission mixing `Ldc_I4` (32-bit) and `Ldc_I8` (64-bit) operations.

**Goal**: All loop counters, index offsets, vector counts, and related variables should be `long` from declaration through IL emission.

---

## Table of Contents

1. [Background: Why Int64?](#background-why-int64)
2. [The Problem: Mixed Int32/Int64 IL](#the-problem-mixed-int32int64-il)
3. [IL Fundamentals](#il-fundamentals)
4. [The Correct Pattern](#the-correct-pattern)
5. [The Incorrect Pattern](#the-incorrect-pattern)
6. [Variable Classification](#variable-classification)
7. [File-by-File Analysis](#file-by-file-analysis)
8. [Migration Checklist](#migration-checklist)
9. [Testing Strategy](#testing-strategy)
10. [Appendix: IL Opcode Reference](#appendix-il-opcode-reference)

---

## Background: Why Int64?

### NumPy Alignment

NumPy uses `npy_intp` for all index operations, which is defined as `Py_ssize_t` - a 64-bit signed integer on 64-bit platforms. This allows NumPy to handle arrays larger than 2GB (2,147,483,647 elements).

```c
// numpy/_core/include/numpy/npy_common.h:217
typedef Py_ssize_t npy_intp;
```

### The 2GB Barrier

With 32-bit (`int`) indexing:
- Maximum array size: 2,147,483,647 elements
- For `double` (8 bytes): ~16GB of data
- For `float` (4 bytes): ~8GB of data

This seems large, but modern ML workloads routinely exceed these limits. A single batch of image data or a large embedding matrix can easily surpass 2B elements.

### Performance Impact

Benchmarking shows 1-3% overhead for scalar loops, <1% for SIMD loops. This is acceptable for enabling >2GB array support.

---

## The Problem: Mixed Int32/Int64 IL

### What's Happening

The IL Kernel Generator creates `DynamicMethod` instances that are JIT-compiled to native code. These methods receive `long` parameters (array sizes, counts) but internally use `int` variables for loop control, which creates type mismatches in the generated IL.

### Example of the Problem

Consider this IL generation code from `ILKernelGenerator.Unary.cs`:

```csharp
// C# code generating IL
int vectorCount = GetVectorCount(key.InputType);  // Returns int (e.g., 8)
int unrollStep = vectorCount * 4;                  // 32 (int)

// Emitting IL for: unrollEnd = totalSize - unrollStep
il.Emit(OpCodes.Ldarg_S, (byte)5);  // Load totalSize (long) onto stack
il.Emit(OpCodes.Ldc_I4, unrollStep); // Load 32 as INT32 onto stack
il.Emit(OpCodes.Sub);                // Subtract: long - int32 = ???
il.Emit(OpCodes.Stloc, locUnrollEnd);
```

### What's Wrong

The IL stack now has:
1. `totalSize` - a 64-bit integer (int64)
2. `unrollStep` - a 32-bit integer (int32)

When `Sub` executes, the CLR must reconcile these types. Per ECMA-335, binary numeric operations on mismatched integer types have **undefined behavior** in the specification. While current .NET implementations may widen the int32, this is:

1. **Not guaranteed** by the specification
2. **Inconsistent** with properly migrated code
3. **A maintenance hazard** that could break with future CLR changes

---

## IL Fundamentals

### Integer Loading Opcodes

| Opcode | Loads | Stack Type | Use For |
|--------|-------|------------|---------|
| `Ldc_I4` | 32-bit int | int32 | Small constants, int variables |
| `Ldc_I4_0` | 0 | int32 | Zero (shorthand) |
| `Ldc_I4_1` | 1 | int32 | One (shorthand) |
| `Ldc_I4_M1` | -1 | int32 | Negative one |
| `Ldc_I8` | 64-bit int | int64 | Long constants, long variables |

### Conversion Opcodes

| Opcode | Converts | Notes |
|--------|----------|-------|
| `Conv_I4` | to int32 | May truncate |
| `Conv_I8` | to int64 | Safe widening from int32 |
| `Conv_I` | to native int | Platform-dependent (32 or 64 bit) |
| `Conv_U8` | to uint64 | Unsigned widening |

### Stack Type Rules for Binary Operations

Per ECMA-335 Partition III, binary numeric operations (`Add`, `Sub`, `Mul`, `Div`) require operands to be:
- Both int32, OR
- Both int64, OR
- Both native int, OR
- Both floating-point (F)

Mixing int32 and int64 is **not specified** and relies on implementation behavior.

### Local Variable Types

When declaring locals:
```csharp
var locI = il.DeclareLocal(typeof(long));  // 64-bit local
var locJ = il.DeclareLocal(typeof(int));   // 32-bit local
```

The local's type determines what values can be stored without conversion.

---

## The Fundamental Rule

**The C# variable type is irrelevant. What matters is the IL stack type at the point of operation.**

IL binary operations (`Add`, `Sub`, `Mul`, `Div`) require both operands to be the **same type**:
- Both int32, OR
- Both int64, OR
- Both native int, OR
- Both floating-point

**The Decision Process:**

```
When emitting a value that will be used in IL arithmetic:
  1. What type is ALREADY on the stack? (from Ldloc, Ldarg, or previous operation)
  2. If int64 → emit the new value as Ldc_I8
  3. If int32 → emit the new value as Ldc_I4
  4. After Conv_I → native int, Ldc_I4 is acceptable
```

### Quick Reference Table

| Stack State | Operation | Correct Emission |
|-------------|-----------|------------------|
| `[int64]` (from `Ldloc` of long local) | `+ 1` | `Ldc_I8, 1L` |
| `[int64]` (from `Ldarg` of long param) | `- vectorCount` | `Ldc_I8, (long)vectorCount` |
| `[int64]` (from `Ldloc` of long local) | `+ offset` | `Ldc_I8, (long)offset` |
| `[int64]` (from `Ldloc` of long local) | `+ j` (C# loop var) | `Ldc_I8, (long)j` |
| `[int32]` (from `Ldloc` of int local) | `- 1` | `Ldc_I4_1` |
| `[native int]` (after `Conv_I`) | `* elementSize` | `Ldc_I4, elementSize` |
| `[uint32]` (bit mask) | `>> j` | `Ldc_I4, j` (shifts take int32) |

### The Key Insight

C# compile-time loops generate IL, but the **loop variable's C# type doesn't determine the emitted opcode**:

```csharp
// C# loop with int variable
for (int j = 0; j < vectorCount; j++)
{
    il.Emit(OpCodes.Ldloc, locI);      // Stack: [int64]
    if (j > 0)
    {
        il.Emit(OpCodes.Ldc_I8, (long)j);  // Must be int64 to match!
        il.Emit(OpCodes.Add);              // int64 + int64 = OK
    }
}
```

Even though `j` is a C# `int`, when emitted for addition with an `int64` stack value, it **must** be `Ldc_I8`.

---

## IL Local Types

The IL local's declared type determines the stack type after `Ldloc`:

```csharp
var locI = il.DeclareLocal(typeof(long));   // Ldloc → int64 on stack
var locD = il.DeclareLocal(typeof(int));    // Ldloc → int32 on stack
```

**Rule**: If a local is `typeof(long)`, ALL arithmetic with it must use `Ldc_I8`.

---

## Correct Patterns

### Pattern 1: Long Index Loops

```csharp
var locI = il.DeclareLocal(typeof(long));

// Initialize: i = 0
il.Emit(OpCodes.Ldc_I8, 0L);
il.Emit(OpCodes.Stloc, locI);

// Increment: i++
il.Emit(OpCodes.Ldloc, locI);     // Stack: [int64]
il.Emit(OpCodes.Ldc_I8, 1L);      // Stack: [int64, int64]
il.Emit(OpCodes.Add);             // int64 + int64 = OK
il.Emit(OpCodes.Stloc, locI);

// Increment by step: i += vectorCount
il.Emit(OpCodes.Ldloc, locI);     // Stack: [int64]
il.Emit(OpCodes.Ldc_I8, (long)vectorCount);  // Stack: [int64, int64]
il.Emit(OpCodes.Add);             // int64 + int64 = OK
il.Emit(OpCodes.Stloc, locI);
```

### Pattern 2: Loop Bounds from Long Parameters

```csharp
// vectorEnd = totalSize - vectorCount
il.Emit(OpCodes.Ldarg_S, (byte)7);  // totalSize (long param) → Stack: [int64]
il.Emit(OpCodes.Ldc_I8, (long)vectorCount);  // Stack: [int64, int64]
il.Emit(OpCodes.Sub);               // int64 - int64 = OK
il.Emit(OpCodes.Stloc, locVectorEnd);
```

### Pattern 3: Pointer Arithmetic (After Conv_I)

```csharp
// ptr + i * elementSize
il.Emit(OpCodes.Ldarg_0);         // ptr
il.Emit(OpCodes.Ldloc, locI);     // Stack: [ptr, int64]
il.Emit(OpCodes.Conv_I);          // Stack: [ptr, native int]
il.Emit(OpCodes.Ldc_I4, elementSize);  // OK - native int accepts int32
il.Emit(OpCodes.Mul);             // native int * int32 → native int
il.Emit(OpCodes.Add);             // ptr + native int = OK
```

### Pattern 4: Int Dimension Loops

```csharp
var locD = il.DeclareLocal(typeof(int));

// d = ndim - 1
il.Emit(OpCodes.Ldarg_S, (byte)4);  // ndim (int param) → Stack: [int32]
il.Emit(OpCodes.Ldc_I4_1);          // Stack: [int32, int32]
il.Emit(OpCodes.Sub);               // int32 - int32 = OK
il.Emit(OpCodes.Stloc, locD);

// d--
il.Emit(OpCodes.Ldloc, locD);     // Stack: [int32]
il.Emit(OpCodes.Ldc_I4_1);        // Stack: [int32, int32]
il.Emit(OpCodes.Sub);             // int32 - int32 = OK
il.Emit(OpCodes.Stloc, locD);
```

### Pattern 5: C# Unroll Loop Emitting Offsets

```csharp
long vectorCount = GetVectorCount(type);  // Widen for convenience

for (int n = 0; n < 4; n++)  // C# loop - int is fine
{
    long offset = n * vectorCount;  // C# arithmetic - result is long

    il.Emit(OpCodes.Ldloc, locI);   // Stack: [int64]
    if (offset > 0)
    {
        il.Emit(OpCodes.Ldc_I8, offset);  // Stack: [int64, int64]
        il.Emit(OpCodes.Add);             // int64 + int64 = OK
    }
}
```

### Pattern 6: Bit Operations (Stay Int)

```csharp
// (bits >> j) & 1  -- bit manipulation uses int32
il.Emit(OpCodes.Ldloc, locBits);  // Stack: [uint32]
il.Emit(OpCodes.Ldc_I4, j);       // Shift amount - always int32
il.Emit(OpCodes.Shr_Un);
il.Emit(OpCodes.Ldc_I4_1);        // Mask - int32
il.Emit(OpCodes.And);
```

---

## Incorrect Patterns

### Wrong: Ldc_I4 with Long Stack

```csharp
il.Emit(OpCodes.Ldloc, locI);     // Stack: [int64]
il.Emit(OpCodes.Ldc_I4_1);        // Stack: [int64, int32] ← MISMATCH!
il.Emit(OpCodes.Add);             // UNDEFINED BEHAVIOR
```

### Wrong: Forgetting C# Loop Variable Needs Cast

```csharp
for (int j = 0; j < vectorCount; j++)
{
    il.Emit(OpCodes.Ldloc, locI);   // Stack: [int64]
    il.Emit(OpCodes.Ldc_I4, j);     // Stack: [int64, int32] ← WRONG!
    il.Emit(OpCodes.Add);
}
```

**Fix:**
```csharp
    il.Emit(OpCodes.Ldc_I8, (long)j);  // Stack: [int64, int64] ← CORRECT
```

---

## C# Variable Type Recommendations

The C# variable type doesn't affect IL correctness, but it affects code clarity:

| Approach | Pros | Cons |
|----------|------|------|
| `int` + cast at emission | Matches API return types | Easy to forget casts |
| `long` from start | No casts needed, self-documenting | Implicit widening may surprise |

**Recommendation**: Use `long` for variables that will be emitted to IL for index arithmetic. Use `int` for variables that stay in C# (loop bounds for code generation, API parameters).

```csharp
// Recommended style
long vectorCount = GetVectorCount(type);  // Will be emitted as Ldc_I8
long unrollStep = vectorCount * 4;         // C# arithmetic stays long
int unrollFactor = 4;                       // C# constant, not emitted directly

for (int n = 0; n < unrollFactor; n++)     // C# loop control
{
    long offset = n * vectorCount;          // Will be emitted as Ldc_I8
    // ...
}
```

---

## File-by-File Analysis

### Status Legend

- **CORRECT**: Properly uses `Ldc_I8` and/or `long` variables
- **PARTIAL**: Some correct, some incorrect patterns
- **INCORRECT**: Uses `Ldc_I4` without conversion for index values

---

### ILKernelGenerator.Binary.cs

**Status**: CORRECT (uses cast pattern)

**Current Implementation**:
```csharp
int vectorCount = GetVectorCount<T>();
int unrollStep = vectorCount * 4;
// ...
il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
il.Emit(OpCodes.Ldc_I8, (long)unrollStep);
il.Emit(OpCodes.Ldc_I8, 1L);
```

**Recommendation**: Change to `long vectorCount` for consistency, but functionally correct as-is.

**Lines of Interest**:
- Line 217: `int vectorCount = GetVectorCount<T>();`
- Line 218: `int unrollStep = vectorCount * 4;`
- Line 243: `int offset = vectorCount * u;`
- Line 222: `il.Emit(OpCodes.Ldc_I8, (long)vectorCount);`

---

### ILKernelGenerator.Shift.cs

**Status**: CORRECT (uses cast pattern)

**Current Implementation**:
```csharp
int vectorCount = GetShiftVectorCount<T>();
int unrollStep = vectorCount * 4;
// ...
il.Emit(OpCodes.Ldc_I8, (long)vectorCount);
il.Emit(OpCodes.Ldc_I8, (long)unrollStep);
il.Emit(OpCodes.Ldc_I8, 1L);
```

**Recommendation**: Change to `long vectorCount` for consistency.

**Lines of Interest**:
- Line 163-164: Variable declarations
- Line 219, 225: Correct `Ldc_I8` usage
- Line 340, 398, 534: Correct `Ldc_I8, 1L` usage

---

### ILKernelGenerator.Unary.cs

**Status**: INCORRECT

**Issues Found**:

1. **Line 254-256**: Variables declared as `int`
   ```csharp
   int vectorCount = GetVectorCount(key.InputType);
   int unrollFactor = 4;
   int unrollStep = vectorCount * unrollFactor;
   ```

2. **Line 271**: `Ldc_I4` for unrollStep
   ```csharp
   il.Emit(OpCodes.Ldc_I4, unrollStep);  // WRONG
   ```

3. **Line 277**: `Ldc_I4` for vectorCount
   ```csharp
   il.Emit(OpCodes.Ldc_I4, vectorCount);  // WRONG
   ```

4. **Line 296**: `int offset`
   ```csharp
   int offset = n * vectorCount;  // Should be long
   ```

5. **Lines 303, 320**: `Ldc_I4` for offset
   ```csharp
   il.Emit(OpCodes.Ldc_I4, offset);  // WRONG
   ```

6. **Lines 331-332**: `Ldc_I4` for unrollStep increment
   ```csharp
   il.Emit(OpCodes.Ldc_I4, unrollStep);  // WRONG
   il.Emit(OpCodes.Add);
   ```

7. **Lines 369-370**: `Ldc_I4` for vectorCount increment
   ```csharp
   il.Emit(OpCodes.Ldc_I4, vectorCount);  // WRONG
   il.Emit(OpCodes.Add);
   ```

8. **Lines 416-417, 487-488, 543-544, 594-595, 635-636**: `Ldc_I4_1` for increment
   ```csharp
   il.Emit(OpCodes.Ldc_I4_1);  // WRONG
   il.Emit(OpCodes.Add);
   ```

**Required Changes**:
```csharp
// Change declarations
long vectorCount = GetVectorCount(key.InputType);
int unrollFactor = 4;  // This can stay int
long unrollStep = vectorCount * unrollFactor;

// In unroll loop
long offset = n * vectorCount;  // n is int, result is long

// All emissions
il.Emit(OpCodes.Ldc_I8, vectorCount);
il.Emit(OpCodes.Ldc_I8, unrollStep);
il.Emit(OpCodes.Ldc_I8, offset);
il.Emit(OpCodes.Ldc_I8, 1L);  // Instead of Ldc_I4_1
```

---

### ILKernelGenerator.MixedType.cs

**Status**: INCORRECT

**Issues Found**:

1. **Multiple function declarations**: `int vectorCount`, `int unrollStep`

2. **EmitMixedTypeSimdLoop** (around line 445):
   - Line 457-458: `Ldc_I4, vectorCount` without Conv_I8
   - Line 463-464: `Ldc_I4, unrollStep` without Conv_I8
   - Lines 489-491, 503-505, 520-522: `Ldc_I4, offset` without Conv_I8
   - Line 532-533: `Ldc_I4, unrollStep` for increment
   - Line 579-580: `Ldc_I4, vectorCount` for increment

3. **EmitMixedTypeScalarBroadcastLoop** (around line 638):
   - Line 624-625: `Ldc_I4_1` for increment
   - Line 687-688: `Ldc_I4_1` for increment

4. **EmitMixedTypeSimdBroadcastLoop** (around line 772):
   - Line 795-796: `Ldc_I4, vectorCount`
   - Line 837-838: `Ldc_I4, vectorCount` for increment
   - Line 886-887: `Ldc_I4_1` for increment

5. **EmitMixedTypeSimdScalarLoop** (around line 908):
   - Line 931-932: `Ldc_I4, vectorCount`
   - Line 973-974: `Ldc_I4, vectorCount` for increment
   - Line 1022-1023: `Ldc_I4_1` for increment

6. **EmitMixedTypeStridedLoop** (around line 1053):
   - Lines 1094-1095, 1159-1160: `Ldc_I4_1` for d-- (dimension counter - OK as int)
   - Line 1203-1204: `Ldc_I4_1` for i++ - WRONG (i is long)

**Required Changes**: Same pattern as Unary.cs - change variable declarations to `long` and use `Ldc_I8`.

---

### ILKernelGenerator.Comparison.cs

**Status**: INCORRECT

**Issues Found**:

1. **EmitComparisonSimdLoop** (around line 310):
   - Line 319: `var locI = il.DeclareLocal(typeof(long));` - CORRECT
   - Lines 332, 338: `Ldc_I4, unrollStep-1` / `Ldc_I4, vectorCount-1`
   - Lines 370-371, 384-385: `Ldc_I4, offset` + Add (no Conv_I8)
   - Lines 408-409: `Ldc_I4, offset` + Conv_I8 - PARTIALLY CORRECT
   - Lines 419-420: `Ldc_I4, unrollStep` + Add
   - Lines 458-459: `Ldc_I4, vectorCount` + Add

2. **Scalar loops**:
   - Lines 507-508, 850-851, 915-916, 956-957: `Ldc_I4_1` for increment

3. **Some correct patterns exist**:
   - Lines 343-345: `Ldc_I4_0; Conv_I8; Stloc` - CORRECT for initialization
   - Lines 669, 731, 793: `Ldc_I8, 1L` - CORRECT

**Note**: This file is inconsistent - some patterns correct, others not.

---

### ILKernelGenerator.MatMul.cs

**Status**: PARTIAL (float path mostly correct, double path incorrect)

**Issues Found**:

1. **EmitMatMulFloat** (line 131):
   - Line 145: `int vectorCount = Vector256<float>.Count;`
   - Lines 174-176: `Ldc_I4_0; Conv_I8; Stloc` - CORRECT
   - Lines 196-198: `Ldc_I8, 1L` - CORRECT
   - Lines 205-207: `Ldc_I4, vectorCount; Conv_I8` - CORRECT
   - Line 291: `Ldc_I8, (long)vectorCount` - CORRECT
   - Lines 335, 344, 353: `Ldc_I8, 1L` - CORRECT

2. **EmitMatMulDouble** (line 443):
   - Line 456: `int vectorCount = Vector256<double>.Count;`
   - Lines 479-481: `Ldc_I4_0; Conv_I8` - CORRECT
   - Lines 497-500: `Ldc_I4_1; Add` - WRONG (should be Ldc_I8, 1L)
   - Lines 507-508: `Ldc_I4, vectorCount; Sub` - WRONG (no Conv_I8)
   - Lines 511-513: `Ldc_I4_0; Stloc` - WRONG (should have Conv_I8)
   - Lines 543-544: `Ldc_I4_0; Stloc` - WRONG
   - Lines 573-574: `Ldc_I4_0; Stloc` - WRONG
   - Lines 583-585: `Ldc_I4, vectorCount; Add` - WRONG
   - Lines 617-620, 626-628, 635-637: `Ldc_I4_1; Add` - WRONG

**Analysis**: The float path was migrated correctly. The double path was not migrated.

---

### ILKernelGenerator.Reduction.cs

**Status**: INCORRECT

**Issues Found**:

1. **EmitReductionSimdLoop** (around line 170):
   - Line 180: `var locI = il.DeclareLocal(typeof(long));` - CORRECT
   - Lines 209-211, 215-217: `Ldc_I4, unrollStep/vectorCount` without Conv_I8
   - Lines 247-248, 287-288, 339-340: `Ldc_I4, vectorCount/unrollStep` for increment
   - Line 378-379: `Ldc_I4_1` for increment

2. **EmitArgMaxMinSimdLoop** (around line 430):
   - Lines 440, 493: Uses helper methods (correct)

3. **EmitReductionStridedLoop** (around line 518):
   - Lines 565-566, 616-617: `Ldc_I4_1` for dimension decrement (OK - d is int)
   - But other increments may be wrong

---

### ILKernelGenerator.Reduction.NaN.cs

**Status**: PARTIAL

**Issues Found**:

1. **Some correct patterns**:
   - Lines 202-203: `Ldc_I4, vectorCount; Conv_I8` - CORRECT
   - Lines 258-259: `Ldc_I4, vectorCount; Conv_I8` - CORRECT
   - Lines 331, 483, 651, 821: `Ldc_I8, 1L` - CORRECT

2. **Incorrect patterns**:
   - Lines 425-427: `Ldc_I4, vectorCount; Conv_I8` then Add - CORRECT
   - Lines 601-603: Same - CORRECT
   - Lines 726-727, 775-776: `Ldc_I4_1` for dimension decrement - OK (d is int)

**Analysis**: This file is mostly correct. The `Ldc_I4` + `Conv_I8` pattern works, though `Ldc_I8` would be cleaner.

---

### ILKernelGenerator.Scan.cs

**Status**: INCORRECT

**Issues Found**:

1. **EmitCumSumContiguousLoop** (around line 195):
   - Line 205: `var locI = il.DeclareLocal(typeof(long));` - CORRECT
   - Line 253-254: `Ldc_I4_1; Add` - WRONG

2. **EmitCumSumStridedLoop** (around line 268):
   - Lines 306-307, 357-358: `Ldc_I4_1; Sub` for dimension decrement - OK (d is int)
   - Line 391-392: `Ldc_I4_1; Add` for i++ - WRONG

---

### ILKernelGenerator.Masking.VarStd.cs

**Status**: INCORRECT

**Issues Found**:

Approximately 20 occurrences of:
```csharp
int vectorCount = Vector512<double>.Count;  // or 256, 128
```

These should be `long vectorCount`.

---

### ILKernelGenerator.Reduction.Axis.cs

**Status**: Needs Review

This file uses helper methods and C# loops rather than IL emission for most logic. The axis parameter and dimension indices correctly stay as `int`.

---

### ILKernelGenerator.Reduction.Axis.Simd.cs

**Status**: PARTIAL

**Correct patterns found**:
- Line 237: `long unrollStep = vectorCount * 4;` - CORRECT
- Line 295: `long unrollStep = vectorCount * 4;` - CORRECT

But `vectorCount` itself is likely `int`. Should verify and change to `long`.

---

### ILKernelGenerator.Clip.cs

**Status**: Needs Review

Contains C# helper methods, not IL emission. Variables like `offset` in:
```csharp
long offset = shape.TransformOffset(i);
```
are correctly `long`.

---

### ILKernelGenerator.Modf.cs

**Status**: Needs Review

Contains C# helper methods with loop variable `long i`. Likely correct.

---

### Other Files

The following files need review for the same patterns:

- `ILKernelGenerator.Masking.cs`
- `ILKernelGenerator.Masking.Boolean.cs`
- `ILKernelGenerator.Masking.NaN.cs`
- `ILKernelGenerator.Reduction.Boolean.cs`
- `ILKernelGenerator.Reduction.Arg.cs`
- `ILKernelGenerator.Reduction.Axis.Arg.cs`
- `ILKernelGenerator.Reduction.Axis.VarStd.cs`
- `ILKernelGenerator.Reduction.Axis.NaN.cs`
- `ILKernelGenerator.Unary.Math.cs`
- `ILKernelGenerator.Unary.Decimal.cs`
- `ILKernelGenerator.Unary.Predicate.cs`
- `ILKernelGenerator.Unary.Vector.cs`
- `ILKernelGenerator.Scalar.cs`

---

## Migration Checklist

### For Each IL-Generating Method

#### Step 1: Identify Variable Declarations

Find all declarations of:
- `vectorCount`
- `unrollStep`
- `offset`
- Any variable used in index arithmetic

#### Step 2: Change Types

```csharp
// Before
int vectorCount = GetVectorCount<T>();
int unrollStep = vectorCount * 4;
int offset = n * vectorCount;

// After
long vectorCount = GetVectorCount<T>();
long unrollStep = vectorCount * 4;
long offset = n * vectorCount;
```

#### Step 3: Update Loop Bound Calculations

```csharp
// Before
il.Emit(OpCodes.Ldarg_3);            // totalSize (long)
il.Emit(OpCodes.Ldc_I4, vectorCount);
il.Emit(OpCodes.Sub);

// After
il.Emit(OpCodes.Ldarg_3);            // totalSize (long)
il.Emit(OpCodes.Ldc_I8, vectorCount); // Now long, no cast needed
il.Emit(OpCodes.Sub);
```

#### Step 4: Update Increment Operations

```csharp
// Before (WRONG)
il.Emit(OpCodes.Ldloc, locI);
il.Emit(OpCodes.Ldc_I4, vectorCount);
il.Emit(OpCodes.Add);
il.Emit(OpCodes.Stloc, locI);

// After (CORRECT)
il.Emit(OpCodes.Ldloc, locI);
il.Emit(OpCodes.Ldc_I8, vectorCount);
il.Emit(OpCodes.Add);
il.Emit(OpCodes.Stloc, locI);
```

#### Step 5: Update Single-Step Increments

```csharp
// Before (WRONG)
il.Emit(OpCodes.Ldloc, locI);
il.Emit(OpCodes.Ldc_I4_1);
il.Emit(OpCodes.Add);
il.Emit(OpCodes.Stloc, locI);

// After (CORRECT)
il.Emit(OpCodes.Ldloc, locI);
il.Emit(OpCodes.Ldc_I8, 1L);
il.Emit(OpCodes.Add);
il.Emit(OpCodes.Stloc, locI);
```

#### Step 6: Update Offset Calculations in Unrolled Loops

```csharp
// Before
for (int n = 0; n < 4; n++)
{
    int offset = n * vectorCount;  // int
    // ...
    il.Emit(OpCodes.Ldc_I4, offset);
}

// After
for (int n = 0; n < 4; n++)  // n stays int
{
    long offset = n * vectorCount;  // now long
    // ...
    il.Emit(OpCodes.Ldc_I8, offset);
}
```

#### Step 7: Verify Loop Initialization

```csharp
// CORRECT: Ldc_I8 directly - preferred, single instruction
il.Emit(OpCodes.Ldc_I8, 0L);
il.Emit(OpCodes.Stloc, locI);
```

**Do NOT use** `Ldc_I4_0; Conv_I8` - it wastes an instruction. Always use `Ldc_I8` directly for long values.

---

## Testing Strategy

### Unit Tests

Existing tests should pass after migration. Key test categories:

1. **Large array tests**: Arrays approaching int.MaxValue elements
2. **SIMD path tests**: Verify SIMD kernels produce correct results
3. **Scalar tail tests**: Small arrays that don't fill a vector
4. **Strided array tests**: Non-contiguous memory access
5. **Type promotion tests**: Mixed-type operations

### Manual Verification

For each migrated file:

1. Build the project
2. Run tests: `dotnet test -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"`
3. Check for any new test failures

### IL Verification (Optional)

Use IL disassembly to verify generated code:

```csharp
// Add to test code temporarily
var kernel = ILKernelGenerator.GetContiguousKernel<float>(BinaryOp.Add);
// Use reflection to get the DynamicMethod and inspect IL
```

---

## Appendix: IL Opcode Reference

### Integer Load Operations

| Opcode | Operand | Stack Result | Notes |
|--------|---------|--------------|-------|
| `Ldc_I4` | int32 value | int32 | 32-bit constant |
| `Ldc_I4_S` | int8 value | int32 | Short form (-128 to 127) |
| `Ldc_I4_0` | none | int32 | Push 0 |
| `Ldc_I4_1` | none | int32 | Push 1 |
| `Ldc_I4_2` | none | int32 | Push 2 |
| `Ldc_I4_3` | none | int32 | Push 3 |
| `Ldc_I4_4` | none | int32 | Push 4 |
| `Ldc_I4_5` | none | int32 | Push 5 |
| `Ldc_I4_6` | none | int32 | Push 6 |
| `Ldc_I4_7` | none | int32 | Push 7 |
| `Ldc_I4_8` | none | int32 | Push 8 |
| `Ldc_I4_M1` | none | int32 | Push -1 |
| `Ldc_I8` | int64 value | int64 | 64-bit constant |

### Conversion Operations

| Opcode | Input | Output | Notes |
|--------|-------|--------|-------|
| `Conv_I1` | any int | int8 | Truncate to signed byte |
| `Conv_I2` | any int | int16 | Truncate to signed short |
| `Conv_I4` | any int | int32 | Truncate to signed int |
| `Conv_I8` | any int | int64 | Sign-extend to long |
| `Conv_U1` | any int | uint8 | Truncate to unsigned byte |
| `Conv_U2` | any int | uint16 | Truncate to unsigned short |
| `Conv_U4` | any int | uint32 | Truncate to unsigned int |
| `Conv_U8` | any int | uint64 | Zero-extend to ulong |
| `Conv_I` | any int | native int | Platform-dependent |
| `Conv_U` | any int | native uint | Platform-dependent |

### Binary Arithmetic

| Opcode | Operation | Notes |
|--------|-----------|-------|
| `Add` | a + b | Both operands must be same type |
| `Sub` | a - b | Both operands must be same type |
| `Mul` | a * b | Both operands must be same type |
| `Div` | a / b | Signed division |
| `Div_Un` | a / b | Unsigned division |
| `Rem` | a % b | Signed remainder |
| `Rem_Un` | a % b | Unsigned remainder |

### Local Variable Operations

| Opcode | Operation | Notes |
|--------|-----------|-------|
| `Ldloc` | Load local | Push local variable value |
| `Ldloc_S` | Load local (short) | Index 0-255 |
| `Ldloc_0` | Load local 0 | Shorthand |
| `Stloc` | Store local | Pop and store to local |
| `Stloc_S` | Store local (short) | Index 0-255 |
| `Stloc_0` | Store local 0 | Shorthand |

### Argument Operations

| Opcode | Operation | Notes |
|--------|-----------|-------|
| `Ldarg` | Load argument | Push argument value |
| `Ldarg_S` | Load argument (short) | Index 0-255 |
| `Ldarg_0` | Load arg 0 | Shorthand |
| `Ldarg_1` | Load arg 1 | Shorthand |
| `Ldarg_2` | Load arg 2 | Shorthand |
| `Ldarg_3` | Load arg 3 | Shorthand |

---

## Summary of Changes Required

### High Priority (IL Emission Issues)

| File | Estimated Changes | Complexity |
|------|-------------------|------------|
| ILKernelGenerator.Unary.cs | ~15 locations | Medium |
| ILKernelGenerator.MixedType.cs | ~25 locations | High |
| ILKernelGenerator.Comparison.cs | ~15 locations | Medium |
| ILKernelGenerator.MatMul.cs (double path) | ~10 locations | Medium |
| ILKernelGenerator.Reduction.cs | ~10 locations | Medium |
| ILKernelGenerator.Scan.cs | ~5 locations | Low |

### Medium Priority (Variable Declarations)

| File | Changes |
|------|---------|
| ILKernelGenerator.Binary.cs | Change `int` to `long` for consistency |
| ILKernelGenerator.Shift.cs | Change `int` to `long` for consistency |
| ILKernelGenerator.Masking.VarStd.cs | ~20 `int vectorCount` declarations |

### Low Priority (Review Needed)

- All other ILKernelGenerator partial files
- Helper method files that use C# loops (likely already correct)

---

## Conclusion

The int64 migration in ILKernelGenerator is partially complete. The core issue is that C# variables like `vectorCount`, `unrollStep`, and `offset` are declared as `int`, leading to `Ldc_I4` IL emissions that create type mismatches with `long` loop counters and array sizes.

The fix is straightforward:
1. Declare these variables as `long`
2. Use `Ldc_I8` for all emissions
3. Replace `Ldc_I4_1` with `Ldc_I8, 1L` for increments

This ensures type consistency throughout the generated IL and aligns with the project's int64 indexing goals.
