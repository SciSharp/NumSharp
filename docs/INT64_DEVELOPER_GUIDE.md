# Int64 Indexing Migration - Developer Guide

This guide provides patterns and rules for developers continuing the int32 to int64 indexing migration.

---

## Core Principle

**Think before casting.** The goal is to use `long` everywhere indices, sizes, strides, and offsets are involved. Only cast to `int` when absolutely required by external APIs.

---

## Decision Tree: Should This Be `long`?

```
Is it an index, size, stride, offset, or count?
├── YES → Use `long`
│   └── Exception: Does external API require int?
│       ├── YES → Cast at the boundary, document why
│       └── NO → Keep as `long`
└── NO → Keep original type
```

---

## Pattern 1: Loop Counters Over Array Elements

**WRONG:**
```csharp
for (int i = 0; i < array.size; i++)  // size is now long
    Process(array[i]);
```

**CORRECT:**
```csharp
for (long i = 0; i < array.size; i++)
    Process(array[i]);
```

**Rule:** If iterating over array indices, use `long` loop counter.

---

## Pattern 2: Coordinate Arrays

**WRONG:**
```csharp
var coords = new int[2];
coords[0] = i;  // i is long
coords[1] = j;  // j is long
array.GetValue(coords);  // GetValue now takes long[]
```

**CORRECT:**
```csharp
var coords = new long[2];
coords[0] = i;
coords[1] = j;
array.GetValue(coords);
```

**Rule:** Coordinate arrays are `long[]`, not `int[]`.

---

## Pattern 3: Matrix Dimensions (M, K, N)

**WRONG:**
```csharp
int M = (int)left.shape[0];   // Defeats the purpose!
int K = (int)left.shape[1];
int N = (int)right.shape[1];
```

**CORRECT:**
```csharp
long M = left.shape[0];   // shape[] returns long
long K = left.shape[1];
long N = right.shape[1];
```

**Rule:** Matrix dimensions are `long`. They come from shape which is now `long[]`.

---

## Pattern 4: Pointer Arithmetic (Works Naturally)

Pointer arithmetic already supports `long` offsets:

```csharp
T* ptr = (T*)Address;
long offset = 3_000_000_000L;
T value = ptr[offset];  // OK! Pointer indexing accepts long
```

**Rule:** Pointer arithmetic is already correct. Focus on the index variables.

---

## Pattern 5: Method Signatures

When updating method signatures, change ALL index-related parameters:

**BEFORE:**
```csharp
private static void MatMulCore<T>(NDArray left, NDArray right, T* result, int M, int K, int N)
```

**AFTER:**
```csharp
private static void MatMulCore<T>(NDArray left, NDArray right, T* result, long M, long K, long N)
```

**Rule:** Update the signature AND all callers simultaneously.

---

## Pattern 6: Unsafe Pointer Parameters

**BEFORE:**
```csharp
public static unsafe bool IsContiguous(int* strides, int* shape, int ndim)
```

**AFTER:**
```csharp
public static unsafe bool IsContiguous(long* strides, long* shape, int ndim)
```

**Note:** `ndim` stays `int` (max ~32 dimensions).

---

## Pattern 7: Local Variables in Algorithms

**BEFORE:**
```csharp
int expectedStride = 1;
for (int d = ndim - 1; d >= 0; d--)
{
    expectedStride *= shape[d];  // shape[d] is now long
}
```

**AFTER:**
```csharp
long expectedStride = 1;
for (int d = ndim - 1; d >= 0; d--)  // d stays int (dimension index)
{
    expectedStride *= shape[d];
}
```

**Rule:** Variables that accumulate products of dimensions must be `long`. Dimension indices (`d`) can stay `int`.

---

## Valid Exceptions: When int Cast IS Correct

### 1. Span<T> Operations

Span has hard `int` limitation:

```csharp
if (Count > int.MaxValue)
    throw new InvalidOperationException("Storage size exceeds Span<T> maximum.");
return new Span<T>(Address, (int)Count);
```

### 2. Managed Array Allocation

.NET arrays limited to int indexing:

```csharp
if (size > int.MaxValue)
    throw new InvalidOperationException("Cannot allocate managed array exceeding int.MaxValue.");
var array = new T[(int)size];
```

### 3. Algorithm Complexity Constraints

When O(n*m) complexity makes large arrays impractical anyway:

```csharp
// Convolution is O(na * nv), so practical limits are well under int.MaxValue
int na = (int)a.size;
int nv = (int)v.size;
```

**Document these exceptions with comments explaining why the cast is safe.**

---

## What Stays `int`

| Item | Reason |
|------|--------|
| `ndim` | Maximum ~32 dimensions |
| `Slice.Start/Stop/Step` | Python slice semantics |
| Dimension indices (`d` in loops) | Iterating over dimensions, not elements |
| `NPTypeCode` values | Small enum |
| Vector lane counts | Hardware-limited |

---

## Checklist for Each File

When migrating a file:

1. [ ] **Find all `int` variables** related to indices/sizes/strides/offsets
2. [ ] **Change to `long`** unless exception applies
3. [ ] **Update method signatures** if parameters are index-related
4. [ ] **Update callers** of changed methods
5. [ ] **Check loop counters** iterating over array elements
6. [ ] **Check coordinate arrays** - must be `long[]`
7. [ ] **Check pointer params** - `int*` → `long*` for strides/shapes
8. [ ] **Add overflow checks** where external APIs require `int`
9. [ ] **Document exceptions** with comments

---

## Common Error Patterns

### Error: Cannot convert long to int

```
error CS0266: Cannot implicitly convert type 'long' to 'int'
```

**Fix:** Change the receiving variable to `long`, OR if external API requires `int`, add explicit cast with overflow check.

### Error: Argument type mismatch

```
error CS1503: Argument 1: cannot convert from 'int[]' to 'long[]'
```

**Fix:** Change the array type at declaration site to `long[]`.

### Error: Iterator type mismatch

```
error CS0029: Cannot implicitly convert type 'int' to 'long' in foreach
```

**Fix:** Check if the enumerated collection now yields `long`. Update the loop variable type.

---

## File Categories and Priority

### Priority 1: Core Types (Done)
- Shape.cs - dimensions, strides, offset, size
- IArraySlice.cs - index parameters
- UnmanagedStorage.cs - Count field
- UnmanagedStorage.Getters.cs - index parameters
- UnmanagedStorage.Setters.cs - index parameters

### Priority 2: Supporting Infrastructure (In Progress)
- ArraySlice.cs / ArraySlice`1.cs - Allocate count, index operations
- Incrementors (6 files) - coordinate arrays
- StrideDetector.cs - pointer parameters

### Priority 3: IL Kernel System (Major Effort)
- IKernelProvider.cs - interface
- ILKernelGenerator.*.cs (13 files) - IL emission, delegate signatures
- SimdKernels.cs, SimdMatMul.cs - SIMD helpers

### Priority 4: DefaultEngine Operations
- Default.Clip.cs, Default.ATan2.cs
- Default.Reduction.*.cs
- Default.NonZero.cs, Default.Transpose.cs

### Priority 5: API Functions
- np.*.cs files
- NDArray.*.cs files

---

## Testing Strategy

After each batch of changes:

1. **Build** - Fix all compilation errors
2. **Run tests** - `dotnet test -- --treenode-filter "/*/*/*/*[Category!=OpenBugs]"`
3. **Check for regressions** - Compare output with NumPy

---

## Git Commit Guidelines

Commit in logical batches with descriptive messages:

```
int64 indexing: <component> <what changed>

- <specific change 1>
- <specific change 2>
- <specific change 3>
```

Example:
```
int64 indexing: StrideDetector pointer params int* -> long*

- IsContiguous: int* strides/shape -> long* strides/shape
- IsScalar: int* strides -> long* strides
- CanSimdChunk: int* params -> long*, innerSize/lhsInner/rhsInner -> long
- Classify: int* params -> long*
- expectedStride local -> long
```

---

## Quick Reference

| Old | New | Notes |
|-----|-----|-------|
| `int size` | `long size` | Array/storage size |
| `int offset` | `long offset` | Memory offset |
| `int[] dimensions` | `long[] dimensions` | Shape dimensions |
| `int[] strides` | `long[] strides` | Memory strides |
| `int[] coords` | `long[] coords` | Index coordinates |
| `int* shape` | `long* shape` | Unsafe pointer |
| `int* strides` | `long* strides` | Unsafe pointer |
| `for (int i` | `for (long i` | Element iteration |
| `int M, K, N` | `long M, K, N` | Matrix dimensions |
| `int ndim` | `int ndim` | **KEEP** - dimension count |
| `int d` (dim index) | `int d` | **KEEP** - dimension loop |

---

## Getting Help

- GitHub Issue: #584
- Migration Plan: `docs/INT64_INDEX_MIGRATION.md`
- NumPy Reference: `src/numpy/_core/include/numpy/npy_common.h:217`
