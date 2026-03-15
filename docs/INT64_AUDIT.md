# Int64 Migration Audit

## Purpose

This document tracks the audit of commits from `198f34f4` to `HEAD` for compliance with `INT64_MIGRATION_GUIDE.md`.

## Commits Under Review (19 total)

| Commit | Description |
|--------|-------------|
| fe5ed380 | Create INT64_MIGRATION_GUIDE.md |
| 00d1c694 | fix: add long[] overloads for NDArray typed getters |
| f4d00fbd | int64 indexing: complete migration from NumSharp |
| 8d07ff03 | int64 indexing: comprehensive migration progress |
| 970da0a6 | docs: Add int64 indexing developer guide |
| b9df3e42 | int64 indexing: StrideDetector pointer params int* -> long* |
| 2b657586 | int64: ILKernelGenerator.Clip.cs TransformOffset and Default.ATan2.cs fixed statements |
| 7cf157e6 | int64: ILKernelGenerator.Clip.cs and Default.Dot.NDMD.cs migration |
| 019351a3 | int64: AllSimdHelper totalSize and loop counters to long |
| 97dd0443 | int64 indexing: SimdMatMul, np.nonzero, NDArray indexer, Transpose |
| 788e9819 | int64 indexing: NonZero returns long[], IKernelProvider uses long size |
| 82e6d8fe | int64 indexing: IKernelProvider, Transpose, Clip, TensorEngine, Shape.Unmanaged |
| 30cce8f1 | int64 indexing: MatMul, unique, itemset, convolve fixes |
| a90e9e07 | int64 indexing: partial fixes for ArraySlice, random, incrementor |
| cf04cc29 | refactor(int64): migrate all Incrementors to long indices |
| a86d79c6 | refactor(int64): core type fixes - NDArray, Storage, Iterator |
| 9e3e0671 | refactor(int64): UnmanagedStorage.Getters/Setters - migrate to long indices |
| 36bb824c | refactor(int64): NDArray.String.cs - complete straightforward fixes |
| dbb81789 | refactor(int64): NDArray.String.cs - handle long Count with overflow check |
| bd68d691 | fix: NDArray.cs - all params int[] methods now have long[] primary + int[] convenience |
| edeeb5b7 | fix: ArraySlice<T> interface implementation for long indexing |
| 28ff483c | fix: additional int64 conversions for Phase 1 |

## Compliance Areas (from INT64_MIGRATION_GUIDE.md)

### Area 1: Core Types (Parts 1-3)
- Shape fields: dimensions, strides, offset, size, bufferSize must be `long`
- Slice/SliceDef fields must be `long`
- Storage/Iterator fields must be `long`
- Method signatures maintain original API
- int[] overloads use Shape.ComputeLongShape()
- params only on long[] overloads

### Area 2: Memory & Iteration (Parts 4-6)
- Use UnmanagedMemoryBlock, not Array.CreateInstance
- Use pointers, not Span<T> for long indexing
- Loop variables must be `long`
- IL generation: typeof(long), Ldc_I8, Conv_I8 only for ndim/axis

### Area 3: Algorithms & Tests (Parts 7-10)
- No int.MaxValue constraints in algorithms (only at string boundary)
- Business logic preserved (Min/Max constraints)
- No (int) downcasts
- Test assertions use long[] for shape comparisons

## Audit Status

| Area | Agent | Status | Findings |
|------|-------|--------|----------|
| Core Types (Parts 1-3) | core-types-auditor | COMPLETE | 93+ violations |
| Memory & Iteration (Parts 4-6) | memory-iter-auditor | COMPLETE | 8 violations |
| Algorithms & Tests (Parts 7-10) | algo-tests-auditor | COMPLETE | 17 violations |

---

## Findings Summary

**Total Violations:** ~118 (some overlap between auditors)
**Acceptable Exceptions:** 14 (documented .NET boundaries)

| Severity | Count | Category |
|----------|-------|----------|
| HIGH | 10 | Algorithms throw instead of long support, IL type mismatches |
| MEDIUM | 95 | `params int[]` overloads, ad-hoc ConvertAll |
| LOW | 50+ | Test assertions use `new[]` instead of `new long[]` |

---

## HIGH Priority Violations

### H1. Algorithms Throw Instead of Supporting Long

Per guide Part 7: "No int.MaxValue constraints in algorithms"

| File | Line | Issue |
|------|------|-------|
| `np.random.choice.cs` | 17-19 | Throws NotSupportedException for size > int.MaxValue, downcasts |
| `np.random.shuffle.cs` | 19-21 | Throws NotSupportedException for size > int.MaxValue, downcasts |
| `np.searchsorted.cs` | 57-62 | Throws OverflowException, uses int loop variable |
| `SimdMatMul.cs` | 41-44 | Accepts long params, throws if > int.MaxValue, downcasts to int |
| `Default.MatMul.2D2D.cs` | 121-124 | Same pattern - accepts long, throws, downcasts |

**Fix:** Rewrite algorithms to natively use long loop variables and pointer arithmetic.

### H2. Span with (int) Cast Index

Per guide Part 5: "Use pointers, not Span<T> for long indexing"

| File | Line | Issue |
|------|------|-------|
| `np.all.cs` | 153 | `inputSpan[(int)inputIndex]` - truncates for >2B elements |
| `np.any.cs` | 157 | `inputSpan[(int)inputIndex]` - truncates for >2B elements |
| `NDArray.Indexing.Masking.cs` | 95,122,125,143,145 | Multiple `(int)i`, `(int)srcIdx` casts |

**Fix:** Replace Span indexing with pointer access: `T* ptr = (T*)storage.Address; ptr[inputIndex]`

### H3. IL Kernel Type Mismatches

Per guide Part 6: "Loop variables declared as typeof(long)"

| File | Line | Issue |
|------|------|-------|
| `ILKernelGenerator.Comparison.cs` | 403 | `locIOffset` declared as `typeof(int)`, holds long result |
| `ILKernelGenerator.Comparison.cs` | 343 | `Ldc_I4_0` stored to long local without Conv_I8 |
| `ILKernelGenerator.MixedType.cs` | 1066-1067 | `Ldc_I4_0` stored to long local without Conv_I8 |

**Fix:** Change `typeof(int)` to `typeof(long)` for index offset locals; add `Conv_I8` after `Ldc_I4` when storing to long locals.

---

## MEDIUM Priority Violations

### M1. `params int[]` on Backwards-Compatibility Overloads (93 occurrences)

Per guide Part 2: "params only on long[] overloads"

**Key files affected:**
- `Shape.cs:559,641,797` - constructor and GetOffset/GetSubshape
- `NDArray.cs:775,806,834,862,876,890,904` and typed setters (1007-1103)
- `UnmanagedStorage.Getters.cs:18,141,385,424-524`
- `UnmanagedStorage.Setters.cs:117,132,200,230,294,445-625`
- `np.zeros.cs:14,36`, `np.ones.cs:17,39,50`, `np.empty.cs:14,36`, `np.full.cs:17,41`
- All `np.random.*` dimension parameters

**Fix:** Remove `params` keyword from all `int[]` overloads to avoid CS0121 ambiguity.

### M2. Missing `Shape.ComputeLongShape()` Method

Per guide Part 3: "Use Shape.ComputeLongShape() for int[] to long[] conversion"

The migration guide specifies this method, but it does NOT exist. Instead, code uses ad-hoc conversion:

| File | Line | Pattern |
|------|------|---------|
| `np.full.cs` | 19,43 | `System.Array.ConvertAll(shapes, i => (long)i)` |
| `np.empty.cs` | 16,38 | `System.Array.ConvertAll(shapes, i => (long)i)` |
| `NdArray.ReShape.cs` | 53,118 | `System.Array.ConvertAll(shape, i => (long)i)` |
| `NdArray`1.ReShape.cs` | 43,99 | `System.Array.ConvertAll(shape, i => (long)i)` |

**Fix:** Create `Shape.ComputeLongShape(int[] dims)` and use consistently.

### M3. Remaining `int*` Pointer Method

Per guide Part 2: "Pointer parameters use long* not int*"

| File | Line | Issue |
|------|------|-------|
| `Shape.cs` | 1295 | `InferNegativeCoordinates(long[] dimensions, int* coords, int ndims)` |

**Fix:** Migrate to `long*` or deprecate if `long*` version in Shape.Unmanaged.cs covers all usages.

### M4. Size Variables Using int

| File | Line | Issue |
|------|------|-------|
| `NdArray.Convolve.cs` | 47-48,83-84,185-186,206-207 | `int na = (int)a.size; int nv = (int)v.size;` |
| `np.arange.cs` | 119,199,308 | `int length = (int)Math.Ceiling(...)` |

**Fix:** Change size variables to `long` with long-supporting algorithms.

### M5. List Capacity with int Cast

Per guide Part 4: "No List<T> with long capacity"

| File | Line | Issue |
|------|------|-------|
| `Default.NonZero.cs` | 71 | `new List<long>(Math.Max(16, (int)Math.Min(size / 4, int.MaxValue)))` |
| `ILKernelGenerator.Masking.cs` | 189 | Same pattern |

**Fix:** Use `ArraySlice<long>` or `UnmanagedMemoryBlock<long>` instead.

---

## LOW Priority Violations

### L1. Test Assertions Use `new[]` Instead of `new long[]` (50+ occurrences)

Per guide Part 8: "Shape comparisons use explicit long[]"

**Files affected:**
- `test/*/Manipulation/NDArray.astype.Truncation.Test.cs:371,393`
- `test/*/Creation/np.empty_like.Test.cs` (50+ occurrences)
- `test/*/NpBroadcastFromNumPyTests.cs:1021,1025,1029,1033,1037,1449`
- `test/*/NDArray.Base.Test.cs:383`

**Pattern:** `result.shape.Should().BeEquivalentTo(new[] { 2, 3 });`
**Fix:** `result.shape.Should().BeEquivalentTo(new long[] { 2, 3 });`

---

## Acceptable Exceptions (Per Guide Part 9)

These use int.MaxValue at legitimate .NET API boundaries:

| File | Line | Reason |
|------|------|--------|
| `NDArray.String.cs` | 31-32,85-86,98-99 | String conversion boundary - .NET string length limit |
| `ArraySlice.cs` | 332-334 | Span creation - Span is int-indexed by design |
| `Shape.cs` | 1050-1052,1068-1070 | `GetIntDimensions`/`GetIntSize` for .NET interop |
| `NdArrayToMultiDimArray.cs` | 34,48 | .NET Array.SetValue requires int[] indices |
| `Randomizer.cs` | 13,38,131,231,232,262,264,268,324,368 | PRNG algorithm constants |
| `ILKernelGenerator.Reduction.cs` | 851,854 | SIMD identity values for Min reduction |
| `ReductionKernel.cs` | 385-386 | Identity values for typed reductions |
| `StackedMemoryPool.cs` | 77,227 | Pool threshold optimization parameters |
| `UnmanagedStorage.cs` | 186 | GetSpan() with documented limitation |
| `UnmanagedMemoryBlock.cs` | 584,639 | Fast path uint.MaxValue boundary check |
| `Arrays.cs` | 390,405 | Array.CreateInstance - .NET limitation |
| `np.load.cs` | 34,141,162 | .npy file format uses int32 shapes |
| `Hashset.cs` | 1950,1971 | Hash codes are int by definition |
| `Converts.Native.cs` | 1021-1062,2406-2409 | Value conversion (not indices) |

---

## Definition of Done

- [x] All 22 commits reviewed for compliance
- [x] All red flag patterns searched and validated
- [x] Violations documented with file:line references
- [x] Fix recommendations provided for each violation
- [x] Summary report delivered to user

---

## Next Steps

1. **Create `Shape.ComputeLongShape(int[] dims)`** - Standard conversion method per guide
2. **Remove `params` from 93 `int[]` overloads** - Avoid CS0121 ambiguity
3. **Fix HIGH priority violations** - Algorithms throwing instead of long support
4. **Fix IL kernel type mismatches** - typeof(int) → typeof(long), add Conv_I8
5. **Replace Span indexing with pointers** - np.all.cs, np.any.cs, Masking.cs
6. **Update test assertions** - `new[]` → `new long[]` for shape comparisons
