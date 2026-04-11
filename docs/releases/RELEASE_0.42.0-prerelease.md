# NumSharp 0.42.0 - Long Indexing Release

This release introduces **Int64/Long Indexing** - a complete architectural migration enabling arrays larger than 2.1 billion elements (>2GB), along with comprehensive **NumPy 2.x type system alignment**, new type introspection APIs, and the Python container protocol.

## TL;DR

- **Int64/Long Indexing**: Full migration from `int` to `long` across Shape, NDArray, Storage, Iterators, and ILKernelGenerator - ndarrays >2GB now supported
- **12 New Type APIs**: `np.can_cast`, `np.promote_types`, `np.result_type`, `np.min_scalar_type`, `np.common_type`, `np.issubdtype`, `np.finfo`, `np.iinfo`, `np.isreal`, `np.iscomplex`, `np.isrealobj`, `np.iscomplexobj`
- **6 Comparison Functions**: `np.equal`, `np.not_equal`, `np.less`, `np.greater`, `np.less_equal`, `np.greater_equal`
- **4 Logical Functions**: `np.logical_and`, `np.logical_or`, `np.logical_not`, `np.logical_xor`
- **Container Protocol**: `__contains__`, `__len__`, `__iter__`, `__getitem__`, `__setitem__` - NumPy-compatible iteration
- **New NDArray Methods**: `tolist()`, `item()` for NumPy parity
- **NumPy 2.x Type System**: `np.arange()` returns Int64, `NPTypeHierarchy` encoding NumPy's exact type tree, Bool NOT under Number
- **`np.frombuffer()` Rewrite**: Full NumPy signature with `count`, `offset`, big-endian support, `IntPtr`/`void*` overloads, view semantics
- **0D Scalar Arrays**: `np.array(5)` now creates 0D arrays (matching NumPy)
- **`np.arange()` Fixes**: Negative step, integer arithmetic, inlined type-specific loops, full NumPy parity.
- **`np.any/np.all`**: 0D array support with axis parameter
- **Random API Alignment** (#582): Parameter names match NumPy, `np.shuffle` fixed
- **Empty Array Handling**: Proper NaN returns for mean/std/var on empty arrays
- **NaN Sorting**: `np.unique` now sorts NaN to end (matches NumPy)
- **ValueType to Object Migration**: All scalar returns now `object` (NumPy alignment), discarded usages of ValueType 
- **UnmanagedSpan<T>**: Ported from dotnet/runtime for Span-like semantics with `long` length
- **Operator Cleanup**: 74% reduction in NDArray.Primitive.cs (150 → 40 overloads)
- **600+ Battle Tests**: All validated against actual NumPy 2.x output
- **145 Test Fixes**: 71 for Int64 alignment + 74 previously failing tests now passing


## Contents

- [Int64/Long Indexing](#int64long-indexing-support)
- [NumPy 2.x Type System](#numpy-2x-type-system)
- [Container Protocol](#container-protocol)
- [New APIs](#new-apis)
- [Bug Fixes](#bug-fixes)
- [Performance](#performance-improvements)
- [Refactoring](#refactoring)
- [Test Improvements](#test-improvements)
- [Breaking Changes](#breaking-changes)
- [Known Issues](#known-issues)
- [Documentation](#documentation)
- [Installation](#installation)



## Int64/Long Indexing Support

Complete migration from `int` to `long` indexing across the entire codebase, enabling arrays larger than 2.1 billion elements (~2GB for byte arrays, ~16GB for doubles).

### Core Type Changes

- `Shape.dimensions`: `int[]` -> `long[]`
- `Shape.strides`: `int[]` -> `long[]`
- `Shape.size`: `int` -> `long`
- `Shape.offset`: `int` -> `long`
- `NDArray.size`: `int` -> `long`
- `NDArray.len`: `int` -> `long`
- All NDArray indexers: `int` -> `long`
- `ArraySlice<T>`: `int` indexing -> `long` indexing
- `UnmanagedMemoryBlock<T>`: `int` indexing -> `long` indexing
- `UnmanagedStorage`: `int` indexing -> `long` indexing
- `NDIterator` coordinates: `int[]` -> `long[]`
- `MultiIterator`: `int` offsets -> `long` offsets
- `np.nonzero()`: Returns `NDArray<long>[]` instead of `NDArray<int>[]`
- `np.argmax/argmin`: Returns `long` indices

### ILKernelGenerator Migration (20+ files)

All ILKernelGenerator partial classes updated for `long` loop counters and offsets:

- `ILKernelGenerator.Binary.cs` - Loop counters to `long`
- `ILKernelGenerator.Reduction.cs` - Index variables to `long`
- `ILKernelGenerator.Reduction.Axis.cs` - Axis iteration with `long`
- `ILKernelGenerator.Reduction.Axis.Simd.cs` - SIMD paths with `long`
- `ILKernelGenerator.Reduction.Axis.NaN.cs` - NaN handling with `long`
- `ILKernelGenerator.Reduction.Axis.Arg.cs` - ArgMax/ArgMin with `long`
- `ILKernelGenerator.Reduction.Axis.VarStd.cs` - Variance/StdDev with `long`
- `ILKernelGenerator.Reduction.NaN.cs` - **NEW** NaN reductions IL generation
- `ILKernelGenerator.Scan.cs` - CumSum/CumProd with `long` indices
- `ILKernelGenerator.MatMul.cs` - Matrix dimensions to `long`
- `ILKernelGenerator.Clip.cs` - TransformOffset calculations
- `ILKernelGenerator.Masking.cs` - Boolean masking with `long`
- `ILKernelGenerator.Masking.Boolean.cs` - Boolean operations with `long`
- `ILKernelGenerator.Masking.NaN.cs` - NaN masking with `long`
- `ILKernelGenerator.Masking.VarStd.cs` - Variance masking with `long`

### New Infrastructure

- `UnmanagedSpan<T>` - Ported from dotnet/runtime Span<T> - Span-like with `long` length
- `ReadOnlyUnmanagedSpan<T>` - Read-only variant
- `UnmanagedSpanExtensions` - Extension methods for Span<T> parity
- `UnmanagedSpanHelpers` - SIMD-optimized value type methods
- `UnmanagedSpanHelpers.T.cs` - Generic type helpers
- `UnmanagedBuffer` - Buffer management for long arrays
- `LongIntroSort` - Sorting algorithm for large arrays (port of .NET IntroSort)
- `LongIndexBuffer` - Unmanaged index collection (replaces List<long> for >2B indices)
- `BitHelperLong` - Long-indexed bit marking (for >2B bits)
- `IndexCollector.cs` - **NEW** Index collection for masking operations
- `Hashset<T>` - Upgraded to long-based indexing with 33% growth

### New API Overloads

- `NDArray.GetInt32(long[])` - Long coordinate access
- `NDArray.GetInt64(long[])` - Long coordinate access
- `NDArray.GetSingle(long[])` - Long coordinate access
- `NDArray.GetDouble(long[])` - Long coordinate access
- `NDArray.GetBoolean(long[])` - Long coordinate access
- `NDArray.GetByte(long[])` - Long coordinate access
- All 9 typed setters with `long[]` coordinates
- All other typed getters with `long[]` coordinates
- `np.random.choice` - `long` population size support
- `np.repeat` - `long` repeat counts
- `np.linspace` - `long` num parameter
- `np.roll` - `long` shift parameter
- All random sampling functions - `long[]` size parameters



## NumPy 2.x Type System

### `np.arange()` Returns Int64

- `np.arange(10)` - Before: `Int32`, After: `Int64`
- `np.arange(10.0)` - Unchanged: `Float64`
- Integer arithmetic now performed in target dtype (matches NumPy's template approach)
- Inlined type-specific loops matching NumPy's `arraytypes.c.src` implementation

### NPTypeHierarchy

New `NPTypeHierarchy.cs` (294 lines) encoding NumPy's exact type tree structure from `multiarraymodule.c`:

- Bool is NOT under Number (NumPy 2.x critical behavior)
- `NPTypeHierarchy.IsSubType(Bool, Number)` returns `false`
- `issubdtype(int32, int64)` returns `false` (concrete types are siblings)
- `isdtype(bool, 'numeric')` returns `false` (bool excluded from numeric)

### New Type Introspection APIs (12)

- `np.can_cast(from, to, casting)` - Full NumPy-compatible type casting checks with 'no', 'equiv', 'safe', 'same_kind', 'unsafe' modes
- `np.promote_types(type1, type2)` - Type promotion following NumPy rules
- `np.result_type(*args)` - Result type inference for arrays and dtypes
- `np.min_scalar_type(value)` - Minimum scalar type for a value
- `np.common_type(*arrays)` - Common type for multiple arrays
- `np.issubdtype(arg1, arg2)` - Type hierarchy checking
- `np.finfo(dtype)` - Machine limits for floating-point types (eps, min, max, resolution)
- `np.iinfo(dtype)` - Machine limits for integer types (min, max, bits)
- `np.isreal(x)` - Check if array has no imaginary part
- `np.iscomplex(x)` - Check if array has imaginary part
- `np.isrealobj(x)` - Check if object is real type
- `np.iscomplexobj(x)` - Check if object is complex type

### C#-Friendly Overloads

- `iinfo<T>()`, `finfo<T>()` - Generic overloads
- `can_cast<TFrom, TTo>()` - Generic type checking
- `promote_types<T1, T2>()` - Generic promotion
- NDArray and string dtype overloads for all functions

### can_cast Refactoring

Replaced 80+ lines of switch cases with single-line derivation:
```csharp
CanCastSafe(A, B) = (A == B) || (_FindCommonType_Array(A, B) == B)
```
Verified against NumPy for all 121 type pairs (11x11 matrix).



## Container Protocol

New `NDArray.Container.cs` implementing Python's container protocol for NumPy compatibility.

### Implemented Methods

- `__contains__` / `Contains()` - Membership testing via element-wise comparison
- `__hash__` / `GetHashCode()` - Throws `NotSupportedException` (NDArray is mutable/unhashable)
- `__len__` - Returns first dimension length, throws `TypeError` for 0-d scalars
- `__iter__` / `GetEnumerator()` - NumPy-compatible iteration over first axis
- `__getitem__` - Indexing with int/long/string slice notation
- `__setitem__` - Assignment with int/long/string slice notation

### Iteration Behavior (BREAKING)

- **0-D arrays (scalars)**: Throws `TypeError` (not iterable)
- **1-D arrays**: Yields scalar elements
- **N-D arrays (N > 1)**: Yields (N-1)-D NDArray slices along first axis

This matches NumPy:
```python
>>> for x in np.array([[1,2],[3,4]]): print(x)
[1 2]
[3 4]
```

### New Exception

- `TypeError.cs` - For NumPy-compatible error messages



## New APIs

### Comparison Functions (6 new)

- `np.equal(x1, x2)` - Element-wise equality (wraps `==`)
- `np.not_equal(x1, x2)` - Element-wise inequality (wraps `!=`)
- `np.less(x1, x2)` - Element-wise less than (wraps `<`)
- `np.greater(x1, x2)` - Element-wise greater than (wraps `>`)
- `np.less_equal(x1, x2)` - Element-wise less or equal (wraps `<=`)
- `np.greater_equal(x1, x2)` - Element-wise greater or equal (wraps `>=`)

### Logical Functions (4 new)

- `np.logical_and(x1, x2)` - Element-wise logical AND
- `np.logical_or(x1, x2)` - Element-wise logical OR
- `np.logical_not(x)` - Element-wise logical NOT
- `np.logical_xor(x1, x2)` - Element-wise logical XOR

### NDArray Methods

**`NDArray.tolist()`** - Convert NDArray to nested lists (NumPy parity):
```csharp
var arr = np.array(new int[,] {{1,2}, {3,4}});
var list = arr.tolist();  // List<object> containing nested lists
```

**`NDArray.item(*args)`** - Copy element to standard scalar:
- `item()` - Extract scalar from size-1 arrays
- `item(index)` - Flat indexing with negative index support
- `item(i, j)` / `item(i, j, k)` - Multi-dimensional indexing
- `item<T>()` - Type-converting variant
- `np.asscalar()` marked as `[Obsolete]` (removed in NumPy 2.0)

### Operator Files

- `NDArray.BitwiseNot.cs` - `~` operator implementation
- `NDArray.XOR.cs` - `^` operator with object pattern

### `np.frombuffer()` Complete Rewrite

**NumPy-compatible signature:**
- `np.frombuffer(buffer, dtype=float64, count=-1, offset=0)`

**Features:**
- `count` parameter - Number of items to read (-1 = all available)
- `offset` parameter - Byte offset into buffer
- Big-endian support via dtype strings (`">u4"`, `">i4"`, `"<i2"`)
- `ArraySegment<byte>` overload - Uses built-in Offset/Count
- `Memory<byte>` overload - View if array-backed, otherwise copies
- `IntPtr` + dispose overload - Native interop with optional ownership
- `void*` overload - Unsafe pointer convenience
- `frombuffer<TSource>(TSource[], dtype)` - Reinterpret typed arrays
- View semantics - Pinned buffer, modifications affect original

### Scalar Array Creation

- `np.array(5)` now correctly creates 0D arrays (matching NumPy)
- Previously created 1D single-element arrays

### keepdims Parameter

- `np.argmax(axis, keepdims)` - Added keepdims parameter
- `np.argmin(axis, keepdims)` - Added keepdims parameter

### Parameter Rename

- `outType` renamed to `dtype` in 19 np.*.cs files to match NumPy



## Changes and Fixes

- **`np.arange(10, 0, -2)`**: Before returned `[9, 7, 5, 3, 1]`, now correctly returns `[10, 8, 6, 4, 2]`
- **`np.arange(0, 5, 0.5, int32)`**: Before returned `[0,0,1,1,2,2,3,3,4,4]`, now correctly returns `[0,0,0,0,0,0,0,0,0,0]` (NumPy behavior)
- **`np.any(0D_array, axis=0)`**: Before threw `ArgumentException`, now returns 0D bool scalar
- **`np.all(0D_array, axis=-1)`**: Before threw `ArgumentException`, now returns 0D bool scalar
- **`Contains([1,2], array([1,2,3]))`**: Before returned `False`, now throws `IncorrectShapeException` (matches NumPy)
- **`np.shuffle` axis parameter**: Removed non-existent `axis` param, now matches NumPy legacy API
- **`np.random.standard_normal`**: Fixed typo (`stardard_normal` → `standard_normal`)
- **Scalar broadcast assignment**: Fixed cross-dtype conversion failure
  - Root cause: `AsOrMakeGeneric<T>()` called `new NDArray<T>(astype(...))` which triggered implicit scalar → size constructor
  - Fix: Use `.Storage` to pass storage directly, avoiding implicit conversion
- **Fancy indexing dtypes**: Now supports all integer dtypes (Int16, Int32, Int64), not just Int32
  - Added `NormalizeIndexArray()` helper that keeps Int32/Int64 as-is, converts smaller types to Int64
  - Throws `IndexOutOfRangeException` for non-integer types (float, decimal)
- NDArray.ToString() now formats 100% identical to numpy.
- **`np.mean([])`**: Returns `NaN` (was throwing or returning 0)
- **`np.mean(zeros((0,3)), axis=0)`**: Returns `[NaN, NaN, NaN]`
- **`np.mean(zeros((0,3)), axis=1)`**: Returns empty array `[]`
- **`np.std/var` single element**: Returns `NaN` with `ddof >= size`
- **Empty comparison**: All 6 comparison operators now return empty boolean arrays (was returning scalar)
- **`np.unique` NaN sorting**: NaN now sorts to end (matches NumPy: `[-inf, 1, 2, inf, nan]`)
- **`ArgMax/ArgMin` NaN**: First NaN always wins (NaN takes precedence over any value)
- **Single-element axis reduction**: Changed `Storage.Alias()` and `squeeze_fast()` to return copies (was sharing memory)
- **Clip mixed-dtype**: Fixed bug where int32 min/max arrays were read as int64
- **`np.invert(bool)`**: Now uses logical NOT (`!x`) instead of bitwise NOT (`~x`)
- **`np.square(int)`**: Preserves integer dtype instead of promoting to double
- **`np.negate(bool)`**: Removed buggy linear-indexing path, now routes through `ExecuteUnaryOp`
- Fixed ATan2 non-contiguous array handling by adding `np.broadcast_arrays()` and `.copy()` materialization
- Fixed ATan2 wrong pointer type (byte*) for x operand in all non-byte cases
- **finfo**: Use `MathF.BitIncrement` for float eps (was using `Math.BitIncrement` which only works on double)
- **issctype**: Properly reject string type (was returning true for `typeof(string)`)
- **`NDArray.unique()`**: Fixed for long indexing support
- **`np.repeat`**: Fixed dtype handling and long count support
- **`np.random.choice`**: Fixed for long population sizes
- **`np.argmax/argmin` IL fix**: Removed `Conv_I4` instruction that truncated long indices to int32
- **ILKernel loop counters**: Fixed numerous int32 overflow issues
- **`TransformOffset` calculations**: Fixed for >2GB arrays
- **SIMD helper functions**: Fixed for long indexing
- **AVX2 gather**: Added stride check (falls back to scalar for stride > int.MaxValue)
- Parameter names now match NumPy (`size`, `a`, `b`, `p`, `d0`)
- `np.random()` added as alias for uniform distribution
- `np.shuffle` removed non-existent axis parameter
- ValueType to Object Migration
    - All scalar return types migrated from `ValueType` to `object`
    - `NPTypeCode.GetDefaultValue()` now returns `object`
    - All operators migrated to NumPy-aligned object pattern
    - NDArray null checks converted from `== null` to `is null` pattern
- Operator Overload Cleanup
    - `NDArray.Primitive.cs`: 159 → 42 lines (74% reduction)
    - ~150 explicit scalar overloads → ~40 object-based overloads
    - Added missing `implicit operator NDArray(byte)`
    - Changed `ushort` from explicit to implicit
- Implicit Scalar Conversion
    - `(int)ndarray_float64` now works via `Converts.ChangeType`
    - `scalar → NDArray`: implicit (safe, creates 0-d array)
    - `NDArray → scalar`: explicit (requires 0-d, throws `IncorrectShapeException`)
    - Matches NumPy's `int(arr)`, `float(arr)`, `bool(arr)` pattern
- All `== null` changed to `is null` (because `==` now returns `NDArray<bool>` as does numpy)
- All `!= null` changed to `is not null`
- Type System Consolidation
    - `can_cast` derived from promotion tables (replaced 80+ lines of switch cases)
    - Single source of truth: `NPTypeHierarchy`
    - Removed duplicate `TypeKind` enum and category helper methods

## Performance Improvements

### SIMD NaN Statistics

New `ILKernelGenerator.Reduction.NaN.cs` (1,097 lines) providing SIMD optimization for:

- `np.nansum` - Sum ignoring NaN
- `np.nanprod` - Product ignoring NaN
- `np.nanmin` - Minimum ignoring NaN
- `np.nanmax` - Maximum ignoring NaN
- `np.nanmean` - Mean ignoring NaN
- `np.nanvar` - Variance ignoring NaN
- `np.nanstd` - Standard deviation ignoring NaN

SIMD Algorithm (NaN masking via self-comparison):
```
nanMask = Equals(vec, vec)           // True for non-NaN
cleaned = BitwiseAnd(vec, nanMask)   // Zero out NaN values
```

### SIMD Helper Functions (IKernelProvider)

- `AllSimdHelper<T>()` - SIMD-accelerated boolean all() with early-exit
- `AnySimdHelper<T>()` - SIMD-accelerated boolean any() with early-exit
- `ArgMaxSimdHelper<T>()` - Two-pass SIMD: find max value, then find index
- `ArgMinSimdHelper<T>()` - Two-pass SIMD: find min value, then find index
- `NonZeroSimdHelper<T>()` - Collects indices where elements != 0
- `CountTrueSimdHelper()` - Counts true values in bool array
- `CopyMaskedElementsHelper<T>()` - Copies elements where mask is true
- `ConvertFlatIndicesToCoordinates()` - Converts flat indices to arrays

### np.arange() Optimization

- Inlined type-specific loops (no delegate overhead per element)
- Direct pointer casts: `addr[i] = (int)(start + i * step)`
- Matches NumPy's `arraytypes.c.src` fill pattern

### MatMul Long Indexing

- SIMD MatMul updated for `long` indices
- V128/V256/V512 support preserved
- Matrices >2GB now supported

### Code Reduction

- `Default.Clip.cs`: 914 → 240 lines (76% reduction)
- `Default.MatMul.2D2D.cs`: 19,862 → 284 lines (98.6% reduction)



## Test Improvements

### New Test Infrastructure

- `[LargeMemoryTest]` attribute - Inherits from OpenBugs for CI exclusion
- `TestMemoryTracker` - Diagnose CI OOM failures
- TUnit category exclusion - Proper CI filtering with `--treenode-filter`

### New Test Files (30+)

**Type API Tests:**
- `np.can_cast.BattleTest.cs`
- `np.promote_types.BattleTest.cs`
- `np.result_type.BattleTest.cs`
- `np.min_scalar_type.BattleTest.cs`
- `np.common_type.BattleTest.cs`
- `np.issubdtype.BattleTest.cs`
- `np.finfo.BattleTest.cs`
- `np.iinfo.BattleTest.cs`
- `np.isreal_iscomplex.BattleTest.cs`
- `np.type_checks.BattleTest.cs`
- `np.typing.Test.cs`
- `NPTypeHierarchy.BattleTest.cs` (74 tests)

**Long Indexing Tests:**
- `LongIndexingSmokeTest.cs` - 96 np.* functions with 1M elements
- `LongIndexingBroadcastTest.cs` - 2.36 billion element iterations
- `LongIndexingMasterTest.cs` - Full 2.4GB array allocations
- `ArgsortInt64Tests.cs`
- `HashsetLongIndexingTests.cs`
- `MatMulInt64Tests.cs`
- `NonzeroInt64Tests.cs`

**Container Protocol Tests:**
- `ContainerProtocolTests.cs` - 69 basic tests
- `ContainerProtocolBattleTests.cs` - Round 1 battle tests
- `ContainerProtocolBattleTests2.cs` - 51 tests (round 2)
- `ContainsNumPyAlignmentTests.cs`

**Comprehensive Tests:**
- `ArgMaxArgMinComprehensiveTests.cs` - 480 lines covering all dtypes, shapes, axes
- `VarStdComprehensiveTests.cs` - 462 lines covering ddof, empty arrays, edge cases
- `CumSumComprehensiveTests.cs` - 381 lines covering accumulation, overflow, dtypes
- `np_nonzero_strided_tests.cs` - 221 lines for strided/transposed arrays
- `NonContiguousTests.cs` - 35+ tests for strided/broadcast arrays
- `DtypeCoverageTests.cs` - 26 parameterized tests for all 12 dtypes
- `np.comparison.Test.cs` - Comparison function tests

**Array Creation Tests:**
- `np.arange.BattleTests.cs` (50+ cases)
- `np.array.BattleTests.cs`
- `np.ToString.BattleTests.cs`

**Other Tests:**
- `TolistTests.cs`
- `ViewTests.cs`

### Test Counts

- Type introspection battle tests: 200+
- Container protocol tests: 120 (69 + 51)
- np.arange battle tests: 50+
- Type hierarchy tests: 74
- Contains battle tests: 50
- Long indexing smoke tests: 96 functions
- Linear algebra tests: 389 (dot: 195, matmul: 106, outer: 88)
- Fancy indexing tests: 20
- Scalar broadcast tests: 13
- **Total new battle tests: 600+**
- NumPy 2.x Int64 alignment: 71 tests fixed
- OpenBugs now passing: 74 tests enabled
- **Total fixes: 145**
- 5,020 tests in CI (+74 tests from fixed OpenBugs)

## Platform Limitations (Cannot Fix)

These are .NET platform limitations, not NumSharp bugs:

- `Span<T>` limited to `int.MaxValue` elements by .NET runtime. We introduced UnmanagedSpan<T> that has identical API to Span<T> with long support.
- `List<T>.Count` returns `int` (use `LongCount()` extension) but still the internal array can't exceed int.MaxValue.
- `Hashset<T>.Count` returns `int` (NumSharp adds `LongCount` property)
- .NET managed arrays limited to `int.MaxValue` elements
- .NET string length limited to `int.MaxValue` characters
