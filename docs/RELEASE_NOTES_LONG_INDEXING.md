# NumSharp Release Notes - Long Indexing Branch

**456 files changed | 95,375 insertions | 8,238 deletions**

---

## Major Features

### Int64/Long Indexing Support (Arrays >2GB)

Complete migration from `int` to `long` indexing across the entire codebase, enabling arrays larger than 2.1 billion elements (~2GB for byte arrays, ~16GB for doubles).

**Core Changes:**
- `Shape.dimensions`, `Shape.strides`, `Shape.size`, `Shape.offset` changed to `long[]` / `long`
- `NDArray.size`, `NDArray.len`, all indexers changed to `long`
- `ArraySlice<T>`, `UnmanagedMemoryBlock<T>`, `UnmanagedStorage` migrated to `long` indexing
- `NDIterator`, `MultiIterator` updated for `long` coordinates and offsets
- All 20+ ILKernelGenerator partial classes updated for `long` loop counters and offsets
- Added `UnmanagedSpan<T>` (ported from dotnet/runtime) for Span-like semantics with `long` length
- Added `LongIntroSort` for sorting large arrays
- `Hashset<T>` upgraded to long-based indexing with 33% growth for large collections

**API Additions for Long Indexing:**
- `long[]` overloads for `NDArray.GetInt32()`, `GetInt64()`, `GetSingle()`, etc.
- `long` population size support in `np.random.choice`
- `long` repeat counts in `np.repeat`
- All random sampling functions support `long[]` size parameters

---

### NumPy 2.x Alignment

**`np.arange()` Now Returns Int64:**
- Integer inputs now return `Int64` arrays (NumPy 2.x behavior)
- Fixed negative step behavior to match NumPy exactly
- Fixed integer arithmetic for dtype casting (matches NumPy's template approach)
- Inlined type-specific loops matching NumPy's `arraytypes.c.src` implementation

**Type System Overhaul:**
- Added `NPTypeHierarchy.cs` encoding NumPy's exact type tree structure (from `multiarraymodule.c`)
- `Bool` is NOT under `Number` (NumPy 2.x critical behavior)
- `issubdtype(int32, int64)` correctly returns `False` (concrete types are siblings)

**New Type Introspection APIs:**
- `np.can_cast(from, to, casting)` - Full NumPy-compatible type casting checks
- `np.promote_types(type1, type2)` - Type promotion following NumPy rules
- `np.result_type(*arrays_and_dtypes)` - Result type inference
- `np.min_scalar_type(value)` - Minimum scalar type for a value
- `np.common_type(*arrays)` - Common type for arrays
- `np.issubdtype(arg1, arg2)` - Type hierarchy checking
- `np.isreal()`, `np.iscomplex()`, `np.isrealobj()`, `np.iscomplexobj()`
- `np.finfo(dtype)` - Machine limits for floating-point types
- `np.iinfo(dtype)` - Machine limits for integer types

**Container Protocol (Python `in` operator):**
- `NDArray.Contains()` now propagates broadcasting errors (matches NumPy's `__contains__`)
- `[1,2] in np.array([1,2,3])` now throws `IncorrectShapeException`
- Type mismatch returns `False` (e.g., `"hello" in np.array([1,2,3])`)

---

### New NDArray Methods

- **`NDArray.tolist()`** - Convert NDArray to nested lists (NumPy parity)
- **`NDArray.item(*args)`** - Copy element to standard Python scalar
- **`np.frombuffer()`** - Complete rewrite with full NumPy-compatible signature:
  - `count` and `offset` parameters
  - Big-endian byte swap support via dtype strings (`">u4"`, `">i4"`)
  - `ArraySegment<byte>`, `Memory<byte>`, `IntPtr`, `void*` overloads
  - Optional dispose callback for native memory ownership
  - View semantics (pinned buffer, modifications affect original)

---

## Performance Improvements

### SIMD-Optimized MatMul
- **35-100x speedup** over scalar path for matrix multiplication
- Added `SimdMatMul` with V128/V256/V512 vector support
- Long indexing support for matrices >2GB

### SIMD NaN Statistics
- `nansum`, `nanmean`, `nanstd`, `nanvar`, `nanmin`, `nanmax` optimized with SIMD
- Added `ILKernelGenerator.Reduction.NaN.cs` (1,097 lines of IL generation)

### General SIMD Improvements
- All reduction operations (sum, prod, min, max, mean, std, var) with SIMD paths
- Scan operations (cumsum, cumprod) with SIMD optimization
- Boolean reductions (any, all) with SIMD fast paths

### np.arange() Performance
- Inlined type-specific loops (no delegate overhead per element)
- Direct pointer casts matching NumPy's template-generated fill functions

---

## Bug Fixes

### Core Functionality
- **`np.any/np.all` with axis parameter**: Now supports 0D (scalar) arrays with `axis=0` or `axis=-1`
- **`np.arange` negative step**: Fixed to return `[10,8,6,4,2]` instead of `[9,7,5,3,1]` for `arange(10,0,-2)`
- **Scalar broadcast assignment**: Fixed cross-dtype conversion
- **Fancy indexing**: Support for all integer dtypes (Int16, Int32, Int64, etc.)
- **`NDArray.unique()`**: Fixed for long indexing support
- **`np.repeat`**: Fixed dtype handling and long count support
- **`np.random.choice`**: Fixed for long population sizes
- **`np.shuffle`**: Aligned with NumPy legacy API (removed axis parameter that didn't exist)
- **`np.random.standard_normal`**: Fixed typo in API
- **`np.random()`**: Added alias for uniform distribution

### ILKernel Fixes
- Fixed numerous int32 overflow issues in loop counters
- Fixed `TransformOffset` calculations for >2GB arrays
- Fixed SIMD helper functions for long indexing

### Test Fixes
- Fixed 71 test failures from NumPy 2.x Int64 alignment
- Removed `[OpenBugs]` from 74 now-passing tests
- Fixed dtype-specific getter mismatches throughout test suite

---

## Refactoring

### ValueType to Object Migration
- All scalar return types migrated from `ValueType` to `object`
- `NPTypeCode.GetDefaultValue()` returns `object`
- All operators migrated to NumPy-aligned object pattern
- NDArray null checks converted from `== null` to `is null` pattern

### Type System Consolidation
- `can_cast` derived from type promotion tables (replaced 80+ lines of switch cases)
- Single source of truth for type hierarchy (`NPTypeHierarchy`)
- Removed duplicate `TypeKind` enum and category helper methods

### Code Cleanup
- Removed unused `Fx.cs` (953 lines of pooling code)
- Removed `KernelKey.cs`, `KernelSignatures.cs`, `SimdThresholds.cs`, `TypeRules.cs`
- Removed `StorageType.cs`, `np.linalg.norm.cs` (incomplete LAPACK bindings)
- Removed `LongList<T>` utility class
- Removed LINQ extension files (`IEnumeratorExtensions.cs`, `MaxBy.cs`)

---

## Documentation

### New Guides
- **Buffering, Arrays and Unmanaged Memory** - Memory architecture, view vs copy semantics, ownership model
- **IL Kernel Generation** - How ILKernelGenerator works with SIMD
- **NumPy .npy/.npz Format Reference** - Binary format implementation details
- **Int64 Indexing Migration Guide** - Patterns for large array support

### API Documentation
- Complete typing function documentation with NumPy alignment notes
- `np.frombuffer` overloads and ownership model documentation

---

## Test Improvements

### New Test Infrastructure
- `[HighMemory]` attribute for tests requiring 8GB+ RAM
- `[SkipOnLowMemory]` runtime memory check attribute
- `TestMemoryTracker` for diagnosing CI OOM failures
- Proper TUnit category exclusion in CI

### New Test Coverage
- **~500+ new battle tests** validated against actual NumPy 2.x output
- `LongIndexingSmokeTest` - 96 np.* function coverage with 1M element arrays
- `LongIndexingBroadcastTest` - 2.36 billion element broadcast iterations
- `LongIndexingMasterTest` - Full 2.4GB array allocations
- Comprehensive `np.arange` battle tests (50+ cases)
- Container protocol tests (100+ cases)
- Type hierarchy tests (74 cases)
- All typing functions have battle test files

### Test Fixes
- Fixed 71 tests for NumPy 2.x Int64 alignment
- Enabled 74 previously failing tests (marked as OpenBugs but passing)
- CI workflow updated to properly exclude `[HighMemory]` tests on Ubuntu

---

## Breaking Changes

| Change | Migration |
|--------|-----------|
| `Shape.dimensions` changed to `long[]` | Update code accessing dimensions directly |
| `Shape.strides` changed to `long[]` | Update code accessing strides directly |
| `NDArray.size` changed to `long` | Use `long` or cast to `int` where safe |
| `np.arange(int)` returns `Int64` | Use `.astype(np.int32)` if Int32 needed |
| `Contains()` throws on shape mismatch | Wrap in try-catch if relying on `False` |
| `ValueType` returns changed to `object` | Cast return values explicitly |
| `np.shuffle` removed axis parameter | Was non-functional, use correct NumPy API |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Commits | 166 |
| Files Changed | 456 |
| Lines Added | 95,375 |
| Lines Removed | 8,238 |
| New C# Files | 50+ |
| New Test Files | 30+ |
| Battle Tests Added | ~500+ |
| Previously Failing Tests Fixed | 145 |
