# Compliance and Compatibility

NumSharp aims for **1-to-1 behavioral compatibility with NumPy 2.x** and compliance with the **Python Array API Standard**. This page tracks our progress toward these goals.

---

## Compliance Goals

| Standard | Target | Status | Milestone |
|----------|--------|--------|-----------|
| [NumPy 2.x](https://numpy.org/doc/stable/) | 2.4.2+ | In Progress | [NumPy 2.x Compliance](https://github.com/SciSharp/NumSharp/milestone/9) |
| [Array API Standard](https://data-apis.org/array-api/latest/) | 2024.12 | In Progress | [Array API Standard](https://github.com/SciSharp/NumSharp/milestone/6) |
| [NumPy Enhancement Proposals](https://numpy.org/neps/) | Key NEPs | In Progress | [NEP Compliance](https://github.com/SciSharp/NumSharp/milestone/7) |

---

## NumPy 2.x Compliance

NumPy 2.0 introduced significant breaking changes. NumSharp is working to align with these changes.

### Type Promotion (NEP 50)

NumPy 2.x changed from **value-based** to **weak scalar** promotion:

```csharp
// NumPy 1.x behavior (OLD - value-based)
np.result_type(np.int8, 1) == np.int8      // 1 fits in int8
np.result_type(np.int8, 255) == np.int16   // 255 doesn't fit - UPCASTED

// NumPy 2.x behavior (NEW - weak scalar)
uint8(1) + 2 → uint8(3)           // Python scalar defers to array dtype
uint8(1) + 255 → uint8(0)         // Overflow with warning
```

**NumSharp Status:** [#529](https://github.com/SciSharp/NumSharp/issues/529) - Type promotion diverges from NEP 50

### API Cleanup (NEP 52)

NumPy 2.0 removed ~100 deprecated functions and aliases:

| Removed | Use Instead |
|---------|-------------|
| `np.round_` | `np.round` |
| `np.product` | `np.prod` |
| `np.sometrue` | `np.any` |
| `np.alltrue` | `np.all` |

### Array API Standard Functions (NEP 56)

New functions and aliases added in NumPy 2.x:

| Category | Functions |
|----------|-----------|
| **Aliases** | `acos`, `asin`, `atan`, `atan2`, `concat`, `permute_dims`, `pow` |
| **New** | `isdtype()`, `unique_values()`, `unique_counts()`, `unique_inverse()`, `unique_all()` |
| **Properties** | `ndarray.mT` (matrix transpose), `ndarray.device` |

### copy= Semantics

```csharp
np.asarray(x, copy: true)   // Always copy
np.asarray(x, copy: false)  // Never copy (raise if needed)
np.asarray(x, copy: null)   // Copy if necessary (default)
```

---

## Array API Standard Compliance

The [Python Array API Standard](https://data-apis.org/array-api/latest/) defines a common API for array computing libraries, enabling code portability across NumPy, PyTorch, JAX, CuPy, and Dask.

### Data Types

| Category | Required | NumSharp |
|----------|----------|----------|
| Boolean | `bool` | ✅ |
| Signed Integer | `int8`, `int16`, `int32`, `int64` | ✅ (as short, int, long) |
| Unsigned Integer | `uint8`, `uint16`, `uint32`, `uint64` | ✅ |
| Floating-Point | `float32`, `float64` | ✅ |
| Complex | `complex64`, `complex128` | ❌ Not implemented |

**Gap:** Complex number support is not yet implemented.

### Required Constants

| Constant | Description | NumSharp |
|----------|-------------|----------|
| `e` | Euler's constant | ✅ |
| `inf` | Positive infinity | ✅ |
| `nan` | Not a Number | ✅ |
| `newaxis` | Dimension expansion | ✅ |
| `pi` | Mathematical pi | ✅ |

### Array Attributes

| Attribute | Description | NumSharp |
|-----------|-------------|----------|
| `dtype` | Data type | ✅ |
| `device` | Hardware device | ❌ |
| `ndim` | Number of dimensions | ✅ |
| `shape` | Dimensions | ✅ |
| `size` | Total element count | ✅ |
| `T` | Transpose | ✅ |
| `mT` | Matrix transpose | ❌ |

### Function Coverage

| Category | Required | NumSharp | Coverage |
|----------|----------|----------|----------|
| Creation | 16 | ~14 | 87% |
| Element-wise | 67 | ~50 | 75% |
| Data Types | 6 | ~3 | 50% |
| Linear Algebra | 4 | 4 | 100% |
| Manipulation | 14 | ~10 | 71% |
| Statistical | 9 | ~7 | 78% |
| Searching | 6 | ~4 | 67% |
| Sorting | 2 | 2 | 100% |
| Set | 4 | 1 | 25% |
| Indexing | 2 | 0 | 0% |
| Utility | 3 | 2 | 67% |
| **Total Core** | **133** | **~80** | **~60%** |

### Optional Extensions

| Extension | Functions | NumSharp Status |
|-----------|-----------|-----------------|
| **linalg** | 23 (cholesky, det, eigh, inv, qr, svd, solve...) | Partial (most are stubs) |
| **fft** | 14 (fft, ifft, rfft, fftfreq...) | Not implemented |

See [#560](https://github.com/SciSharp/NumSharp/issues/560) for full specification details.

---

## NEP Compliance

NumPy Enhancement Proposals (NEPs) define behavioral standards. Key NEPs for NumSharp:

### Critical - NumPy 2.0 Breaking Changes

| NEP | Title | Status | Issue |
|-----|-------|--------|-------|
| [NEP 50](https://numpy.org/neps/nep-0050-scalar-promotion.html) | Promotion Rules for Python Scalars | In Progress | [#547](https://github.com/SciSharp/NumSharp/issues/547) |
| [NEP 52](https://numpy.org/neps/nep-0052-python-api-cleanup.html) | Python API Cleanup | In Progress | [#547](https://github.com/SciSharp/NumSharp/issues/547) |
| [NEP 56](https://numpy.org/neps/nep-0056-array-api-main-namespace.html) | Array API Standard Support | In Progress | [#547](https://github.com/SciSharp/NumSharp/issues/547) |

### High Priority - Feature Completeness

| NEP | Title | Status | Issue |
|-----|-------|--------|-------|
| [NEP 07](https://numpy.org/neps/nep-0007-datetime-proposal.html) | DateTime and Timedelta Types | Not Started | [#554](https://github.com/SciSharp/NumSharp/issues/554) |
| [NEP 19](https://numpy.org/neps/nep-0019-rng-policy.html) | Random Number Generator Policy | Implemented | [#553](https://github.com/SciSharp/NumSharp/issues/553) |
| [NEP 27](https://numpy.org/neps/nep-0027-zero-rank-arrarys.html) | Zero Rank Arrays (Scalars) | Partial | [#550](https://github.com/SciSharp/NumSharp/issues/550) |
| [NEP 55](https://numpy.org/neps/nep-0055-string_dtype.html) | UTF-8 Variable-Width String DType | Not Started | [#549](https://github.com/SciSharp/NumSharp/issues/549) |

### Medium Priority - Behavioral Considerations

| NEP | Title | Status | Issue |
|-----|-------|--------|-------|
| [NEP 05](https://numpy.org/neps/nep-0005-generalized-ufuncs.html) | Generalized Universal Functions | Partial | [#551](https://github.com/SciSharp/NumSharp/issues/551) |
| [NEP 21](https://numpy.org/neps/nep-0021-advanced-indexing.html) | Advanced Indexing Semantics | In Progress | [#552](https://github.com/SciSharp/NumSharp/issues/552) |
| [NEP 42](https://numpy.org/neps/nep-0042-new-dtypes.html) | New and Extensible DTypes | Not Started | [#549](https://github.com/SciSharp/NumSharp/issues/549) |

### Performance - SIMD Optimization

| NEP | Title | Status | Issue |
|-----|-------|--------|-------|
| [NEP 10](https://numpy.org/neps/nep-0010-new-iterator-ufunc.html) | Iterator/UFunc Optimization | Not Started | [#548](https://github.com/SciSharp/NumSharp/issues/548) |
| [NEP 38](https://numpy.org/neps/nep-0038-simd-optimizations.html) | SIMD Optimization Instructions | Not Started | [#548](https://github.com/SciSharp/NumSharp/issues/548) |
| [NEP 54](https://numpy.org/neps/nep-0054-simd-highway.html) | SIMD Infrastructure (Highway) | Not Started | [#548](https://github.com/SciSharp/NumSharp/issues/548) |

---

## Supported Data Types

NumSharp currently supports 12 numeric data types:

| NPTypeCode | C# Type | NumPy Equivalent | Array API |
|------------|---------|------------------|-----------|
| Boolean | `bool` | `np.bool_` | ✅ |
| Byte | `byte` | `np.uint8` | ✅ |
| Int16 | `short` | `np.int16` | ✅ |
| UInt16 | `ushort` | `np.uint16` | ✅ |
| Int32 | `int` | `np.int32` | ✅ |
| UInt32 | `uint` | `np.uint32` | ✅ |
| Int64 | `long` | `np.int64` | ✅ |
| UInt64 | `ulong` | `np.uint64` | ✅ |
| Char | `char` | (no equivalent) | ❌ |
| Single | `float` | `np.float32` | ✅ |
| Double | `double` | `np.float64` | ✅ |
| Decimal | `decimal` | (no equivalent) | ❌ |

### Missing Types

| Type | Status | Priority |
|------|--------|----------|
| `complex64` | Not implemented | High (Array API required) |
| `complex128` | Not implemented | High (Array API required) |
| `datetime64` | Not implemented | Medium (NEP 07) |
| `timedelta64` | Not implemented | Medium (NEP 07) |
| `StringDType` | Not implemented | Low (NEP 55) |

---

## Memory Layout

| Feature | NumPy | NumSharp |
|---------|-------|----------|
| C-order (row-major) | ✅ | ✅ |
| F-order (column-major) | ✅ | ❌ |

**Note:** NumSharp only supports C-order (row-major) memory layout. The `order` parameter on `ravel`, `flatten`, `copy`, and `reshape` is accepted but ignored.

See [#546](https://github.com/SciSharp/NumSharp/issues/546) for F-order support tracking.

---

## File Format Interoperability

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| `.npy` | ✅ | ✅ | NumPy array format (NEP 01) |
| `.npz` | ✅ | ❌ | Compressed archive of .npy files |
| Binary | ✅ | ✅ | `fromfile()` / `tofile()` |

Files created by NumSharp can be read by NumPy and vice versa.

---

## Random Number Generation

NumSharp's `np.random` module provides **1-to-1 seed matching** with NumPy (NEP 19):

```csharp
// Same seed produces identical sequences
np.random.seed(42);
var a = np.random.rand(5);

// Equivalent Python:
// np.random.seed(42)
// a = np.random.rand(5)
```

This enables reproducible results when porting code between NumPy and NumSharp.

---

## Implementation Roadmap

### Phase 1: Core Compatibility (Current)
- Fix type promotion to match NEP 50
- API cleanup audit (NEP 52)
- Add Array API functions and aliases (NEP 56)

### Phase 2: Feature Completeness
- Add complex number support (`complex64`, `complex128`)
- Add `datetime64` / `timedelta64` types (NEP 07)
- Implement missing Array API functions

### Phase 3: Extensions
- Complete `linalg` extension (currently stubs)
- Add `fft` extension

### Phase 4: Performance
- SIMD optimization (NEP 38/54)
- Iterator optimization (NEP 10)

---

## Contributing

Help us achieve NumPy compatibility! See our [GitHub milestones](https://github.com/SciSharp/NumSharp/milestones) for tracked issues:

- [NumPy 2.x Compliance](https://github.com/SciSharp/NumSharp/milestone/9)
- [Array API Standard Compliance](https://github.com/SciSharp/NumSharp/milestone/6)
- [NEP Compliance](https://github.com/SciSharp/NumSharp/milestone/7)

---

## References

- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
- [Python Array API Standard](https://data-apis.org/array-api/latest/)
- [NumPy Enhancement Proposals](https://numpy.org/neps/)
- [NumPy Source (v2.4.2)](https://github.com/numpy/numpy/tree/v2.4.2) - Reference implementation at `src/numpy/`
