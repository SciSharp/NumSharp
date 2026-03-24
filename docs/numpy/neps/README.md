# NumPy Enhancement Proposals (NEPs) - NumSharp Reference

This directory contains documentation for NEPs relevant to NumSharp's goal of 1-to-1 NumPy compatibility.

## What are NEPs?

NEPs (NumPy Enhancement Proposals) are design documents describing new features, processes, or environments for NumPy. Similar to Python's PEPs.

## NEP Index by Impact

### CRITICAL - Breaking Changes in NumPy 2.0

| NEP | Title | Impact |
|-----|-------|--------|
| [NEP 50](NEP50.md) | Promotion Rules for Python Scalars | Type promotion completely changed |
| [NEP 52](NEP52.md) | Python API Cleanup for NumPy 2.0 | ~100 functions removed/renamed |
| [NEP 56](NEP56.md) | Array API Standard Support | New functions, changed behaviors |

### HIGH - Significant Implementation Requirements

| NEP | Title | Impact |
|-----|-------|--------|
| [NEP 01](NEP01.md) | .npy File Format | Must match exactly for interop |
| [NEP 07](NEP07.md) | DateTime Types | NOT IMPLEMENTED in NumSharp |
| [NEP 19](NEP19.md) | Random Number Generator Policy | NumSharp claims 1-to-1 seed matching |
| [NEP 27](NEP27.md) | Zero Rank Arrays | Affects scalar handling |
| [NEP 38](NEP38.md) | SIMD Optimization Instructions | Performance via SIMD (#544, #545) |
| [NEP 54](NEP54.md) | SIMD Infrastructure (Highway) | Modern SIMD patterns |
| [NEP 55](NEP55.md) | UTF-8 Variable-Width String DType | New string handling |

### MEDIUM - Behavioral Considerations

| NEP | Title | Impact |
|-----|-------|--------|
| [NEP 05](NEP05.md) | Generalized Ufuncs | Affects operation signatures |
| [NEP 10](NEP10.md) | Iterator/UFunc Optimization | Cache-coherency, dimension coalescing |
| [NEP 20](NEP20.md) | Gufunc Signature Enhancement | Frozen/flexible dimensions |
| [NEP 21](NEP21.md) | Advanced Indexing Semantics | oindex vs vindex (deferred but informative) |
| [NEP 34](NEP34.md) | Disallow dtype=object Inference | Ragged array creation |
| [NEP 42](NEP42.md) | New and Extensible DTypes | Future dtype architecture |
| [NEP 43](NEP43.md) | Extensible UFuncs | Custom dtype ufuncs |
| [NEP 51](NEP51.md) | Scalar Representation | ToString() output |

### LOW - Informational / Python-Specific

| NEP | Title | Impact |
|-----|-------|--------|
| [NEP 13](NEP13.md) | Ufunc Override Mechanism | Python duck-typing |
| [NEP 18](NEP18.md) | Array Function Dispatch | Python duck-typing |
| [NEP 32](NEP32.md) | Remove Financial Functions | DO NOT IMPLEMENT |
| [NEP 49](NEP49.md) | Data Allocation Strategies | Custom allocator patterns |
| [NEP 53](NEP53.md) | C-API Evolution for NumPy 2.0 | API versioning patterns |

## NEPs NOT Documented (Not Relevant to NumSharp)

| NEP | Title | Reason |
|-----|-------|--------|
| 0 | Purpose and Process | Meta |
| 14 | Dropping Python 2.7 | Python-specific |
| 15 | Merging multiarray/umath | Internal restructuring |
| 22 | Duck Typing Overview | Python-specific |
| 23 | Backwards Compatibility Policy | Meta |
| 28 | Website Redesign | Meta |
| 29 | Version Support Policy | Meta |
| 36 | Fair Play | Meta |
| 40 | Legacy Datatype Docs | Informational only |
| 41 | First Step Toward New Dtypes | Covered by NEP 42 |
| 44 | Documentation Restructuring | Meta |
| 45 | C Style Guide | Meta |
| 46-48 | Sponsorship/Spending | Meta |
| 57 | Platform Support | Meta |

## Quick Reference: NumPy 1.x vs 2.x Changes

### Type Promotion (NEP 50)
```python
# NumPy 1.x: Value-based promotion
uint8(1) + 2 → int64(3)

# NumPy 2.x: Weak scalar promotion
uint8(1) + 2 → uint8(3)
```

### Removed Functions (NEP 52)
```python
# Use instead:
np.round_     → np.round
np.product    → np.prod
np.sometrue   → np.any
np.alltrue    → np.all
```

### New Functions (NEP 56)
```python
# Aliases
np.acos, np.asin, np.atan   # for arccos, arcsin, arctan
np.concat                    # for concatenate
np.permute_dims             # for transpose

# New
np.isdtype(dtype, kind)
np.unique_values(), np.unique_counts()
ndarray.mT  # Matrix transpose
```

### copy Keyword (NEP 56)
```python
np.asarray(x, copy=True)   # Always copy
np.asarray(x, copy=False)  # Never copy (error if needed)
np.asarray(x, copy=None)   # Copy if necessary
```

## Implementation Priority for NumSharp

### Phase 1: Core Compatibility
1. NEP 50 - Fix type promotion
2. NEP 52 - API cleanup audit
3. NEP 56 - Add new functions/aliases

### Phase 2: Feature Completeness
4. NEP 07 - Add datetime64/timedelta64
5. NEP 55 - Improve string handling
6. NEP 27 - Verify zero-rank behavior

### Phase 3: Interoperability
7. NEP 01 - Verify .npy format compliance
8. NEP 19 - Verify RNG seed matching

### Phase 4: Performance (SIMD)
9. NEP 38/54 - SIMD optimization (#544, #545)

## SIMD Implementation for NumSharp

Related GitHub issues: **#544**, **#545**

### .NET SIMD Options

| API | .NET Version | Portability | Control |
|-----|--------------|-------------|---------|
| `Vector<T>` | .NET Core 2.0+ | High | Low |
| `Vector128/256/512<T>` | .NET Core 3.0+ | Medium | High |
| Native P/Invoke | Any | Low | Full |

### Priority Operations

| Operation | Speedup | Complexity |
|-----------|---------|------------|
| Element-wise (+, -, *, /) | 4-8x | Low |
| Reductions (sum, mean) | 2-4x | Medium |
| Comparisons | 4-8x | Low |
| Dot/matmul | 4-16x | High |

### Example Pattern

```csharp
// Runtime dispatch like NumPy
ISimdBackend backend = Avx2.IsSupported ? new Avx2Backend()
                     : Avx.IsSupported ? new AvxBackend()
                     : Sse2.IsSupported ? new Sse2Backend()
                     : new ScalarBackend();
```

See [NEP38.md](NEP38.md) and [NEP54.md](NEP54.md) for detailed patterns.

## References

- [NumPy NEPs Index](https://numpy.org/neps/)
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
- [Array API Standard](https://data-apis.org/array-api/latest/)
