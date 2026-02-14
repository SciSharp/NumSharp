# NEP 56 - Array API Standard Support

**Status:** Final
**NumSharp Impact:** HIGH - Defines cross-library compatibility standard

## Summary

NumPy 2.0 adds support for the **Array API standard (2022.12)**, enabling code portability across NumPy, CuPy, JAX, PyTorch, and Dask.

## What is the Array API Standard?

A specification for array library APIs designed for cross-library compatibility. Code written to the standard works with any compliant library.

```python
def vq_py(obs, code_book):
    xp = array_namespace(obs, code_book)  # Get namespace (numpy, cupy, etc.)
    dist = xp.asarray(cdist(obs, code_book))
    return xp.argmin(dist, axis=1)
```

## Major Changes in NumPy 2.0

### A. New Strict Behaviors

| Function | Change |
|----------|--------|
| `.T` | Errors for ndim > 2 |
| `cross()` | Errors on size-2 vectors (only size-3) |
| `solve()` | Strict validation of inputs |
| `outer()` | Raises on >1-D inputs (was: flatten) |

### B. DType Changes

| Function | Change |
|----------|--------|
| `ceil()` | Returns integer dtype (was: float) |
| `floor()` | Returns integer dtype (was: float) |
| `trunc()` | Returns integer dtype (was: float) |

### C. Numerical Behavior Changes

| Function | Change |
|----------|--------|
| `pinv()` | `rtol` default now dtype-dependent |
| `matrix_rank()` | `tol` renamed to `rtol` |

### D. `copy` Keyword Semantics

```python
np.asarray(obj, copy=True)   # Always copy
np.asarray(obj, copy=False)  # Never copy (raise if needed)
np.asarray(obj, copy=None)   # Copy if necessary (old default)
```

### E. FFT Precision

All `numpy.fft` functions now preserve 32-bit precision (was: upcast to float64).

## New Functions and Aliases

### New Aliases (Trigonometry)
| New Name | Alias For |
|----------|-----------|
| `acos` | `arccos` |
| `acosh` | `arccosh` |
| `asin` | `arcsin` |
| `asinh` | `arcsinh` |
| `atan` | `arctan` |
| `atanh` | `arctanh` |
| `atan2` | `arctan2` |

### New Aliases (Other)
| New Name | Alias For |
|----------|-----------|
| `concat` | `concatenate` |
| `permute_dims` | `transpose` |
| `pow` | `power` |
| `bitwise_left_shift` | `left_shift` |
| `bitwise_right_shift` | `right_shift` |
| `bitwise_invert` | `invert` |

### New Functions

| Function | Description |
|----------|-------------|
| `isdtype(dtype, kind)` | Check dtype kind |
| `unique_all()` | All unique info |
| `unique_counts()` | Unique values + counts |
| `unique_inverse()` | Unique values + inverse indices |
| `unique_values()` | Just unique values |
| `matrix_transpose()` | Transpose last two axes |
| `vecdot()` | Vector dot product |
| `matrix_norm()` | Matrix norm (gufunc) |
| `vector_norm()` | Vector norm (gufunc) |

### New Properties

| Property | Description |
|----------|-------------|
| `ndarray.mT` | Matrix transpose (last 2 axes) |
| `ndarray.device` | Returns CPU device object |

### New Keywords

| Function | New Keyword | Description |
|----------|-------------|-------------|
| `std()`, `var()` | `correction` | Clearer than `ddof` |
| `sort()`, `argsort()` | `stable` | Complement to `kind` |

## NumSharp Implementation Checklist

### Required for Array API Compliance

**High Priority:**
- [ ] `copy` keyword with three-way semantics
- [ ] `.T` error for ndim > 2
- [ ] `ceil/floor/trunc` return integer dtypes
- [ ] `isdtype()` function

**Medium Priority:**
- [ ] Add trig aliases: `acos`, `asin`, `atan`, etc.
- [ ] Add `concat`, `permute_dims`, `pow` aliases
- [ ] Implement `unique_*` family of functions
- [ ] Add `matrix_transpose()` and `.mT` property

**Lower Priority:**
- [ ] `vecdot()`, `matrix_norm()`, `vector_norm()`
- [ ] `correction` keyword for `std/var`
- [ ] `stable` keyword for `sort/argsort`
- [ ] `.device` property (always returns CPU)

### Breaking Changes to Consider

NumSharp should decide whether to:
1. Match NumPy 2.0 exactly (breaking from 1.x)
2. Provide compatibility layer
3. Target specific NumPy version

## `np.bool` Reintroduced

```python
np.bool  # Now points to numpy.bool_ (was removed in NumPy 1.20)
```

NumSharp note: `np.bool` should map to `NPTypeCode.Boolean`.

## References

- [NEP 56 Full Text](https://numpy.org/neps/nep-0056-array-api-main-namespace.html)
- [Array API Standard](https://data-apis.org/array-api/latest/)
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
