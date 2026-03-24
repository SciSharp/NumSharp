# NEP 52 - Python API Cleanup for NumPy 2.0

**Status:** Final
**NumSharp Impact:** HIGH - Many functions removed/renamed in NumPy 2.0

## Summary

NumPy 2.0 cleaned up the main namespace, removing ~100 entries including deprecated aliases, redundant functions, and legacy code.

## Removed/Deprecated Functions

### Removed Aliases

| Removed | Use Instead |
|---------|-------------|
| `np.round_` | `np.round` |
| `np.product` | `np.prod` |
| `np.cumproduct` | `np.cumprod` |
| `np.sometrue` | `np.any` |
| `np.alltrue` | `np.all` |
| `np.inf` (8 aliases) | `np.inf` (single) |
| `np.nan` (8 aliases) | `np.nan` (single) |

### Removed Functions

| Removed | Replacement |
|---------|-------------|
| `byte_bounds` | (internal use only) |
| `disp` | `print()` |
| `safe_eval` | `ast.literal_eval` |
| `who` | (debugging tool) |
| `maximum_sctype` | Use dtype directly |

### Removed Namespaces

| Namespace | Status |
|-----------|--------|
| `np.compat` | Removed (Python 2-3 transition) |
| `numpy.core` | Renamed to `numpy._core` (private) |
| `numpy.version` | Renamed to `numpy._version` |

## Namespace Reorganization

### Regular Namespaces (Recommended)
```python
numpy
numpy.exceptions
numpy.fft
numpy.linalg
numpy.polynomial
numpy.random
numpy.testing
numpy.typing
```

### Special-Purpose
```python
numpy.array_api
numpy.ctypeslib
numpy.dtypes
numpy.lib.stride_tricks
numpy.rec
```

### Legacy (De-emphasized)
```python
numpy.char        # Use np.strings instead (NEP 55)
numpy.distutils   # Use meson/setuptools
numpy.ma          # Masked arrays
numpy.matlib      # Matrix library
```

## DType Alias Simplification

### Removed Redundant Aliases

```python
# These were all equivalent but confusing:
np.float_   # Removed, use np.float64
np.int_     # Changed meaning
np.complex_ # Removed, use np.complex128

# Platform-specific cleanup:
np.float96  # May not exist on all platforms
np.float128 # May not exist on all platforms
```

## NumSharp Implementation Checklist

### Functions to Verify

Check if NumSharp implements any removed functions:

- [ ] `np.round_` → should be `np.round` only
- [ ] `np.product` → should be `np.prod` only
- [ ] `np.cumproduct` → should be `np.cumprod` only
- [ ] `np.sometrue` → should be `np.any` only
- [ ] `np.alltrue` → should be `np.all` only

### NDArray Method Cleanup

Removed/deprecated methods:
- [ ] `.itemset()` - discouraged
- [ ] `.newbyteorder()` - too niche
- [ ] `.ptp()` - use `np.ptp()` function

### Namespace Organization

NumSharp's namespace structure in `APIs/np.cs`:
```csharp
public static partial class np {
    // Core functions
    public static NDArray array(...) { }

    // Submodules
    public static class random { }
    public static class linalg { }
    public static class fft { }  // Does NumSharp have this?
}
```

### Dead Code Audit

Per CLAUDE.md, NumSharp has dead code that should NOT be exposed:
- `np.linalg.norm` - private static (not accessible)
- `nd.inv()` - returns null
- `nd.qr()` - returns default
- `nd.svd()` - returns default
- `nd.lstsq()` - returns null
- `nd.multi_dot()` - returns null
- `np.isnan`, `np.isfinite`, `np.isclose` - return null
- `operator &`, `operator |` - return null

These should either be:
1. Properly implemented
2. Removed entirely
3. Throw `NotImplementedException`

## References

- [NEP 52 Full Text](https://numpy.org/neps/nep-0052-python-api-cleanup.html)
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
