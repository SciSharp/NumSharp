# NEP 53 - Evolving the NumPy C-API for NumPy 2.0

**Status:** Draft (Open)
**NumSharp Impact:** LOW - C-API specific, but informs API evolution patterns

## Summary

Defines strategy for evolving NumPy's C-API while maintaining backwards compatibility. Introduces API versioning and compatibility packages.

## Key Changes in NumPy 2.0

### Removed Functions
| Function | Replacement |
|----------|-------------|
| `PyArray_Mean` | Use `arr.mean()` method |
| `PyArray_Std` | Use `arr.std()` method |
| `MapIter` API | N/A (advanced indexing internals) |

### Struct Layout Changes

**`PyArray_Descr` (dtype struct):**
- Larger maximum itemsize
- New flags for custom user dtypes
- Direct field access → macro access

```c
// OLD
npy_intp size = descr->elsize;

// NEW
npy_intp size = PyDataType_ITEMSIZE(descr);
```

**`NPY_MAXDIMS`:** Increased from 32 to 64 dimensions

## Backwards Compatibility Strategy

### Step 1: NumPy 1.25+
```c
#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
```
- Default exports older API for compatibility
- New API requires explicit opt-in

### Step 2: NumPy 2.0
- Requires recompilation against NumPy 2.0
- `numpy2_compat` package for dual 1.x/2.0 support

## NumSharp Relevance

### Why This Matters (Informational)

1. **API Evolution Pattern:** Shows how NumPy handles breaking changes
2. **Versioning Strategy:** Informs NumSharp's own API versioning decisions
3. **Compatibility Layer:** Concept of compatibility packages

### NumSharp Considerations

NumSharp doesn't use NumPy's C-API directly, but similar patterns apply:

```csharp
// Similar pattern: accessor methods instead of direct field access
public class Shape {
    // BAD: public fields that can't evolve
    public int[] Dimensions;

    // GOOD: properties/methods that can change implementation
    public int NDim => _dimensions.Length;
    public int GetDimension(int axis) => _dimensions[axis];
}
```

### Breaking Changes in NumSharp

If NumSharp follows similar patterns:

1. **Deprecation Period:** Warn before removing APIs
2. **Accessor Methods:** Use methods/properties instead of fields
3. **Versioning:** Consider major version bumps for breaking changes

## Impact on Downstream Packages

| Package Type | Impact | Migration |
|--------------|--------|-----------|
| C-API Users | Must adapt code | Minor changes |
| Cython Users | Less impact with Cython 3 | Use macros |
| End Users | Transparent | None |

## References

- [NEP 53 Full Text](https://numpy.org/neps/nep-0053-c-abi-evolution.html)
- [NumPy 2.0 Migration Guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html)
