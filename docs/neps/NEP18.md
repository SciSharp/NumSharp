# NEP 18 - A Dispatch Mechanism for NumPy's High Level Array Functions

**Status:** Final
**NumSharp Impact:** LOW (Python-specific) - Informational only

## Summary

Defines `__array_function__`, a protocol enabling arguments of NumPy functions to define how those functions operate. Complements NEP 13 for non-ufunc functions.

## What is `__array_function__`?

A dispatch mechanism for NumPy's general-purpose functions (not ufuncs):

```python
def __array_function__(self, func, types, args, kwargs):
    """
    func:   NumPy function being called
    types:  Collection of argument types
    args:   Original positional arguments
    kwargs: Original keyword arguments
    """
```

## Functions Affected

### Dispatched via `__array_function__`
- `np.concatenate`, `np.stack`, `np.broadcast_to`
- `np.mean`, `np.sum`, `np.prod`
- `np.tensordot`, matrix operations
- Most array-processing functions

### NOT Dispatched (use other protocols)
- **Ufuncs:** Use `__array_ufunc__` (NEP 13)
- **`np.array()`, `np.asarray()`:** Explicit coercion
- **Methods:** On RandomState, etc.

## Why This Matters for NumSharp (Informational)

### Python Ecosystem Integration

This enables:
```python
# Generic code works with any compliant array
def f(x):
    y = np.tensordot(x, x.T)  # Dispatches via __array_function__
    return np.mean(np.exp(y))  # Mixed protocols
```

### NumSharp Architecture Insight

NumSharp's `TensorEngine` abstraction serves similar purpose:

```csharp
public abstract class TensorEngine {
    public abstract NDArray Sum(NDArray a, int? axis);
    public abstract NDArray Mean(NDArray a, int? axis);
    public abstract NDArray Concatenate(NDArray[] arrays, int axis);
}
```

The difference:
- Python: Runtime dispatch via special methods
- C#: Compile-time dispatch via abstract methods

## Supported Array Implementations

Libraries using this protocol:
- **CuPy** - GPU arrays
- **Dask** - Parallel/distributed arrays
- **scipy.sparse** - Sparse arrays
- **XArray** - Labeled arrays

## No Direct Implementation Needed

NumSharp doesn't implement `__array_function__` because:
1. Python-specific protocol
2. NumSharp IS the primary array implementation
3. C# uses different extensibility patterns

## References

- [NEP 18 Full Text](https://numpy.org/neps/nep-0018-array-function-protocol.html)
- Related: NEP 13 (ufunc overrides)
