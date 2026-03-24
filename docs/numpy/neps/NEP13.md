# NEP 13 - A Mechanism for Overriding Ufuncs

**Status:** Final
**NumSharp Impact:** LOW (Python-specific) - Informational only

## Summary

Defines `__array_ufunc__`, a special method allowing classes to override NumPy ufunc behavior. This is Python's duck-typing mechanism.

## What is `__array_ufunc__`?

A Python protocol enabling custom array types to intercept and customize ufunc operations.

```python
def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    """
    ufunc:   The ufunc object (e.g., np.add)
    method:  How called: "__call__", "reduce", "accumulate", etc.
    inputs:  Positional arguments
    kwargs:  Keyword arguments
    """
```

## Why This Matters for NumSharp (Informational)

### Python Duck Typing

This protocol enables:
- CuPy arrays to work with NumPy operations
- Dask arrays to work with NumPy operations
- Custom array types in the NumPy ecosystem

### C# Equivalent

C# doesn't have this dynamic dispatch. Instead, NumSharp uses:
1. **Interface-based polymorphism**
2. **Generic methods with type constraints**
3. **Runtime type checking**

```csharp
// NumSharp pattern
public static NDArray Add(NDArray a, NDArray b) {
    // Type-check and dispatch
    return a.TensorEngine.Add(a, b);
}
```

### No Direct Implementation Needed

NumSharp doesn't need to implement `__array_ufunc__` because:
1. It's a Python-specific protocol
2. C# has different extensibility mechanisms
3. NumSharp is the primary array type (not integrating with others)

## Understanding NumPy's Design

This NEP helps understand why NumPy:
- Has a `TensorEngine` abstraction
- Separates operation dispatch from implementation
- Supports multiple backend implementations

## References

- [NEP 13 Full Text](https://numpy.org/neps/nep-0013-ufunc-overrides.html)
- Related: NEP 18 (array function protocol)
