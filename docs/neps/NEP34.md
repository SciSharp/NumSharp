# NEP 34 - Disallow Inferring dtype=object from Sequences

**Status:** Final
**NumSharp Impact:** MEDIUM - Affects array creation from ragged sequences

## Summary

NumPy no longer silently creates `dtype=object` arrays from ragged nested sequences. Instead, it raises `ValueError`.

## Behavior Change

### Old Behavior (Pre-NEP 34)
```python
>>> np.array([[1, 2], [1]])
array([[1, 2], [1]], dtype=object)  # Silent object dtype
```

### New Behavior (Post-NEP 34)
```python
>>> np.array([[1, 2], [1]])
ValueError: cannot guess the desired dtype from the input
```

## When ValueError is Raised

1. **Ragged nested sequences:**
   ```python
   np.array([[1, 2], [1]])  # Different lengths
   ```

2. **Mixed sequences and non-sequences:**
   ```python
   np.array([np.arange(10), [10]])
   ```

3. **Mixed sequence types within sequences:**
   ```python
   np.array([[range(3), range(3)], [range(3), 0]])
   ```

## Explicit Object Dtype

Users who intentionally want object arrays must explicitly request them:

```python
# Explicit dtype=object works
np.array([[1, 2], [1]], dtype=object)

# For structured object arrays
arr = np.empty(correct_shape, dtype=object)
arr[...] = values
```

## NumSharp Implementation

### Current Behavior Check

Verify NumSharp's `np.array()` behavior with ragged sequences:
```csharp
// Should this throw or create object array?
var arr = np.array(new object[] {
    new int[] { 1, 2 },
    new int[] { 1 }
});
```

### Recommended Approach

1. **Detect ragged sequences** during array creation
2. **Throw `ArgumentException`** if shapes don't match
3. **Allow explicit `dtype: NPTypeCode.Object`** to override

### Why This Matters

Prevents silent errors where users accidentally:
- Pass mismatched sequence lengths
- Get unexpected object dtype instead of numeric
- Experience poor performance (object arrays are slow)

## References

- [NEP 34 Full Text](https://numpy.org/neps/nep-0034-infer-dtype-is-object.html)
- `src/NumSharp.Core/Creation/np.array.cs`
