# NEP 51 - Changing the Representation of NumPy Scalars

**Status:** Accepted
**NumSharp Impact:** LOW - Affects ToString() output only

## Summary

Changes how NumPy scalars are displayed in `repr()` to make types explicit.

## Representation Changes

### Before (NumPy 1.x)
```python
>>> np.float32(3.0)
3.0

>>> np.int64(34)
34

>>> np.True_
True
```

### After (NumPy 2.x)
```python
>>> np.float32(3.0)
np.float32(3.0)

>>> np.int64(34)
np.int64(34)

>>> np.True_
np.True_
```

## Affected Types

| Type | Old repr | New repr |
|------|----------|----------|
| `np.bool_` | `True` | `np.True_` |
| `np.int64` | `34` | `np.int64(34)` |
| `np.float32` | `3.0` | `np.float32(3.0)` |
| `np.complex128` | `(3+4j)` | `np.complex128(3.0+4.0j)` |
| `np.str_` | `'text'` | `np.str_('text')` |

## Rationale

1. **Type Awareness:** Makes type distinctions clearer
2. **Behavior Differences:** NumPy scalars differ from Python builtins:
   - NumPy integers can overflow (Python cannot)
   - Lower precision types need caution
   - Division by zero behavior differs
3. **Debugging:** Easier to identify type-related bugs

## NumSharp Implications

### Current Behavior

NumSharp's `ToString()` for scalars likely returns just the value:
```csharp
var x = np.array(3.0f).GetSingle();  // Returns float
Console.WriteLine(x);  // "3"
```

### Potential Alignment

If matching NumPy 2.x representation:
```csharp
public override string ToString() {
    // For scalar NDArray or typed scalar
    return $"np.{TypeName}({Value})";
}
```

### Priority

**LOW** - This is cosmetic output formatting. Focus on behavioral compatibility first.

## References

- [NEP 51 Full Text](https://numpy.org/neps/nep-0051-scalar-representation.html)
