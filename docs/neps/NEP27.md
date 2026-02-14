# NEP 27 - Zero Rank Arrays

**Status:** Final (Informational)
**NumSharp Impact:** HIGH - Affects scalar handling throughout the library

## Summary

Documents the behavior and rationale of zero-rank arrays (`shape=()`), which represent scalar quantities as arrays.

## What are Zero-Rank Arrays?

Arrays with no dimensions:
```python
>>> x = np.array(1)
>>> x.shape
()
>>> x.ndim
0
```

## Zero-Rank vs Array Scalars vs Python Scalars

| Feature | Zero-Rank Array | Array Scalar | Python Scalar |
|---------|-----------------|--------------|---------------|
| Type | `ndarray` | e.g., `np.int64` | `int`, `float` |
| Mutable | Yes | No | N/A |
| Has `.shape` | Yes `()` | Yes `()` | No |
| Can be output buffer | Yes | No | No |
| Shares memory | Can | Cannot | N/A |

## Indexing Zero-Rank Arrays

```python
>>> x = np.array(1)
>>> x[...]     # Returns scalar, not array
1
>>> x[()]      # Same as above
1
>>> x[np.newaxis]  # Adds dimension
array([1])
```

## Critical Use Cases

### 1. Output Arguments
Zero-rank arrays can be used as output buffers (scalars cannot):
```python
>>> y = np.int_(5)
>>> np.add(5, 5, y)  # TypeError - scalar can't be output

>>> x = np.array(10)
>>> np.add(5, 5, x)  # Works - zero-rank array is valid output
array(10)
```

### 2. Shared Memory Views
```python
>>> x = np.array([1, 2])
>>> y = x[1:2]
>>> y.shape = ()  # Reshape to zero-rank
>>> y
array(2)
>>> x[1] = 20
>>> y  # Reflects change
array(20)
```

### 3. Generic Code
Zero-rank arrays allow writing code that works uniformly across all array ranks.

## NumSharp Implementation Notes

### Current Behavior
NumSharp's `Shape` class supports zero-rank:
```csharp
var shape = new Shape();  // shape.NDim == 0, shape.Size == 1
var scalar = np.array(5); // Creates zero-rank array
```

### Key Considerations

1. **Indexing:** `arr[...]` and `arr[()]` should return scalar value, not array
2. **Operations:** All operations must handle zero-rank input gracefully
3. **Broadcasting:** Zero-rank arrays broadcast to any shape
4. **Type preservation:** Zero-rank arrays maintain dtype

### Related Shape Properties

```csharp
shape.IsScalar      // True for shape ()
shape.IsEmpty       // False for scalars (size > 0)
shape.NDim          // 0 for scalars
shape.Size          // 1 for scalars
```

## References

- [NEP 27 Full Text](https://numpy.org/neps/nep-0027-zero-rank-arrarys.html)
- `src/NumSharp.Core/View/Shape.cs`
