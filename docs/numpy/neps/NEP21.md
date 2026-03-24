# NEP 21 - Simplified and Explicit Advanced Indexing

**Status:** Deferred
**NumSharp Impact:** MEDIUM - Documents indexing semantics issues

## Summary

Proposes explicit `oindex` (outer) and `vindex` (vectorized) indexing to resolve confusing advanced indexing behavior. Though deferred, it documents important edge cases NumSharp should handle correctly.

## The Three Problems

### 1. Outer vs. Vectorized Ambiguity

For `x[[0, 1], [0, 1]]` on a 2D array:

| Mode | Result | Shape |
|------|--------|-------|
| **Outer** (intuitive) | All combinations | `(2, 2)` |
| **Vectorized** (NumPy) | Diagonal elements | `(2,)` |

```python
x = np.arange(9).reshape(3, 3)
# [[0, 1, 2],
#  [3, 4, 5],
#  [6, 7, 8]]

# What users often expect (outer):
# [[x[0,0], x[0,1]],
#  [x[1,0], x[1,1]]]
# = [[0, 1], [3, 4]]

# What NumPy does (vectorized):
x[[0, 1], [0, 1]]  # = [x[0,0], x[1,1]] = [0, 4]
```

### 2. Confusing Mixed Indexing Transpose

```python
arr = np.zeros((X, Y, Z))  # 3D array

arr[:, [0,1], 0].shape      # (X, 2) - intuitive
arr[[0,1], 0, :].shape      # (2, Z) - intuitive
arr[0, :, [0,1]].shape      # (2, Y) NOT (Y, 2) - surprising!
```

### 3. Difficulty for Other Libraries

Dask, h5py, xarray struggle to implement NumPy indexing consistently.

## Proposed Solution (Deferred)

### `arr.oindex[...]` - Outer/Orthogonal Indexing

Array indices behave like slices - result axes where indices appear:

```python
arr.oindex[:, [0], [0, 1], :].shape  # (5, 1, 2, 8)
arr.oindex[:, [0], :, [0, 1]].shape  # (5, 1, 7, 2)
```

### `arr.vindex[...]` - Vectorized Indexing (Explicit)

Array indices broadcast together, **result always at front**:

```python
arr.vindex[:, [0], [0, 1], :].shape  # (2, 5, 8)
arr.vindex[:, [0], :, [0, 1]].shape  # (2, 5, 7)
```

## NumSharp Current Behavior

NumSharp should match NumPy's current (vectorized) behavior:

```csharp
var arr = np.arange(9).reshape(3, 3);
var idx = np.array(new[] { 0, 1 });

// This should return diagonal: [0, 4]
var result = arr[idx, idx];  // Vectorized indexing
```

### Implementation Considerations

```csharp
public NDArray this[params NDArray[] indices] {
    get {
        if (HasMultipleArrayIndices(indices)) {
            // Vectorized indexing: broadcast and zip
            return VectorizedIndex(indices);
        } else {
            // Simple indexing
            return SimpleIndex(indices);
        }
    }
}

private NDArray VectorizedIndex(NDArray[] indices) {
    // 1. Broadcast all array indices to common shape
    var broadcastedIndices = np.broadcast_arrays(indices);

    // 2. Iterate through broadcast shape
    // 3. Gather elements at (idx0[i], idx1[i], ...)

    // Result shape: broadcast shape (placed where array indices are)
}
```

### Edge Case: Mixed Slices and Arrays

The confusing transpose behavior:

```csharp
// arr[0, :, [0,1]] should have shape (2, Y) not (Y, 2)
// This is because array index results go to front when separated by slice
```

Rule: When array indices are **separated** by slices, their result dimensions are transposed to the front.

## Behavior Reference Table

| Expression | Shape | Explanation |
|------------|-------|-------------|
| `arr[0, 1, 2]` | `()` | Scalar |
| `arr[0, :, 2]` | `(Y,)` | Slice in middle |
| `arr[[0,1], 0, 0]` | `(2,)` | Array index first |
| `arr[0, [0,1], 0]` | `(2,)` | Array index middle |
| `arr[0, 0, [0,1]]` | `(2,)` | Array index last |
| `arr[[0,1], :, 0]` | `(2, Y)` | Array, slice, int |
| `arr[0, :, [0,1]]` | `(2, Y)` | **Confusing!** Array at end, but result at front |
| `arr[:, [0,1], :]` | `(X, 2, Z)` | Array in middle, stays there |

## Workaround for Outer Indexing

Until `oindex` is implemented, use `np.ix_`:

```python
# Outer indexing workaround
rows = [0, 1]
cols = [0, 1]
result = arr[np.ix_(rows, cols)]  # Shape (2, 2)
```

```csharp
// NumSharp equivalent
var rows = np.array(new[] { 0, 1 });
var cols = np.array(new[] { 0, 1 });
var result = arr[np.ix_(rows, cols)];  // Shape (2, 2)
```

## References

- [NEP 21 Full Text](https://numpy.org/neps/nep-0021-advanced-indexing.html)
- `src/NumSharp.Core/Selection/NDArray.Indexing.Selection.cs`
