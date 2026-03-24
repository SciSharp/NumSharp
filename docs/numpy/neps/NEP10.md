# NEP 10 - Optimizing Iterator/UFunc Performance

**Status:** Final
**NumSharp Impact:** HIGH - Core optimization patterns for array iteration

## Summary

Introduces advanced iterator optimizations that significantly improve ufunc performance, especially for non-contiguous arrays.

## Key Optimizations

### 1. Cache-Coherency (order='K')

New memory layout option that preserves input layout instead of forcing C-contiguous:

```python
# Memory layout options
NPY_ANYORDER = -1    # F if all inputs F, else C
NPY_CORDER = 0       # C-contiguous (row-major)
NPY_FORTRANORDER = 1 # Fortran-contiguous (column-major)
NPY_KEEPORDER = 2    # Match input layout (NEW)
```

**Performance impact:**
```python
a = np.arange(1000000).reshape(10,10,10,10,10,10)
timeit a + a.copy()           # 28.5 ms (C-contiguous)
timeit a.T + a.T.copy()       # 237 ms (8.3x slower without order='K')
```

### 2. Dimension Coalescing

Merge adjacent dimensions when memory is contiguous:
```
If strides[i+1] * shape[i+1] == strides[i]:
    Merge dimensions i and i+1
```

Enables single-loop iteration instead of nested loops.

### 3. Inner Loop Specialization

- Load constants once instead of repeatedly
- SSE/SIMD for aligned data
- Reduction operation optimizations

## Casting Modes

```c
typedef enum {
    NPY_NO_CASTING = 0,        // Identical types only
    NPY_EQUIV_CASTING = 1,     // + byte-swapped
    NPY_SAFE_CASTING = 2,      // Safe casts only
    NPY_SAME_KIND_CASTING = 3, // + same-kind casts
    NPY_UNSAFE_CASTING = 4     // Any casts
} NPY_CASTING;
```

## Performance Examples

### Image Compositing (19x speedup)
```python
# Poor memory layout: 3.51s
# With buffered iterator: 180ms
```

### Python-Level UFunc (2.3x speedup)
```python
# Standard: 138ms
timeit 3*a + b - (a/c)

# Lambda UFunc with iterator: 60.9ms
timeit luf(lambda a,b,c: 3*a + b - (a/c), a, b, c)
```

## NumSharp Implementation Patterns

### 1. Layout Detection

```csharp
public enum ArrayOrder {
    C,          // Row-major (default)
    Fortran,    // Column-major
    Any,        // F if all F, else C
    Keep        // Match input layout
}

public bool IsContiguous(ArrayOrder order) {
    // Check if strides match expected layout
}
```

### 2. Dimension Coalescing

```csharp
public (int[] shape, int[] strides) CoalesceDimensions() {
    var newShape = new List<int>();
    var newStrides = new List<int>();

    for (int i = 0; i < NDim; i++) {
        if (i > 0 && CanCoalesce(i-1, i)) {
            // Merge with previous dimension
            newShape[^1] *= Dimensions[i];
        } else {
            newShape.Add(Dimensions[i]);
            newStrides.Add(Strides[i]);
        }
    }
    return (newShape.ToArray(), newStrides.ToArray());
}

bool CanCoalesce(int i, int j) {
    return Strides[i] == Strides[j] * Dimensions[j];
}
```

### 3. Iterator with Buffering

```csharp
public class BufferedIterator<T> {
    private readonly T[] _buffer;
    private readonly int _bufferSize;

    public void Iterate(NDArray arr, Action<Span<T>> process) {
        // Copy chunks to buffer for cache-friendly access
        // Process buffer
        // Copy back if needed
    }
}
```

### 4. Keep-Order Output Allocation

```csharp
public NDArray AllocateOutput(params NDArray[] inputs) {
    // Determine best output layout from inputs
    var order = DetermineOptimalOrder(inputs);
    return np.empty(shape, dtype, order);
}
```

## Key Takeaways

| Optimization | When to Use | Speedup |
|--------------|-------------|---------|
| order='K' | Non-contiguous inputs | Up to 8x |
| Dimension coalescing | Many small dimensions | 2-4x |
| Buffering | Poor memory layout | Up to 20x |
| Inner specialization | Repeated operations | 2-3x |

## References

- [NEP 10 Full Text](https://numpy.org/neps/nep-0010-new-iterator-ufunc.html)
- `src/NumSharp.Core/Backends/Iterators/NDIterator.cs`
