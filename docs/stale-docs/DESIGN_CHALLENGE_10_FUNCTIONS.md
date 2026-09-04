# Design Challenge: 10 Diverse NumSharp Functions

Testing the dynamic `yield return` approach against 10 challenging functions.

---

## The 10 Functions

| # | Function | Challenge | How Dynamic Approach Handles It |
|---|----------|-----------|--------------------------------|
| 1 | `np.unique` | Variable return count (1-4) based on flags | Just works - compare whatever comes back |
| 2 | `np.modf` | Fixed tuple return (2 arrays) | Just works - tuples compare element-wise |
| 3 | `np.linspace` | Multiple params, edge cases | `Vary()` for custom values |
| 4 | `np.meshgrid` | String param (`indexing`) | `Vary("xy", "ij")` |
| 5 | `np.concatenate` | Array tuple input | Custom `array_tuples` marker |
| 6 | `np.dot` | Behavior varies by ndim | Test all combos, errors are valid results |
| 7 | `np.searchsorted` | String param (`side`) | `Vary("left", "right")` |
| 8 | `np.nonzero` | Returns ndim-tuple | Just works - compare tuple lengths |
| 9 | `np.random.choice` | Probability array param | Context-aware `probs` marker |
| 10 | `np.broadcast_to` | Shape compatibility | Context-aware `broadcast_shapes` marker |

---

## Contract Definitions

```csharp
public class ChallengeContracts : Contract
{
    public override IEnumerable<dynamic> TestCases()
    {
        // 1. np.unique - variable return count based on flags
        //    No special handling needed - framework compares whatever NumPy returns
        yield return np.unique(arrays);
        yield return np.unique(arrays, return_index: bools);
        yield return np.unique(arrays, return_index: bools, return_inverse: bools);
        yield return np.unique(arrays, return_index: bools, return_inverse: bools, return_counts: bools);

        // 2. np.modf - always returns (fractional, integral) tuple
        yield return np.modf(arrays);
        yield return np.modf(arrays_float);

        // 3. np.linspace - scalar params with edge cases
        yield return np.linspace(scalars, scalars, Vary(0, 1, 2, 50, 100));
        yield return np.linspace(Vary(0, -10, 1e10), Vary(1, 10, -1e10), 50);

        // 4. np.meshgrid - string indexing parameter
        yield return np.meshgrid(arange(3), arange(4), indexing: Vary("xy", "ij"));
        yield return np.meshgrid(arange(5), arange(3), indexing: Vary("xy", "ij"), sparse: bools);

        // 5. np.concatenate - tuple of arrays
        yield return np.concatenate(array_tuples, axis: Vary(0, 1, -1));
        yield return np.concatenate(array_tuples_3, axis: 0);  // 3 arrays

        // 6. np.dot - behavior varies by ndim, incompatible shapes should error
        yield return np.dot(arrays, arrays);  // All combos, errors are valid
        yield return np.dot(arange(6), arange(6));  // 1D inner product
        yield return np.dot(zeros(3, 4), zeros(4, 5));  // 2D matmul
        yield return np.dot(arange(6), zeros(3, 4));  // Should error

        // 7. np.searchsorted - side parameter
        yield return np.searchsorted(sorted_arrays, scalars, side: Vary("left", "right"));
        yield return np.searchsorted(arange(10), Vary(0, 5, 9, 10, -1, 100), side: Vary("left", "right"));

        // 8. np.nonzero - returns tuple with length = ndim
        yield return np.nonzero(arrays);
        yield return np.nonzero(eye(5));
        yield return np.nonzero(zeros(3, 4, 5));  // Returns 3-tuple of empty arrays

        // 9. np.random.choice - probability array must match size
        yield return np.random.choice(Vary(5, 10), size: shapes, replace: bools, p: probs);
        yield return np.random.choice(5, p: Vary(null, array(0.1, 0.2, 0.3, 0.2, 0.2)));
        yield return np.random.choice(3, p: array(0.5, 0.5, 0.5));  // Invalid - should error

        // 10. np.broadcast_to - shape must be compatible
        yield return np.broadcast_to(arrays, broadcast_shapes);
        yield return np.broadcast_to(arange(4), Vary(new[]{3,4}, new[]{2,3,4}));
        yield return np.broadcast_to(zeros(3, 1), Vary(new[]{3,4}, new[]{2,3,4}));
        yield return np.broadcast_to(arange(4), new[]{3,3});  // Should error
    }

    // Custom markers
    Marker array_tuples => new("array_tuples");      // Pairs of compatible arrays
    Marker array_tuples_3 => new("array_tuples_3");  // Triples of compatible arrays
    Marker sorted_arrays => new("sorted_arrays");
    Marker probs => new("probs");                    // Context-aware probabilities
    Marker broadcast_shapes => new("broadcast_shapes");
}
```

---

## Why This Works

### No Special Attributes Needed

The old design required:
- `[ConditionalReturns]` for np.unique
- `[NdimMatchingReturns]` for np.nonzero
- `[ArraySequenceParameter]` for np.concatenate
- `[VariadicInput]` for np.meshgrid
- etc.

The dynamic approach needs **none of these**. The framework simply:
1. Runs the Python code
2. Runs the C# code
3. Compares whatever comes back

### Variable Returns Just Work

```csharp
// np.unique with all flags = returns 4 arrays
yield return np.unique(arrays, return_index: true, return_inverse: true, return_counts: true);

// NumPy returns: (unique, indices, inverse, counts)
// NumSharp returns: (NDArray, NDArray, NDArray, NDArray)
// Comparison: Element-wise on each tuple position
```

No need to declare the return structure - just compare what you get.

### Errors Are Valid Test Results

```csharp
yield return np.dot(arange(6), zeros(3, 4));  // Incompatible shapes
```

- NumPy throws: `ValueError: shapes (6,) and (3,4) not aligned`
- NumSharp throws: `IncorrectShapeException: Shapes not aligned for dot product`
- **Both threw = PASS** (same behavior, lenient by default)

The report shows both messages for review, but doesn't fail on text differences.

### Context-Aware Markers Handle Dependencies

```csharp
yield return np.random.choice(Vary(5, 10), p: probs);
```

The `probs` marker sees that `a=5` or `a=10` and generates:
- For a=5: `null`, `[0.2, 0.2, 0.2, 0.2, 0.2]`, `[0.9, 0.025, ...]`
- For a=10: `null`, `[0.1, 0.1, ...]`, etc.

### Inline Vary() for Edge Cases

```csharp
yield return np.searchsorted(arange(10), Vary(0, 5, 9, 10, -1, 100), side: Vary("left", "right"));
```

Generates 12 test cases (6 values × 2 sides) with one line.

---

## Generated Test Counts

| Function | Contract Lines | Expanded Cases |
|----------|---------------|----------------|
| np.unique | 4 | 16+ (2 arrays × 2³ flag combos) |
| np.modf | 2 | 4+ |
| np.linspace | 2 | 50+ |
| np.meshgrid | 2 | 8+ |
| np.concatenate | 2 | 6+ |
| np.dot | 4 | 20+ |
| np.searchsorted | 2 | 24+ |
| np.nonzero | 3 | 6+ |
| np.random.choice | 3 | 36+ |
| np.broadcast_to | 4 | 12+ |
| **Total** | **~28** | **~180+** |

---

## Conclusion

The dynamic `yield return` approach handles all 10 challenges with:
- **No special attributes** — Just yield what you want to test
- **No complex type system** — Compare whatever comes back
- **Semantic yields** — Descriptions, not execution
- **Python-first execution** — Python creates artifacts, NumSharp validates
- **Chained verification** — Every intermediate step is compared
- **Grid expansion** — `np.dot(arrays, arrays)` = N² combinations
- **Errors are valid** — Both throwing = PASS

The framework's job is simple: run Python, save artifacts, run NumSharp, compare.
