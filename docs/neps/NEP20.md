# NEP 20 - Expansion of Generalized Universal Function Signatures

**Status:** Final
**NumSharp Impact:** MEDIUM - Extends NEP 5 gufunc signatures

## Summary

Adds two signature enhancements to generalized ufuncs:
1. **Frozen dimensions** - Fixed size requirements (e.g., size-3 for cross product)
2. **Flexible dimensions** - Optional dimensions that can be absent

## Signature Syntax

```
<Core dimension> ::= <name> <modifier>
<name> ::= variable_name | integer
<modifier> ::= "" | "?"
```

## Frozen (Fixed-Size) Dimensions

Use integer instead of variable name to require specific size.

### Examples

| Signature | Description |
|-----------|-------------|
| `()->(2)` | Polar angle → 2D cartesian unit vector |
| `(3),(3)->(3)` | Cross product of two 3-vectors |
| `(),()->(3)` | Two angles → 3D unit vector |

### Implementation

```csharp
// Validate frozen dimensions
public void ValidateFrozenDimension(int axis, int requiredSize) {
    if (Shape[axis] != requiredSize) {
        throw new ArgumentException(
            $"Axis {axis} must have size {requiredSize}, got {Shape[axis]}");
    }
}

// Cross product requires size 3
public static NDArray Cross(NDArray a, NDArray b) {
    a.ValidateFrozenDimension(-1, 3);
    b.ValidateFrozenDimension(-1, 3);
    // ... implementation
}
```

## Flexible (Optional) Dimensions

Suffix with `?` to make dimension optional.

### Matrix Multiplication Example

**Signature:** `(m?,n),(n,p?)->(m?,p?)`

This single signature covers four cases:

| Case | Input Shapes | Output Shape |
|------|--------------|--------------|
| matrix × matrix | `(m,n),(n,p)` | `(m,p)` |
| vector × matrix | `(n),(n,p)` | `(p)` |
| matrix × vector | `(m,n),(n)` | `(m)` |
| vector × vector | `(n),(n)` | `()` (scalar) |

### Implementation

```csharp
// Handle optional dimensions in matmul
public static NDArray Matmul(NDArray a, NDArray b) {
    bool aIsVector = a.NDim == 1;
    bool bIsVector = b.NDim == 1;

    // Expand vectors to 2D for computation
    var a2d = aIsVector ? a.reshape(1, -1) : a;
    var b2d = bIsVector ? b.reshape(-1, 1) : b;

    // Compute matrix product
    var result = MatrixProduct(a2d, b2d);

    // Squeeze output based on input shapes
    if (aIsVector && bIsVector) return result.squeeze();  // scalar
    if (aIsVector) return result.squeeze(axis: 0);        // (p,)
    if (bIsVector) return result.squeeze(axis: -1);       // (m,)
    return result;                                         // (m,p)
}
```

## NumSharp Functions Affected

| Function | Signature | Frozen | Flexible |
|----------|-----------|--------|----------|
| `np.cross` | `(3),(3)->(3)` | Yes (size 3) | No |
| `np.dot` | `(n),(n)->()` | No | No |
| `np.matmul` | `(m?,n),(n,p?)->(m?,p?)` | No | Yes |
| `np.inner` | `(n),(n)->()` | No | No |
| `np.outer` | `(m),(n)->(m,n)` | No | No |

## Validation Patterns

### Frozen Dimension Check

```csharp
public static void ValidateSignature(
    string signature,
    params NDArray[] arrays)
{
    // Parse signature for frozen dimensions
    // e.g., "(3),(3)->(3)" requires size 3

    var frozenDims = ParseFrozenDimensions(signature);
    foreach (var (arrayIdx, axis, size) in frozenDims) {
        if (arrays[arrayIdx].Shape[axis] != size) {
            throw new ValueError(
                $"Input {arrayIdx} axis {axis} must be {size}");
        }
    }
}
```

### Flexible Dimension Handling

```csharp
public static (NDArray[], int[]) PrepareFlexibleInputs(
    string signature,
    params NDArray[] arrays)
{
    var squeezedAxes = new List<int>();
    var prepared = new List<NDArray>();

    // Identify which flexible dims are missing
    // Expand missing dims to size 1
    // Track which to squeeze in output

    return (prepared.ToArray(), squeezedAxes.ToArray());
}
```

## References

- [NEP 20 Full Text](https://numpy.org/neps/nep-0020-gufunc-signature-enhancement.html)
- [NEP 5 - Basic gufuncs](NEP05.md)
