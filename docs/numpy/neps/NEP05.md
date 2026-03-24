# NEP 05 - Generalized Universal Functions (gufuncs)

**Status:** Final
**NumSharp Impact:** MEDIUM - Affects operation signatures like matmul, dot

## Summary

Extends ufuncs to operate on sub-arrays instead of just scalars. Enables "sub-array by sub-array" operations with defined core dimensions.

## Key Concept

Regular ufuncs operate element-by-element. Generalized ufuncs operate on sub-arrays with a defined **signature** specifying core dimensions.

## Signature Syntax

```
<Signature> ::= <Input arguments> "->" <Output arguments>
<Argument> ::= "(" <Core dimension list> ")"
```

### Examples

| Function | Signature | Description |
|----------|-----------|-------------|
| `add` | `(),()->()` | Two scalars → scalar |
| `inner1d` | `(i),(i)->()` | Two 1-D arrays → scalar |
| `matmul` | `(m,n),(n,p)->(m,p)` | Matrix multiplication |
| `sum1d` | `(i)->()` | 1-D array → scalar |

## Broadcasting with gufuncs

1. Core dimensions are mapped to the **last dimensions** of arrays
2. Remaining dimensions are "loop dimensions" and are broadcast
3. Same dimension name = same size (or broadcastable)

### Example: `inner1d` with signature `(i),(i)->()`

```
Input a: (3, 5, N)  →  Core dim i=N, loop dims (3, 5)
Input b: (5, N)     →  Core dim i=N, loop dims (5,) → broadcast to (3, 5)
Output:  (3, 5)     →  No core dims, just loop dims
```

## NumSharp Relevance

NumSharp operations that follow gufunc semantics:
- `np.dot` / `np.matmul` - matrix multiplication
- `np.tensordot` - tensor contraction
- `np.inner` - inner product
- `np.outer` - outer product

### Implementation Pattern

When implementing gufunc-like operations:
1. Extract core dimensions from last axes
2. Broadcast remaining "loop" dimensions
3. Apply operation over core dimensions
4. Output shape = loop dims + output core dims

## References

- [NEP 5 Full Text](https://numpy.org/neps/nep-0005-generalized-ufuncs.html)
