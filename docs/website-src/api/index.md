---
uid: api-index
---

# NumSharp API Reference

Browse the complete API documentation for NumSharp, including all classes, methods, and properties.

## Namespaces

- @NumSharp - Core types including `NDArray`, `Shape`, `np`
- @NumSharp.Generic - Generic typed arrays like `NDArray<T>`
- @NumSharp.Backends - Storage and computation backends

## Key Types

| Type | Description |
|------|-------------|
| @NumSharp.NDArray | The main multi-dimensional array type |
| @NumSharp.Shape | Represents array dimensions and strides |
| @NumSharp.np | Static API class for NumPy-style operations |
| @NumSharp.Slice | Represents array slicing operations |

## Getting Started

```csharp
using NumSharp;

// Create arrays
var a = np.array(new[] { 1, 2, 3, 4, 5 });
var b = np.zeros((3, 4));
var c = np.random.randn(2, 3);

// Operations
var sum = np.sum(a);
var transposed = b.T;
var sliced = a["1:4"];
```
