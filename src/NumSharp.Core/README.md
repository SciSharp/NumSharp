# NumSharp

**NumPy for .NET** — a native .NET array library with a NumPy-shaped API: `NDArray`,
broadcasting, slicing views, dtype-aware `np.*` functions, unmanaged storage, and
runtime-generated SIMD kernels. The compatibility target is **NumPy 2.x**; where behavior
differs, NumPy is treated as the source of truth.

`NumSharp.Core` is **100% managed C#** with no native dependency and no P/Invoke — every
kernel is its own managed code. An optional OpenBLAS matrix-product backend ships separately
as [`NumSharp.Interop.OpenBLAS`](https://www.nuget.org/packages/NumSharp.Interop.OpenBLAS).

## Install

```bash
dotnet add package NumSharp
```

## Quick start

```csharp
using NumSharp;

var a = np.arange(12).reshape(3, 4);
var window = a[":, 1::2"];

Console.WriteLine(window);
Console.WriteLine(np.sum(window, axis: 0));
```

For Python readers, the shape is deliberately close:

```python
import numpy as np

a = np.arange(12).reshape(3, 4)
print(a[:, 1::2].sum(axis=0))
```

## Features

- **NumPy-style `NDArray`** — N-dimensional arrays with shape, strides, offsets, slicing, and
  view semantics (slices return views that share memory).
- **Broadcasting** — NumPy-style shape expansion without materializing repeated values.
- **Dtype-aware operations** — 15 core dtypes with NumPy-oriented promotion (NEP50) and
  conversion behavior.
- **Broad `np.*` surface** — creation, manipulation, math, reductions, comparisons, logic,
  linear algebra, FFT, random sampling, and `.npy`/`.npz` I/O.
- **Generated IL + SIMD kernels** — runtime-specialized kernels (V128/V256/V512) for supported
  dtype and layout combinations.

## Links

- Documentation: <https://scisharp.github.io/NumSharp/>
- Getting started: <https://scisharp.github.io/NumSharp/docs/NDArray.html>
- API reference: <https://scisharp.github.io/NumSharp/api/>
- Coverage & support dashboard: <https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html>
- Source: <https://github.com/SciSharp/NumSharp>

Licensed under the Apache License 2.0.
