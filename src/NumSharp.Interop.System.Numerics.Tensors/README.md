# NumSharp.Interop.System.Numerics.Tensors

Zero-copy interop between NumSharp's `NDArray` and the BCL's **`System.Numerics.Tensors`** types —
`Tensor<T>`, `TensorSpan<T>` and `ReadOnlyTensorSpan<T>`.

It is a **conversion library** in the mould of `NumSharp.Interop.OnnxRuntime` /
`NumSharp.Interop.MLNet`: it moves data (or rather, descriptions of the same memory) across the
boundary. It is **not** an engine backend — `System.Numerics.Tensors` does not compute NumSharp
operations, so there is no `TensorEngine` seam, no `[ModuleInitializer]`, and **no native
dependency**. Referencing the package changes nothing until you call a verb.

> Do **not** route NumSharp's own operations through `TensorPrimitives` — NumSharp's kernels are
> bit-exact with NumPy (and often faster than `TensorPrimitives`). This bridge is a **data** bridge.

## Verbs

`As…` shares memory (zero-copy); `To…` copies.

| Verb | Direction | Semantics |
|------|-----------|-----------|
| `nd.AsTensorSpan<T>()` → `TensorSpanHandle<T>` | NDArray → Tensors | **zero-copy** `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` over the NDArray's buffer, **any non-negative-stride layout** |
| `nd.ToTensor<T>()` → `Tensor<T>` | NDArray → Tensors | independent **dense copy** (any source layout, logical order) |
| `tensor.ToNDArray()` → `NDArray` | Tensors → NDArray | owning **copy** (the safe default) |
| `tensor.AsNDArray()` → `NDArray` | Tensors → NDArray | **zero-copy** view over the pinned tensor backing store |
| `span.ToNDArray()` | Tensors → NDArray | **copy** (a `ref struct` span cannot be leased) |

## Dtypes — all 15 cross as themselves

Unlike ONNX Runtime (a fixed `TensorElementType` enum with real gaps), `System.Numerics.Tensors`
containers are **unconstrained generics** over an unmanaged `T`. So **every** NumSharp dtype crosses
zero-copy as its own CLR type — `bool`, the eight integers, `char`, **`System.Half`** (directly, no
`Float16` wrapper), `float`, `double`, **`System.Decimal`** and **`System.Numerics.Complex`**. Nothing
is refused and nothing is converted.

## Layout — strided views share, negative strides copy

A `TensorSpan<T>` carries explicit lengths + strides, so a **sliced, transposed, strided or broadcast**
NDArray view shares zero-copy (a broadcast view through `ReadOnlySpan` only — writing overlapping
stride-0 lanes would corrupt data). The one refusal is a **negative-stride** view (a reversed slice
`a[::-1]`): `System.Numerics.Tensors` forbids negative strides — materialize it first
(`np.ascontiguousarray(nd)`, `nd.copy()`) or use `ToTensor`.

## Example

```csharp
using System.Numerics.Tensors;
using NumSharp;
using NumSharp.Interop.Tensors;

var nd = np.arange(6).reshape(2, 3).astype(NPTypeCode.Single);

// Zero-copy: mutate the NDArray through a TensorSpan.
using (var h = nd.AsTensorSpan<float>())
{
    TensorSpan<float> span = h.Span;          // shares nd's buffer
    for (nint i = 0; i < span.Lengths[0]; i++)
        for (nint j = 0; j < span.Lengths[1]; j++)
            span[new[] { i, j }] += 10f;       // writes straight into nd
}

// Read a tensor produced elsewhere back as an NDArray, no copy.
var output = Tensor.Create(new[] { 0.1f, 0.7f, 0.2f }, new nint[] { 3 });
using NDArray probs = output.AsNDArray();
long best = np.argmax(probs).GetInt64(0);      // 1
```

## Frameworks

Builds assemblies for **net8.0** and **net10.0**; **net9.0** and **net11.0** consumers are served by the
net8.0 / net10.0 assets via NuGet nearest-TFM selection. Depends only on the managed
`System.Numerics.Tensors` package.

## Experimental

`Tensor<T>`/`TensorSpan<T>` are `[Experimental("SYSLIB5001")]` in the BCL; this package re-marks its own
surface `[Experimental("NUMSHARP_TENSORS")]` so you acknowledge that churn once. Suppress with
`<NoWarn>NUMSHARP_TENSORS</NoWarn>` or `#pragma warning disable NUMSHARP_TENSORS`.
