# NumSharp.Interop.MLNet

Interop between **NumSharp** `NDArray` and **ML.NET** (`Microsoft.ML`) data — the `IDataView`
pipeline currency and the `VBuffer<T>` vector primitive.

Expose an `NDArray` to a trained ML.NET pipeline as an `IDataView` over the **very same unmanaged
buffer** — read lazily through the array's strides, no per-row POCO fill, no densifying copy — and
read a transformer's output column straight back into an `NDArray`. The whole pipeline bridge is two
verbs:

```csharp
using NumSharp;
using NumSharp.Interop.MLNet;

// A (rows, features) matrix in NumSharp -> feed a trained model -> read the scores back
using var featuresView = features.AsDataView("Features");     // no copy; any layout
IDataView output = transformer.Transform(featuresView);       // your ML.NET pipeline
using NDArray scores = output.ToNDArray("Score");             // straight into NumSharp
```

This is a **conversion library** in the mould of `NumSharp.Interop.OnnxRuntime` /
`NumSharp.Interop.pythonnet`: no engine backend, no `[ModuleInitializer]`, no native dependency.
It depends only on the standalone **`Microsoft.ML.DataView`** contract package (`IDataView`,
`VBuffer<T>`, `DataViewSchema`, `DataViewType`) and composes with whatever `Microsoft.ML` (MLContext,
trainers, transforms) you already reference — NuGet unifies the DataView dependency upward.

## The verbs

`As…` shares memory (a lazy, pinned view); `To…` copies.

| Verb | Direction | What it does |
|------|-----------|--------------|
| `nd.AsDataView("Features")` | NumSharp → ML.NET | An `IDataView` with **one vector column**: a 2-D `(R, C)` array becomes `R` rows of a length-`C` `VBuffer`; a 1-D `(C,)` array becomes one row (a single sample). Read lazily through the array's strides (any layout), ARC-pinned. |
| `nd.AsDataView(new[]{"a","b",...})` | NumSharp → ML.NET | An `IDataView` with **one scalar column per feature** (the tabular / training-data shape). |
| `nd.ToDataView(...)` | NumSharp → ML.NET | Same, over an independent snapshot (no lifetime coupling to the source). |
| `nd.ToVBuffer<T>()` | NumSharp → ML.NET | A dense `VBuffer<T>` (a copy — see [zero-copy note](#zero-copy-and-vbuffer)). |
| `view.ToNDArray("col")` | ML.NET → NumSharp | Materialize a column across all rows: a scalar column → 1-D `(R,)`; a fixed-size vector column → 2-D `(R, C)`. |
| `vbuffer.AsNDArray<T>()` | ML.NET → NumSharp | A **zero-copy** 1-D view over a dense `VBuffer<T>`'s own backing array (write-through), pinned for the view's lifetime; sparse/empty fall back to a copy. |
| `vbuffer.ToNDArray()` | ML.NET → NumSharp | A dense 1-D `NDArray` copy (sparse buffers are densified). |

### Zero-copy and VBuffer

`vbuffer.AsNDArray<T>()` is a genuine **zero-copy view** — it reaches the private `T[]` backing a dense
`VBuffer<T>` through a compiled-expression accessor, pins it, and wraps it write-through (mutations visible
both ways), exactly like the pythonnet bridge. **Caveat:** the view shares the VBuffer's array, so it is
valid only while that array is not refilled — a VBuffer from a cursor getter reuses one buffer across rows,
so use `ToNDArray()` (a copy) there; zero-copy is safe for a standalone VBuffer you own.

The **reverse** direction (`NDArray → VBuffer`) cannot be zero-copy: `VBuffer<T>` is backed by a managed
`T[]` at every ML.NET version (verified 2.0.0 / 3.0.0 / 4.0.2 — no `ReadOnlyMemory<T>`, no non-public
constructor), and a managed array cannot alias NumSharp's unmanaged buffer. `ToVBuffer<T>` copies. (The bulk
NumSharp→ML.NET path — `AsDataView` — is already zero-copy on its source; the VBuffer is a per-vector
convenience.)

`Postprocess.{Softmax, LogSoftmax, Sigmoid, Argmax, TopK}` turn a score column into a prediction with
`np.*` compositions instead of hand-written loops.

## Dtypes

Eleven dtypes map directly to an ML.NET column type — `bool` → `BooleanDataViewType`; the eight
integers, `float` and `double` → the matching `NumberDataViewType`. `Char` crosses as
`NumberDataViewType.UInt16` (UTF-16 code units), one-way. ML.NET has no half / decimal / complex
column type: **`Half` is converted to Single** (lossless) and **`Decimal` to Double** by the
view-producing verbs (in a temporary the view owns); **`Complex` is refused** (split into real /
imaginary planes with `np.real` / `np.imag`).

## Lifetime

`AsDataView` returns an `NDArrayDataView` (which, unlike a bare `IDataView`, **is** `IDisposable`):
dispose it — a `using`, or wire it into your `MLContext` lifetime — to release the buffer pin. The pin
takes its own ARC reference, so it is even safe to dispose the source array while the view (or its
cursors) still read it. Not disposing leaks the pin until finalization.

`NDArrayMLNetInterop.LiveExports` counts live views, for leak detection in tests.

## Dependency

The package references **`Microsoft.ML.DataView`** only. There is no standalone `Microsoft.ML.Data`
NuGet package, and `ITransformer` is not needed here: `IDataView` is ML.NET's universal currency, so
feeding a transformer and reading its output are both just conversions. Bring the `Microsoft.ML` you
already use for training / transforms.

## License

Apache-2.0, same as NumSharp.
