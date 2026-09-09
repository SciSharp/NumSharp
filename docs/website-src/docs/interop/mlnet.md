# ML.NET — feed a pipeline from NumSharp memory, read it back into NumSharp

`NumSharp.Interop.MLNet` connects `NDArray` to [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)'s
data model — the `IDataView` pipeline currency and the `VBuffer<T>` vector primitive. An array becomes an
`IDataView` that a trained `Microsoft.ML` pipeline reads **lazily through the array's own strides** (no
per-row POCO fill, no densifying copy), and a transformer's output column comes straight back as an
`NDArray`, ready for `np.argmax`, `Postprocess.Softmax`, `TopK` and everything else NumSharp does. The
whole "shape my data into an `IDataView` / read the scores out of one" boilerplate every ML.NET sample
hand-writes with a POCO class and a `PredictionEngine` becomes two verbs.

**On this page:** [From zero to a prediction](#from-zero-to-a-prediction) · [The verbs](#the-verbs) ·
[The two shapes: vector column vs scalar columns](#the-two-shapes) · [Lifetime](#lifetime-who-frees-what) ·
[Post-processing](#post-processing) · [Dtypes](#dtypes) · [Why only `Microsoft.ML.DataView`](#dependency) ·
[Limits](#limits)

> Verified on ML.NET 2.0.1 (the floor — `Microsoft.ML.DataView` 2.0.0) and 4.0.2 · net8.0/net10.0 ·
> Windows, Linux, macOS. Every claim below is reproduced by a test in `NumSharp.Tests.Interop.MLNet`
> (real `Microsoft.ML` transforms and trainers; no Python at test time).

---

## From zero to a prediction

```bash
dotnet add package NumSharp.Interop.MLNet
dotnet add package Microsoft.ML                    # the pipeline runtime you already use for training
```

The interop depends only on `Microsoft.ML.DataView` — the standalone package that defines `IDataView`,
`VBuffer<T>`, `DataViewSchema` and `DataViewType`. ML.NET's runtime (MLContext, transforms, trainers)
lives in `Microsoft.ML`, which you reference to train or load a model anyway; the package composes with
whichever version you bring (NuGet unifies the DataView dependency upward).

```csharp
using NumSharp;
using NumSharp.Interop.MLNet;
using Microsoft.ML;

// A (rows, features) matrix in NumSharp.
NDArray features = np.array(new float[,] { { 5.1f, 3.5f }, { 4.9f, 3.0f }, { 6.3f, 3.3f } });

// Feed it to a trained pipeline as an IDataView, read the Score column straight back into NumSharp.
using NDArrayDataView input = features.AsDataView("Features");
IDataView output = trainedModel.Transform(input);
using NDArray scores = output.ToNDArray("Score");

using NDArray predictions = Postprocess.Argmax(scores, axis: -1);   // np, not a hand-written loop
```

That is the whole bridge. `IDataView` is ML.NET's universal currency — everything a pipeline consumes and
produces is an `IDataView` — so once an `NDArray` can *become* one and a column can be *read out* of one,
you have bridged training-data preparation, transforms and predictions alike, with no per-transformer
runner and no POCO class.

---

## The verbs

`As…` shares memory (a lazy, pinned view of the array); `To…` copies.

| Verb | Direction | What it does |
|------|-----------|--------------|
| `nd.AsDataView("Features")` | NumSharp → ML.NET | An `IDataView` with **one vector column** — the feature-matrix shape a model consumes. Reads the array lazily through its strides (any layout), ARC-pinned. |
| `nd.AsDataView(new[]{"a","b",…})` | NumSharp → ML.NET | An `IDataView` with **one scalar column per feature** — the tabular / training-data shape. |
| `nd.ToDataView(…)` | NumSharp → ML.NET | Same, over an independent snapshot (mutate/dispose the source freely afterward). |
| `nd.ToVBuffer<T>()` | NumSharp → ML.NET | A dense `VBuffer<T>` (always a copy — see [Limits](#limits)). |
| `view.ToNDArray("col")` | ML.NET → NumSharp | Materialize a column across all rows: a scalar column → 1-D `(R,)`; a fixed-size vector column → 2-D `(R, C)`. |
| `vbuffer.ToNDArray()` | ML.NET → NumSharp | A dense 1-D `NDArray` (sparse buffers are densified: absent entries become zero). |

The input view reads **any layout** through the array's strides — C-contiguous, Fortran, sliced,
transposed, negative-stride and broadcast views all work with no densifying copy, because the cursor
computes each element's physical offset itself. (`AsOrtValue` in the ONNX bridge needs C-contiguity;
`AsDataView` does not, because ML.NET pulls values one row at a time rather than handing a dense pointer
to a native runtime.)

---

## The two shapes

ML.NET has two idioms for "a table of numbers", and the two `AsDataView` overloads produce each:

**One vector column** (`AsDataView("Features")`) — the shape a trained pipeline consumes. A 2-D `(R, C)`
array becomes `R` rows, each a length-`C` `VBuffer`; a 1-D `(C,)` array becomes a single row (one sample).

```csharp
using var v = features.AsDataView("Features");   // (150, 4) -> 150 rows of a 4-vector
```

**One scalar column per feature** (`AsDataView(string[] names)`) — the shape for building training data or
feeding pipelines that reference named scalar columns (then `Concatenate` them into `"Features"`). A 2-D
`(R, C)` array becomes `R` rows and `C` named scalar columns; a 1-D `(R,)` array with one name becomes `R`
rows and one scalar column.

```csharp
using var t = data.AsDataView(new[] { "SepalLength", "SepalWidth", "Label" });   // (150, 3) table
var pipeline = ml.Transforms.Concatenate("Features", "SepalLength", "SepalWidth")
    .Append(ml.Regression.Trainers.Sdca(labelColumnName: "Label"));
var model = pipeline.Fit(t);
```

Reading a column back mirrors the two shapes: a scalar `"Score"` column of `R` rows comes back as a 1-D
`(R,)` array; a fixed-size vector column of width `C` comes back as `(R, C)`. A variable-length vector
column (rows disagree on length) is refused — NumSharp arrays are rectangular; pad the column to a fixed
size in the pipeline first.

---

## Lifetime: who frees what

`AsDataView` returns an `NDArrayDataView`. Unlike a bare `IDataView`, it **is** `IDisposable`: dispose it
(a `using`, or wire it into your `MLContext`'s lifetime) to release the pin it holds on the NumSharp
buffer. The pin takes its own atomic reference, so:

- The array stays valid for the view even if you dispose or drop the source array while the pipeline still
  reads it — the opposite of a raw ML.NET data source over a managed array, which would read freed data.
- Every cursor takes its **own** pin too, so a cursor mid-iteration survives the view being disposed
  underneath it.
- Not disposing the view leaks the pin until the finalizer reclaims it (a safety net, not a lifetime
  model).

`ToDataView` copies first, so its view has no coupling to the source at all.

`NDArrayMLNetInterop.LiveExports` counts live views — the tests assert it returns to its baseline after
every case, so each test doubles as a no-leak gate.

---

## Post-processing

`Postprocess` turns a score column into a prediction with `np.*` compositions instead of hand-written
loops — the softmax / argmax / top-k every classifier re-implements over a `float[]`:

```csharp
using NDArray scores = output.ToNDArray("Score");              // (batch, classes)
using NDArray probs  = Postprocess.Softmax(scores);            // stable exp(x-max)/Σ, rows sum to 1
using NDArray top    = Postprocess.Argmax(probs, axis: -1);    // class id, first on ties
var (values, idx)    = Postprocess.TopK(scores, k: 5);         // sorted, ties lower-index-first
```

`Softmax`, `LogSoftmax`, `Sigmoid`, `Argmax`, `TopK` — nothing new, just `np.max` / `np.exp` / `np.sum` /
`np.argmax` / `np.argsort` / `np.take_along_axis`.

---

## Dtypes

Eleven dtypes map directly to an ML.NET column type:

| NumSharp | ML.NET column type |
|----------|--------------------|
| `Boolean` | `BooleanDataViewType` |
| `Byte` / `SByte` / `Int16` / `UInt16` / `Int32` / `UInt32` / `Int64` / `UInt64` | the matching `NumberDataViewType` |
| `Single` / `Double` | `NumberDataViewType.Single` / `.Double` |
| `Char` | `NumberDataViewType.UInt16` (UTF-16 code units, one-way — reads back as `UInt16`) |

ML.NET has no half, decimal or complex column type, so the view-producing verbs **convert**:
`Half` → `Single` (lossless — every `Half` is an exact `Single`) and `Decimal` → `Double` (lossy beyond
~16 significant digits), in a temporary the view owns. `Complex` is refused — split it into real and
imaginary planes with `np.real` / `np.imag` and feed those.

---

## Dependency

The package references **`Microsoft.ML.DataView`** alone — the standalone contract package. There is no
separate `Microsoft.ML.Data` NuGet package (ML.NET ships `Microsoft.ML.Data.dll` inside the full
`Microsoft.ML`), and `ITransformer` is deliberately **not** needed: because `IDataView` is the universal
currency, feeding a transformer (`transformer.Transform(nd.AsDataView(...))`) and reading its output
(`output.ToNDArray("Score")`) are both just conversions this package provides. So the dependency stays the
lightweight contract, and the heavy `Microsoft.ML` you already reference for training composes on top —
the exact mirror of how `NumSharp.Interop.OnnxRuntime` depends on `Microsoft.ML.OnnxRuntime.Managed` and
leaves the native runtime to the consumer.

---

## Limits

- **`VBuffer<T>` is copy-only, both ways.** `VBuffer<T>` exposes no public constructor over foreign memory
  (its only public constructors take a managed `T[]`), by design — it is polymorphic between dense and
  sparse and pools its buffers — so a zero-copy `VBuffer` over NumSharp's unmanaged buffer is not possible
  through the public API. `ToVBuffer<T>` copies out; `vbuffer.ToNDArray()` copies in. The **`IDataView`**
  path is where the sharing happens: the view holds the array live and reads it lazily.
- **Rank 1 or 2 only.** An `IDataView` is a table of rows; `AsDataView` accepts 1-D and 2-D arrays.
  Reshape higher-rank data first.
- **No `Complex` column.** ML.NET has no complex type (see [Dtypes](#dtypes)).
- **`ToNDArray` rectangularizes.** A variable-length vector column cannot become a rectangular NDArray and
  is refused with a message naming the cause.
