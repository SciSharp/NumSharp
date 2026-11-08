# NumSharpMLNetExample

A console app that trains, evaluates and uses [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)
models on data held in NumSharp arrays, with no POCO class and no row-by-row fill. It references
[`NumSharp.Interop.MLNet`](../../src/NumSharp.Interop.MLNet/README.md) and `Microsoft.ML`.

ML.NET reads all of its data through `IDataView`. `AsDataView` makes an `NDArray` one, reading the
array's own memory row by row while a pipeline trains or scores; `ToNDArray` reads a column of a
pipeline's output back into an `NDArray`, where NumSharp takes over.

## Run it

You need the .NET 10 SDK. From this folder:

```bash
dotnet run -c Release
```

```text
classifier  accuracy 93.5 % on 200 test rows, mean confidence 92.5 %
ML.NET      accuracy 93.5 %, AUC 0.982
k-means     clusters match the clouds for 93.8 % of 1000 points
live views  0
```

The data is two overlapping clouds of points made with NumSharp's random generator, which is NumPy's,
so the numbers are the same on every machine.

## What each file shows

| File | What it shows |
|---|---|
| [`Program.cs`](Program.cs) | Training a binary classifier from a `(rows, 5)` array exposed as five named scalar columns; scoring test rows and reading `PredictedLabel` and `Probability` back as `NDArray`s, where an accuracy is one comparison and a mean (and agrees with ML.NET's own evaluation); k-means on the features as one vector column, the cluster ids read back as a `uint32` array. |
| [`Blobs.cs`](Blobs.cs) | The dataset: `RandomState.normal`, `np.hstack` / `np.vstack`, a shuffle by fancy indexing with `permutation`, and the split into training and test rows. |

## Notes

- **The two shapes.** `AsDataView("Features")` gives one vector column, the shape a trained model
  reads; `AsDataView(names)` gives one scalar column per array column, the shape for training data
  that a pipeline concatenates (`Concatenate("Features", names)`).
- **The label.** One view exposes one array, so every column has the array's dtype: the label travels
  as a float column and `ConvertType` turns it into the boolean that binary trainers want. That step is
  fitted on its own, outside the model, so the trained model reads the four feature columns and
  nothing else, and new data needs no label column to be scored. (A model whose chain maps the label,
  such as ML.NET's `MapValueToKey("Label")` recipe for multiclass training, also asks for a `Label`
  column when it scores.)
- **The views read NumSharp's memory.** `AsDataView` copies nothing: the cursor reads each value in
  place, through the array's strides, so a slice such as `test[":, :4"]` is exposed as it is. Each view
  holds a reference on the array's buffer until it is disposed; `NDArrayMLNetInterop.LiveExports`
  counts the views still alive, and is back to 0 at the end.
- **Every array is disposed.** NumSharp arrays live in pooled unmanaged memory. The `using`
  declarations give each buffer back as soon as its array is no longer needed, which is also what keeps
  the build free of the `NDW012` leak warnings that the NumSharp package's analyzer reports.
