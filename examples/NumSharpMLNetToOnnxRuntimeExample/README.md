# NumSharpMLNetToOnnxRuntimeExample

A console app that trains a model with [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet) on data
held in NumSharp arrays, exports it to ONNX, and serves it with [ONNX Runtime](https://onnxruntime.ai) on NumSharp
arrays. It uses both bridges: [`NumSharp.Interop.MLNet`](../../src/NumSharp.Interop.MLNet/README.md) to train and
[`NumSharp.Interop.OnnxRuntime`](../../src/NumSharp.Interop.OnnxRuntime/README.md) to serve, with `Microsoft.ML`,
`Microsoft.ML.OnnxConverter` (ML.NET's ONNX exporter) and `Microsoft.ML.OnnxRuntime`.

Training needs ML.NET; serving the exported model needs only ONNX Runtime. The app checks that both runtimes give the
same prices for the same houses.

## Run it

You need the .NET 10 SDK. From this folder:

```bash
dotnet run -c Release
```

```text
trained     R² 0.9870, RMSE 15.25k
exported    1,833 bytes of ONNX
house 0      60 m², 2 rooms, 30 years: ONNX Runtime  136.1k, ML.NET  136.1k
house 1      95 m², 3 rooms,  5 years: ONNX Runtime  243.4k, ML.NET  243.4k
house 2     140 m², 4 rooms, 40 years: ONNX Runtime  321.8k, ML.NET  321.8k
house 3     180 m², 5 rooms, 12 years: ONNX Runtime  440.9k, ML.NET  440.9k
house 4     210 m², 6 rooms, 55 years: ONNX Runtime  484.7k, ML.NET  484.7k
largest difference between the runtimes 3.05E-05
live        ML.NET views 0, ONNX Runtime handles 0 / 0
```

The data is made with NumSharp's random generator, which is NumPy's, and SDCA trains on one thread, so the numbers are
the same on every run and every machine (Windows and Linux print the lines above).

## What each file shows

| File | What it shows |
|---|---|
| [`Program.cs`](Program.cs) | Training on a `(rows, 4)` array exposed as four named scalar columns (`AsDataView(names)`, then `Concatenate`, `NormalizeMinMax` and SDCA); `ConvertToOnnx` into memory; an `InferenceSession` over the bytes fed from NumSharp column slices through `session.Run`; the same rows scored by ML.NET and compared with NumSharp; an `NDScope` that releases every array the serving step makes. |
| [`Houses.cs`](Houses.cs) | The dataset: whole-array arithmetic on NumSharp's generator inside an `NDScope`, the table handed back with `scope.Returns`. |

## Notes

- **The exported model's inputs.** ML.NET's exporter declares one `[N, 1]` input per column of the training data,
  the label included, although scoring never reads it. Feed the label as zeros of its shape: left out, ONNX Runtime
  fails inside the graph with a message about an `Identity` node, not about the missing input.
- **The exported model's outputs.** One per column, named after it with `.output`: the predictions are
  `Score.output`, an `[N, 1]` float32 array.
- **Column slices.** `houses[":, 0:1"]` is a view with a stride; ONNX Runtime reads dense memory, so the bridge feeds a
  C-order copy for it (its `Auto` mode) and releases it with the call.
