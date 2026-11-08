// NumSharpMLNetToOnnxRuntimeExample: train a model in ML.NET on NumSharp arrays, export it to ONNX, and serve it with
// ONNX Runtime on NumSharp arrays.
//
// The route that uses both bridges. NumSharp.Interop.MLNet hands the training table to ML.NET as an IDataView; ML.NET's
// own exporter writes the trained pipeline as an ONNX model; NumSharp.Interop.OnnxRuntime feeds new rows to that model
// from NumSharp memory. The service that serves the model needs ONNX Runtime only, not ML.NET.
//
//     dotnet run -c Release
using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.Trainers;
using NumSharp;
using NumSharp.Interop.MLNet;
using NumSharp.Interop.OnnxRuntime;
using NumSharpMLNetToOnnxRuntimeExample;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
string[] features = ["Size", "Rooms", "Age"];
var ml = new MLContext(seed: 1);

// 1. Data made in NumSharp: 2,000 houses as a (2000, 4) float32 table - size, rooms, age, price.
using NDArray table = Houses.Make(count: 2000, seed: 42);

// 2. Train in ML.NET on the table's four named scalar columns: concatenate the features, scale them, fit a linear
//    regression (SDCA on one thread, so every run prints the same numbers). 3. Export the trained pipeline as an ONNX
//    model; the exporter takes the training data for its schema.
ITransformer model;
byte[] onnx;
using (NDArrayDataView train = table.AsDataView([.. features, "Label"]))
{
    var sdca = ml.Regression.Trainers.Sdca(new SdcaRegressionTrainer.Options { NumberOfThreads = 1, MaximumNumberOfIterations = 100 });
    model = ml.Transforms.Concatenate("Features", features)
        .Append(ml.Transforms.NormalizeMinMax("Features"))
        .Append(sdca)
        .Fit(train);
    var metrics = ml.Regression.Evaluate(model.Transform(train));
    Console.WriteLine($"trained     R² {metrics.RSquared:F4}, RMSE {metrics.RootMeanSquaredError:F2}k");

    using var stream = new MemoryStream();
    ml.Model.ConvertToOnnx(model, train, stream);
    onnx = stream.ToArray();
    Console.WriteLine($"exported    {onnx.Length:N0} bytes of ONNX");
}

// 4. Serve it with ONNX Runtime. The exported model declares one [N, 1] input per column of the training data - the
//    label too, which scoring never reads, so it is fed as zeros - and one output per column: Score.output holds the
//    predictions. Each feature is a column slice of the new rows; the bridge feeds the C-order copy each one needs.
//    5. Score the same rows with ML.NET itself: both runtimes give the same prices. The scope releases every array
//    made here (the slices, the outputs, the comparison) when it closes.
using (var session = new InferenceSession(onnx))
using (var scope = NDScope.Open())
{
    NDArray houses = np.array(new float[,] { { 60, 2, 30 }, { 95, 3, 5 }, { 140, 4, 40 }, { 180, 5, 12 }, { 210, 6, 55 } });
    var feed = new Dictionary<string, NDArray>
    {
        ["Size"] = houses[":, 0:1"], ["Rooms"] = houses[":, 1:2"], ["Age"] = houses[":, 2:3"],
        ["Label"] = np.zeros(new Shape(houses.shape[0], 1), np.float32),
    };
    NDArray served = session.Run(feed, ["Score.output"])["Score.output"];               // (5, 1) float32

    NDArray scored;
    using (NDArrayDataView rows = houses.AsDataView(features))
        scored = model.Transform(rows).ToNDArray("Score");                              // (5,) float32

    for (int i = 0; i < houses.shape[0]; i++)
        Console.WriteLine($"house {i}     {houses.GetSingle(i, 0),3:F0} m², {houses.GetSingle(i, 1):F0} rooms, {houses.GetSingle(i, 2),2:F0} years: " +
                          $"ONNX Runtime {served.GetSingle(i, 0),6:F1}k, ML.NET {scored.GetSingle(i),6:F1}k");
    NDArray difference = np.abs(served.reshape(-1) - scored);
    Console.WriteLine($"largest difference between the runtimes {(float)np.max(difference):G3}");
}

Console.WriteLine($"live        ML.NET views {NDArrayMLNetInterop.LiveExports}, ONNX Runtime handles {NDArrayOnnxInterop.LiveExports} / {NDArrayOnnxInterop.LiveImports}");
