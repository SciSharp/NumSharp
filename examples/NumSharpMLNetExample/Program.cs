// NumSharpMLNetExample: train, evaluate and use ML.NET models on NumSharp arrays, with no POCO class.
//
// ML.NET reads all of its data through IDataView. NumSharp.Interop.MLNet makes an NDArray one: AsDataView reads the
// array's own memory, row by row, while a pipeline trains or scores; ToNDArray reads a column of a pipeline's output
// back into an NDArray, where NumSharp takes over.
//
//     dotnet run -c Release
using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Data;
using NumSharp;
using NumSharp.Interop.MLNet;
using NumSharpMLNetExample;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
var ml = new MLContext(seed: 1);
string[] features = ["f0", "f1", "f2", "f3"];

// 1. Data made in NumSharp: two overlapping clouds of points in four dimensions, 600 points each, labelled 0 and 1.
//    The first 1,000 rows train, the last 200 test.
var (train, test) = Blobs.Make(pointsPerCloud: 600, testRows: 200, seed: 7);   // (rows, 5) float32: 4 features + label
using (train)
using (test)
{
    // 2. Train a binary classifier. One NumSharp view holds one array of one dtype, so the label rides as a fifth
    //    float column; it becomes the boolean ML.NET's binary trainers want in a step of its own, outside the model,
    //    so the trained model reads the four feature columns and nothing else.
    var label = ml.Transforms.Conversion.ConvertType("Label", outputKind: DataKind.Boolean);
    using NDArrayDataView trainView = train.AsDataView([.. features, "Label"]);
    IDataView labelled = label.Fit(trainView).Transform(trainView);
    ITransformer model = ml.Transforms.Concatenate("Features", features)
        .Append(ml.BinaryClassification.Trainers.LbfgsLogisticRegression())
        .Fit(labelled);

    // 3. Score the test rows and read the predictions back into NumSharp: an accuracy is one comparison and a mean.
    using NDArray testFeatures = test[":, :4"];
    using NDArray testLabels = test[":, 4"];
    using NDArrayDataView testView = testFeatures.AsDataView(features);
    IDataView scored = model.Transform(testView);
    using NDArray predicted = scored.ToNDArray("PredictedLabel");   // (200,) bool
    using NDArray probability = scored.ToNDArray("Probability");    // (200,) float32
    using NDArray truth = testLabels.astype(NPTypeCode.Boolean);
    using NDArray hits = predicted == truth;
    using NDArray doubt = 1f - probability;                          // the probability of the other class
    using NDArray confidence = np.maximum(probability, doubt);       // the probability of the predicted class
    Console.WriteLine($"classifier  accuracy {(double)np.mean(hits):P1} on {predicted.size} test rows, " +
                      $"mean confidence {(double)np.mean(confidence):P1}");

    // ML.NET's own evaluation of the same rows agrees.
    using NDArrayDataView evalView = test.AsDataView([.. features, "Label"]);
    CalibratedBinaryClassificationMetrics metrics = ml.BinaryClassification.Evaluate(model.Transform(label.Fit(evalView).Transform(evalView)));
    Console.WriteLine($"ML.NET      accuracy {metrics.Accuracy:P1}, AUC {metrics.AreaUnderRocCurve:F3}");

    // 4. Unlabelled data: k-means on the features as ONE vector column, the shape a model consumes. The cluster ids
    //    come back as a uint32 key column (1 and 2); NumSharp measures how well they match the clouds.
    using NDArray trainFeatures = train[":, :4"];
    using NDArrayDataView points = trainFeatures.AsDataView("Features");
    ITransformer kmeans = ml.Clustering.Trainers.KMeans(numberOfClusters: 2).Fit(points);
    using NDArray cluster = kmeans.Transform(points).ToNDArray("PredictedLabel");   // (1000,) uint32
    using NDArray cloud = train[":, 4"];
    using NDArray inSecond = cluster == 2u;
    using NDArray isSecondCloud = cloud == 1f;
    using NDArray same = inSecond == isSecondCloud;
    double agreement = (double)np.mean(same);
    Console.WriteLine($"k-means     clusters match the clouds for {Math.Max(agreement, 1 - agreement):P1} of {cluster.size} points");
}

Console.WriteLine($"live views  {NDArrayMLNetInterop.LiveExports}");
