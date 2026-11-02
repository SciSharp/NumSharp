// NumSharpOnnxRuntimeExample: image classification with ONNX Runtime, the pre- and post-processing in NumSharp.
//
// ONNX Runtime's C# ResNet50 tutorial fills its input tensor pixel by pixel and ranks the 1,000 scores with LINQ.
// Here both loops are array operations: ImageSharp decodes and resizes each photo, NumSharp normalizes it and lays
// it out channels-first, the session reads the NumSharp batch in place, and the scores come back as an NDArray.
//
//     dotnet run -c Release                        classifies a sample photo
//     dotnet run -c Release -- a.jpg b.png ...     classifies your photos, all in one batch
using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using NumSharp.Interop.OnnxRuntime;
using NumSharpOnnxRuntimeExample;

CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// MobileNetV2 from the ONNX model zoo (14 MB), the names of its 1,000 ImageNet classes and a sample photo,
// downloaded once into ./data.
string modelPath = await Downloads.FetchAsync("mobilenetv2-12.onnx", Downloads.MobileNetV2);
string[] labels = File.ReadAllLines(await Downloads.FetchAsync("synset.txt", Downloads.ImageNetLabels));
string[] photos = args.Length > 0 ? args : [await Downloads.FetchAsync("dog.jpg", Downloads.SamplePhoto)];

using var session = new InferenceSession(modelPath);

// One (N, 3, 224, 224) float32 batch. The model declares its batch dimension symbolic ('batch_size'), so every
// photo runs in a single call. np.stack writes a new C-contiguous array, which ONNX Runtime reads in place.
NDArray[] images = photos.Select(ImageNet.Preprocess).ToArray();
using NDArray batch = np.stack(images, axis: 0);
foreach (NDArray image in images)
    image.Dispose();

using NDArray logits = session.Run(batch);                       // (N, 1000) float32, an array of its own
using NDArray probabilities = Postprocess.Softmax(logits);       // every row sums to 1
var (top, classes) = Postprocess.TopK(probabilities, 5);         // the five best of each row, best first
using (top)
using (classes)
{
    for (int i = 0; i < photos.Length; i++)
    {
        Console.WriteLine(Path.GetFileName(photos[i]));
        for (int k = 0; k < 5; k++)
            Console.WriteLine($"  {top.GetSingle(i, k),6:P1}  {labels[classes.GetInt64(i, k)]}");
    }
}

// A service keeps its buffers: RunInto has ONNX Runtime write the scores into an array allocated once, so a
// request costs no allocation for the input or the output and nothing to dispose.
var inputs = new Dictionary<string, NDArray> { [session.InputNames[0]] = batch };
using NDArray scores = np.zeros(logits.Shape, NPTypeCode.Single);
var outputs = new Dictionary<string, NDArray> { [session.OutputNames[0]] = scores };
session.RunInto(inputs, outputs);
Console.WriteLine($"RunInto wrote the same scores: {np.array_equal(scores, logits)}");
