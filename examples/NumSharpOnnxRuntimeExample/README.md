# NumSharpOnnxRuntimeExample

A console app that classifies photos with an ImageNet model through
[ONNX Runtime](https://onnxruntime.ai), with the pre- and post-processing in NumSharp. It references
[`NumSharp.Interop.OnnxRuntime`](../../src/NumSharp.Interop.OnnxRuntime/README.md), ONNX Runtime's
CPU package and [SkiaSharp](https://github.com/mono/SkiaSharp), which decodes and resizes the photos.

ONNX Runtime's C# ResNet50 tutorial fills its input tensor pixel by pixel and ranks the 1,000 scores
with LINQ. Here both loops are array operations: NumSharp normalizes each photo and lays it out
channels-first, the session reads the NumSharp batch in place (no copy into a `DenseTensor<float>`),
and the scores come back as an `NDArray` for `Postprocess.Softmax` and `Postprocess.TopK`.

## Run it

You need the .NET 10 SDK and a network connection for the first run. From this folder:

```bash
dotnet run -c Release                          # classifies a sample photo
dotnet run -c Release -- a.jpg b.png c.webp    # classifies your photos, all three in one batch
```

```text
downloading mobilenetv2-12.onnx ...
downloading synset.txt ...
downloading dog.jpg ...
dog.jpg
   86.7 %  n02111889 Samoyed, Samoyede
    4.5 %  n02112018 Pomeranian
    1.6 %  n02120079 Arctic fox, white fox, Alopex lagopus
    1.5 %  n02112350 keeshond
    1.2 %  n02114548 white wolf, Arctic wolf, Canis lupus tundrarum
RunInto wrote the same scores: True
```

The first run downloads MobileNetV2 from the [ONNX model zoo](https://github.com/onnx/models) (14 MB),
the names of the 1,000 classes and a sample photo into a `data` folder under the working directory;
later runs reuse them. Any ImageNet classifier of the zoo with a `(batch, 3, 224, 224)` float32 input
works the same way (ResNet50 v2: `resnet50-v2-7.onnx`), as long as its batch dimension is symbolic;
SqueezeNet 1.1's input is fixed at one photo.

## What each file shows

| File | What it shows |
|---|---|
| [`Program.cs`](Program.cs) | One batch of photos through `session.Run(batch)`, then `Postprocess.Softmax` and `Postprocess.TopK` on the scores; `RunInto` writing the scores into an array allocated once, the pattern for a service that keeps its buffers. |
| [`ImageNet.cs`](ImageNet.cs) | The tutorial's preprocessing as NumSharp operations: the RGBA bytes as a `(224, 224, 4)` array, the alpha channel dropped by a slice, `np.transpose` to channels-first, the scaling and the per-channel normalization by broadcasting. |
| [`Downloads.cs`](Downloads.cs) | The first-run downloads, written under a temporary name and renamed once complete. |

## Notes

- **Float32 constants.** The mean and the standard deviation are `float32` arrays. With `double`
  constants the image would become `float64`, and `session.Run` refuses a `float64` array for the
  model's `float32` input under its default `casting: "safe"`, as Python's onnxruntime does. Pass
  `casting: "same_kind"` to let the bridge cast it instead.
- **The batch is shared, not copied.** `np.stack` writes a new C-contiguous array, and the bridge hands
  ONNX Runtime a tensor over that very buffer. A non-contiguous input (a transposed or sliced view) is
  copied for the call under the default mode; `mode: OnnxCodecMode.View` makes that an error instead.
- **Every array is disposed.** NumSharp arrays live in pooled unmanaged memory. The `using`
  declarations give each buffer back as soon as its array is no longer needed, which is also what keeps
  the build free of the `NDW012` leak warnings that the NumSharp package's analyzer reports.
- **Why SkiaSharp.** ImageSharp, which ONNX Runtime's tutorial uses, needs a license key to build from
  version 3.2.0 on, and its 3.1 releases carry advisories that `dotnet restore` reports. SkiaSharp is
  MIT-licensed; its Linux package is referenced too, so the example runs on Windows, macOS and Linux.
