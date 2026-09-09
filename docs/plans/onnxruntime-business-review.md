# Business review: ONNX Runtime × NumSharp (`NumSharp.Interop.OnnxRuntime`)

**Question.** How widely is ONNX Runtime (ORT) used — in production and on GitHub — and what does
NumSharp gain by interoperating with it fully? **Date:** 2026-09-08. Every count below was taken live on
that date with the queries shown; re-run them to refresh.

## 1. How big is ONNX Runtime

### Production

ORT is Microsoft's inference engine, and Microsoft says so about its own products: "High-scale Microsoft
services such as Bing, Office, and Azure AI use ONNX Runtime … these Microsoft services average a 2x
performance gain on CPU" ([Azure ML docs](https://learn.microsoft.com/azure/machine-learning/concept-onnx)).
It is **built into Windows** as the engine behind Windows ML — "runs on hundreds of millions of devices"
— and Windows ML 2.x ships ORT 1.24+ through `Microsoft.Windows.AI.MachineLearning`; it is the scoring
engine of **Azure SQL Edge / Managed Instance** and of **ML.NET's `OnnxTransformer`**. The onnxruntime.ai
homepage lists Windows, Office, Azure Cognitive Services and Bing as Microsoft products, a "trusted by"
wall of 38+ organisations (Adobe, AMD, Ant Group, Autodesk, Hugging Face, Intel, NVIDIA, Oracle, Redis,
…) and "thousands of other projects across the world". The C# API is a first-class, Microsoft-maintained
binding — not a community wrapper — and every Windows-ML / WinUI AI sample on Microsoft Learn is written
against `Microsoft.ML.OnnxRuntime` in C#.

### NuGet (total downloads, nuget.org search API)

| Package | Downloads | Latest |
|---|---:|---|
| `Microsoft.ML.OnnxRuntime` (CPU) | 14,627,670 | 1.29.0 |
| `Microsoft.ML.OnnxRuntime.Managed` (the C# API every flavour depends on) | 13,438,823 | 1.29.0 |
| `Microsoft.ML.OnnxRuntime.Gpu.Linux` | 20,975,622 | 1.29.0 |
| `Microsoft.ML.OnnxRuntime.Gpu` | 3,770,002 | 1.29.0 |
| `Microsoft.ML.OnnxRuntime.Gpu.Windows` | 1,059,955 | 1.29.0 |
| `Microsoft.ML.OnnxRuntime.DirectML` | 1,072,718 | 1.24.4 |
| `Microsoft.ML.OnnxRuntimeGenAI` | 1,105,319 | 0.15.2 |
| `Microsoft.ML.OnnxTransformer` (ML.NET's ORT integration) | 2,271,389 | 5.0.0 |
| *for scale:* `NumSharp` | 5,704,070 | 0.70.0 |
| *for scale:* `TensorFlow.NET` | 3,298,019 | 0.150.0 |
| *for scale:* `TorchSharp` | 1,414,881 | 0.107.0 |
| *for scale:* `Numpy` + `Numpy.Bare` (Numpy.NET) | 490,674 | 3.11.x |

Read: the .NET ORT ecosystem is **~55 M package downloads across the native flavours**, and the managed
API alone (13.4 M) is **2.4× NumSharp's lifetime downloads**. ORT is by a wide margin the largest
inference surface a .NET numerical library can attach to — larger than TensorFlow.NET and TorchSharp
combined.

### GitHub

| Measure (GitHub search API, 2026-09-08) | Count |
|---|---:|
| `microsoft/onnxruntime` stars / forks | 21,797 / 4,208 |
| Repositories matching `onnxruntime` | 2,799 |
| C# repositories mentioning onnxruntime in name/description/readme | 796 |
| Code files: `"Microsoft.ML.OnnxRuntime"` in `.csproj` | 4,584 |
| Code files: `using Microsoft.ML.OnnxRuntime` (C#) | 7,776 |
| Code files: `DenseTensor<float>` (C#) — the hand-filled input tensor | 3,904 |
| Code files: `NamedOnnxValue.CreateFromTensor` (C#) — the legacy feed | 3,888 |
| Code files: `OrtValue.CreateTensorValueFromMemory` (C#) — the modern feed | 475 |
| Code files: `GetTensorDataAsSpan` (C#) — the modern output read | 506 |
| Code files: `CreateTensorValueWithData` (C#) — raw-pointer feed (what this package uses) | 211 |
| Code files: `MultipleTensorByFloat` (C#) — the copy-pasted Stable Diffusion `TensorHelper` | 60 |
| *for scale:* code files `using NumSharp` (C#) | 2,060 |
| *for scale:* `NumSharp` in `.csproj` | 423 |
| C# files mentioning **both** `NumSharp` and `OnnxRuntime` | 115 |

Read: **4,584 C# projects** reference ORT on GitHub alone — ten times the 423 that reference NumSharp —
and the dominant usage pattern is still the *legacy* one: 3,904 files fill a `DenseTensor<float>` by
hand and 3,888 wrap it in `NamedOnnxValue.CreateFromTensor`, against ~500 that have moved to the
`OrtValue` API Microsoft has recommended since 1.16 (2023). The `TensorHelper` class — a hand-rolled
NumSharp (`DivideTensorByFloat`, `AddTensors`, `SplitTensor`, …) — is copy-pasted across 60 files
including Microsoft's own `ai-dev-gallery`. Only 115 files already combine NumSharp with ORT: the
overlap is small today, which is the opportunity, not a ceiling.

## 2. What the tutorials actually teach (the gap, verbatim)

The seven official C# tutorials (https://onnxruntime.ai/docs/tutorials/csharp/) were read in full.
Every one hand-writes the two loops this package deletes:

| Tutorial | Input | Hand-written INPUT loop | Hand-written OUTPUT loop |
|---|---|---|---|
| ResNet50v2 | `(1,3,224,224)` f32 | per-pixel `((px.R/255f)-mean)/std` into `DenseTensor<float>` → `CreateTensorValueFromMemory(…Buffer…)` | `Math.Exp`/`Sum`/`Select` softmax + `OrderByDescending().Take(10)` |
| Faster R-CNN | `(3,H',W')` f32, padded to ×32 | per-pixel `pixel.B - mean[0]` with pad offsets | stride-4 box unpack + `>= 0.7` filter |
| BERT QA | 3 × `(1,seq)` int64 | `long[]` arrays → `CreateTensorValueFromMemory` | two `GetMaxValueIndex` loops (argmax) |
| YOLOv3 / OpenVINO | `(1,3,416,416)` f32 | letterbox + per-pixel fill (sample repo) | box decode + NMS |
| Stable Diffusion | latents `(1,4,64,64)`, embeddings `(2,77,768)` | a 10-method `TensorHelper` (Duplicate, Divide, Multiply, Add, Sum, Split, …) | per-pixel `clip(x/2+0.5,0,1)*255` |
| Basic / get-started | flat `float[]` + `long[]` shape | — | flat span read |

The "basic" tutorial itself says the `OrtValue` API "is the recommended approach", that `NamedOnnxValue`
/ `DisposableNamedOnnxValue` / `FixedBufferOnnxValue` are slated for deprecation, and that pre-allocated
`OrtValue` outputs make `IOBinding` largely unnecessary — the exact shape `NumSharp.Interop.OnnxRuntime`
targets (`AsOrtValue`, `Run(inputs, outputs)` writing into NumSharp arrays).

## 3. What NumSharp gains

1. **A distribution channel ten times its size.** Every ORT C# project is a potential NumSharp consumer
   with a concrete, immediate pain (the fill loop, the post-processing loop). The package turns
   "NumSharp is NumPy for .NET" into "NumSharp is the pre-/post-processing layer for .NET inference" —
   a use case with a Microsoft-maintained on-ramp (Windows ML, WinUI samples, ML.NET) instead of a
   Python-porting niche.
2. **Zero-copy is genuinely new on the input side and additive on the output side.** ORT already offers
   zero-copy from a *managed* array, so the honest input win is *vectorized preprocessing + zero-copy*;
   the output side has no substitute at all — `session.Run` hands back a flat span, and turning it into a
   decision is exactly NumSharp's core (argmax / softmax / top-k / masking / slicing). Both are delivered
   here with no per-element loop and no NumSharp-side copy.
3. **It closes issue #512 on its own terms** — "Is there a faster way?" than a per-pixel `DenseTensor`
   fill: yes, four vectorized ops and a pointer handoff (`DocExampleTests.Issue512_…`).
4. **It is cheap to own.** The package is ~1,200 lines of conversion code with no native asset, no
   engine seam, no `[ModuleInitializer]`, one `.Managed` dependency, and a 128-test gate over committed
   models that runs on all three CI OSes without Python or GPU. Maintenance is "bump the ORT version the
   CI's second run uses".
5. **It composes with the rest of the stack.** `AsNDArray` views feed straight back into `AsOrtValue`
   (chained models), into `np.fft`, into `NumSharp.Interop.pythonnet` (an ORT output viewed as a numpy
   array with no copy), and into the OpenBLAS-backed linear algebra — one buffer, every ecosystem's API,
   the interop story the docs already tell.

## 4. Risks and non-goals

- **GPU-resident tensors** cannot be viewed as `NDArray`s (not CPU-addressable); users on CUDA/DirectML
  bind CPU outputs or copy — the same caveat the pythonnet bridge carries for CUDA torch tensors.
- **Complex / BFloat16 / String** are genuine mutual gaps; BFloat16 is the likely first post-ship ask and
  is a NumSharp dtype question, not an interop one.
- **ORT's `OrtValue` API churn** is bounded: the floor 1.16.0 (Sept 2023) is stable, the upper bound
  `< 2.0.0` protects consumers, and CI runs both ends.
- **No `System.Numerics.Tensors` bridge yet**: ORT is migrating toward `Tensor<T>` (.NET 9+); the
  `DenseTensor<T>` legacy tier is what the installed base uses today (3,900+ files), so it ships first.

## 5. Recommendation

Ship `NumSharp.Interop.OnnxRuntime` as the third interop package (done on the `onnxruntime` branch:
package, 128 tests, docs page, CI job, release wiring). Lead the messaging with the two loops every
tutorial hand-writes, show the #512 and Stable-Diffusion examples, and open an issue on
`microsoft/onnxruntime-inference-examples` offering a NumSharp-based variant of the ResNet50 sample
once the package is on nuget.org.

### Sources

- https://onnxruntime.ai/ (adopters), https://onnxruntime.ai/docs/tutorials/csharp/ (the seven tutorials),
  https://onnxruntime.ai/docs/tutorials/csharp/basic_csharp.html (OrtValue guidance)
- https://learn.microsoft.com/azure/machine-learning/concept-onnx (Bing/Office/Azure AI; Windows ML)
- https://learn.microsoft.com/windows/ai/new-windows-ml/onnx-versions (Windows ML ↔ ORT versions)
- NuGet search API `https://azuresearch-usnc.nuget.org/query?q=packageid:<id>` · GitHub REST
  `search/repositories`, `search/code` (authenticated), `repos/microsoft/onnxruntime`
- Release notes v1.16.0: "Expose OrtValue API as the new preferred API to run inference in C#"
