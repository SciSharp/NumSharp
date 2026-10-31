---
paths:
  - "src/NumSharp.Interop.*/**"
  - "test/NumSharp.Tests.Interop*/**"
  - "examples/NumSharpPythonnet*/**"
  - "examples/NumSharp.Interop.pythonnet.Examples/**"
  - ".github/workflows/{build-and-release,example-pythonnet}.yml"
  - "docs/releases/**"
  - "docs/website-src/**"
  - "README.md"
---

# Interop packages

## Unreleased packages

`NumSharp.Interop.OnnxRuntime`, `NumSharp.Interop.MLNet` and `NumSharp.Interop.ParquetNet` (Parquet columns decoded
straight into `NDArray` buffers) are unreleased: they build, test and merge like everything else, but no NuGet package is
created and no release or public documentation mentions them. Unreleased is exactly three things:
1. No package. Each csproj sets `<IsPackable>false</IsPackable>`, which makes `dotnet pack` a silent no-op (exit 0, no
   `.nupkg`, not even a compile). CI still names all three in the signing check and in `build-nuget`'s Build and Pack
   steps, and `create-release` lists exactly the `.nupkg` files that were packed (ids read from each nuspec), so
   packing, signing, publishing and the release text all follow that one property.
2. No docs. `docs/website-src/docfx.json` excludes `docs/interop/{onnxruntime,mlnet}.md`, because DocFX publishes every
   `.md` under `docs/website-src`, toc entry or not. `toc.yml`, `interop/index.md`, `advanced/index.md`,
   `absolute-basics.md` and the System.Numerics.Tensors page do not name or link them. ParquetNet has no page (its
   `PackageProjectUrl` points at the page to write when it ships).
3. Gated. The `interop-test` job builds and tests all three (ONNX Runtime and ML.NET at their floor and current
   versions, ParquetNet on both TFMs). Nothing else compiles them in CI.

Keep them out of release notes (`docs/releases/RELEASE_<version>.md`; when that file is missing, GitHub generates the
notes from merged PR titles, so PR titles stay free of them too), the website, the root README, and the shipped XML docs
and README of every released package: a released package's IntelliSense must not point at a bridge nobody can install.
To release one: set its `IsPackable` back to `true` (or delete the property), drop its page from the docfx.json exclude,
and restore its `toc.yml` entry and its `interop/index.md` row and "See also" link from git history. CI needs no edit.

## The conversion packages

None of these computes NumSharp operations: each converts tensors between NumSharp and another library, so none has a
`TensorEngine` seam, a `[ModuleInitializer]` or a native asset (like the pythonnet bridge).

### `NumSharp.Interop.OnnxRuntime` (unreleased)

- `NDArrayOnnxInterop` verbs, `InferenceSessionExtensions`, `Postprocess`. Sources (`src/NumSharp.Interop.OnnxRuntime/`):
  `NDArrayOnnxInterop.{cs,Export.cs,Import.cs,NonTensor.cs}` (dtype maps, the export and import verbs, non-tensor
  outputs), `OrtTensor.cs` (the `OrtTensor` / `OrtTensor<T>` handles and the `ImportLease`), `UnmanagedMemoryManager.cs`
  (a `Memory<T>` over the unmanaged buffer), `InferenceSessionExtensions.cs`, `Postprocess.cs`. ONNX Runtime's C# sources
  are vendored for reference at `refs/onnxruntime/csharp/` (a sparse submodule), its official samples at
  `refs/onnxruntime-inference-examples/c_sharp/`.
- Export: `nd.AsOrtValue()` (a zero-copy `OrtValue` over the NumSharp buffer, C-contiguous only, held by the ARC-rooted
  handle `OrtTensor`), `nd.ToOrtValue()` (an ORT-allocated copy), `AsDenseTensor<T>` / `ToDenseTensor<T>` (ORT's legacy
  tier).
- Import: `ortValue.ToNDArray()` (an owning copy), `ortValue.AsNDArray(ownsValue)` (a zero-copy lease over ORT's buffer,
  CPU only). Importing a tensor of 2 GB or more is refused up front: ORT exposes tensor memory only as an `int`-length
  Span.
- `session.Run(NDArray…)`: input names from the model metadata, NumPy-`casting` dtype coercion, declared-shape
  validation, and copy / owning-view / pre-allocated outputs. `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}`.
- Depends on `Microsoft.ML.OnnxRuntime.Managed [1.16.0, 2.0.0)` only: the native flavours (CPU, Gpu, DirectML) are
  mutually exclusive, so the consumer picks one.
- Dtypes: 12 zero-copy, including a Half↔Float16 bit reinterpretation; Char→UInt16; Decimal→Double through a temp the
  handle owns; Complex and BFloat16 refused.
- Non-tensor outputs read as NumSharp pieces: `ortValue.ToNDArrays()` (a sequence of tensors), `ToMap()` / `ToMaps()`
  (a map, or a ZipMap sequence of maps → (keys, values) NDArrays), `ReadStringTensor()` → `string[]` (NumSharp has no
  string dtype; a string or sequence model input is refused).
- Lifetime: finalizers abandon, never free eagerly; an `OrtValue` keeps its handle alive through a ConditionalWeakTable,
  and a tensor's memory manager keeps it by back-reference.
- Gate: `test/NumSharp.Tests.Interop.OnnxRuntime`, over committed `.onnx` models from `test/oracle/gen_onnx_models.py`:
  the every-dtype × layout export matrix, property-based, metamorphic, ORT-consumer (six feed routes through ORT's own
  Identity), bit-pattern fuzz, concurrency stress, ARC/fault injection, boundary and API-approval suites, and
  `LifetimeGcTests` (every finalization order of eight scenarios as a staged real collection). The `interop-test` CI job
  runs it against ONNX Runtime 1.16.0 and 1.29.0, with no Python.
- Docs: `docs/website-src/docs/interop/onnxruntime.md` (excluded from the site).

### `NumSharp.Interop.MLNet` (unreleased)

- `NDArrayMLNetInterop` verbs, `NDArrayDataView`, `Postprocess`. Sources (`src/NumSharp.Interop.MLNet/`):
  `NDArrayMLNetInterop.{cs,Export.cs,Import.cs}` (dtype maps, the export verbs, `ToNDArray` and the zero-copy
  `VBuffer.AsNDArray<T>` with `WrapExternal` / `Pin` / `LiveExports` / `LiveImports`), `NDArrayDataView.cs` (the
  NDArray-backed `IDataView`, its strided lazy cursor and ARC-pin lifetime), `VBufferAccessor.cs` (the compiled-expression
  getter for the private `VBuffer<T>._values`), `ImportLease.cs` (the GCHandle pin that the last view releases),
  `Postprocess.cs`.
- Export: `nd.AsDataView(colName)` (a vector column: a lazy `IDataView` that reads the NumSharp buffer through its
  strides, any layout, ARC-pinned; `NDArrayDataView` is `IDisposable`), `nd.AsDataView(string[])` (one scalar column per
  feature), `nd.ToDataView(…)` (a snapshot), `nd.ToVBuffer<T>()` (a copy).
- Import: `view.ToNDArray("col")` (a scalar column → `(R,)`, a fixed-size vector column → `(R,C)`),
  `vbuffer.AsNDArray<T>()` (zero-copy over a dense `VBuffer`'s private `T[]` through the compiled-expression accessor
  `VBufferAccessor<T>`: pinned, write-through, leased; sparse or empty → a copy), `vbuffer.ToNDArray()` (a copy).
- Feeding a model: `transformer.Transform(nd.AsDataView("Features")).ToNDArray("Score")`. `IDataView` is the whole
  bridge, so there is no per-transformer runner and no `ITransformer` dependency.
- Depends on `Microsoft.ML.DataView [2.0.0, 6.0.0)` only. There is no standalone `Microsoft.ML.Data` package:
  `ITransformer`/`MLContext` are in the full `Microsoft.ML` the consumer brings, which NuGet unifies upward.
- Dtypes: 11 direct, plus Char→UInt16 one way; Half→Single (lossless) and Decimal→Double converted by the view verbs
  into an owned temp; Complex refused (ML.NET has no half, decimal or complex column type). Rank 1–2 only.
- `VBuffer` zero-copy is one-directional. Import is zero-copy through reflected compiled-expression access to the
  private `T[] _values` (a `VBuffer` is backed by a managed `T[]` in every version 2.0.0, 3.0.0 and 4.0.2, with no
  `ReadOnlyMemory<T>` or non-public constructor). Export can never be zero-copy, because a managed array cannot alias
  NumSharp's unmanaged buffer, so `ToVBuffer` copies; the bulk export path `AsDataView` is zero-copy anyway.
- Gate: `test/NumSharp.Tests.Interop.MLNet`: the dtype map with every column type and CLR type, export/import
  round-trips, the zero-copy import (write-through, pooled slice, sparse fallback, lease lifetime, derived slice), every
  layout in both modes, special values bit-exact through every path including a real pipeline, sparse/empty/boundary
  VBuffers, the cursor contract, variable-length and text columns throwing, key → UInt32, lazy null row count,
  lifetime (resize refcheck refuses while exported, finalizer net), thread-parallel exports and cursors, real
  `Microsoft.ML` pipelines (NormalizeMinMax, Concatenate, Sdca, SdcaMaximumEntropy multiclass, OneHot, MapValueToKey,
  a POCO-loader cross-check), the technique suites (layout matrix, property-based, metamorphic, cursor-state
  differential against ML.NET's own cursors, real-pipeline consumer, bit-pattern fuzz, concurrency, ARC/fault injection,
  boundary, API approval) and `LifetimeGcTests` (a getter keeps its cursor reachable until its row is committed). The
  `interop-test` CI job runs it against ML.NET 2.0.1 and 4.0.2, with no Python.
- Docs: `docs/website-src/docs/interop/mlnet.md` (excluded from the site).

### `NumSharp.Interop.System.Numerics.Tensors` (released)

- `NDArrayTensorsInterop` verbs, `TensorSpanHandle<T>`.
- `nd.AsTensorSpan<T>()` → `TensorSpanHandle<T>`: a zero-copy `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` over the
  NumSharp buffer for any non-negative-stride layout (strided, transposed and broadcast included), ARC-pinned. `.Span` /
  `.ReadOnlySpan` are rebuilt on each access, because a ref struct cannot be stored.
- `nd.ToTensor<T>()` (a dense copy), `tensor.ToNDArray()` (an owning copy), `tensor.AsNDArray()` (a zero-copy view over
  the tensor's pinned buffer), `span.ToNDArray()` (a copy).
- Routing NumSharp's kernels through `TensorPrimitives` would abandon bit parity with NumPy, so the package only converts.
- Depends on `System.Numerics.Tensors [10.0.0, 11.0.0)` only, pinned to 10.x because `Tensor<T>`/`TensorSpan<T>` are
  `[Experimental("SYSLIB5001")]`; the package re-marks its surface `[Experimental("NUMSHARP_TENSORS")]`, so a consumer
  opts in once.
- All 15 dtypes cross as their own CLR type (the BCL containers are unconstrained generics). A negative-stride view is
  refused, because the BCL forbids negative strides (`ToTensor` or `np.ascontiguousarray` first).
- Namespace `NumSharp.Interop.Tensors`, not the assembly name: a `System.*` namespace would shadow the types it uses.
- Gate: `test/NumSharp.Tests.Interop.System.Numerics.Tensors`, both TFMs, in the `interop-test` CI job.
- Docs: `docs/website-src/docs/interop/system-numerics-tensors.md`.

## The pythonnet examples

- `.github/workflows/example-pythonnet.yml` runs `examples/NumSharpPythonnetGhActionsCicd`: an MSTest project that
  embeds CPython through pythonnet and asserts NumPy, pandas, SciPy, scikit-learn, Matplotlib and PyTorch over NumSharp
  arrays through `NumSharp.Interop.pythonnet`, with a `LiveExports`/`LiveImports` leak check after each test.
- It runs on the three OSes when the example or the workflow changes (a `paths` filter; also `workflow_dispatch`):
  `setup-python` 3.12 with a pip cache keyed on the example's `requirements.txt`, `pip install` into the job's
  interpreter, then `dotnet test` with `NUMSHARP_PYTHONNET_PYTHON` = `steps.python.outputs.python-path`. That variable is
  binding: unset under `CI=true` the tests fail, never skip; unset on a dev machine they are Inconclusive.
- Its console twin `examples/NumSharpPythonnetLocalExample` finds a local Python (`NUMSHARP_PYTHONNET_PYTHON`, then a
  `.venv` beside the project, then `PATH`) and is written like the tutorials (silent, outcome comments; the tests are its
  proof).
- Both csprojs are the regular form, so they build against the published package and never this repo's source: they
  may use only API that NumSharp 0.70.0 has (`typecode`, never `dtype`; `NDArrayPythonInterop.RegisterCodec()`, never
  `Py.RegisterNumSharpCodec()`). The workflow tests the released package; the interop suite of the main workflow tests
  the source.
- pythonnet facts both hosts encode:
  - `PythonEngine.ProgramName` is set for every interpreter, venv or not. Without it a venv's packages are absent; on
    Windows `sys.executable` is the .NET host; on Linux the engine takes its paths from the first `python3` on `PATH`
    (a segfault when that is another Python).
  - pythonnet 3.0.5's `Runtime.Shutdown` runs 87 full `PyGC_Collect` passes (7.3 s with the six libraries imported). The
    Local example calls `gc.collect(); gc.freeze()` first (0.2 s); the test fixture keeps the plain shutdown.
  - `torch.__version__` is a `str` subclass that `As<string>()` refuses: use `ToString()`.
  - `torch.optim` lazily imports about 800 modules (`torch._dynamo`, `sympy`), so the examples update weights by hand.
  - The .NET app host (`<app>.exe`, `testhost.exe`) is not long-path aware on Windows, so a module file past 260
    characters fails to import as `No module named …` (`python.exe` and `dotnet <app>.dll` are fine).
