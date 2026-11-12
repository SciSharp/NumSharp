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

`NumSharp.Interop.ParquetNet` (Parquet columns decoded straight into `NDArray` buffers) and `NumSharp.Interop.TorchSharp`
(NDArray ↔ TorchSharp tensors, below) are unreleased: they build, test and merge like everything else, but no NuGet
package is created and no release or public documentation mentions them. `NumSharp.Interop.OnnxRuntime` and
`NumSharp.Interop.MLNet` were unreleased the same way from 2026-10-02 and ship from 0.80.0 (the owner's decision of
2026-11-02). Unreleased is exactly three things:
1. No package. The csproj sets `<IsPackable>false</IsPackable>`, which makes `dotnet pack` a silent no-op (exit 0, no
   `.nupkg`, not even a compile). CI still names it in the signing check and in `build-nuget`'s Build and Pack steps,
   and `create-release` lists exactly the `.nupkg` files that were packed (ids read from each nuspec), so packing,
   signing, publishing and the release text all follow that one property.
2. No docs. ParquetNet and TorchSharp have no page (each `PackageProjectUrl` points at the page to write when it ships).
   DocFX publishes every `.md` under `docs/website-src`, toc entry or not, so a hidden package's page needs a
   `docfx.json` exclude.
3. Gated. The `interop-test` job builds and tests it on both TFMs; apart from `build-nuget`'s Build step on a release
   tag (which compiles it so that releasing it needs no workflow edit), nothing else compiles it in CI. It READS text
   (`ParquetConvert.TextKindOf`, keyed on both Parquet.Net's CLR type - `ReadOnlyMemory<char>` / `<byte>` in Parquet.Net
   6, `string` / `byte[]` - and the schema, whose STRING / ENUM / JSON annotation makes text of bytes: Parquet.Net 6.1.0
   hands JSON over as `ReadOnlyMemory<byte>`): a UTF-8 column becomes `StringDType(na_object=None)`, a null as the NA, a
   binary column (variable or fixed length) an `S` array as wide as its longest value, a null as `b""`; a list of text
   is refused as repeated, and a UTF-8-annotated column Parquet.Net decodes to another CLR type is refused AS TEXT (a
   guard for other versions). Its gate is `TextColumnTests`, over pyarrow fixtures (`test/oracle/gen_parquet_text_fixture.py`,
   one row group and three): numpy's view of every column, both streaming faces (each group's `S` as wide as its own
   longest value), the null policies over text. The numeric edge is `NumericColumnParityTests`, over pyarrow fixtures
   (`test/oracle/gen_parquet_numeric_fixture.py`: every width at its extremes in four writer configurations, a zero-row
   file, wide decimals, a float16 column): numpy's `to_numpy()` view bit for bit across row groups, both faces streaming
   the same bytes, and the null policy against numpy's NaN / None widening (`default(T)`, a per-call fill never cached,
   Raise only where a null is actually read). A fill converts as NumPy's `a[i] = fill` (`ResolveFill` packs it with Core's
   `PyArray_Pack` port, `ReductionInitial.PackStore`: a fraction truncates, NaN into an integer is numpy's `ValueError`, a
   text fill into text stays text and NaN becomes `'nan'`), and only at the first null met, so a file-level NaN fill
   leaves the null-free integer columns readable. Parquet.Net 6.1.0's own limits are `[Misaligned]`: data page v2 RLE
   booleans throw `DivideByZeroException`, DELTA_BINARY_PACKED int8 / uint8 / decimal are refused, a float16 column
   refuses the whole schema, a decimal past 28-29 digits rounds or overflows, and the decimal's trailing zeros are dropped.

Keep it out of release notes (`docs/releases/RELEASE_<version>.md`; when that file is missing, GitHub generates the
notes from merged PR titles, so PR titles stay free of it too), the website, the root README, and the shipped XML docs
and README of every released package: a released package's IntelliSense must not point at a bridge nobody can install.
To release it: delete its `IsPackable` property, write its page and add its `toc.yml` entry and its `interop/index.md`
row and "See also" link (the two packages released for 0.80.0 got theirs back from git history). CI needs no
edit, but the package is a NEW id on nuget.org: check before the tag that the Trusted Publishing policy may push new
packages (`ci-release.md`).

## The conversion packages

None of these computes NumSharp operations: each converts tensors between NumSharp and another library, so none has a
`TensorEngine` seam, a `[ModuleInitializer]` or a native asset (like the pythonnet bridge).

### `NumSharp.Interop.OnnxRuntime` (released from 0.80.0)

- `NDArrayOnnxInterop` verbs, `InferenceSessionExtensions`, `Postprocess`. Sources (`src/NumSharp.Interop.OnnxRuntime/`):
  `NDArrayOnnxInterop.{cs,Export.cs,Import.cs,NonTensor.cs,Text.cs}` (dtype maps, the export and import verbs,
  non-tensor outputs, text tensors), `OnnxCodecMode.cs`, `OrtTensor.cs` (the `OrtTensor` / `OrtTensor<T>` handles and the
  `ImportLease`), `OrtDisposalProbe.cs` (reads ORT's private `_disposed` flags, so a disposed session or value throws
  `ObjectDisposedException`), `OrtValueBooking.cs` (the ARC token of every value the bridge touches, an `IPinnable` it
  stores in ORT's private `OrtValue._memHandle`: books a `ToOrtValue` result's bytes, counts call and view leases, and
  takes ORT's native release over so it happens once, after the last lease), `NDInferenceSession.cs` (the session
  whose `Dispose` waits for the bridge's runs, and the `SessionLease` the four `Run` verbs take),
  `UnmanagedMemoryManager.cs` (a `Memory<T>` over the unmanaged buffer),
  `InferenceSessionExtensions{,.Async}.cs`, `RunOutputs.cs` (the result dictionary, an `INDArrayCarrier`), `Postprocess.cs`
  (the one public `Postprocess`: the ML.NET bridge references this package for it). ONNX Runtime's C# sources are vendored for reference at `refs/onnxruntime/csharp/` (a
  sparse submodule), its official samples at `refs/onnxruntime-inference-examples/c_sharp/`.
- `OnnxCodecMode { Auto, View, Copy }` (the members of the pythonnet codec's `NumpyCodecMode`) on every entry that could
  share memory: Auto is a view when ORT can read the buffer in place (C-contiguous and a bit-exact dtype), else a copy
  the handle owns; View throws instead of copying; Copy always copies.
- Export: `nd.AsOrtValue(mode)` (an `OrtValue` over the NumSharp buffer, held by the ARC-rooted handle `OrtTensor`, whose
  `IsView` says view or copy), `nd.ToOrtValue()` (an ORT-allocated copy that ORT frees: NumSharp never frees memory a
  native holder such as `OrtIoBinding` may still read, it only BOOKS the bytes until `Dispose(true)` releases the token's
  `MemoryHandle` or the token's finalizer runs), `AsDenseTensor<T>(mode)` / `ToDenseTensor<T>`
  (ORT's legacy tier, always row-major).
- Import: `ortValue.ToNDArray()` (an owning copy; the legacy tier's `DenseTensor<T>`, any `Tensor<T>` - what
  `AsTensor<T>()` is declared to return, a non-dense one densified first - and `NamedOnnxValue` copy the same way),
  `ortValue.AsNDArray(ownsValue, mode)` (a zero-copy lease over ORT's
  buffer, CPU only; a counted NumSharp alias for a value the package made). Imports of NumSharp memory (a value the
  package made, a fed input passed through) are counted aliases that inherit read-only. A view books
  (`SizeBucketedBufferPool.BookForeign`) the memory it keeps alive: an OWNING view its ORT buffer - not over a value that
  carries a `MemoryHandle` (a `ToOrtValue` result is booked already; a `CreateTensorValueFromMemory` value's memory is
  counted elsewhere or not freed with it) - and a `DenseTensor<T>` view the MANAGED array it pins (`MemoryMarshal.TryGetArray`;
  a legacy output's native buffer books nothing). Every lease over a pinned managed array books (this one, ML.NET's,
  Tensors'): the GC counts the array, but a forgotten view's pin keeps it alive past the collection that found the view
  dead, and unbooked forgotten views of large arrays grew to 1.9-2.7 GB. Importing a tensor of 2 GB or more that ORT owns is refused up front: ORT exposes tensor memory only as
  an `int`-length Span.
- `session.Run(NDArray…)`: input names from the model metadata (overridable initializers and `optional(tensor)` inputs
  included), NumPy-`casting` dtype coercion checked on every call, declared-shape validation, copy / owning-view outputs.
  `session.RunInto` fills pre-allocated outputs (an output that overlaps an input is computed from a copy under Auto,
  refused under View). `session.RunIntoAsync` / `RunAsync` + `IsRunAsyncSupported` are ORT's native async (ORT 1.30.0 or
  later, a session with two or more intra-op threads; `RunAsync` allocates only a numeric output of a fixed declared
  shape). `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}`.
- Depends on `Microsoft.ML.OnnxRuntime.Managed` 1.16.0 or later, with NO upper bound (the form ORT's own packages
  declare it in), only: the native flavours (CPU, Gpu, DirectML) are mutually exclusive, so the consumer picks one. A
  cap below a consumer's ONNX Runtime is a restore ERROR (NU1107), and the ML.NET bridge references this package, so a
  cap would fail ML.NET users too. The bridge uses ORT's public C# API plus the private `_disposed` flags
  `OrtDisposalProbe` reads (a missing field turns its check off), the private `OrtValue._memHandle` slot
  `OrtValueBooking` writes (a missing or retyped field leaves `ToOrtValue` unbooked; `ToOrtValueBookingTests` asserts
  the slot exists) and the members its takeover needs (`_handle`, `_disposed`, the internal `(IntPtr, OnnxValueType)`
  constructor; `_owned` and `_compositeMembers` where they exist; any one missing turns the leases off and leaves ORT's
  own rules, `OrtValueArcTests.Arc_HooksIntoTheLoadedOnnxRuntime` asserts them), so a breaking ORT major shows at run time; CI's advisory "newest" step (floating `*-*`) sees it first.
- Dtypes: 12 zero-copy, including a Half↔Float16 bit reinterpretation; Char→UInt16; Decimal→Double through a temp the
  handle owns (Auto / Copy; `Run` needs `casting: "same_kind"`); text: fixed-width `U` / `S` ↔ ORT string tensors, always
  a copy (a trailing NUL is lost; the bridge casts no text ↔ number pair: the caller's `astype` does); a `StringDType`
  array shares the text lane, so `IsText` is true for it and `RefuseStringDType` throws `NotSupportedException` in
  every export verb and `Run` (Python's onnxruntime refuses it too); the remedy is a SIZED `U` cast, because NumPy and
  NumSharp both raise `TypeError` for `astype(np.dtype("U"))` from `StringDType`; Complex and BFloat16 refused; sparse
  tensors, non-tensor `Run` inputs and non-`NDArray` inputs are not served.
- Non-tensor outputs read as NumSharp pieces: `ortValue.ToNDArrays()` (a sequence of tensors), `ToMap()` / `ToMaps()`
  (a map, or a ZipMap sequence of maps → (keys, values) NDArrays), `GetMapKeys`, `ReadStringTensor()` → `string[]`; each
  numeric array is a view that OWNS ORT's copy of its element.
- Five ORT facts the package is built around (`docs/plans/onnxruntime-rework.md`): (1) a model OUTPUT that is a FED
  input (no node produces it) is never written into a pre-allocated buffer, and the value ORT returns for it lies in the
  fed memory, so `InferenceSessionExtensions.FedTensor` (by name, ORT's own rule) makes `RunInto` copy it and the
  zero-copy form alias it; (2) ORT completes `RunAsync` on its OWN intra-op thread, where `session.Dispose()` never
  returns, and awaits without `ConfigureAwait(false)`, so the bridge's task uses `RunContinuationsAsynchronously` and
  hides the caller's context and scheduler from ORT; before 1.30.0 the managed `RunAsync` does not pin its argument
  arrays, so the forms refuse older ORT; (3) ORT does not guard a RELEASED `OrtValue` (a native call on one is an access
  violation that ends the process), so the handle records its disposed value in a weak table and every import verb
  throws `ObjectDisposedException`; (4) `NodeMetadata` reports a declared scalar and an unknown rank identically
  (`dims=[]`), so `RunAsync` does not allocate such an output and `Run`'s shape check leaves it to ORT; (5) the 1.16.0
  managed assembly carries AssemblyVersion 0.0.0.0, so only the NuGet range sets the floor.
- Lifetime: finalizers abandon, never free eagerly; an `OrtValue` keeps its handle alive through a ConditionalWeakTable,
  and a tensor's memory manager keeps it by back-reference.
- Threads (the owner's decisions of 2026-11-07, `docs/plans/interop-onnx-mlnet-release-qa.md` section 10): ORT's own
  `Dispose` takes no lock (two disposers of a value or a session corrupted the heap; a `Dispose` during a verb or the
  bridge's `Run` was an access violation; a view read after its value's `Dispose` returned another value's data 400 of
  400). Every value the bridge touches carries an ARC token: from birth for `ToOrtValue` / `AsOrtValue` / `GetMapKeys`
  results, adopted by the first public verb for a value ORT allocated (never: a slot already holding a
  `MemoryHandle`, `_owned == false`, `_compositeMembers` set; nor the `Run` tier's outputs and the readers' elements,
  which pass `adopt: false`). The synchronous `Run` / `RunInto`'s own input and pre-allocated-output handles are
  TRANSIENT (`OrtTensor(..., transient: true)`): no token, no `ValueOwners` / `DisposedValues` entry, ORT's own `Dispose`
  in the call's `finally` - they never leave the call or its thread (27 inputs of a GPT-2 decoder step: 42 us / 17 KB
  in that machinery, `RunTierTransientHandleTests` pins it through `TakenOverReleases`). The asynchronous verbs and every
  public export keep it; never pass `transient: true` for a value that can outlive the call or cross a thread. A verb holds a CALL lease (`OrtValueBooking.Enter`, which also makes the disposal checks)
  while it calls ORT; a view holds a VIEW lease for its life. The token's `Unpin` runs inside ORT's `Dispose(true)`
  before ORT's own release: it waits for the calls, moves the native value into a shadow `OrtValue` (ORT's internal
  constructor), zeroes `_handle` (ORT's own release becomes a no-op) and sets `_disposed`; the shadow is disposed at
  once or by the last view, and the booking ends then. Never dispose a value under its own call lease (it waits for
  itself): the owning verbs dispose after theirs. Sessions: `NDInferenceSession` (sealed, ORT's eight constructors)
  overrides `Dispose(bool)` with a once-guard and a run count; the four `Run` verbs take a `SessionLease`, the
  asynchronous ones until `PendingRun.Finish`, whose release hops to the thread pool. Not covered: ORT's own calls
  (its `Run` overloads on any session, `GetTensorDataAsSpan`, `OrtIoBinding`), a plain `InferenceSession` (its
  `Dispose` frees the runs' name buffers before any hook), and a `Dispose` racing a value's FIRST verb (adoption is a
  plain struct store). Diagnostics: `OrtValueBooking.TakenOverReleases` / `HasToken`, `NDInferenceSession.Releases`.
- Gate: `test/NumSharp.Tests.Interop.OnnxRuntime`, over committed `.onnx` models from `test/oracle/gen_onnx_models.py`
  (which takes model names, so a new model does not rewrite the committed ones): the every-dtype × layout export matrix
  (a cell per mode), property-based, metamorphic, ORT-consumer (six feed routes through ORT's own Identity), bit-pattern
  fuzz, concurrency stress, ARC/fault injection, boundary and API-approval suites, `LifetimeGcTests` (every finalization
  order of eight scenarios as a staged real collection), and `SessionRunModeTests`, `ImportAliasTests`, `TextTests`,
  `PassThroughOutputTests`, `SessionRunAsyncTests`, `ValueOwnershipTests`, `MemoryManagerTests`, `OrtValueArcTests` and
  `NDInferenceSessionTests` (every race above, 8 threads disposing one value or one session included), and `TextModelsEndToEndTests`
  (text through eleven string models - Concat, Transpose, Gather, Reshape, Slice, Unique, Where, Expand, Tile,
  StringNormalizer, LabelEncoder - held to NumSharp's numpy-verified twins and to `np.array(list)`'s width; Python
  onnxruntime's edges pinned `[Misaligned]`: an embedded NUL survives where it truncates, U+FFFD where it raises on a
  lone surrogate or non-UTF-8 output; onnxruntime 1.29's string Gather on axis 0 of a 2-D tensor copies one string per
  row, so the model gathers on axis 1), and `TextRunTierEdgeTests` (text input through `RunInto` into a preallocated
  numeric output, `Run`, `RunAsync` / `RunIntoAsync` over the fixed-shape LabelEncoder model - `RunAsync` cannot allocate
  for an output declared without dimensions -, the declared rank validated, StringNormalizer stopwords removing elements
  down to one empty string, eight concurrent string runs), and the execution providers: `ProviderEndToEndTests` (each
  CPU-side provider the loaded ORT carries runs `matmul_const_f32` - X times a CONSTANT W, the MatMul form XNNPACK takes
  - and its output is imported through `AsNDArray`, `ToNDArray` and `Run(zeroCopyOutputs)`, the output's allocator
  asserted so a silent CPU fallback cannot pass; each GPU provider's output, bound to its own memory through
  `OrtIoBinding`, is refused) and `HelperEdgeTests`' table of every provider's allocator from ORT's sources (CPU-device
  memory read, GPU / NPU memory refused, via ORT 1.23's `CreateMemoryInfo_V2`). Desktop ORT packages carry only the CPU
  provider, so `test/NumSharp.Tests.Interop.OnnxRuntime.Android` (an instrumentation-only app, not in the sln: no
  Android workload in CI; run by hand over adb, its README) runs the same `ProviderEndToEnd` code on a device: CPU,
  XNNPACK (outputs in `XnnpackExecutionProvider` memory) and NNAPI imported, WebGPU's `WebGPU_Buf` refused. The
  `interop-test` CI job
  runs it against ONNX Runtime 1.16.0 (the async tests Inconclusive, the refusal tested) and 1.31.1, the current
  release (the only gating leg that runs the async tests: never set it below 1.30.0), and (advisory) against the newest
  `Microsoft.ML.OnnxRuntime` on nuget.org, release candidates included, with no Python. Its "Package consumer:" steps
  (`src/NumSharp.Interop.{OnnxRuntime,MLNet}/tools/verify_package_consumer.sh`) pack the two bridges into a throwaway
  local feed under a pre-release version unique to the run (`openblas.md`) and run a consumer that restores them by
  `PackageReference`, so the package that ships is the one gated. Mutation runs restore a file from memory only: copy
  untracked files out of the tree before one.
- Docs: `docs/website-src/docs/interop/onnxruntime.md` (on the site's Interoperability toc since 0.80.0).
- Example: `examples/NumSharpOnnxRuntimeExample` (regular csproj form over the published 0.80.0; no workflow runs
  it): MobileNetV2 downloaded on the first run, SkiaSharp decoding (ImageSharp 3.1.x restores with security advisories
  and 3.2+ needs a license key to build), NumSharp preprocessing into one NCHW batch, `Postprocess`, `RunInto`.

### `NumSharp.Interop.MLNet` (released from 0.80.0)

- `NDArrayMLNetInterop` verbs, `NDArrayDataView`; `Postprocess` is the ONNX Runtime bridge's, which this package
  references (`NumSharp.Interop.OnnxRuntime.Postprocess`: one public type, no CS0104 in an app that uses both). Sources
  (`src/NumSharp.Interop.MLNet/`):
  `NDArrayMLNetInterop.{cs,Export.cs,Import.cs}` (dtype maps, the export verbs, `ToNDArray` and the zero-copy
  `VBuffer.AsNDArray<T>` with `WrapExternal` / `Pin` / `LiveExports` / `LiveImports`), `NDArrayDataView.cs` (the
  NDArray-backed `IDataView`, its strided lazy cursor - shuffled through a per-cursor permutation when ML.NET passes a
  `Random`, `CanShuffle` true - and ARC-pin lifetime), `VBufferAccessor.cs` (the compiled-expression
  getter for the private `VBuffer<T>._values`), `ImportLease.cs` (the GCHandle pin that the last view releases; it books
  the pinned bytes with `SizeBucketedBufferPool.BookForeign`, never one `GC.AddMemoryPressure` per lease).
- Export: `nd.AsDataView(colName)` (a vector column: a lazy `IDataView` that reads the NumSharp buffer through its
  strides, any layout, ARC-pinned; `NDArrayDataView` is `IDisposable`), `columns.AsDataView()` over an
  `IReadOnlyDictionary<string, NDArray>` (one column per array, each its own dtype and strides, equal rows, every pin
  taken or none; `NDArrayDataView` holds an array of `Source`s and each `ColumnPlan` names its source),
  `nd.AsDataView(string[])` (one scalar column per
  feature; a repeated name is refused, ordinal as in ML.NET, because ML.NET would hide the earlier column),
  `nd.ToDataView(…)` (a snapshot), `nd.ToVBuffer<T>()` (a copy).
- Import: `view.ToNDArray("col")` (a scalar column → `(R,)`, a fixed-size vector column → `(R,C)`),
  `vbuffer.AsNDArray<T>()` (zero-copy over a dense `VBuffer`'s private `T[]` through the compiled-expression accessor
  `VBufferAccessor<T>`: pinned, write-through, leased; sparse or empty → a copy), `vbuffer.ToNDArray()` (a copy).
- Feeding a model: `transformer.Transform(nd.AsDataView("Features")).ToNDArray("Score")`. `IDataView` is the whole
  bridge, so there is no per-transformer runner and no `ITransformer` dependency.
- Depends on `Microsoft.ML.DataView` 2.0.0 or later, with NO upper bound (the form `Microsoft.ML` itself declares it
  in), and on `NumSharp.Interop.OnnxRuntime` for `Postprocess` (bringing `Microsoft.ML.OnnxRuntime.Managed`, the C# API
  only, about 250 KB, never loaded by `Postprocess`). A cap below a consumer's `Microsoft.ML` is a restore ERROR (NU1107, cousin ranges that do
  not overlap): the former `[2.0.0, 6.0.0)` would have failed every user's restore the day they moved to ML.NET 6. The
  bridge uses ML.NET's public data contract plus the private `VBuffer<T>._values`, whose accessor falls back to a copy,
  so a breaking ML.NET major shows at run time; CI's advisory "newest" step (floating `*-*`) sees it first. There is no
  standalone `Microsoft.ML.Data` package: `ITransformer`/`MLContext` are in the full `Microsoft.ML` the consumer
  brings, which NuGet unifies upward.
- Dtypes: 11 direct, plus Char→UInt16 one way; Half→Single (lossless) and Decimal→Double converted by the view verbs
  into an owned temp; Complex refused (ML.NET has no half, decimal or complex column type). Rank 1–2 only.
- `VBuffer` zero-copy is one-directional. Import is zero-copy through reflected compiled-expression access to the
  private `T[] _values` (a `VBuffer` is backed by a managed `T[]` in every version 2.0.0, 3.0.0, 4.0.2 and 5.0.1 and in
  the 6.0.0 previews, with no `ReadOnlyMemory<T>` or non-public constructor: the zero-copy import suites pass on each). Export can never be zero-copy, because a managed array cannot alias
  NumSharp's unmanaged buffer, so `ToVBuffer` copies; the bulk export path `AsDataView` is zero-copy anyway.
- Gate: `test/NumSharp.Tests.Interop.MLNet`: the dtype map with every column type and CLR type, export/import
  round-trips, the zero-copy import (write-through, pooled slice, sparse fallback, lease lifetime, derived slice), every
  layout in both modes, special values bit-exact through every path including a real pipeline, sparse/empty/boundary
  VBuffers, the cursor contract, variable-length and text columns throwing, key → UInt32, lazy null row count,
  lifetime (resize refcheck refuses while exported, finalizer net), thread-parallel exports and cursors, real
  `Microsoft.ML` pipelines (NormalizeMinMax, Concatenate, Sdca, SdcaMaximumEntropy multiclass, OneHot, MapValueToKey,
  a POCO-loader cross-check), the technique suites (layout matrix, property-based, metamorphic, cursor-state
  differential against ML.NET's own cursors, real-pipeline consumer, bit-pattern fuzz, concurrency, ARC/fault injection,
  boundary, API approval), `LifetimeGcTests` (a getter keeps its cursor reachable until its row is committed) and
  `TextEndToEndTests` (every text class × layout × verb refused before any rank check, leasing nothing; NumSharp text
  through `MapValueToKey` gives `np.unique`'s inverse + 1; an ML.NET text column through `GetColumn<string>` is
  `np.array(list)`; `FeaturizeText` rows cross back) and `TextEdgeEndToEndTests` (ML.NET reads a `null` text as empty
  and keys the empty text 0, its missing value, so a `StringDType(na_object=None)` NA - `null` through `AsStringArray` -
  lands on key 0 and the twin is `np.unique` over the non-empty texts; a `na_object=nan` NA is refused by `AsStringArray`,
  `tolist()` is the route; `TokenizeIntoWords` rows are `np.char.split`'s and ragged rows refuse `np.array` with
  numpy's `ValueError`; a `TextLoader` CSV's numbers cross the bridge as `np.loadtxt` reads them, quoted text by the
  route). The `interop-test` CI job runs it against ML.NET 2.0.1 (the floor), 5.0.1 (current) and 4.0.2 (the previous
  major), and (advisory) against the newest `Microsoft.ML` on nuget.org, previews included, with no Python.
- Docs: `docs/website-src/docs/interop/mlnet.md` (on the site's Interoperability toc since 0.80.0).
- Example: `examples/NumSharpMLNetExample` (regular csproj form over the published 0.80.0 and Microsoft.ML 5.0.1;
  no workflow runs it): a binary classifier trained from a `(rows, 5)` array as five scalar columns with the label
  converted OUTSIDE the model (so the model scores a features-only view), L-BFGS for run-to-run identical output (SDCA
  is multi-threaded and varied), k-means on one vector column.

### `NumSharp.Interop.System.Numerics.Tensors` (released)

- `NDArrayTensorsInterop` verbs, `TensorSpanHandle<T>`.
- `nd.AsTensorSpan<T>()` → `TensorSpanHandle<T>`: a zero-copy `TensorSpan<T>` / `ReadOnlyTensorSpan<T>` over the
  NumSharp buffer for any non-negative-stride layout (strided, transposed and broadcast included), ARC-pinned. `.Span` /
  `.ReadOnlySpan` are rebuilt on each access, because a ref struct cannot be stored. A span holds no reference of its own:
  the handle or the source must stay undisposed for its whole life (read after both were disposed, it read another
  array's data 800 / 800; the owner chose to document the rule, 2026-11-07: "NDArray can wrap Unsafe memory but has
  to be disposed in the duration of the life of the span").
- `nd.ToTensor<T>()` (a dense copy), `tensor.ToNDArray()` (an owning copy), `tensor.AsNDArray()` (a zero-copy view over
  the tensor's pinned buffer; its `ImportLease` in `TensorSpanHandle.cs` books the pinned bytes with `BookForeign` until
  the last view releases the pin), `span.ToNDArray()` (a copy).
- Routing NumSharp's kernels through `TensorPrimitives` would abandon bit parity with NumPy, so the package only converts.
- Depends on `System.Numerics.Tensors [10.0.0, 11.0.0)` only, pinned to 10.x because `Tensor<T>`/`TensorSpan<T>` are
  `[Experimental("SYSLIB5001")]`; the package re-marks its surface `[Experimental("NUMSHARP_TENSORS")]`, so a consumer
  opts in once.
- All 15 dtypes cross as their own CLR type (the BCL containers are unconstrained generics). A negative-stride view is
  refused, because the BCL forbids negative strides (`ToTensor` or `np.ascontiguousarray` first).
- Namespace `NumSharp.Interop.Tensors`, not the assembly name: a `System.*` namespace would shadow the types it uses.
- Text crosses as a COPY both ways, since no string can be shared (`NDArrayTensorsInterop.Text.cs`; the owner's Auto
  rule - view when possible, else copy - always lands on the copy): `ToStringTensor()` copies a `U`, `S` (decoded as
  UTF-8, strictly) or `StringDType` array into a `Tensor<string>` (an NA whose object is not a string as `null`, a 0-d
  array as `[1]`), and the non-generic `ToNDArray` overloads copy a `Tensor<string>` / string span into
  `StringDType(na_object=None)` (a `null` as the NA). The unmanaged verbs and `ToTensorElementClrType` refuse a text array
  with ONE message (`TextRefusal`, the ML.NET bridge's rule) naming its class and `ToStringTensor()`: the verbs pass the
  array's `dtype`, since its type code's implicit `NPTypeCode → DType` conversion throws Core's storage-lane error.
- Gate: `test/NumSharp.Tests.Interop.System.Numerics.Tensors`, both TFMs, in the `interop-test` CI job. Text:
  `TextCrossingTests` (every class × layout against numpy's `tolist()`, imports, round trips, numpy's
  `UnicodeDecodeError` / `UnicodeEncodeError`) and `TextRefusalTests` (every class × layout × unmanaged verb refused
  before any layout check, naming the class and `ToStringTensor()`); `ApiSurfaceTests` approves the text verbs.
- Docs: `docs/website-src/docs/interop/system-numerics-tensors.md`.

### `NumSharp.Interop.TorchSharp` (unreleased)

- `NDArrayTorchInterop` verbs and `TorchCodecMode { Auto, View, Copy }` (the owner's codec modes; NumPy/PyTorch's
  `copy=None/False/True`). Sources (`src/NumSharp.Interop.TorchSharp/`): `NDArrayTorchInterop.{cs,Export.cs,Import.cs}`
  (counters, dtype maps, `MayShareMemory`, shared plumbing; `AsTensor` / `ToTensor`; `AsNDArray` / `ToNDArray`),
  `LibTorchSharpNative.cs` (the five natives the bridge binds itself, the `THSTensor_new` ABI detection),
  `TorchLeases.cs` (`ExportLease`, `ImportLease`). TorchSharp's sources are not vendored: read them on GitHub.
- How memory crosses. TorchSharp has no public way to wrap memory it did not allocate (`from_array` / `frombuffer` pin a
  managed `Array` only, its `Memory<T>` overloads copy), so the export calls LibTorchSharp's `THSTensor_new` (libtorch's
  `from_blob` with a deleter) itself and wraps the handle with the public `Tensor.UnsafeCreateTensor`. A C-contiguous
  array is one `from_blob`; any other non-negative-stride layout is `from_blob` over its reach + `as_strided`; a broadcast
  over writeable memory (`Storage.BaseStorage`'s flag, never `nd.@base`, which allocates a pinned NDArray) is the memory
  under it + torch's `expand`. Each export lease holds one ARC reference (`TryAddRef`) and a per-lease delegate rooted in a
  static dictionary until libtorch's deleter runs; the deleter `Release`s (not `Abandon`s: libtorch calls it exactly when
  nothing in torch can read the memory) on whichever thread drops the storage. The import holds a `THSTensor_detach` of the
  tensor (its own `TensorImpl` over the storage) as a raw handle in an `ImportLease` (no managed `Tensor`, so no
  `DisposeScope` can dispose it under the view), books the storage's `THSStorage_nbytes`, and is released by the view's
  memory-block Disposer. Copies are NumSharp buffers: `np.copyto(…, casting)` into a fresh array, lent like a view.
- The owner's decisions (2026-11-12): a broadcast of writeable memory crosses as `expand`, every other read-only array
  (NumPy scalars, read-only foreign memory, a broadcast of either) copies under Auto and throws under View; NumSharp's
  uint16 / uint32 / uint64 / char take NEP 50's `np.promote_types(dtype, np.int8)` (int32, int64, float64, int32), decimal
  float64, and an explicit `dtype` converts through `np.copyto`'s `casting` (default `"same_kind"`), so `dtype: Int64`
  downcasts uint64; torch's complex64 / bfloat16 widen exactly in NumSharp (float pairs → `.real` / `.imag`; bf16 bits
  zero-extended into a uint32 view of the result and `left_shift` 16), View throws; a `requires_grad` tensor is a
  READ-ONLY view (no TorchSharp / LibTorchSharp API bumps autograd's version counter; `t.detach().AsNDArray()` is the
  writeable route); floor TorchSharp 0.100.0, tested with 0.107.0, no upper bound; unreleased until the owner decides.
- Torch facts the package is built around (measured on 0.107.0 unless noted): `conj()` is lazy (the memory holds the
  unconjugated values; `.imag` of it carries the neg bit), so a view refuses `is_conj()` / `is_neg()`; `THSTensor_data` of
  a META tensor is null with NO recorded error (decide by `device_type`), of a sparse one null with an error; an empty
  tensor's data pointer is null; `THSTensor_alias` shares the caller's `TensorImpl` (its data pointer moved after
  `t.set_(other)`), `detach` does not; the deleter fires even for a 0-byte storage and on the GC finalizer thread
  through TorchSharp's `~Tensor`; `from_blob` storage is not resizable; `as_strided` refuses negative strides and any
  layout past the storage; libtorch refuses in-place ops over overlapping (expanded) lanes but `t[0].fill_()` writes one;
  TorchSharp's `Tensor.Dispose` takes no lock.
- Version traps (the package compiles against 0.100.0): every `Tensor.to(...)` overload gained parameters by 0.107.0
  (MissingMethodException for a floor binary), so the package never calls `to` (conversions in NumSharp, device moves
  via `cpu()`, which never changed; no `device:` parameter - `nd.AsTensor().cuda()`); `THSTensor_new` took 6 arguments in
  0.100.x (`TorchSharp.PInvoke.LibTorchSharp`) and 9 from 0.101.0 (`TorchSharp.PInvoke.NativeMethods`):
  `LibTorchSharpNative.DetectAbi` reads TorchSharp's own declaration and refuses an unknown form; `ScalarType` names the
  quantized codes (12-14) only from 0.107.0 and never names libtorch's uint16/32/64 (27-29, imported as NumSharp's
  unsigned dtypes), so the sources use the ordinals; 0.100.0 ships no osx-arm64 natives; TorchSharp 0.100.0 brings
  SkiaSharp 2.88.3 (NU1903 audit warning; current TorchSharp brings 2.88.6).
- The namespace is the package name, so inside `NumSharp.Interop.TorchSharp` (and the test namespace) a qualified
  `TorchSharp.` resolves to this namespace: import TorchSharp with file-level `using`s, never qualify.
- What no reference covers (documented, not guarded): an in-place resize of an imported tensor's storage (libtorch's
  `resize_`) frees the memory a view reads - PyTorch's `numpy()` marks the storage non-resizable, which TorchSharp offers
  no way to; a tensor disposed on another thread while a verb reads it.
- Gate: `test/NumSharp.Tests.Interop.TorchSharp` (`TorchSharpVersion` / `LibTorchVersion` properties, default the floor
  0.100.0 / 2.0.1.1; it references `TorchSharp` + the runner's `libtorch-cpu-<rid>` package, not `TorchSharp-cpu`, which
  restores all three OSes' natives, ~1.2 GB per version). The `interop-test` job runs the floor leg (Windows, Linux) and
  the current leg (0.107.0 / 2.10.0, all three OSes) on both TFMs; the current leg is the binary-compatibility gate for
  the floor-compiled package, and runs the sparse / quantized / libtorch-uint16 tests the floor leg reports Inconclusive.
  Suites: `AbiTests` (the detected form, the floor reference), `DtypeMapTests`, `ExportTests`, `ExportLayoutTests` (every
  shared dtype × layout, broadcast `expand`, reversed / read-only / misaligned copies, views ending at the buffer's last
  element), `ImportTests`, `LifetimeTests` (dispose / scope / finalizer / torch views / `set_` / booking),
  `ConcurrencyTests`, `ApiSurfaceTests`, `DocExampleTests`; the leak gate (`LiveExports` / `LiveImports` back to the
  baseline) after every test. Run it under `NUMSHARP_DEBUG_GUARD_PAGES=1` after touching the export layouts (green on
  both legs).

## The pythonnet bridge carries text

- `S` / `U` are ordinary buffers and VIEW both ways (`<U1` and a 4-byte `wchar_t` buffer import as `U1`, never `Char`;
  a big-endian `U` copies byte-swapped; `np.str_` / `np.bytes_` are 0-d `U` / `S`).
- A `T` array crosses as an exact COPY through numpy's `_ARRAY_API` capsule
  (`src/NumSharp.Interop.pythonnet/NumpyStringApi.cs`: slots 313 load, 314 pack, 315 pack_null, 316 acquire, 318
  release, 160 `PyArray_GetPtr` for the data address) with `NpyString_dup`'s rules, so its cells equal numpy's own
  copy's. An object array of `str` copies to `StringDType()` (a `str` SUBCLASS element too: numpy's `coerce=False` cast
  takes only an exact `str` / `np.str_`, so such an array goes through the coercing cast, which stores `str(x)`), any
  other is refused, and so is an NA object with no C# form (`pd.NA`). An EMPTY source keeps its writeability like any
  other (a read-only empty one is refused by default, non-writeable on opt-in; an empty broadcast exports read-only).
- Traps: numpy's `na_eq_cmp` is not reflexive for an `np.float16` NaN (only identity matches), so write a `T` array's NA
  with `descr.na_object` (a second boxed `Half.NaN` is the TEXT `'nan'`); pythonnet's `As<double>()` refuses an
  `np.float16` (convert through Python's `float()`); `np.empty(shape, T)` gives the result a NEW descriptor instance with
  its own allocator, so lock `result.dtype`'s allocator, not the descriptor passed in; a failed `memoryview` through
  pythonnet costs ~10 µs of .NET exceptions, so the import routes probe with `PythonRuntimeInterop.TryMemoryview`
  (Python's own `try`, `None` on failure); pythonnet's `As<string>()` compares the type with `str` EXACTLY, so it refuses
  an `np.str_` (read a possible `str` subclass through Python's `str()`); a `T` cell written short over an arena string
  keeps the old offset's high bytes behind its text and numpy's `NpyString_dup` moves them: compare cells, and they must
  move too.
- End to end: `TextOpsLiveParityTests`, `TextConversionLiveParityTests`, `TextPandasLiveParityTests` over
  `TextLiveParityHarness` (numpy-born text crosses, NumSharp computes, the result crosses back, live numpy's answer
  over the same memory: ~13,500 cells of `numpy.strings`, the text ufuncs, sort / unique, casts both ways, printing,
  `.npy`, writes through views and the numeric neighbours; its conventions are in the project's CLAUDE.md, 7c), and
  `TextEdgeLiveParityTests` over the same harness: the EXPORT direction (NumSharp-born layouts handed to numpy), operand
  pairs of two text classes / widths / NA objects, broadcasting shapes, `out=` / `where=` / `dtype=` (the `out=` object
  returned), every `StringDType` NA kind, `numpy.char` and its chararray, `partition` / `mod` / `translate`, fused
  `np.evaluate` text expressions against numpy's composition, and numpy's functions called with NumSharp arrays through
  the codec. Their Python-free twins are `TextBridgeEdgeTwinTests` in the unit suite: NumPy's recorded answers
  (`test/oracle/gen_text_edge_twins.py`) for the cross-class, NA-kind, keyword and char-comparison families.
- Gates: `TextFixedWidthCrossingTests`, `TextStringDTypeCrossingTests` (live numpy, every NA kind × `coerce` × layout),
  `TextCrossingEdgeCaseTests` (cell histories, UTF-8 size boundaries, ranks 5–6, empty / read-only sources, a 64 KiB+
  arena, seeded random arrays and views, eight threads, `str` subclasses, corrupt cells on both sides failing cleanly with
  numpy's allocator lock released, 64 dimensions both ways and numpy's own refusal of 65).

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
