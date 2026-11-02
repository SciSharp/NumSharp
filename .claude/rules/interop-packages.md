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

`NumSharp.Interop.ParquetNet` (Parquet columns decoded straight into `NDArray` buffers) is unreleased: it builds, tests
and merges like everything else, but no NuGet package is created and no release or public documentation mentions it.
`NumSharp.Interop.OnnxRuntime` and `NumSharp.Interop.MLNet` were unreleased the same way from 2026-10-02 and ship from
0.80.0-beta1 (the owner's decision of 2026-11-02). Unreleased is exactly three things:
1. No package. The csproj sets `<IsPackable>false</IsPackable>`, which makes `dotnet pack` a silent no-op (exit 0, no
   `.nupkg`, not even a compile). CI still names it in the signing check and in `build-nuget`'s Build and Pack steps,
   and `create-release` lists exactly the `.nupkg` files that were packed (ids read from each nuspec), so packing,
   signing, publishing and the release text all follow that one property.
2. No docs. ParquetNet has no page (its `PackageProjectUrl` points at the page to write when it ships). DocFX publishes
   every `.md` under `docs/website-src`, toc entry or not, so a hidden package's page needs a `docfx.json` exclude.
3. Gated. The `interop-test` job builds and tests it on both TFMs; apart from `build-nuget`'s Build step on a release
   tag (which compiles it so that releasing it needs no workflow edit), nothing else compiles it in CI. Its text edge is
   `TextColumnTests`, over a pyarrow-written fixture (`test/oracle/gen_parquet_text_fixture.py`): the loaders skip text
   columns, the direct readers refuse them; under Parquet.Net 6 a text column's CLR type is `ReadOnlyMemory<char>`, so
   the bridge's text refusal never fires (`[OpenBugs]`). Its numeric edge is `NumericColumnParityTests`, over pyarrow
   fixtures (`test/oracle/gen_parquet_numeric_fixture.py`: every width at its extremes in four writer configurations, a
   zero-row file, wide decimals, a float16 column): numpy's `to_numpy()` view bit for bit across row groups, both faces
   streaming the same bytes, and the null policy against numpy's NaN / None widening (`default(T)`, a per-call fill
   never cached, Raise only where a null is actually read). Two `[OpenBugs]`: `ParquetConvert.ResolveFill` converts the
   fill for every column before any null is met (a NaN fill breaks every integer and decimal column, the required ones
   included), and converts it with `Convert.ChangeType` (a fraction ROUNDS, `"7"` is no Boolean) where numpy's setitem
   and `np.full` truncate. Parquet.Net 6.1.0's own limits are `[Misaligned]`: data page v2 RLE booleans throw
   `DivideByZeroException`, DELTA_BINARY_PACKED int8 / uint8 / decimal are refused, a float16 column refuses the whole
   schema, a decimal past 28-29 digits rounds or overflows, and the decimal's trailing zeros are dropped.

Keep it out of release notes (`docs/releases/RELEASE_<version>.md`; when that file is missing, GitHub generates the
notes from merged PR titles, so PR titles stay free of it too), the website, the root README, and the shipped XML docs
and README of every released package: a released package's IntelliSense must not point at a bridge nobody can install.
To release it: delete its `IsPackable` property, write its page and add its `toc.yml` entry and its `interop/index.md`
row and "See also" link (the two packages released for 0.80.0-beta1 got theirs back from git history). CI needs no
edit.

## The conversion packages

None of these computes NumSharp operations: each converts tensors between NumSharp and another library, so none has a
`TensorEngine` seam, a `[ModuleInitializer]` or a native asset (like the pythonnet bridge).

### `NumSharp.Interop.OnnxRuntime` (released from 0.80.0-beta1)

- `NDArrayOnnxInterop` verbs, `InferenceSessionExtensions`, `Postprocess`. Sources (`src/NumSharp.Interop.OnnxRuntime/`):
  `NDArrayOnnxInterop.{cs,Export.cs,Import.cs,NonTensor.cs,Text.cs}` (dtype maps, the export and import verbs,
  non-tensor outputs, text tensors), `OnnxCodecMode.cs`, `OrtTensor.cs` (the `OrtTensor` / `OrtTensor<T>` handles and the
  `ImportLease`), `UnmanagedMemoryManager.cs` (a `Memory<T>` over the unmanaged buffer),
  `InferenceSessionExtensions{,.Async}.cs`; `Postprocess.cs` is one source file compiled into both bridges
  (`src/NumSharp.Interop.Shared/`). ONNX Runtime's C# sources are vendored for reference at `refs/onnxruntime/csharp/` (a
  sparse submodule), its official samples at `refs/onnxruntime-inference-examples/c_sharp/`.
- `OnnxCodecMode { Auto, View, Copy }` (the members of the pythonnet codec's `NumpyCodecMode`) on every entry that could
  share memory: Auto is a view when ORT can read the buffer in place (C-contiguous and a bit-exact dtype), else a copy
  the handle owns; View throws instead of copying; Copy always copies.
- Export: `nd.AsOrtValue(mode)` (an `OrtValue` over the NumSharp buffer, held by the ARC-rooted handle `OrtTensor`, whose
  `IsView` says view or copy), `nd.ToOrtValue()` (an ORT-allocated copy), `AsDenseTensor<T>(mode)` / `ToDenseTensor<T>`
  (ORT's legacy tier, always row-major).
- Import: `ortValue.ToNDArray()` (an owning copy), `ortValue.AsNDArray(ownsValue, mode)` (a zero-copy lease over ORT's
  buffer, CPU only; a counted NumSharp alias for a value the package made). Imports of NumSharp memory (a value the
  package made, a fed input passed through) are counted aliases that inherit read-only; only an OWNING view adds GC
  memory pressure. Importing a tensor of 2 GB or more that ORT owns is refused up front: ORT exposes tensor memory only as
  an `int`-length Span.
- `session.Run(NDArray…)`: input names from the model metadata (overridable initializers and `optional(tensor)` inputs
  included), NumPy-`casting` dtype coercion checked on every call, declared-shape validation, copy / owning-view outputs.
  `session.RunInto` fills pre-allocated outputs (an output that overlaps an input is computed from a copy under Auto,
  refused under View). `session.RunIntoAsync` / `RunAsync` + `IsRunAsyncSupported` are ORT's native async (ORT 1.30.0 or
  later, a session with two or more intra-op threads; `RunAsync` allocates only a numeric output of a fixed declared
  shape). `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}`.
- Depends on `Microsoft.ML.OnnxRuntime.Managed [1.16.0, 2.0.0)` only: the native flavours (CPU, Gpu, DirectML) are
  mutually exclusive, so the consumer picks one.
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
- Gate: `test/NumSharp.Tests.Interop.OnnxRuntime`, over committed `.onnx` models from `test/oracle/gen_onnx_models.py`
  (which takes model names, so a new model does not rewrite the committed ones): the every-dtype × layout export matrix
  (a cell per mode), property-based, metamorphic, ORT-consumer (six feed routes through ORT's own Identity), bit-pattern
  fuzz, concurrency stress, ARC/fault injection, boundary and API-approval suites, `LifetimeGcTests` (every finalization
  order of eight scenarios as a staged real collection), and `SessionRunModeTests`, `ImportAliasTests`, `TextTests`,
  `PassThroughOutputTests`, `SessionRunAsyncTests`, `ValueOwnershipTests`, `MemoryManagerTests`, and `TextModelsEndToEndTests`
  (text through eleven string models - Concat, Transpose, Gather, Reshape, Slice, Unique, Where, Expand, Tile,
  StringNormalizer, LabelEncoder - held to NumSharp's numpy-verified twins and to `np.array(list)`'s width; Python
  onnxruntime's edges pinned `[Misaligned]`: an embedded NUL survives where it truncates, U+FFFD where it raises on a
  lone surrogate or non-UTF-8 output; onnxruntime 1.29's string Gather on axis 0 of a 2-D tensor copies one string per
  row, so the model gathers on axis 1), and `TextRunTierEdgeTests` (text input through `RunInto` into a preallocated
  numeric output, `Run`, `RunAsync` / `RunIntoAsync` over the fixed-shape LabelEncoder model - `RunAsync` cannot allocate
  for an output declared without dimensions -, the declared rank validated, StringNormalizer stopwords removing elements
  down to one empty string, eight concurrent string runs). The `interop-test` CI job
  runs it against ONNX Runtime 1.16.0 (the async tests Inconclusive, the refusal tested) and 1.31.0 (the only leg that
  runs the async tests: never set it below 1.30.0), with no Python. Mutation runs restore a file from memory only: copy
  untracked files out of the tree before one.
- Docs: `docs/website-src/docs/interop/onnxruntime.md` (on the site's Interoperability toc since 0.80.0-beta1).
- Example: `examples/NumSharpOnnxRuntimeExample` (regular csproj form over the published 0.80.0-beta1; no workflow runs
  it): MobileNetV2 downloaded on the first run, SkiaSharp decoding (ImageSharp 3.1.x restores with security advisories
  and 3.2+ needs a license key to build), NumSharp preprocessing into one NCHW batch, `Postprocess`, `RunInto`.

### `NumSharp.Interop.MLNet` (released from 0.80.0-beta1)

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
  route). The `interop-test` CI job runs it against ML.NET 2.0.1 and 4.0.2, with no Python.
- Docs: `docs/website-src/docs/interop/mlnet.md` (on the site's Interoperability toc since 0.80.0-beta1).
- Example: `examples/NumSharpMLNetExample` (regular csproj form over the published 0.80.0-beta1 and Microsoft.ML 5.0.1;
  no workflow runs it): a binary classifier trained from a `(rows, 5)` array as five scalar columns with the label
  converted OUTSIDE the model (so the model scores a features-only view), L-BFGS for run-to-run identical output (SDCA
  is multi-threaded and varied), k-means on one vector column.

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
- Gate: `test/NumSharp.Tests.Interop.System.Numerics.Tensors`, both TFMs, in the `interop-test` CI job. Text never
  crosses (`TextRefusalTests`): every text class × layout × verb is refused before any layout check, and no import
  verb instantiates for `string`; the refusal's TEXT is Core's storage-lane conversion error, not a text message
  (`[OpenBugs]` `TextExports_RefuseWithATextMessage_NotTheStorageLaneError`).
- Docs: `docs/website-src/docs/interop/system-numerics-tensors.md`.

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
