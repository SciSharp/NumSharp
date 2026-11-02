# Release QA: `NumSharp.Interop.OnnxRuntime` and `NumSharp.Interop.MLNet` (0.80.0-beta1)

**The tracking file of an 8-hour unattended QA run.** Read it top to bottom to resume: the brief, how the run works,
the work list (section 3), the findings (section 4), the list for the owner (section 5), the examples (section 6) and the
log (section 7). Every new idea becomes a work-list row before it is worked on.

| | |
|---|---|
| Started | 2026-11-02 21:52 +0200; planned end 2026-11-03 05:52 +0200 |
| Branch | `qa/interop-release` (local), worktree `K:\source\NumSharp\.claude\worktrees\interop-qa` |
| Base | `strings` `23ce02ea` (journey4 + strings, the beta's line); release prep commit `1c001253` on top |
| Scratch | `C:\Users\ELI\AppData\Local\Temp\claude\K--source-NumSharp\a4c6ff8e-6ba9-48a1-8c05-d4202f4019c4\scratchpad` (wiped nightly: nothing durable lives only there) |
| Python | `C:\Users\ELI\.claude\python\python.exe` (3.12, numpy 2.4.2, onnx 1.22.0, onnxruntime 1.29.0, torch 2.12.1, transformers 4.45.2, scikit-learn 1.8.0, Pillow 11.3.0); scratch `venv/` adds skl2onnx |

## 1. The brief (the owner, 2026-11-02)

QA both packages that 0.80.0-beta1 releases, deeply, for 8 hours, unattended (no questions): collect gaps, problems,
issues and difficulties; plan and prepare them for integration; find the real use cases and prove the packages work in
them; produce examples. Fix only what is not controversial and does not change designed behavior; defer the rest and
list it at the end. The owner put a lot of effort in already: a problem must be proven (a probe, a test, a measurement),
not suspected. Be file-oriented: this file holds everything; each new direction is added here with its reason.

## 2. How the run works

- One angle at a time from section 3. Each angle ends with: its findings in section 4, a log line in section 7, and its
  row marked `done`. A fix is its own commit on `qa/interop-release`, with its test; a deferral gets a section 5 entry.
- When the list runs dry: write the state down, then look for a different angle (real use cases, other personas, other
  platforms, scale, failure injection, docs, competitors, packaging) and add rows.
- Evidence first: a finding states how it was proven (probe file, test name, command, numbers). Probes live under
  `test/NumSharp.Tests.Interop.{OnnxRuntime,MLNet}` when they become tests, else in the scratch folder (named in the
  finding) with the essential facts copied here.
- Consumer probes run against the PACKED 0.80.0-beta1 nupkgs from a local feed (scratch `pkgs/`, `consumers/NuGet.Config`,
  an isolated `NUGET_PACKAGES` = scratch `nugetcache/`), as file-based apps:
  `cd consumers && NUGET_PACKAGES=<nugetcache> dotnet run X.cs -- args`. Source probes (scratch `probes/`) reference the
  worktree's projects with `#:project`. Both need `#:property PublishAot=false` (F05).
- Inputs already read: `docs/plans/onnxruntime-rework.md` (the owner's decisions per item, the deferred items V8a, C1,
  C2a, C2c and the open V12b: never re-propose those as new findings, cite them), `onnxruntime-business-review.md`
  (the use cases: the ResNet50 / Faster R-CNN / BERT QA / YOLOv3 / Stable Diffusion tutorials and their two
  hand-written loops), `.claude/rules/interop-packages.md`, the memories `onnxruntime-interop-package`,
  `mlnet-interop-package`, `interop-codec-modes-rule`, `interop-lifetime-gc-analysis`. Still to read:
  `onnxruntime-gap-review.md`, `onnxruntime.md`, `interop-lifetime-gc-analysis.md` (plans), the two docs pages (A13).

## 3. Work list

Status: `todo`, `doing`, `done`, `dropped` (with the reason).

| # | Angle | Why | Status |
|---|---|---|---|
| A1 | Baseline: both suites on Windows, net8.0 + net10.0, ORT 1.16.0 (floor) + 1.31.0, ML.NET 2.0.1 + 4.0.2 | Know what is green before probing; durations for CI | done (F08; ML.NET 4.0.2 leg re-run with the fixes) |
| A2 | Package inspection: nuspec (ids, versions, dependency ranges, TFMs, metadata), files, XML docs, symbols, strong names, SourceLink | A first release is judged on its package page and its IntelliSense | doing (ranges done: F04; metadata/XML docs/symbols left) |
| A3 | Consumer QA from a local feed: net8.0 (C# 12) and net10.0 console apps, floor and current dependency versions | Only an outside consumer sees restore warnings, CS0121 ambiguities, missing natives | todo |
| A4 | Real use case: image classification with a real ONNX model (SqueezeNet / MobileNet / ResNet), NumSharp preprocessing and post-processing, parity with Python onnxruntime | The business review's lead scenario (the ResNet50 tutorial's two loops) | done (F15) |
| A5 | Real use case: a transformer (BERT-family) with int64 token ids, attention masks, embeddings / QA logits, parity with Python | The BERT QA tutorial; int64 inputs, multiple inputs and outputs | done (F15) |
| A6 | Real use case: scikit-learn models exported to ONNX (ZipMap seq-of-maps, string labels, probabilities) | Classic tabular inference; the non-tensor readers and text | done (F09, F15) |
| A7 | Real use case: detection-style post-processing (boxes, scores, NMS) with NumSharp | YOLO / Faster R-CNN tutorials' output loops | todo |
| A8 | Stable-Diffusion `TensorHelper` replaced by NumSharp ops (latents, guidance, decode) | 60 copy-pasted files in the wild; a strong example | todo |
| A9 | ORT ergonomics against the raw ORT API: inputs/outputs by name, dynamic shapes, batching, sessions options, errors | A user compares with what they already write | todo |
| A10 | Performance: per-call overhead of `Run(NDArray)` against raw `OrtValue` code, zero-copy vs copy, allocations per call | The package promises "no copy"; overhead must not eat the gain | done (F19) |
| A11 | Leaks: 100K-run loops (Run, RunInto, AsOrtValue, AsDataView, cursors), process memory, live counters | Long-running inference services | done (F20, F21) |
| A12 | Lifetime in real apps: dispose orders, `using` patterns of the docs, async + cancellation | The suites prove the contract in a harness; apps hold arrays differently | todo |
| A13 | Docs QA: every snippet of `interop/onnxruntime.md`, `interop/mlnet.md`, both READMEs compiles and runs; claims match code | The pages go live with the beta | done for the page snippets (F10, F17, F18; scratch `consumers/docs_snippets.cs`); README blocks = the page's or `DocExampleTests` |
| A14 | ML.NET real use cases: train + predict on NumSharp data, OnnxTransformer, TextLoader to NumSharp, clustering, normalization | The ML.NET bridge's reason to exist | done (F01, F07) |
| A15 | ML.NET training needs Label + Features in ONE `IDataView`: can the bridge build a multi-array view? | Training scenarios, not just scoring | doing (F01: proposal for section 5) |
| A16 | Both bridges in one project: `Postprocess` and extension-method name clashes | A real app uses ORT and ML.NET together | done (F16) |
| A17 | CI coverage of the two packages as released (interop-test job, the packs, consumer steps) | The tag's gate | todo |
| A18 | Metadata and release text: descriptions, tags, project URLs, notes section 7A accuracy | First impression on nuget.org | todo |
| A19 | Dependency floors: the 1.16.0 / 2.0.0 floors against what consumers really install | The ranges are the contract | todo |
| A20 | Robustness: malformed models, huge or negative dims, odd strings (NUL, invalid UTF-8), unusual map keys | Inference services meet bad input | todo |
| A21 | `StringDType` (T) arrays through the ORT bridge | strings landed T; the bridge's TODO is due; another session saw an InvalidCastException | done (F03 fixed, F10, F11) |
| A22 | ML.NET text columns (`ReadOnlyMemory<char>`) against NumSharp's text dtypes | String labels and FeaturizeText are everyday ML.NET | todo (F02 fixed the refusal; the design is for section 5) |
| A23 | Examples: an ORT example and an ML.NET example in the regular NuGet form, proven from a local feed | The brief asks for examples | todo |
| A24 | NumSharp array classes as inputs: `NDMatrix`, `NDCharArray`, `NDMaskedArray`, views, F-order, broadcast, float16, int8 | A masked array's mask could be silently ignored | done (F14) |
| A25 | Error-message quality: wrong name / dtype / shape / disposed / null | Users debug through messages | todo |
| A26 | Thread safety: concurrent `Run` on one session with NumSharp inputs; concurrent `AsDataView` cursors | ORT sessions are shared across requests | todo |
| A27 | Linux (WSL) run of both suites | CI's ubuntu leg; macOS cannot run here | todo |
| A28 | Public API review: XML docs on every public member, naming consistency between the two bridges | The owner's documentation rule; a first release freezes names in users' code | todo |

## 4. Findings

Severity: **blocker** (cannot ship), **high** (a real user hits it), **medium**, **low**, **info**. Status: `fixed <sha>`,
`deferred` (section 5), `closed` (not a problem after all, with the reason).

| ID | Sev | Package | Finding | Evidence | Status |
|---|---|---|---|---|---|
| F01 | high | MLNet | A training view needs a vector `Features` column AND a `Label` column of another dtype in ONE `IDataView`; the bridge builds a view from ONE array, so the natural `X.AsDataView("Features")` + `y` cannot train (`ArgumentOutOfRangeException: Could not find label column 'Label'`). Working route found: `np.hstack(X, y.astype(float32))` as scalar columns + `ConvertType("Label", DataKind.Boolean)` + `Concatenate("Features", f0..fn)` - three extra steps, the label forced into the features' dtype. Training is ML.NET's main use. Root: `NDArrayDataView` holds one `_fed` array, one `_slice`, one `_itemsize`, so every column shares one array and one dtype. **Scoring hits it too:** a model trained with ML.NET's standard multiclass recipe (`MapValueToKey("Label")` + a trainer, the Iris sample) keeps the label mapping in its chain, so `model.Transform(X.AsDataView("Features"))` throws `Could not find input column 'Label'`; the workaround is the trainer alone, `((IEnumerable<ITransformer>)model).Last().Transform(view)` (proven: scores (4,6)). | scratch `consumers/ml_train.cs` (attempt 1 fails, attempt 2 trains, accuracy 0.9995); `consumers/docs_snippets.cs` (the recipe's last two steps) | deferred (P1) |
| F02 | medium | MLNet | Every export verb (`AsDataView` x2, `ToDataView` x2, `ToVBuffer`) of a text array (`U`, `S`, `StringDType`, 2-D `U`, `NDCharArray`) failed inside the implicit `NPTypeCode` -> `DType` conversion with Core's storage-lane message ("does not support fixed-width string arrays: it asked for a dtype by storage lane ... Use the array's own dtype"), which also calls a `StringDType` array fixed-width; the dtype map said only "NumSharp dtype String has no ML.NET column type." | scratch `probes/mlnet_text.cs` (5 forms x 8 calls, before the fix) | fixed `8e607929`: `FeedTypeCode` refuses `String` with a text message naming `NDArray.AsStringArray` + `LoadFromEnumerable`; test `ErrorTests.TextArrays_AllExportVerbs_RefuseWithTheBridgesTextMessage` |
| F03 | high | OnnxRuntime | A `StringDType` (`T`) array shares the text lane, so `IsText` is true and `TextToOrtValue` read it as `S` bytes: `ToOrtValue`, `AsOrtValue` and `Run` (string model) all threw `InvalidCastException: Unable to cast object of type 'System.String' to type 'System.Byte[]'` (with or without an NA). Python's onnxruntime refuses StringDType ("Numpy_type 2056 can't be converted to MLDataType"). | scratch `consumers/ort_stringdtype.cs` (packed beta) | fixed `8e607929`: `RefuseStringDType` in `TextToOrtValue`, `TextAsOrtValue` (before the View refusal) and `PrepareTextInput` (names the input); test `TextTests.StringDTypeArray_IsRefusedByEveryTextExport_AndTheSizedUCastItNamesRuns` |
| F04 | low | both | The NumSharp dependency is open-ended (`NumSharp >= 0.80.0-beta1`) while each bridge calls Core INTERNALS through the friend grant (OnnxRuntime: `ArraySlice.WrapShared`, `NDArray.Storage`; MLNet: `NDArray.Storage`) and Core's public API breaks between 0.x minors (118 breaking commits 0.70 -> 0.80). NuGet lets a user upgrade NumSharp alone. The 0.70.0 OpenBLAS / pythonnet / Bitmap packages have the same open range (OpenBLAS 0.70.0 references 8 Core internals; all 8 still resolve by name in 0.80). | scratch `internals_probe/internals.cs` + `out.txt`; nuspecs of the 0.80.0-beta1 packs and the cached 0.70.0 ones | deferred (P4) |
| F05 | medium | release (Core) | A .NET 10 file-based app (`dotnet run app.cs`, the quickest way to try a package) defaults to `PublishAot=true`, so NumSharp's first IL kernel throws `PlatformNotSupportedException: Dynamic code generation is not supported` (here: `X > 0` inside the ML.NET probe). `#:property PublishAot=false` fixes it. Every doc snippet a reader pastes into a file-based app hits this. | scratch `consumers/ml_train.cs` first run | deferred (P5) |
| F06 | info | release (Core) | NDW012 fires on ordinary consumer code: a 40-line ML.NET program got 4 warnings of ~600 characters each (`NDArray X = rng.randn(...)`). Deliberate analyzer design; noted for the beta's first impression. | same run | deferred (P5, mentioned) |
| F07 | info | MLNet | Works end to end on ML.NET 4.0.2 through a local feed (`NumSharp*` 0.80.0-beta1 packs): TrainTestSplit + Sdca, CrossValidate x5, AveragedPerceptron, FastTree, Sdca regression, MapValueToKey + SdcaMaximumEntropy (Score `(N,K)` back as float32, rows sum to 1), KMeans, NormalizeMinMax + PCA, Cache + ShuffleRows, model save + load + transform of a new view, NaN + ReplaceMissingValues, TextLoader numeric column -> NDArray. `LiveExports` back to 0. | scratch `consumers/ml_scenarios.cs` | closed (works) |
| F08 | info | both | Baseline (Windows, Release): OnnxRuntime 291 passed + 20 Inconclusive at ORT 1.16.0 on net8.0 / net10.0, 311 / 311 at 1.31.0 on both; MLNet 208 / 208 at ML.NET 2.0.1 on both. | scratch `baseline/baseline.txt` | closed |
| F09 | medium | OnnxRuntime | `session.Run(dict)` with the default `outputNames` (every output) on a scikit-learn classifier (the ZipMap `seq(map)` output, the most common tabular export) RUNS the model, converts `output_label`, then throws `NotSupportedException: ToNDArray needs a dense tensor OrtValue, but this value is ONNX_TYPE_SEQUENCE. A sequence of tensors is read with ToNDArrays() ...`. The message names `ToNDArray` (not called by the user), not the output, and points at readers the caller has no `OrtValue` for (Run disposed them). The metadata knew before the run (`session.OutputMetadata["output_probability"].IsTensor == false`); `RunInto` / `RunAsync` already refuse a non-tensor output up front with its name ("output 'seq' is ONNX_TYPE_SEQUENCE, not a tensor"). Not fixed here: the post-run path and its release of the first output are pinned by `SessionRunEdgeTests.Run_WithANonTensorOutputAfterATensorOutput_ReleasesWhatItMadeForTheFirst`, `optional(tensor)` outputs make an up-front check a design decision, and non-tensor outputs through `Run` are the owner's deferred C2a. | scratch `consumers/ort_sklearn.cs` (iris_lr, step 2) | deferred (P2) |
| F10 | medium | OnnxRuntime (docs) | The documented `StringDType` remedy, `nd.astype(np.dtype("U"))` (docs page Text section, README, the Text.cs TODO), throws `TypeError: cannot cast dtype StringDType() to <class 'numpy.dtypes.StrDType'>`: NumPy 2.4.2 refuses an UNSIZED `U` target for StringDType ("specify an explicit size") and NumSharp matches it. A sized cast works (`astype("U5")`: `alpha|βeta|`; NA -> `None` / `nan` as in NumPy). | Python 2.4.2 + scratch `probes/core_t_to_u.cs` | fixed `8e607929`: the docs, README, TODO and the refusal name a sized `U` |
| F11 | low | OnnxRuntime | `NDArrayOnnxInterop.TryToTensorElementType(np.dtypes.StringDType())` answers `true` / `String` (the lane is shared) although no verb crosses a StringDType array; its XML doc says "true when the dtype crosses". Left as is: `String` IS the element type such data would have, and a refusal there is an API answer change. | `NDArrayOnnxInterop.cs:151` (`case NPTypeCode.String`) | deferred (P3) |
| F12 | info | (Python ORT) | Python onnxruntime 1.29.0 reads an `S2` array past each element's field (`b'ab'` -> `'abc'` when the next field starts with `c`): a Python ORT bug, not ours; the NumSharp bridge strips the padding correctly. | earlier probe (pre-compaction) | closed (not ours) |
| F13 | info | Core | `nd.transpose(2, 0, 1)` does not compile (CS1501): Core's signature is `transpose(int[] premute = null)`, NumPy's `transpose(*axes)`; preprocessing code from NumPy needs `transpose(new[] {2, 0, 1})`. Owner-known NumPy-signature gap; not a bridge issue. | scratch `consumers/ort_vision.cs` first compile | closed (Core, known) |
| F14 | info | both | Array classes and layouts as inputs: `NDMatrix`, broadcast views, negative-stride views and an empty batch run through `session.Run` (values consistent with the base rows); `NDMatrix.AsDataView`, float16 (-> `Vector<Single, 10>`), int8 scalar columns work; a 0-d array is refused by `AsDataView` with a clear rank message; `NDMaskedArray` does not convert to `NDArray` (CS0029), so no bridge can drop a mask silently; `NDCharArray` hit F02. | scratch `consumers/subclasses.cs` | closed (works; F02 fixed) |
| F16 | low | both | An app that imports both bridges' namespaces (scores with ONNX Runtime, trains with ML.NET) cannot write `Postprocess.Softmax(...)`: `error CS0104: 'Postprocess' is an ambiguous reference between 'NumSharp.Interop.MLNet.Postprocess' and 'NumSharp.Interop.OnnxRuntime.Postprocess'` (one source file compiled into both assemblies under two namespaces). Everything else coexists: no other public name collides (the extension verbs take different receiver types), ORT output -> `np.hstack` -> `AsDataView` -> `NormalizeMinMax` -> `ToNDArray` runs, and both classes give identical results. Workaround: a `using` alias. The names freeze at the first release, so this is the time to decide. | scratch `consumers/both_bridges.cs` (+ `both_bridges_amb.cs` for the CS0104) | deferred (P6) |
| F17 | low | OnnxRuntime (docs) | The page's hero snippet used `mean` / `std` without defining them; a reader who writes them as doubles (`np.array(new[] { 0.485, ... })`) gets a float64 input, which `Run` refuses for the float32 model under the default `casting="safe"` (`InvalidCastException: input 'data' holds Double but the model declares Single (Float)`), as Python's onnxruntime refuses a float64 array. With float32 constants the snippet runs (ResNet50 v2: logits (1,1000) float32, top-5 int64, probabilities sum to 1). | scratch `consumers/docs_snippets.cs` steps 1-2 | fixed (this commit): the snippet defines `mean` / `std` as float32 and says why |
| F18 | medium | OnnxRuntime | `RunAsync` (the bridge allocates the outputs) refuses every output with a dimension fixed only at run time, and real models almost always have one: of the 8 real models in this run only SqueezeNet 1.1 (fixed batch 1) qualifies; the page's own hero model fails (`RunAsync: output 'resnetv24_dense0_fwd' is declared with the shape ('N', 1000) ... call RunIntoAsync`). Documented, and the advice works (`RunIntoAsync` with an output allocated by the caller ran in the same step), but for named symbols the shape IS known before the run: resnet50 `data ['N',3,224,224]` -> `['N',1000]`, MobileNetV2 `input ['batch_size',...]` -> `['batch_size',1000]`, MiniLM `['batch_size','sequence_length']` -> `['batch_size','sequence_length',384]` (every output symbol appears on an input). The skl2onnx exports use unnamed dynamic dims (no symbol), which only shape inference resolves. | scratch `consumers/docs_snippets.cs` (async steps), `probes/model_dims.py` output in this row | deferred (P7) |
| F19 | info | OnnxRuntime | Per-call cost (Release, pinned to 2 CPUs, ORT 1 thread, best of 7 interleaved rounds, ORT 1.31.0). Identity, 16 floats: raw ORT 5.6-6.3 us; `Run(x)` 9.3-9.5 us; zero-copy outputs 9.3 us; `RunInto` 6.6 us; `AsOrtValue` + ORT `Run` + `ToNDArray` 8.5-8.7 us; the bridge allocates 1.3-2.3 KB managed per call (raw ORT 0.5-0.6 KB). Identity, 602 KB: the typical raw code (`CreateTensorValueFromMemory` + `Run` + `ToArray`) 85-97 us with a 603 KB LOH array per call (1,600 gen2 GCs per 14,000 calls); `Run(x)` 52-53 us (the copy out; no gen2); zero-copy outputs 32 us; `RunInto` 21.5 us = raw ORT with a pre-made input (21 us). SqueezeNet 1.1: every route 8.69-8.79 ms (overhead < 0.1%). The zero-copy outputs' `GC.AddMemoryPressure` induces gen2 collections in a small process (496 per 10,000 calls of 602 KB with an empty heap, 223 ms of pauses) but not in a large one (5 per 10,000 with a 229 MB live heap); the route stays faster than the copy route in both. | scratch `consumers/ort_perf.cs`, `ort_perf_gc.cs`, `ort_pressure.cs` | closed (no problem) |
| F20 | low | OnnxRuntime | A FORGOTTEN `ToOrtValue` result balloons native memory: 20,000 dropped copies of a 602 KB array peaked at +4.4 GB private bytes (2.3 / 1.0 / 3.3 / 0.7 / 3.0 GB at each fifth) before the finalizers caught up, back to +3 MB after a full collection. The value is ORT's own (`OrtValue`, ORT-allocated memory), and ORT's C# API calls `GC.AddMemoryPressure` nowhere (vendored sources), so the GC never learns about the memory; the same drop of raw ORT values behaves the same. The bridge's OWN objects do not: forgotten `Run` outputs peaked at +24 MB, forgotten owning views +60 MB (they report pressure), forgotten `AsOrtValue` / `AsOrtValue(Copy)` handles +9 / +11 MB. | scratch `consumers/leak_loops.cs` | docs fixed (this commit: the page's Lifetime section says to dispose it, with the numbers); the pressure itself deferred (P8) |
| F21 | info | both | No leak in any route: five phases of 200,000 calls each, after a full collection per phase, plateau after the first phase (private MB above start: raw ORT 15; `Run(x)` 24.5-25.8; `AsOrtValue` disposed 23-27, forgotten 37-48 with a flat 5.6 MB managed heap; plain `np.ones` 23-31) and the live counters end at 0/0. Single long loops (200K-1M calls) of `Run`, zero-copy outputs, `AsOrtValue`, `AsDataView` + cursor reads (disposed and forgotten) and ML.NET `ToNDArray` stay bounded (peaks +11 to +75 MB) with every live counter back at 0. | scratch `consumers/leak_loops.cs`, `leak_long.cs`, `leak_phases.cs` | closed (no problem) |
| F15 | info | OnnxRuntime | Real models, bit-exact parity: ImageNet SqueezeNet 1.1 / MobileNetV2 / ResNet50 v2 (preprocessing bytes == NumPy; logits bytes == Python onnxruntime 1.29.0; top-5 equal); all-MiniLM-L6-v2 sentence embeddings (three int64 inputs; `last_hidden_state` bytes == Python; NumSharp mean pooling + L2 norm bytes == NumPy; cosine-similarity matrix max diff 2.98e-7, from the managed matmul's summation order); scikit-learn LogisticRegression / RandomForest (string labels) / GradientBoosting (labels, probabilities, regression bytes == Python; ZipMaps through ORT's `Run` + `ToMaps`). | scratch `consumers/ort_vision.cs`, `ort_text.cs`, `ort_sklearn.cs` + `realdata/make_*_ref.py` | closed (works) |

## 5. For the owner (deferred, with what was proven)

Each entry: the finding, what was proven, the options, and a recommendation. Nothing here was changed.

- **P1 (F01) ML.NET training views.** Proposal pending (A15).
- **P2 (F09) Non-tensor outputs in the default `Run` fetch.** Proposal pending.
- **P3 (F11) The dtype map's answer for `StringDType`.** Proposal pending.
- **P4 (F04) The NumSharp dependency range.** Proposal pending.
- **P5 (F05, F06) First-run experience in file-based apps.** Proposal pending.
- **P6 (F16) `Postprocess` in both bridges.** Proposal pending (decide before the tag: names freeze).
- **P7 (F18) `RunAsync` and run-time dimensions.** Proposal pending.
- **P8 (F20) Memory pressure for `ToOrtValue` results.** Proposal pending.

## 6. Examples

## 7. Log

| Time (+0200) | Entry |
|---|---|
| 21:52 | Run started. Prep commit `1c001253` (bridges packable, version 0.80.0, notes renamed) on `qa/interop-release`; packs of NumSharp / OnnxRuntime / MLNet 0.80.0-beta1 build. |
| 22:10 | Context compacted; this file brought up to date from the transcript: A1, A4, A5, A6, A14, A21, A24 done; F09-F15 recorded. |
| 22:25 | F02, F03, F10 fixed with tests (ML.NET `ErrorTests`, ORT `TextTests`: 23 + 16 green on net10.0 at the floors). F09 deferred, not fixed (pinned post-run design, C2a). Committed `8e607929`. |
| 22:50 | A16 (F16: `Postprocess` CS0104 with both namespaces) and A13 page snippets (all run as written given float32 constants and a features-only model; F17 fixed in the page; F18 `RunAsync` vs run-time dims; F01 extended to scoring with the MapValueToKey recipe). Committed `a4a4b03b`. |
| 23:25 | A10 + A11: overhead ~3 us / call, `RunInto` = raw ORT, the copy route beats typical raw code; no leak in any route (phased plateau); F20 `ToOrtValue` forgotten -> +4.4 GB peak (ORT's value, no pressure): documented on the page. |
