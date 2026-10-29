# Gap review: `NumSharp.Interop.OnnxRuntime` vs. the full ONNX Runtime surface

**Date:** 2026-09-23 · **Package at:** `journey4` @ `33a4d2cf` · **Compared against:** ORT C# + native C API
source in `refs/onnxruntime` (a FULL checkout at `v1.29.0`, with tags through `v1.30.0` fetched) · **NuGet current:**
`Microsoft.ML.OnnxRuntime.Managed` **1.30.0**.

Companion to `docs/plans/onnxruntime.md` (the design, decisions §13 settled — not re-litigated here) and
`docs/plans/onnxruntime-business-review.md`.

## 0. Method — every finding below was executed, not inferred

| Evidence | What it settles |
|---|---|
| The package's 9 source files read in full; ORT C# `OrtValue`, `InferenceSession`, `OrtIoBinding`, `OrtAllocator`, `SessionOptions`, `ManagedProjections`, `OrtTypeInfo` read; native `IOBinding.cc`, `onnxruntime_c_api.cc`, `onnxruntime_typeinfo.cc`, `TensorSeq.h`, `inference_session.cc`, `feeds_fetches_manager.h`, `qnn_allocator.cc`, `cuda_allocator.h` read at the cited lines | what ORT actually does, per version |
| The existing suite: **159/159 green** on ORT **1.16.0** (floor, net10.0) and on **1.30.0** (`-p:OnnxRuntimeVersion=1.30.0`) | the "bump the current pin" claim |
| Three scratch probes (`dotnet run`, REAL sessions) over 10 purpose-built `.onnx` models + the committed ones. Probe A (`P*`) ran on **1.29.0 AND 1.16.0** with identical outcomes except P12 (see G10) and P5b (1.29.0 only — it kills the process); probe B (`B*`) on 1.29.0; probe C (`C1`) on the 1.16.0 floor | every finding's evidence column |
| The NumSharp ownership analyzer (`NumSharp.Build.Analyzer`) run over the package sources AND over a consumer file | G11 |

Model/probe reproduction recipes are in §6. Probe IDs (`P1`, `B1`, `C1`, …) are cited per finding.

## 1. Verdict

The dense-tensor core is sound and well-gated: 12 zero-copy dtypes (Half ↔ Float16 bit-exact), the ARC-pin
export / lease import lifetime model, contiguity rules, the metadata-driven `Run` tier with NumPy `casting`, and
pre-allocated outputs all behave as documented on both ends of the supported range — and the package sources
are **clean under the ownership analyzer (0 NDW012/016/017)**.

The gaps cluster in five places:

1. **Two correctness defects** on documented routes — Fortran-order `AsDenseTensor` (G1) and >2 GB import (G2).
2. **One lifetime hole in the model**: three ORT APIs keep dereferencing NumSharp memory AFTER the `OrtTensor`
   handle is gone — `OrtIoBinding`, `SessionOptions.AddInitializer`, `OrtValue.CreateSequence` (G3).
3. **The `Run` tier only speaks dense tensors**: every scikit-learn-shaped model (ZipMap probabilities, string
   labels) fails through it, and optional / overridable / string / sequence / map inputs cannot be fed (G4–G7).
4. **Missing tiers**: `RunAsync` (G8, with an ORT < 1.30 caveat), IoBinding (folded into G3).
5. **Polish**: double copies (G9), a CPU-memory allowlist that misses newer memory kinds (G10), a
   non-disposable multi-output result (G11), BFloat16 bridging (G12), docs/CI drift (G13).

## 2. Findings

Severity: **High** = silent wrong answer, or documented advice that fails · **Medium** = a common workflow is
blocked · **Low** = niche, perf, or polish.

| ID | Sev | Finding | Evidence |
|---|---|---|---|
| G1 | High | F-order `AsDenseTensor<T>` builds a `reverseStride` tensor ORT refuses to feed; the `.Memory` route feeds it **transposed** | P1, P1b |
| G2 | High | `AsNDArray` / `ToNDArray` on tensors > 2 GB fail (2–4 GB) or wrap (≥ 4 GB); docs say "share zero-copy instead" | P2, P2b, P2c |
| G3 | High | IoBinding / `AddInitializer` / `CreateSequence` outlive the `OrtTensor` pin — use-after-free if the NDArray dies first | P3, C1, B2 |
| G4 | Medium | `Run(dict)` throws on any model with a non-tensor output in the fetched set (scikit-learn classifiers) | P7, P8 |
| G5 | Medium | `ToMap` / `ToMaps` refuse string-keyed maps (every string-labelled ZipMap) | P6 |
| G6 | Medium | The `Run` tier accepts only `NDArray` inputs — no string / sequence / map / raw `OrtValue`, so mixed models are unfeedable | P9, B2 |
| G7 | Medium | `PrepareInput` refuses `optional(tensor)` inputs and overridable initializers that ORT accepts | P4, B1 |
| G8 | Low | No `RunAsync` tier — and ORT's own `RunAsync` is only GC-safe from **1.30.0** | P14, ORT #32015 |
| G9 | Low | `ToNDArrays` / `ToMap(s)` copy every element twice | P13, B3 |
| G10 | Low | `IsCpuAccessible` refuses real CPU memory kinds on newer ORT (`QnnHtpShared`, `CpuAligned4K`) | P12 |
| G11 | Low | `Run(dict)` returns a non-disposable `IReadOnlyDictionary<string, NDArray>` the analyzer cannot see | consumer probe |
| G12 | Low | BFloat16 refused although a lossless widen / RNE narrow needs no Core dtype | P11 |
| G13 | Docs/CI | CI "current" is 1.29.0 (NuGet: 1.30.0, suite green); test counts, 2 GB advice, F-order claims drifted | suite runs |

### G1 — Fortran-order `AsDenseTensor<T>` is unusable by ORT, and one route is a silent wrong answer (High)

`NDArrayOnnxInterop.Export.cs:185-194` shares an F-contiguous array as `new DenseTensor<T>(memory, dims,
reverseStride: true)`, and the README verbs table / website page (L65, L119-120) advertise it "for
`NamedOnnxValue.CreateFromTensor`". But ORT's only DenseTensor→OrtValue projection, `PinAsTensor`
(`OrtValue.shared.cs:1567-1571`), throws for **any** reversed-stride tensor:

- **P1:** `NamedOnnxValue.CreateFromTensor("X", h.Tensor)` + `session.Run(...)` → `NotSupportedException: Tensor
  of reverseStride is not supported` (1.16.0 and 1.29.0).
- **P1b:** the other documented route — `OrtTensor<T>.Memory` "for `OrtValue.CreateTensorValueFromMemory`"
  (`OrtTensor.cs:170`) — has no stride parameter, so ORT reads column-major memory as row-major: identity model
  output row 0 = `[0,4,8,1]` for a source row 0 of `[0,1,2,3]`. **No error, wrong data.**

The existing test (`DenseTensorTests.AsDenseTensor_FortranOrder_SharesColumnMajor_WithReverseStride`) checks
element indexing on the `DenseTensor` and never feeds it to ORT, which is why this stayed green.

**Fix:** refuse non-C-contiguous input in `AsDenseTensor` with the same message as `AsOrtValue` (a view verb
must not copy; `ToDenseTensor` already copies in logical order). Keep `reverseStride` support on the IMPORT side
(`denseTensor.ToNDArray()` / `AsNDArray()` of a user-built column-major tensor) — that direction is correct.
Add a test that feeds the result through a real `NamedOnnxValue` run.

### G2 — Import of tensors > 2 GB is broken (High)

ORT's managed `GetTensorBufferRawData` computes the byte length as `long` and constructs
`new Span<byte>(ptr, (int)bufferLenInBytes)` — an unchecked narrowing (`OrtValue.shared.cs:712-716`). Every
public data accessor (`GetTensorMutableRawData`, `Get…DataAsSpan<T>`, the experimental `TensorSpan` forms) goes
through it, and so do `ToNDArray` (`Import.cs:41`) and `AsNDArray` (`Import.cs:93-96`, the pointer is taken from
that span):

| Tensor | `AsOrtValue` (export) | `AsNDArray` | `ToNDArray` |
|---|---|---|---|
| 2.2 GB uint8, NumSharp-owned (P2) | ✅ works (native pointer + `long` length) | ❌ `ArgumentOutOfRangeException` from inside ORT | ❌ same |
| 2.2 GB uint8, ORT-allocated like a session output (P2b) | — | ❌ same | — |
| 4.4 GB uint8 (P2c) | ✅ | ⚠️ "works": `(int)` wrapped to 105,032,704 → the lease's `GC.AddMemoryPressure` is 42× too small | ❌ misleading `InvalidOperationException: ORT reports 105032704 tensor bytes but shape (4400000000)…` |

The README ("copying a tensor over 2 GB … throws — share it zero-copy instead", L160-161), the website (L185-186)
and `TooLargeForSpanMessage` (`Export.cs:263-265`, "Share … with AsOrtValue / AsNDArray") are right for export
and wrong for import. Real sizes: LLM prefill logits `(1, 8192, 128256)` float32 (Llama-3 vocabulary) =
4,202,692,608 bytes — inside the throwing band; long-context KV-cache `present.*` outputs.

**Fix, in two steps.** (a) Now: pre-check `ElementCount × itemsize > int.MaxValue` in `ToNDArray`/`AsNDArray`
and throw a `NotSupportedException` that names the ORT limitation; correct the three texts. (b) Real fix: the
data pointer is not reachable through ORT's public managed API for such tensors — it needs the internal
`OrtValue.Handle` (reflection) plus `OrtGetTensorMutableData` (the internal `NativeMethods` delegate, or our own
`OrtGetApiBase()` table call — the `OrtApi` struct is append-only, so the slot is stable). Either way it is
trimming-hostile and should light up only when the fast path fails. File the truncation upstream (§4, U2).

### G3 — Three ORT APIs keep reading NumSharp memory after the `OrtTensor` pin is released (High)

The handle model is "keep the `OrtTensor` alive across the call". Three ORT APIs retain the data pointer beyond any
single call, so that rule is not enough and the package offers no way to tie the pin to the right lifetime:

- **`OrtIoBinding.BindInput` / `BindOutput`** store their OWN copy of the OrtValue (`IOBinding.cc:14-36`, 81-104;
  ORT's C# even disposes its temporary OrtValue right after binding, `OrtIoBinding.shared.cs:218-222`). **P3:**
  bind `nd.AsOrtValue().Value`, dispose the handle (`LiveExports == 0`), mutate `nd`, `RunWithBinding` → output
  reads the mutated value `42`. Had `nd` been disposed instead, that read is a use-after-free.
- **`SessionOptions.AddInitializer(name, OrtValue)`** (present at 1.16.0, `SessionOptions.shared.cs:589`; ORT's doc:
  the buffer "must outlive the session object"). **C1:** weights injected from an `NDArray` work zero-copy
  (`y = [1001, 2002, 3003]` vs. stored `[11, 22, 33]`), and a later mutation of the NDArray changes the next run's
  output — the session reads NumSharp memory for its whole lifetime.
- **`OrtValue.CreateSequence`** — `TensorSeq::Add` shares the element tensors (`TensorSeq.h:81-85`). **B2:** a
  sequence input built from two `AsOrtValue` handles runs (`len = 2`); the pins must outlive the sequence.

**Fix:** one "keeper" concept reused three ways — the owner holds the `OrtTensor` handles until the owning ORT
object is disposed/cleared:
`NDArrayIoBinding` (wraps `OrtIoBinding`: `BindInput(name, NDArray)`, `BindOutput(name, NDArray)`,
`BindOutputToDevice`, `Run`, `ClearBoundInputs` releases pins); `sessionOptions.AddInitializer(name, NDArray)`
returning a disposable keeper (or an `NDArrayInitializers : IDisposable` the caller disposes after the session);
`nd[].ToOrtSequence()` returning a handle that owns the sequence + element pins (also closes half of G6). Until
then, document the rule next to "keep the handle alive across `Run`".

### G4 — The `Run` tier cannot return non-tensor outputs (Medium)

`CopyOutputs` calls `ToNDArray()` on every fetched output (`InferenceSessionExtensions.cs:168`), and `outputNames`
defaults to ALL outputs. skl2onnx classifiers emit `label` + a ZipMap `sequence(map(…, float))` by default:

- **P7** (int labels): `session.Run(dict)` → `NotSupportedException` (sequence output); `session.Run(nd)` →
  "the model has 2 outputs"; only `outputNames: ["label"]` works, by dropping the probabilities.
- **P8** (string labels): the `label` output itself is a string tensor → unreadable through the tier at all.

**Fix:** a result type that carries every ONNX value kind — tensors as `NDArray`, string tensors as `string[]`
(+ shape), sequences as `NDArray[]`, maps as `(keys, values)` — or at minimum a mode that returns non-tensor
outputs as raw `OrtValue`s instead of throwing. Making it `IDisposable` also fixes G11.

### G5 — String-keyed maps are refused (Medium)

`ToMap` throws on string keys (`NonTensor.cs:70-73`), so `ToMaps` fails on every string-labelled ZipMap
(`sequence(map(string, float))`). **P6:** ORT reads it fine — keys `[bird, cat, dog]` via
`GetStringTensorAsArray()`, values `[0.2, 0.1, 0.7]` via `ToNDArray()`. **Fix:** return `(string[] keys, NDArray
values)` for string-keyed maps (a sibling method or a small `OnnxMap` result with either key form).

### G6 — The `Run` tier accepts only `NDArray` inputs (Medium)

A model with one string input and one float input cannot go through the tier (**P9**: omitting the string input
fails inside ORT; there is no way to pass it), although raw ORT runs it with a string `OrtValue` next to
`x.AsOrtValue().Value`. Same for sequence inputs (**B2** shows the raw route works over NumSharp memory) and map
inputs (skl2onnx `DictVectorizer`). **Fix:** accept `NDArray | OrtValue | OrtTensor | string[] (+shape) |
NDArray[]` per input (an `object`-valued dictionary or an `OnnxInput` union), plus export verbs
`ToOrtStringTensor(string[] values, long[] shape)`, `ToOrtSequence(NDArray[])` (G3) and `ToOrtMap(keys, values)`.

### G7 — Optional and overridable inputs are refused though ORT accepts them (Medium)

`PrepareInput` (`InferenceSessionExtensions.cs:225-228`) looks only in `session.InputMetadata` and refuses
`!meta.IsTensor`:

- **Optional (P4):** an `optional(tensor(float))` input's metadata is `ONNX_TYPE_OPTIONAL` (`IsTensor == false`), so
  the tier throws — with a wrong hint ("feed it with … `OrtValue.CreateSequence / CreateMap`"). ORT accepts a
  plain tensor there (`inference_session.cc:3016-3031`, `IsOptionalTensor`); the raw run returns `[1, 2, 3]`.
- **Overridable initializers (B1):** ORT's `LookupInputMetadata` also consults `OverridableInitializerMetadata`
  (`InferenceSession.shared.cs:935-936`). A model whose initializer `w` is also a graph input: the tier throws
  "the model has no input named 'w'"; raw ORT feeds it (`y = [101, 202, 303]`). This arises from exports with
  `keep_initializers_as_inputs=True` and whenever a caller wants to swap weights per run.

**Fix:** unwrap `meta.AsOptionalMetadata().ElementMeta` when the element is a tensor (dtype/shape checks then
apply to it), fall back to `OverridableInitializerMetadata`, and fix the hint. Related, from the same property:
`NodeMetadata.IsTensor` is also true for **sparse**-tensor metadata (`InferenceSession.shared.cs:1970`), so a
sparse input would pass the tier's checks and fail in ORT — refuse it up front with a clear message.

### G8 — No `RunAsync` tier; ORT's own `RunAsync` is GC-safe only from 1.30.0 (Low)

`InferenceSession.RunAsync(runOptions, names, values, outputNames, outputValues)` exists at the 1.16.0 floor
(#16890) and composes with `AsOrtValue` inputs and pre-allocated `AsOrtValue` outputs (**P14**: `y = [5, 6, 7]`).
An NDArray tier is the natural owner of the handle lifetimes across the `await`. **But** before **1.30.0** ORT
passed its name/handle arrays to native code pinned only for the P/Invoke call, not until completion — fixed by
`8239590229 Pin C# RunAsync arguments until completion (#32015)`; no wrapper can pin ORT's private arrays. So:
gate the tier on the loaded ORT version (≥ 1.30.0) or document it as unsafe below. Map a `CancellationToken` to
`RunOptions.Terminate`.

### G9 — Non-tensor readers copy every element twice (Low)

`OrtValue.GetValue(i, allocator)` already returns a FRESH ORT-allocated copy for sequence and map elements
(`onnxruntime_c_api.cc:2058-2069`; **P13**: two `GetValue(0)` calls return distinct buffers), and
`ToNDArrays`/`ToMap(s)` then `ToNDArray()` it again (`NonTensor.cs:43-48, 68-74`). **B3** (16 MB element): 5.99 ms
vs 4.76 ms with `element.AsNDArray(ownsValue: true)` — 1.26×, and half the peak memory. **Fix:** wrap the
element instead of copying it; the result still owns its memory and the source may be disposed.

### G10 — `IsCpuAccessible` refuses real CPU memory kinds on newer ORT (Low)

`NDArrayOnnxInterop.cs:292-300` accepts `Name == "Cpu"`, mem-type `CpuInput`/`CpuOutput`, or equality with the
default CPU info. Real `CudaPinned` reports `OrtMemTypeCPUOutput` (`cuda_allocator.h:54-61`) — accepted, the docs
are right. But `QnnHtpShared` (the QNN EP shared-memory allocator on Snapdragon / Windows-on-ARM NPUs) is a
**CPU, host-accessible** device with `OrtMemTypeDefault` (`qnn_allocator.cc:115-121`), and `CpuAligned4K` is a CPU
device too; **P12** (1.29.0): both refused ("…which the CPU cannot address"). Neither kind exists at 1.16.0, so
this is a newer-ORT gap. **Fix:** add both names, and light up `OrtMemoryInfo.GetDeviceMemoryType()` (absent from
the C# API through 1.23.0) when present. Context: plain `Run` outputs always land on CPU even for GPU EPs (the
fetch target defaults to CPU, `feeds_fetches_manager.h:75`), so device tensors arise only from IoBinding device
outputs; `OrtEnv.CopyTensors` (C# since **1.23.1**) could light up a device→host `ToNDArray` for those.

### G11 — The multi-output `Run` result is not disposable (Low)

`Run(dict)` returns `IReadOnlyDictionary<string, NDArray>`: it cannot be `using`-disposed, and NumSharp's
ownership analyzer — which every NumSharp consumer gets — does not see it. Consumer probe: a discarded
`s.Run(x)`, `v.ToNDArray()`, `v.AsNDArray(ownsValue: true)`, `v.ToNDArrays()` and even `x.AsOrtValue()` are all
flagged NDW012; a discarded `s.Run(dict)` is not. **Fix:** fold into G4's result type (`IDisposable`, or an
`INDArrayCarrier`-style carrier the analyzer already understands).

### G12 — BFloat16 can be bridged without a Core dtype (Low)

Refusal is the current, documented decision. But import is a lossless widen (**P11**: `[1.5, -2.25, 3]` via
ORT's own `BFloat16 → float`), and export is one round-to-nearest-even narrow — so `ToNDArray(…, widen: true)` →
float32 and `ToOrtValue(nd, TensorElementType.BFloat16)` would unblock bf16 model I/O today, without waiting for a
Core `bfloat16` dtype. (Float8 / Int4 / Float4 are NOT bridgeable: see U3.)

### G13 — Docs / CI drift (Docs/CI)

- CI's "current" leg pins **1.29.0** (`build-and-release.yml:413-421`); NuGet current is **1.30.0** and the suite
  is **159/159** on it — bump the one number (plus the README/website "Verified against" banners, and the
  `refs/onnxruntime` checkout to `v1.30.0`).
- "148 tests" in README L28 and website L19 — the suite has **159**.
- The > 2 GB advice (G2), the F-order `NamedOnnxValue` claims (G1), the optional-input hint (G7).
- `NumSharp.Interop.OnnxRuntime.csproj` explains the floor via "a consumer whose resolved version is OLDER … gets
  a hard `FileLoadException`": every 1.16.0 asset of `Microsoft.ML.OnnxRuntime.dll` carries AssemblyVersion
  **0.0.0.0** (1.29.0 carries 1.29.0.0 — read from the NuGet cache's DLL metadata), so an older ORT would bind and
  fail later with `MissingMethodException`, not `FileLoadException`; the NuGet range is what actually enforces
  the floor.
- `docs/plans/onnxruntime.md` §6.5 says `GetTensorDataAsSpan` "throws for non-CPU memory"; it does not
  (`GetTensorBufferRawData` never checks the location) — the package's own `EnsureCpuAccessible` pre-check is
  what makes the copy path safe. Keep that pre-check.

## 3. Verified — NOT gaps

- Export and every numeric dtype / 0-d / empty / view-at-offset path, on both 1.16.0 and 1.29.0/1.30.0 (suite).
- `AsOrtValue` itself handles > 2 GB (P2: 2.2 GB exported fine) — only import is affected (G2).
- Optional **outputs holding a value** read fine through the tier (P5: `y = [1, 2]`).
- GPU sessions: plain `Run` returns CPU outputs, and pre-allocated NumSharp outputs receive the device→host copy
  — both work with today's verbs.
- The ownership analyzer reports **0** diagnostics on the package sources (built with the analyzer the NumSharp
  package ships), and flags consumers' discarded results except the one in G11.
- Thread safety, finalizer safety nets, `LiveExports`/`LiveImports` (suite + every probe ended at 0/0).

## 4. Upstream ORT issues (not fixable here — document, and consider filing)

- **U1 — A None optional graph output crashes the process.** Running a model whose `optional(tensor)` output is
  None through ORT's C# `session.Run` → native **AV 0xC0000005** in ORT's own `OrtValue.InitOnnxType()` (P5b, 1.29.0):
  `OrtTypeInfo::FromOrtValue` calls `value.Get<Tensor>()` on a tensor-typed value with no data
  (`onnxruntime_typeinfo.cc:169-176`). No wrapper can intercept it; the C# API also never exposes `HasValue`.
- **U2 — `GetTensorBufferRawData` truncates the byte length to `int`** (the root cause of G2): throws for 2–4 GB,
  silently returns a short span for ≥ 4 GB.
- **U3 — Float8 / Int4 / Float4 graph I/O is unreachable from C#.** The C# `TensorElementType` enum ends at
  `BFloat16 = 16, DataTypeMax = 17`; a model with a float8 output fails in ORT's metadata lookup ("Unregistered
  TensorElementType value of: DataTypeMax", `InferenceSession.shared.cs:1677-1683`) before any data exists (P10).
- **U4 — No sparse-tensor API in C#** (only `IsSparseTensor`); the native `CreateSparseTensor*` / `Fill*` /
  `GetSparseTensor*` functions are in the `OrtApi` table but not bound.

## 5. Recommended order

1. **G1** — refuse F-order in `AsDenseTensor` + a real-session test + docs. Small; removes a silent wrong answer.
2. **G2(a)** — clear > 2 GB pre-check + correct the three texts; then decide on G2(b).
3. **G3** — the keeper concept: `NDArrayIoBinding`, `AddInitializer(name, NDArray)`, `ToOrtSequence`.
4. **G4 + G5 + G11** — one disposable, all-value-kinds `Run` result + string-keyed maps: unlocks scikit-learn models.
5. **G7 + G6** — optional / overridable / sparse-refusal in `PrepareInput`; mixed input kinds.
6. **G13** — bump CI current to 1.30.0, fix counts and banners (can ride with any of the above).
7. **G9, G10, G12, G8** — double copy, memory allowlist, BFloat16 bridge, `RunAsync` (gated ≥ 1.30.0).

**Adjacent, out of scope for this package:** ORT Training (last stable `Microsoft.ML.OnnxRuntime.Training` is
1.19.2; its API compiles only under `__ENABLE_TRAINING_APIS__`); ORT GenAI (`Microsoft.ML.OnnxRuntimeGenAI`
0.16.0 — own `Tensor` type, not in `refs/`, not evaluated here); image decode/resize for vision preprocessing;
embedding-model post-processing (`MeanPool` + `L2Normalize`) as further `Postprocess` compositions — NMS stays
documented-only per plan decision 7.

## 6. Reproduction

**Models** — built with `onnx` 1.22 the same way as `test/oracle/gen_onnx_models.py` (opset 17, IR 8; ai.onnx.ml
opset 2 where noted):

| Model | Graph |
|---|---|
| `optional_input` | `X: optional(tensor(float))` → `OptionalGetElement` → `Y` |
| `optional_some` / `optional_none` | `Optional(X)` / `Optional(type=tensor(float))` → `Y: optional(tensor(float))` |
| `zipmap_string` | `X[N,3]` → `ZipMap(classlabels_strings=[cat,dog,bird])` → `sequence(map(string,float))` (ml) |
| `classifier_int` | `ArgMax(axis=1)` → `label` int64 ; `ZipMap(classlabels_int64s)` → `probabilities` (ml) |
| `classifier_str` | `ArgMax` → `LabelEncoder(int64→string)` → `label` string ; `ZipMap(strings)` → `probabilities` (ml) |
| `mixed_string_float` | inputs `text: string[N]`, `x: float[N]` → `y = Identity(x)` |
| `overridable_init` | `y = x + w`, `w` an initializer also listed as a graph input (checker-valid) |
| `float8_out` / `bf16_out` | `Cast(FLOAT8E4M3FN)` (opset 19, IR 9) / `Cast(BFLOAT16)` |

**Probe skeleton** — a file-based app outside the repo root (see CLAUDE.md "Scripting with `dotnet run`"):

```csharp
#:project K:/source/NumSharp/src/NumSharp.Interop.OnnxRuntime
#:package Microsoft.ML.OnnxRuntime@1.29.0          // or @1.16.0 for the floor
#:property AssemblyName=NumSharp.DotNetRunScript   // the package grants it InternalsVisibleTo
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

Key repros (each is a few lines on top of that header):

```csharp
// G1 / P1b — silent transpose through OrtTensor<T>.Memory
using NDArray c = np.arange(12).astype(NPTypeCode.Int32).reshape(3, 4), f = np.asfortranarray(c);
using OrtTensor<int> h = f.AsDenseTensor<int>();
using OrtValue v = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, h.Memory, new long[] { 3, 4 });
// run identity_int32.onnx on v: row 0 comes back [0,4,8,1]; NamedOnnxValue.CreateFromTensor("X", h.Tensor) throws.

// G2 / P2 — 2.2 GB import
using var nd = new NDArray(NPTypeCode.Byte, new Shape(2_200_000_000L), fillZeros: false);
using OrtTensor t = nd.AsOrtValue();         // fine
t.Value.AsNDArray();                         // ArgumentOutOfRangeException (inside ORT's (int) cast)

// G3 / P3 — IoBinding outlives the pin
var b = session.CreateIoBinding(); var h2 = x.AsOrtValue();
b.BindInput("X", h2.Value); b.BindOutputToDevice("Y", OrtMemoryInfo.DefaultInstance);
h2.Dispose();                                // LiveExports == 0 — nothing pins x any more
x[0] = 42f; session.RunWithBinding(new RunOptions(), b);   // output reads 42 from x's memory
```
