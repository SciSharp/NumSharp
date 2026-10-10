# `NumSharp.Interop.OnnxRuntime`: rework plan after the 2026-10-09 review

**Written:** 2026-10-10 · **Branch:** `strings` (the work commits directly on it; no rebase onto `journey4` now) ·
**Review base:** `journey4` `0e88dd53` (the bridge files are identical on `strings` `cd9c53e6`) ·
**Package state:** unreleased, so **no backward compatibility is kept** (owner's decision).

This plan holds the owner's decision for every item of the review, the design that follows from the decisions,
the work packages, and the items that are deferred with the owner's design notes. It replaces the recommended
order of `onnxruntime-gap-review.md` (2026-09-23), whose gap ids (G1–G13) appear here in parentheses.

## 1. Evidence the review produced

| Check | Result |
|---|---|
| Suite at the review base | 209 / 209 on net8.0 and net10.0 with ORT 1.16.0 (the floor); 209 / 209 with 1.29.0 and 1.31.0 |
| Coverage (coverlet, Debug, net10.0) | 89.2 % lines (1,027 / 1,151), 83.3 % branches (490 / 588) |
| Mutation sample | 24 planted bugs, 20 caught, 4 survive (T1 below) |
| Probes | about 150 checks in real sessions on ORT 1.31.0, 13 scratch models |

How to repeat it without the live tree: `git archive HEAD src tools test/NumSharp.Tests.Interop.OnnxRuntime
test/NumSharp.Tests/Utilities/GcQuiescence.cs Directory.Build.props Open.snk LICENSE global.json .editorconfig
docs/website-src/images/numsharp.icon128.png | tar -x -C <scratch>` builds standalone. Coverage: add
`coverlet.collector` to the snapshot's test project and collect with `<Include>[NumSharp.Interop.OnnxRuntime]*</Include>`.
A mutant that removes a `GC.KeepAlive` kills the test host: the run then reports "Failed: 0" over fewer tests,
which is a kill, not a survival.

## 2. Decisions

Status: `todo`, `done` (with the commit), `deferred`, `closed` (no change by decision).

### 2.1 Validity

| ID | Item | Decision | Status |
|---|---|---|---|
| V1 (G1) | Fortran-order `AsDenseTensor<T>`: ORT refuses it through `NamedOnnxValue` and reads it transposed through `Memory` | The export verbs and `Run` get the three modes of the pythonnet codec as `OnnxCodecMode` (`Auto`, `View`, `Copy`). `Auto` shares when ORT can read the buffer and copies otherwise. The `reverseStride` share is removed. Design in §3.1. | done (WP 2 export verbs, WP 4 `Run`) |
| V2 | `Postprocess.Softmax` / `LogSoftmax` / `Sigmoid` wrap for integer input (`Softmax(uint8 [1,3])` = `[NaN, 0]`) | Promote bool and integer input to NumPy's exp tier before the computation (bool, int8, uint8 → float16; int16, uint16, char → float32; wider → float64). Result dtypes do not change. Tests check values. | done (WP 1) |
| V2b | `LogSoftmax([inf, 1])` = `[NaN, NaN]`; SciPy gives `[nan, -inf]` | Follow `scipy.special.log_softmax`: a non-finite row maximum is replaced by 0 before the shift. | done (WP 1) |
| V3 | `nd.AsOrtValue().Value.AsNDArray()` reads another array's memory after the handle and the source are disposed | A view of an `OrtValue` that the package made is a counted NumSharp alias of the buffer (as `Tensor.AsNDArray()` already is). | done (WP 3) |
| V4 | The read-only flag is lost through `Value.AsNDArray()` and `Tensor.AsNDArray()` | Those views inherit the source's read-only flag. A read-only array stays a legal model input; the documents say that ORT has no read-only tensors. | done (WP 3; documents in WP 9) |
| V5 | A repeated output name loses one array until a collection | Accepted as it is. | closed |
| V6 | `casting` is checked only when a cast is necessary; `"no"` still converts Decimal | `Run` always validates the string. Decimal and Char follow NumSharp's `can_cast` table: the default `"safe"` refuses a Decimal input (`"same_kind"` permits it); Char → UInt16 stays permitted under `"safe"` and stays zero-copy. | done (WP 4) |
| V7 | `TryToTensorElementType` / `ToTensorElementClrType` throw for a datetime64 / timedelta64 descriptor | The `Try` form returns `false`, `ToTensorElementClrType` returns `null`, `ToTensorElementType` throws the package's own message. A non-native descriptor keeps mapping as native. | done (WP 5) |
| V8a | A sparse-tensor model input passes the checks of `Run` and fails inside ORT | Deferred. | deferred |
| V8b (G7) | `Run` refuses an `optional(tensor)` input with a wrong hint | `Run` unwraps the declaration and feeds a plain tensor. | done (WP 4) |
| V9 | A pre-allocated output that overlaps an input gives wrong values (MatMul 64×64: 3,072 of 4,096 elements) | `Auto`: `RunInto` feeds a copy of the overlapping input, as NumPy does for `out=`. `View`: it throws. Two outputs in one array always throw. | done (WP 4) |
| V10a | `AsNDArray(value, ownsValue: true)` can dispose `value` when it throws | The code follows the docs: a throw never disposes `value`. | done (WP 3) |
| V10b | Every view adds full-size GC memory pressure | Only a view that owns its `OrtValue` adds pressure, once. A copy needs none from the bridge (Core's allocator reports it). | done (WP 3) |
| V11a | `session.Run(inputs, null)` is CS0121 | The pre-allocated-outputs overload becomes `RunInto`. | done (WP 4) |
| V11b | `NamedOnnxValue.ToNDArray` refuses `DenseTensor<char>` and `DenseTensor<Half>` | Accept both. | done (WP 3) |
| V11c | `Postprocess` exists 2 times (ONNX bridge, ML.NET bridge) | One source file compiled into both bridges; the 2 public names stay; a caller with both bridges uses a `using` alias. | done (WP 1) |
| V11d | `Postprocess.Argmax(x2d, axis: -3)` returns a result | `Postprocess.Argmax` checks its axis, as `TopK` does. | done (WP 1) |
| V11e | Unused internal members | Kept: each one was checked and is correct. | closed |
| V12 (new, found by the probes for WP 7) | A model output that no node produces and that the caller feeds (an output that is a model input, or a fed overridable initializer). `RunInto` leaves the pre-allocated array unwritten (`[-777, -777, -777]` stays), or gives it the initializer's DEFAULT while the run uses the fed value; no error. `Run(zeroCopyOutputs: true)` returns a view of the fed memory that holds no reference on it: with an input fed as a temporary, the view read `[555, 555, 555]` from the next array that took the temporary's block. | Not an owner decision: a defect found during execution, fixed by the precedents V9 and V3. `RunInto` writes the fed values into the array after the run and checks the array's shape before it. The zero-copy form returns a counted NumSharp alias of the fed buffer; the copy form copies from the fed buffer. The criterion is the output's name among the fed names, which is ORT's own rule. Design in §3.7. The low-level `AsNDArray(OrtValue)` stays open (§5.1). | done (WP 10) |
| V13 (new, found by a test of WP 8) | An import verb given the value of a DISPOSED export handle (`v = h.Value; h.Dispose(); v.AsNDArray()`) ended the process: the handle forgets its value on dispose, so the verbs took the value for one of ONNX Runtime's own and called into it, and ORT does not guard a released value (access violation `0xC0000005` in `OrtValue.GetTensorTypeAndShape`). | Not an owner decision: a defect found during execution. The handle records its value as disposed (a weak table that keeps nothing alive), and the 7 verbs that take an `OrtValue` throw `ObjectDisposedException` for it before the first native call. A value of ONNX Runtime's own that the caller disposed cannot be recognized through the public API. | done (WP 8) |

### 2.2 Completeness

| ID | Item | Decision | Status |
|---|---|---|---|
| C1 (G3) | No owner for the pins of `OrtIoBinding`, `AddInitializer`, `CreateSequence` | Deferred. Owner's note: consider wrapping `Run` (overloads) so that the wrapper holds the references. | deferred |
| C2a (G4, G11) | `Run` cannot return non-tensor outputs; the dictionary result is not disposable | Deferred. Owner's note: consider overloads and typed struct support. Text outputs are covered by S1. | deferred |
| C2b (G5) | String-keyed maps are refused | Covered by S1: the keys come back as a text `NDArray`. | done (WP 6) |
| C2c (G6) | The `Run` tier accepts only `NDArray` inputs | Deferred. Owner's note: a `Run<T>(T inputs) where T : struct` with an interface that yields the `Dictionary<string, OnnxInput>` from the struct values; the default implementation is reflection with statically cached getters (expression trees or IL generation). Text inputs are covered by S1. | deferred |
| C2d (G7) | `Run` refuses an overridable initializer | Accept: look a name up in `InputMetadata`, then in `OverridableInitializerMetadata`. | done (WP 4) |
| C3a (G8) | No `RunAsync` | Add it on ORT's own native async, for ORT 1.30.0 and newer only (older versions do not pin their argument arrays). Design in §3.6. | done (WP 7) |
| C3b (G9) | The non-tensor readers copy each element 2 times | Each result becomes a view that owns ORT's copy. | done (WP 3) |
| C3c (G10) | `IsCpuAccessible` refuses `QnnHtpShared` and `CpuAligned4K` | Ask ORT for the device memory kind when the loaded ORT has `GetDeviceMemoryType`; the name list stays as the fallback. | done (WP 3) |
| C3e (G12) | BFloat16 is refused | Kept. | closed |
| S1 (new) | Text crossing, possible because `strings` gives `NDArray` a text dtype | Build all of it now on the fixed-width `S` / `U` dtypes, as a copy: `ToNDArray`, the map readers, `ToOrtValue`, `AsOrtValue` and `Run` take and return text. Leave a prominent TODO for the variable-width `StringDType` (phase 3 of the string plan). | done (WP 6) |

### 2.3 Tests, documents, CI

| ID | Item | Decision | Status |
|---|---|---|---|
| T1 | 4 planted bugs pass the suite | Add all 4 tests: a `WeakReference` test for the rooting, an end-to-end device-memory refusal, and 2 disposal tests that read ORT's private `_handle` by reflection (Inconclusive if a future ORT renames it). | done (WP 3: device memory; WP 8: the other 3) |
| T2 | Reachable behaviour without a test | Add a test for each listed path; `gen_onnx_models.py` gets the models they need. | done (WP 8) |
| D1 | Stale test counts in the README and the docs page | Remove the counts there. `CLAUDE.md` keeps the one number. | todo |
| D2 (G13) | "Current" ORT is 1.29.0 in CI and the documents; NuGet has 1.31.0 | 1.31.0 everywhere: CI, README, docs page, `CLAUDE.md`, and the `refs/onnxruntime` submodule at `v1.31.0`. | todo |
| D3 (G13) | The csproj comment promises `FileLoadException` for an older ORT | Correct it: the 1.16.0 assembly has AssemblyVersion 0.0.0.0, so only the NuGet range sets the floor. | todo |
| D4 | 9 XML-doc warnings that CI hides | Skipped as an item. Doc comments that this work rewrites are written correctly. | closed |
| D5 | Public methods without `<param>` / `<returns>` text | Skipped as an item. Code that this work writes or edits is documented in full (the standing rule). | closed |

## 3. Design

### 3.1 `OnnxCodecMode` (V1)

```csharp
public enum OnnxCodecMode
{
    Auto = 0,   // a zero-copy view when the dtype and the layout permit it, otherwise an independent copy
    View = 1,   // always a zero-copy view; a crossing that needs a copy throws
    Copy = 2,   // always an independent copy
}
```

Same member names and meanings as pythonnet's `NumpyCodecMode`. One enum serves both directions.

**A view is possible** when ORT can read the caller's buffer as the tensor: the dtype crosses bit for bit (the 12
numeric dtypes and Char → UInt16) and the array is C-contiguous. A read-only array qualifies.

| Source | `Auto` | `View` | `Copy` |
|---|---|---|---|
| C-contiguous, bit-exact dtype | view | view | copy |
| not C-contiguous (Fortran, stepped, reversed, transposed, broadcast) | C-order copy | throws | copy |
| Decimal (no ONNX type) | float64 copy | throws | float64 copy |
| text (`S` / `U`, see §3.5) | ORT string tensor (a copy) | throws | ORT string tensor |
| Complex | throws in every mode (no ONNX type; a copy cannot repair it) | | |
| more elements than a `DenseTensor` holds (`AsDenseTensor` only) | throws in every mode | | |
| released buffer | `ObjectDisposedException` in every mode | | |

Entry points that take `mode` (default `Auto`):

```csharp
OrtTensor      AsOrtValue(this NDArray source, OnnxCodecMode mode = OnnxCodecMode.Auto)
OrtTensor<T>   AsDenseTensor<T>(this NDArray source, OnnxCodecMode mode = OnnxCodecMode.Auto)
NDArray        AsNDArray(this OrtValue value, bool ownsValue = false, OnnxCodecMode mode = OnnxCodecMode.Auto)
NDArray        AsNDArray<T>(this DenseTensor<T> tensor, OnnxCodecMode mode = OnnxCodecMode.Auto)
session.Run(…, OnnxCodecMode mode = OnnxCodecMode.Auto)        // every Run / RunInto / RunAsync form: the INPUT mode
```

- A copy that an export verb makes lives in a temporary that the handle owns (the existing Decimal mechanism).
  ORT's writes then land in the temporary. `OrtTensor.IsView` / `OrtTensor<T>.IsView` say which case a handle is.
- `ToOrtValue` and `ToDenseTensor<T>` stay: the plain copy verbs that return a bare value with no handle.
- The `reverseStride` export is removed: no ORT route consumes it. The import of a column-major tensor that a
  caller built stays.
- Import verbs: a copy helps only for a text tensor (`Auto` copies it, `View` throws). Device memory, tensors of
  2 GB or more and non-tensor values stay refused in every mode.
- `Run` inputs: `Auto` is today's behaviour. `View` throws when an input needs a cast, a conversion or a C-order
  copy, so a hot loop can prove that it copies nothing. `Copy` feeds a copy of every input.

### 3.2 The `Run` tier (V6, V8b, V9, V11a, C2d)

```csharp
NDArray Run(this InferenceSession session, NDArray input, RunOptions runOptions = null,
            string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
NDArray Run(this InferenceSession session, string inputName, NDArray input, string outputName,
            RunOptions runOptions = null, string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
IReadOnlyDictionary<string, NDArray> Run(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyCollection<string> outputNames = null, RunOptions runOptions = null, string casting = "safe",
            bool zeroCopyOutputs = false, OnnxCodecMode mode = OnnxCodecMode.Auto)
void RunInto(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
            IReadOnlyDictionary<string, NDArray> outputs, RunOptions runOptions = null, string casting = "safe",
            OnnxCodecMode mode = OnnxCodecMode.Auto)
```

- `casting` is validated first, on every call, with NumSharp's own `ValueError` text.
- An input name is looked up in `InputMetadata`, then in `OverridableInitializerMetadata` (C2d).
- An `optional(tensor)` declaration is unwrapped to its element declaration; dtype and shape checks use it (V8b).
- Dtype rule: equal dtype → no cast. Otherwise `np.can_cast(have, want, casting)` decides. Char → UInt16 is a
  zero-copy reinterpretation when the rule permits it; every other conversion is a copy, which `View` refuses.
- `RunInto` overlap rule (V9): the memory range of each output is compared with every input and every other
  output. Output against output always throws. Output against input: `Auto` and `Copy` feed a copy of that input;
  `View` throws.

### 3.3 Import side (V3, V4, V10a, V10b, V11b, C3b, C3c)

- `OrtTensor.ValueOwners` already maps each package-made `OrtValue` to its handle. `AsNDArray(value)` uses it: the
  result is a counted alias of the handle's buffer (`ArraySlice.WrapShared`), read-only when the source is.
  With `ownsValue: true` the handle is disposed once the alias exists, because the alias keeps the buffer.
- `UnmanagedMemoryManager<T>` records the source's writeable flag, so `Tensor.AsNDArray()` can apply it.
- A foreign `OrtValue` keeps the lease route. The release action disposes the value only after the view was
  returned (V10a). Memory pressure is added only for an owning lease (V10b).
- The non-tensor readers return `element.AsNDArray(ownsValue: true)` (C3b). Their results report
  `owndata == False` and count as live imports.
- `IsCpuAccessible` asks ORT through a cached delegate over `OrtMemoryInfo.GetDeviceMemoryType()` when that member
  exists (ORT 1.2x and newer; the package compiles against 1.16.0, so the member is found by reflection once):
  `HOST_ACCESSIBLE` is readable. Otherwise the existing name and memory-type checks decide.

### 3.4 `Postprocess` (V2, V2b, V11c, V11d)

- One source file, `src/NumSharp.Interop.Shared/Postprocess.cs`, compiled into both bridges. The namespace is
  chosen by a compilation symbol that each project defines.
- `Softmax`, `LogSoftmax`, `Sigmoid`: a bool or integer input is cast to its float tier first, then the same
  formulas run. `LogSoftmax` replaces a non-finite maximum by 0.
- `Argmax` validates its axis against `[-ndim, ndim)`.

### 3.5 Text crossing (S1)

Everything here is a copy: ORT keeps each string as a separate native object, so no layout can share it.

- dtype maps: `NPTypeCode.String` ↔ `TensorElementType.String`.
- Import: `ToNDArray` reads a string tensor into a fixed-width `U` array (width = the longest string, as
  `np.array(list_of_str)`), with the tensor's shape. `ReadStringTensor` stays for callers that want `string[]`.
  The map readers return string keys (and string values) as text arrays. `ToNDArrays` reads a sequence of strings.
- Export: `ToOrtValue` makes a string tensor from a `U` or `S` array (`S` bytes are passed as UTF-8, which is what
  Python `onnxruntime` does). `AsOrtValue` does the same under `Auto` and `Copy` and throws under `View`.
  `AsDenseTensor<T>` refuses text (its `T` is unmanaged).
- `Run`: a text array feeds a string model input; text outputs come back as text arrays. With zero-copy outputs a
  text output is a copy. `mode: View` refuses a text input, because it cannot be a view.
- Text and numbers do not convert: NumSharp cannot cast between a string dtype and a numeric one yet. A numeric
  array for a string input, and a text array for a numeric input, are refused under every casting rule, with a
  message that names the two dtypes. This is a second TODO in the code.
- `RunInto` refuses a text output array: ORT cannot write its strings into a fixed-width buffer. `Run` returns
  text outputs.
- Handles: a text `OrtTensor` holds no NumSharp buffer (`IsView` is `false`). The import verbs therefore read it
  as a string tensor and never alias it.
- **TODO (prominent, in the code):** when NumSharp gets the variable-width `StringDType` (NEP 55, phase 3 of
  `strings-phase2.md`), map ORT strings to it. The fixed-width form pads with NUL, so a string that ends in NUL
  characters loses them, and one long string widens every element.

### 3.6 `RunAsync` (C3a)

ORT's C# `RunAsync` schedules the run on ORT's intra-op thread pool and completes a task from a native callback.
It is a real async: no .NET thread waits. Three limits come from ORT:

1. It reads `.Handle` of every output value, so it needs outputs that already exist.
2. It is memory-safe only from 1.30.0 (commit `8239590229`, #32015); 1.29.0 does not pin its argument arrays.
3. The intra-op pool needs 2 or more threads (`IntraOpNumThreads = 1` makes ORT refuse the call).

```csharp
Task RunIntoAsync(this InferenceSession session, IReadOnlyDictionary<string, NDArray> inputs,
                  IReadOnlyDictionary<string, NDArray> outputs, RunOptions runOptions = null,
                  string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
Task<IReadOnlyDictionary<string, NDArray>> RunAsync(this InferenceSession session,
                  IReadOnlyDictionary<string, NDArray> inputs, IReadOnlyCollection<string> outputNames = null,
                  RunOptions runOptions = null, string casting = "safe", OnnxCodecMode mode = OnnxCodecMode.Auto)
```

- `RunIntoAsync` is the real async for every model. The bridge holds the input and output handles until the task
  completes, then disposes them.
- `RunAsync` allocates each output from the model's declared shape, so it works only when every fetched output
  declares a fixed shape and a numeric dtype; otherwise it throws `NotSupportedException` that names `RunIntoAsync`.
- On an ORT older than 1.30.0 both throw `NotSupportedException` with the reason. The loaded version is read
  once from the managed assembly (1.16.0 reports 0.0.0.0, which counts as older).

As built (WP 7), with what the probes on ORT 1.31.0 added:

- `InferenceSessionExtensions.IsRunAsyncSupported` tells a caller which case applies. On an older ORT both forms
  throw from the call, before every check of the arguments but the null checks.
- ORT completes its task on its own intra-op thread, and `session.Dispose()` called there never returns
  (measured). So the bridge's task runs a caller's continuation off that thread
  (`TaskCreationOptions.RunContinuationsAsynchronously`). The handles are released first, on ORT's thread, and
  then the task completes; a thread that blocks on the task is released with no thread-pool hop.
- ORT awaits without `ConfigureAwait(false)`: it captures the caller's synchronization context, or else the
  caller's task scheduler. A thread that owned a context and blocked on the task waited forever (measured:
  `Wait(2 s)` false, then true once the context was pumped). The bridge hides the context for the call, and makes
  the call from the default scheduler when the caller runs in a task of another one.
- Argument errors throw from the call. ORT's errors fault the task: a wrong output shape, `RunOptions.Terminate`
  (the way to cancel), a session with 1 intra-op thread.
- `RunAsync` refuses, before it reads an input: a dimension fixed only at run time, a declaration without
  dimensions (a scalar or an unknown rank: the metadata reports both as `dims=[]`, measured, so the bridge does
  not guess), a string tensor, a sequence or a map. An output that is a fed input is allocated from the fed
  tensor. A name asked 2 times is fetched once. A failed run disposes the arrays it allocated.
- The overlap rule (V9) and the pass-through rule (V12) hold for the async forms: they use the same code.
- While a run is in flight, ORT reads the shared inputs and writes the outputs. The caller must not write an
  input before the task completes; `OnnxCodecMode.Copy` feeds a snapshot. To DISPOSE an array in flight is safe:
  the run holds a reference on every buffer.

### 3.7 Outputs that are fed inputs (V12)

ONNX permits a graph output that no node produces. ONNX Runtime's execution frame fills its value slots with the
fetch buffers, then with the initializers, then with the feeds, each by name. So a feed replaces what the slot of
the same name held. Measured on ORT 1.16.0 and 1.31.0:

| Output | Pre-allocated buffer | Value ORT returns when it allocates |
|---|---|---|
| A node's output | written | ORT's own buffer |
| An initializer that is not fed | written (ORT copies the initializer) | ORT's own copy (a write through it does not reach the weight) |
| A model input, or an overridable initializer that IS fed | NOT written with the fed value | a value at the FED tensor's address |

The bridge therefore treats an output whose name is among the fed names (`FedTensor`) apart:

- `RunInto` (and `RunIntoAsync`): before the run it checks that the array has the fed input's shape (ORT checks
  nothing for this buffer); after a successful run it copies the fed handle's bytes into the array.
- `Run`, copy form: the output is copied from the fed handle's NumSharp memory (no span, so any size).
- `Run`, zero-copy form: the output is a counted NumSharp alias of the fed buffer (`AliasExportedValue`). It is a
  view of the caller's input array when the input was shared, and of the temporary when the input needed a cast or
  a C-order copy; the temporary then lives as long as the view. It is read-only when the fed array was.
- A fed TEXT input is not this case: its string tensor is ORT's own object, and the ordinary text copy reads it.

## 4. Work packages

Each package is one commit on `strings`, limited to the bridge files, with its tests. Gate for every package: the
ONNX suite on net10.0 at the floor; at milestones also net8.0 and ORT 1.31.0, and the ML.NET suite when the shared
`Postprocess` changes.

| WP | Content | Items | Status |
|---|---|---|---|
| 0 | This plan | | done |
| 1 | `Postprocess`: shared source, float promotion, SciPy `log_softmax`, axis check | V2, V2b, V11c, V11d | done |
| 2 | `OnnxCodecMode` on the export verbs; `IsView`; `reverseStride` export removed | V1 | done |
| 3 | Import side: counted alias, read-only, ownership on failure, memory pressure, legacy types, readers, device kind | V3, V4, V10a, V10b, V11b, C3b, C3c | done |
| 4 | `Run` tier: `mode`, `RunInto`, casting rule, optional inputs, overridable initializers, overlap | V1, V6, V8b, V9, V11a, C2d | done |
| 5 | dtype maps over descriptors without storage | V7 | done |
| 6 | Text crossing | S1, C2b | done |
| 7 | `RunIntoAsync` / `RunAsync` | C3a | done |
| 8 | Remaining tests and models; the disposed-handle guard they found | T1, T2, V13 | done |
| 9 | Documents and CI: README, docs page, `CLAUDE.md`, csproj comment, ORT 1.31.0, API approval list | D1, D2, D3 | todo |
| 10 | Outputs that are fed inputs. Run before WP 7, whose forms share the code. | V12 | done |

## 5. Deferred, with the owner's design notes

- **V8a, sparse tensors.** ORT's C# API has no method that creates or reads one (only `IsSparseTensor` and the raw
  native function table). A clear refusal up front was the alternative on the table.
- **C1, owners for bindings, initializers and sequences.** Note: consider wrapping `Run` (overloads) so that the
  wrapper is where the references are maintained.
- **C2a, non-tensor outputs and a disposable result.** Note: consider overloads and the typed struct support of C2c.
- **C2c, inputs that are not `NDArray`.** Note: a `Run<T>(T inputs) where T : struct`; an interface that yields the
  `Dictionary<string, OnnxInput>` from the struct's values; the default implementation is a reflection-built getter,
  cached statically per `T`, made with expression trees or IL generation.
- **Rebase of `strings` onto `journey4`.** Not now. A trial merge on 2026-10-10 showed conflicts in 13 files (4 in
  Core: `np.asarray`, `np.asmatrix`, `np.all`, `np.any`; 6 oracle test files; the coverage map and its test;
  `CLAUDE.md`; 1 matrix test). The bridge files are identical on the two branches.

### 5.1 Open items found during execution (not decided by the owner)

- **V12b, a pass-through output through the low-level verb.** A caller who runs the session with ORT's own
  `session.Run` and feeds an `AsOrtValue` handle gets, for an output that is a fed input, a value over the handle's
  NumSharp memory. `value.AsNDArray(ownsValue: true)` then returns a view that holds no reference on that memory:
  it dangles when the handle is disposed and the handle owned a temporary. The `Run` tier is fixed because it knows
  what it fed. The low-level verb would need a registry of the live export handles by address (a lock, several
  handles for one address, entries that do not root their handle). The remarks of `AsNDArray` state the limit.
  Recommended: build the registry when the low-level tier gets a caller that returns owning views.
- **ORT's `RunAsync` on a session whose task continues on ORT's thread.** Measured on 1.31.0: ORT completes its
  task on its intra-op thread, and `session.Dispose()` called there never returns. The bridge's async forms
  complete their task off that thread (§3.6). A caller of ORT's own `RunAsync` must do the same.

## 6. Execution log

| Date | WP | Commit | Result |
|---|---|---|---|
| 2026-10-10 | 0 | `32d47557` | plan written |
| 2026-10-10 | 1 | `720d81cb` | shared `Postprocess`; ONNX suite 212 / 212 on net10.0 and net8.0 (ORT 1.16.0), ML.NET post-processing and API-surface classes 24 / 24 on both |
| 2026-10-10 | 2 | `40d546aa` | `OnnxCodecMode` on `AsOrtValue` / `AsDenseTensor<T>`, `IsView`, row-major tensors only; ONNX suite 216 / 216 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0 |
| 2026-10-10 | 3 | `eade99a5` | import side: counted alias, read-only inheritance, ownership on failure, memory pressure, legacy `char` / `Half` tensors, owning reader results, device kind from ORT; ONNX suite 232 / 232 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0; mutation sample 16 of 18 (1 equivalent, 1 closed by a new test) |
| 2026-10-10 | 4 | `3139d111` | `Run` tier: input mode on every form, `RunInto`, the cast table for every dtype pair, optional inputs, overridable initializers, the overlap rule; ONNX suite 241 / 241 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0; mutation sample 18 of 18 |
| 2026-10-10 | 5 | `09248ea1` | dtype maps answer false / null for a datetime64 / timedelta64 descriptor; ONNX suite 242 / 242 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0 |
| 2026-10-10 | 6 | `17231da9` | text crossing on the fixed-width `U` / `S` dtypes, as a copy: the dtype maps, `ToOrtValue` / `AsOrtValue`, `ToNDArray` / `AsNDArray`, the legacy `Tensor<string>`, the sequence and map readers, and `Run`; 3 new models (string labels, a string-keyed ZipMap, text next to numbers); ONNX suite 257 / 257 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0; mutation sample 21 of 21 |
| 2026-10-10 | 10 | `944817d4` | outputs that are fed inputs: `RunInto` writes them, the zero-copy form returns a counted alias of the fed buffer, the copy form copies from it; 3 new models (`passthrough`, `overridable_out`, `passthrough_text`); ONNX suite 266 / 266 in Release on net10.0 and net8.0 (ORT 1.16.0) and on net10.0 with ORT 1.31.0; mutation sample 6 of 9 (2 equivalent: a zero-byte copy, and the copy form below 2 GB; 1 not observable from outside: the early dispose of the value ORT returned) |
| 2026-10-10 | 7 | `da9b1be8` | `RunIntoAsync`, `RunAsync`, `IsRunAsyncSupported` on ORT's native asynchronous run; 1 new model (`cast_fanout`); ONNX suite 286 / 286 in Release on net10.0 and net8.0 with ORT 1.31.0, and 267 passed + 19 Inconclusive (the async behaviour tests) on both with ORT 1.16.0, where the refusal is tested; mutation sample 18 of 18 (run with ORT 1.31.0), the three deadlock guards included |
| 2026-10-10 | 8 | (this commit) | 21 tests for the review's 3 surviving planted bugs and for the reachable paths without a test (the list came from a new coverage run); 6 new models (`tensor_then_seq`, `anon_dim`, `bf16_out`, `bf16_in`, `seq_two`, `two_dtypes`); the disposed-handle guard (V13); ONNX suite 307 / 307 in Release on net10.0 and net8.0 with ORT 1.31.0, and 288 passed + 19 Inconclusive on both with ORT 1.16.0; mutation sample 22 of 22 (the review's M07, M09 and M23 among them; 3 mutants of the guard end the test host, which is the crash the guard prevents) |
