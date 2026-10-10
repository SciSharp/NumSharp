# NumSharp.Interop.OnnxRuntime

Zero-copy interop between NumSharp `NDArray` and [ONNX Runtime](https://onnxruntime.ai) tensors. Feed an
`NDArray` to `InferenceSession.Run` as an `OrtValue` built **over the very same unmanaged buffer** — no
`DenseTensor` fill loop, no copy — read outputs back as `NDArray`s (an owning copy, or a zero-copy view),
and do the pre-/post-processing (normalize, transpose, softmax, argmax, top-k, clip) as vectorized NumSharp
ops instead of the per-pixel `for` loops every ORT C# tutorial hand-writes.

```csharp
using NumSharp;
using NumSharp.Interop.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

using var session = new InferenceSession("resnet50-v2-7.onnx");

NDArray input = /* (1,3,224,224) float32, C-contiguous */;
using NDArray logits = session.Run(input);                 // zero-copy in, owning NDArray out
using NDArray probs  = Postprocess.Softmax(logits);        // (1,1000)
var (top5, classes) = Postprocess.TopK(probs, 5);          // values + int64 indices
```

This is a **conversion library**, in the mould of `NumSharp.Interop.pythonnet`, not an engine backend: ONNX
Runtime does not compute NumSharp operations, it consumes tensors NumSharp produces and produces tensors
NumSharp reads. So there is no `TensorEngine` seam, no `[ModuleInitializer]`, and referencing the package
changes nothing until a verb is called.

> Verified against ONNX Runtime **1.16.0** (the package floor) and **1.31.0** (current) on net8.0 / net10.0,
> by `test/NumSharp.Tests.Interop.OnnxRuntime`: committed `.onnx` models that ONNX Runtime really executes,
> with an every-dtype × every-layout matrix and property-based, metamorphic, bit-pattern-fuzz, concurrency,
> boundary, reference-accounting, lifetime and API-approval suites. The package is unreleased, so its surface
> is not frozen.

## Install

```bash
dotnet add package NumSharp.Interop.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime          # or .Gpu / .DirectML / .QNN — exactly ONE native flavour
```

The package depends only on **`Microsoft.ML.OnnxRuntime.Managed`** (the C# API). ORT ships its native
runtime as mutually exclusive flavours — `Microsoft.ML.OnnxRuntime` (CPU), `.Gpu`, `.DirectML`, ... — and
two of them side by side is a documented misconfiguration, so the flavour is yours to pick: the one you
already reference to run inference at all. The dependency range is `[1.16.0, 2.0.0)`; a consumer's own ORT
reference (say 1.31.0) carries the same `.Managed` version and NuGet unifies upward to it. The NuGet range is
the only thing that sets the floor: the 1.16.0 managed assembly reports assembly version 0.0.0.0, so the
runtime's binder refuses nothing.

## Views, copies and `OnnxCodecMode`

An ORT tensor is a pointer, a row-major shape and a fixed-size element type. It has **no strides**. So ORT can
read a NumSharp buffer in place only when the array is C-contiguous and its dtype crosses bit for bit (bool,
the eight integers, float16, float32, float64, and char as UInt16). Every entry point that could share
memory takes an `OnnxCodecMode`, with the member names and meanings of the pythonnet bridge's
`NumpyCodecMode`:

| Mode | What it does |
|---|---|
| `Auto` (default) | A zero-copy view when the dtype and the layout permit it, an independent copy otherwise. It never fails because of a layout or a convertible dtype. |
| `View` | Always a view. A crossing that needs a copy throws `InvalidOperationException` and names what prevents the view. Use it to prove that a hot loop copies nothing, or that ORT writes into your array. |
| `Copy` | Always an independent copy. |

A copy is needed for a Fortran-ordered, stepped, reversed, transposed or broadcast view (a C-order copy), for
a `Decimal` array (a float64 conversion) and for text. No mode repairs a dtype with no ONNX element type
(`Complex`), a released buffer, or a size the target cannot hold.

## The verbs

**`As…` gives a view when it can** (and a handle or an array that says which it gave), **`To…` always copies.**

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → ORT | `nd.AsOrtValue(mode)` → `OrtTensor` | An `OrtValue` over the NDArray's buffer (`CreateTensorValueWithData` on NumSharp's pointer) when a view is possible; otherwise, under `Auto`, an `OrtValue` over a temporary copy that the handle owns. `handle.IsView` says which, `handle.Source` is the array ORT reads. The handle owns the `OrtValue` and a counted reference on the buffer; keep it alive across `Run`. |
| NumSharp → ORT | `nd.ToOrtValue()` → `OrtValue` | An ORT-allocated **copy**, any layout (a view is copied in logical order); no lifetime coupling. |
| NumSharp → ORT | `nd.AsDenseTensor<T>(mode)` → `OrtTensor<T>` / `nd.ToDenseTensor<T>()` | The legacy `DenseTensor<T>` twins for `NamedOnnxValue.CreateFromTensor`. The view's `Buffer` is a `Memory<T>` over NumSharp's memory. The tensor is always row-major: a Fortran-ordered array is a copy under `Auto` (no ORT route reads a column-major tensor correctly). |
| ORT → NumSharp | `ortValue.ToNDArray()` → `NDArray` | A new **owning**, C-contiguous **copy** — the safe default; dispose the `OrtValue` whenever you like. A string tensor comes back as a text array. |
| ORT → NumSharp | `ortValue.AsNDArray(ownsValue = false, mode)` → `NDArray` | A **zero-copy view** over the tensor's buffer (CPU tensors only), mutations visible both ways. `ownsValue: true` makes the last view dispose the `OrtValue`. `mode: Copy` returns an independent copy. A string tensor cannot be viewed: `Auto` copies it, `View` throws. |
| ORT → NumSharp | `denseTensor.ToNDArray()` / `denseTensor.AsNDArray(mode)` / `namedOnnxValue.ToNDArray()` | The legacy-surface twins (a column-major tensor transposes back on copy and stays Fortran-ordered as a view; `NamedOnnxValue` dispatches on the element type at run time, `char`, `Half` and `string` tensors included). |
| ORT → NumSharp | `ortValue.ToNDArrays()` / `ToMap()` / `ToMaps()` / `GetMapKeys()` / `ReadStringTensor()` | **Non-tensor** outputs: a sequence of tensors → `NDArray[]`; a map, or a scikit-learn ZipMap `sequence(map(K, V))` → `(keys, values)` arrays; a string tensor → `string[]`. A numeric array is a view that **owns** ORT's copy of its element (one copy per element, made by ORT); a string part is a text array. |

Plus the dtype maps `ToTensorElementType` / `FromTensorElementType` (+ `Try…` forms, `ToTensorElementClrType`),
`IsCpuAccessible`, and two counters, `NDArrayOnnxInterop.LiveExports` / `LiveImports`, that make every live
crossing observable.

## The ergonomic tier — `session.Run(NDArray)`

`InferenceSessionExtensions` hides every `OrtValue` lifetime:

```csharp
NDArray y = session.Run(x);                                   // single input/output: names from metadata
NDArray y2 = session.Run("X", x, "Y2");                       // named
IReadOnlyDictionary<string, NDArray> outs = session.Run(      // several inputs, all (or some) outputs
    new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
var views = session.Run(inputs, zeroCopyOutputs: true);       // views that OWN their OrtValue
session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", preallocated } });   // ORT writes INTO your array
await session.RunIntoAsync(inputs, outputs);                  // the same, without blocking a thread (ORT 1.30+)
```

- **Inputs.** An input that is C-contiguous and of the declared dtype is shared. Any other is fed as a
  temporary under `Auto` (a cast, a C-order copy of a view), refused under `View`, and every input is copied
  under `Copy`. A dtype that differs from the declaration is **cast to it** when NumPy's `casting` rule allows
  it (`"safe"` by default — so an int32 array into an int64 BERT input or float32 into float64 just works; a
  lossy float64 → float16, and a `Decimal` array into a float64 input, need `casting: "same_kind"` or an
  `astype` at the call site). `Char` into a UInt16 input is safe and stays zero-copy. The rule name is checked
  on every call. Declared fixed extents and rank are checked **before** ORT sees the tensor, with a message
  naming the input (`input 'data' expects shape (1, 3, 4, 4), but the array has shape (1, 3, 4, 5): dimension 3
  must be 4, not 5`); symbolic dims (`'batch'`, `'seq'`) are free.
- **Names.** An input name is looked up among the model's inputs, then among its overridable initializers. An
  input declared `optional(tensor)` takes a plain array.
- **Outputs are copied** into owning arrays by default and every ORT value is released before the call
  returns. `zeroCopyOutputs: true` returns views that **own** their `OrtValue` — ORT's memory is freed when the
  array (and every slice derived from it) dies — for large outputs you consume in place.
- **`RunInto`** writes into arrays you allocated: zero-copy in *both* directions, the idiom for a hot loop that
  reuses its buffers (Stable Diffusion's UNet step). An output array must be C-contiguous, writeable and of the
  declared dtype (it cannot be cast). ORT may write an output while it still reads an input, so an output that
  shares memory with an input is computed from a copy of that input under `Auto` (NumPy's rule for `out=`) and
  refused under `View`; two outputs that overlap are always refused.
- **An output that is a fed input.** ONNX permits a model output that no node produces (a model input passed
  through, or an overridable initializer you feed). ORT does not write a pre-allocated array for it and returns
  a value over the fed memory. The tier handles it: `RunInto` writes the fed values into your array, the copy
  form copies them, and the zero-copy form returns a counted view of the fed buffer.
- **Text.** A text array (`U` or `S`) feeds a string input, and a string output comes back as a text array;
  see below.

### Asynchronous runs

`RunIntoAsync` and `RunAsync` build on ORT's own native asynchronous run: the model executes on the session's
intra-op thread pool and no .NET thread waits.

- They need ONNX Runtime **1.30.0** or newer (`InferenceSessionExtensions.IsRunAsyncSupported`). Older managed
  assemblies do not pin the arrays the native run keeps reading (microsoft/onnxruntime#32015), so both forms
  throw `NotSupportedException` there.
- The session needs **2 or more intra-op threads** (`SessionOptions.IntraOpNumThreads` is not 1). With one, ORT
  refuses and the task is faulted.
- `RunIntoAsync` is `RunInto` for every model. `RunAsync` allocates the outputs itself, which it can do only
  before the run: for a numeric tensor with a fixed declared shape, and for an output that is a fed input.
  For a dimension fixed only at run time, a declaration without dimensions, a string tensor or a non-tensor
  output it throws `NotSupportedException` with what to do instead.
- Usage errors throw from the call. A failure inside ORT faults the task. To cancel, pass a `RunOptions` and
  set `Terminate = true`.
- ORT completes its own task on an intra-op thread of the session. Code that continues there and disposes the
  session never returns. The bridge's task runs your continuation off that thread, and a thread that blocks on
  it under a synchronization context is released (ORT's own task waits forever there).
- Until the task completes, ORT reads the shared inputs and writes the outputs: do not write an input (pass
  `OnnxCodecMode.Copy` to feed a snapshot). Disposing an array in flight is safe; the run holds a reference.

## Text

NumSharp's fixed-width text dtypes (`U`: UCS-4 code points; `S`: bytes) cross as ORT string tensors, in both
directions and through every route (`ToOrtValue`, `AsOrtValue`, `ToNDArray`, `AsNDArray`, the readers, `Run`).

- **Everything is a copy.** ORT keeps each string as a separate object in its own memory. `Auto` and `Copy`
  do the same; `View` throws.
- A `U` element crosses as its text (ORT stores UTF-8). An `S` element crosses as its bytes, unchecked, as
  Python's `onnxruntime` does. A string tensor reads back as a `U` array as wide as its longest string.
- **The fixed width costs two things**, until NumSharp has the variable-width `StringDType`: a string that
  ENDS in NUL characters loses them (`ReadStringTensor` keeps them), and one long string widens every element.
- Text and numbers do not convert into each other yet: a numeric array for a string input, or a text array
  for a numeric input, is refused under every casting rule.
- `RunInto` and `RunAsync` cannot take a string output (ORT cannot write its strings into NumSharp memory);
  the synchronous `Run` returns it.

## Dtypes

| NumSharp | ORT `TensorElementType` | Crossing |
|---|---|---|
| Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double | Bool, UInt8, Int8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float, Double | **zero-copy** |
| Half | Float16 | **zero-copy reinterpret** — ORT's `Float16` is a blittable 16-bit IEEE struct, binary-identical to `System.Half`; NaN payloads and subnormals cross untouched |
| Char | UInt16 | zero-copy as UTF-16 code units; directional — an ORT UInt16 comes back as UInt16, not Char |
| String (`U`, `S`) | String | always a copy; see Text |
| Decimal | — | no ONNX type: the array-producing verbs convert to Double under `Auto` and `Copy` (a float64 temporary the handle owns; lossy beyond ~16 digits), `View` refuses; `Run` needs `casting: "same_kind"`; the low-level maps refuse it |
| Complex | Complex64/128 exist in the enum but **no ORT API or kernel accepts them** | refused — split into `np.real` / `np.imag` |
| — | BFloat16 | no NumSharp dtype — refused with the ORT-side alternative named |

The maps answer for every descriptor: `TryToTensorElementType` returns `false` and `ToTensorElementClrType`
returns `null` for a dtype without storage (a `datetime64` descriptor).

## Post-processing

`Postprocess` is the output half of every tutorial as `np` compositions — no new kernels, kept apart from
the verbs so those stay conversion-only:

| | Replaces |
|---|---|
| `Postprocess.Softmax(logits, axis = -1)` (max-shifted, rows sum to 1), `LogSoftmax`, `Sigmoid` | the `Math.Exp` / `Sum` / `Select` loop |
| `Postprocess.Argmax(scores, axis = -1)` → int64, first occurrence | `GetMaxValueIndex` |
| `Postprocess.TopK(scores, k, axis = -1, largest = true)` → `(values, indices)` | `OrderByDescending().Take(k)` — ONNX `TopK` semantics, ties lower-index-first |

They agree with ORT's own `Softmax` / `ArgMax` / `TopK` operators run on the same data (the tests do exactly
that). Float32 in gives float32 out. A bool or integer input is promoted to NumPy's float tier before the
computation (bool, int8, uint8 → float16; int16, uint16, char → float32; wider → float64), so the result is
the mathematical one, not a wrapped integer. `LogSoftmax` follows `scipy.special.log_softmax` for a row whose
maximum is not finite. `Argmax` and `TopK` check their axis.

## The #512 example — a BGRA image to an NCHW tensor, no pixel loop

The [issue](https://github.com/SciSharp/NumSharp/issues/512) fills a `DenseTensor<float>` one pixel at a time
(`/255`, channel reorder). With NumSharp it is four vectorized ops feeding ORT zero-copy:

```csharp
NDArray bgra  = np.frombuffer(data, NPTypeCode.Byte).reshape(H, W, 4);                      // interleaved B,G,R,A
NDArray chw   = np.stack(new[] { bgra[":,:,2"], bgra[":,:,1"], bgra[":,:,0"] }, axis: 0)    // (3,H,W) RGB
                  .astype(NPTypeCode.Single) / 255f;                                        // fresh C-contiguous f32
NDArray input = chw.reshape(1, 3, H, W);                                                    // NCHW

using OrtTensor t = input.AsOrtValue(OnnxCodecMode.View);                                   // ZERO COPY, or it throws
using var results = session.Run(runOptions, new[] { "data" }, new[] { t.Value }, session.OutputNames);
NDArray output = results[0].ToNDArray();
```

The one copy left — `stack` + `astype`/divide producing the planar normalized array — is genuine work
(layout + dtype + scale), not marshaling. Or, one line: `NDArray output = session.Run(input);`.

## Lifetime & memory safety

- **Exports** (`AsOrtValue` / `AsDenseTensor<T>`) take their own atomic reference on the NumSharp buffer
  (NumSharp's ARC), so the memory survives even if the source `NDArray` is disposed or collected while ORT
  holds the tensor. Release is deterministic and owned by the returned handle (`OrtTensor.Dispose` /
  `OrtTensor<T>.Dispose`), which also disposes the temporary copy when the handle holds one. **Keep the handle
  (or its `Value`) reachable across `Run`** — pass `handle.Value`, dispose after the run (`using`): ORT's own
  `Run` holds no reference to the managed inputs while native code reads them, and neither do `OrtIoBinding`,
  `SessionOptions.AddInitializer` or `OrtValue.CreateSequence`. While exported, `ndarray.resize(refcheck: true)`
  on the source refuses (`cannot resize an array that references or is referenced by another array`), exactly
  NumPy's rule.
- **Reachable means valid.** Holding only `handle.Value`, or only `Tensor` / `Memory`, keeps the handle — and
  its reference on the buffer — alive, so a collection never disposes an `OrtValue` or disables a tensor you
  still hold. A forgotten handle is released by a finalizer safety net once nothing reaches any of them; it only
  gives the reference back (never frees eagerly), so an alias of the buffer you still hold stays valid, and it
  never disposes the `OrtValue` or a `Source` conversion you can still reach.
- **A value of a disposed handle is refused, not passed to ORT.** After `handle.Dispose()`, ONNX Runtime has
  released `handle.Value`, and ORT does not guard a released value (a native call on one ends the process).
  Every verb that takes an `OrtValue` throws `ObjectDisposedException` for such a value. A value of ORT's own
  that you disposed yourself cannot be recognized.
- **Pins of an `AsDenseTensor<T>` handle's `Memory<T>` outlive the handle.** Whoever pins that memory —
  ORT's `CreateTensorValueFromMemory`, any `Memory<T>.Pin()` — takes a reference of its own on the NumSharp
  buffer, given back exactly once: when the pin is disposed (two copies of one `MemoryHandle` count once) or
  when it is collected undisposed (ORT's `OrtValue` finalizer never disposes the pin it holds). A disposed
  handle only refuses NEW access: its tensor's `Memory<T>` throws `ObjectDisposedException` for a new span or
  pin, and so do `ToNDArray` / `AsNDArray` of that tensor (before allocating anything).
- **A released source is refused** with `ObjectDisposedException` by every verb that reads it (the exports,
  the copies, `Run`'s cast and C-order copy): each takes its reference on the buffer before its first read,
  so it never reads memory the pool may have handed to another array. Dtype and size refusals come first, so
  nothing is converted, copied or allocated for an array that cannot cross.
- **Imports of a buffer ORT owns** (`AsNDArray` of a run output) lease it through NumSharp's memory-block
  reference counting: the lease fires when the last NumSharp view over the memory — derived slices included —
  is disposed or collected. A non-owning view roots the `OrtValue` (it cannot be finalized under the view) but
  the caller must not `Dispose` it while a view is in use — the rule ORT documents for `GetTensorDataAsSpan`.
  `ownsValue: true` transfers ownership: the last view disposes the `OrtValue`, and the caller must not. A
  call that throws never disposes the value. Only an owning view reports the buffer's size to the GC. Views do
  not own their data (`flags.owndata == False`): a size-changing `resize` refuses instead of detaching.
- **Imports of NumSharp memory are ordinary NumSharp views.** A value this package made
  (`nd.AsOrtValue().Value.AsNDArray()`), a `tensor.AsNDArray()` of an `AsDenseTensor<T>` handle and an output
  that is a fed input are views of a NumSharp buffer. They hold their own counted reference, stay valid after
  the handle and the source are disposed, and are read-only when the exported array was. ORT itself has no
  read-only tensors: a read-only array is a legal model input, and ORT could write a tensor it was given.
- **A value ORT returned over memory that is not ORT's.** When you run the session with ORT's own API and feed
  an `AsOrtValue` handle, ORT returns a value over the handle's memory for an output that is a fed input.
  `AsNDArray` cannot know that: such a view is valid only while the handle, or the array you fed, is alive.
  `session.Run(…, zeroCopyOutputs: true)` of this package knows what it fed and returns a counted view.
- **A `DenseTensor<T>` view keeps its tensor reachable.** `denseTensor.AsNDArray()` over a legacy `session.Run`
  output (`DisposableNamedOnnxValue.AsTensor<T>()`) roots the tensor — and through it the `OrtValue` — so
  results that are merely dropped are not finalized under the view. Disposing the results explicitly while a
  view lives still frees the memory under it: keep them alive, or copy with `ToNDArray`.
- **Only CPU-addressable tensors** can be read or viewed. A CUDA / DirectML / TensorRT-resident output is
  refused (`the tensor lives in 'Cuda' memory ... which the CPU cannot address`) — bind a CPU output or copy
  to CPU first. Host-accessible device memory (`CudaPinned`, `HipPinned`, `QnnHtpShared`) and `CpuAligned4K`
  are fine: the bridge asks ORT for the device memory kind where the loaded ORT can tell, and uses the
  allocator name otherwise.
- No GIL, no interpreter lifetime, no thread-affinity: ORT is thread-safe and the verbs are plain calls.

## Limits (stated up front)

- **Complex** (ORT has no complex tensors), **BFloat16** / float8 / 4-bit (no NumSharp dtype).
- **Sparse tensors.** ORT's C# API cannot create or read one. A model input declared sparse passes the checks
  of `Run` and fails inside ORT.
- **Non-tensor outputs through `Run`.** `Run` returns tensors only (text included). Fetch a sequence or a map
  with ORT's own `session.Run` and read it with `ToNDArrays` / `ToMap` / `ToMaps`. A string-keyed map's keys
  come back sorted: ORT keeps a map ordered by key.
- **GPU-resident tensors** cannot be viewed as `NDArray`s.
- ORT's managed data API (`GetTensorMutableRawData` and every span over it) is `int`-indexed — it hands out a
  tensor's memory only as a `Span` of bytes, whose length it casts to `int` unchecked. **Export:** an array over
  2 GB is refused by `ToOrtValue` before anything is copied or allocated — share it zero-copy with `AsOrtValue`,
  which hands ORT a native pointer (no limit). **Import:** a tensor of 2 GB or more that ORT owns can be neither
  copied (`ToNDArray`) nor viewed (`AsNDArray`) out of an `OrtValue`; both refuse up front, before anything is
  allocated. Let ORT write such an output into NumSharp's memory instead: pre-allocate it and call `RunInto`.
  A value this package made, and a `DenseTensor<T>` (up to `int.MaxValue` elements), cross at any byte size.
- `ToDenseTensor<T>` copies into a managed `T[]`, which holds at most `Array.MaxLength` (2,147,483,591)
  elements; a larger array is refused up front. `AsDenseTensor<T>` takes up to `int.MaxValue` elements (a
  `Memory<T>` cannot index more), `AsOrtValue` has no element limit.
- `Run(...)` shape validation can only see what `NodeMetadata` reports: a declaration without dimensions is
  either a scalar or an unknown-rank tensor (indistinguishable there) and is left to ORT's own check. For the
  same reason `RunAsync` does not allocate such an output.
- The plan and the record of this surface: `docs/plans/onnxruntime-rework.md`.
