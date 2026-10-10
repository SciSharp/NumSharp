# ONNX Runtime - feed a model from NumSharp memory, read it back into NumSharp

`NumSharp.Interop.OnnxRuntime` connects `NDArray` to [ONNX Runtime](https://onnxruntime.ai)'s C# API
without copying: an array becomes an `OrtValue` tensor over the very same unmanaged bytes and goes
straight into `InferenceSession.Run`; the outputs come back as `NDArray`s - an owning copy by default, or
a zero-copy view that owns ORT's buffer - ready for `np.argmax`, `Postprocess.Softmax`, `TopK`, `np.clip`
and everything else NumSharp does. The whole pre-processing loop every ORT tutorial hand-writes
(`processedImage[0, c, y, x] = (pixel / 255f - mean) / std`) and the whole post-processing loop
(`Math.Exp` / `Sum` / `OrderByDescending().Take(k)`) become vectorized NumSharp ops. This page is the
package's reference: setup, the verbs and their modes, the ergonomic `Run`, asynchronous runs, text, what
crosses zero-copy and what cannot, who frees what, dtypes and versions.

**On this page:** [From zero to an inference](#from-zero-to-an-inference) ·
[Views, copies and modes](#views-copies-and-onnxcodecmode) · [The verbs](#the-verbs) ·
[`session.Run(NDArray)`](#sessionrunndarray---the-ergonomic-tier) · [Asynchronous runs](#asynchronous-runs) ·
[Text](#text) · [Zero-copy: the rules](#zero-copy-the-rules) · [Lifetime](#lifetime-who-frees-what) ·
[Post-processing](#post-processing) · [Dtypes](#dtypes) · [Versions](#versions) · [Limits](#limits) ·
[Claims](#claims-ledger)

> Verified on ONNX Runtime 1.16.0 (the floor) and 1.31.0 · net8.0/net10.0 · Windows, Linux, macOS.
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop.OnnxRuntime`, over committed `.onnx`
> models that ORT executes; no Python at test time.

---

## From zero to an inference

```bash
dotnet add package NumSharp.Interop.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime        # the native runtime: CPU here; or .Gpu / .DirectML / .QNN
```

The interop depends only on `Microsoft.ML.OnnxRuntime.Managed` - the C# API. ORT's *native* runtime ships
as mutually exclusive flavours, and you must reference exactly one to run any model at all, so the
package leaves that choice to you and composes with whichever flavour you pick.

```csharp
using NumSharp;
using NumSharp.Interop.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

using var session = new InferenceSession("resnet50-v2-7.onnx");

// preprocessing as array ops: (H,W,3) uint8 -> (1,3,224,224) float32, normalized
NDArray rgb   = np.frombuffer(pixels, NPTypeCode.Byte).reshape(224, 224, 3);
NDArray chw   = np.transpose(rgb, [2, 0, 1]).astype(NPTypeCode.Single) / 255f;
NDArray input = ((chw - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)).reshape(1, 3, 224, 224);

using NDArray logits = session.Run(input);           // zero-copy in (C-contiguous f32), owning NDArray out
using NDArray probs  = Postprocess.Softmax(logits);  // (1,1000), rows sum to 1
var (top5, classes)  = Postprocess.TopK(probs, 5);   // float32 values + int64 class ids, descending
```

One `Run` call: the input is handed to ORT as an `OrtValue` over NumSharp's buffer, the output tensor is
copied into an owning `NDArray`, and every ORT object created on the way is disposed before the call
returns.

## Views, copies and `OnnxCodecMode`

An ORT tensor is a pointer, a row-major shape and a fixed-size element type - **no strides**. ORT can read
a NumSharp buffer in place only when the array is C-contiguous and its dtype crosses bit for bit. Every
entry point that could share memory takes an `OnnxCodecMode` (the member names and meanings of the
[pythonnet bridge](pythonnet-numpy.md)'s `NumpyCodecMode`):

| Mode | What it does |
|---|---|
| `Auto` (default) | A zero-copy view when the dtype and the layout permit it, an independent copy otherwise. It never fails because of a layout or a convertible dtype. |
| `View` | Always a view. A crossing that needs a copy throws `InvalidOperationException` and names what prevents the view. |
| `Copy` | Always an independent copy. |

A copy is needed for a Fortran-ordered, stepped, reversed, transposed or broadcast view, for a `Decimal`
array (a float64 conversion) and for text. No mode repairs a dtype with no ONNX element type (`Complex`), a
released buffer, or a size the target cannot hold. Writes by ORT land in the memory it was given: with a
view that is your array, with a copy it is a temporary. `handle.IsView` says which.

## The verbs

Everything is packaging over a few operations on the static `NDArrayOnnxInterop` class (also extension
methods). **`As…` gives a view when it can, `To…` always copies.**

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → ORT | `nd.AsOrtValue(mode)` → `OrtTensor` | an `OrtValue` over the array's buffer when a view is possible, else (under `Auto`) over a temporary copy the handle owns; `IsView` says which; the handle owns the `OrtValue` and an ARC reference on the buffer - keep it alive across `Run` |
| NumSharp → ORT | `nd.ToOrtValue()` → `OrtValue` | ORT-allocated **copy**; any layout, read in logical order; no lifetime coupling |
| NumSharp → ORT | `nd.AsDenseTensor<T>(mode)` → `OrtTensor<T>` · `nd.ToDenseTensor<T>()` | the legacy `DenseTensor<T>` twins (for `NamedOnnxValue.CreateFromTensor`); always row-major, so a Fortran-ordered array is a copy under `Auto` |
| ORT → NumSharp | `ortValue.ToNDArray()` | fresh **owning** C-contiguous **copy** - the safe default; a string tensor becomes a text array |
| ORT → NumSharp | `ortValue.AsNDArray(ownsValue: false, mode)` | **zero-copy view** over the tensor's buffer (CPU tensors only); `ownsValue: true` → the last view disposes the `OrtValue`; `mode: Copy` → an independent copy |
| ORT → NumSharp | `denseTensor.ToNDArray()` · `denseTensor.AsNDArray(mode)` · `namedOnnxValue.ToNDArray()` | the legacy-surface twins; column-major tensors transpose back on copy and stay Fortran-ordered as a view |
| ORT → NumSharp | `ortValue.ToNDArrays()` · `ToMap()` · `ToMaps()` · `GetMapKeys()` · `ReadStringTensor()` | non-tensor outputs: a sequence of tensors, a map, a ZipMap sequence of maps, a string tensor as `string[]`; each numeric array is a view that owns ORT's copy of its element |

Plus the dtype maps (`ToTensorElementType`, `FromTensorElementType`, `Try…`, `ToTensorElementClrType`),
`IsCpuAccessible`, and the counters `NDArrayOnnxInterop.LiveExports` / `LiveImports`, which count live
handles and leases so a leak is observable.

```csharp
using OrtTensor t = input.AsOrtValue(OnnxCodecMode.View);               // ZERO COPY, or it throws
using var results = session.Run(runOptions, new[] { "data" }, new[] { t.Value }, session.OutputNames);
NDArray copy = results[0].ToNDArray();                                  // owning; dispose `results` freely
NDArray view = results[0].AsNDArray();                                  // shares ORT's buffer while `results` lives
```

## `session.Run(NDArray)` - the ergonomic tier

`InferenceSessionExtensions` turns a session into a function on arrays and hides every `OrtValue`:

```csharp
NDArray y  = session.Run(x);                                  // one input, one output: names from metadata
NDArray y2 = session.Run("X", x, "Y2");                       // named
var outs   = session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });          // -> name -> NDArray
var views  = session.Run(inputs, zeroCopyOutputs: true);      // views that OWN their OrtValue
session.RunInto(inputs, new Dictionary<string, NDArray> { { "Y", y } });                      // ORT writes INTO y
```

What each call does for you, from the session's metadata:

- **Names default** when the model has one input and one output; otherwise name them. A name is looked up
  among the model's inputs, then among its overridable initializers. An `optional(tensor)` input takes a
  plain array.
- **Dtype coercion, NumPy's way.** An input whose dtype differs from the model's declaration is cast to it
  when `casting` (default `"safe"`) allows - int32 → int64 (the BERT `input_ids` case), float32 → float64,
  bool → anything. A lossy cast (float64 → float16) is refused with the exact rule to relax:
  `input 'X' holds Double but the model declares Half (Float16); the cast is not allowed under casting='safe'.
  Convert explicitly - nd.astype(NPTypeCode.Half) - or relax the rule (casting: "same_kind" / "unsafe")`.
  The table decides every pair: a `Decimal` array into a float64 input needs `"same_kind"`, `Char` into a
  UInt16 input is safe and stays zero-copy. The rule name is checked on every call.
- **Shape validation before ORT.** Declared rank and fixed extents are checked, symbolic dims
  (`'batch'`, `'seq'`) are free, and the error names the input and the offending dimension.
- **The input mode.** `Auto` shares an input that is C-contiguous and of the declared dtype and feeds a
  temporary for any other; `View` throws for an input that would need one; `Copy` feeds a copy of every
  input. Every temporary is released with the call.
- **Outputs**: copies by default; `zeroCopyOutputs: true` returns views that own their `OrtValue` (ORT's
  memory is freed when the array - or its last derived slice - dies).
- **`RunInto`** writes into arrays you allocated (C-contiguous, writeable, declared dtype), zero-copy in both
  directions - the pattern for a denoising loop that reuses its buffers. ORT may write an output while it
  still reads an input: an output that shares memory with an input is computed from a copy of that input
  under `Auto` (NumPy's rule for `out=`) and refused under `View`; two outputs that overlap are always
  refused.
- **An output that is a fed input.** ONNX permits a model output that no node produces: a model input
  passed through, or an overridable initializer you feed. ORT does not write a pre-allocated array for it,
  and the value it returns lies in the fed memory. `RunInto` writes the fed values into your array, the copy
  form copies them, and the zero-copy form returns a counted view of the fed buffer.

## Asynchronous runs

```csharp
await session.RunIntoAsync(inputs, outputs);                  // ORT writes into your arrays; no thread waits
IReadOnlyDictionary<string, NDArray> outs = await session.RunAsync(inputs);   // outputs allocated for you
```

Both build on ORT's own native asynchronous run: the model executes on the session's intra-op thread pool
and the task completes from a native callback.

- **ONNX Runtime 1.30.0 or newer.** Older managed assemblies do not pin the arrays the native run keeps
  reading (microsoft/onnxruntime#32015). Both forms throw `NotSupportedException` there;
  `InferenceSessionExtensions.IsRunAsyncSupported` tells which applies.
- **2 or more intra-op threads** (`SessionOptions.IntraOpNumThreads` is not 1). With one, ORT refuses and
  the task is faulted.
- **`RunAsync` allocates its outputs before the run**, so it needs their shape then: a numeric tensor with
  a fixed declared shape, or an output that is a fed input. A dimension fixed only at run time, a
  declaration without dimensions (a scalar or an unknown rank: ORT's metadata does not tell which), a string
  tensor and a non-tensor output are refused with what to do instead. `RunIntoAsync` works for every model
  whose outputs you can allocate.
- **Errors.** A usage error throws from the call. A failure inside ORT faults the task. To cancel, pass a
  `RunOptions` and set `Terminate = true`.
- **Where your code continues.** ORT completes its own task on an intra-op thread of the session, and code
  that continues there and disposes the session never returns (measured). The bridge's task runs your
  continuation off that thread. ORT also awaits without `ConfigureAwait(false)`, so its own task waits for
  the caller's synchronization context: a thread that blocked on it waited forever. The bridge hides the
  context from ORT, so a blocking wait is released.
- **In flight.** Until the task completes, ORT reads the shared inputs and writes the outputs. Do not write
  an input (pass `OnnxCodecMode.Copy` to feed a snapshot). To dispose an array in flight is safe: the run
  holds a reference on every buffer.

## Text

NumSharp's fixed-width text dtypes (`U`: UCS-4 code points; `S`: bytes) cross as ORT string tensors in
both directions: `ToOrtValue`, `AsOrtValue`, `ToNDArray`, `AsNDArray`, the readers and `Run`.

- **Every crossing is a copy.** ORT keeps each string as a separate object in its own memory. `Auto` and
  `Copy` do the same; `View` throws.
- A `U` element crosses as its text (ORT stores UTF-8); an `S` element crosses as its bytes, unchecked. A
  string tensor reads back as a `U` array as wide as its longest string.
- **The fixed width has a cost**, until NumSharp has the variable-width `StringDType`: a string that ends
  in NUL characters loses them (`ReadStringTensor` returns the strings exactly), and one long string widens
  every element.
- Text and numbers do not convert into each other yet, under any casting rule.
- `RunInto` and `RunAsync` cannot take a string output; the synchronous `Run` returns it.

## Zero-copy: the rules

- **C-contiguous arrays share.** Scalars (0-d) and empty arrays cross too; a contiguous view at an offset
  (`np.split` / `np.unstack` children) shares exactly its window.
- **Anything else is copied under `Auto` and refused under `View`** - transposed, stepped,
  negative-stride, broadcast, Fortran - with the reason in the message.
- **Outputs are always dense row-major**, so `ToNDArray` / `AsNDArray` never translate strides.
- **CPU memory only.** A tensor resident on CUDA / DirectML / TensorRT is not CPU-addressable: both read
  verbs refuse it (`the tensor lives in 'Cuda' memory ... which the CPU cannot address`); bind a CPU output
  or copy to CPU first. Host-accessible device memory (`CudaPinned`, `HipPinned`, `QnnHtpShared`) and
  `CpuAligned4K` are fine: the bridge asks ORT for the device memory kind where the loaded ORT can tell.

Under the hood the export is `OrtValue.CreateTensorValueWithData(memInfo, elementType, shape, pointer,
bytes)` on `slice.Address + Shape.Offset × itemsize`; the import takes the pointer behind
`GetTensorMutableRawData()` and wraps it as a NumSharp memory block with a release hook - the same
"wrap foreign memory" primitive the [pythonnet bridge](pythonnet-numpy.md) leases Python buffers with.

## Lifetime: who frees what

- **Exports.** `AsOrtValue` / `AsDenseTensor<T>` take an atomic reference on the NumSharp buffer, so the
  memory outlives the source array if the array is disposed or collected while ORT holds the tensor.
  Disposing the handle disposes the `OrtValue`, drops the reference and disposes the temporary copy when the
  handle holds one. While exported, `ndarray.resize(refcheck: true)` refuses - NumPy's rule for a referenced
  buffer - and works again once the handle is gone.
- **Reachable means valid.** The handle and what it hands out keep each other alive: holding only
  `handle.Value`, or only `Tensor` / `Memory`, keeps the handle and its reference, so a collection never
  disposes an `OrtValue` or disables a tensor you still hold, and never lets `resize` reallocate under it. A
  forgotten handle is released by a finalizer safety net once nothing reaches any of them; it only gives the
  reference back (never frees eagerly), so an alias of the buffer you still hold stays valid.
- **A value of a disposed handle is refused.** After `handle.Dispose()`, ONNX Runtime has released
  `handle.Value`, and ORT does not guard a released value: a native call on one ends the process. Every verb
  that takes an `OrtValue` throws `ObjectDisposedException` for such a value instead.
- **Pins outlive the handle.** Whoever pins an `AsDenseTensor<T>` handle's `Memory<T>` - ORT's
  `CreateTensorValueFromMemory`, any `Memory<T>.Pin()` - takes its own reference on the NumSharp buffer,
  given back exactly once: when the pin is disposed (two copies of one `MemoryHandle` count once), or when
  it is collected undisposed. A disposed handle refuses only NEW access: a new span or pin of its
  `Memory<T>`, and `ToNDArray` / `AsNDArray` of its tensor, throw `ObjectDisposedException` before anything
  is allocated.
- **A released source is refused.** Every verb that reads an array - the exports, the copies, and `Run`'s
  cast and C-order copy of an input - takes its reference on the array's buffer before its first read, and
  throws `ObjectDisposedException` if the buffer has already been released. Refusals that need only the dtype
  and the shape come first, so nothing is converted, copied or allocated for an array that cannot cross.
- **Imports of a buffer ORT owns.** `AsNDArray` leases the buffer through NumSharp's memory-block reference
  count: the lease fires when the *last* NumSharp view over the memory - slices derived from it included -
  is disposed or collected. A non-owning view roots the `OrtValue` so it cannot be finalized under it, but
  you must not `Dispose` the `OrtValue` (or the `Run` result collection) while a view is in use - exactly
  ORT's rule for `GetTensorDataAsSpan`. With `ownsValue: true` the last view disposes the `OrtValue` and you
  must not; a call that throws never disposes it. Only an owning view reports the buffer's size to the GC.
  Views do not own their data (`owndata == false`): a size-changing `resize` refuses instead of detaching.
- **Imports of NumSharp memory are ordinary NumSharp views.** A value this package made
  (`nd.AsOrtValue().Value.AsNDArray()`), a `tensor.AsNDArray()` of an `AsDenseTensor<T>` handle, and an
  output that is a fed input are views of a NumSharp buffer: counted on it like any view, valid after the
  handle and the source are disposed, and read-only when the exported array was. ORT has no read-only
  tensors: a read-only array is a legal model input.
- **The readers return owning views.** Each numeric array of `ToNDArrays` / `ToMap` / `ToMaps` owns ORT's
  copy of its element and frees it when disposed; the value it was read from may be disposed at once.
- **A `DenseTensor<T>` view roots its tensor.** ORT's memory manager for a legacy `session.Run` output
  (`DisposableNamedOnnxValue.AsTensor<T>()`) pins nothing, so `denseTensor.AsNDArray()` keeps the tensor - and
  through it the `OrtValue` - reachable itself. Disposing the results explicitly while a view lives still
  frees the memory under it - the same rule as a non-owning `AsNDArray(OrtValue)`.
- **A pass-through output through ORT's own API.** When you run the session yourself and feed an
  `AsOrtValue` handle, the value ORT returns for an output that is a fed input lies in the handle's memory,
  and `AsNDArray` cannot know that: the view is valid only while the handle, or the array you fed, is alive.
- **No GIL, no interpreter, no thread affinity** - ORT is thread-safe and the verbs are plain calls. The
  one rule is the handle rule: **pass `handle.Value`, keep the handle (or its `Value`) reachable across
  `Run`, dispose after.** ORT's own `Run` reads the inputs' native handles and holds no reference to the
  managed values during the run, so an input referenced only by the argument list can be collected while
  native code reads it; the same holds for anything that keeps a value past a call (`OrtIoBinding`,
  `SessionOptions.AddInitializer`, `OrtValue.CreateSequence`). `session.Run(NDArray…)` does this for you.
  The full lifetime and garbage-collection analysis: `docs/plans/interop-lifetime-gc-analysis.md`.

## Post-processing

`Postprocess` is the output half of the tutorials as `np` compositions - no new kernels, and kept in its
own type so the verbs stay conversion-only:

| Method | Replaces | Contract |
|---|---|---|
| `Softmax(logits, axis = -1)` · `LogSoftmax` · `Sigmoid` | the `Math.Exp` / `Sum` / `Select` loop | max-shifted (stable for huge logits); float32 in, float32 out; a bool or integer input is promoted to NumPy's float tier first; `LogSoftmax` follows `scipy.special.log_softmax` for a non-finite row maximum |
| `Argmax(scores, axis = -1)` | `GetMaxValueIndex` | int64, first occurrence - ONNX `ArgMax(select_last_index=0)`; the axis is checked |
| `TopK(scores, k, axis = -1, largest = true)` → `(values, indices)` | `OrderByDescending().Take(k)` | ONNX `TopK`: sorted, ties **lower index first**; values keep the input dtype |

The tests run ORT's own `Softmax` / `ArgMax` / `TopK` operators on the same data and compare - to float
tolerance for softmax, exactly for the indices.

## Dtypes

| NumSharp | ORT | Crossing |
|---|---|---|
| Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double | Bool, UInt8, Int8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float, Double | zero-copy |
| Half | Float16 | zero-copy **reinterpret** - ORT's `Float16` is a blittable 16-bit IEEE struct, the same bits as `System.Half`; NaN payloads and subnormals cross untouched |
| Char | UInt16 | zero-copy, UTF-16 code units; **directional** - UInt16 imports as UInt16 |
| String (`U`, `S`) | String | always a copy; see [Text](#text) |
| Decimal | - | no ONNX type: the array verbs convert to Double under `Auto` and `Copy` (an owned float64 temporary; lossy past ~16 digits), `View` refuses; the dtype maps refuse |
| Complex | Complex64/128 are in the enum but **no ORT API or kernel takes them** | refused; feed `np.real` / `np.imag` |
| - | BFloat16 | no NumSharp dtype; refused with the ORT-side alternative named |

## Versions

The dependency is `Microsoft.ML.OnnxRuntime.Managed [1.16.0, 2.0.0)`. **1.16.0** (September 2023) is the
floor because it is the release that made the `OrtValue` API the preferred C# path and gave `Float16` its
full surface - every member the package uses is present there (verified against the tagged source). The
package compiles against the floor, so any newer ORT rolls forward; your own native-flavour reference
(say `Microsoft.ML.OnnxRuntime 1.31.0`) carries a matching `.Managed` and NuGet unifies upward to it. CI
runs the suite at both ends of the range. The asynchronous forms need 1.30.0; on an older ORT the tests of
their behaviour report Inconclusive and the test of their refusal runs.

## Limits

- Complex tensors (ORT has none), BFloat16 / float8 / 4-bit (NumSharp has none).
- Sparse tensors: ORT's C# API cannot create or read one. A model input declared sparse passes the checks
  of `Run` and fails inside ORT.
- `Run` returns tensors only. Fetch a sequence or a map with ORT's own `session.Run` and read it with the
  readers. A string-keyed map's keys come back sorted: ORT keeps a map ordered by key.
- GPU-resident tensors cannot be viewed or copied through this package - bring them to CPU first.
- ORT's managed data API is `int`-indexed: it exposes a tensor's memory only as a `Span` of bytes, whose length
  it casts to `int` unchecked, and offers no other route to the pointer.
  - **Export**: `ToOrtValue` refuses an array over 2 GB before it copies or allocates anything; share it with
    `AsOrtValue` (a native pointer, no limit).
  - **Import**: a tensor of 2 GB or more that ORT owns can be neither copied (`ToNDArray`) nor viewed
    (`AsNDArray`) out of an `OrtValue`. Both refuse it up front, allocating nothing. To get such an output, let
    ORT write it into memory you allocated: `session.RunInto(inputs, outputs)`.
  - A value this package made, and a `DenseTensor<T>` (limited by elements, `int.MaxValue`, not bytes), cross
    at any byte size.
- `ToDenseTensor<T>` copies into a managed `T[]`, which holds at most `Array.MaxLength` (2,147,483,591)
  elements; a larger array is refused up front. `AsDenseTensor<T>` takes up to `int.MaxValue` elements,
  `AsOrtValue` has no element limit.
- `Run`'s shape check sees what `NodeMetadata` reports; a declaration with no dimensions is either a scalar
  or unknown-rank (indistinguishable there) and is left to ORT's own validation. `RunAsync` does not allocate
  such an output.

## Claims ledger

| # | Claim | Gate |
|---|---|---|
| 1 | `AsOrtValue` round-trips bit-exact for 12 dtypes × {0-d, empty, 1-D, 2-D, 3-D} | `ExportTests.AsOrtValue_RoundTrips_BitExact_EveryDtype_EveryShape` |
| 2 | The tensor pointer IS the NumSharp buffer; mutations flow both ways | `ExportTests.AsOrtValue_IsZeroCopy_TensorPointerIsTheNumSharpBuffer`, `…_MutationsAreVisibleBothWays` |
| 3 | Non-contiguous views are copied under `Auto` and refused under `View`; `ToOrtValue` copies them in logical order | `ExportTests.AsOrtValue_NonContiguousViews_AreCopiedUnderAuto_AndRefusedUnderView`, `ToOrtValue_CopiesAnyLayout…` |
| 4 | Half ↔ Float16 crosses bit-identical, NaN payloads and subnormals included | `ExportTests.AsOrtValue_Half_CrossesAsFloat16_BitIdentical…` |
| 5 | A real session output copies bit-exact for every dtype; a view shares ORT's buffer both ways | `ImportTests.ToNDArray_FromARealOutput…`, `AsNDArray_View_SharesOrtsOutputBuffer_BothWays` |
| 6 | A non-owning view roots its value; an owning view disposes it with the last view; the copy form of `Run` disposes ORT's values | `ValueOwnershipTests` |
| 7 | The source may die while exported; `resize` refuses while exported; forgotten handles/views are released by GC | `LifetimeTests` |
| 8 | `Run` coerces int32 → int64 under `safe`, refuses float64 → float16, validates declared shapes with the input named | `SessionRunTests` |
| 9 | The input mode on every form; the casting rule on every call; optional inputs; overridable initializers; an output that overlaps an input is computed from a copy or refused | `SessionRunModeTests` |
| 10 | Softmax / Argmax / TopK agree with ORT's own operators, ties lower-index-first; integer inputs give the mathematical result | `PostprocessTests`, `PostprocessEdgeTests` |
| 11 | The #512 BGRA → NCHW example and the Stable-Diffusion step run as written | `DocExampleTests` |
| 12 | A released source is refused by every verb and `Run`, in every layout, conversion and cast; dtype and size refusals come first, and an oversize copy is refused before anything is allocated | `ReleasedSourceTests` |
| 13 | Every export verb satisfies its contract in every mode (bytes against NumSharp's and ORT's own reads, write-through, round trips, the refusals) for every dtype × 47 catalogued layouts - 0-d, empties, offsets, negative strides, broadcasts, F-order, rank 40 and 70 - and for seeded random views per dtype plus every slice of a small vector and every axis-slice pair of a small matrix | `LayoutMatrixTests`, `PropertyBasedTests` |
| 14 | Every feed route through ORT's own `Identity` returns the elements byte for byte, for every dtype and layout; a pre-allocated output window receives exactly the result; `Add` / `MatMul` over strided and broadcast inputs match NumSharp | `OrtConsumerTests` |
| 15 | Routes that must agree do: slicing commutes with exporting, a view and its copy cross identically, round trips are idempotent | `MetamorphicTests` |
| 16 | Every path - success, refusal, failure half-way - leaves the buffer references and the pool balanced; `Memory<T>` pins outlive the handle; a stale tensor and a shape-decided refusal allocate nothing | `ArcAccountingTests` |
| 17 | Handles disposed by many threads release once; pins racing a dispose stay valid or are refused; one session run from many threads over one shared input | `ConcurrencyStressTests` |
| 18 | Random bit patterns - every NaN payload, signalling NaNs, bool bytes other than 0 and 1 - cross every verb and ORT's `Identity` byte-identical | `BitPatternFuzzTests` |
| 19 | Tensors of 2 GB or more that ORT owns are refused by both imports before anything is allocated; `ToOrtValue` refuses exactly past the span limit; `AsDenseTensor` refuses more elements than a `Memory<T>` indexes | `BoundaryTests`, `DenseTensorEdgeTests` |
| 20 | The public surface equals the approved list (defaults included) | `ApiSurfaceTests` |
| 21 | In every order the runtime may run their finalizers, the handles, pins, views and imports never free a buffer an alias still reaches and give each reference back exactly once | `LifetimeGcTests` |
| 22 | A view of a value this package made is a counted alias, valid after the handle and the source; it inherits the read-only flag; only an owning view adds memory pressure | `ImportAliasTests` |
| 23 | Text crosses in both directions through every route, as a copy; `View` refuses it; a trailing NUL is lost in a text array and kept by `ReadStringTensor` | `TextTests` |
| 24 | An output that is a fed input: `RunInto` writes the fed values, the zero-copy form returns a counted view of the fed buffer, also for a temporary | `PassThroughOutputTests` |
| 25 | The asynchronous forms: the version rule, the completion off ORT's thread, a blocking wait under a context or a single-thread scheduler, arrays disposed in flight, errors from the call and through the task, the outputs `RunAsync` can and cannot allocate | `SessionRunAsyncTests` |
| 26 | The value of a disposed handle is refused by every import verb instead of ending the process | `ValueOwnershipTests.ValueOfADisposedHandle_IsRefused_ByEveryImportVerb_InsteadOfCrashingInOnnxRuntime` |
| 27 | Non-tensor readers: sequences in order with their own shapes, int64- and string-keyed maps, `GetMapKeys`, each reader's refusals | `NonTensorTests` |

## See also

- [Interoperability overview](index.md) - the contract every bridge builds on
- [Python & numpy (pythonnet)](pythonnet-numpy.md) - the sibling bridge whose lifetime model this one mirrors
- Package README: `src/NumSharp.Interop.OnnxRuntime/README.md` · plan: `docs/plans/onnxruntime.md` ·
  the 2026-10 rework and its record: `docs/plans/onnxruntime-rework.md`
