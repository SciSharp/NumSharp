# ONNX Runtime — feed a model from NumSharp memory, read it back into NumSharp

`NumSharp.Interop.OnnxRuntime` connects `NDArray` to [ONNX Runtime](https://onnxruntime.ai)'s C# API
without copying: an array becomes an `OrtValue` tensor over the very same unmanaged bytes and goes
straight into `InferenceSession.Run`; the outputs come back as `NDArray`s — an owning copy by default, or
a zero-copy view that owns ORT's buffer — ready for `np.argmax`, `Postprocess.Softmax`, `TopK`, `np.clip`
and everything else NumSharp does. The whole pre-processing loop every ORT tutorial hand-writes
(`processedImage[0, c, y, x] = (pixel / 255f - mean) / std`) and the whole post-processing loop
(`Math.Exp` / `Sum` / `OrderByDescending().Take(k)`) become vectorized NumSharp ops. This page is the
package's reference: setup, the verbs, the ergonomic `Run`, what crosses zero-copy and what cannot, who
frees what, dtypes and versions.

**On this page:** [From zero to an inference](#from-zero-to-an-inference) · [The verbs](#the-verbs) ·
[`session.Run(NDArray)`](#sessionrunndarray--the-ergonomic-tier) · [Zero-copy: the rules](#zero-copy-the-rules) ·
[Lifetime](#lifetime-who-frees-what) · [Post-processing](#post-processing) · [Dtypes](#dtypes) ·
[Versions](#versions) · [Limits](#limits) · [Claims](#claims-ledger)

> Verified on ONNX Runtime 1.16.0 (the floor) and 1.29.0 · net8.0/net10.0 · Windows, Linux, macOS.
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop.OnnxRuntime` (128 tests over 20
> committed `.onnx` models ORT executes; no Python at test time).

---

## From zero to an inference

```bash
dotnet add package NumSharp.Interop.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime        # the native runtime: CPU here; or .Gpu / .DirectML / .QNN
```

The interop depends only on `Microsoft.ML.OnnxRuntime.Managed` — the C# API. ORT's *native* runtime ships
as mutually exclusive flavours, and you must reference exactly one to run any model at all, so the
package leaves that choice to you and composes with whichever flavour you pick.

```csharp
using NumSharp;
using NumSharp.Interop.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

using var session = new InferenceSession("resnet50-v2-7.onnx");

// preprocessing as array ops: (H,W,3) uint8 -> (1,3,224,224) float32, normalized
NDArray rgb   = np.frombuffer(pixels, NPTypeCode.Byte).reshape(224, 224, 3);
NDArray chw   = np.transpose(rgb, new[] { 2, 0, 1 }).astype(NPTypeCode.Single) / 255f;
NDArray input = ((chw - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)).reshape(1, 3, 224, 224);

using NDArray logits = session.Run(input);           // zero-copy in (C-contiguous f32), owning NDArray out
using NDArray probs  = Postprocess.Softmax(logits);  // (1,1000), rows sum to 1
var (top5, classes)  = Postprocess.TopK(probs, 5);   // float32 values + int64 class ids, descending
```

One `Run` call: the input is handed to ORT as an `OrtValue` over NumSharp's buffer, the output tensor is
copied into an owning `NDArray`, and every ORT object created on the way is disposed before the call
returns.

## The verbs

Everything is packaging over six operations on the static `NDArrayOnnxInterop` class (also extension
methods). The naming is the house convention: **`As…` shares memory, `To…` copies.**

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → ORT | `nd.AsOrtValue()` → `OrtTensor` | **zero-copy** `OrtValue` over the array's buffer; C-contiguous only; the handle owns the `OrtValue` and an ARC reference on the buffer — keep it alive across `Run` |
| NumSharp → ORT | `nd.ToOrtValue()` → `OrtValue` | ORT-allocated **copy**; any layout, read in logical order; no lifetime coupling |
| NumSharp → ORT | `nd.AsDenseTensor<T>()` → `OrtTensor<T>` · `nd.ToDenseTensor<T>()` | the legacy `DenseTensor<T>` twins (for `NamedOnnxValue.CreateFromTensor`); a Fortran-contiguous array shares column-major via `reverseStride` |
| ORT → NumSharp | `ortValue.ToNDArray()` | fresh **owning** C-contiguous **copy** — the safe default |
| ORT → NumSharp | `ortValue.AsNDArray(ownsValue: false)` | **zero-copy view** over ORT's buffer (CPU tensors only); `ownsValue: true` → the last view disposes the `OrtValue` |
| ORT → NumSharp | `denseTensor.ToNDArray()` · `denseTensor.AsNDArray()` · `namedOnnxValue.ToNDArray()` | the legacy-surface twins; column-major tensors transpose back on copy and stay Fortran-ordered as a view |

Plus the dtype maps (`ToTensorElementType`, `FromTensorElementType`, `Try…`, `ToTensorElementClrType`) and
the counters `NDArrayOnnxInterop.LiveExports` / `LiveImports`, which count live handles and leases so a
leak is observable.

```csharp
using OrtTensor t = input.AsOrtValue();                                  // ZERO COPY: pointer + shape + dtype
using var results = session.Run(runOptions, new[] { "data" }, new[] { t.Value }, session.OutputNames);
NDArray copy = results[0].ToNDArray();                                  // owning; dispose `results` freely
NDArray view = results[0].AsNDArray();                                  // shares ORT's buffer while `results` lives
```

## `session.Run(NDArray)` — the ergonomic tier

`InferenceSessionExtensions` turns a session into a function on arrays and hides every `OrtValue`:

```csharp
NDArray y  = session.Run(x);                                  // one input, one output: names from metadata
NDArray y2 = session.Run("X", x, "Y2");                       // named
var outs   = session.Run(new Dictionary<string, NDArray> { { "A", a }, { "B", b } });          // -> name -> NDArray
var views  = session.Run(inputs, zeroCopyOutputs: true);      // views that OWN their OrtValue
session.Run(inputs, outputs: new Dictionary<string, NDArray> { { "Y", y } });                 // ORT writes INTO y
```

What each call does for you, from the session's metadata:

- **Names default** when the model has one input and one output; otherwise name them.
- **Dtype coercion, NumPy's way.** An input whose dtype differs from the model's declaration is cast to it
  when `casting` (default `"safe"`) allows — int32 → int64 (the BERT `input_ids` case), float32 → float64,
  bool → anything. A lossy cast (float64 → float16) is refused with the exact rule to relax:
  `input 'X' holds Double but the model declares Half (Float16); the cast is not allowed under casting='safe'.
  Convert explicitly — nd.astype(NPTypeCode.Half) — or relax the rule (casting: "same_kind" / "unsafe")`.
  `Char` and `Decimal` take the dtype map's own conversions (UInt16, Double).
- **Shape validation before ORT.** Declared rank and fixed extents are checked, symbolic dims
  (`'batch'`, `'seq'`) are free, and the error names the input and the offending dimension.
- **Layout.** A C-contiguous array is fed zero-copy; a view is copied in logical order for the run; every
  temporary is released with the call.
- **Outputs**: copies by default (`ToNDArray`); `zeroCopyOutputs: true` returns views that own their
  `OrtValue` (ORT's memory is freed when the array — or its last derived slice — dies); the
  `Run(inputs, outputs)` overload writes into arrays you allocated (C-contiguous, writeable, declared
  dtype), zero-copy in both directions — the pattern for a denoising loop that reuses its buffers.

## Zero-copy: the rules

An ORT tensor is a pointer, a row-major shape and an element type — **no strides**. So:

- **C-contiguous arrays share.** Scalars (0-d) and empty arrays cross too; a contiguous view at an offset
  (`np.split` / `np.unstack` children) shares exactly its window.
- **Anything else is refused by `AsOrtValue`** — transposed, stepped, negative-stride, broadcast, Fortran
  — with the fix in the message: `np.ascontiguousarray(nd)` / `nd.copy()`, or `ToOrtValue` (copy).
  `AsDenseTensor<T>` additionally shares a *Fortran-contiguous* array (`DenseTensor` can express
  column-major through `reverseStride: true`).
- **Outputs are always dense row-major**, so `ToNDArray` / `AsNDArray` never translate strides.
- **CPU memory only.** A tensor resident on CUDA / DirectML / TensorRT is not CPU-addressable: both read
  verbs refuse it (`the tensor lives in 'Cuda' memory ... which the CPU cannot address`); bind a CPU output
  or copy to CPU first. Host-accessible device memory (`CudaPinned`) is fine.

Under the hood the export is `OrtValue.CreateTensorValueWithData(memInfo, elementType, shape, pointer,
bytes)` on `slice.Address + Shape.Offset × itemsize`; the import takes the pointer behind
`GetTensorMutableRawData()` and wraps it as a NumSharp memory block with a release hook — the same
"wrap foreign memory" primitive the [pythonnet bridge](pythonnet-numpy.md) leases Python buffers with.

## Lifetime: who frees what

- **Exports.** `AsOrtValue` / `AsDenseTensor<T>` take an atomic reference on the NumSharp buffer, so the
  memory outlives the source array if the array is disposed or collected while ORT holds the tensor.
  Disposing the handle disposes the `OrtValue` and drops the reference; a forgotten handle is released by
  a finalizer safety net. While exported, `ndarray.resize(refcheck: true)` refuses — NumPy's rule for a
  referenced buffer — and works again once the handle is gone.
- **Imports.** `AsNDArray` leases ORT's buffer through NumSharp's memory-block reference count: the lease
  fires when the *last* NumSharp view over the memory — slices derived from it included — is disposed or
  collected. A non-owning view roots the `OrtValue` so it cannot be finalized under it, but you must not
  `Dispose` the `OrtValue` (or the `Run` result collection) while a view is in use — exactly ORT's rule for
  `GetTensorDataAsSpan`. With `ownsValue: true` the last view disposes the `OrtValue` and you must not.
  Views do not own their data (`owndata == false`): a size-changing `resize` refuses instead of detaching.
- **No GIL, no interpreter, no thread affinity** — ORT is thread-safe and the verbs are plain calls. The
  one rule is the handle rule: **pass `handle.Value`, keep the handle alive across `Run`, dispose after.**

## Post-processing

`Postprocess` is the output half of the tutorials as `np` compositions — no new kernels, and kept in its
own type so the verbs stay conversion-only:

| Method | Replaces | Contract |
|---|---|---|
| `Softmax(logits, axis = -1)` · `LogSoftmax` · `Sigmoid` | the `Math.Exp` / `Sum` / `Select` loop | max-shifted (stable for huge logits); float32 in, float32 out |
| `Argmax(scores, axis = -1)` | `GetMaxValueIndex` | int64, first occurrence — ONNX `ArgMax(select_last_index=0)` |
| `TopK(scores, k, axis = -1, largest = true)` → `(values, indices)` | `OrderByDescending().Take(k)` | ONNX `TopK`: sorted, ties **lower index first**; values keep the input dtype |

The tests run ORT's own `Softmax` / `ArgMax` / `TopK` operators on the same data and compare — to float
tolerance for softmax, exactly for the indices.

## Dtypes

| NumSharp | ORT | Crossing |
|---|---|---|
| Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double | Bool, UInt8, Int8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float, Double | zero-copy |
| Half | Float16 | zero-copy **reinterpret** — ORT's `Float16` is a blittable 16-bit IEEE struct, the same bits as `System.Half`; NaN payloads and subnormals cross untouched |
| Char | UInt16 | zero-copy, UTF-16 code units; **directional** — UInt16 imports as UInt16 |
| Decimal | — | no ONNX type: the array verbs convert to Double (an owned float64 temporary; lossy past ~16 digits); the dtype maps refuse |
| Complex | Complex64/128 are in the enum but **no ORT API or kernel takes them** | refused; feed `np.real` / `np.imag` |
| — | BFloat16, String | no NumSharp dtype; refused with the ORT-side alternative named |

## Versions

The dependency is `Microsoft.ML.OnnxRuntime.Managed [1.16.0, 2.0.0)`. **1.16.0** (September 2023) is the
floor because it is the release that made the `OrtValue` API the preferred C# path and gave `Float16` its
full surface — every member the package uses is present there (verified against the tagged source). The
package compiles against the floor, so any newer ORT rolls forward; your own native-flavour reference
(say `Microsoft.ML.OnnxRuntime 1.29.0`) carries a matching `.Managed` and NuGet unifies upward to it. CI
runs the suite at both ends of the range.

## Limits

- Complex tensors (ORT has none), BFloat16 / String / float8 / 4-bit (NumSharp has none).
- GPU-resident tensors cannot be viewed or copied through this package — bring them to CPU first.
- ORT's managed data API is `int`-indexed: copying a tensor over 2 GB (`ToOrtValue`, `ToNDArray`) throws;
  share it zero-copy instead.
- `Run`'s shape check sees what `NodeMetadata` reports; a declaration with no dimensions is either a scalar
  or unknown-rank (indistinguishable there) and is left to ORT's own validation.

## Claims ledger

| # | Claim | Gate |
|---|---|---|
| 1 | `AsOrtValue` round-trips bit-exact for 12 dtypes × {0-d, empty, 1-D, 2-D, 3-D} | `ExportTests.AsOrtValue_RoundTrips_BitExact_EveryDtype_EveryShape` |
| 2 | The tensor pointer IS the NumSharp buffer; mutations flow both ways | `ExportTests.AsOrtValue_IsZeroCopy_TensorPointerIsTheNumSharpBuffer`, `…_MutationsAreVisibleBothWays` |
| 3 | Non-contiguous views are refused with the materialize hint; `ToOrtValue` copies them in logical order | `ExportTests.AsOrtValue_NonContiguousViews_AreRefused…`, `ToOrtValue_CopiesAnyLayout…` |
| 4 | Half ↔ Float16 crosses bit-identical, NaN payloads and subnormals included | `ExportTests.AsOrtValue_Half_CrossesAsFloat16_BitIdentical…` |
| 5 | A real session output copies bit-exact for every dtype; a view shares ORT's buffer both ways | `ImportTests.ToNDArray_FromARealOutput…`, `AsNDArray_View_SharesOrtsOutputBuffer_BothWays` |
| 6 | An owning view disposes the `OrtValue` when its last derived slice dies | `ImportTests.AsNDArray_OwnsValue_TheLastView_DisposesTheOrtValue` |
| 7 | The source may die while exported; `resize` refuses while exported; forgotten handles/views are released by GC | `LifetimeTests` |
| 8 | `Run` coerces int32 → int64 under `safe`, refuses float64 → float16, validates declared shapes with the input named | `SessionRunTests` |
| 9 | Pre-allocated outputs are written in place; zero-copy outputs own their values | `SessionRunTests.Run_PreallocatedOutputs…`, `Run_ZeroCopyOutputs…` |
| 10 | Softmax / Argmax / TopK agree with ORT's own operators, ties lower-index-first | `PostprocessTests` |
| 11 | The #512 BGRA → NCHW example and the Stable-Diffusion step run as written | `DocExampleTests` |
| 12 | Special values (NaN sign/payload/signaling, ±inf, ±0, subnormals, every dtype extreme, BMP code units) survive every crossing AND a real `Identity` session bit-exact; a CLR NaN is preserved uncanonicalized | `SpecialValueFidelityTests` |
| 13 | An import view is non-owning: a same-size `resize` reshapes in place and stays on ORT's memory, a derived slice inherits it, a 0-d value goes through the real lease path | `ImportViewOwnershipTests` |

## See also

- [Interoperability overview](index.md) — the contract every bridge builds on
- [Python & numpy (pythonnet)](pythonnet-numpy.md) — the sibling bridge whose lifetime model this one mirrors
- Package README: `src/NumSharp.Interop.OnnxRuntime/README.md` · plan: `docs/plans/onnxruntime.md`
