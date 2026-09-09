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

> Verified against ONNX Runtime **1.16.0** (the package floor) and **1.29.0** (current) on net8.0 / net10.0,
> by `test/NumSharp.Tests.Interop.OnnxRuntime` (117 tests over 20 committed `.onnx` models ORT actually executes).

## Install

```bash
dotnet add package NumSharp.Interop.OnnxRuntime
dotnet add package Microsoft.ML.OnnxRuntime          # or .Gpu / .DirectML / .QNN — exactly ONE native flavour
```

The package depends only on **`Microsoft.ML.OnnxRuntime.Managed`** (the C# API). ORT ships its native
runtime as mutually exclusive flavours — `Microsoft.ML.OnnxRuntime` (CPU), `.Gpu`, `.DirectML`, ... — and
two of them side by side is a documented misconfiguration, so the flavour is yours to pick: the one you
already reference to run inference at all. The dependency range is `[1.16.0, 2.0.0)`; a consumer's own ORT
reference (say 1.29.0) carries the same `.Managed` version and NuGet unifies upward to it.

## The verbs

Same convention as the pythonnet bridge: **`As…` shares memory** (zero-copy view), **`To…` copies**.

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → ORT | `nd.AsOrtValue()` → `OrtTensor` | **zero-copy** `OrtValue` over the NDArray's buffer (`CreateTensorValueWithData` on NumSharp's pointer). C-contiguous only. The handle owns the `OrtValue` + an ARC reference on the buffer; keep it alive across `Run`. |
| NumSharp → ORT | `nd.ToOrtValue()` → `OrtValue` | an ORT-allocated **copy**, any layout (a view is copied in logical order); no lifetime coupling. |
| NumSharp → ORT | `nd.AsDenseTensor<T>()` → `OrtTensor<T>` / `nd.ToDenseTensor<T>()` | the legacy `DenseTensor<T>` twins for `NamedOnnxValue.CreateFromTensor`; the view's `Buffer` is a `Memory<T>` over NumSharp's memory (C-contiguous → row-major, Fortran-contiguous → `reverseStride: true`). |
| ORT → NumSharp | `ortValue.ToNDArray()` → `NDArray` | a fresh **owning**, C-contiguous **copy** — the safe default; dispose the `OrtValue` whenever you like. |
| ORT → NumSharp | `ortValue.AsNDArray(ownsValue = false)` → `NDArray` | a **zero-copy view** over ORT's own buffer (CPU tensors only), mutations visible both ways. `ownsValue: true` makes the last view dispose the `OrtValue`. |
| ORT → NumSharp | `denseTensor.ToNDArray()` / `denseTensor.AsNDArray()` / `namedOnnxValue.ToNDArray()` | the legacy-surface twins (column-major tensors transpose back on copy, stay Fortran-ordered as a view; `NamedOnnxValue` dispatches on the element type at runtime). |

Plus the dtype maps `ToTensorElementType` / `FromTensorElementType` (+ `Try…` forms, `ToTensorElementClrType`)
and two counters, `NDArrayOnnxInterop.LiveExports` / `LiveImports`, that make every live crossing observable.

## The ergonomic tier — `session.Run(NDArray)`

`InferenceSessionExtensions` hides every `OrtValue` lifetime:

```csharp
NDArray y = session.Run(x);                                   // single input/output: names from metadata
NDArray y2 = session.Run("X", x, "Y2");                       // named
IReadOnlyDictionary<string, NDArray> outs = session.Run(      // several inputs, all (or some) outputs
    new Dictionary<string, NDArray> { { "A", a }, { "B", b } });
session.Run(inputs, outputs: new Dictionary<string, NDArray> { { "Y", preallocated } });   // ORT writes INTO your array
```

- **Inputs are fed zero-copy** when C-contiguous and of the model's declared dtype. A view is copied in
  logical order for the run; a dtype mismatch is **cast to the declared dtype** when NumPy's `casting` rule
  allows it (`"safe"` by default — so an int32 array into an int64 BERT input or float32 into float64 just
  works; a lossy float64 → float16 needs `casting: "same_kind"`, or an `astype` at the call site). Declared
  fixed extents and rank are checked **before** ORT sees the tensor, with a message naming the input
  (`input 'data' expects shape (1, 3, 4, 4), but the array has shape (1, 3, 4, 5): dimension 3 must be 4, not 5`);
  symbolic dims (`'batch'`, `'seq'`) are free.
- **Outputs are copied** into owning arrays by default (`ToNDArray`) and every ORT value is released before
  the call returns. `zeroCopyOutputs: true` returns views that **own** their `OrtValue` — ORT's memory is freed
  when the array (and every slice derived from it) dies — for large outputs you consume in place.
- **Pre-allocated outputs** (the `Run(inputs, outputs)` overload) are zero-copy in *both* directions: ORT writes
  straight into your arrays, the idiom for a hot inference loop that reuses its buffers (Stable Diffusion's
  UNet step). An output must be C-contiguous, writeable and of the declared dtype (it cannot be cast).

## Dtypes

| NumSharp | ORT `TensorElementType` | Crossing |
|---|---|---|
| Boolean, Byte, SByte, Int16, UInt16, Int32, UInt32, Int64, UInt64, Single, Double | Bool, UInt8, Int8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Float, Double | **zero-copy** |
| Half | Float16 | **zero-copy reinterpret** — ORT's `Float16` is a blittable 16-bit IEEE struct, binary-identical to `System.Half`; NaN payloads and subnormals cross untouched |
| Char | UInt16 | zero-copy as UTF-16 code units; directional — an ORT UInt16 comes back as UInt16, not Char |
| Decimal | — | no ONNX type: the array-producing verbs convert to Double (a float64 temporary the handle owns; lossy beyond ~16 digits); the low-level maps refuse it |
| Complex | Complex64/128 exist in the enum but **no ORT API or kernel accepts them** | refused — split into `np.real` / `np.imag` |
| — | BFloat16, String | no NumSharp dtype — refused with the ORT-side alternative named |

## Post-processing

`Postprocess` is the output half of every tutorial as `np` compositions — no new kernels, kept apart from
the verbs so those stay conversion-only:

| | Replaces |
|---|---|
| `Postprocess.Softmax(logits, axis = -1)` (max-shifted, rows sum to 1), `LogSoftmax`, `Sigmoid` | the `Math.Exp` / `Sum` / `Select` loop |
| `Postprocess.Argmax(scores, axis = -1)` → int64, first occurrence | `GetMaxValueIndex` |
| `Postprocess.TopK(scores, k, axis = -1, largest = true)` → `(values, indices)` | `OrderByDescending().Take(k)` — ONNX `TopK` semantics, ties lower-index-first |

They agree with ORT's own `Softmax` / `ArgMax` / `TopK` operators run on the same data (the tests do exactly
that). Float32 in gives float32 out; integer logits land on NumPy's float tier for `exp`.

## The #512 example — a BGRA image to an NCHW tensor, no pixel loop

The [issue](https://github.com/SciSharp/NumSharp/issues/512) fills a `DenseTensor<float>` one pixel at a time
(`/255`, channel reorder). With NumSharp it is four vectorized ops feeding ORT zero-copy:

```csharp
NDArray bgra  = np.frombuffer(data, NPTypeCode.Byte).reshape(H, W, 4);                      // interleaved B,G,R,A
NDArray chw   = np.stack(new[] { bgra[":,:,2"], bgra[":,:,1"], bgra[":,:,0"] }, axis: 0)    // (3,H,W) RGB
                  .astype(NPTypeCode.Single) / 255f;                                        // fresh C-contiguous f32
NDArray input = chw.reshape(1, 3, H, W);                                                    // NCHW

using OrtTensor t = input.AsOrtValue();                                                     // ZERO COPY
using var results = session.Run(runOptions, new[] { "data" }, new[] { t.Value }, session.OutputNames);
NDArray output = results[0].ToNDArray();
```

The one copy left — `stack` + `astype`/divide producing the planar normalized array — is genuine work
(layout + dtype + scale), not marshaling. Or, one line: `NDArray output = session.Run(input);`.

## Lifetime & memory safety

- **Exports** (`AsOrtValue` / `AsDenseTensor<T>`) take their own atomic reference on the NumSharp buffer
  (NumSharp's ARC), so the memory survives even if the source `NDArray` is disposed or collected while ORT
  holds the tensor. Release is deterministic and owned by the returned handle (`OrtTensor.Dispose` /
  `OrtTensor<T>.Dispose`); a forgotten handle is released by a finalizer safety net. **Keep the handle alive
  across `Run`** — pass `handle.Value`, dispose after the run (`using`). While exported,
  `ndarray.resize(refcheck: true)` on the source refuses (`cannot resize an array that references or is
  referenced by another array`), exactly NumPy's rule.
- **Imports** (`AsNDArray`) lease ORT's buffer through NumSharp's memory-block reference counting: the lease
  fires when the last NumSharp view over the memory — derived slices included — is disposed or collected.
  Non-owning views root the `OrtValue` (it cannot be finalized under them) but the caller must not `Dispose`
  it while a view is in use — the rule ORT documents for `GetTensorDataAsSpan`. `ownsValue: true` transfers
  ownership: the last view disposes the `OrtValue`, and the caller must not. Views do not own their data
  (`flags.owndata == False`): a size-changing `resize` refuses instead of detaching from ORT's memory.
- **Only CPU-addressable tensors** can be read or viewed. A CUDA / DirectML / TensorRT-resident output is
  refused (`the tensor lives in 'Cuda' memory ... which the CPU cannot address`) — bind a CPU output or copy
  to CPU first. Host-accessible device memory (`CudaPinned`) is fine.
- **Only C-contiguous arrays share.** ORT tensors are dense row-major with no strides; a sliced / transposed /
  negative-stride / broadcast / Fortran view throws from `AsOrtValue` (`ToOrtValue` copies it; `AsDenseTensor`
  also shares Fortran-contiguous memory via `reverseStride`).
- No GIL, no interpreter lifetime, no thread-affinity: ORT is thread-safe and the verbs are plain calls.

## Limits (stated up front)

- **Complex** (ORT has no complex tensors), **BFloat16** / **String** / float8 / 4-bit (no NumSharp dtype).
- **GPU-resident tensors** cannot be viewed as `NDArray`s.
- ORT's managed data API (`GetTensorMutableRawData`, spans) is `int`-indexed: copying a tensor over 2 GB
  through `ToOrtValue` / `ToNDArray` throws — share it zero-copy instead.
- `Run(...)` shape validation can only see what `NodeMetadata` reports: a declaration without dimensions is
  either a scalar or an unknown-rank tensor (indistinguishable there) and is left to ORT's own check.
