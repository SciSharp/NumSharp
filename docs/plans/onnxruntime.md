# Plan: `NumSharp.Interop.OnnxRuntime`

Zero-copy interop between NumSharp `NDArray` and **ONNX Runtime** (`Microsoft.ML.OnnxRuntime`)
tensors, shipped as its **own NuGet package** — the third interop package, in the exact mould of
`NumSharp.Interop.OpenBLAS` and `NumSharp.Interop.pythonnet`. `NumSharp.Core` gains **no ONNX
dependency**; the seam lives entirely in the optional package.

- **Driving issue:** [#512](https://github.com/SciSharp/NumSharp/issues/512) — a user hand-writing a
  per-pixel scalar loop to fill a `DenseTensor<float>` for inference, asking "Is there a faster way?".
  The answer this package gives: **yes — build the tensor directly on an `NDArray`'s buffer (no copy,
  no loop), and do the pixel math as vectorized NDArray ops.**
- **Status:** IMPLEMENTED (2026-09-08, branch `onnxruntime` on top of `journey3`) — package
  `src/NumSharp.Interop.OnnxRuntime/`, gate `test/NumSharp.Tests.Interop.OnnxRuntime/` (128 tests over 20
  committed `.onnx` models from `test/oracle/gen_onnx_models.py`, green on net8.0/net10.0 against ORT 1.16.0
  and 1.29.0), docs page `docs/website-src/docs/interop/onnxruntime.md`, CI job `onnxruntime-interop-test`,
  release build/pack wiring. Open decisions (§13) resolved as: 1 handle (`OrtTensor`); 2 copy-default
  (`ToNDArray`) with `AsNDArray(ownsValue)` opt-in; 3 Char→UInt16 one-way; 4 the two `Run` shapes PLUS a
  pre-allocated-outputs `Run(inputs, outputs)` (zero-copy both ways); 5 floor **1.16.0** (verified at the tag —
  `GetTensorSizeInBytes` and `OrtMemoryInfo.GetDeviceMemoryType` are newer and deliberately unused); 6 ORT
  `DenseTensor<T>`, `System.Numerics.Tensors` deferred; 7 `Postprocess.{Softmax,LogSoftmax,Sigmoid,Argmax,TopK}`
  shipped (ORT's own operators are the test oracle), NMS documented only; 8 BFloat16 listed as a gap; 9 coerce
  under NumPy `casting` (default `"safe"`, lossy needs `"same_kind"`/`"unsafe"`). One refinement to §10: the
  package depends on `Microsoft.ML.OnnxRuntime.Managed` ONLY (the native flavours are mutually exclusive; the
  consumer picks CPU/Gpu/DirectML), and it takes a keyed `InternalsVisibleTo` from Core (for
  `Storage.InternalArray` and the strided `Shape` ctor), like the pythonnet package. Business review:
  `docs/plans/onnxruntime-business-review.md`.
- **Evidence base (2026-09-08):** the official ORT C# tutorials
  (https://onnxruntime.ai/docs/tutorials/csharp/) and the `OrtValue`/`OrtTensorTypeAndShapeInfo`/
  `OrtMemoryInfo` API pages were read in full and drive §2.5 (what the tutorials actually do), §5c
  (post-processing), §6.5 (verbatim API surface) and §9b (the Stable Diffusion flagship). The short
  version: **every** tutorial hand-writes a scalar loop to *build* the input tensor **and** a scalar
  loop to *interpret* the output tensor — this package deletes both.

---

## 1. Why this is a conversion package, not an engine backend

The two existing interop packages sit at opposite ends of a spectrum, and ONNX Runtime belongs firmly
on the pythonnet end:

| Package | Kind | Core seam | `[ModuleInitializer]`? |
|---|---|---|---|
| `NumSharp.Interop.OpenBLAS` | **engine backend** — changes *which implementation* computes a product | `TensorEngine.Blas` (`IBlasBackend`) | yes — referencing the package IS the opt-in |
| `NumSharp.Interop.pythonnet` | **conversion library** — moves *data* across a boundary | none (pure `NDArray`⇄`PyObject` verbs) | no |
| **`NumSharp.Interop.OnnxRuntime`** | **conversion library** — moves *data* across a boundary | **none** (pure `NDArray`⇄`OrtValue`/`DenseTensor` verbs) | **no** |

ONNX Runtime does not compute NumSharp operations; it consumes tensors NumSharp produces and produces
tensors NumSharp reads. So there is **no engine seam, no `IBlasBackend`-style interface, no property on
`TensorEngine`**. The package is a set of extension methods and a couple of small `IDisposable`
handles — modelled line-for-line on `NDArrayPythonInterop` (see
`src/NumSharp.Interop.pythonnet/NDArrayPythonInterop.Export.cs`), but **simpler**, because ORT has no
GIL, no interpreter lifetime, and no `atexit` pathology to work around.

---

## 2. The two ORT crossing surfaces

ORT exposes two tensor faces, and NumSharp's **unmanaged-memory** model (`NDArray` data is a raw
`byte*` behind `UnmanagedStorage`, never a managed `T[]`) decides which one fits cleanly.

### 2a. `OrtValue` — the preferred surface (ORT ≥ 1.14)

```csharp
// Wraps a RAW POINTER. Does NOT own or free the memory. Caller guarantees validity for the
// OrtValue's lifetime. Row-major / C-contiguous only.
public static OrtValue CreateTensorValueWithData(
    OrtMemoryInfo memoryInfo, TensorElementType elementType,
    long[] shape, IntPtr dataBufferPointer, long bufferLengthInBytes);

// Pins a managed Memory<T>/T[] for the OrtValue's lifetime, unpinned on Dispose. Unmanaged T only.
public static OrtValue CreateTensorValueFromMemory<T>(
    OrtMemoryInfo memoryInfo, Memory<T> memory, long[] shape) where T : unmanaged;

// Zero-copy typed spans over the tensor buffer, valid while the OrtValue lives.
public Span<T>         GetTensorMutableDataAsSpan<T>() where T : unmanaged;
public ReadOnlySpan<T> GetTensorDataAsSpan<T>()        where T : unmanaged;
```

`CreateTensorValueWithData` is the **exact fit** for NumSharp: it takes a native pointer NumSharp
already has (`slice.Address + offset·itemsize`), does not pin, does not own, and requires only that the
memory stay valid — which the package guarantees by holding an **ARC reference** on the NumSharp buffer
for the OrtValue's lifetime. This is the primary export path.

### 2b. `DenseTensor<T>` — the legacy surface (what #512 uses)

```csharp
// Wraps a Memory<T> zero-copy. Row-major default; reverseStride=true => column-major (Fortran).
public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride = false);
public Memory<T> Buffer { get; }   // direct access to the backing store
// : Tensor<T> : TensorBase
```

`DenseTensor<T>` needs a `Memory<T>`. NumSharp's buffer is unmanaged, and there is **no `Memory<T>`
over a raw pointer** without a custom `MemoryManager<T>`. So the DenseTensor path requires an
`UnmanagedMemoryManager<T> : MemoryManager<T>` shim that (a) exposes the NumSharp pointer as a
`Memory<T>`, and (b) holds/releases the ARC reference in its `Dispose(bool)`. It is offered for
compatibility with the legacy `NamedOnnxValue`/`DenseTensor` `session.Run` overloads (and because #512
is written against it), but `OrtValue` is the recommended path.

**Decision:** implement `OrtValue` first-class; provide `DenseTensor` as a thin second surface sharing
the same dtype map and ARC-root logic.

---

## 2.5. Evidence base — what the official ORT C# tutorials actually do

Read from https://onnxruntime.ai/docs/tutorials/csharp/ (2026-09-08). Every tutorial hand-writes the
two things this package eliminates — a **scalar loop to build the input tensor** and a **scalar loop
to interpret the output tensor**. The pattern is not incidental; it is what Microsoft's own
documentation teaches, so every developer following the getting-started path writes it.

| Tutorial | Model | Input tensor | Hand-written INPUT loop | Hand-written OUTPUT loop |
|---|---|---|---|---|
| ResNet50v2 | image classification | `(1,3,224,224)` f32 NCHW | per-pixel `((px.R/255)-mean)/stddev` into a `DenseTensor<float>` | `softmax` via `Math.Exp`/`Sum`/`Select` + `OrderByDescending().Take(10)` |
| Faster R-CNN | object detection | `(3,H',W')` f32, H'/W' padded up to ×32 | per-pixel mean-subtract + pad math | stride-4 box unpack, `if conf >= 0.7` filter |
| BERT | NLP QA | 3 × `(1,seq)` **int64** | `input_ids`/`input_mask`/`segment_ids` `long[]` arrays | `argmax` over start/end logits (`GetMaxValueIndex` loop) |
| YOLOv3 (OpenVINO) | detection | `(1,3,416,416)` f32 | letterbox resize + per-pixel fill | box decode + NMS |
| Stable Diffusion | generative | latents `(1,4,64,64)` f32, embeddings `(2,77,768)` f32 | a whole `TensorHelper` class (see below) | VAE decode `clip(x/2+0.5,0,1)*255` per pixel |
| Basic | general | flat `float[]` + `long[]` shape | — | flat span read |

**The `DenseTensor` is only ever a fillable buffer.** The modern tutorials build a
`DenseTensor<float>`, run the scalar fill loop, then immediately wrap its `.Buffer`:

```csharp
using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(
    OrtMemoryInfo.DefaultInstance, processedImage.Buffer, new long[] { 1, 3, 224, 224 });
```

So the DenseTensor exists *only* to be filled and wrapped — `nd.AsOrtValue()` deletes both the
allocation and the loop. This directly validates the §2 decision to make **`OrtValue` first-class and
`DenseTensor` legacy-only**: nearly nobody wants a `DenseTensor` per se, they want a fed `OrtValue`.

**`TensorHelper.cs` — NumSharp, reimplemented by hand, in Microsoft's own sample gallery.** The
Stable Diffusion tutorial ships a 10-method `TensorHelper` class whose entire job is scalar-loop
tensor math — each method is a NumSharp one-liner:

| `TensorHelper` method | Hand-written body | NumSharp |
|---|---|---|
| `DivideTensorByFloat` | `data[i] = data[i] / value` | `nd / s` |
| `MultipleTensorByFloat` | `data[i] = data[i] * value` | `nd * s` |
| `AddTensors` | `a[i] = a[i] + b[i]` | `nd + nd` |
| `SubtractTensors` | `a[i] = a[i] - b[i]` | `nd - nd` |
| `SumTensors` | `sum[i] += t[i]` over an array | `np.sum` / stacked add |
| `SplitTensor` | nested `t1[i,j,k,l] = src[...]` | `np.split` / slicing |
| `Duplicate` | `data.Concat(data).ToArray()` | `np.concatenate` / `np.tile` |
| `GetRandomTensor` | Box–Muller loop | `np.random.normal` |
| `CreateTensor<T>` | `new DenseTensor<T>(data, dims)` | `np.array` / `frombuffer` |

This exact file is **copy-pasted verbatim** across `microsoft/ai-dev-gallery`,
`cassiebreviu/StableDiffusion`, the `onnxruntime` SD demos, `sergiosolorzano/TalkomicApp-Unity`,
`gianni-rg/SharpDiffusion` (an fp16 `Float16` variant), `CU-Production/sd15cs` and more (found via
GitHub code search on `MultipleTensorByFloat` / `class TensorHelper … SplitTensor`). It is the single
strongest justification for the package: **developers are already shipping a worse NumSharp next to
ORT, and Microsoft is one of them.**

---

## 3. The verbs (house style)

Same `As…` = view/share, `To…` = copy convention as pythonnet, and the same fluent+static duality.

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → ORT | `nd.AsOrtValue()` | **zero-copy** `OrtValue` over the NDArray buffer (raw pointer). Requires C-contiguity; ARC-rooted. Returns an `IDisposable` handle owning the OrtValue + the ARC ref. |
| NumSharp → ORT | `nd.ToOrtValue()` | independent `OrtValue` over an ORT-allocated copy (no shared memory, no lifetime coupling). Any layout. |
| NumSharp → ORT | `nd.AsDenseTensor<T>()` / `nd.ToDenseTensor<T>()` | the `DenseTensor<T>` twins (view via `UnmanagedMemoryManager<T>` / copy). |
| ORT → NumSharp | `ortValue.ToNDArray()` | **copy** ORT's output tensor into a fresh owning C-contiguous `NDArray` (safe default). |
| ORT → NumSharp | `ortValue.AsNDArray([allowReadonly])` | **zero-copy** `NDArray` view over ORT's output buffer — leased and rooted on the OrtValue (the OrtValue must outlive the view). CPU tensors only. |
| ORT → NumSharp | `denseTensor.ToNDArray()` / `denseTensor.AsNDArray()` | the `DenseTensor<T>` twins. |

High-level convenience (tier 2, see §8): `session.Run(inputs, ...)` overloads that take and return
`NDArray`s directly, hiding all OrtValue lifetime management inside one call.

---

## 4. NumSharp → ORT (feeding inputs)

The primary win of the package: eliminate the copy loop.

### The zero-copy path — `AsOrtValue`

Mirrors `NDArrayPythonInterop.ToNumpy` exactly, minus the GIL:

```csharp
public static unsafe OrtTensor AsOrtValue(this NDArray source)
{
    if (source is null) throw new ArgumentNullException(nameof(source));

    TensorElementType et = ToTensorElementType(source.typecode);   // dtype gate (§6)
    Shape shape = source.Shape;

    // ORT tensors are DENSE ROW-MAJOR with no stride metadata: only a C-contiguous NDArray
    // can be shared. Anything else (slice / transpose / negative-stride / broadcast / F-order)
    // must be materialized first — that is a copy, so route it to ToOrtValue instead.
    if (!shape.IsContiguous)
        throw new InvalidOperationException(
            "the array is not C-contiguous; ORT has no strides. Materialize first " +
            "(np.ascontiguousarray(nd) / nd.copy()) or call ToOrtValue(nd) for an owned copy.");

    IArraySlice slice = source.Storage.InternalArray;
    if (!slice.TryAddRef())          // ARC: pin the buffer for the OrtValue's lifetime
        throw new ObjectDisposedException(nameof(source), "the NumSharp buffer has already been released.");

    try
    {
        int itemsize = source.dtypesize;
        long nbytes  = shape.Size * itemsize;
        IntPtr data  = (IntPtr)((byte*)slice.Address + shape.Offset * itemsize);
        long[] dims  = shape.Dimensions;   // row-major

        OrtValue ov = OrtValue.CreateTensorValueWithData(
            OrtMemoryInfo.DefaultInstance, et, dims, data, nbytes);

        // OrtTensor owns BOTH the OrtValue and the ARC ref; disposing it disposes the OrtValue
        // and releases the ref (idempotent). This is the analog of pythonnet's ExportKeeper,
        // but deterministic — no weakref.finalize, because ORT gives us a real Dispose hook.
        return new OrtTensor(ov, slice, source);
    }
    catch { slice.Release(); throw; }
}
```

`OrtTensor : IDisposable` exposes `.Value` (the `OrtValue`) for `session.Run`, and its `Dispose`
disposes the OrtValue then releases the ARC ref. **The caller must keep the `OrtTensor` alive until
`Run` returns** — exactly like keeping a `using PyObject` alive across a numpy call.

### The copy path — `ToOrtValue`

For non-contiguous inputs, or when the caller wants an ORT-owned buffer independent of NumSharp
lifetime: densify in logical order (`np.ascontiguousarray`) and copy into an ORT-allocated tensor
(`CreateTensorValueFromMemory` / `CreateAllocatedTensorValue` + span fill). No ARC ref needed — ORT
owns the memory, freed on `OrtValue.Dispose`.

### What ORT REQUIRES from NumSharp here (the contract)

- **A stable native pointer to C-contiguous data** — `(byte*)slice.Address + shape.Offset·itemsize`
  (all public: `Storage.InternalArray`, `IArraySlice.Address`, `Shape.Offset`, `NDArray.dtypesize`).
- **Row-major shape as `long[]`** — `shape.Dimensions`, plus the byte count for the pointer overload.
- **An `NPTypeCode → TensorElementType` map** (§6).
- **A lifetime root** — `slice.TryAddRef()` held for the OrtValue's lifetime, released on dispose.
  While rooted, the source's `ndarray.resize(refcheck:true)` already refuses (the buffer is not
  uniquely referenced) — the same guard pythonnet and NumPy apply.
- **C-contiguity** — enforced up front; F-contiguous can share only to a `DenseTensor(reverseStride:true)`,
  never to an `OrtValue`.

---

## 5. ORT → NumSharp (reading outputs)

An ORT output `OrtValue` (from `session.Run`) **owns native CPU memory allocated by ORT's allocator**,
freed when the OrtValue is disposed. Two routes, split by who owns the result:

- **`ToNDArray()` — copy (the safe default).** Read `GetTensorTypeAndShape` for dtype+shape, copy
  `GetTensorDataAsSpan<T>()` into a fresh owning C-contiguous `NDArray`. The OrtValue can then be
  disposed freely. This is the recommended path — outputs are usually consumed once.
- **`AsNDArray()` — lease-and-root (zero-copy view).** Build an `NDArray` over ORT's output pointer and
  **root the OrtValue** (dispose it when the last NDArray view over the memory dies) — the mirror of
  pythonnet's *import* lease (`NDArrayPythonInterop.Import`), which leases Python memory into an NDArray
  via NumSharp's memory-block refcounting with a release callback. The **OrtValue must outlive every
  view**; a size-changing resize on such a view refuses (`does not own its data`, like `np.frombuffer`).

### Constraints on the output side

- **CPU memory only.** An output allocated on **CUDA / DirectML / TensorRT** is not CPU-addressable —
  its `OrtValue` cannot be viewed as an `NDArray`. It must be copied to CPU first (ORT's
  `OrtValue.GetTensorDataAsSpan` throws for non-CPU memory). This is the direct analog of pythonnet's
  torch-on-CUDA caveat. `AsNDArray` checks `GetTensorMemoryInfo` and declines non-CPU tensors; use
  `ToNDArray` after copying to CPU, or bind a CPU output.
- **Dense row-major.** ORT outputs are always C-contiguous, so the produced NDArray is C-contiguous;
  no stride translation is needed on this side.
- **Only decline what is genuinely unreadable.** `OrtValue.OnnxType` may be `Tensor`, `Sequence`,
  `Map` or `Optional`; the output verbs handle `Tensor` (via `IsTensor`) and throw a clear message
  for the others (some models — e.g. ZipMap classifiers — return a sequence of maps, read with
  `GetValueCount()` / `GetValue(i, allocator)`, out of scope for a dense `NDArray`).

### 5c. Output post-processing is a first-class use case (not just "read the span")

This is the half of the value proposition §1–§4 under-sell. The **input** side has a partial
substitute — ORT already offers zero-copy from a managed array (`CreateTensorValueFromMemory`), so the
honest input win is *vectorized preprocessing + zero-copy*, not zero-copy alone. **The output side has
no such substitute.** `session.Run` hands back a flat `ReadOnlySpan<T>`; turning it into a decision is
exactly the reductions/sorts/masks that are NumSharp's core — and every tutorial hand-writes them:

| Tutorial output step | Hand-written today | NumSharp |
|---|---|---|
| ResNet top-K | `Math.Exp`/`Sum`/`Select` + `OrderByDescending().Take(k)` | `np.exp`, `np.sum`, `np.argsort`/`argpartition`, `argmax` |
| BERT answer span | two `GetMaxValueIndex` loops | `np.argmax(start)`, `np.argmax(end)` |
| Faster R-CNN filter | stride-4 unpack + `if conf >= 0.7` | `boxes.reshape(-1,4)`, mask `scores > 0.7`, fancy index |
| SD VAE decode | per-pixel `clip(x/2+0.5,0,1)*255` | `(np.clip(x/2+0.5, 0, 1) * 255).astype(Byte)` |

So `ortValue.ToNDArray()` is not the end of the story — it is the *entry* to the workload NumSharp is
uniquely good at, and this side carries **no "zero-copy already exists" caveat**: it is pure additive
value. The output-side pitch should lead with **`ToNDArray`/`AsNDArray` + argmax/softmax/top-k/clip**.
Softmax and NMS ship as thin `np`-composition helpers (a `NumSharp.Interop.OnnxRuntime.Postprocess`
static class) or documented one-liners — **not** new kernels; NumSharp already has argmax/argsort/
argpartition/clip/where/boolean-masking. This turns the package from "a bridge" into "the pre- and
post-processing layer for .NET inference".

---

## 6. The dtype map — the seam's coverage

`Microsoft.ML.OnnxRuntime.Tensors.TensorElementType`: `Float=1, UInt8=2, Int8=3, UInt16=4, Int16=5,
Int32=6, Int64=7, String=8, Bool=9, Float16=10, Double=11, UInt32=12, UInt64=13, Complex64=14,
Complex128=15, BFloat16=16`.

| NumSharp `NPTypeCode` | C# | ORT `TensorElementType` | Crossing |
|---|---|---|---|
| Boolean | bool | Bool (9) | **zero-copy** (1 byte both) |
| Byte | byte | UInt8 (2) | **zero-copy** |
| SByte | sbyte | Int8 (3) | **zero-copy** |
| Int16 | short | Int16 (5) | **zero-copy** |
| UInt16 | ushort | UInt16 (4) | **zero-copy** |
| Int32 | int | Int32 (6) | **zero-copy** |
| UInt32 | uint | UInt32 (12) | **zero-copy** |
| Int64 | long | Int64 (7) | **zero-copy** |
| UInt64 | ulong | UInt64 (13) | **zero-copy** |
| Single | float | Float (1) | **zero-copy** |
| Double | double | Double (11) | **zero-copy** |
| Half | System.Half | Float16 (10) | **zero-copy reinterpret** — ORT's `Float16` is a *blittable* struct, binary-identical to `ushort` and to IEEE-754 `System.Half`. No conversion; the pointer crosses as-is. |
| Char | char | UInt16 (4) | **zero-copy** as UTF-16 code units — the same `char→uint16` rule pythonnet uses. (Directional: an ORT UInt16 comes back as UInt16, not Char, unless requested.) |
| Decimal | decimal | — | **no ONNX dtype** → the array-producing verbs convert to Double (lossy beyond ~16 digits), exactly as pythonnet's `ToNumpy` does; the low-level dtype map stays honest and refuses it. |
| Complex | System.Numerics.Complex | Complex64/128 (14/15) | **in ORT's enum but NOT supported** for real tensors/inference → no working path (declined). |
| — | — | BFloat16 (16) | **no NumSharp dtype** (NumSharp has no bfloat16) → out of scope. |
| — | — | String (8) | **no NumSharp dtype**, separate non-unmanaged API → out of scope. |

So the seam is **clean and zero-copy on the entire numeric core — 12 dtypes including Half** — with
Char/Decimal degrading precisely as pythonnet already established, and Complex/BFloat16/String being
genuine mutual gaps documented up front (a `[Misaligned]`-class list). `ToTensorElementType(NPTypeCode)`
and `FromTensorElementType(TensorElementType)` are the two map functions; a refused dtype throws a
message naming the gap and the workaround.

---

## 6.5. Confirmed ORT API surface (verbatim — the ORT ≥ 1.14 `OrtValue` API, checked on 1.29)

Signatures verified against the ORT C# API docs (2026-09-08). Implement against **these**, not the
approximations in §2/§4 — one correction: the pointer factory's parameter is `nint dataBufferPtr`.

**Input factories (three, in preference order):**

```csharp
// zero-copy over a RAW POINTER — the primary AsOrtValue target. Does not own/free; caller keeps valid.
public static OrtValue CreateTensorValueWithData(
    OrtMemoryInfo memInfo, TensorElementType elementType,
    long[] shape, nint dataBufferPtr, long bufferLengthInBytes);

// managed COPY paths — the ToOrtValue / AsDenseTensor targets.
public static OrtValue CreateTensorValueFromMemory<T>(OrtMemoryInfo memoryInfo, Memory<T> memory, long[] shape) where T : unmanaged;
public static OrtValue CreateTensorValueFromMemory<T>(T[] data, long[] shape) where T : unmanaged;
public static OrtValue CreateAllocatedTensorValue(OrtAllocator allocator, TensorElementType elementType, long[] shape);
```

**Run (modern `OrtValue` path — what all current tutorials use):**

```csharp
IDisposableReadOnlyCollection<OrtValue> Run(
    RunOptions options, IReadOnlyDictionary<string, OrtValue> inputs,
    IReadOnlyCollection<string> outputNames);          // inputs is a Dictionary<string, OrtValue>
// + a single-output-name overload; the legacy path is session.Run(IReadOnlyCollection<NamedOnnxValue>)
//   -> IDisposableReadOnlyCollection<DisposableNamedOnnxValue> (the DenseTensor/NamedOnnxValue tier).
```

**Output readers (drive `ToNDArray`/`AsNDArray`):**

```csharp
ReadOnlySpan<T> GetTensorDataAsSpan<T>()       where T : unmanaged;   // ToNDArray copies from this
Span<T>         GetTensorMutableDataAsSpan<T>() where T : unmanaged;   // AsNDArray writes through this
OrtTensorTypeAndShapeInfo GetTensorTypeAndShape();  // .ElementDataType (TensorElementType), .Shape (long[]),
                                                    // .ElementCount (long), .DimensionsCount (int), .IsString
OrtMemoryInfo GetTensorMemoryInfo();
bool          IsTensor { get; }
OnnxValueType OnnxType { get; }                     // Tensor | Sequence | Map | Optional — decline non-Tensor
```

`FromTensorElementType` reads `GetTensorTypeAndShape().ElementDataType`; the produced `NDArray`'s shape
comes from `.Shape` (already `long[]`, C-contiguous). No `int[]`/`long[]` juggling on the read side.

**CPU-memory gate (for the zero-copy `AsNDArray` on an output):** decline non-CPU tensors *before*
touching a span —

```csharp
bool cpu = value.GetTensorMemoryInfo().Equals(OrtMemoryInfo.DefaultInstance);  // OrtCompareMemoryInfo
// fallback discriminators: GetMemoryType(), GetAllocatorType(), GetDeviceMemoryType()
```

A CUDA/DirectML/TensorRT output fails this and routes to the "copy to CPU first, then `ToNDArray`"
message (§5). `ToNDArray` (copy) is unaffected — `GetTensorDataAsSpan` itself throws for non-CPU
memory, so the copy path fails loudly rather than silently.

**Tier-2 metadata (auto-name + dtype/shape validate):** `session.InputNames`, `session.OutputNames`,
and `session.InputMetadata` / `session.OutputMetadata` (each entry exposes `ElementDataType` and
`Dimensions`). Tier-2 uses these to (a) default the single input/output name so the one-arg
`Run(this InferenceSession, NDArray)` overload needs no names, and (b) cast/convert the NDArray to the
model's *declared* input dtype before feeding — e.g. an int32 NDArray into an int64 BERT input, or a
float NDArray into an fp16 model, using `nd.astype(FromTensorElementType(meta.ElementDataType))`.

---

## 7. Lifetime & memory safety (the design)

- **Exports** (`AsOrtValue`/`AsDenseTensor`) take their own atomic ARC reference on the NumSharp buffer
  (`IArraySlice.TryAddRef`), so the memory survives even if the source `NDArray` is disposed/collected
  while ORT holds the tensor. Release is deterministic and owned by the returned handle
  (`OrtTensor.Dispose` / `UnmanagedMemoryManager<T>.Dispose`) — **no `weakref.finalize` needed**,
  because ORT hands us a real `Dispose` hook, unlike Python. While rooted, source `resize` refuses.
- **Imports** (`AsNDArray` on an output) lease ORT's output buffer through NumSharp's memory-block
  refcounting; released when the last NDArray view (including derived slices) dies. The **output
  OrtValue is rooted** and disposed by the lease, so it outlives every view. `ToNDArray` (copy) has no
  lifetime coupling at all — the preferred, foolproof output route.
- **No threading model to fight.** ORT is thread-safe and has no GIL, so there is no `requireGIL`
  parameter, no re-entrancy dance, no engine-shutdown sweep. Disposal ordering (`OrtTensor` before the
  buffer it roots is naturally satisfied because the handle owns both) is the only rule.
- **Handle ownership rule (the one foot-gun):** `AsOrtValue` returns a disposable that owns both the
  `OrtValue` and the ARC ref. Passing `.Value` to `session.Run` and then letting the handle be
  collected mid-`Run` would be a use-after-free — so the verbs return the *handle*, not a bare
  `OrtValue`, and the docs say "keep the handle alive across `Run`" (compiler-nudged by `using`).

---

## 8. API surface (two tiers)

### Tier 1 — the verbs (low-level, full control)

```csharp
namespace NumSharp.Interop.OnnxRuntime
{
    public static partial class NDArrayOnnxInterop
    {
        // NumSharp -> ORT
        public static OrtTensor        AsOrtValue(this NDArray source);                 // zero-copy view (C-contig)
        public static OrtValue         ToOrtValue(this NDArray source);                 // owned copy (any layout)
        public static OrtTensor<T>     AsDenseTensor<T>(this NDArray source) where T : unmanaged;
        public static DenseTensor<T>   ToDenseTensor<T>(this NDArray source) where T : unmanaged;

        // ORT -> NumSharp
        public static NDArray ToNDArray(this OrtValue value);                           // copy (safe default)
        public static NDArray AsNDArray(this OrtValue value, bool allowReadonly = false);// lease-and-root view (CPU only)
        public static NDArray ToNDArray<T>(this DenseTensor<T> t) where T : unmanaged;
        public static NDArray AsNDArray<T>(this DenseTensor<T> t) where T : unmanaged;

        // dtype maps
        public static TensorElementType ToTensorElementType(NPTypeCode code);
        public static NPTypeCode        FromTensorElementType(TensorElementType type);
    }

    public sealed class OrtTensor : IDisposable { public OrtValue Value { get; } /* owns OrtValue + ARC ref */ }
}
```

### Tier 2 — the ergonomic default (`InferenceSession` helpers, hide all lifetime)

```csharp
// One call: build inputs zero-copy, run, copy outputs into owned NDArrays. No OrtValue juggling.
public static NDArray               Run(this InferenceSession s, string inputName, NDArray input,
                                        string outputName);
public static IReadOnlyDictionary<string, NDArray> Run(this InferenceSession s,
                                        IReadOnlyDictionary<string, NDArray> inputs,
                                        IReadOnlyCollection<string> outputNames = null);
```

The Tier-2 helpers own every `OrtTensor`/`OrtValue` inside the call, feed zero-copy inputs, run, and
`ToNDArray`-copy the outputs — so the user never touches ORT lifetime. This is what most #512-style
users actually want.

Tier-2 reads **session metadata** to stay ergonomic and safe (§6.5): with a single model input/output
the name arguments default from `session.InputNames[0]`/`session.OutputNames[0]`; and each input
NDArray is coerced to the model's declared dtype via
`nd.astype(FromTensorElementType(session.InputMetadata[name].ElementDataType))` before feeding, so an
int32 NDArray into an int64 input (BERT) or a float NDArray into an fp16 model just works instead of
throwing a shape/type error from deep inside ORT. A dtype that cannot be coerced, or a shape that does
not match the metadata's fixed dimensions, is rejected with a message naming the model input.

---

## 9. The #512 answer (worked example)

The reported code builds an NCHW `[1,3,H,W]` float tensor from interleaved **BGRA** bytes with a `/255`
scale and channel reorder, one pixel at a time. With this package it becomes vectorized NDArray ops
feeding a zero-copy OrtValue:

```csharp
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

// data: interleaved BGRA bytes, length H*W*4 (B,G,R,A per pixel)
NDArray bgra = np.frombuffer(data, NPTypeCode.Byte).reshape(H, W, 4);

// planar RGB, normalized: gather the R,G,B channels (strided views), stack to CHW, cast+scale.
NDArray chw = np.stack(new[] { bgra[":,:,2"], bgra[":,:,1"], bgra[":,:,0"] }, axis: 0)  // (3,H,W)
                .astype(NPTypeCode.Single) / 255f;                                       // fresh C-contiguous f32
NDArray input = chw.reshape(1, 3, H, W);                                                 // NCHW, still C-contiguous

using OrtTensor t = input.AsOrtValue();          // ZERO COPY — no per-pixel loop, no tensor.Buffer fill
using var results = session.Run(runOptions, new[] { "input" }, new[] { t.Value }, new[] { "output" });
NDArray output = results[0].ToNDArray();         // copy ORT's output into an owned NDArray
```

The scalar `for`-loop is replaced by ~4 vectorized NumSharp ops (SIMD channel slices + a stacking
copy + a fused cast/divide), and the resulting contiguous float buffer is handed to ORT with **no
copy at all**. The one unavoidable copy is the `stack`/`astype` that produces the planar normalized
array — which is genuine work (layout + dtype + scale change), not marshaling overhead.

### 9b. Flagship example — Stable Diffusion, the whole `TensorHelper` deleted

The SD tutorial is where the payoff compounds, because it is *all* tensor math and almost no I/O. Its
hand-written plumbing — the 10-method `TensorHelper` (§2.5), the inline embedding-stack loop,
`torch.cat([latents]*2)`, classifier-free guidance, sigma scaling, the `(2,4,64,64)` split, and the
per-pixel VAE decode — collapses into NDArray ops, and each denoising step feeds the UNet zero-copy:

```csharp
using NumSharp;
using NumSharp.Interop.OnnxRuntime;

// text embeddings (2,77,768): stack uncond + cond          (TensorHelper: an inline nested loop)
NDArray textEmbeddings = np.stack(new[] { uncondEmb, textEmb }, axis: 0);

for (int step = 0; step < timesteps.Length; step++)
{
    // torch.cat([latents] * 2) -> (2,4,64,64)               (TensorHelper.Duplicate)
    NDArray batch = np.concatenate(new[] { latents, latents }, axis: 0);

    // scheduler ScaleInput: latents / sqrt(sigma^2 + 1)     (TensorHelper.Divide / Multiple)
    NDArray scaled = batch / np.sqrt(sigma[step] * sigma[step] + 1f);

    // zero-copy in, copy out — no DenseTensor, no fill loop
    using OrtTensor sIn = scaled.AsOrtValue();
    using OrtTensor eIn = textEmbeddings.AsOrtValue();
    using var r = unet.Run(run, new Dictionary<string, OrtValue> {
        { "sample", sIn.Value }, { "encoder_hidden_states", eIn.Value },
        { "timestep", NDArray.Scalar(timesteps[step]).AsOrtValue().Value } }, unet.OutputNames);
    NDArray noisePred = r[0].ToNDArray();                    // (2,4,64,64)

    // split into uncond / text                              (TensorHelper.SplitTensor)
    NDArray uncondPred = noisePred["0:1"], textPred = noisePred["1:2"];

    // classifier-free guidance: uncond + scale*(text-uncond)  (TensorHelper: Subtract+Multiple+Add)
    NDArray guided = uncondPred + guidanceScale * (textPred - uncondPred);

    latents = scheduler.Step(guided, step, latents);         // scheduler math is more NDArray ops
}

// VAE decode -> image bytes                                 (the per-pixel clip loop)
NDArray vae = DecodeLatents(latents, vaeDecoder);            // one more AsOrtValue/ToNDArray round-trip
NDArray img = (np.clip(vae / 2f + 0.5f, 0f, 1f) * 255f).astype(NPTypeCode.Byte);
```

Every commented line is a `TensorHelper` method or an inline loop in the shipped Microsoft sample. The
fp16 variant (`SharpDiffusion`) exercises the **`Half`→`Float16` zero-copy reinterpret** the dtype map
(§6) gives for free — no conversion, the pointer crosses as-is — which is exactly the case fp16
diffusion models need. This is the plan's strongest demo: it replaces an entire hand-maintained
tensor-math file with the library whose whole purpose is tensor math, and the per-step UNet handoff is
genuinely zero-copy.

---

## 10. Packaging

- **Project:** `src/NumSharp.Interop.OnnxRuntime/NumSharp.Interop.OnnxRuntime.csproj`.
- **Target frameworks:** `net8.0;net10.0` (match the other interop packages).
- **Dependencies:** `Microsoft.ML.OnnxRuntime` (the managed API + CPU EP). Do **not** hard-depend on a
  GPU EP package — a consumer adds `Microsoft.ML.OnnxRuntime.Gpu`/`.DirectML` themselves; the interop
  only touches the CPU-addressable surface. `ProjectReference` to `NumSharp.Core`.
- **`Microsoft.ML.OnnxRuntime` version:** floor at the first release exposing the `OrtValue`
  `CreateTensorValueWithData` + `GetTensorDataAsSpan<T>` API (**1.14**; confirm the exact minor before
  pinning). Range `[1.14.0, )` or a `[x, y)` window per repo convention.
- **Strong naming:** inherits `Open.snk` from the repo-root `Directory.Build.props` (token
  `cc7b13ffcd2ddd51`) like every assembly. No `InternalsVisibleTo` into it is needed; if the seam ever
  reads a NumSharp `internal`, add a keyed friend declaration (never keyless — `CS1726`).
- **No `[ModuleInitializer]`** — this is a conversion library; referencing it changes nothing until a
  verb is called (unlike `NumSharp.Interop.OpenBLAS`, which auto-installs its backend on load).
- **README.md** in the package dir (the pythonnet package's README is the template): the verbs table,
  the dtype map, the lifetime rules, the #512 example.

---

## 11. Testing & gates

Conversion, not numerics — so **no differential-fuzz oracle tier** (there is no NumPy analog to
bit-compare; ORT is not NumPy). The gate is a dedicated interop test project mirroring
`test/NumSharp.Tests.Interop`:

- **Round-trip bit-exactness** — `nd.AsOrtValue().Value.ToNDArray()` and the `DenseTensor` twins are
  byte-identical to `nd` across all **12 supported dtypes** × {0-d, 1-D, 2-D, 3-D, empty, unit}.
- **Zero-copy proof** — mutate through the `OrtValue`'s `GetTensorMutableDataAsSpan` and observe the
  change in the source `NDArray` (and vice-versa for `AsNDArray` output views); assert *no copy* by
  comparing the ORT tensor data pointer to `slice.Address + offset·itemsize`.
- **Contiguity gate** — a sliced/transposed/negative-stride/broadcast/F-order `NDArray` throws from
  `AsOrtValue` with the materialize-first message; `ToOrtValue` copies it correctly.
- **Half reinterpret** — a `Half` NDArray crosses to `Float16` with identical bits (incl. NaN
  payloads / subnormals / ±0), no conversion.
- **Dtype refusals** — Complex, Decimal (low-level map), BFloat16-round-trip refusal texts.
- **Lifetime** — dispose the source `NDArray` while the `OrtTensor` lives → ORT still reads valid data
  (ARC kept it alive); dispose the `OrtTensor` → ARC released, `LiveExports`-style counter returns to 0;
  `resize(refcheck:true)` on a rooted source refuses.
- **Output lease** — `AsNDArray` over an output view holds the OrtValue alive until the last derived
  slice dies; disposing the OrtValue underneath a live view is refused/guarded.
- **GPU decline** — a CUDA/DML output OrtValue is declined by `AsNDArray` with the CPU-only message
  (skipped when no GPU EP is present).
- **End-to-end** — a tiny real `.onnx` model (identity / add) run through the Tier-2 `Run` helper with
  `NDArray` in and out.

Category: reuse the interop test conventions; live-ORT tests skip cleanly when the native ORT binaries
are unavailable on the CI leg (as the pythonnet tests skip without a Python).

---

## 12. Non-goals / known gaps (state up front)

- **Complex128/64** — in ORT's type enum but unsupported for real tensors; declined, not silently
  copied.
- **BFloat16, String, float8, 4-bit** ORT types — no NumSharp dtype; out of scope.
- **GPU-resident tensors** — cannot be viewed as `NDArray`s (not CPU-addressable); copy to CPU first.
- **Non-contiguous zero-copy** — impossible (ORT has no strides); F-order shares only via
  `DenseTensor(reverseStride:true)`. Everything else densifies (a copy).
- **`ndarray` subclasses / structured / object dtypes** — NumSharp has none; N/A.

---

## 13. Open decisions (to settle before coding)

1. **Handle vs. bare `OrtValue`.** Return the `OrtTensor` handle (safe, `using`-nudged) vs. a bare
   `OrtValue` with the ARC ref tracked in a side table keyed on the OrtValue (ergonomic but a
   collect-mid-`Run` foot-gun). **Recommendation:** the handle.
2. **Output default — copy vs. view.** `ToNDArray` (copy) as the documented default, `AsNDArray` (view)
   as the opt-in — because an output view's OrtValue lifetime is easy to get wrong. **Recommendation:**
   copy-default.
3. **Char direction.** Export `Char`→`UInt16` always; on import, `UInt16`→`UInt16` (not `Char`) unless
   an explicit `asChar` is requested — matches pythonnet's asymmetry. **Recommendation:** yes.
4. **Tier-2 `Run` surface breadth.** Single-in/single-out + dictionary is enough for #512; a full
   `IOBinding`/`RunOptions` passthrough can wait. **Recommendation:** ship the two, defer the rest.
5. **ORT version floor.** Confirm the exact `Microsoft.ML.OnnxRuntime` minor that first ships
   `CreateTensorValueWithData` + `GetTensorDataAsSpan<T>` (≈1.14) and pin it.
6. **Which `DenseTensor`?** There are now two tensor libraries: the classic
   `Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<T>` (what the tutorials + `NamedOnnxValue` use) and
   the newer `System.Numerics.Tensors` (`Tensor<T>`, imported in the get-started page, .NET 9+). Target
   the **ORT `DenseTensor<T>`** for the legacy tier (it is what `session.Run(NamedOnnxValue…)` consumes);
   decide whether to *also* offer a `System.Numerics.Tensors.Tensor<T>` bridge, or defer it until ORT's
   own migration settles. **Recommendation:** ORT `DenseTensor` now, `System.Numerics.Tensors` deferred.
7. **Ship post-processing helpers, or document one-liners?** §5c makes the output side a headline. A
   tiny `Postprocess` static class (`Softmax`, `TopK`, `Nms`, `Argmax`) is thin `np`-composition and a
   big ergonomics win, but it is scope beyond pure conversion. **Recommendation:** ship `Softmax`/`TopK`/
   `Argmax` (pure `np` composition, no kernels), document `Nms`/box-decode as examples; keep them in a
   separate `Postprocess` type so the core verbs stay conversion-only.
8. **BFloat16 priority.** BFloat16 is a genuine mutual gap (no NumSharp dtype) and increasingly common
   in modern models. It is out of scope for v1, but likely the first post-ship ask. **Recommendation:**
   note it explicitly in the README's gap list and track as a follow-up (it needs a NumSharp bf16 dtype,
   not interop work — so it is a Core issue the package would consume, not own).
9. **Tier-2 dtype auto-coercion (§6.5).** Should Tier-2 silently `astype` an NDArray to the model's
   declared input dtype, or require an exact match? Silent coercion is what "most #512-style users
   actually want" (int32→int64, float→fp16) but hides a class of bugs. **Recommendation:** coerce with a
   *widening/safe* cast silently, require opt-in for a *lossy* one (float64→fp16), mirroring NumSharp's
   own casting rules.

---

### References
- Issue: https://github.com/SciSharp/NumSharp/issues/512
- `OrtValue` API: https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtValue.html
- `OrtTensorTypeAndShapeInfo` API: https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtTensorTypeAndShapeInfo.html
- `OrtMemoryInfo` API: https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.OrtMemoryInfo.html
- `DenseTensor<T>` API: https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.Tensors.DenseTensor-1.html
- `TensorElementType` / `Float16`: `microsoft/onnxruntime` `csharp/src/Microsoft.ML.OnnxRuntime/Tensors/Tensor.shared.cs`
- C# tutorials (read for §2.5/§5c/§6.5/§9b): https://onnxruntime.ai/docs/tutorials/csharp/
  — get-started: https://onnxruntime.ai/docs/get-started/with-csharp.html
  — ResNet50v2 (preprocess loop + softmax): https://onnxruntime.ai/docs/tutorials/csharp/resnet50_csharp.html
  — Faster R-CNN (pad math + box filter): https://onnxruntime.ai/docs/tutorials/csharp/fasterrcnn_csharp.html
  — BERT (int64 inputs + argmax): https://onnxruntime.ai/docs/tutorials/csharp/bert-nlp-csharp-console-app.html
  — Stable Diffusion (the `TensorHelper` case): https://onnxruntime.ai/docs/tutorials/csharp/stable-diffusion-csharp.html
- `TensorHelper.cs` (the hand-rolled NumSharp, copy-pasted across repos incl. Microsoft's own):
  `microsoft/ai-dev-gallery` `AIDevGallery/Samples/SharedCode/StableDiffusionCode/TensorHelper.cs`,
  `cassiebreviu/StableDiffusion` `StableDiffusion.ML.OnnxRuntime/TensorHelper.cs`
- House model to copy: `src/NumSharp.Interop.pythonnet/NDArrayPythonInterop.Export.cs`, `.../README.md`
