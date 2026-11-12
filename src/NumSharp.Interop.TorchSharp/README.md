# NumSharp.Interop.TorchSharp

Interop between NumSharp's `NDArray` and **TorchSharp** tensors (.NET bindings for libtorch, the engine under PyTorch).
`nd.AsTensor()` hands a NumSharp buffer to libtorch without copying, and `tensor.AsNDArray()` views a tensor's storage as an
`NDArray`. Each verb takes a `TorchCodecMode`: **Auto** shares when it can and copies otherwise, **View** shares or throws,
**Copy** always gives an independent result. These are the pythonnet codec's `NumpyCodecMode` and the ONNX Runtime bridge's
`OnnxCodecMode`, which are NumPy's and PyTorch's `copy=None` / `copy=False` / `copy=True`.

It is a **conversion library**: no `TensorEngine` seam, no `[ModuleInitializer]`, no native asset of its own. It binds
TorchSharp's own `LibTorchSharp` library. Referencing the package changes nothing until you call a verb.

The package is **unreleased**: it builds and is tested in CI, but no NuGet package is published yet.

```csharp
using NumSharp;
using NumSharp.Interop.TorchSharp;
using TorchSharp;
using static TorchSharp.torch;

var batch = np.random.rand(64, 3, 32, 32).astype(np.float32);   // NumSharp preprocessing
using var x = batch.AsTensor();                                  // zero-copy: x is a view of batch
using var y = nn.functional.relu(x);                             // libtorch computes
var result = y.AsNDArray();                                      // zero-copy: a view of y's storage
```

## Verbs

| Verb | Direction | What it returns |
|------|-----------|-----------------|
| `nd.AsTensor(mode = Auto, dtype = null, casting = "same_kind")` | NumSharp → torch | a CPU `Tensor` over the array's own buffer when torch can read it in place; otherwise a copy (Auto) or `InvalidOperationException` (View) |
| `nd.ToTensor(dtype = null, casting = "same_kind")` | NumSharp → torch | `AsTensor(Copy)`: an independent C-contiguous tensor |
| `tensor.AsNDArray(mode = Auto)` | torch → NumSharp | a view of a CPU tensor's storage; otherwise a copy (Auto) or `InvalidOperationException` (View) |
| `tensor.ToNDArray()` | torch → NumSharp | `AsNDArray(Copy)`: an owning, C-contiguous array |
| `nd.MayShareMemory(tensor)` | both | whether the two byte ranges overlap (NumPy's `np.may_share_memory`): how to tell whether Auto viewed or copied |
| `NDArrayTorchInterop.ToScalarType(dtype)` / `ToDType(scalarType)` | maps | the dtype each side crosses as |
| `NDArrayTorchInterop.LiveExports` / `LiveImports` | diagnostics | buffers lent to libtorch / torch storages held by NumSharp views; a steady rise is a leak |

## When a crossing shares, and when it copies

**NumSharp → torch.** The array's buffer becomes the tensor's storage when all of these hold:

- the dtype has a torch counterpart bit for bit and no `dtype` conversion was asked for;
- every stride is non-negative;
- the data is element-aligned;
- the array is writeable.

A C-contiguous array goes through libtorch's `from_blob`. Any other non-negative-stride layout (sliced, stepped, transposed,
Fortran-ordered) goes through `from_blob` over the elements it reaches, plus `as_strided`. A **broadcast view of writeable
memory** crosses as torch's own **`expand`** of the memory under it:

- it is zero-copy, with stride 0;
- libtorch refuses an in-place op over the repeated lanes (`t.add_(1)`);
- a write to one lane (`t[0].fill_(7)`) changes the one row the broadcast repeats.

The array crosses as a copy under Auto, and is refused under View, when it is:

- **reversed** (torch has no negative strides);
- **read-only**: a NumPy scalar, read-only foreign memory, or a broadcast of either (torch has no read-only tensors);
- **misaligned**;
- **converted** to another dtype.

**torch → NumSharp.** A CPU tensor with strided storage, a dtype NumSharp has and no pending conjugate or negative bit is
viewed in place, with torch's strides (transposed, narrowed, overlapping). An expanded (stride-0) tensor becomes a read-only
broadcast view, NumPy's rule.

These tensors cross as a copy under Auto, and are refused under View:

- a tensor on another device, brought over with `cpu()`;
- a sparse tensor, brought over with `to_dense()`;
- a conjugated or negated view, resolved first;
- a complex64 or bfloat16 tensor, widened exactly.

Under Auto, the private CPU copy of a CUDA tensor (or the resolved copy of a conjugated one) is viewed, not copied a second
time.

**A tensor that requires grad is viewed read-only.** No TorchSharp or LibTorchSharp API bumps autograd's version counter,
and NumSharp's kernels write memory directly, so a write through NumSharp would corrupt a backward pass silently. NumSharp
refuses such a write instead. `t.detach().AsNDArray()` is the writeable route, and Copy gives an independent array.

## Dtypes

| NumSharp | torch | |
|----------|-------|-|
| bool, uint8, int8, int16, int32, int64, float16, float32, float64, complex128 | Bool, Byte, Int8, Int16, Int32, Int64, Float16, Float32, Float64, ComplexFloat64 | bit for bit, both ways |
| uint16, char | Int32 | NEP 50: `np.promote_types(dtype, np.int8)`; a copy |
| uint32 | Int64 | NEP 50; a copy |
| uint64 | Float64 | NEP 50 (exact up to 2^53); a copy. `AsTensor(dtype: ScalarType.Int64)` downcasts instead (values from 2^63 wrap, as `astype(np.int64)` does) |
| decimal | Float64 | the bridges' rule (rounded to 53 bits); a copy |
| complex128 | ComplexFloat32 | import only: widened exactly from complex64 |
| float32 | BFloat16 | import only: widened exactly (the 16 bits shifted into the top half) |
| uint16, uint32, uint64 | libtorch's own uint16/uint32/uint64 (c10 codes 27-29, unnamed in TorchSharp's `ScalarType`) | import only, viewed in place |
| text (`U`, `S`, `StringDType`) | none | refused in every mode (torch has no strings) |
| none | QInt8, QUInt8, QInt32 | refused in every mode (`dequantize()` or `int_repr()` first) |

An explicit `dtype` on `AsTensor` converts through NumSharp's `np.copyto` and its `casting` check. The default is
`"same_kind"`: float64 to float32 is allowed, float64 to int32 throws NumPy's `InvalidCastException`, and
`casting: "unsafe"` allows it.

## Lifetime

- **An exported tensor** holds its own reference on the NumSharp buffer. libtorch releases it, through the deleter
  `from_blob` was given, when the last tensor over that storage dies: this one and every torch view of it, whether
  disposed, collected by a `DisposeScope`, or finalized. The array may be disposed meanwhile, and `resize` refuses to move
  a buffer torch still reads. The tensor joins the thread's current `DisposeScope` like any tensor TorchSharp makes. An
  undisposed torch view (an indexing temporary such as `t[0, 0].item<float>()`) keeps the buffer until the GC finalizes it.
- **An imported view** holds its own `detach()` of the tensor: a separate `TensorImpl` over the same storage, outside every
  `DisposeScope`. It is released when the last NumSharp view over it, derived slices included, is disposed or finalized.
  The caller may dispose the tensor, or `set_` it to other storage, while NumSharp reads. The view books the storage's
  bytes with NumSharp's memory accounting, so forgotten views pace a collection.
- **What no reference covers:** an in-place resize of the tensor's storage (libtorch's `resize_`) frees the memory a view
  reads. PyTorch's own `numpy()` marks the storage non-resizable for that reason, and TorchSharp offers no way to do so.
- **Threads:** TorchSharp's `Tensor.Dispose` takes no lock. Disposing a tensor on one thread while a verb reads it on
  another is the caller's race, as with any TorchSharp call. libtorch may release an exported buffer on any thread.

## TorchSharp versions

The package depends on the managed **`TorchSharp`** package only, version **0.100.0 or later** with no upper bound. The
consumer picks the libtorch flavour: `TorchSharp-cpu`, `TorchSharp-cuda-windows`, `TorchSharp-cuda-linux`, or a
`libtorch-*` package. The package compiles against 0.100.0 and is tested against 0.100.0 and 0.107.0. Two differences
between those releases shape the code:

- every `Tensor.to(...)` overload gained parameters, so a binary built against 0.100.0 cannot call `to` on 0.107.0. The
  package therefore converts dtypes in NumSharp, and brings a device's values over with `cpu()`. `nd.AsTensor().cuda()` is
  the way to put an array on the GPU.
- the native `THSTensor_new` (libtorch's `from_blob`) took 6 arguments in 0.100.x and takes 9 from 0.101.0 on. The package
  reads which form the loaded TorchSharp declares and calls that one.

TorchSharp 0.100.0 brings SkiaSharp 2.88.3, which NuGet's audit flags (NU1903). Any current TorchSharp brings 2.88.6.
