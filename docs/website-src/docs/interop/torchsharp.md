# TorchSharp - hand NumSharp arrays to libtorch, read tensors back, without copying

`NumSharp.Interop.TorchSharp` connects `NDArray` to [TorchSharp](https://github.com/dotnet/TorchSharp)'s
`torch.Tensor` - the .NET bindings for libtorch, the engine under PyTorch - with no Python anywhere. An array
becomes a CPU tensor over the very same unmanaged bytes: any non-negative-stride layout (contiguous, sliced,
stepped, transposed, Fortran-ordered), and a broadcast view as torch's own `expand`. A CPU tensor comes back as an
`NDArray` view of its storage, with torch's strides. Each verb takes a `TorchCodecMode`: **Auto** shares when it can
and copies otherwise, **View** shares or throws, **Copy** is always independent. It is a **conversion library**: no
`TensorEngine` seam, no `[ModuleInitializer]`, no native asset of its own - it binds the `LibTorchSharp` library
TorchSharp already ships. This page is its reference: setup, the modes and verbs, what shares and what copies in each
direction, who frees what, dtypes and versions.

**On this page:** [From zero](#from-zero) · [Views, copies and modes](#views-copies-and-torchcodecmode) ·
[The verbs](#the-verbs) · [NumSharp → torch](#numsharp--torch) · [torch → NumSharp](#torch--numsharp) ·
[Lifetime](#lifetime-who-frees-what) · [Dtypes](#dtypes) · [Versions](#torchsharp-versions) · [Limits](#limits) ·
[Not a compute backend](#not-a-compute-backend) · [Claims](#claims-ledger)

> Verified on TorchSharp 0.100.0 (the floor, libtorch 2.0.1) and 0.107.0 (libtorch 2.10.0) · net8.0/net10.0.
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop.TorchSharp` (222 tests). The layout and flag
> tables are held to NumPy 2.5.4 (the pinned release) and PyTorch 2.13.0 through a committed fixture, so no Python runs at
> test time.

---

## From zero

```bash
dotnet add package NumSharp.Interop.TorchSharp
dotnet add package TorchSharp-cpu        # libtorch for the CPU (or one TorchSharp-cuda-* package)
```

The bridge depends only on the managed `TorchSharp` package - the C# API and its small `LibTorchSharp` shim. libtorch
itself ships as mutually exclusive packages (`TorchSharp-cpu`, `TorchSharp-cuda-windows`, `TorchSharp-cuda-linux`, or a
`libtorch-*` package directly), and you reference exactly one: TorchSharp's own build refuses two. The bridge composes
with whichever you pick.

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

`x` is a tensor over `batch`'s own buffer, so libtorch reads NumSharp's memory in place. `result` is a NumSharp view of
the storage libtorch allocated for `y`, and it stays valid after `y` is disposed: the view holds its own reference to
that storage.

## Views, copies and `TorchCodecMode`

Every verb that could share memory takes a `TorchCodecMode`, with the member names and meanings of the
[pythonnet bridge](pythonnet-numpy.md)'s `NumpyCodecMode` and the [ONNX Runtime bridge](onnxruntime.md)'s
`OnnxCodecMode` - NumPy's and PyTorch's `copy=None` / `copy=False` / `copy=True`:

| Mode | What it does |
|---|---|
| `Auto` (default) | A view when the layout and the dtype permit it, an independent copy otherwise. It never fails because of a layout or a convertible dtype. |
| `View` | Always a view. A crossing that needs a copy throws `InvalidOperationException`, and the message names what prevents the view. |
| `Copy` | Always an independent copy. |

```csharp
NDArray a = np.arange(12).reshape(3, 4).astype(np.float32);

using Tensor shared = a.AsTensor();                     // Auto: a tensor over a's own buffer
using Tensor copy   = a.ToTensor();                     // always an independent copy
// a.MayShareMemory(shared) == true, a.MayShareMemory(copy) == false

NDArray reversed = a["::-1"];                           // a negative stride, which torch has no form for
using Tensor r = reversed.AsTensor();                   // Auto: a copy, in reversed order
reversed.AsTensor(TorchCodecMode.View);                 // throws InvalidOperationException:
// AsTensor(View): the array has a negative stride (a reversed view such as a[::-1]), and torch has no
// negative strides. Auto and Copy cross it as a copy.
```

Writes go both ways through a view, and never through a copy. `nd.MayShareMemory(tensor)` - NumPy's
`np.may_share_memory` test - tells which one Auto returned: a view overlaps its source, a copy never does.

## The verbs

Everything is packaging over four operations on the static `NDArrayTorchInterop` class (also extension methods).
**`As…` takes a mode and shares when it can, `To…` is its Copy form.**

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → torch | `nd.AsTensor(mode = Auto, dtype = null, casting = "same_kind")` | a CPU `Tensor` over the array's own buffer when torch can read it in place; otherwise a copy (Auto) or `InvalidOperationException` (View). `dtype` asks for a conversion, which is always a copy, checked by NumPy's `casting` rule |
| NumSharp → torch | `nd.ToTensor(dtype = null, casting = "same_kind")` | `AsTensor(Copy)`: an independent C-contiguous tensor; any layout, read in logical order |
| torch → NumSharp | `tensor.AsNDArray(mode = Auto)` | a view of a CPU tensor's storage with torch's strides (read-only when the tensor requires grad); otherwise a copy (Auto) or `InvalidOperationException` (View) |
| torch → NumSharp | `tensor.ToNDArray()` | `AsNDArray(Copy)`: an owning, C-contiguous array; any device, sparse, conjugated, complex64 and bfloat16 tensors included |

Plus `nd.MayShareMemory(tensor)`, the dtype maps `NDArrayTorchInterop.ToScalarType(dtype)` (the torch dtype an array
crosses as) and `ToDType(scalarType)` (the NumSharp dtype a tensor imports as), and the counters
`NDArrayTorchInterop.LiveExports` / `LiveImports`: the NumSharp buffers libtorch still holds, and the torch storages
NumSharp views still hold. A count that stays above its baseline after everything was disposed is a leak.

```csharp
using Tensor t  = nd.AsTensor(TorchCodecMode.View);          // ZERO COPY, or it throws
using Tensor t2 = nd.ToTensor(dtype: ScalarType.Float32);    // a converted copy (casting "same_kind")
NDArray v = tensor.AsNDArray();                              // a view of a CPU tensor's storage
NDArray c = tensor.ToNDArray();                              // an owning copy
```

## NumSharp → torch

The array's buffer becomes the tensor's storage when four things hold: the dtype is one of the ten torch shares bit for
bit and no `dtype` conversion was asked for, every stride is non-negative, the data is element-aligned, and the array is
writeable - or a broadcast of writeable memory. A C-contiguous array is one libtorch `from_blob`. Any other
non-negative-stride layout is `from_blob` over the elements it reaches plus `as_strided`, so the tensor has exactly
NumPy's strides. A broadcast view of writeable memory is the memory under it plus torch's own `expand`.

The table holds every layout of the state matrix: 45 array states, each over the ten shared dtypes, built the same way
in NumPy and in NumSharp. NumSharp's own flags (C, F, OWNDATA, WRITEABLE, ALIGNED) and strides equal NumPy's in all 386
cases. The last column is what `torch.from_numpy` does with NumPy's array.

| The array | Auto | View | `torch.from_numpy` |
|---|---|---|---|
| C-contiguous (1-D to rank 8), 0-d, one element, a contiguous slice at an offset, a reshape view, `np.frombuffer` over writeable memory, an `np.asmatrix`, a byte-reinterpreting `view(dtype)` | shares (`from_blob`) | shares | shares |
| F-contiguous (an owner or a view), transposed, permuted, stepped (rows, columns, at an offset), overlapping windows (an `np.ndarray` whose strides revisit elements), a complex array's `.real` / `.imag` | shares, with NumPy's strides (`from_blob` + `as_strided`) | shares | shares |
| A broadcast of writeable memory (`np.broadcast_to`, a row or a column) | shares as torch's `expand` (stride 0) | shares | shares |
| A singleton axis, `np.newaxis`, or a negative stride on a length-1 axis | shares: a length-1 axis's stride never matters | shares | shares, but **refuses** the negative stride |
| Reversed: a negative stride on any longer axis (`a[::-1]`) | copies, in logical order | throws | refuses: torch has no negative strides |
| Read-only: a NumPy scalar, read-only foreign memory, `writeable` turned off, a view of any of them | copies | throws | shares, warning that writes are undefined behavior |
| A broadcast of read-only memory | copies | throws | shares |
| Misaligned: a byte-reinterpreting view at an odd offset; complex128 at 8 mod 16 | copies | throws | shares |
| Empty (`(0, 3)`, an empty view, an empty step) | crosses with its shape | crosses with its shape | an empty tensor |
| `uint16`, `uint32`, `uint64`, `char`, `decimal` | copies, NEP 50's dtype ([Dtypes](#dtypes)) | throws | `uint16` / `uint32` / `uint64` share as torch's own unsigned dtypes (PyTorch 2.3 and later) |

Where the bridge differs from `torch.from_numpy`, it is on purpose:

- **Torch has no read-only tensors.** `from_numpy` shares a read-only array anyway and warns that a write is undefined
  behavior. The bridge copies instead, so no torch operation can write NumSharp memory that NumSharp declared read-only.
  A broadcast of *writeable* memory is read-only in NumSharp only because its lanes overlap (NumPy's rule), so it does
  share - as torch's own `expand`, whose overlap checks libtorch applies itself: an in-place `t.add_(1)` is refused, and
  a write to one lane (`t[0].fill_(7)`) changes the one row the broadcast repeats.
- **libtorch's kernels assume element alignment** (`c10::complex<double>` is declared `alignas(16)`, so the compiler may
  use aligned 16-byte loads). NumSharp's own buffers are aligned; only a byte-reinterpreting view at an odd offset is
  not, and it copies.
- **A length-1 axis is never stepped**, so its stride is normalized rather than refused.

The copy is a fresh NumSharp buffer filled by `np.copyto(…, casting)` and lent to libtorch the same way as a view, so it
is freed when the tensor is, and nothing else holds it. A device is the caller's next call: `nd.AsTensor().cuda()`.

```csharp
NDArray row  = np.arange(5).astype(np.float32);
NDArray grid = np.broadcast_to(row, new Shape(4, 5));     // NumSharp: stride 0, read-only
using Tensor g = grid.AsTensor();                         // torch's expand: no copy, strides (0, 1)
// libtorch refuses g.add_(1) (overlapping lanes); g[0].fill_(7) writes the one row all four repeat
```

## torch → NumSharp

A CPU tensor with strided storage, a dtype NumSharp has, and no pending conjugation or negation is viewed in place with
torch's strides, writeable both ways. The table holds the 24 tensor states of the state matrix; the last column is what
PyTorch's own `t.numpy()` does.

| The tensor | Auto | View | `t.numpy()` |
|---|---|---|---|
| Contiguous, transposed, permuted, narrowed, stepped, a `select`, `unsqueeze`, channels-last, overlapping `as_strided`, a length-1 axis with any stride, rank 8, 0-d | a view with torch's strides and NumPy's flags for them | a view | shares |
| Expanded (stride 0) | a **read-only** broadcast view | a read-only view | shares, writeable |
| Requires grad (every `nn.Parameter`, and what is computed from one) | a **read-only** view | a read-only view | refuses; `numpy(force=True)` shares |
| A pending conjugation (`conj()`) or negation (`conj().imag`) | a copy, resolved | throws | refuses; `numpy(force=True)` copies |
| complex64, bfloat16 | a copy, widened exactly to complex128 / float32 | throws | complex64 shares; bfloat16 refuses |
| On another device (CUDA, MPS, ...) | a copy through `cpu()` | throws | refuses; `numpy(force=True)` copies |
| Sparse | a copy through `to_dense()` | throws | refuses |
| Empty | crosses with its shape | crosses with its shape | an empty array |
| Quantized, meta | refused in every mode | refused | refuses |

Where the bridge differs from `t.numpy()`:

- **A tensor that requires grad is viewed read-only.** No TorchSharp or LibTorchSharp API bumps autograd's version
  counter, and NumSharp's kernels write memory directly, so a write through NumSharp would corrupt a backward pass
  silently. NumSharp refuses such a write instead. `t.detach().AsNDArray()` is the writeable route - you have opted out
  of autograd - and Copy gives an independent array.
- **An expanded tensor is a read-only broadcast view**, NumPy's rule for stride-0 lanes: NumSharp marks every broadcast
  read-only, whoever made it.
- **NumSharp has no complex64 or bfloat16**, so those widen, exactly: complex64's float pairs become complex128, and
  bfloat16's 16 bits move into the top half of a float32. The values are the tensor's, bit for bit.
- **Auto never copies twice.** The private CPU copy of a CUDA tensor, or the resolved copy of a conjugated one, is itself
  viewed rather than copied again; Copy always returns an owning, C-contiguous array.

```csharp
using Tensor t  = torch.arange(12, dtype: ScalarType.Float32).reshape(3, 4);
using Tensor tt = t.t();                                  // a transposed torch view, strides (1, 4)
using NDArray v = tt.AsNDArray();                         // a NumSharp view: F-contiguous, like NumPy's

using Tensor w = torch.randn(3, 3, requires_grad: true);
using NDArray wv = w.AsNDArray();                         // read-only: wv.flags.writeable == false
using Tensor wd = w.detach();
using NDArray wdv = wd.AsNDArray();                       // writeable: autograd is out of the picture
```

## Lifetime: who frees what

- **An exported tensor holds its own reference on the NumSharp buffer.** libtorch releases it, through the deleter
  `from_blob` was given, when the last tensor over that storage dies: this one and every torch view of it, whether
  disposed, collected by a `DisposeScope`, or finalized. The array may be disposed meanwhile, and `nd.resize(...)`
  refuses to move a buffer torch still reads. The tensor joins the thread's current `DisposeScope` like any tensor
  TorchSharp makes. An undisposed torch view - an indexing temporary such as `t[0, 0].item<float>()` - keeps the buffer
  until the GC finalizes it, so use a `DisposeScope` (`torch.NewDisposeScope()`) around code that indexes.
- **An imported view holds its own `detach()` of the tensor**: a separate `TensorImpl` over the same storage, outside
  every `DisposeScope`. It is released when the last NumSharp view over it - derived slices included - is disposed or
  finalized. You may dispose the tensor, close its `DisposeScope`, or `set_` it to other storage while NumSharp reads:
  the view keeps reading the storage it was made over. The view books the storage's bytes with NumSharp's memory
  accounting, so forgotten views pace a collection.
- **What no reference covers:** an in-place resize of an imported tensor's storage (libtorch's `resize_`) frees the
  memory a view reads. PyTorch's own `numpy()` marks the storage non-resizable for that reason; TorchSharp offers no way
  to. Do not `resize_` a tensor whose storage a NumSharp view reads.
- **Threads.** TorchSharp's `Tensor.Dispose` takes no lock: disposing a tensor on one thread while a verb reads it on
  another is the caller's race, as with any TorchSharp call. The bridge's own leases are thread-safe, and libtorch may
  release an exported buffer on any thread, the GC's finalizer thread included.

## Dtypes

| NumSharp | torch | Crossing |
|---|---|---|
| bool, uint8, int8, int16, int32, int64, float16, float32, float64, complex128 | Bool, Byte, Int8, Int16, Int32, Int64, Float16, Float32, Float64, ComplexFloat64 | bit for bit, both ways |
| uint16, char | Int32 | NEP 50: `np.promote_types(dtype, np.int8)`; a copy |
| uint32 | Int64 | NEP 50; a copy |
| uint64 | Float64 | NEP 50 (exact up to 2^53); a copy. `AsTensor(dtype: ScalarType.Int64)` downcasts instead (values from 2^63 wrap, as `astype(np.int64)` does) |
| decimal | Float64 | the bridges' rule (rounded to 53 bits); a copy |
| complex128 | ComplexFloat32 | import only: widened exactly from complex64 |
| float32 | BFloat16 | import only: widened exactly (the 16 bits shifted into the top half) |
| uint16, uint32, uint64 | libtorch's own uint16 / uint32 / uint64 (libtorch 2.3 and later; TorchSharp's `ScalarType` does not name them) | import only, viewed in place |
| text (`U`, `S`, `StringDType`) | none | refused in every mode: torch has no strings |
| none | QInt8, QUInt8, QInt32 | refused in every mode: choose `dequantize()` or `int_repr()` first |

An explicit `dtype` on `AsTensor` / `ToTensor` converts through NumSharp's `np.copyto` and its `casting` check, so the
ten shared dtypes are the targets. The default rule is `"same_kind"`: float64 to float32 is allowed, float64 to int32
throws NumPy's `InvalidCastException`, and `casting: "unsafe"` allows it. BFloat16 and complex64 are not targets NumSharp
can write: cross the array as it is and convert on the torch side (`t.to_type(ScalarType.BFloat16)`).

```csharp
NDArray ids = np.array(new ulong[] { 1, (1UL << 63) + 5 });
using Tensor f = ids.AsTensor();                          // Float64: NEP 50 promotes uint64 with int8
using Tensor i = ids.AsTensor(dtype: ScalarType.Int64);   // Int64 on request: 2^63 + 5 wraps to -2^63 + 5
```

## TorchSharp versions

The package depends on the managed `TorchSharp` package **0.100.0 or later, with no upper bound**. 0.100.0 (May 2023,
libtorch 2.0.1) is the oldest release exposing every member the bridge calls. The package compiles against it, and the
suite runs against 0.100.0 and 0.107.0, so the current release is the binary-compatibility check for the floor-built
package. There is no ceiling because a ceiling below your TorchSharp would be a restore error (NU1107) the day you
upgrade. Two differences between those releases shape the code:

- every `Tensor.to(...)` overload gained parameters, so a binary built against 0.100.0 cannot call `to` on 0.107.0. The
  bridge therefore converts dtypes in NumSharp, and brings a device's values over with `cpu()`, which never changed.
  That is also why `AsTensor` has no `device` parameter: `nd.AsTensor().cuda()` puts an array on the GPU.
- the native `THSTensor_new` (libtorch's `from_blob` behind TorchSharp) took 6 arguments in 0.100.x and takes 9 from
  0.101.0 on. The bridge reads which form the loaded TorchSharp declares and calls that one; a form it does not know is
  refused with `NotSupportedException` in every mode, before anything is lent.

TorchSharp 0.100.0 brings SkiaSharp 2.88.3, which NuGet's audit flags (NU1903); any current TorchSharp brings 2.88.6.
TorchSharp 0.100.0 ships no osx-arm64 natives.

## Limits

- **CPU memory only.** A view needs memory the CPU can address: a CUDA or MPS tensor crosses as a copy through `cpu()`
  (Auto, Copy) and is refused under View. In the other direction the tensor is a CPU tensor; move it yourself.
- **No negative strides in torch**, so a reversed array always copies.
- **No read-only tensors in torch**, so a read-only array copies (Auto) or is refused (View) - except a broadcast of
  writeable memory, which crosses as `expand`.
- **The storage of an imported tensor must not be resized** (`resize_`) while a NumSharp view reads it.
- **Text and quantized tensors cross in no mode**; a meta tensor holds no values; complex32 and the float8 family have
  no NumSharp dtype even widened.

## Not a compute backend

This bridge moves *data*, not *computation*. NumSharp's operations stay NumSharp's own kernels, bit-exact with NumPy, and
libtorch's operations stay libtorch's: the bridge only lets each read the other's memory. That is why there is no
`TensorEngine` seam here - use it to hand a NumSharp buffer to a TorchSharp model, and to read a model's tensors back
into `np.*`.

## Claims ledger

| # | Claim | Gate |
|---|---|---|
| 1 | The ten shared dtypes map both ways; the five others follow NEP 50; torch-only dtypes import widened or as libtorch's unsigned dtypes; text, quantized and unknown dtypes are refused | [`DtypeMapTests`][gate] |
| 2 | The shared dtypes view in Auto and View and copy in Copy; a view writes through both ways and a copy does not; converted dtypes cross as NEP 50 copies and View refuses them | [`ExportTests`][gate] |
| 3 | An explicit `dtype` converts with NumPy's casting rule; uint64 downcasts to int64 on request; a torch dtype the bridge cannot write names the ten it can | [`ExportTests`][gate], [`ExplicitDtypeTests`][gate] |
| 4 | Every non-negative-stride layout shares; a strided view writes the right elements; negative strides copy in Auto and throw in View; a broadcast of writeable memory crosses as `expand` and one of read-only memory copies; strided views that end at the buffer's last element read in bounds | [`ExportLayoutTests`][gate] |
| 5 | For 45 array states × the ten shared dtypes × the three modes: NumSharp's flags and strides are NumPy's for the same recipe; View shares or throws for the documented reason; a shared tensor has `torch.from_numpy`'s strides and contiguity; the bits match; writes reach the other side in both directions; the crossing leaves the array's flags unchanged | [`StateMatrixExportTests`][gate] |
| 6 | A tensor view has torch's strides; an expanded tensor is a read-only broadcast view; a tensor that requires grad is a read-only view; conjugated and negated tensors copy resolved and View throws; complex64 and bfloat16 widen exactly; sparse tensors densify; meta and quantized tensors are refused | [`ImportTests`][gate] |
| 7 | For 24 tensor states × their dtypes × the three modes: a view has `t.numpy()`'s strides and NumPy's flags for them, a copy is a fresh C-contiguous array, the bits are the tensor's (or its exact widening) | [`StateMatrixImportTests`][gate] |
| 8 | NumSharp → torch → NumSharp in Auto returns the values, in the same memory for a shared state and in independent memory for a copied one | [`StateMatrixRoundTripTests`][gate] |
| 9 | uint16, uint32, uint64, char and decimal take NEP 50's dtype across 13 layouts, are never shared, and View refuses them naming NEP 50 | [`ConvertedDtypeStateTests`][gate] |
| 10 | NumPy scalars and their broadcasts copy; foreign memory, `NDArray<T>` and `np.asmatrix` arrays share; an array an `NDScope` disposed lives on in its tensor; `writeable` changed before or after an export is honored; `.real` / `.imag` and `view(dtype)` share their bits | [`SpecialStateTests`][gate] |
| 11 | The source array may be disposed while torch reads; torch views keep the buffer; `resize` refuses while torch reads; a tensor disposed, scope-closed or `set_` elsewhere leaves the view valid; a derived slice keeps the lease; the view books the whole storage; forgotten tensors and views are released by the finalizer | [`LifetimeTests`][gate] |
| 12 | An export joins the current `DisposeScope` and leaves only its result on it | [`ExportTests`][gate] |
| 13 | Parallel exports and imports release every lease; a tensor disposed on another thread is released once | [`ConcurrencyTests`][gate] |
| 14 | The loaded TorchSharp's native `from_blob` form is detected; only the two exact forms are accepted; an unknown form refuses every export and gives the array's reference back | [`AbiTests`][gate], [`FailurePathTests`][gate] |
| 15 | Failures half-way release what they took: libtorch refusing `from_blob`, a handle without storage, double lease releases | [`FailurePathTests`][gate] |
| 16 | The public surface is exactly the approved one, and `TorchCodecMode` has the three codec modes | [`ApiSurfaceTests`][gate] |
| 17 | The examples on this page and in the package README run as written | [`DocExampleTests`][gate] |

Every test also checks `LiveExports` and `LiveImports`: they must return to their baseline after the test, so each test
is also a no-leak and no-premature-free assertion.

## See also

- [Interoperability overview](index.md) - the contract every bridge builds on
- [PyTorch (pythonnet)](pytorch.md) - the Python route: NumSharp ⇄ PyTorch CPU tensors through `torch.from_numpy` /
  `Tensor.numpy`, when your torch code is Python
- [ONNX Runtime](onnxruntime.md) - the sibling .NET bridge whose codec modes this one shares
- Package README: `src/NumSharp.Interop.TorchSharp/README.md`

[gate]: https://github.com/SciSharp/NumSharp/tree/master/test/NumSharp.Tests.Interop.TorchSharp
