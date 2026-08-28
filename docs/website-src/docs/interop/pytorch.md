# PyTorch — shared CPU tensors with NumSharp

`NumSharp.Interop.pythonnet` can hand NumSharp memory to PyTorch and bring PyTorch CPU storage back
without copying it. The bridge follows PyTorch's own supported interchange path:
[`torch.from_numpy`](https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) in one
direction and [`Tensor.numpy`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numpy.html)
in the other. There is no TorchSharp dependency and no compile-time PyTorch dependency — `torch` is
imported from the CPython environment only when a Torch helper is called.

**On this page:** [Setup](#setup) · [The four operations](#the-four-operations) ·
[NumSharp → PyTorch](#numsharp--pytorch) · [PyTorch → NumSharp](#pytorch--numsharp) ·
[Autograd](#autograd) · [Dtypes](#dtype-map) · [Layouts and devices](#layouts-devices-and-copies) ·
[Other Python libraries](#what-pythonnet-supports) · [Claims](#claims-ledger)

> Verified live on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 ·
> **PyTorch 2.13.0+cpu** · net8.0/net10.0. PyTorch 2.13.0 was the latest stable release when this
> page was validated; 2.14 was still a release candidate. Every compatibility claim below is
> reproduced against the real installed PyTorch by `PyTorchInteropTests`.

---

## Setup

Install the .NET bridge and PyTorch into the same CPython environment pythonnet will embed:

```bash
dotnet add package NumSharp.Interop.pythonnet

# CPU wheel used by the compatibility gate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

Initialize pythonnet once per process:

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";  // or PYTHONNET_PYDLL
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();
```

The [pythonnet embedding guide](https://pythonnet.github.io/pythonnet/dotnet.html) requires the GIL
around every direct `PyObject` operation. The bridge methods acquire it themselves by default, but
the surrounding `PyObject` work still belongs inside `using (Py.GIL())`.

---

## The four operations

| Direction | Operation | Result | Memory |
|---|---|---|---|
| NumSharp → PyTorch | `nd.ToTorch()` | CPU `torch.Tensor` | **shared where representable** (`Decimal` converts) |
| NumSharp → PyTorch | `nd.ToTorch(copy: true)` | CPU `torch.Tensor` | independent C-contiguous copy |
| PyTorch → NumSharp | `tensor.AsTorchNDArray()` | `NDArray` view | **shared** |
| PyTorch → NumSharp | `tensor.ToTorchNDArray(force: false)` | owning `NDArray` | independent copy |
| PyTorch → NumSharp | `tensor.ToTorchNDArray(force: true)` | owning `NDArray` | detach + CPU transfer if needed, then copy |

The naming follows the rest of the package: **`As…` shares and `To…` copies**. `ToTorch` is the one
headline exception, like `ToNumpy`: it shares by default and makes a copy only when asked.

Internally these are deliberately small compositions of the existing interop:

```text
nd.ToTorch()              = torch.from_numpy(nd.ToNumpy())
nd.ToTorch(copy: true)    = torch.from_numpy(nd.ToNumpyCopy())
tensor.AsTorchNDArray()   = tensor.numpy().AsNDArray()
tensor.ToTorchNDArray()   = tensor.numpy().ToNDArray()
```

That structure matters: NumPy is not an accidental middle format. These are the public APIs PyTorch
documents for sharing CPU storage, and the NumSharp↔NumPy bridge already owns the layout and lifetime
contract.

---

## NumSharp → PyTorch

### Zero-copy, including positive strides

```csharp
using NDArray matrix = np.arange(12)
    .astype(NPTypeCode.Double)
    .reshape(3, 4);
using NDArray everyOtherColumn = matrix[":, ::2"];

using (Py.GIL())
using (PyObject tensor = everyOtherColumn.ToTorch())
{
    using var scope = Py.CreateScope();
    scope.Set("t", tensor);
    scope.Exec("t.add_(1000)");
}
```

`tensor.shape == (3, 2)`, `tensor.stride() == (4, 2)`, and the tensor's `data_ptr()` is the
NumSharp view's actual first-element address. The in-place add changes columns 0 and 2 of `matrix`;
columns 1 and 3 remain untouched. No element was moved to construct the tensor.

The returned tensor also owns the lifetime chain. Disposing the C# `NDArray` or the temporary numpy
wrapper cannot free its buffer while PyTorch still holds the tensor; the export pin is released when
the last Python-side storage reference dies.

### An independent tensor

```csharp
using NDArray reversed = matrix["::-1"];
using (Py.GIL())
using PyObject tensor = reversed.ToTorch(copy: true);
```

`copy:true` materializes logical values into C order before calling PyTorch. Tensor writes are then
detached from NumSharp. Use it for negative strides, broadcast/read-only arrays, or whenever a model
should own its input storage.

---

## PyTorch → NumSharp

### Shared CPU view

```csharp
using (Py.GIL())
using (var scope = Py.CreateScope())
{
    scope.Exec("import torch");
    using PyObject tensor = scope.Eval(
        "torch.arange(12, dtype=torch.float64).reshape(3, 4)[:, ::2]");
    using NDArray view = tensor.AsTorchNDArray();

    view[1, 1] = 77.5;                  // writes tensor[1, 1]
    scope.Set("t", tensor);
    scope.Exec("t[2, 0] = -12.0");     // immediately visible through view
}
```

The NumSharp view preserves shape `(3, 2)` and element strides `(4, 2)`. Its Python-buffer lease
keeps the numpy adapter and PyTorch storage alive until the last NumSharp-derived view is disposed,
even if Python deletes every other reference first.

This operation intentionally uses `tensor.numpy()` without `force`: sharing is only honest when the
tensor is already on CPU, has a NumPy-compatible dtype/layout, does not require gradients, and has no
unresolved conjugate or negative bit.

### Detached snapshot, including device/autograd tensors

```csharp
using NDArray copy = tensor.ToTorchNDArray();

// Equivalent to detach().cpu().resolve_conj().resolve_neg().numpy(), then a NumSharp copy:
using NDArray forcedCopy = tensor.ToTorchNDArray(force: true);
```

`force:true` matches the sequence documented for `Tensor.numpy(force=True)`. A CUDA/MPS tensor must
transfer to CPU, so that route cannot be zero-copy; the method makes the copy explicit in its name.

---

## Autograd

A tensor created from NumSharp is a normal PyTorch leaf tensor. Autograd can operate on it, and its
result can return as an owning NumSharp array:

```csharp
using NDArray x = np.array(new[] { -2.0, 1.5, 3.0 });

using (Py.GIL())
using (PyObject tensor = x.ToTorch())
using (var scope = Py.CreateScope())
{
    scope.Set("x", tensor);
    scope.Exec("x.requires_grad_()");
    scope.Exec("loss = (x * x).sum(); loss.backward()");

    using PyObject gradient = scope.Eval("x.grad");
    using NDArray dx = gradient.ToTorchNDArray();
    // dx == [-4.0, 3.0, 6.0]
}
```

Do not take a writeable shared NumSharp view directly from a tensor that requires gradients:
PyTorch itself rejects `tensor.numpy()` in that state. Use a detached tensor deliberately if shared
storage is appropriate, or `ToTorchNDArray(force: true)` for the safe independent result.

---

## Dtype map

PyTorch 2.13.0 accepts the entire NumSharp dtype surface through this bridge:

| NumSharp | NumPy adapter | PyTorch 2.13 |
|---|---|---|
| `Boolean` | `bool` | `torch.bool` |
| `Byte` / `SByte` | `uint8` / `int8` | `torch.uint8` / `torch.int8` |
| `Int16` / `UInt16` | `int16` / `uint16` | `torch.int16` / `torch.uint16` |
| `Int32` / `UInt32` | `int32` / `uint32` | `torch.int32` / `torch.uint32` |
| `Int64` / `UInt64` | `int64` / `uint64` | `torch.int64` / `torch.uint64` |
| `Char` | `uint16` UTF-16 code units | `torch.uint16` |
| `Half` | `float16` | `torch.float16` |
| `Single` / `Double` | `float32` / `float64` | `torch.float32` / `torch.float64` |
| `Complex` | `complex128` | `torch.complex128` |
| `Decimal` | converted to `float64` | `torch.float64` |

`Decimal` is necessarily a conversion: NumPy and PyTorch have no CLR-decimal dtype, so values lose
precision beyond what IEEE float64 can represent. `Char` is data, not text semantics — PyTorch sees
UTF-16 code units as unsigned 16-bit integers.

The unsigned 16/32/64 rows are live-probed behavior of PyTorch 2.13.0 and are pinned by the test;
older PyTorch documentation and releases list a narrower `torch.from_numpy` dtype set.

---

## Layouts, devices and copies

| Source | Shared operation | Result |
|---|---|---|
| NumSharp C-contiguous | `ToTorch()` | zero-copy |
| NumSharp transpose / F-order / positive slice | `ToTorch()` | zero-copy, strides preserved |
| NumSharp negative stride | `ToTorch()` | rejected; use `copy:true` |
| NumSharp broadcast/read-only | `ToTorch()` | rejected; use `copy:true` |
| PyTorch CPU, no grad, NumPy-compatible | `AsTorchNDArray()` | zero-copy |
| PyTorch CPU, positive non-contiguous stride | `AsTorchNDArray()` | zero-copy, strides preserved |
| PyTorch tensor requiring grad | `AsTorchNDArray()` | PyTorch rejects; detach deliberately or copy with `force:true` |
| PyTorch CUDA/MPS tensor | `AsTorchNDArray()` | cannot share CPU NumSharp memory; copy with `force:true` |

The read-only guard is a safety boundary, not a convenience restriction. PyTorch warns that writing
through a tensor created from a non-writeable numpy array is undefined behavior. A NumSharp
broadcast has stride 0 and is deliberately non-writeable, so `ToTorch()` refuses it before PyTorch
can create an unsafe tensor; `copy:true` produces ordinary owned storage.

For a contiguous raw byte route, `nd.ToMemoryView()` can also feed
[`torch.frombuffer`](https://docs.pytorch.org/docs/stable/generated/torch.frombuffer.html). That API
requires the caller to spell dtype, count, offset and reshape manually. Prefer `ToTorch()` for shaped
arrays; use `frombuffer` when integrating an API that is explicitly buffer-oriented.

---

## What pythonnet supports

pythonnet does not maintain an allowlist of scientific libraries. It embeds real CPython, so any
package importable in that interpreter can be called. Array interoperability then depends on the
package's data protocol:

| Python library shape | NumSharp route | Copy contract |
|---|---|---|
| NumPy/SciPy/scikit-learn code accepting `ndarray` | `nd.ToNumpy()` | input view is zero-copy; the operation decides its output |
| Buffer consumers such as `torch.frombuffer`, `pa.py_buffer`, `PIL.Image.frombuffer` | `nd.ToMemoryView()` | zero-copy for compatible contiguous CPU buffers |
| PyTorch CPU tensors | `ToTorch` / `AsTorchNDArray` | zero-copy where the table above allows |
| pandas/polars/OpenCV APIs accepting NumPy | pass `nd.ToNumpy()` | library may retain, view, or copy according to its own API |
| GPU/device arrays (PyTorch CUDA, CuPy, JAX, TensorFlow) | device-specific transfer or DLPack | NumSharp currently has no device-memory/DLPack object; a CPU crossing copies |

See [Any library via `np.frombuffer`](np-frombuffer.md) for the PEP 3118 consumer route and
[Python & numpy](pythonnet-numpy.md) for the general four-verb/lifetime contract.

One distinction prevents a common mistake: **`torch.Tensor` is not a PEP 3118 buffer exporter** —
`memoryview(tensor)` raises `TypeError`. PyTorch *consumes* Python buffers through `torch.frombuffer`,
and exposes CPU tensors to NumPy through `Tensor.numpy` / `__array__`. The bridge uses the correct
direction-specific protocol instead of pretending those are the same capability.

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | The validated runtime is stable PyTorch 2.13.x | live `torch.__version__` | [`Runtime_IsLatestStablePyTorch213`][gate] |
| 2 | A contiguous tensor has NumSharp's exact pointer and shares writes both ways | `data_ptr` equality + mutations | [`NumSharpToTorch_Contiguous_IsZeroCopyAndMutatesBothWays`][gate] |
| 3 | Positive strides survive NumSharp → Torch | shape `(3,2)`, stride `(4,2)`, skipped columns untouched | [`NumSharpToTorch_PositiveStridedView_PreservesShapeStrideAndStorage`][gate] |
| 4 | All 15 NumSharp dtypes map on 2.13 | exhaustive dtype table | [`NumSharpToTorch_All15Dtypes_MapOnPyTorch213`][gate] |
| 5 | Negative/read-only layouts cannot enter an unsafe shared tensor | directed rejection; `copy:true` succeeds and detaches | [`NumSharpToTorch_UnsafeLayoutsRequireExplicitCopy`][gate] |
| 6 | A positive-strided CPU tensor becomes a lifetime-safe NumSharp view | shape/strides + both-way writes + delete-source survival | [`TorchToNumSharp_PositiveStridedCpuTensor_IsZeroCopyAndSurvivesTensorWrapper`][gate] |
| 7 | Copy and force semantics are detached and autograd-safe | PyTorch rejection without detach; force copy remains unchanged | [`TorchToNumSharp_CopyIsDetached_AndForceHandlesAutogradTensor`][gate] |
| 8 | Real autograd output returns exactly to NumSharp | `(x*x).sum().backward()` gives `[-4,3,6]` | [`NumSharpToTorch_AutogradCompute_GradientReturnsToNumSharp`][gate] |
| 9 | Tensor is not a buffer exporter | `memoryview(tensor)` raises; `__array__` exists | [`TorchTensor_IsNotABufferExporter_TheNumpyAdapterIsIntentional`][gate] |

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/PyTorchInteropTests.cs
