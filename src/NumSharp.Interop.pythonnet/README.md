# NumSharp.Interop.pythonnet

Zero-copy interop between NumSharp `NDArray` and the Python ecosystem via [Python.NET (pythonnet)](https://github.com/pythonnet/pythonnet) — with **no Numpy.NET dependency**, so it works with any numpy, any Python, and any object implementing the PEP 3118 buffer protocol (numpy, `memoryview`, `bytes`, `bytearray`, `array.array`, PIL, torch, ...).

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = "python312.dll";     // or the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
```

## The four verbs

Everything is packaging over four operations on the static `NDArrayInterop` engine:

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → Python | `ToNumpy(nd)` | **zero-copy numpy view** of NumSharp's buffer — shared mutation, source rooted; full layout fidelity (slices, transposes, Fortran order, negative strides; broadcasts become read-only; scalars become 0-d) |
| NumSharp → Python | `ToNumpyCopy(nd)` | independent numpy array (no shared memory, no lifetime coupling) |
| Python → NumSharp | `ToNDArray(py)` | **copy** any PEP 3118 exporter into a fresh C-contiguous `NDArray` (honors strides / Fortran order; complex64 widens to complex128; 0-d becomes a scalar) |
| Python → NumSharp | `ToNDArrayView(py[, allowReadonly])` | **zero-copy NDArray view** over Python memory — shared mutation; C-contiguous buffers via a locked `PyBuffer` lease, non-contiguous numpy arrays via `__array_interface__` as true strided views |

Plus `ToMemoryView(nd)` (a writable Python `memoryview` of raw bytes for non-numpy consumers) and the dtype maps `ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`.

## Extension methods (the ergonomic default)

The verbs above double as extension methods — `NDArrayInterop` provides them, so importing its
namespace lets you call them fluently on `NDArray` / `PyObject` as well as statically:

```csharp
using NumSharp.Interop.PythonNet;

PyObject a = nd.ToNumpy();          // zero-copy view (also: nd.ToPython())
PyObject c = nd.ToNumpyCopy();      // independent copy
NDArray  b = py.ToNDArray();        // copy   (To… = copy)
NDArray  v = py.AsNDArray();        // view   (As… = share, like numpy array/asarray)
```

## Codec (auto-marshaling)

```csharp
NDArrayInterop.RegisterCodec();     // once per engine session; idempotent

using (Py.GIL())
{
    scope.Set("x", nd);                          // NDArray auto-encoded to a numpy view
    dynamic np = Py.Import("numpy");
    NDArray r = ((PyObject)np.matmul(a, b)).As<NDArray>();   // decoded back, no explicit calls
}
```

`NumpyCodecOptions` controls the policies: `EncodeAsView` (default `true`), `DecodeAsView` (default `false` — decode copies), `DecodeAnyBuffer` (default `true` — `memoryview`/`bytes`/`bytearray`/`array.array` also decode). numpy `ndarray` subclasses (`matrix`, `memmap`) decode via an `__mro__` walk. Arrays with no numpy dtype (`decimal`) fall back to pythonnet's default CLR wrapping instead of failing.

## Lifetime & memory safety (the design)

**Exports** take their own atomic reference on the NumSharp buffer (NumSharp's ARC), so the memory survives even if every C# reference — including the returned `PyObject` wrapper — is disposed or collected. Release is owned by Python: a `weakref.finalize` on the exported array's *base* buffer object (which every derived numpy view chains to) fires when the last Python-side view dies. If the engine shuts down while Python still holds views, the pin is swept right after `PythonEngine.Shutdown()` completes — pythonnet runs no Python atexit pass, so the finalize callbacks cannot fire during shutdown (probed; the sweep waits until the interpreter is provably gone, so it can never race Python reads). While exported, `ndarray.resize(refcheck: true)` on the source refuses to reallocate — the same guard NumPy applies to referenced arrays.

**Imports** lease the Python buffer through NumSharp's memory-block reference counting: the lease is released when the last NumSharp view over the memory — *including derived slices like `nd["2:"]`* — is disposed or collected. C-contiguous leases hold a real `Py_buffer`, so resizable exporters (`bytearray`) are locked against reallocation, and numpy's `resize` refcheck refuses while the lease lives. The Python-side release is marshaled to the GIL safely (never raw on a finalizer thread); `NDArrayInterop.LiveExports` / `LiveImports` expose the counters.

Read-only sources (`bytes`, non-writeable numpy arrays) are **refused** for views by default — writing through them would corrupt immutable Python objects. Pass `allowReadonly: true` to opt in: the view comes back **non-writeable** (`Shape.IsWriteable == false`, exactly numpy's `writeable=False`), so guarded write paths raise `assignment destination is read-only` instead of corrupting the source. Or use `ToNDArray` to copy.

Import views also **do not own their data** (like `np.frombuffer(...)`, whose `flags.owndata` is `False`): a size-changing `ndarray.resize` refuses with NumPy's `cannot resize this array: it does not own its data` instead of silently reallocating away from the shared Python memory, and `np.require(..., "O")` produces an owning copy.

## Rules & limits

- **GIL**: every method acquires the GIL itself (re-entrant). Your own `PyObject` usage still follows pythonnet's rules.
- **Engine lifetime**: import views die with the interpreter. `PythonEngine.Shutdown()` releases all outstanding leases crash-free (a registered shutdown handler), but the NDArrays over that memory must not be touched afterwards (disposing them stays safe). Exports still held by Python are swept right after shutdown completes — no leak survives the engine.
- **`PythonEngine.Shutdown` on .NET 8+**: pythonnet 3.0.x crashes in its own BinaryFormatter state-stashing. Opt out first: `RuntimeData.FormatterType = typeof(NoopFormatter);`.
- **dtypes**: all NumSharp dtypes map except `Decimal` (no numpy equivalent — convert first); `Char` is exported as `uint16` (UTF-16 code units). Big-endian buffers are rejected (byte-swap first: `arr.astype(arr.dtype.newbyteorder('<'))`). complex64 imports copy-widen to complex128 (no zero-copy view).
- **pythonnet versions**: floor is 3.0.1. Its `PyBuffer` is broken for shape/strides/format flags, so metadata is read through Python's `memoryview` and only the crash-free `PyBUF.SIMPLE`/`PyBUF.WRITABLE` are used — the same code path works on every 3.0.x. Python 3.12+ requires pythonnet ≥ 3.0.4.
