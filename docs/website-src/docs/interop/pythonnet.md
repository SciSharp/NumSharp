# Pythonnet — NumSharp ⇄ Python

The **NumSharp.Interop.pythonnet** package bridges `NDArray` to the Python ecosystem through [Python.NET (pythonnet)](https://github.com/pythonnet/pythonnet) — with **no Numpy.NET dependency**. It binds only to pythonnet's `PyObject` and Python's PEP 3118 buffer protocol, so it works with *any* numpy, *any* Python, and **any** buffer-exporting object: numpy arrays, `memoryview`, `bytes`, `bytearray`, `array.array`, `ctypes` arrays, `io.BytesIO` buffers, PIL images, torch CPU tensors, custom C extensions.

Both directions are **zero-copy by default**: Python mutates NumSharp's memory and NumSharp mutates Python's, with lifetimes coupled so neither side can free the buffer while the other can still see it. Measured across 48 exporter varieties, **45 share memory and 2 copy** — and those 2 are layouts that genuinely cannot be shared, not gaps. See [The Zero-Copy Model](zero-copy-model.md) for the decision tree.

> Migrating from or coexisting with SciSharp's Numpy.NET packages? See [Numpy.NET — coexistence & migration](numpy-net.md); every sample there maps 1:1 to a test in `NumSharp.Interop.UnitTests`.

**On this page:** [Installation](#installation) · [Quick Start](#quick-start) · [The Four Verbs](#the-four-verbs) · [Layout Fidelity](#layout-fidelity) · [Lifetime & Memory Safety](#lifetime--memory-safety) · [Auto-Marshaling](#auto-marshaling-the-codec) · [Dtypes](#dtype-mapping) · [Threading & the GIL](#threading--the-gil) · [Engine Lifecycle](#engine-lifecycle) · [Versions](#version-compatibility) · [Troubleshooting](#troubleshooting)

---

## Installation

```bash
dotnet add package NumSharp.Interop.pythonnet
```

The package depends on `pythonnet` `[3.0.5, 4.0.0)` and co-versions with NumSharp. You bring the Python: point pythonnet at a CPython shared library and initialize the engine once per process:

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";   // or set the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();                    // release the GIL from the init thread
```

Each pythonnet release hard-caps the Python it can drive, and **NuGet resolves the lowest applicable version, never the latest** — so the package's floor *is* the out-of-box experience. That floor is **3.0.5**, which drives **Python 3.7 – 3.13** with no configuration at all. Only **Python 3.14** needs you to opt in:

```xml
<PackageReference Include="pythonnet" Version="3.1.0" />
```

| Your Python | Minimum pythonnet | That pythonnet supports |
|---|---|---|
| 3.7 – 3.10 | 3.0.0 | 3.7 – 3.10 |
| 3.11 | 3.0.1 | 3.7 – 3.11 |
| 3.12 | 3.0.3 | 3.7 – 3.12 |
| 3.13 | 3.0.5 | 3.7 – 3.13 |
| 3.14 | 3.1.0 | 3.7 – 3.14 |

Read straight out of each package's own `PythonEngine.MinSupportedVersion`/`MaxSupportedVersion`, not the release notes. `MinSupportedVersion` has been 3.7 for the entire v3 line, so each release only ever *adds* the next minor — every version is a strict superset of its predecessors, which is exactly why raising the floor to 3.0.5 costs nothing. The rows below 3.0.5 are there to explain the mapping (and are what the error message quotes back at you); you cannot actually resolve them through this package.

Get it wrong and pythonnet usually dies inside `PythonEngine.Initialize()` with a bare `Failed to load symbol <PyFoo>` — and *sometimes* initializes anyway, which is the dangerous case. The bridge therefore checks the pairing once per session and raises an actionable error instead:

```text
Python 3.14 is not supported by the loaded pythonnet 3.0.5 (it supports Python 3.7 - 3.13).
Upgrade pythonnet to 3.1.0 or later: <PackageReference Include="pythonnet" Version="3.1.0" />.
```

The check reads each package's own `PythonEngine.MaxSupportedVersion`/`MinSupportedVersion`, so it stays correct as pythonnet moves; it is best-effort and never fails a session over its own parsing.

---

## Quick Start

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

var nd = np.arange(6).reshape(2, 3);

using (Py.GIL())
{
    using var scope = Py.CreateScope();
    scope.Exec("import numpy as np");

    // NumSharp -> numpy: a zero-copy VIEW of NumSharp's buffer
    using (PyObject x = nd.ToNumpy())
        scope.Set("x", x);

    scope.Exec("x[1, 2] = 99");        // Python writes...
    // ...NumSharp sees it: nd now holds 99 at (1, 2) — same memory.

    // numpy -> NumSharp
    using PyObject result = scope.Eval("np.sin(x / 3.0)");
    NDArray copy = result.ToNDArray();      // independent copy
    NDArray view = result.AsNDArray();      // zero-copy view (shared mutation)
}
```

Or skip the explicit calls entirely — register the codec once and every pythonnet boundary converts itself:

```csharp
NDArrayPythonInterop.RegisterCodec();       // Auto: share when possible, copy when not

using (Py.GIL())
{
    scope.Set("x", nd);                                 // NDArray -> numpy view, automatically
    using PyObject r = scope.Eval("np.sin(x / 3.0)");
    NDArray back = r.As<NDArray>();                     // numpy -> NDArray, automatically
}
```

---

## The Four Verbs

Everything in the package is packaging over four operations on the static `NDArrayPythonInterop` engine:

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → Python | `ToNumpy(nd, bool? requireGIL = null)` | **Zero-copy numpy view** — shared mutation, source rooted, full layout fidelity |
| NumSharp → Python | `ToNumpyCopy(nd, bool? requireGIL = null)` | Independent numpy array — no shared memory, no lifetime coupling |
| Python → NumSharp | `ToNDArray(py, bool? requireGIL = null)` | **Copy** any PEP 3118 exporter into a fresh C-contiguous `NDArray` |
| Python → NumSharp | `ToNDArrayView(py, bool allowReadonly = false, bool? requireGIL = null)` | **Zero-copy `NDArray` view** over Python memory — shared mutation, exporter leased |

All four are static members of `NDArrayPythonInterop`. The trailing `requireGIL` is the [GIL opt-out](#threading--the-gil); `null` (the default) means "manage it for me". `ToNumpy(nd, copy: true)` is a routing overload equivalent to `ToNumpyCopy`.

Plus `NDArrayPythonInterop.ToMemoryView(nd)` — a writable Python `memoryview` of raw bytes, for non-numpy consumers of the buffer protocol (`PIL.Image.frombuffer`, `struct.unpack_from`, sockets, ...) — and the dtype maps (`ToNumpyDtypeStr`, `FromNumpyDtypeStr`, `ToBufferFormat`, `FromBufferFormat`).

### Extension methods

The verbs double as extension methods on `NDArrayPythonInterop`, so `using NumSharp.Interop.PythonNet;` also adds the fluent forms. The convention mirrors numpy's `array`/`asarray`: **`To…` copies, `As…` shares** (`ToNumpy` is the exception — the zero-copy view is the package's headline, and `ToNumpyCopy` is the explicit copy).

```csharp
PyObject a = nd.ToNumpy();          // zero-copy view (alias: nd.ToPython())
PyObject c = nd.ToNumpyCopy();      // independent copy
PyObject m = nd.ToMemoryView();     // raw-bytes memoryview

NDArray  b = py.ToNDArray();        // copy
NDArray  v = py.AsNDArray();        // zero-copy view
NDArray  r = py.AsNDArray(allowReadonly: true);   // NON-WRITEABLE view of a read-only exporter
```

---

## Layout Fidelity

`ToNumpy` exports **every** NumSharp layout zero-copy — the numpy side sees the exact same strided window:

| NumSharp source | numpy result |
|---|---|
| C-contiguous | C-contiguous array |
| Sliced view (`nd["1:3, ::2"]`), any offset | Strided view over the same buffer |
| Transposed / Fortran-order | F-order strided view |
| Reversed (`nd["::-1"]`, negative strides) | Negative-stride view |
| Broadcast (stride-0) view | **Read-only** array (`flags.writeable == False`), matching NumSharp's own write protection |
| Scalar (0-d) | 0-d array |
| Empty | Empty array of the right shape/dtype |

Imports are symmetric. `ToNDArrayView` has **three** zero-copy routes — see [The Zero-Copy Model](zero-copy-model.md) for the full decision tree:

- **C-contiguous PEP 3118 exporters** (any object): the buffer is acquired with `PyBUF.WRITABLE`, which pins the exporter — resizable objects like `bytearray` are locked against reallocation for the lease's lifetime (`ba.append(...)` raises `BufferError` while a view lives).
- **Non-contiguous numpy arrays** (slices, transposes, Fortran order, broadcasts): imported through `__array_interface__` as true strided NumSharp views with identical layout. Broadcast sources become read-only NumSharp views; the numpy array is kept alive by a strong reference, so numpy's own `resize(refcheck=True)` refuses to reallocate under the view.
- **Non-contiguous non-numpy exporters** (a sliced / offset / reversed `memoryview`, a strided `array.array` memoryview): the base pointer comes from a `PyBUF.STRIDED` request and the exact shape/strides from the memoryview itself, so these are true views too rather than copies.

The lease is always acquired **through the exporter's `memoryview`**, never the raw object: pythonnet 3.0.x's `obj.GetBuffer` is per-exporter buggy (on a raw `ctypes` array it hard-crashes for every flag), while the memoryview over the same memory leases cleanly and keeps the source pinned. Measured coverage across 48 exporter varieties: **45 view, 2 copy** (`complex64`, sub-item strides — both genuinely unrepresentable).

Read-only sources (`bytes`, arrays with `writeable=False`) are **refused** for views by default — writing through them would corrupt immutable Python objects. Pass `allowReadonly: true` to opt in: the view comes back **non-writeable** (numpy's `writeable=False`, carried as `Shape.IsWriteable == false`), so guarded write paths raise `assignment destination is read-only` instead of corrupting the source. Or use `ToNDArray` to copy.

Import views also **do not own their data** — like `np.frombuffer(...)`, whose `flags.owndata` is `False`: a size-changing `ndarray.resize` refuses with NumPy's `cannot resize this array: it does not own its data` instead of silently reallocating away from the shared Python memory, and `np.require(..., "O")` produces an owning copy.

---

## Lifetime & Memory Safety

This is the part that makes the bridge production-grade. The rules:

**Exports** (`ToNumpy`, `ToMemoryView`) take their own atomic reference on the NumSharp buffer, so the memory survives even if every C# reference — including the returned `PyObject` wrapper — is disposed or garbage-collected. Release is owned by Python: a `weakref.finalize` on the exported array's *base* buffer object (which every derived numpy view — `arr[1:]`, `arr.T`, `np.asarray(arr)` — chains to) fires when the **last Python-side view** dies. If the engine shuts down while Python still holds views, the pin is swept right after `PythonEngine.Shutdown()` completes — pythonnet runs no Python atexit pass, so finalize callbacks cannot fire during shutdown (the sweep waits until the interpreter is provably gone, so it can never race Python reads). While exported, `ndarray.resize(refcheck: true)` on the source refuses to reallocate — the same guard NumPy applies to referenced arrays.

**Imports** (`ToNDArrayView`) lease the Python buffer through NumSharp's memory-block reference counting: the lease is released when the **last NumSharp view over the memory** — including derived slices like `nd["2:"]` — is disposed or collected. Explicitly disposing the original `NDArray` while a derived slice lives does *not* free the buffer; the refcount decides, not disposal order. The Python-side release is marshaled to the GIL safely (never raw on a finalizer thread).

In practice this means the patterns real applications use just work:

```csharp
// Store exported PyObjects in a cache and read them much later —
// the source NDArrays can be long gone:
_cache["features"] = nd.ToNumpy();

// Hand imported views across threads / queues / closures / awaits —
// Python can delete its own references, the lease keeps the data alive:
PyExec("del big_dataset");
double total = (double)np.sum(importedView);   // NumSharp kernels over Python-owned memory

// Chain freely: python array -> NumSharp view -> re-export -> stored in a python dict.
// The dict entry alone keeps the WHOLE transitive chain alive; clearing it cascades
// the teardown in order and frees the root.
```

Two observability counters expose the live state — useful in tests and leak hunts:

```csharp
int pins   = NDArrayPythonInterop.LiveExports;   // NumSharp buffers rooted by live Python views
int leases = NDArrayPythonInterop.LiveImports;   // Python buffers leased by live NumSharp views
```

---

## Auto-Marshaling (the Codec)

Register once and `NDArray` ⇄ numpy conversion happens automatically at every pythonnet boundary — `scope.Set`, Python call arguments, `ToPython()`, and `pyObj.As<NDArray>()` on the way back:

```csharp
NDArrayPythonInterop.RegisterCodec();   // once per engine session; idempotent

using (Py.GIL())
{
    scope.Set("x", nd);                                   // auto-encoded as a numpy view
    dynamic model = scope.Get("model");
    NDArray pred = ((PyObject)model.predict(nd)).As<NDArray>();   // in and out, no explicit calls
}
```

Policies via `NumpyCodecOptions`. `EncodeMode` / `DecodeMode` take a **`NumpyCodecMode`** — one enum for both directions:

| Mode | Meaning |
|---|---|
| `Auto` *(default)* | **Zero-copy view when the dtype/layout permits, an independent copy only when a view is impossible** — never a blanket copy |
| `View` | Always share; **decline** the conversion if a view is impossible — a loud failure instead of a silent copy |
| `Copy` | Always an independent copy — no shared memory, no `Py_buffer` lock, total coverage |

| Option | Default | Meaning |
|---|---|---|
| `EncodeMode` | `Auto` | How `NDArray` crossing into Python is materialized (a view and a copy have identical coverage on encode, so `Auto` always yields a view) |
| `DecodeMode` | `Auto` | How `As<NDArray>()` materializes — view-first, copy-fallback |
| `DecodeAnyBuffer` | `true` | Besides numpy, **any** buffer exporter decodes: `memoryview`/`bytes`/`bytearray`/`array.array` by name, plus everything else (`ctypes` arrays, C-extension buffers) via the PEP 688 `__buffer__` capability check on Python 3.12+ |

> **`Auto` decode shares memory.** `pyObj.As<NDArray>()` on a viewable source is a zero-copy view: mutations flow both ways, read-only sources decode as non-writeable views, and the view holds a `Py_buffer` lock (a `bytearray` cannot be resized while it lives). Use `DecodeMode = Copy` for a detached snapshot. Full rationale: [The Zero-Copy Model](zero-copy-model.md).

numpy `ndarray` subclasses (`matrix`, `memmap`, user subclasses) decode via an `__mro__` walk. Arrays with no numpy dtype (`decimal`) fall back to pythonnet's default CLR-object wrapping instead of failing the conversion.

---

## Dtype Mapping

| NumSharp | numpy | Notes |
|---|---|---|
| `Boolean` | `bool` (`\|b1`) | |
| `Byte` / `SByte` | `uint8` / `int8` | |
| `Int16` / `UInt16` | `int16` / `uint16` | |
| `Int32` / `UInt32` | `int32` / `uint32` | |
| `Int64` / `UInt64` | `int64` / `uint64` | |
| `Half` / `Single` / `Double` | `float16` / `float32` / `float64` | |
| `Complex` | `complex128` | complex64 sources **copy-widen** to `Complex` in `ToNDArray` (no zero-copy view) |
| `Char` | `uint16` (`<u2`) | UTF-16 code units — numpy has no char dtype |
| `Decimal` | — | No numpy equivalent (16-byte, non-IEEE); throws with guidance. Convert first: `nd.astype(NPTypeCode.Double)` |

**Big-endian** buffers (`>i4`, `!H`, ...) are rejected rather than silently byte-swapped — byte-swap on the Python side first: `arr.astype(arr.dtype.newbyteorder('<'))`. Exotic numpy dtypes (`datetime64`, `object`, structured/void) have no NumSharp representation and are rejected with clear errors.

---

## Threading & the GIL

- Every `NDArrayPythonInterop` method **acquires the GIL itself** (re-entrantly — nesting under your own `Py.GIL()` is fine). You can call the API from any thread, including threads that never touched Python before.
- **Optional opt-out.** Every verb takes a nullable `requireGIL`; `null` (the default) follows the process-wide `NDArrayPythonInterop.RequireGIL` (default `true`). Passing `false` skips the per-call `PyGILState_Ensure`/`Release` — useful for hot loops under one outer acquisition — but then **the caller must already hold the GIL**:

  ```csharp
  using (Py.GIL())                                              // ONE acquisition...
      foreach (var batch in batches)
          using (PyObject p = batch.ToNumpy(requireGIL: false))  // ...N conversions inside
              consumer.Invoke(p);
  ```

  > **Trap:** a .NET method or delegate invoked *from* Python does **not** hold the GIL — pythonnet's binder releases it around managed bodies. Keep GIL management **on** inside Python → .NET callbacks. Only pythonnet's argument/return marshaling (where the codec runs) executes under the GIL.

- Your own `PyObject` handling still follows pythonnet's rules — in particular, **dispose `PyObject`s under the GIL** (pythonnet 3.0.x requires it for the final decref).
- Internal releases never block: dropping the last NumSharp view of a lease only *enqueues* the Python-side release, which is drained under the GIL by a background worker, inline at the next conversion, and at engine shutdown. A thread that holds the GIL indefinitely cannot deadlock the interop.

---

## Engine Lifecycle

- One engine per process: initialize once; CPython + numpy do not support re-initialization after `Py_Finalize`.
- **Import views die with the interpreter.** `PythonEngine.Shutdown()` runs the package's shutdown handler first, which releases every outstanding lease crash-free — but the `NDArray`s over that memory must not be touched afterwards (disposing them stays safe). Exported buffers still referenced by Python are swept right after shutdown completes — never during teardown, never a use-after-free, and no leak survives the engine.
- On .NET 8+, pythonnet 3.0.x's own `Shutdown` crashes in its BinaryFormatter state-stashing (removed from the runtime). Opt out first:

```csharp
RuntimeData.FormatterType = typeof(NoopFormatter);
PythonEngine.Shutdown();
```

---

## Version Compatibility

| Component | Supported |
|---|---|
| pythonnet | `[3.0.5, 4.0.0)` — 3.0.5 is the floor (a strict superset of 3.0.0–3.0.4, so nothing is lost), the 4.x cap keeps a future breaking major out of consumers automatically. Pair with your Python per the [table above](#installation). The bridge reads all buffer metadata through Python's `memoryview` and uses only the `SIMPLE`/`WRITABLE`/`STRIDED`/`STRIDED_RO` buffer flags — the `PyBuffer` shape/strides/format flags are broken across 3.0.x, and this code path is correct on every one of them |
| Python | 3.7 – 3.14, bounded by your pythonnet (validated at first use with an actionable error) |
| numpy | Any — the bridge talks buffer protocol and `__array_interface__`, not numpy's C API. Verified against numpy 2.4.2 |
| .NET | `net8.0` and `net10.0` |

**Two pythonnet defects the bridge routes around**, so you don't have to:

- `PyBuffer` is unusable for shape/strides/format flags on 3.0.x, so **all metadata is read through Python's own `memoryview`** and `PyBuffer` is used only to obtain the pointer.
- `obj.GetBuffer` is *per-exporter* buggy — on a raw `ctypes` array it hard-crashes the process for **every** flag, and requesting a writable buffer from a read-only `memoryview` segfaults. So leases are acquired **through the exporter's `memoryview`** (uniformly well-behaved), and a writable buffer is only ever requested when `memoryview.readonly` says the source is writable.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Python 3.x is not supported by the loaded pythonnet 3.y.z` | Version pairing — install the pythonnet the message names (see the [matrix](#installation)) |
| `Failed to load symbol <PyFoo>` during `Initialize()` | The same pairing problem, raised by pythonnet before our check can run. Same fix |
| `Python engine is not initialized` | Call `PythonEngine.Initialize()` (and set `Runtime.PythonDLL`) before any conversion |
| `the exporter's buffer is read-only …` | You asked for a *writable* view of `bytes` / a `writeable=False` array. Pass `allowReadonly: true` for a non-writeable view, or `ToNDArray` to copy |
| `assignment destination is read-only` | You wrote through a non-writeable view (read-only or broadcast source). Copy it first if you need to mutate |
| `BufferError: Existing exports of data: object cannot be re-sized` (Python side) | A live NumSharp view holds a `Py_buffer` lease on that `bytearray`. `Dispose()` the view — see [the lock trade-off](zero-copy-model.md#the-trade-off-a-live-view-locks-the-source) |
| `ValueError: cannot resize an array that references or is referenced by …` (Python side) | Same, for numpy's `resize(refcheck=True)` |
| `cannot resize this array: it does not own its data` | Import views don't own their memory (numpy's `owndata == False`). Use `np.require(…, "O")` for an owning copy |
| `big-endian dtype … cannot be shared` | Byte-swap on the Python side: `arr.astype(arr.dtype.newbyteorder('<'))` |
| `decimal has no numpy dtype` | Convert first: `nd.astype(NPTypeCode.Double)` |
| complex64 came back as `Complex` | Expected — complex64 has no zero-copy NumSharp dtype and is widened on copy. See [why](zero-copy-model.md#complex64) |
| `As<NDArray>()` returned a **view** and a later Python write changed my data | That's the `Auto` default. Use `DecodeMode = NumpyCodecMode.Copy` for a snapshot |
| Access violation / process exit around a conversion | Almost always `requireGIL: false` (or `RequireGIL = false`) on a thread that does **not** hold the GIL — including inside a Python → .NET callback, where pythonnet releases it |
| Crash after `PythonEngine.Shutdown()` | Set `RuntimeData.FormatterType = typeof(NoopFormatter)` before shutting down, and don't touch import views afterwards |

---

## Testing

The package ships with a dedicated suite — `test/NumSharp.Interop.UnitTests` (149 tests on each of `net8.0` and `net10.0`) — covering every dtype and layout in both directions, all three conversion routes, the codec modes, the GIL policy, the lifetime model (orphaned exports, derived-view leases, transitive chains, premature/double disposal, GC hammers, cross-thread handoffs, async flows), and full made-up applications (ML inference, image pipelines, telemetry rings, co-simulation). Every test doubles as a leak test: it fails unless `LiveExports`/`LiveImports` return to baseline. The tests self-skip when no Python + numpy is found on the machine.

```bash
cd test/NumSharp.Interop.UnitTests
dotnet test                                   # both frameworks
dotnet test --filter "ClassName~StridedBufferViewTests"
```
