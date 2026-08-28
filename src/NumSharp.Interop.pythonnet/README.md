# NumSharp.Interop.pythonnet

Zero-copy interop between NumSharp `NDArray` and the Python ecosystem via [Python.NET (pythonnet)](https://github.com/pythonnet/pythonnet) — with **no Numpy.NET dependency**. NumPy and PEP 3118 buffers enter the memory bridge directly; registered `IPythonArrayAdapter` instances feed library objects such as `torch.Tensor` and Pandas containers through that same bridge. PyTorch and Pandas adapters are built in; neither duplicates the memory implementation.

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = "python312.dll";     // or the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
```

## The four verbs

Everything is packaging over four operations on the static `NDArrayPythonInterop` engine:

| Direction | Verb | Semantics |
|---|---|---|
| NumSharp → Python | `ToNumpy(nd)` | **zero-copy numpy view** of NumSharp's buffer — shared mutation, source rooted; full layout fidelity (slices, transposes, Fortran order, negative strides; broadcasts become read-only; scalars become 0-d) |
| NumSharp → Python | `ToNumpyCopy(nd)` | independent numpy array (no shared memory, no lifetime coupling) |
| Python → NumSharp | `ToNDArray(py)` | **copy** any PEP 3118 exporter or registered-adapter source into a fresh C-contiguous `NDArray` (honors logical strides/order; complex64 widens; adapter Copy may detach/transfer) |
| Python → NumSharp | `ToNDArrayView(py[, allowReadonly])` | **zero-copy NDArray view** over Python memory — adapters only select a canonical shareable object; the existing locked-buffer / `__array_interface__` / strided-buffer routes make every memory and lifetime decision |

The lease buffer is always acquired **through the exporter's `memoryview`**, never the raw object: the memoryview is CPython's canonical, uniformly-behaved buffer exporter, so this sidesteps pythonnet 3.0.x's per-exporter `GetBuffer` bugs — a raw `ctypes` array hard-crashes `obj.GetBuffer` on *every* flag, while the memoryview over the same memory leases cleanly. Measured coverage across 50 exporter varieties: **47 view, 2 copy** (`complex64` widens, sub-item strides linearize). Big-endian multi-byte data is a **third copy** — the view path declines it (a native-endian shared view is impossible), but `ToNDArray` byte-reverses each element on copy — so the only refusals left are element types with no NumSharp dtype at all (`datetime64`, structured, object, void).

Plus `ToMemoryView(nd)` (a writable Python `memoryview` of raw bytes for non-numpy consumers) and the dtype maps `ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`.

## PyTorch through the existing bridge

The built-in `TorchPythonArrayAdapter` makes the ordinary verbs work directly; there is no TorchSharp
or compile-time PyTorch dependency:

```csharp
using (Py.GIL())
{
    using PyObject tensor = nd.ToTorch();                  // shared CPU tensor
    using PyObject owned = nd.ToTorch(copy: true);         // independent tensor
    using NDArray view = tensor.AsNDArray();               // existing shared bridge
    using NDArray copy = tensor.ToNDArray();               // existing copy bridge (force permitted)

    // Torch-specific aliases/policy controls remain available:
    using NDArray sameView = tensor.AsTorchNDArray();
    using NDArray strictCopy = tensor.ToTorchNDArray();     // no detach/device transfer
    using NDArray forced = tensor.ToTorchNDArray(force: true); // detach + CPU transfer, then copy
}
```

The sharing path preserves positive strides and roots the owner on both sides. `Decimal` is the
documented exception: neither NumPy nor PyTorch has that dtype, so it converts to float64.
Negative-stride and non-writeable NumSharp arrays require `copy:true`; this avoids PyTorch's
documented undefined behavior for tensors over read-only NumPy memory. A Tensor requiring gradients
or living on CUDA/MPS cannot be shared directly with CPU NumSharp memory — use an explicit detach or
`force:true` copy.

Rare tensor families follow PyTorch's NumPy compatibility boundary: `complex64` copy-widens to
NumSharp `Complex`; `bfloat16`, float8, sparse and quantized tensors cannot cross through NumPy;
meta tensors have no data even for `force:true`. PyTorch `expand()` tensors import safely as
non-writeable NumSharp broadcasts. Scalars, empty arrays, storage offsets, F-order/transposes,
conjugate/negative view bits, derived-view lifetimes, resize refusal, all dtype boundary values,
caller-owned GIL mode and concurrent churn are pinned by the live interop test matrix.

## Pandas through the existing bridge

The built-in `PandasPythonArrayAdapter` recognizes `DataFrame`, `Series`, `Index`,
`ExtensionArray`, and their subclasses. Pandas → NumSharp uses the ordinary verbs:

```csharp
using NDArray view = series.AsNDArray(allowReadonly: true); // verified shared projection
using NDArray copy = frame.ToNDArray();                     // owning C-contiguous snapshot

NDArrayPythonInterop.RegisterCodec();
using NDArray automatic = frame.As<NDArray>();              // Auto: view, then copy fallback
```

Pandas documents that `to_numpy(copy:false)` does not guarantee a view. View mode therefore checks
that two independent official projections overlap before the existing bridge leases one. Homogeneous
NumPy-backed frames, numeric Series/Indexes, and some extension arrays share; mixed numeric blocks,
nullable arrays with missing values, categorical, and sparse arrays copy through Auto/Copy. Pandas 3
usually exposes shared projections read-only, so direct view callers pass `allowReadonly:true` and
receive a non-writeable `NDArray`. Object/string/datetime/timedelta outputs remain unsupported because
NumSharp has no corresponding dtype. Axis labels are metadata and do not enter the value array.

NumSharp → Pandas is the existing NumPy encoder; request Pandas `copy:false` when sharing is intended:

```csharp
NDArrayPythonInterop.RegisterCodec();
scope.Set("values", nd);
scope.Exec("series = pd.Series(values, copy=False)");
```

## Adapter-aware pythonnet codec

After `NDArrayPythonInterop.RegisterCodec()`, pythonnet uses the same adapter registry implicitly:

```csharp
dynamic torch = Py.Import("torch");
using PyObject tensor = (PyObject)torch.from_numpy(nd); // registered encoder: NDArray → NumPy
using NDArray a = tensor.As<NDArray>();                 // registered decoder: Torch → shared bridge
using NDArray b = (NDArray)tensor.AsManagedObject(typeof(NDArray));
```

`NumpyCodecOptions.DecodeArrayAdapters` is on by default and independent of `DecodeAnyBuffer` and
`DecodeArrayLike`. View permits only share-preserving adaptation; Copy permits library-specific
materialization (including Pandas coercion or Torch detach/device transfer); Auto tries View and
falls back to Copy.

`nd.ToPython()` itself produces the bridge's NumPy view because pythonnet's encoder sees the managed
source type, not the eventual Python library target. That is also why `torch.from_numpy(nd)` works:
the callable receives the implicitly encoded NumPy view. On return, `As<NDArray>()` and
`AsManagedObject(typeof(NDArray))` do carry an explicit CLR target and select the adapter-aware decoder.

Other libraries can register the same kind of front door without writing memory code:

```csharp
PythonArrayAdapterRegistry.Register(myAdapter); // implements IPythonArrayAdapter
```

The adapter returns a canonical buffer/NumPy object. `NDArrayPythonInterop` still owns dtype, shape,
strides, writeability, leases, copy fallback, GIL and shutdown behavior. Adapters may decline an
instance by returning `null`; the registry then tries the next adapter. Disabling codec adapters does
not suppress a recognized object's independent NumPy/buffer capability.

## Extension methods (the ergonomic default)

The verbs above double as extension methods — `NDArrayPythonInterop` provides them, so importing its
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
NDArrayPythonInterop.RegisterCodec();     // once per engine session; idempotent

using (Py.GIL())
{
    scope.Set("x", nd);                          // NDArray auto-encoded to a numpy view
    dynamic np = Py.Import("numpy");
    NDArray r = ((PyObject)np.matmul(a, b)).As<NDArray>();   // decoded back, no explicit calls
}
```

`NumpyCodecOptions` controls the policies via **`NumpyCodecMode`** — one enum for both directions (`EncodeMode` / `DecodeMode`):

| Mode | Meaning |
|---|---|
| `Auto` *(default)* | **Zero-copy view when the dtype/layout permits, an independent copy only when a view is impossible** — never a blanket copy. On decode the fallback is real (complex64 widens, UCS-4 text narrows, sub-item strides linearize, big-endian multi-byte byteswaps — all as copies; everything else stays a view). On encode a view and a copy have identical dtype coverage, so Auto always yields a view. |
| `View` | Always share; **decline** the conversion (return no value) if a view is impossible — a loud failure for callers who depend on shared memory. |
| `Copy` | Always an independent copy — no shared memory, no Py_buffer lock, total coverage. |

Plus `DecodeArrayAdapters` (default `true`, including PyTorch and Pandas), `DecodeAnyBuffer` (default `true`) and `DecodeArrayLike` (default `true` for numeric builtin containers). NumPy `ndarray` subclasses decode via an `__mro__` walk.

> **`Auto` decode shares memory.** Under the default, `pyObj.As<NDArray>()` on a contiguous/strided-numpy source is a zero-copy view: mutations flow both ways, read-only sources decode as non-writeable views, and the view holds a `Py_buffer` lock on the source (a `bytearray` cannot be resized while it lives). Use `DecodeMode = Copy` when you want a detached snapshot that never touches the Python object.

## Lifetime & memory safety (the design)

**Exports** take their own atomic reference on the NumSharp buffer (NumSharp's ARC), so the memory survives even if every C# reference — including the returned `PyObject` wrapper — is disposed or collected. Release is owned by Python: a `weakref.finalize` on the exported array's *base* buffer object (which every derived numpy view chains to) fires when the last Python-side view dies. If the engine shuts down while Python still holds views, the pin is swept right after `PythonEngine.Shutdown()` completes — pythonnet runs no Python atexit pass, so the finalize callbacks cannot fire during shutdown (probed; the sweep waits until the interpreter is provably gone, so it can never race Python reads). While exported, `ndarray.resize(refcheck: true)` on the source refuses to reallocate — the same guard NumPy applies to referenced arrays.

**Imports** lease the Python buffer through NumSharp's memory-block reference counting: the lease is released when the last NumSharp view over the memory — *including derived slices like `nd["2:"]`* — is disposed or collected. C-contiguous leases hold a real `Py_buffer`, so resizable exporters (`bytearray`) are locked against reallocation, and numpy's `resize` refcheck refuses while the lease lives. The Python-side release is marshaled to the GIL safely (never raw on a finalizer thread); `NDArrayPythonInterop.LiveExports` / `LiveImports` expose the counters.

Read-only sources (`bytes`, non-writeable numpy arrays) are **refused** for views by default — writing through them would corrupt immutable Python objects. Pass `allowReadonly: true` to opt in: the view comes back **non-writeable** (`Shape.IsWriteable == false`, exactly numpy's `writeable=False`), so guarded write paths raise `assignment destination is read-only` instead of corrupting the source. Or use `ToNDArray` to copy.

Import views also **do not own their data** (like `np.frombuffer(...)`, whose `flags.owndata` is `False`): a size-changing `ndarray.resize` refuses with NumPy's `cannot resize this array: it does not own its data` instead of silently reallocating away from the shared Python memory, and `np.require(..., "O")` produces an owning copy.

## Rules & limits

- **GIL**: every method acquires the GIL itself (re-entrant). Your own `PyObject` usage still follows pythonnet's rules.
- **GIL opt-out**: every conversion verb takes a nullable `requireGIL` parameter; `null` (the default) follows the process-wide `NDArrayPythonInterop.RequireGIL` (default `true`). An effective `false` replaces `Py.GIL()` with a shared no-op guard — the calling thread must **already hold the GIL** (an enclosing `Py.GIL()` block); converting GIL-less without holding it is an immediate access violation. **Trap:** a .NET method/delegate body invoked *from* Python does **not** hold the GIL — pythonnet's binder releases it around managed bodies (probed on 3.0.5/3.1.0: `PyGILState_Check() == 0` inside the body) — so keep GIL management on inside Python→.NET callbacks. The interop's background machinery (deferred lease disposal, shutdown drain) always manages the GIL itself, regardless of the policy.

  ```csharp
  using (Py.GIL())                                     // ONE acquisition...
      for (int i = 0; i < n; i++)
          using (PyObject p = batch[i].ToNumpy(requireGIL: false))   // ...N conversions inside
              consumer.Invoke(p);

  NDArrayPythonInterop.RequireGIL = false;                   // or process-wide, when EVERY call site holds the GIL
  ```
- **Engine lifetime**: import views die with the interpreter. `PythonEngine.Shutdown()` releases all outstanding leases crash-free (a registered shutdown handler), but the NDArrays over that memory must not be touched afterwards (disposing them stays safe). Exports still held by Python are swept right after shutdown completes — no leak survives the engine.
- **`PythonEngine.Shutdown` on .NET 8+**: pythonnet crashes in its own BinaryFormatter state-stashing. Opt out first: `RuntimeData.FormatterType = typeof(NoopFormatter);` (`NoopFormatter` ships from pythonnet 3.0.4 onward — guaranteed by the 3.0.5 floor below).
- **dtypes**: all NumSharp dtypes map except `Decimal` — it has no numpy dtype, so the array-producing verbs (`ToNumpy` / `ToNumpyCopy`, and the codec) auto-convert it to `float64` (the `astype(Double)` guidance, automated — lossy beyond ~16 digits), while the low-level maps (`ToNumpyDtypeStr` / `ToBufferFormat` / `ToMemoryView`) stay honest and still refuse it. `Char` is exported as `uint16` (UTF-16 code units) and imported from 2-byte wchar buffers (PEP 3118 `'u'`: `array.array('u')` on Windows, `ctypes.c_wchar`) as a zero-copy view. Big-endian multi-byte buffers cannot be *viewed* (a native-endian shared view is impossible), but `ToNDArray` **byteswaps them on copy** (`>c16` reverses each 8-byte half; `>c8` reverses then widens; single-byte `>i1`/`|u1`/`|b1` view regardless). complex64 imports copy-widen to complex128, UCS-4 text (numpy `<U1`, 4-byte wchar, `'w'`) copy-narrows to `Char` (BMP only) — neither has a zero-copy view.
- **array-like decode (on by default)**: `py.FromArrayLike()` materializes a Python `list` / `tuple` / nested sequence / scalar (`int`/`float`/`bool`/`complex`) into an owning `NDArray` via `numpy.asarray`, and the registered codec does it automatically (`NumpyCodecOptions.DecodeArrayLike`, default `true`) so a numpy call or plain Python that returns a list crosses without a manual conversion. It fires ONLY for genuine non-buffer builtins — a numpy array the view/copy paths declined is never rescued into a silent copy. Set `DecodeArrayLike = false` to restrict decode to buffer exporters (fully numpy-agnostic).
- **pythonnet versions**: the dependency is `[3.0.5, 4.0.0)`. `PyBuffer` is broken for shape/strides/format flags, so metadata is read through Python's `memoryview` and only the crash-free `PyBUF.SIMPLE`/`PyBUF.WRITABLE` are used — the same code path works on every v3. The source compiles clean against all of 3.0.0–3.1.0; 3.0.5 is the floor because it is the oldest release that covers the Python versions people actually run (see below).

### Choosing a pythonnet version

Every pythonnet release hard-caps the Python it can drive. `MinSupportedVersion` has been 3.7 for the
entire v3 line, so the ceiling is the only thing that moves — which makes **3.0.5 a strict superset of
every earlier release**. Read from each package's own `PythonEngine.MaxSupportedVersion`:

| pythonnet | Python supported | note |
|-----------|------------------|------|
| 3.0.0 | 3.7 – 3.10 | |
| 3.0.1 / 3.0.2 | 3.7 – 3.11 | |
| 3.0.3 / 3.0.4 | 3.7 – 3.12 | `NoopFormatter` added in 3.0.4 |
| **3.0.5** | **3.7 – 3.13** | **the floor — what you get by default** |
| 3.1.0 | 3.7 – 3.14 | release notes drop 3.7–3.9; opt in for Python 3.14 |

**NuGet installs the floor, not the latest.** Resolution is lowest-applicable, so a plain
`dotnet add package NumSharp.Interop.pythonnet` gives you pythonnet **3.0.5** even though newer
releases exist. On **Python 3.14**, ask for it explicitly:

```xml
<PackageReference Include="pythonnet" Version="3.1.0" />
```

Overshooting a release's ceiling usually kills `PythonEngine.Initialize()` with a bare
`MissingMethodException: Failed to load symbol PyUnicode_AsUnicode` (or similar). If the engine
initializes anyway, the interop checks the range itself on first use and raises an actionable error
naming the version to install.
