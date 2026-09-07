---
name: pythonic
description: Write or use NumSharp's `NumSharp.Interop.pythonnet` layer — the Python-shaped facade (`np`/`ctypes`/`weakref`/`builtins` module classes + `PyObject` extension members) that makes C# driving embedded CPython read like the Python it runs, the GIL / `requireGIL` / PyObject-lifetime discipline, and the ZERO-COPY `NDArray`⇄numpy⇄torch/pandas crossing (verbs `ToNumpy`/`ToNDArray`/`ToNDArrayView`/`ToTorch`/… plus the `NumpyCodec`/`TupleCodec` implicit codecs) — so you never hand-roll element loops. Use when adding or editing a facade member or a `PyObject` extension, a crossing verb, a codec or `IPythonArrayAdapter`, writing an interop parity test (`Python.np.eval/with/evalStmts`, `ByteContract`), deciding typed-facade vs snippet-runner or explicit-verb vs codec crossing, or debugging a GIL / lease / lifetime / "numpy scalar isn't a Python int" / `CS9202` extension-members bug.
keywords: pythonnet interop layer, Python-shaped facade, Python.np, PyObject extension members, zero-copy NDArray crossing, NDArray to numpy, NDArray to torch, ToNumpy ToNDArray ToNDArrayView ToTorch, NumpyCodec TupleCodec, RegisterCodec, IPythonArrayAdapter, GIL requireGIL discipline, PyObject lifetime, PythonRuntimeInterop, weakref.finalize export rooting, ExportKeeper lease, byte-contract interop tests, evalStmts snippet runner
keywords-sparse: pythonnet interop, Python-shaped facade, zero-copy numpy/torch crossing, ToNumpy/ToNDArray/ToTorch verbs, NumpyCodec/TupleCodec, GIL and PyObject lifetime, typed facade vs evalStmts, PyObject extension member
---

# pythonic — `NumSharp.Interop.pythonnet` (read like Python, cross data zero-copy)

How NumSharp's **`NumSharp.Interop.pythonnet`** package drives **embedded CPython** so the C# reads like
the Python it runs — and how to move arrays/tensors across the boundary **without copying, element by
element, ever**. This package IS the zero-copy `NDArray`⇄numpy⇄torch/pandas bridge; downstream pythonnet
bridges (OptunaSharp's BoTorch/OptunaHub, any consumer) build on it. Built on `Python.Runtime`
(pythonnet).

**Reference implementations to read (in this repo):**
- `src/NumSharp.Interop.pythonnet/Pythonic.cs` — the **canonical typed facade** (`np`, `ctypes`,
  `weakref`, `builtins` module classes + `PyObjectPythonic` extension members): one C# member per Python
  call, dispatched to **pre-resolved cached callables**, **zero Python source strings**.
- `test/NumSharp.Tests.Interop/Pythonic.cs` — the **hybrid**: a `Python.np` facade PLUS the
  `eval`/`with`/`evalStmts`/`truthy` snippet-runner escape hatch, `NumpyExtensions`, and `ByteContract`
  (the byte-for-byte oracle).
- `src/NumSharp.Interop.pythonnet/NDArrayPythonInterop.cs` (+ `.Export.cs` / `.Import.cs`) — the crossing
  verbs (`ToNumpy`/`ToNumpyCopy`/`ToNDArray`/`ToNDArrayView`) + `RegisterCodec`.
- `src/NumSharp.Interop.pythonnet/TorchInterop.cs` — the optional torch bridge
  (`ToTorch`/`AsTorchNDArray`/`ToTorchNDArray`).
- `src/NumSharp.Interop.pythonnet/NumpyCodec.cs` + `TupleCodec.cs` + `EncoderHandoff.cs` — the implicit
  codec layer + the leak-safe encoder handoff.
- `src/NumSharp.Interop.pythonnet/PyExtensions.cs` — the **public** C# 14 extension member on
  pythonnet's `Py`: `Py.RegisterNumSharpCodec()`, the SINGLE switch that enables all support (aliases
  `NDArrayPythonInterop.RegisterCodec()`).
- `src/NumSharp.Interop.pythonnet/PythonRuntimeInterop.cs` — the session/lease backend (`EnsureEngine`,
  `DrainPending`, the `AcquireGil` policy, pre-resolved callables, interned names + cached literals).

---

## 1 — The two flavors (pick the right one)

A pythonic layer makes call sites read `np.frombuffer(...)` / `Python.np.eval(...)`. It comes in two
shapes; the repo keeps both, for different jobs:

**(A) Typed facade — one C# member per Python call, NO source strings.** Dispatches straight to a
**session-cached, pre-resolved callable** — a veneer over `PythonRuntimeInterop`'s cache, not a second
cache. The C# *reads* like Python but *is* C#. This is `np` in `Pythonic.cs`. It lives in
`namespace NumSharp.Interop.PythonNet` and is **`internal`**, so the lowercase `np`/`ctypes`/`weakref`/
`builtins` can never collide with `NumSharp.np` (or user code) outside the assembly:

```csharp
internal static class np                              // namespace NumSharp.Interop.PythonNet
{
    // np.frombuffer(buffer, dtype) — dtype routed through the cached per-dtype PyString
    internal static PyObject frombuffer(PyObject buffer, NPTypeCode dtype)
        => PythonRuntimeInterop.NpFrombuffer.Invoke(buffer, PythonRuntimeInterop.DtypeString(dtype));
}
```

Plus **extension members on `PyObject`** for the instance-side, spelled as Python (`mv.tobytes("C")`,
`arr.setflags(write:false)`, `mv.shape`) — C# 14 `extension(...)` blocks:

```csharp
internal static class PyObjectPythonic               // namespace NumSharp.Interop.PythonNet
{
    extension(PyObject obj)
    {
        internal void setflags(bool write)           // arr.setflags(write=False)
        {
            using PyObject method = obj.GetAttr(PythonRuntimeInterop.NameSetflags);
            using PyObject none = method.Invoke(write ? PythonRuntimeInterop.TrueLiteral
                                                      : PythonRuntimeInterop.FalseLiteral);
        }
        internal PyObject tobytes(string order)      // mv.tobytes('C') — "C" routes through a cached literal
        {
            using PyObject method = obj.GetAttr(PythonRuntimeInterop.NameTobytes);
            if (order == "C") return method.Invoke(PythonRuntimeInterop.StrC);
            using var o = new PyString(order); return method.Invoke(o);
        }
    }
}
```

**(B) Snippet runner — execute Python SOURCE.** For a body that is a little *program* (a slice, a
transpose, a broadcast, a multi-operand numpy expression) and must **diff 1:1 against real Python /
observe what numpy actually does**, don't translate it to `GetAttr`/`Invoke` chains — run it as text.
This is the test-side `Python.np`, deliberately in **`namespace Python`** (beside `Python.Runtime`) so
`Python.np.eval(...)` reads as "Python's numpy" while `np.*` stays NumSharp:

```csharp
internal static class np                              // namespace Python (test side)
{
    // with named operands pre-bound: let REAL numpy compute over arrays this interop exported
    internal static PyObject with(string npExpr, params (string name, PyObject val)[] vars)
    {
        using var scope = Runtime.Py.CreateScope();
        scope.Exec("import numpy as np");
        foreach (var (name, val) in vars) scope.Set(name, val);   // operands stay caller-owned (scope increfs)
        return scope.Eval(npExpr);                                // result outlives the throwaway scope
    }
    internal static PyObject eval(string npExpr)      { /* CreateScope + Exec("import numpy as np") + Eval */ }
    internal static PyObject evalStmts(string setup, string resultVar) { /* Exec(setup) then Eval(resultVar) */ }
}
```
Call site (the byte-oracle idiom):
```csharp
using (Gil())                                                    // Gil() == Py.GIL()
{
    using PyObject a  = Python.np.eval("np.arange(12, dtype='<i4').reshape(3,4).T");   // numpy makes it
    using NDArray  nd = a.ToNDArray();                                                 // crossed
    ByteContract.AssertSameBytes(nd, a);                                              // byte-for-byte
}
```

**The decision, in one line:** a single clean call → a **facade row** (A: internal, cached callable);
a multi-statement numpy program you want to diff/observe → the **snippet runner** (B:
`Python.np.eval/with/evalStmts`). Don't push single calls into strings, and don't unroll a numpy
expression into `PyObject` chains. Keep *both* on purpose.

---

## 2 — GIL discipline (non-negotiable)

Everything that touches a `PyObject` — including its `Dispose` — runs **under the GIL**: `using
(Py.GIL())` (tests use the `Gil()` helper = `Py.GIL()`). `Py.GIL()` is **re-entrant**.

The **crossing verbs (§4) acquire the GIL themselves** via `NDArrayPythonInterop.AcquireGil(requireGIL)`:
```csharp
internal static IDisposable AcquireGil(bool? requireGIL) => (requireGIL ?? _requireGil) ? Py.GIL() : NoGil;
```
- `requireGIL: null` (the default on every verb) → follows the process-wide `RequireGIL` policy (default
  **true** → takes `Py.GIL()`). So calling `nd.ToTorch()` / `pyobj.ToNDArray()` from a **non-GIL** context
  just works.
- `requireGIL: false` → "I already hold the GIL, don't re-acquire" — the verbs pass this to their own
  nested calls (e.g. `ToTorch` → `source.ToNumpy(copy, requireGIL: false)`). Because the GIL is
  re-entrant, nesting verbs inside your own `using (Py.GIL())` block is always safe; `requireGIL:false`
  just skips the redundant re-acquire.
- Compute the **C# side first (no GIL)**, then open the GIL only for the Python part + the comparison.
- Python→.NET callbacks do **not** inherit the GIL (pythonnet releases it around the managed body). If a
  facade member can run from such a callback, keep GIL management on.
- The **engine must already be initialized** — `PythonRuntimeInterop.EnsureEngine()` throws with guidance
  otherwise. The consuming app/test owns `PythonEngine.Initialize()` (§6), not this package.

---

## 3 — Lifetime & ownership (where the leaks and use-after-frees hide)

- **The session cache is process-scoped — never dispose it.** `PythonRuntimeInterop`'s pre-resolved
  callables (`NpEmpty`, `NpFrombuffer`, `TorchFromNumpy`, `CCharMul`, `WeakrefFinalize`, …), interned
  attribute-name PyStrings (`NameShape`, `NameTobytes`, …), and cached literals (`TrueLiteral`,
  `StrC`, `DtypeString(dtype)`) are reused every call and disposed en masse only on engine `Shutdown`.
  A `using` on one decrefs the shared handle — don't.
- **A returned `PyObject` is owned by the caller — dispose it** (`using`). Operands passed into a scope
  stay owned by the caller (the throwaway scope only increfs them).
- **Fluent chains LEAK — use `using` locals.** `t.detach().cpu().tolist()` drops the `detach()` and
  `cpu()` intermediates on the floor. Spell each step with its own `using`. (Done *right* on purpose: the
  ctypes `SizedArray.from_address` is a deliberate **one-shot** — it disposes the type wrapper in a
  `finally`, so the fluent `ctypes.c_char.mul(n).from_address(addr)` is leak-free by construction.)
- **A Python context manager becomes a disposable scope.** `with …:` → `using` (enter on ctor / import,
  `__exit__`/dispose at block end, both under the GIL) — e.g. `using var scope = Py.CreateScope();`.
- **`DrainPending()` runs for you.** Every verb calls `PythonRuntimeInterop.DrainPending()` first to
  dispose export/import lease handles queued from finalizers that fired off-GIL — which is why exports and
  imports don't accumulate. You don't call it; know it's the reason the leak gate returns to baseline.

---

## 4 — Data crossing: NEVER hand-roll element loops — the verbs do it

This package's **own public API** crosses arrays as a **view** or **one bulk copy**. There is never a
reason to build a numpy array / torch tensor element by element.

**The four numpy verbs** (`NumSharp.Interop.PythonNet.NDArrayPythonInterop`; each takes `bool? requireGIL`):
| Verb | Direction | Copy? |
|---|---|---|
| `nd.ToNumpy()` / `nd.ToNumpy(copy:false)` / `nd.ToPython()` | NDArray → numpy | **zero-copy view** (buffer rooted by a Python `weakref.finalize` on the numpy base) |
| `nd.ToNumpy(copy:true)` / `nd.ToNumpyCopy()` | NDArray → numpy | independent copy |
| `obj.ToNDArray()` | numpy / buffer / adapter → NDArray | copy into a fresh C-contiguous NDArray |
| `NDArrayPythonInterop.ToNDArrayView(obj, allowReadonly)` | numpy → NDArray | zero-copy view (leases the buffer) — **static, not an extension** |

**torch** (`TorchInterop`, imported only when first called): `nd.ToTorch(copy:false)` =
`torch.from_numpy(nd.ToNumpy())` (zero-copy; **throws** on a non-writeable source or negative strides —
pass `copy:true`; `Decimal` has no torch/numpy dtype → float64 copy); `tensor.AsTorchNDArray()` =
zero-copy view of a CPU tensor (via `ToNDArrayView`); `tensor.ToTorchNDArray(force:true)` = the documented
`detach().cpu().resolve_conj().resolve_neg().numpy()` → `ToNDArray` (one bulk copy, detached; crosses
CUDA/MPS/grad/conj tensors).

```csharp
// NumSharp NDArray (its unmanaged buffer) → numpy view → torch tensor: one buffer end-to-end
using PyObject t   = nd.ToTorch();                 // torch.from_numpy(nd.ToNumpy()); zero-copy
NDArray        back = t.ToTorchNDArray(force: true); // torch result → NDArray (one bulk copy; safe to dispose t after)
```

**Copyless from a MANAGED array too — pin it.** An NDArray can be created zero-copy over a managed buffer:
`ArraySlice<T>.FromArray(arr, copy:false)` does `GCHandle.Alloc(arr, GCHandleType.Pinned)` and views the
pinned address (`UnmanagedMemoryBlock<T>.FromArray`). Combined with `ToNumpy`/`ToTorch` the whole chain is
a view of one buffer — so the managed array must be **short-lived and not mutated** by the Python side (pin
lifetime = view lifetime). The old PyList-append / `tolist()`-loop marshaling is the anti-pattern this
replaces.

**Why this is genuinely zero-copy — not just "no loop".** Export builds `ctypes.c_char.mul(nbytes)
.from_address(dataPtr)` over the NDArray's own buffer (`slice.Address + offset`) → `np.frombuffer` →
`np.lib.stride_tricks.as_strided(dims, byteStrides)`, so numpy gets the EXACT layout — strides, offset,
negative strides, Fortran order, and broadcast (stride-0 → `setflags(write=False)`) — with **zero**
per-element objects (**O(1)** in element count), where a PyList/`tolist` loop allocates one Python object
*per element* (O(n) churn + boxing). Import is symmetric: a contiguous exporter is leased with a
`PyBUF.WRITABLE` lock, a non-contiguous numpy array is viewed through `__array_interface__`, a
non-contiguous non-numpy exporter through `PyBUF.STRIDED` — full stride fidelity, read-only sources →
non-writeable views. **Lifetime is handled:** exports hand ownership to an `ExportKeeper` released by a
`weakref.finalize` on the numpy base (survives C# `NDArray` GC; `resize(refcheck=True)` refuses while
exported); imports lease the buffer until the last NumSharp view dies. Only irreducible cases fall back to
a copy, and **loudly** — complex64 widens, UCS-4 text narrows to Char, big-endian is **refused** (never
silently byte-swapped), `Decimal` has no numpy dtype so it copies to float64.

---

## 5 — Codecs: implicit crossing at every boundary (the ergonomic layer)

`NDArrayPythonInterop.RegisterCodec()` (or the public C# 14 extension `Py.RegisterNumSharpCodec()`)
registers `NumpyCodec` (an `IPyObjectEncoder` + `IPyObjectDecoder`) so from then on `NDArray` ⇄ Python
conversion happens **automatically at every pythonnet boundary** — no explicit verb:

```csharp
Py.RegisterNumSharpCodec();               // == NDArrayPythonInterop.RegisterCodec(); per session (idempotent); PROCESS-GLOBAL
scope.Set("x", nd);                        // NDArray -> numpy VIEW implicitly (encode)
using PyObject r = pyfunc.Invoke(nd);      // an NDArray argument auto-encodes
NDArray back = r.As<NDArray>();            // numpy / torch / pandas / list -> NDArray implicitly (decode)
```

**`RegisterNumSharpCodec` is the ONE switch — it enables every support below, and `NumpyCodecOptions` is
the single control surface with all flags defaulting ON.** So torch/pandas, buffers, array-likes and
tuples are all part of this one call — there is NO separate "register the adapters" step. Opt a support
out by passing options (e.g. `DecodeArrayAdapters = false`); the only thing NOT toggled here is adding
your OWN adapter (below).

- **Encode** (`NumpyCodecMode.Auto`, default) is view-first (`ToNumpy`), copy only when unviewable — so it
  stays zero-copy by default.
- **Decode covers far more than numpy:** registered **adapters** (`TorchPythonArrayAdapter`,
  `PandasPythonArrayAdapter` — `DataFrame`/`Series`), **any PEP 3118 buffer**
  (`memoryview`/`bytes`/`bytearray`/`array.array`/ctypes, via the PEP 688 `__buffer__` capability check),
  and **array-like builtins** (`list`/`tuple`/nested/scalar through `numpy.asarray`, always a copy). Modes:
  `Auto` (view→copy) · `View` (decline if no view) · `Copy` (always) — per-direction `NumpyCodecOptions`.
- **`ConvertTuples` (default on)** also registers `TupleCodec`, so a C# `(long,long)` shape crosses as a
  Python `tuple` and back — pythonnet has no tuple conversion of its own.
- **Extending it (the one thing NOT part of `RegisterNumSharpCodec`):**
  `NDArrayPythonInterop.RegisterArrayAdapter(IPythonArrayAdapter)` feeds a NEW library's objects into the
  same memory bridge; idempotent by `IPythonArrayAdapter.Name`. Torch + Pandas are built in and enabled by
  the codec switch (`DecodeArrayAdapters`), so this is only for your OWN adapter — it takes an adapter
  instance, it is not a support toggle, and it is deliberately NOT on the `Py.` facade.
- **Why implicit encoding is leak-safe — `EncoderHandoff`.** pythonnet takes its OWN reference from an
  encoder's wrapper and **never disposes it**, so a naive codec would keep every exported view pinned until
  the CLR finalizer. The codec routes each encode through a one-slot-per-thread handoff that disposes the
  *previous* encode's wrapper — bounding the outstanding pin to one per thread. Invisible to you, but it is
  why implicit encoding at scale doesn't leak.
- Registration is **per engine session** (pythonnet clears codecs on `Shutdown`; `RegisterCodec` re-adds
  after re-init) and **process-global** — it changes conversion for the WHOLE process.

**Explicit verbs (§4) vs codecs — the decision:**
- **Explicit** — local, zero global footprint, per-call view/copy control; the typed `nd.ToPython()` even
  yields numpy **without any registration** (it wins overload resolution over pythonnet's
  `object.ToPython`). **Default for a focused bridge** (build torch tensors explicitly; a codec wouldn't
  touch the torch-encode side anyway).
- **Codecs** — implicit and broad (numpy + torch/pandas + buffers + array-likes + tuples), best when you
  **own the process** or pass lots of numpy/lists and want it ergonomic. The cost is the process-global
  scope: never register from a library that shouldn't dictate conversion for unrelated code.

---

## 6 — The backend & who owns the engine

`PythonRuntimeInterop` (internal, in this package) is the **session/lease manager**, not an engine owner:
`EnsureEngine()` (asserts `PythonEngine.IsInitialized`, throws with guidance otherwise), `DrainPending()`
(off-GIL finalizer-queue drain), the `AcquireGil(requireGIL)` policy (§2), the pre-resolved cached
callables + interned names + cached literals the facade dispatches to (§1/§3), and the export/import lease
lifecycle across engine `Initialize`→`Shutdown` sessions.

**The consuming app/test owns the engine lifecycle.** In the interop suite, `PythonSession`
(`[AssemblyInitialize]`/`[AssemblyCleanup]`) discovers libpython from **`PYTHONNET_PYDLL`** (else probes
`python3`/`python` on PATH; resolves `libpython3.x.{so,dylib}` on unix), sets `Runtime.PythonDLL`, calls
`PythonEngine.Initialize()` + `BeginAllowThreads()`, and shuts down at the end — **once per process**
(pythonnet cannot re-`Initialize` after `Py_Finalize` in-process). After `Initialize()` the GIL is held,
so a suite that computes on the C# thread releases it with `BeginAllowThreads()` and re-takes it per op.

Discovery pin for this machine: `PYTHONNET_PYDLL=C:/Users/ELI/.claude/python/python312.dll`.

---

## 7 — Gotchas (each cost real time)

- **Extension members need C# 14.** The `extension(...)` facade members and the public
  `Py.RegisterNumSharpCodec()` extension require LangVersion 14 — a consumer on the SDK-default LangVersion
  (even against this package's net8.0 target) fails with **`CS9202: Feature 'extensions' is not
  available`**. Such consumers call the underlying `NDArrayPythonInterop.*` / `TorchInterop.*` methods
  directly — identical, and they compile on every C# version.
- **A numpy scalar is not a Python `int`.** pythonnet's primitive conversions never consult codecs, so
  `new PyInt(arr[i])` on an int64 array throws — use `int(arr[i])` / `.item()`. numpy floats ARE Python
  floats; a 0-d ndarray is neither (`float(z0)`).
- **Don't dispose the session cache** (§3): the pre-resolved callables, interned name PyStrings, and cached
  literals are shared and reused every call.
- **`using static` does NOT bring in extension members or keep the facade prefix.** To read
  `Python.np.eval(...)` use the namespace form (`using Python;`); inside the package the facade is reached
  because you're in `namespace NumSharp.Interop.PythonNet`. `using static` would also drop the `np.` prefix
  that makes it read like the API.
- **Never build arrays/tensors element by element.** No PyList-append loops in, no `tolist()` loops out —
  §4's crossing does it as a view or one bulk copy.
- **`GetEnumerator()` must not return `this`** for an object that owns unmanaged/GIL state — `foreach`
  disposes the enumerator (closes the iterator). Hand out a thin wrapper (see NumSharp's `nditer`).
- **The facade is a veneer, not a second cache** — every member resolves through `PythonRuntimeInterop`'s
  session cache and calls `Invoke` directly. Don't add a parallel per-call attribute walk or a second cache.
- **Keep flavor-B snippets diffable against real Python** — `import numpy as np` in the scope, port
  near-verbatim, don't paraphrase; that's what makes them a faithful oracle.

---

## Definition of done

The interop reads like the Python it runs: single calls are **facade rows** or **`PyObject` extension
members** (internal, `namespace NumSharp.Interop.PythonNet` for the package; `namespace Python` for the
test side, `using Python;`), multi-statement numpy programs are **`Python.np.eval/with/evalStmts`
snippets** that diff 1:1 against real numpy; every `PyObject` touch is under the GIL (`Py.GIL()` /
`AcquireGil(requireGIL)`); the session cache is never disposed and no fluent chain leaks an intermediate;
and **all** array/tensor data crosses through the zero-copy verbs (`nd.ToNumpy()`/`nd.ToTorch()` in,
`ToNDArray`/`ToTorchNDArray`/`ToNDArrayView` out) or a registered `NumpyCodec` — **never** a hand-rolled
element loop. Validate with the live interop gate (`ByteContract` byte-for-byte vs real numpy/torch) on
both TFMs; a leak gate that returns to baseline proves the lifetimes.
