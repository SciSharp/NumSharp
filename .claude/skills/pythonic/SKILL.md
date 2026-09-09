---
name: pythonic
description: Write a "pythonic" pythonnet interop layer — a root `static class Python` whose nested subclasses form a tree that makes C# driving embedded CPython read like the Python it drives, plus the GIL / PyObject-lifetime discipline and the ZERO-COPY NDArray⇄numpy⇄torch crossing (via NumSharp's `NumSharp.Interop.pythonnet`, so you never hand-roll element loops). Use when building or editing any pythonnet bridge, adding a `Python.*` facade member or a nested class that exposes a Python object's methods/properties, crossing array/tensor data between C# and numpy/torch, deciding typed-facade vs `evalStmts` snippet or explicit-verb vs codec (RegisterCodec/NumpyCodec) crossing, or debugging a GIL / leak / lifetime / "numpy scalar isn't a Python int" bug in the interop. keywords: pythonic layer, static class Python tree, Python.* facade, pythonnet bridge, make C# read like Python, nested class for object members, zero-copy tensor crossing, NDArray to torch, ToTorch, ToTorchNDArray, GIL discipline, PyObject lifetime, evalStmts snippet runner. keywords-sparse: pythonic interop, static class Python, Python.* surface, zero-copy numpy/torch crossing, GIL and PyObject lifetime, typed facade vs evalStmts, nested class not extensions
---

# pythonic — the `Python.*` interop layer (read like Python, cross data zero-copy)

How to drive **embedded CPython** from C# (via pythonnet) so the C# reads like the Python it drives — and
how to move arrays/tensors across the boundary **without copying, element by element, ever**. Two
ingredients: a small **library-agnostic backend** that owns engine startup + the GIL + a module-import
cache, and NumSharp's **`NumSharp.Interop.pythonnet`** zero-copy crossing. Everything below is the pattern;
adapt the module/class names to whatever library your bridge drives.

**The shared references** (SciSharp foundation, not tied to any one bridge):
- `NumSharp.Interop.pythonnet` — the zero-copy `NDArray`⇄numpy⇄torch/pandas crossing (§4/§5) every bridge
  builds on. Its own `Pythonic.cs` is a worked **typed facade** and its test-side one adds the
  `eval`/`with`/`evalStmts` **snippet runner** — read them for the crossing and for flavor B. (NumSharp
  spells its own facade through a *namespace + C# extension members*; that is a different surface style —
  see §1/§7 — so copy it for the crossing, not for the tree.)

---

## 1 — The two flavors (pick the right one)

The pythonic surface is a root **`static class Python`** whose **nested subclasses form a tree** reaching
each Python call, so call sites read `Python.torch.tensor(...)` / `Python.somelib.evalStmts(...)`. It is
always spelled **`Python.*`** — never a `using` of a namespace, never an extension method. The class
shadows pythonnet's `Python.Runtime` namespace in scope, so the surface stays explicit and collision-free,
and any Python object whose methods/properties you need is exposed as **a nested class under `Python`**
(not an extension). It comes in two shapes; use both, for different jobs:

**(A) Typed facade — one C# member per Python call, NO source strings.** Nested static classes that
dispatch straight to a cached module/callable. The C# *reads* like Python but *is* C#.

```csharp
internal static class Python                        // the root; always spelled Python.*
{
    internal static class torch
    {
        private static PyObject Module => Import("torch");                 // backend's cached, process-scoped import
        internal static PyObject float64 => Module.GetAttr("float64");     // caller disposes
        internal static PyObject tensor(PyObject data)                     // Python.torch.tensor(data, …)
        {
            using PyObject fn = Module.GetAttr("tensor");
            using PyObject dtype = float64;
            using var kwargs = new PyDict(); kwargs["dtype"] = dtype;
            return fn.Invoke(new[] { data }, kwargs);                      // torch.tensor(data, dtype=float64)
        }
    }
}
```

**A Python object's members go in a nested class — NOT an extension.** Expose them as unbound calls that
take the object, mirroring how Python itself spells them (`torch.Tensor.size(t, -1)`). **Prefer returning
`PyObject`** (stay in Python-land, composable); add a **`_t`-suffixed typed variant only where a final CLR
value leaves Python** and it would otherwise be an ambiguous return-type-only overload:

```csharp
internal static class torch                          // nested under Python
{
    internal static class Tensor                     // Python.torch.Tensor.*
    {
        internal static PyObject size(PyObject t, long dim)               // Python.torch.Tensor.size(t, -1)
        { using PyObject d = new PyInt(dim); return t.InvokeMethod("size", d); }
        internal static PyObject squeeze(PyObject t, long dim)            // caller disposes
        { using PyObject d = new PyInt(dim); return t.InvokeMethod("squeeze", d); }
        internal static long size_t(PyObject t, long dim)                 // typed — a final CLR value
        { using PyObject d = new PyInt(dim); using PyObject s = t.InvokeMethod("size", d); return s.As<long>(); }
    }
}
```

**(B) Snippet runner — execute multi-statement Python SOURCE.** For a body that is a little *program*
(control flow, kwarg-heavy constructor calls) and must **diff 1:1 against the reference Python**, don't
translate it to `GetAttr`/`Invoke` chains — run it as text:

```csharp
internal static PyObject evalStmts(string setup, string resultVar, params (string name, PyObject val)[] vars)
{
    using PyModule scope = Py.CreateScope();
    scope.Exec(Imports);                              // the `import …` / `from … import …` the body needs
    foreach ((string name, PyObject val) in vars) scope.Set(name, val);
    scope.Exec(setup);                                // the multi-statement body
    return scope.Eval(resultVar);
}
```
The body lives as a `const string` that reads 1:1 like the source Python it ports:
```csharp
private const string FitBody =
    "model = SomeModel(train_x, train_y, transform=Standardize(m=train_y.size(-1)))\n" +
    "fit(model)\n" +
    "result = optimize(acqf(model), bounds=bounds)\n" + …;
```

**The decision, in one line:** a single clean call → a **facade row** (A); a multi-statement program you
want to diff against the Python source → the **snippet runner** (B). Don't push single calls into strings,
and don't unroll multi-statement Python into `PyObject` chains. Keep *both* on purpose.

**Only FIXED calls go in the `Python.*` tree.** A call whose name you know at compile time — a module
function, a constructor, an object's protocol method — becomes a facade row / nested-class member. A call
whose target is chosen at RUNTIME — an arbitrary user-loaded class (`module.GetAttr(className)` +
instantiate), a `HasAttr` presence probe, `__name__`/type reflection, generic `dict.items()` / `__len__`
iteration — stays raw at the call site (it can't be a static facade). A plugin loader is the worked
example: `Python.lib.make_thing(...)` is a row; `module.GetAttr(userClassName).Invoke(...)` stays raw.

---

## 2 — GIL discipline (non-negotiable)

Everything that touches a `PyObject` — including its `Dispose` — runs **under the GIL**. Route every
Python-touching body through a backend `Run` helper that holds `Py.GIL()` (and starts the engine if needed):

```csharp
double best = Run(() =>                               // holds Py.GIL() for the whole body
{
    using PyObject t = ToTensor(matrix);             // your bridge's marshaling helper
    return Python.torch.Tensor.size_t(t, 0);
});
```
- Compute the **C# side first (no GIL)**, then open the GIL only for the Python part + the comparison.
- `Py.GIL()` is **re-entrant** — NumSharp's crossing re-acquires it internally, so calling `ToTorch()` /
  `ToTorchNDArray()` *inside* a `Run` body is fine.
- Python→.NET callbacks do **not** inherit the GIL (pythonnet releases it around the managed body). If a
  facade member can run from such a callback, keep GIL management on.

---

## 3 — Lifetime & ownership (where the leaks and use-after-frees hide)

- **Cached modules are process-scoped — never dispose them.** The backend's `Import("torch")` returns a
  session-owned handle; a `using` on it decrefs the cache. Long-lived singletons (a dtype object) may be
  cached the same way; on a race, **loser-dispose** (`ConcurrentDictionary.TryAdd`, dispose the loser).
- **A returned `PyObject` is owned by the caller — dispose it** (`using`). Bound operands passed into a
  scope stay owned by the caller (the throwaway scope only increfs).
- **Fluent chains LEAK — use `using` locals.** `t.detach().cpu().tolist()` drops the `detach()` and
  `cpu()` intermediates on the floor (the interop leak gate goes red). Spell each step:
  ```csharp
  using PyObject detached = t.detach();
  using PyObject cpu = detached.cpu();
  using PyObject list = cpu.tolist();
  ```
  (This is why facade members return owned `PyObject`s and the docs say "dispose it".)
- **A context manager becomes a disposable scope.** A Python `with obj(...):` → an `IDisposable` that
  enters on the ctor and calls `__exit__` on `Dispose` (both under the GIL), used as
  `using (Python.lib.obj(...)) …`.

---

## 4 — Data crossing: NEVER hand-roll element loops — use NumSharp's interop

Add the package (single-source the version with the rest of your NumSharp stack):
```xml
<PackageReference Include="NumSharp.Interop.pythonnet" Version="$(NumSharpVersion)" />
```
It **requires an already-running engine** (`EnsureEngine` throws otherwise) and wires onto the SAME
embedded engine your backend starts — it does NOT start its own, and its GIL is re-entrant. Two ways to
cross data: the **explicit verbs** here (local, no global side effect — the right default for a bridge), or
**codecs** that convert implicitly at every boundary (§5).

**The four verbs** (`NumSharp.Interop.PythonNet.NDArrayPythonInterop`):
| Verb | Direction | Copy? |
|---|---|---|
| `nd.ToNumpy()` | NDArray → numpy | **zero-copy view** (rooted by a Python `weakref.finalize`) |
| `nd.ToNumpyCopy()` | NDArray → numpy | independent copy |
| `pyobj.ToNDArray()` | numpy/adapter → NDArray | copy into a fresh C-contiguous NDArray |
| `NDArrayPythonInterop.ToNDArrayView(pyobj)` | numpy → NDArray | zero-copy view (leases the buffer; **static**, not an extension) |

**torch** (`TorchInterop`): `nd.ToTorch(copy:false)` = `torch.from_numpy(nd.ToNumpy())` (zero-copy view;
NumSharp roots its buffer for the tensor's life); `pyTensor.ToTorchNDArray(force:true)` = the documented
`detach().cpu().resolve_conj().resolve_neg().numpy()` → `ToNDArray` (one bulk copy, detached);
`pyTensor.AsTorchNDArray()` = zero-copy view of a CPU tensor.

**Copyless from a MANAGED array — `copy: false` PINS it.** `np.array(data, copy: false)` does
`GCHandle.Alloc(data, GCHandleType.Pinned)` and views the pinned address (see
`UnmanagedMemoryBlock.FromArray`), so combined with `ToTorch` the whole chain is a view of one buffer:

```csharp
// managed double[,]  →(pin)→ NDArray view →(np.frombuffer)→ numpy view →(from_numpy)→ torch tensor
static PyObject ToTensor(double[,] data) => np.array(data, copy: false).ToTorch();
// torch result → NDArray (one bulk copy, safe to dispose the tensor after)
static NDArray  ToNDArray(PyObject tensor) => tensor.ToTorchNDArray(force: true);
```
NumSharp's `ExportKeeper` + `weakref.finalize` hold the pin until the last Python view dies, then
`UnmanagedMemoryBlock`'s `Disposer` calls `GCHandle.Free()`. So the managed array must be **short-lived
and not mutated** by the Python side (pin lifetime = tensor lifetime, released on Python GC) — true for
per-call matrices. The old PyList-append / `tolist()`-loop marshaling is the anti-pattern this replaces.

**Why this is genuinely zero-copy — not just "no loop".** Export is `ctypes.c_char.from_address(ptr)` →
`np.frombuffer` → `np.lib.stride_tricks.as_strided(dims, byteStrides)`, so numpy gets the EXACT layout —
strides, offset, negative strides, Fortran order, and broadcast (stride-0 → `setflags(write=False)`) —
with **zero** per-element objects (**O(1)** in element count), where a PyList/`tolist` loop allocates one
Python object *per element* (O(n) churn + boxing). Import is symmetric: a contiguous exporter is leased
with a `PyBUF.WRITABLE` lock, a non-contiguous numpy array is viewed through `__array_interface__`, a
non-contiguous non-numpy exporter through `PyBUF.STRIDED` — full stride fidelity, read-only sources → non-
writeable views. **Lifetime is handled:** exports take an ARC ref released by a `weakref.finalize` on the
numpy base (survives C# `NDArray` GC; `resize(refcheck=True)` refuses while exported); imports lease the
buffer until the last NumSharp view dies. Only irreducible cases fall back to a copy, and **loudly** —
complex64 widens, UCS-4 text narrows to Char, big-endian is **refused** (never silently byte-swapped).

---

## 5 — Codecs: implicit crossing at every boundary (the ergonomic layer)

`NDArrayPythonInterop.RegisterCodec()` registers `NumpyCodec` (an `IPyObjectEncoder` + `IPyObjectDecoder`),
so from then on `NDArray` ⇄ Python conversion happens **automatically at every pythonnet boundary** — no
explicit verb:

```csharp
NDArrayPythonInterop.RegisterCodec();     // once per engine session (idempotent); PROCESS-GLOBAL
scope.Set("x", nd);                       // NDArray -> numpy VIEW implicitly (encode)
using PyObject r = pyfunc.Invoke(nd);     // an NDArray argument auto-encodes
NDArray back = r.As<NDArray>();           // numpy / torch / pandas / list -> NDArray implicitly (decode)
```

- **Encode** (`NumpyCodecMode.Auto`, default) is view-first (`ToNumpy`), copy only when unviewable — so it
  stays zero-copy by default.
- **Decode covers far more than numpy:** registered **adapters** (`torch.Tensor`, Pandas
  `DataFrame`/`Series`/…), **any PEP 3118 buffer** (`memoryview`/`bytes`/`bytearray`/`array.array`/ctypes,
  via the PEP 688 `__buffer__` capability check), and **array-like builtins** (`list`/`tuple`/nested/scalar
  through `numpy.asarray`, always a copy). Modes: `Auto` (view→copy) · `View` (decline if no view) · `Copy`
  (always) — per-direction `NumpyCodecOptions` toggles.
- **`ConvertTuples` (default on)** also registers `TupleCodec`, so a C# `(long,long)` shape crosses as a
  Python `tuple` and back — pythonnet has no tuple conversion of its own.
- **Why implicit encoding is leak-safe — `EncoderHandoff`.** pythonnet takes its OWN reference from an
  encoder's wrapper and **never disposes it**, so a naive codec would keep every exported view pinned until
  the CLR finalizer (a loop of 150 encodes → 150 live exports). The codec routes each encode through a
  one-slot-per-thread handoff that disposes the *previous* encode's wrapper — bounding the outstanding pin
  to one per thread. Invisible to you, but it is why implicit encoding at scale doesn't leak.
- Registration is **per engine session** (pythonnet clears codecs on `Shutdown`; `RegisterCodec` re-adds
  after re-init) and **process-global** — it changes conversion for the WHOLE process.

**Explicit verbs (§4) vs codecs — the decision:**
- **Explicit** — local, zero global footprint, per-call view/copy control; the typed `nd.ToPython()` even
  yields numpy **without any registration** (it wins overload resolution over pythonnet's `object.ToPython`).
  **Default for a focused bridge** (build torch tensors explicitly; a codec wouldn't touch the torch-encode
  side anyway).
- **Codecs** — implicit and broad (numpy + torch/pandas + buffers + array-likes + tuples), best when you
  **own the process** or pass lots of numpy/lists and want it ergonomic. The cost is the process-global
  scope: never register from a library that shouldn't dictate conversion for unrelated code.

---

## 6 — The generic backend vs library specifics

Keep a **library-agnostic backend** (one per process) that owns: `Initialize()` (idempotent,
process-scoped; honors `PYTHONNET_PYDLL`, else probes `python`/`python3` for libpython + injects
site-packages so `pip install`ed packages import), `Run(Action)`/`Run<T>(Func<T>)` (the GIL entry),
`Import(name)` (cached), `Builtins`. It does **no** startup imports and hangs **nothing** on pythonnet's
`Py` — there is no `Py.*` extension; the pythonic surface is the `Python.*` tree. Library-specific
accessors (discovered versions, log-quieting, cached module handles) layer on top in their own runtime
class. Give the bridge a typed exception base (e.g. `PythonInteropException`) and derive per-library ones
from it.

Pin discovery when a machine has several interpreters: `PYTHONNET_PYDLL=/path/to/libpython3.x.{dll,so,dylib}`.

---

## 7 — Gotchas (each cost real time)

- **Expose object members as a nested class under `Python`, NOT as extensions.** A tensor's `.size()` /
  `.dim()` become `Python.torch.Tensor.size_t(t, -1)` / `Python.torch.Tensor.dim_t(t)` (unbound, taking the
  object) — mirroring Python's own `torch.Tensor.size(t, -1)`. A `static` extension is allowed only as an
  **`internal`** helper (never a public `Py.*` / `PyObject` surface). Prefer returning `PyObject`; add a
  `_t`-suffixed typed variant only where a final CLR value leaves Python and it would otherwise be an
  ambiguous return-type-only overload.
- **The root `static class Python` shadows the `Python.Runtime` namespace** in scope, so `Python.torch…`
  binds to the class. Reference pythonnet types via `Py` / `PyObject` (from `using Python.Runtime;`), never
  a `Python.Runtime.X` qualifier inside a file where `Python` is the class.
- **A numpy scalar is not a Python `int`.** pythonnet's primitive conversions never consult codecs, so
  `new PyInt(arr[i])` on an int64 array throws — use `int(arr[i])` / `.item()`. numpy floats ARE Python
  floats; a 0-d ndarray is neither (`float(z0)`).
- **Don't dispose cached module handles** (§3). And don't dispose `PyObject.None` casually — it's a shared
  singleton; follow the reference code.
- **`GetEnumerator()` must not return `this`** for an object that owns unmanaged/GIL state — `foreach`
  disposes the enumerator (closes the iterator). Hand out a thin wrapper (see NumSharp's `nditer`).
- **Never build arrays/tensors element by element.** No PyList-append loops in, no `tolist()` loops out —
  §4's crossing does it as a view or one bulk copy.
- **Keep the reference Python close.** Snippet bodies (flavor B) must diff 1:1 against the source Python you
  port — port them near-verbatim, `import`s in the scope, don't paraphrase.

---

## Definition of done

The bridge reads like the Python it drives: single calls are **facade rows** or **nested-class object
members** under the root `static class Python` (`Python.torch.Tensor.size_t(t, -1)` — never an extension),
multi-statement bodies are **`evalStmts` snippets** that diff 1:1 against the source; every `PyObject`
touch is under the GIL via the backend `Run` helper; every returned `PyObject` is `using`-disposed and no
fluent chain leaks an intermediate; and **all** array/tensor data crosses through NumSharp's zero-copy
`NumSharp.Interop.pythonnet` — the explicit verbs (`np.array(x, copy:false).ToTorch()` in, `ToTorchNDArray`
out) or a registered codec — **never** a hand-rolled element loop. Validate with a live interop gate
(byte-exact vs the real library) on every TFM; a leak gate that returns to baseline proves the lifetimes.
