# Numpy.NET — Coexistence & Migration

[Numpy.NET](https://github.com/SciSharp/Numpy.NET) (the SciSharp `Numpy` / `Numpy.Bare` packages) wraps Python's numpy behind a generated C# API — its `NDarray` type is a thin handle around a live `numpy.ndarray`. Plenty of existing .NET codebases are built on it.

This page proves that **NumSharp and Numpy.NET interoperate zero-copy in the same process, on the same interpreter, over the same buffers** — so you can adopt NumSharp incrementally inside a Numpy.NET codebase (or keep using Numpy.NET's API surface over NumSharp-owned data) without a single element copied.

Every section is *proven*: it corresponds 1:1 to a test in [`test/NumSharp.Interop.UnitTests/NumpyNetInteropTests.cs`](https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/NumpyNetInteropTests.cs), executed against live packages on every run (verified with `Numpy.Bare 3.11.1.33` on Python 3.12 + numpy 2.4.2 — newer than anything Numpy.NET was built against, which is rather the point). Every test also runs under the suite's leak gate, so each pattern is proven to release every reference it takes.

---

## Why this works (and why NumSharp.Interop.pythonnet has no Numpy.NET dependency)

Numpy.NET's `NDarray` **is** a `PyObject` — `PythonObject.self` is public, and `NDarray` has a public `NDarray(PyObject)` constructor. Since [NumSharp.Interop.pythonnet](pythonnet.md) speaks raw `PyObject` + buffer protocol, the two meet in the middle with two one-liners:

```csharp
using Numpy;                  // Numpy.NET
using NumSharp.Interop.PythonNet;  // the bridge
using np2 = Numpy.np;         // NumSharp's np wins bare-name lookup; alias theirs

// Wrap: Numpy.NET's ENTIRE C# API now operates over NumSharp's buffer, zero-copy
NDarray wrapped = new NDarray(numsharpArray.ToNumpy());

// Unwrap: NumSharp kernels now run over Numpy.NET's array, zero-copy
NDArray view = numpyNetArray.self.AsNDArray();      // or .ToNDArray() for a copy
```

The bridge deliberately does **not** reference the Numpy.NET packages: `Numpy` and `Numpy.Bare` ship the *same types in different assemblies* (referencing both is a CS0433 compile error), and each release pins a Python/wheel combination. Binding to `PyObject` sidesteps all of it — and is exactly what makes the two one-liners above possible against either flavor.

> Reference **one** flavor in your app: `Numpy` (bundles Python via Python.Included) or `Numpy.Bare` (uses the Python your process provides). Never both.

---

## One engine, shared peacefully

Numpy.NET initializes lazily: its first call runs `PythonEngine.Initialize()` — a no-op if your app (or this bridge) already initialized — and then simply `Py.Import("numpy")` into the *same interpreter*. There is no second engine and no hidden state:

```csharp
Runtime.PythonDLL = @"C:\Python312\python312.dll";
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();

using (Py.GIL())
{
    NDarray a = np2.arange(6);      // Numpy.NET call — binds to the engine above
}
// NumSharp.Interop.pythonnet scopes, codecs and conversions keep working untouched.
```

> Proven by `NumpyNet_BootsOnTheSharedEngine`.

### ⚠ The GIL rule

Numpy.NET performs **no GIL management of its own** — it assumes the GIL stays held after `Initialize()`. The moment your app calls `PythonEngine.BeginAllowThreads()` (which multi-threaded apps and this bridge's test host do), **every Numpy.NET call — including `NDarray.Dispose()` — must be wrapped in `using (Py.GIL())`**. NumSharp.Interop.pythonnet's own API self-acquires the GIL; Numpy.NET's does not. All samples on this page follow the rule.

---

## Wrap: Numpy.NET's API over NumSharp memory

```csharp
var ours = np.arange(6).astype(NPTypeCode.Double);

using (Py.GIL())
{
    using var wrapped = new NDarray(ours.ToNumpy());

    double sum = wrapped.sum().item<double>();   // 15.0 — computed by Numpy.NET

    // the aliasing is real, both ways:
    // a NumSharp write is visible through wrapped.item<double>(...),
    // and wrapped.fill(7.0) is visible from NumSharp's side.
    var data = wrapped.GetData<double>();        // their copy-out API works too
}
```

numpy itself confirms the sharing: `np.shares_memory(wrapped, second_export_of_ours)` is `True` — one buffer with three façades (NumSharp, Numpy.NET, any scope variable).

> Proven by `Wrap_NumSharpBuffer_DrivenByNumpyNetApi_ZeroCopy`.

---

## Unwrap: NumSharp kernels over Numpy.NET arrays

```csharp
NDarray their;
using (Py.GIL())
    their = np2.arange(8).astype(np2.float64);

NDArray view;
using (Py.GIL())
    view = their.self.AsNDArray();       // zero-copy lease on their buffer

// NumSharp writes -> their.item<double>(1) sees it
// their.fill(3.25) -> NumSharp reads see it
double total = (double)np.sum(view);     // NumSharp kernel over Numpy.NET's memory
```

> Proven by `Unwrap_NumpyNetArray_AsNumSharpView_SharedMutation`.

---

## Dtypes and slices survive the boundary

Wrapped NumSharp arrays report the exact numpy dtype names to Numpy.NET (`float64`, `float32`, `int32`, `int64`, `uint8`, `bool`, ...) and `GetData<T>()` returns the right values; Numpy.NET arrays arrive in NumSharp with the exact `NPTypeCode`.

Slices cross in both directions:

```csharp
// their slice (a numpy view) -> our leased view; writing it hits THEIR base array
NDarray slice = their["2:8"];
NDArray sliceView = slice.self.AsNDArray();

// our strided view -> their API; GetData<T> linearizes non-contiguous layouts correctly
var v = b["1:3, ::2"];                              // NumSharp strided window
using var w = new NDarray(v.ToNumpy());
double[] logical = w.GetData<double>();             // logical order, not raw buffer order
```

> Proven by `DtypeFidelity_AcrossBothWrapDirections` and `Slices_CrossTheBoundary_InBothDirections`.

---

## Lifetimes: who keeps what alive

The [bridge's lifetime model](pythonnet.md#lifetime--memory-safety) extends across Numpy.NET transparently, in both directions:

- **A Numpy.NET wrapper can be the only holder of NumSharp memory.** Create `new NDarray(nd.ToNumpy())`, drop every NumSharp-side reference, collect aggressively — the wrapper still computes over valid memory. Disposing it releases the pin (`LiveExports` drains to zero).
- **A NumSharp view can outlive the Numpy.NET wrapper.** Take `their.self.AsNDArray()`, then `their.Dispose()` — the lease keeps the numpy array alive; reads and writes stay valid until the last NumSharp view dies.

```csharp
using (Py.GIL())
    wrapped.Dispose();    // remember the GIL rule — their Dispose decrefs without taking it
```

> Proven by `Lifetime_TheirWrapperIsTheOnlyHolder_OfNumSharpMemory` and `Lifetime_OurViewOutlivesTheirDisposedWrapper`.

---

## Compute pipelines, cross-checked

A NumSharp-owned pipeline can hand matrices to Numpy.NET operations and take results back — element-for-element equal to NumSharp computing the same thing itself:

```csharp
using (Py.GIL())
{
    using var wa = new NDarray(a.ToNumpy());
    using var wb = new NDarray(b.ToNumpy());
    using var product = np2.matmul(wa, wb);      // Numpy.NET drives numpy
    NDArray result = product.self.ToNDArray();   // back to NumSharp
    // result == np.matmul(a, b) computed by NumSharp, verified per element
}
```

> Proven by `Compute_NumpyNetPipeline_CrossCheckedAgainstNumSharp`.

---

## The codec sees their arrays too

A Numpy.NET `NDarray` *is* a `numpy.ndarray`, so with [`NDArrayPythonInterop.RegisterCodec()`](pythonnet.md#auto-marshaling-the-codec) registered, `their.self.As<NDArray>()` decodes like any numpy array — under the default [`Auto`](zero-copy-model.md) mode a contiguous source comes back as a **zero-copy view**, so a NumSharp write lands in their array. The same object can simultaneously live in a plain pythonnet scope: the scope name, the Numpy.NET wrapper and numpy itself are one object. Pass `DecodeMode = NumpyCodecMode.Copy` if you want a detached snapshot instead.

> Proven by `Codec_DecodesNumpyNetArrays_AndScopesInterleave`.

---

## Version notes

| Fact | Detail |
|---|---|
| Verified combination | `Numpy.Bare 3.11.1.33` + pythonnet 3.0.5 + Python 3.12.12 + numpy 2.4.2 |
| pythonnet | Numpy.NET depends on pythonnet 3.0.1; the bridge's floor is `[3.0.5, 4.0.0)`, and NuGet unifies the two upward to 3.0.5+ — which is also what lets the pair run Python 3.12+ |
| numpy version | Numpy.NET binds `Py.Import("numpy")` — it uses whatever numpy your interpreter has. Core array APIs work fine on numpy 2.x; some rarely-used generated wrappers may hit numpy 2.0 API removals |
| `Numpy` vs `Numpy.Bare` | Same types, different assemblies — reference exactly one (CS0433 otherwise). `Bare` is the natural fit when your app already manages Python |
| Name clash | Both libraries define `np` — alias one: `using np2 = Numpy.np;` |

## Running the proof yourself

```bash
dotnet test test/NumSharp.Interop.UnitTests/NumSharp.Interop.UnitTests.csproj --filter "ClassName~NumpyNetInteropTests"
```
