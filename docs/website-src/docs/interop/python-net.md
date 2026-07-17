# Python.NET — Unit-Test-Proven Patterns

This page is a cookbook of **idiomatic Python.NET** (embedding scopes, `dynamic` modules, runtime-defined functions and classes, iterators, exceptions, threads) with NumSharp arrays flowing through all of it.

Every section is *proven*: it corresponds 1:1 to a test in [`test/NumSharp.Interop.UnitTests/PythonNetUsageTests.cs`](https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/PythonNetUsageTests.cs), executed against a live CPython + numpy on every run. If a pattern on this page breaks, the suite fails — and because every test runs under the suite's leak gate, each pattern is also proven to release every reference it takes (`PythonConvert.LiveExports` / `LiveImports` must return to baseline).

The samples assume the [NumSharp.Interop.pythonnet](pythonnet.md) package and an initialized engine:

```csharp
using NumSharp;
using NumSharp.Interop;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";   // or the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();
```

---

## Scopes: variables that flow both ways

Bind a zero-copy view into a `PyModule` scope; Python computes over NumSharp's buffer and writes back into it — and NumSharp's later writes are visible to the scope variable.

```csharp
var nd = np.arange(6).astype(NPTypeCode.Double).reshape(2, 3);

using (Py.GIL())
{
    using var scope = Py.CreateScope();
    using (PyObject x = nd.ToNumpy())
        scope.Set("x", x);

    scope.Exec("total = float(x.sum())\nx[0, 0] = 100.0");
    // total == 15.0, and nd[0, 0] is now 100.0 — same memory.
}
```

> Proven by `Embedding_ScopeVariables_FlowBothWays`.

---

## Dynamic modules: numpy functions on NumSharp arrays

With the codec registered (`PythonConvert.RegisterCodec()`), `NDArray`s pass straight into `dynamic` module calls and results decode back — no explicit conversion anywhere.

```csharp
PythonConvert.RegisterCodec();

var a = np.arange(4).astype(NPTypeCode.Double).reshape(2, 2);
var b = (np.arange(4).astype(NPTypeCode.Double) + 1).reshape(2, 2);

using (Py.GIL())
{
    dynamic numpy = Py.Import("numpy");
    NDArray product = ((PyObject)numpy.matmul(a, b)).As<NDArray>();
    // product agrees element-for-element with np.matmul(a, b) computed by NumSharp
}
```

> Proven by `DynamicModules_NumpyFunctions_OnNumSharpArrays` (the test cross-checks against NumSharp's own `np.matmul`).

---

## The Python standard library: a JSON round trip

The bridge isn't numpy-only — any Python library that touches array data participates. Here NumSharp data travels through `json` and back losslessly:

```csharp
var nd = np.arange(5).astype(NPTypeCode.Double) * 1.5;
// scope: jx = <zero-copy view of nd>

scope.Exec("import json\npayload = json.dumps(jx.tolist())");
// payload == "[0.0, 1.5, 3.0, 4.5, 6.0]"

using PyObject parsed = scope.Eval("np.asarray(json.loads(payload), dtype='f8')");
NDArray back = parsed.ToNDArray();   // bit-equal to nd
```

> Proven by `PythonStdlib_JsonRoundtrip`.

---

## Runtime-defined functions

Define Python at runtime, fetch it as a callable, and call it with `NDArray` arguments:

```csharp
scope.Exec("def zscore(a):\n    return (a - a.mean()) / a.std()");

var samples = np.arange(8).astype(NPTypeCode.Double);
using (Py.GIL())
{
    dynamic zscore = scope.Get("zscore");
    NDArray scored = ((PyObject)zscore(samples)).As<NDArray>();
    // equals (samples - np.mean(samples)) / np.std(samples) computed in NumSharp
}
```

> Proven by `RuntimeDefinedFunction_TypedResults`.

---

## Python classes that store your arrays

A Python object can hold references to passed arrays long-term. Because the codec encodes **views**, the stored batches are live — a later C#-side mutation changes what the object computes:

```csharp
scope.Exec(@"
class Accumulator:
    def __init__(self):
        self.batches = []
    def add(self, arr):
        self.batches.append(arr)      # stores the SHARED view
    def total(self):
        return float(sum(b.sum() for b in self.batches))
");

using (Py.GIL())
{
    dynamic acc = scope.Eval("Accumulator()");
    acc.add(first);                   // NDArray, auto-encoded as a view
    acc.add(second);
    double t1 = (double)acc.total();  // 66.0

    // mutate first[0] on the C# side...
    double t2 = (double)acc.total();  // ...and the accumulator sees it: 166.0
}
// dropping the instance releases the stored views — nothing leaks
```

> Proven by `PythonClass_StoresArrays_SeesLaterMutations`.

---

## Exceptions across the boundary

Python exceptions surface as `PythonException` with the original message; a failed call leaves the engine — and every live conversion — fully usable:

```csharp
try
{
    scope.Exec("raise ValueError('bad shape: ' + str(ex.shape))");
}
catch (PythonException e)
{
    // e.Message contains "bad shape: (3,)"
}

scope.Exec("ex[0] = 7.5");   // the shared view still works after the failure
```

> Proven by `PythonExceptions_SurfaceAsPythonException_EngineStaysUsable`.

---

## Generators: streaming chunks into NumSharp

Iterate a Python generator with `PyIter` and materialize each yielded array; stack the stream with NumSharp:

```csharp
scope.Exec(@"
def chunks(n):
    for i in range(n):
        yield np.arange(3, dtype='f8') + 10 * i
");

var rows = new List<NDArray>();
using (Py.GIL())
{
    using PyObject generator = scope.Eval("chunks(4)");
    using var iterator = PyIter.GetIter(generator);
    while (iterator.MoveNext())
    {
        using PyObject chunk = iterator.Current;
        rows.Add(chunk.ToNDArray());
    }
}

NDArray stacked = np.vstack(rows.ToArray());   // shape (4, 3)
```

> Proven by `Generators_StreamChunksIntoNumSharp`.

---

## Manual `PyObject` work: dicts, strings, arrays together

Nothing stops you from mixing the raw Python.NET object model with conversions — a `PyDict` "record" carrying an array behaves like any Python dict, and its `data` entry is still the shared buffer:

```csharp
using (Py.GIL())
{
    using var record = new PyDict();
    using (PyObject arr = nd.ToNumpy())
        record["data"] = arr;
    using (var tag = new PyString("run-42"))
        record["tag"] = tag;

    scope.Set("record", record);
}

scope.Exec("record['data'][1] = 41.0");   // writes into nd's memory
```

> Proven by `ManualPyObjectWork_ListsAndDicts_MixedWithConversions`.

---

## Caching module references

The typical long-lived-app pattern: import a module once, keep the `PyObject`, reuse it across many calls (dispose it under the GIL when done):

```csharp
PyObject numpyModule;
using (Py.GIL())
    numpyModule = Py.Import("numpy");

// ... any time later, on any thread:
using (Py.GIL())
{
    dynamic numpy = numpyModule;
    using PyObject arr = nd.ToNumpy();
    double mean = (double)numpy.mean(arr);
}
```

> Proven by `ModuleCache_StoredDynamicReference_ReusedAcrossCalls`.

---

## In-place ufuncs writing into NumSharp memory

numpy's `out=` parameter works through `dynamic` named arguments — classic in-place pipelines mutate NumSharp's buffer directly:

```csharp
using (Py.GIL())
{
    dynamic numpy = Py.Import("numpy");
    dynamic u = scope.Get("u");        // the exported view
    numpy.add(u, 1.0, @out: u);        // u += 1, in place
    numpy.clip(u, 2.0, 5.0, @out: u);  // clip, in place
}
// nd now holds [2, 2, 3, 4, 5, 5] — no copies were made anywhere
```

> Proven by `InPlaceUfuncs_WriteThroughTheSharedView`.

---

## Python lists in and out

```csharp
// python list -> NDArray
NDArray fromList = scope.Eval("np.asarray([2.5, 3.5, 4.5])").ToNDArray();

// NDArray -> python list
scope.Exec("as_list = pl.tolist()");   // pl is an exported int64 array
// as_list == [0, 2, 4, 6]
```

> Proven by `PythonLists_ToAndFromArrays`.

---

## Worker threads calling Python

The interop self-acquires the GIL, so worker threads call Python functions with their own arrays using ordinary `Py.GIL()` blocks — no shared state, no deadlocks:

```csharp
var worker = new Thread(() =>
{
    for (int k = 0; k < 10; k++)
    {
        var batch = np.arange(6).astype(NPTypeCode.Double) + id;
        double weighed;
        using (Py.GIL())
        {
            dynamic weigh = scope.Get("weigh");
            weighed = (double)weigh(batch);      // NDArray auto-encoded on a worker thread
        }
    }
});
```

> Proven by `WorkerThreads_CallPythonWithTheirOwnArrays` (three concurrent workers, ten calls each).

---

## Running the proof yourself

```bash
dotnet test test/NumSharp.Interop.UnitTests/NumSharp.Interop.UnitTests.csproj
```

The suite auto-discovers Python (or honors `PYTHONNET_PYDLL`) and self-skips cleanly on machines without Python + numpy. Beyond this page's patterns it also proves the [lifetime model](pythonnet.md#lifetime--memory-safety) — orphaned exports, derived-view leases, transitive chains, premature/double disposal, GC hammers, cross-thread handoffs and async flows.
