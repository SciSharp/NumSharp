# NumSharp.Interop.pythonnet — the examples

Twelve single-file C# tutorials, one per feature area of
[`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md) — the zero-copy
bridge between NumSharp `NDArray` and the Python ecosystem (numpy, PEP 3118 buffers, PyTorch,
Pandas, custom adapters) through Python.NET. Together they touch **every public member** of the
package.

They are written to be **read**, top to bottom: each line does one thing and the comment beside it
states what that line leaves behind (`// nd is now [11. 1. 2. 3.]`, `// (48, 8)`,
`// InvalidOperationException: ...`). They print nothing and assert nothing. The proof lives next
door: every script has a twin under
[`test/NumSharp.Tests.Interop/Examples`](../../test/NumSharp.Tests.Interop/Examples) — the same
code, section by section, with every one of those comments turned into an assertion and run against
live CPython on every CI push.

## Run

Requirements: the .NET 10 SDK, and a `python` on `PATH` that can `import numpy` (or point
`PYTHONNET_PYDLL` at a `python3xx.dll` / `libpython3.xx.so`). Install the Python side once:

```bash
python -m pip install -r requirements.txt     # numpy is required; torch/pandas/pyarrow/pillow are optional
```

Then, from this directory:

```bash
dotnet run 02-four-verbs.cs                   # one script (the first run builds NumSharp; later runs take ~2 s)
pwsh ./verify.ps1                             # every script, in order; exit 1 if any fails to run to completion
pwsh ./verify.ps1 -Examples 06-lifetime, 09-pytorch
```

A script that needs an optional package (`09-pytorch.cs`, `10-pandas.cs`, and the Pillow / PyArrow
/ torch sections of `05-buffers-and-libraries.cs`) skips that part quietly when the package is not
installed. Every file is a .NET 10
[file-based app](https://learn.microsoft.com/dotnet/core/sdk/file-based-programs): its first line
references the in-repo project, and swapping that line for
`#:package NumSharp.Interop.pythonnet@<version>` makes it a standalone program you can copy
anywhere. Because a single-file app cannot share source with its siblings, the ~80-line scaffolding
at the bottom of each file (`PythonHost`: CPython discovery, engine startup, codec registration, a
namespace with `import numpy as np`, crash-free shutdown; `Throws`: the exception a line is expected
to raise) is repeated verbatim; the tutorial is everything above it.

## How the scripts are written

Three things shape every script, and they are the three things worth taking away:

```csharp
using var host = PythonHost.Start();          // CPython + numpy, the engine, the codec
using var scope = NDScope.Open();             // every NDArray built below is released when the scope closes
dynamic py = host.Namespace;                  // a Python module: talk to it under Py.GIL()

NDArray nd = np.arange(4).astype(np.float64);
using (Py.GIL())
{
    py.v = nd;                                // NDArray -> numpy: a zero-copy view (the codec ran ToNumpy)
    py.Exec("v[0] = 11.0");                   // Python writes; nd is now [11. 1. 2. 3.]
    double v2 = py.v[2];                      // read back with the natural C# type
    bool shared = py.np.shares_memory(py.v, py.v);
    NDArray back = py.np.sin(py.v / 3.0);     // numpy -> NDArray: a view of the fresh result (AsNDArray, implicitly)
    (long, long) shape = py.np.zeros((2, 3)).shape;   // C# tuples cross as Python tuples and back
}
```

- **`dynamic` for the Python side.** The namespace is a `PyModule` held as `dynamic`: attributes
  bind by name, `NDArray` arguments and results convert through the registered codec, kwargs are C#
  named arguments (`numpy.arange(6, dtype: "f8")`), and a Python `tuple` reads into a C# one. The
  explicit verbs (`nd.ToNumpy()`, `pyObj.AsNDArray()`, ...) appear where a verb is the subject.
  Two spellings stay Python strings on purpose: an element write (`py.Exec("v[0] = 11.0")` — a
  `dynamic` indexer setter needs a `PyObject`) and a multi-index read (`py.Eval("src[1, 2]")`, or
  `a.item(1, 2)`); a numpy integer scalar converts through `int(...)` / `.item()` (it is not a
  Python `int`).
- **One `NDScope` per script**, opened right after the engine. Every `NDArray` constructed on the
  thread while it is open — slices, transposes, import views — is released when it closes, so no
  script disposes an array by hand; a loop opens a nested scope per iteration, and a helper that
  hands an array out yields it with `scope.Returns(...)`. `06-lifetime.cs` is where this meets the
  interop's own pins and leases.
- **`Py.GIL()` around the Python parts.** The conversion verbs take the GIL themselves; your own
  `PyObject` / `dynamic` work runs inside a `using (Py.GIL())` block, and NumSharp work needs none.

One trap the scripts point out as they go: a `dynamic` read of a numpy array (`py.win`) hands out a
CLR wrapper that keeps its own Python reference until the garbage collector reaches it — harmless
in general, but where a reference count is the subject (`06-lifetime.cs`) the scripts read the
object as `using PyObject` instead, and a tight loop reads a statistic through `py.Eval("win.std()")`
rather than through the array. (Encodes are covered: the codec releases the wrapper it hands
pythonnet at the next conversion, so `py.win = window` in a loop keeps exactly one export live.)

## The tour

| Script | Feature area | What it walks through |
|---|---|---|
| [`01-bootstrap.cs`](01-bootstrap.cs) | Engine lifecycle | discovering CPython, `Initialize` / `BeginAllowThreads`, the pythonnet-vs-Python version guard, `RegisterCodec()` ordering, the first shared buffer, converting from a thread that never touched Python, `Shutdown()` with `NoopFormatter` |
| [`02-four-verbs.cs`](02-four-verbs.cs) | The four verbs | `ToNumpy` (view) / `ToNumpyCopy` / `ToNDArray` (copy) / `AsNDArray` (view), the aliases `ToPython`, `ToNumpy(copy:true)`, `ToNDArrayView`; the codec's implicit spellings `py.x = nd` / `NDArray a = py.x`; write-through both ways; read-only sources and `allowReadonly`; import views own nothing (`resize` refusal, `np.require("O")`); `FromArrayLike` |
| [`03-layouts.cs`](03-layouts.cs) | Layouts | export keeps strides/flags for C, row/column slices, transpose, reversed (negative strides), Fortran, broadcast (read-only), 0-d, empty — as one table; the ctypes-window / `as_strided` base chain; the three zero-copy import routes (contiguous exporters, strided numpy via `__array_interface__`, strided memoryviews via `PyBUF.STRIDED`); the only layouts that copy |
| [`04-dtypes.cs`](04-dtypes.cs) | Dtypes | the 14 mappable dtypes both ways; `ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`; bit-exact specials (NaN payloads, `-0.0`, `uint64` maxima); Decimal → float64, complex64 widening, UCS-4 narrowing, big-endian byte-swapping, refused dtypes |
| [`05-buffers-and-libraries.cs`](05-buffers-and-libraries.cs) | Buffers & libraries | numpy-agnostic import of `bytes`/`bytearray`/`memoryview`/`array.array` (12 typecodes)/ctypes/`BytesIO`/`mmap`/shared memory; `ToMemoryView` for `struct`, `hashlib`, `zlib`, sockets, `np.frombuffer`; Pillow, PyArrow, `torch.frombuffer` |
| [`06-lifetime.cs`](06-lifetime.cs) | Lifetime | `LiveExports` / `LiveImports`; an export outlives every managed reference (derived numpy views included); an import lease outlives every Python reference (derived slices included, yielded out of a nested `NDScope`); resize locks on both sides; 120 rounds of churn under per-round scopes; engine shutdown draining imports and sweeping orphaned exports |
| [`07-codec.cs`](07-codec.cs) | The codec | `RegisterCodec()` and the cached-miss ordering trap; implicit encode at `py.x = nd`, call arguments and `dynamic` numpy; C# tuples as shapes and multi-indices; `As<NDArray>()` / `AsManagedObject` / `NDArray a = py.x`; what `CanDecode` accepts; `NumpyCodecMode` Auto / View / Copy via standalone `NumpyCodec` instances; `DecodeAnyBuffer`, `DecodeArrayLike`, `DecodeArrayAdapters`, `ConvertTuples` |
| [`08-gil.cs`](08-gil.cs) | The GIL | verbs acquire the GIL themselves; the `requireGIL:false` hot loop; the process-wide `RequireGIL` opt-out; a contention proof; worker threads with their own scopes; the Python → .NET callback trap (`PyGILState_Check() == 0`); the background drain being immune to the policy |
| [`09-pytorch.cs`](09-pytorch.cs) | PyTorch | `ToTorch` (view / `copy:true`), `torch.Size` and strides as C# tuples, all 15 dtypes, unsafe layouts; `AsTorchNDArray` / `ToTorchNDArray(force)`; autograd round trip; requires_grad / conj / CUDA / bfloat16 / sparse / meta boundaries; `expand()` as a read-only broadcast; `torch.from_numpy(nd)` through the codec; resize refusal |
| [`10-pandas.cs`](10-pandas.cs) | Pandas | Series / DataFrame / Index / RangeIndex / nullable arrays through the adapter; read-only Copy-on-Write projections and `allowReadonly`; what materializes (mixed blocks, `pd.NA`, categorical, text); labels as metadata; `pd.Series(nd, copy=False)`; lifetime; `DecodeArrayAdapters` |
| [`11-custom-adapter.cs`](11-custom-adapter.cs) | Extensibility | an `IPythonArrayAdapter` for a buffer-less library object; `RegisterArrayAdapter` / `PythonArrayAdapterRegistry.Register` idempotence; declining with `null`; the adapter serving the direct verbs and the codec; a throwing adapter cannot break the pipeline |
| [`12-scenarios.cs`](12-scenarios.cs) | Applications | ML inference hand-off, an in-place image pipeline, a telemetry ring, a Python dataset outliving its references under NumSharp kernels, a mixed-ownership co-simulation, a codec service loop, generators, exceptions, JSON, faulty inputs |

## Where the proof lives

`test/NumSharp.Tests.Interop/Examples/Example01_Bootstrap.cs` … `Example12_Scenarios.cs` are the
twins: one test class per script, one test method per section, the same code with the comments
asserted (`dotnet test test/NumSharp.Tests.Interop -f net10.0 --filter "FullyQualifiedName~Examples"`).
They inherit the suite's per-test leak gate, so each section is also a no-leak proof. The rest of
`test/NumSharp.Tests.Interop` (630+ tests against live CPython) is the package's contract, and the
website pages [Python & numpy](../../docs/website-src/docs/interop/pythonnet-numpy.md),
[PyTorch](../../docs/website-src/docs/interop/pytorch.md),
[Pandas](../../docs/website-src/docs/interop/pandas.md) and
[np.frombuffer](../../docs/website-src/docs/interop/np-frombuffer.md) are its reference.
