# NumSharp.Interop.pythonnet — the examples

Twelve single-file C# scripts, one per feature area of
[`NumSharp.Interop.pythonnet`](../../src/NumSharp.Interop.pythonnet/README.md) — the zero-copy
bridge between NumSharp `NDArray` and the Python ecosystem (numpy, PEP 3118 buffers, PyTorch,
Pandas, custom adapters) through Python.NET. Together they touch **every public member** of the
package. Each script is a runnable tour *and* a gate: it prints one `OK`/`FAIL` line per claim and
exits non-zero if any claim does not hold, so the console output is evidence, not narration.

## Run

Requirements: the .NET 10 SDK, and a `python` on `PATH` that can `import numpy` (or point
`PYTHONNET_PYDLL` at a `python3xx.dll` / `libpython3.xx.so`). Install the Python side once:

```bash
python -m pip install -r requirements.txt     # numpy is required; torch/pandas/pyarrow/pillow are optional
```

Then, from this directory:

```bash
dotnet run 02-four-verbs.cs                   # one script (first run builds NumSharp; later runs are ~2 s)
pwsh ./verify.ps1                             # every script, in order; exit 1 if any fails
pwsh ./verify.ps1 -Examples 06-lifetime, 09-pytorch
```

Scripts that need an optional package (`09-pytorch.cs`, `10-pandas.cs`, and the Pillow/PyArrow/torch
sections of `05-buffers-and-libraries.cs`) self-skip with exit code 0 when it is not installed.

Every file is a .NET 10 [file-based app](https://learn.microsoft.com/dotnet/core/sdk/file-based-programs):
its first line references the in-repo project, and swapping that line for
`#:package NumSharp.Interop.pythonnet@<version>` makes it a standalone program you can copy anywhere.
Because a single-file app cannot share source with its siblings, the ~90-line scaffolding at the
bottom of each file (engine discovery + startup + crash-free shutdown, a numpy scope helper, and
the OK/FAIL checklist) is repeated verbatim; the feature code is everything above it.

## The tour

| Script | Feature area | What it proves |
|---|---|---|
| [`01-bootstrap.cs`](01-bootstrap.cs) | Engine lifecycle | discovering CPython, `Initialize` / `BeginAllowThreads`, the pythonnet-vs-Python version guard, `RegisterCodec()` ordering, the first shared buffer, converting from a thread that never touched Python, `Shutdown()` with `NoopFormatter` |
| [`02-four-verbs.cs`](02-four-verbs.cs) | The four verbs | `ToNumpy` (view) / `ToNumpyCopy` / `ToNDArray` (copy) / `AsNDArray` (view), the aliases `ToPython`, `ToNumpy(copy:true)`, `ToNDArrayView`; write-through both ways; read-only sources and `allowReadonly`; import views own nothing (`resize` refusal, `np.require("O")`); `FromArrayLike` |
| [`03-layouts.cs`](03-layouts.cs) | Layouts | export keeps strides/flags for C, row/column slices, transpose, reversed (negative strides), Fortran, broadcast (read-only), 0-d, empty; the ctypes-window / `as_strided` base chain; the three zero-copy import routes (contiguous exporters, strided numpy via `__array_interface__`, strided memoryviews via `PyBUF.STRIDED`); the only layouts that copy |
| [`04-dtypes.cs`](04-dtypes.cs) | Dtypes | the 14 mappable dtypes both ways; `ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`; bit-exact specials (NaN payloads, `-0.0`, `uint64` maxima); Decimal → float64, complex64 widening, UCS-4 narrowing, big-endian byte-swapping, refused dtypes |
| [`05-buffers-and-libraries.cs`](05-buffers-and-libraries.cs) | Buffers & libraries | numpy-agnostic import of `bytes`/`bytearray`/`memoryview`/`array.array` (12 typecodes)/ctypes/`BytesIO`/`mmap`/shared memory; `ToMemoryView` for `struct`, `hashlib`, `zlib`, sockets, `np.frombuffer`; Pillow, PyArrow, `torch.frombuffer` |
| [`06-lifetime.cs`](06-lifetime.cs) | Lifetime | `LiveExports` / `LiveImports`; an export outlives every managed reference (derived numpy views included); an import lease outlives every Python reference (derived slices included); resize locks on both sides; 120 rounds of churn; engine shutdown drains imports and sweeps orphaned exports |
| [`07-codec.cs`](07-codec.cs) | The codec | `RegisterCodec()` and the cached-miss ordering trap; implicit encode at `scope.Set`, call arguments and `dynamic` numpy; `As<NDArray>()` / `AsManagedObject`; what `CanDecode` accepts; `NumpyCodecMode` Auto / View / Copy through standalone `NumpyCodec` instances; `DecodeAnyBuffer`, `DecodeArrayLike`, `DecodeArrayAdapters` |
| [`08-gil.cs`](08-gil.cs) | The GIL | verbs acquire the GIL themselves; the `requireGIL:false` hot loop; the process-wide `RequireGIL` opt-out; contention proof; worker threads; the Python → .NET callback trap (`PyGILState_Check() == 0`); the background drain is immune to the policy |
| [`09-pytorch.cs`](09-pytorch.cs) | PyTorch | `ToTorch` (view / `copy:true`), strides, all 15 dtypes, unsafe layouts; `AsTorchNDArray` / `ToTorchNDArray(force)`; autograd round trip; requires_grad / conj / CUDA / bfloat16 / sparse / meta boundaries; `expand()` as a read-only broadcast; the codec with `torch.from_numpy(nd)`; resize refusal |
| [`10-pandas.cs`](10-pandas.cs) | Pandas | Series / DataFrame / Index / RangeIndex / nullable arrays through the adapter; read-only Copy-on-Write projections and `allowReadonly`; what materializes (mixed blocks, `pd.NA`, categorical, text); labels are metadata; `pd.Series(nd, copy=False)`; lifetime; `DecodeArrayAdapters` |
| [`11-custom-adapter.cs`](11-custom-adapter.cs) | Extensibility | an `IPythonArrayAdapter` for a buffer-less library object; `RegisterArrayAdapter` / `PythonArrayAdapterRegistry.Register` idempotence; declining with `null`; the adapter serving the direct verbs and the codec; a throwing adapter cannot break the pipeline |
| [`12-scenarios.cs`](12-scenarios.cs) | Applications | ML inference hand-off, an in-place image pipeline, a telemetry ring, a Python dataset outliving its references under NumSharp kernels, a mixed-ownership co-simulation, a codec service loop, generators, exceptions, JSON, faulty inputs |

## Reading the output

```text
-- ToNumpy (view) vs ToNumpyCopy (copy) --
  OK   view: Python's write reached NumSharp (nd[0] == 11)
  OK   copy: Python's write did not (nd[1] still 1)
  OK   np.shares_memory(v, c) is False
  ...
ALL OK
```

Each script ends with a counters section that asserts `LiveExports` and `LiveImports` returned to
baseline, so every script also doubles as a no-leak proof. Two things worth knowing when you write
your own: pythonnet **defers** the decref of wrappers the CLR garbage-collected (the arguments of a
`dynamic` call, for instance) until `Finalizer.Instance.Collect()` runs, which is why the scripts'
`Settle()` helper flushes it before reading the counters; and a `PyObject` must be created *and*
disposed under the GIL (`using (Py.GIL())`) — and never touched once the engine has shut down, which
is why the scaffolding's `PyScope.Dispose` checks `PythonEngine.IsInitialized` first.

## Where the formal gates live

The same claims are pinned by `test/NumSharp.Tests.Interop` (524 tests against live CPython;
`dotnet test test/NumSharp.Tests.Interop -f net10.0`) and documented on the website pages
[Python & numpy](../../docs/website-src/docs/interop/pythonnet-numpy.md),
[PyTorch](../../docs/website-src/docs/interop/pytorch.md),
[Pandas](../../docs/website-src/docs/interop/pandas.md) and
[np.frombuffer](../../docs/website-src/docs/interop/np-frombuffer.md). These examples are the
executable tour of that surface; the tests are its contract.
