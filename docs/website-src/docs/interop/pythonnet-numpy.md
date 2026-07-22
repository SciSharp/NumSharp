# Python & numpy — the same buffer on both sides

`NumSharp.Interop.pythonnet` embeds CPython in your .NET process through pythonnet and hands arrays
across the boundary without copying them: a NumSharp `NDArray` becomes a numpy view over the very
same unmanaged bytes, and any Python buffer becomes a NumSharp view over Python's memory. Four verbs
cover the surface — two per direction, one sharing and one copying — a registrable codec makes the
conversions implicit at every pythonnet boundary, and two counters make every live crossing
observable. This page is the package's reference: setup, the verbs, what they cost, which layouts
survive the crossing, who frees what, the codec, the GIL, dtypes and versions.

**On this page:** [From zero to a shared array](#from-zero-to-a-shared-array) ·
[The four verbs](#the-four-verbs) · [What it costs](#what-it-costs) ·
[Exporting: layouts](#exporting-every-layout-strides-intact) ·
[Importing: three routes](#importing-three-zero-copy-routes) ·
[Lifetime](#lifetime-two-collectors-one-buffer) · [The codec](#the-codec-conversions-without-calls) ·
[The GIL](#the-gil) · [Dtypes](#dtypes) · [Versions](#versions) ·
[Troubleshooting](#troubleshooting) · [Claims](#claims-ledger)

> Verified on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · net8.0/net10.0.
> Every claim below is reproduced by a test in `NumSharp.Interop.UnitTests`.

---

## From zero to a shared array

```bash
dotnet add package NumSharp.Interop.pythonnet
```

That resolves pythonnet 3.0.5 — the floor of the package's `[3.0.5, 4.0.0)` range, and what NuGet
actually installs, since it picks the lowest applicable version — which drives Python 3.7 through
3.13. Point pythonnet at your CPython and start the engine once per process:

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";   // or set the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();   // release the GIL from this thread; conversions re-acquire it
```

`BeginAllowThreads` is what lets any thread convert afterwards — including threads that have never
touched Python. From here, one buffer serves both sides:

```csharp
var nd = np.arange(6).reshape(2, 3);

using (Py.GIL())
{
    using var scope = Py.CreateScope();
    scope.Exec("import numpy as np");

    using (PyObject x = nd.ToNumpy())          // zero-copy view of NumSharp's buffer
        scope.Set("x", x);

    scope.Exec("x[1, 2] = 99");                // Python writes...

    scope.Exec("r = np.sin(x / 3.0)");         // numpy computes a fresh array...
    using PyObject r = scope.Get("r");
    using NDArray copy = r.ToNDArray();        // ...independent copy of it
    using NDArray view = r.AsNDArray();        // ...or a zero-copy view of it
}

Console.WriteLine(nd);                         // ...and NumSharp saw the write
```

```text
x  (dtype=int64):
[[0 1 2]
 [3 4 5]]

nd after x[1, 2] = 99:
[[ 0  1  2]
 [ 3  4 99]]

np.round(r, 4)  (r.dtype=float64):
[[0.     0.3272 0.6184]
 [0.8415 0.9719 0.9999]]

r.ToNDArray()  ->  Double, shape (2, 3), detached
r.AsNDArray()  ->  Double, writeable view over numpy's buffer
```

<sub>See here [`QuickStart_OneBufferBothSides`][gate], [`Bootstrap_AFreshThreadConverts`][gate]</sub>

---

## The four verbs

Everything else in the package is packaging over these four. The naming follows numpy's own
`array` / `asarray` split: **`To…` copies, `As…` shares** — with `ToNumpy` as the deliberate
exception, because the zero-copy view is the package's headline.

| Direction | Verb | Returns | Memory |
|---|---|---|---|
| NumSharp → Python | `nd.ToNumpy()` | numpy view | **shared**, source rooted |
| NumSharp → Python | `nd.ToNumpyCopy()` | numpy array | independent |
| Python → NumSharp | `py.ToNDArray()` | `NDArray` | independent, C-contiguous |
| Python → NumSharp | `py.AsNDArray()` | `NDArray` view | **shared**, exporter leased |

`ToNDArrayView(py, allowReadonly)` is the static spelling of `AsNDArray`; `nd.ToPython()` aliases
`ToNumpy`; `nd.ToNumpy(copy: true)` routes to `ToNumpyCopy`. `nd.ToMemoryView()` — raw writable
bytes for consumers that never touch numpy — has [its own page](np-frombuffer.md).

**A view shares later writes in both directions; a copy never does.**

```csharp
var nd = np.arange(4).astype(NPTypeCode.Double);

using (Py.GIL())
{
    using (PyObject v = nd.ToNumpy())     Scope.Set("v", v);
    using (PyObject c = nd.ToNumpyCopy()) Scope.Set("c", c);
    Scope.Exec("v[0] = 11.0");            // reaches nd
    Scope.Exec("c[1] = 22.0");            // does not
}
```

```text
nd                       [11.  1.  2.  3.]
np.shares_memory(v, c)   False
```

The import pair splits identically: `ToNDArray` copies any PEP 3118 exporter into a fresh
C-contiguous array (non-contiguous sources are linearized by CPython's `memoryview.tobytes('C')`),
while `AsNDArray` leases the exporter's own memory.

<sub>See here [`Verbs_ViewSharesAndCopyDetaches_BothDirections`][gate], [`Verbs_CopyTrueRoutesToTheCopy`][gate], [`Verbs_ToCopiesAndAsShares`][gate]</sub>

---

## What it costs

### Does a view get more expensive as the array grows?

**No. View verbs are flat in *n*; copy verbs are linear.** A view is a pointer, a shape and a
lifetime hook — the elements are never read. Measured at one million `float64`, best of 5, on the
stack in the banner:

| Verb | Time | Scales with n |
|---|---:|---|
| `ToMemoryView` | 0.0069 ms | no |
| `ToNumpy` | 0.0088 ms | no |
| `ToNDArrayView` | 0.0273 ms | no |
| `ToNDArray` (copy) | 0.6755 ms | yes |
| `ToNumpyCopy` | 1.1266 ms | yes |

At ten million elements the gap is three orders of magnitude; at a thousand it inverts — sharing
has a fixed setup cost (a ctypes window, a `frombuffer`, a `weakref.finalize` registration) that
copying eight kilobytes undercuts. The crossover sits around ten thousand elements. Below it, if
you have no reason to share, copy: it is simpler, it locks nothing, and it outlives the
interpreter.

A view also carries obligations a copy does not — the lifetime coupling and the resize locks of
[Lifetime](#lifetime-two-collectors-one-buffer). Choose a view for size or for shared mutation;
choose a copy for a detached snapshot.

<sub>See here [`Costs_ViewVerbsAreFlat_CopyVerbsScale`][gate]</sub>

---

## Exporting: every layout, strides intact

### What does numpy see when I export a slice or a transpose?

**The exact same strided window — never a copy, never a flattening.** `ToNumpy` expresses the
NumSharp layout in numpy's own terms: dimension sizes, byte strides, and the writeable flag.

| NumSharp source | numpy sees |
|---|---|
| C-contiguous `(4,6)` float64 | C-contiguous, strides `(48, 8)` |
| row slice `b["1:3"]` | strides `(48, 8)` over the shared buffer |
| column slice `b[":, ::2"]` | strides `(48, 16)` |
| transpose `b.T` | strides `(8, 48)`, `F_CONTIGUOUS=True` |
| reversed `b["::-1"]` | strides `(-48, 8)` — negative strides survive |
| F-order (`np.asfortranarray`) | strides `(8, 32)`, `F_CONTIGUOUS=True` |
| broadcast `(3,) → (2,3)` | strides `(0, 8)`, **`WRITEABLE=False`** |
| 0-d scalar | `shape=()`; `x[()] = 7.5` writes through |
| empty `(0,3)` | `shape=(0, 3)` — a fresh array; there are no bytes to share |

The table is exhaustive over the layout classes NumSharp produces. Broadcast views are read-only on
the numpy side because they are read-only in NumSharp — one element pretending to be many corrupts
under a blind write, and both libraries guard it the same way.

```text
sl = full[1:3, ::2]-equivalent exported directly:
  sl.tolist()              [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]
  np.shares_memory(full, sl)   True
```

<sub>See here [`Export_TableRows_KeepStridesAndFlags`][gate]</sub>

### What is the exported array made of?

**A numpy view chained onto a ctypes window over NumSharp's pointer.** `ToNumpy` builds
`(ctypes.c_char * nbytes).from_address(ptr)`, runs `np.frombuffer` over it, and reshapes — or, for
strided sources, `np.lib.stride_tricks.as_strided` with the exact byte strides:

```text
full:  strides=(48, 8)   base=ndarray -> c_char_Array_192    OWNDATA=False
slice: strides=(48, 16)  base=DummyArray (as_strided)        OWNDATA=False
```

`OWNDATA=False` is numpy agreeing the memory is NumSharp's. The deepest base object — the ctypes
window — is where the export's `weakref.finalize` is rooted, which is why every derived numpy view
(`arr[1:]`, `arr.T`, `np.asarray(arr)`) extends the buffer's life: they all chain back to it.

<sub>See here [`Export_BaseChain_EndsAtTheCtypesWindow`][gate]</sub>

---

## Importing: three zero-copy routes

### Which Python objects can NumSharp view without copying?

**Almost all of them — 47 of the 50 exporter varieties in the census below.** `AsNDArray` /
`ToNDArrayView` takes one of three routes, chosen by what the exporter is:

1. **C-contiguous PEP 3118 exporters** — any object: numpy arrays, `bytes`, `bytearray`,
   `array.array`, `memoryview`, ctypes arrays, `BytesIO.getbuffer()`, mmaps, shared memory. The
   buffer is acquired with `PyBUF.WRITABLE`, which pins the exporter and blocks reallocation.
2. **Non-contiguous numpy arrays** — slices, transposes, Fortran order, broadcasts: imported
   through `__array_interface__` as a strided NumSharp view with the identical layout; broadcast
   sources arrive read-only.
3. **Non-contiguous non-numpy exporters** — a sliced, offset or reversed `memoryview`, a strided
   window over an `array.array`: the pointer comes from a `PyBUF.STRIDED` request and the exact
   shape/strides from the memoryview itself. Negative strides included.

```csharp
using (Py.GIL())
{
    using PyObject ba = Scope.Eval("bytearray(b'abcd')");
    using NDArray v = ba.AsNDArray();      // route 1: Byte[4], writable, shares the bytearray
    v[0] = (NDArray)(byte)90;              // ...lands in Python
}
```

The census, re-measured on every test run — 50 exporter varieties across builtins, `array.array`'s
twelve typecodes, ctypes element types, numpy dtypes, numpy layouts and memoryview forms:
**47 view, 2 copy, 1 rejected.** The two copies are complex64 (widened to `Complex`) and a
sub-element-stride `as_strided` window (linearized); the one rejection is big-endian multi-byte
data, refused by both paths rather than silently byte-swapped.

<sub>See here [`Import_ThreeRoutes_EachYieldsAView`][gate], [`Import_Census_47Of50VarietiesView`][gate]</sub>

### Can I view something read-only, like `bytes`?

**Only by saying so — and the view arrives non-writeable.** By default a read-only source is
refused with guidance, so a view you can write through is the only thing a default call can return:

```text
InvalidOperationException: the exporter's buffer is read-only; writing through a NumSharp view
would corrupt an immutable Python object. Use ToNDArray (copy), or pass allowReadonly:true to
take a NON-WRITEABLE view (guarded writes through it throw).
```

With `allowReadonly: true` the view's `Shape.IsWriteable` is `false` — numpy's `writeable=False`,
carried across — and a write raises `assignment destination is read-only` instead of corrupting
the source.

<sub>See here [`Import_ReadonlyRefusedByDefault_OptInIsNonWriteable`][gate]</sub>

### Does an imported view own its memory?

**No — and it tells you so, exactly like `np.frombuffer` arrays do.** An import view has numpy's
`owndata == False` semantics: a size-changing `resize` refuses rather than silently reallocating
away from the shared Python memory, and `np.require(view, requirements: "O")` is the escape hatch
that produces an owning, detached copy.

```text
view.resize(new Shape(8))  ->  IncorrectShapeException:
    cannot resize this array: it does not own its data
np.require(view, null, "O") -> owning copy; owned[0] = -4.0 leaves Python's own[0] = 0.0
```

<sub>See here [`ImportView_OwnsNothing_ResizeRefusesAndRequireOCopies`][gate]</sub>

---

## Lifetime: two collectors, one buffer

The rule in both directions is the same: **memory is released when the last reference on the far
side lets go**, never when the near side happens to tidy up first.

### Can Python outlive my `NDArray`?

**Yes.** An export takes its own atomic reference on the NumSharp buffer and hands the release to a
`weakref.finalize` on the exported array's base object. Dispose the `NDArray`, drop the `PyObject`
wrapper, run the GC — Python still reads and writes valid memory, and the buffer is freed when the
last Python-side view dies:

```text
after nd.Dispose() + GC:   a = [0. 1. 2. 3.]     LiveExports = 1
a[0] = 7.0                 a = [7. 1. 2. 3.]     (still writable)
del a, mv; gc.collect()                          LiveExports = 0
```

<sub>See here [`Lifetime_ExportOutlivesItsSource`][gate]</sub>

### Can my `NDArray` outlive Python's object?

**Yes — the lease holds the exporter alive.** An imported view keeps a `Py_buffer` lock (or, for
strided numpy sources, a strong reference) on the exporter; Python can `del` every name it has. The
lease is released when the last NumSharp view over the memory — *including derived slices* — is
disposed or collected. The refcount decides, not disposal order.

Both directions are observable, which is what makes leak tests possible:

```csharp
int pinned = NDArrayPythonInterop.LiveExports;   // NumSharp buffers held by Python
int leased = NDArrayPythonInterop.LiveImports;   // Python buffers held by NumSharp
```

<sub>See here [`Lifetime_LeaseIsHeldByTheLastView`][gate], [`Lifetime_CountersTrackBothDirections`][gate]</sub>

### What does a live view forbid?

**Reallocation, on both sides.** While NumSharp leases a `bytearray`, `ba.append(1)` raises
`BufferError: Existing exports of data: object cannot be re-sized`; numpy's
`resize(refcheck=True)` refuses the same way. Mirror image: while Python holds a view of a NumSharp
array, `nd.resize(...)` raises NumPy's own wording — `cannot resize an array that references or is
referenced by another array in this way`. Dispose the view and both locks lift.

<sub>See here [`Lifetime_ALiveConversionLocksResizing_BothSides`][gate]</sub>

### What happens at engine shutdown?

**Imports are drained crash-free; orphaned exports are swept right after the engine dies.** Import
views are tied to the interpreter that owns their memory: after `PythonEngine.Shutdown()` they must
not be touched, though disposing them stays safe — the shutdown handler releases every outstanding
lease first. Exports still referenced by Python cannot release through `weakref.finalize` (pythonnet
runs no Python atexit pass), so the package snapshots them and drops their pins as soon as the
engine has provably finished dying.

One pythonnet caveat on the way out: on .NET 8+ its shutdown crashes in its own state stashing
(`BinaryFormatter` was removed from the runtime). Opt out before calling it:

```csharp
RuntimeData.FormatterType = typeof(NoopFormatter);   // pythonnet's own opt-out
PythonEngine.Shutdown();
```

<sub>See here [`OrphanExport_HeldOnlyByPython_AwaitsTheShutdownSweep`][gate-shutdown], [`OrphanImport_StillReferencedByCSharp_AwaitsTheShutdownDrain`][gate-shutdown]</sub>

---

## The codec: conversions without calls

### Can the conversions happen automatically?

**Yes — register once, then every pythonnet boundary converts by itself.** `RegisterCodec()` hooks
NumSharp into pythonnet's conversion pipeline: `scope.Set("x", nd)` encodes an `NDArray` as a
zero-copy numpy view, passing one to a Python callable does the same, and `pyObj.As<NDArray>()`
decodes any buffer exporter on the way back:

```csharp
NDArrayPythonInterop.RegisterCodec();              // once per engine session; idempotent

var nd = np.arange(4).astype(NPTypeCode.Double);
using (Py.GIL())
{
    Scope.Set("c", nd);                            // auto-encoded: a shared numpy view
    Scope.Exec("c[1] = 88.5");                     // ...reaches nd
    using PyObject r = Scope.Eval("np.sqrt(c)");
    using NDArray back = r.As<NDArray>();          // auto-decoded: a shared view of r
}
```

```text
type(c).__name__          ndarray       c.flags['OWNDATA']   False
after c[1] = 88.5     ->  nd = [ 0.  88.5  2.   3. ]
As<NDArray>(np.sqrt(c))   Double, writeable view:
                          [0.         9.40744386 1.41421356 1.73205081]
```

<sub>See here [`Codec_RegisterOnce_ThenSetAndAsJustWork`][gate]</sub>

### When must I register it?

**Before the first `As<NDArray>()` anywhere in the process — this is the one ordering trap.**
pythonnet caches the decoder lookup per (Python type, target type) pair, and it caches misses. A
decode attempted before registration fails — and keeps failing for that Python type for the rest of
the engine session, even after you register:

```text
1. BEFORE RegisterCodec: As<NDArray>(ndarray)   -> InvalidCastException:
       'numpy.ndarray' value cannot be converted to NumSharp.NDArray
2. RegisterCodec()                              -> True
3. AFTER RegisterCodec:  As<NDArray>(ndarray)   -> still the same InvalidCastException
4. AFTER RegisterCodec:  As<NDArray>(bytes)     -> OK Byte    (pair never tried before)
```

Register at startup, right after `PythonEngine.Initialize()`. Registration is per engine session —
pythonnet clears all codecs during `Shutdown`, and `RegisterCodec()` knows to re-register after a
later `Initialize()`. The explicit verbs (`ToNumpy`, `AsNDArray`, …) never involve the codec and
work regardless.

<sub>See here [`Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned`][gate]</sub>

### Which mode should I pick?

**`Auto` unless you have a reason: it shares when a view is possible and copies only when one is
impossible.** `NumpyCodecOptions` sets one of three modes per direction:

| Mode | On decode | On encode |
|---|---|---|
| `Auto` (default) | view when representable, else copy (complex64 widens, UCS-4 narrows, sub-element strides linearize) | always a view |
| `View` | view or **decline** — never a silent copy | view |
| `Copy` | always an independent snapshot; never locks the source | always a copy |

Read-only sources decode as non-writeable views under `Auto`/`View`. `Decimal` has no numpy dtype
in any mode: encoding falls back to pythonnet's CLR-object wrapping instead of failing, so the
object still crosses — as a wrapped `NDArray`, not a numpy array:

```text
scope.Set("d", decimalArray)  ->  type(d).__name__ == 'NDArray'
View-mode TryDecode: complex64 -> False (declined), float64 -> True (view)
```

<sub>See here [`Codec_Modes_AutoViewCopy_AndTheDecimalFallback`][gate]</sub>

---

## The GIL

Every conversion verb **acquires the GIL itself**, re-entrantly — nesting inside your own
`Py.GIL()` block is fine, and calling from a thread that never touched Python is fine. What you
still own is your `PyObject`s: create and dispose them inside a `Py.GIL()` scope, because
pythonnet's final decref needs it.

For a hot loop, lift the per-call acquisition:

```csharp
using (Py.GIL())                                                   // ONE acquisition...
    foreach (var batch in batches)
        using (PyObject p = batch.ToNumpy(requireGIL: false))      // ...N conversions inside
            consumer.Invoke(p);
```

`requireGIL: false` means **you** hold the GIL — converting without actually holding it is an
immediate access violation, like any raw C-API misuse. The process-wide default is
`NDArrayPythonInterop.RequireGIL` (`true`); the per-call parameter overrides it, and `null` follows
it.

> **The trap.** A .NET method or delegate invoked *from* Python does **not** hold the GIL —
> pythonnet's binder releases it around managed bodies. Inside a Python → .NET callback, leave GIL
> management on.

The package's background machinery (deferred lease disposal, the shutdown drain) always manages
the GIL itself, so the opt-out never applies to it.

<sub>See here [`Gil_OneAcquisitionManyConversions`][gate], [`PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere`][gate-gil]</sub>

---

## Dtypes

Fourteen of NumSharp's fifteen dtypes cross, and the maps are public
(`ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`):

| NumSharp | numpy | PEP 3118 | numpy calls it |
|---|---|---|---|
| `Boolean` | `\|b1` | `?` | `bool` |
| `Byte` / `SByte` | `\|u1` / `\|i1` | `B` / `b` | `uint8` / `int8` |
| `Int16` / `UInt16` | `<i2` / `<u2` | `h` / `H` | `int16` / `uint16` |
| `Int32` / `UInt32` | `<i4` / `<u4` | `i` / `I` | `int32` / `uint32` |
| `Int64` / `UInt64` | `<i8` / `<u8` | `q` / `Q` | `int64` / `uint64` |
| `Half` / `Single` / `Double` | `<f2` / `<f4` / `<f8` | `e` / `f` / `d` | `float16` / `float32` / `float64` |
| `Complex` | `<c16` | `Zd` | `complex128` |
| `Char` | `<u2` | `H` | `uint16` — a C# `char` is a UTF-16 code unit; numpy has no char dtype |
| `Decimal` | *throws* | *throws* | — 16 bytes, non-IEEE; `astype(NPTypeCode.Double)` first |

Four element types deserve their fine print, all four handled by conversion rather than
misrepresentation:

- **complex64** (`c8` / `Zf`) — two 4-byte floats where `Complex` is two 8-byte doubles. No view is
  possible; `ToNDArray` widens each pair during the copy.
- **UCS-4 text** (`<U1`, 4-byte `wchar_t`) — a 4-byte code point where `Char` is a 2-byte UTF-16
  unit. `ToNDArray` narrows on copy; non-BMP code points throw (they need a surrogate pair). A
  **2-byte** `wchar_t` buffer (`array.array('u')` on Windows) *is* UTF-16 and views zero-copy as
  `Char`.
- **big-endian** (`>i4`, `!H`, …) — refused by both paths for multi-byte types; a native read would
  byte-swap every value. Single-byte dtypes (`>i1`, `\|u1`, `\|b1`) still view — byte order is
  meaningless at one byte. The fix travels in the message:
  `arr.astype(arr.dtype.newbyteorder('<'))`.
- **long double** (`g`) — MSVC's 8-byte long double is IEEE double and views as `Double`; the
  extended-precision widths have no NumSharp dtype and throw with `astype(np.float64)` guidance.

<sub>See here [`Dtypes_EveryRowRoundTrips`][gate]</sub>

---

## Versions

### Which pythonnet do I need for my Python?

**The package floor (3.0.5) covers Python 3.7–3.13; Python 3.14 needs pythonnet 3.1.0.** Each
pythonnet release hard-caps the newest Python it can drive. The mapping below is read out of each
release's own `PythonEngine.MaxSupportedVersion` — and the package checks it once per session,
turning pythonnet's opaque symbol-load failures into an actionable error naming the version to
install:

| Your Python | Minimum pythonnet |
|---|---|
| 3.7 – 3.10 | 3.0.0 |
| 3.11 | 3.0.1 |
| 3.12 | 3.0.3 |
| 3.13 | 3.0.5 |
| 3.14 | 3.1.0 |

The floor is 3.0.5 because its range (3.7–3.13) is a strict superset of every earlier 3.0.x — the
older floors silently shipped defaults that could not run current Pythons. The `4.0.0` upper bound
keeps a future breaking pythonnet from resolving into your build unasked; the source compiles clean
against all of 3.0.0–3.1.0, so pinning 3.1.0 yourself is supported:

```xml
<PackageReference Include="pythonnet" Version="3.1.0" />  <!-- only for Python 3.14 -->
```

One version-specific repair worth knowing exists but not relying on: pythonnet 3.0.1's `PyBuffer`
is broken for shape/strides/format flags, so the package reads all buffer *metadata* through
Python's own `memoryview` and uses `PyBuffer` only with the crash-free flags — which is why every
import route works uniformly across pythonnet 3.0.x.

<sub>See here [`Versions_TableIsTheGuardsOwnMapping`][gate]</sub>

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Python engine is not initialized` | Set `Runtime.PythonDLL` (or `PYTHONNET_PYDLL`) and call `PythonEngine.Initialize()` before any conversion |
| `Python 3.x is not supported by the loaded pythonnet` | Your pythonnet caps out below your Python. Install the version the message names (table above) |
| `'numpy.ndarray' value cannot be converted to NumSharp.NDArray` — even after `RegisterCodec()` | A decode ran before registration and pythonnet cached the miss for that Python type. Register at startup, before the first `As<NDArray>()` |
| `the exporter's buffer is read-only` | You asked for a writable view of `bytes` or a `writeable=False` array. Pass `allowReadonly: true`, or `ToNDArray()` to copy |
| `assignment destination is read-only` | You wrote through a non-writeable view — a read-only or broadcast source. Copy first if you need to mutate |
| `Existing exports of data: object cannot be re-sized` | A live NumSharp view leases that object. `Dispose()` the view to release the lock |
| `cannot resize an array that references or is referenced by another array in this way` | A live Python export (or another NumSharp view) pins the buffer. Release it, or copy first |
| `cannot resize this array: it does not own its data` | Import views never own their memory. `np.require(view, null, "O")` for an owning copy |
| `big-endian dtype '>f8' cannot be shared with a native-endian NumSharp buffer` | Byte-swap on the Python side: `arr.astype(arr.dtype.newbyteorder('<'))` |
| `decimal has no numpy dtype (16-byte, non-IEEE)` | Nothing in numpy describes it. `nd.astype(NPTypeCode.Double)` first |
| `the object does not export a PEP 3118 buffer` | Not every object is an exporter — a `dict`, a PIL `Image`. `np.asarray(obj)` first if numpy understands it |
| A Python write did not show up in NumSharp | You copied. `ToNumpy`/`AsNDArray` share; `ToNumpyCopy`/`ToNDArray` do not |
| Access violation around a conversion | `requireGIL: false` on a thread that does not hold the GIL — including inside a Python → .NET callback, where pythonnet releases it |
| `PythonEngine.Shutdown()` crashes on .NET 8+ | pythonnet's state stashing uses the removed `BinaryFormatter`. Set `RuntimeData.FormatterType = typeof(NoopFormatter)` first |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | After `BeginAllowThreads`, a fresh thread converts | conversion succeeds on a thread that never touched Python | [`Bootstrap_AFreshThreadConverts`][gate] |
| 2 | Export is a shared view; Python writes reach NumSharp | `x[1,2] = 99` → `nd[1,2] == 99` | [`QuickStart_OneBufferBothSides`][gate] |
| 3 | View shares later writes; copy never does — both directions | write visibility + `shares_memory == False` | [`Verbs_ViewSharesAndCopyDetaches_BothDirections`][gate] |
| 4 | `ToNumpy(copy: true)` routes to the copy | no write-through, no shared memory | [`Verbs_CopyTrueRoutesToTheCopy`][gate] |
| 5 | `To…` copies, `As…` shares, incl. read-only opt-in | write-through matrix over the aliases | [`Verbs_ToCopiesAndAsShares`][gate] |
| 6 | View verbs are flat in *n*; copy verbs are linear | 10⁴× size step moves copy time >50×, view time <5× | [`Costs_ViewVerbsAreFlat_CopyVerbsScale`][gate] |
| 7 | Every export layout keeps its strides and flags | strides/flags per table row; `shares_memory == True` | [`Export_TableRows_KeepStridesAndFlags`][gate] |
| 8 | Exports chain to a ctypes base; strided exports via `as_strided` | base types `c_char_Array_N` / `DummyArray`; `OWNDATA=False` | [`Export_BaseChain_EndsAtTheCtypesWindow`][gate] |
| 9 | All three import routes produce writable views | a write crosses back on each route | [`Import_ThreeRoutes_EachYieldsAView`][gate] |
| 10 | 47 of 50 exporter varieties view; 2 copy; 1 rejected | census constants, re-measured each run | [`Import_Census_47Of50VarietiesView`][gate] |
| 11 | Read-only sources: refused by default, non-writeable on opt-in | verbatim refusal; `IsWriteable == false` | [`Import_ReadonlyRefusedByDefault_OptInIsNonWriteable`][gate] |
| 12 | Import views own nothing; `require("O")` detaches | resize refusal text; detached write | [`ImportView_OwnsNothing_ResizeRefusesAndRequireOCopies`][gate] |
| 13 | An export outlives every managed reference | `Dispose()` + GC, then Python reads and writes | [`Lifetime_ExportOutlivesItsSource`][gate] |
| 14 | The import lease is held by the last view, incl. derived slices | counter stays up until the derived slice dies | [`Lifetime_LeaseIsHeldByTheLastView`][gate] |
| 15 | `LiveExports`/`LiveImports` track conversions exactly | counters move by exactly one per conversion | [`Lifetime_CountersTrackBothDirections`][gate] |
| 16 | A live view blocks reallocation on both sides | `BufferError` + both numpy-worded resize refusals | [`Lifetime_ALiveConversionLocksResizing_BothSides`][gate] |
| 17 | Shutdown drains imports and sweeps orphaned exports | post-shutdown counters at zero, no crash | [`OrphanExport_HeldOnlyByPython_AwaitsTheShutdownSweep`][gate-shutdown], [`OrphanImport_StillReferencedByCSharp_AwaitsTheShutdownDrain`][gate-shutdown] |
| 18 | The registered codec converts at every boundary, as views | `type(c) == ndarray`; writes cross both ways | [`Codec_RegisterOnce_ThenSetAndAsJustWork`][gate] |
| 19 | A pre-registration decode poisons that type pair for the session | verbatim `InvalidCastException`; fresh pairs still work | [`Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned`][gate] |
| 20 | Modes: `Auto` falls back, `View` declines, `Copy` detaches; `Decimal` CLR-wraps | `TryDecode` outcomes; `type(d) == 'NDArray'` | [`Codec_Modes_AutoViewCopy_AndTheDecimalFallback`][gate] |
| 21 | One GIL acquisition can host many conversions | hot loop under a single `Py.GIL()` | [`Gil_OneAcquisitionManyConversions`][gate] |
| 22 | Python → .NET callback bodies do not hold the GIL | `PyGILState_Check() == 0` inside the body | [`PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere`][gate-gil] |
| 23 | Every dtype row maps as printed; the four specials convert or refuse | round-trips + verbatim errors | [`Dtypes_EveryRowRoundTrips`][gate] |
| 24 | The version table is the guard's own mapping | table compared against `MinimumPythonnetFor` | [`Versions_TableIsTheGuardsOwnMapping`][gate] |
| 25 | Every quoted Troubleshooting symptom is the message actually raised | verbatim assertion per row | [`Troubleshooting_SymptomsAreVerbatim`][gate] |

---

## See also

- [Any library via np.frombuffer](np-frombuffer.md) — reaching consumers that want bytes, not numpy
  arrays: `ToMemoryView` + the buffer protocol, torch/Pillow/Arrow/OpenCV verified
- [Numpy.NET](numpy-net.md) — driving these same buffers through SciSharp's `Numpy.Bare` C# API
- [Interoperability](index.md) — the contract underneath every NumSharp bridge

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/DocExamples.PythonnetNumpyPage.cs
[gate-gil]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/GilPolicyTests.cs
[gate-shutdown]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Interop.UnitTests/ShutdownLeakTests.cs
