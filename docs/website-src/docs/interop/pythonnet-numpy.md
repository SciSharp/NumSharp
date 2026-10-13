# Python & numpy - the same buffer on both sides

`NumSharp.Interop.pythonnet` embeds CPython in your .NET process through pythonnet and hands arrays
across the boundary without copying them: a NumSharp `NDArray` becomes a numpy view over the very
same unmanaged bytes, and any Python buffer becomes a NumSharp view over Python's memory. Four verbs
cover the surface - two per direction, one sharing and one copying - a registrable codec makes the
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
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop`.

---

## From zero to a shared array

```bash
dotnet add package NumSharp.Interop.pythonnet
```

That resolves pythonnet 3.0.5 - the floor of the package's `[3.0.5, 4.0.0)` range, and what NuGet
actually installs, since it picks the lowest applicable version - which drives Python 3.7 through
3.13. Point pythonnet at your CPython and start the engine once per process:

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";   // or set the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();   // release the GIL from this thread; conversions re-acquire it
```

`BeginAllowThreads` is what lets any thread convert afterwards - including threads that have never
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
`array` / `asarray` split: **`To…` copies, `As…` shares** - with `ToNumpy` as the deliberate
exception, because the zero-copy view is the package's headline.

| Direction | Verb | Returns | Memory |
|---|---|---|---|
| NumSharp → Python | `nd.ToNumpy()` | numpy view | **shared**, source rooted |
| NumSharp → Python | `nd.ToNumpyCopy()` | numpy array | independent |
| Python → NumSharp | `py.ToNDArray()` | `NDArray` | independent, C-contiguous |
| Python → NumSharp | `py.AsNDArray()` | `NDArray` view | **shared**, exporter leased |

`ToNDArrayView(py, allowReadonly)` is the static spelling of `AsNDArray`; `nd.ToPython()` aliases
`ToNumpy`; `nd.ToNumpy(copy: true)` routes to `ToNumpyCopy`. `nd.ToMemoryView()` - raw writable
bytes for consumers that never touch numpy - has [its own page](np-frombuffer.md).

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
lifetime hook - the elements are never read. Measured at one million `float64`, best of 5, on the
stack in the banner:

| Verb | Time | Scales with n |
|---|---:|---|
| `ToMemoryView` | 0.0069 ms | no |
| `ToNumpy` | 0.0088 ms | no |
| `ToNDArrayView` | 0.0273 ms | no |
| `ToNDArray` (copy) | 0.6755 ms | yes |
| `ToNumpyCopy` | 1.1266 ms | yes |

At ten million elements the gap is three orders of magnitude; at a thousand it inverts - sharing
has a fixed setup cost (a ctypes window, a `frombuffer`, a `weakref.finalize` registration) that
copying eight kilobytes undercuts. The crossover sits around ten thousand elements. Below it, if
you have no reason to share, copy: it is simpler, it locks nothing, and it outlives the
interpreter.

A view also carries obligations a copy does not - the lifetime coupling and the resize locks of
[Lifetime](#lifetime-two-collectors-one-buffer). Choose a view for size or for shared mutation;
choose a copy for a detached snapshot.

<sub>See here [`Costs_ViewVerbsAreFlat_CopyVerbsScale`][gate]</sub>

---

## Exporting: every layout, strides intact

### What does numpy see when I export a slice or a transpose?

**The exact same strided window - never a copy, never a flattening.** `ToNumpy` expresses the
NumSharp layout in numpy's own terms: dimension sizes, byte strides, and the writeable flag.

| NumSharp source | numpy sees |
|---|---|
| C-contiguous `(4,6)` float64 | C-contiguous, strides `(48, 8)` |
| row slice `b["1:3"]` | strides `(48, 8)` over the shared buffer |
| column slice `b[":, ::2"]` | strides `(48, 16)` |
| transpose `b.T` | strides `(8, 48)`, `F_CONTIGUOUS=True` |
| reversed `b["::-1"]` | strides `(-48, 8)` - negative strides survive |
| F-order (`np.asfortranarray`) | strides `(8, 32)`, `F_CONTIGUOUS=True` |
| broadcast `(3,) → (2,3)` | strides `(0, 8)`, **`WRITEABLE=False`** |
| 0-d scalar | `shape=()`; `x[()] = 7.5` writes through |
| empty `(0,3)` | `shape=(0, 3)` - a fresh array; there are no bytes to share |

The table is exhaustive over the layout classes NumSharp produces. Broadcast views are read-only on
the numpy side because they are read-only in NumSharp - one element pretending to be many corrupts
under a blind write, and both libraries guard it the same way.

```text
sl = full[1:3, ::2]-equivalent exported directly:
  sl.tolist()              [[6.0, 8.0, 10.0], [12.0, 14.0, 16.0]]
  np.shares_memory(full, sl)   True
```

<sub>See here [`Export_TableRows_KeepStridesAndFlags`][gate]</sub>

### What is the exported array made of?

**A numpy view chained onto a ctypes window over NumSharp's pointer.** `ToNumpy` builds
`(ctypes.c_char * nbytes).from_address(ptr)`, runs `np.frombuffer` over it, and reshapes - or, for
strided sources, `np.lib.stride_tricks.as_strided` with the exact byte strides:

```text
full:  strides=(48, 8)   base=ndarray -> c_char_Array_192    OWNDATA=False
slice: strides=(48, 16)  base=DummyArray (as_strided)        OWNDATA=False
```

`OWNDATA=False` is numpy agreeing the memory is NumSharp's. The deepest base object - the ctypes
window - is where the export's `weakref.finalize` is rooted, which is why every derived numpy view
(`arr[1:]`, `arr.T`, `np.asarray(arr)`) extends the buffer's life: they all chain back to it.

<sub>See here [`Export_BaseChain_EndsAtTheCtypesWindow`][gate]</sub>

---

## Importing: three zero-copy routes

### Which Python objects can NumSharp view without copying?

**Almost all of them - 50 of the 55 exporter varieties in the census below.** `AsNDArray` /
`ToNDArrayView` takes one of three routes, chosen by what the exporter is:

1. **C-contiguous PEP 3118 exporters** - any object: numpy arrays, `bytes`, `bytearray`,
   `array.array`, `memoryview`, ctypes arrays, `BytesIO.getbuffer()`, mmaps, shared memory. The
   buffer is acquired with `PyBUF.WRITABLE`, which pins the exporter and blocks reallocation.
2. **Non-contiguous numpy arrays** - slices, transposes, Fortran order, broadcasts: imported
   through `__array_interface__` as a strided NumSharp view with the identical layout; broadcast
   sources arrive read-only.
3. **Non-contiguous non-numpy exporters** - a sliced, offset or reversed `memoryview`, a strided
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

The census, re-measured on every test run - 55 exporter varieties across builtins, `array.array`'s
twelve typecodes, ctypes element types, numpy dtypes, numpy layouts, memoryview forms and text:
**50 view, 4 copy, 1 rejected.** The four copies are complex64 (widened to `Complex`), a
sub-element-stride `as_strided` window (linearized), and the two text forms with no buffer - a
`StringDType` array and an object array of `str`, both copied to a NumSharp `StringDType` array
([How does text cross?](#how-does-text-cross)); fixed-width `U` and `S` arrays view. The one rejection
is `datetime64`, whose element type has no NumSharp dtype at any byte order. **Big-endian multi-byte data is a third copy** - the
view path refuses it (a native-endian shared view is impossible), but `ToNDArray` byte-reverses each
element into a value-correct native array, exactly as it widens complex64.

<sub>See here [`Import_ThreeRoutes_EachYieldsAView`][gate], [`Import_Census_50Of55VarietiesView`][gate]</sub>

### Can I view something read-only, like `bytes`?

**Only by saying so - and the view arrives non-writeable.** By default a read-only source is
refused with guidance, so a view you can write through is the only thing a default call can return:

```text
InvalidOperationException: the exporter's buffer is read-only; writing through a NumSharp view
would corrupt an immutable Python object. Use ToNDArray (copy), or pass allowReadonly:true to
take a NON-WRITEABLE view (guarded writes through it throw).
```

With `allowReadonly: true` the view's `Shape.IsWriteable` is `false` - numpy's `writeable=False`,
carried across - and a write raises `assignment destination is read-only` instead of corrupting
the source.

<sub>See here [`Import_ReadonlyRefusedByDefault_OptInIsNonWriteable`][gate]</sub>

### Does an imported view own its memory?

**No - and it tells you so, exactly like `np.frombuffer` arrays do.** An import view has numpy's
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
wrapper, run the GC - Python still reads and writes valid memory, and the buffer is freed when the
last Python-side view dies:

```text
after nd.Dispose() + GC:   a = [0. 1. 2. 3.]     LiveExports = 1
a[0] = 7.0                 a = [7. 1. 2. 3.]     (still writable)
del a, mv; gc.collect()                          LiveExports = 0
```

<sub>See here [`Lifetime_ExportOutlivesItsSource`][gate]</sub>

### Can my `NDArray` outlive Python's object?

**Yes - the lease holds the exporter alive.** An imported view keeps a `Py_buffer` lock (or, for
strided numpy sources, a strong reference) on the exporter; Python can `del` every name it has. The
lease is released when the last NumSharp view over the memory - *including derived slices* - is
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
array, `nd.resize(...)` raises NumPy's own wording - `cannot resize an array that references or is
referenced by another array in this way`. Dispose the view and both locks lift.

<sub>See here [`Lifetime_ALiveConversionLocksResizing_BothSides`][gate]</sub>

### What happens at engine shutdown?

**Imports are drained crash-free; orphaned exports are swept right after the engine dies.** Import
views are tied to the interpreter that owns their memory: after `PythonEngine.Shutdown()` they must
not be touched, though disposing them stays safe - the shutdown handler releases every outstanding
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

**Yes - register once, then every pythonnet boundary converts by itself.** `RegisterCodec()` hooks
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

An implicit encode does not linger: pythonnet takes its own reference from the wrapper an encoder
returns and never disposes that wrapper, which would keep every `scope.Set("x", nd)` / `py.x = nd` /
`numpy.sum(nd)` export pinned until the CLR finalizer ran. The codec hands its wrapper to a per-thread
slot that is disposed at the next encode on that thread and at the next conversion verb, so rebinding
a name in a loop keeps exactly one export live. (A `dynamic` *read* of a numpy array is an ordinary
pythonnet wrapper and lives until the GC; where a reference count matters, read the object as
`using PyObject` or read a scalar through `scope.Eval(...)`.)

<sub>See here [`RebindingANameThroughTheCodec_KeepsTheLiveExportCountBounded`][gate-handoff]</sub>

### When must I register it?

**Before the first `As<NDArray>()` anywhere in the process - this is the one ordering trap.**
pythonnet caches the decoder lookup per (Python type, target type) pair, and it caches misses. A
decode attempted before registration fails - and keeps failing for that Python type for the rest of
the engine session, even after you register:

```text
1. BEFORE RegisterCodec: As<NDArray>(ndarray)   -> InvalidCastException:
       'numpy.ndarray' value cannot be converted to NumSharp.NDArray
2. RegisterCodec()                              -> True
3. AFTER RegisterCodec:  As<NDArray>(ndarray)   -> still the same InvalidCastException
4. AFTER RegisterCodec:  As<NDArray>(bytes)     -> OK Byte    (pair never tried before)
```

Register at startup, right after `PythonEngine.Initialize()`. Registration is per engine session -
pythonnet clears all codecs during `Shutdown`, and `RegisterCodec()` knows to re-register after a
later `Initialize()`. The explicit verbs (`ToNumpy`, `AsNDArray`, …) never involve the codec and
work regardless.

<sub>See here [`Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned`][gate]</sub>

### Which mode should I pick?

**`Auto` unless you have a reason: it shares when a view is possible and copies only when one is
impossible.** `NumpyCodecOptions` sets one of three modes per direction:

| Mode | On decode | On encode |
|---|---|---|
| `Auto` (default) | view when representable, else copy (complex64 widens, big-endian byte-swaps, `StringDType` and object arrays of `str` copy, sub-element strides linearize) | a view; a copy for a `StringDType` array, which has no buffer |
| `View` | view or **decline** - never a silent copy | view, or **decline** (a `StringDType` array stays a CLR object) |
| `Copy` | always an independent snapshot; never locks the source | always a copy |

Read-only sources decode as non-writeable views under `Auto`/`View`. `Decimal` has no numpy dtype,
so encoding **converts it to a float64 numpy array** (the `astype(NPTypeCode.Double)` guidance,
automated - lossy beyond ~16 significant digits) rather than refusing, in every mode:

```text
scope.Set("d", decimalArray)  ->  d.dtype == float64
View-mode TryDecode: complex64 -> False (declined), float64 -> True (view)
```

<sub>See here [`Codec_Modes_AutoViewCopy_AndTheDecimalConversion`][gate]</sub>

### Do C# tuples cross as Python tuples?

**Yes - `RegisterCodec()` also registers `TupleCodec`, because pythonnet has no tuple conversion of
its own.** Without it a shape written the numpy way through `dynamic`, `numpy.zeros((2, 3))`, reaches
numpy as an opaque wrapped `System.ValueTuple` ("expected a sequence of integers or a single integer"),
and `(long, long) shape = a.shape` cannot convert. With it every place a tuple is idiomatic Python reads
the same in C#:

```csharp
using (Py.GIL())
{
    dynamic numpy = Py.Import("numpy");
    dynamic z = numpy.zeros((2, 3));                 // a C# tuple is the shape
    (long, long) strides = z.strides;                // a Python tuple decodes into a C# one: (24, 8)
    double corner = z.item((1, 2));                  // a tuple multi-index
    dynamic b = numpy.broadcast_to(nd, (2, 4));      // an NDArray and a tuple in one call
}
```

Any arity encodes (8+ elements ride the `Rest` slot), nested tuples and mixed element types too, and
every element crosses through the registered codecs - an `NDArray` element becomes a numpy view, `null`
becomes `None`. Decoding accepts a Python `tuple` or subclass (a namedtuple, `torch.Size`) into a C#
tuple of the **same** arity; a length mismatch or a `list` is declined, never truncated. Opt out with
`NumpyCodecOptions.ConvertTuples = false`.

<sub>See here [`Encode_ValueTupleShape_ReachesNumpyAsATuple`][gate-tuples], [`Decode_ShapeAndStrides_IntoValueTuples_ThroughDynamic`][gate-tuples]</sub>

---

## The GIL

Every conversion verb **acquires the GIL itself**, re-entrantly - nesting inside your own
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

`requireGIL: false` means **you** hold the GIL - converting without actually holding it is an
immediate access violation, like any raw C-API misuse. The process-wide default is
`NDArrayPythonInterop.RequireGIL` (`true`); the per-call parameter overrides it, and `null` follows
it.

> **The trap.** A .NET method or delegate invoked *from* Python does **not** hold the GIL -
> pythonnet's binder releases it around managed bodies. Inside a Python → .NET callback, leave GIL
> management on.

The package's background machinery (deferred lease disposal, the shutdown drain) always manages
the GIL itself, so the opt-out never applies to it.

<sub>See here [`Gil_OneAcquisitionManyConversions`][gate], [`PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere`][gate-gil]</sub>

---

## Dtypes

Fourteen of NumSharp's fifteen numeric dtypes cross, and so does text: fixed-width `S` and `U` as
views in both directions, NEP 55 `StringDType` as an exact copy. The maps are public
(`ToNumpyDtypeStr` / `FromNumpyDtypeStr` / `ToBufferFormat` / `FromBufferFormat`; the two `From…`
maps return a `DType`, because a text dtype carries its width):

| NumSharp | numpy | PEP 3118 | numpy calls it |
|---|---|---|---|
| `Boolean` | `\|b1` | `?` | `bool` |
| `Byte` / `SByte` | `\|u1` / `\|i1` | `B` / `b` | `uint8` / `int8` |
| `Int16` / `UInt16` | `<i2` / `<u2` | `h` / `H` | `int16` / `uint16` |
| `Int32` / `UInt32` | `<i4` / `<u4` | `i` / `I` | `int32` / `uint32` |
| `Int64` / `UInt64` | `<i8` / `<u8` | `q` / `Q` | `int64` / `uint64` |
| `Half` / `Single` / `Double` | `<f2` / `<f4` / `<f8` | `e` / `f` / `d` | `float16` / `float32` / `float64` |
| `Complex` | `<c16` | `Zd` | `complex128` |
| `Char` | `<u2` | `H` | `uint16` - a C# `char` is a UTF-16 code unit; numpy has no char dtype |
| `S<n>` (bytes) | `\|S<n>` | `<n>s` | `bytes<8n>` - `n` bytes, NUL-padded |
| `U<n>` (text) | `<U<n>` | `<n>w` | `str<32n>` - `n` UCS-4 code points, NUL-padded |
| `StringDType(...)` | `StringDType(...)` | *none - copies* | `StringDType` - NEP 55 variable-width UTF-8 |
| `Decimal` | *throws* | *throws* | 16 bytes, non-IEEE; the maps refuse honestly, but `ToNumpy`/`ToNumpyCopy` auto-convert it to `float64` |

Four numeric element types deserve their fine print, all four handled by conversion rather than
misrepresentation:

- **complex64** (`c8` / `Zf`) - two 4-byte floats where `Complex` is two 8-byte doubles. No view is
  possible; `ToNDArray` widens each pair during the copy.
- **wide characters** (`wchar_t`) - a 2-byte `wchar_t` buffer (`array.array('u')` and
  `ctypes.c_wchar` on Windows) *is* UTF-16 and views zero-copy as `Char`; a 4-byte one
  (`array.array('w')`, and `array('u')` / `c_wchar` on Linux and macOS) holds UCS-4 code points and
  views as `U1`.
- **big-endian** (`>i4`, `!H`, `>U2`, …) - no zero-copy *view* for multi-byte types (a native read
  would byte-swap every value), so the view path refuses it with the byte-swap fix in the message.
  But `ToNDArray` **byte-reverses each element during the copy** - complex128 reverses each 8-byte
  half, complex64 reverses then widens, `U` reverses each 4-byte code point - so big-endian data
  crosses value-correct as a copy. Single-byte dtypes (`>i1`, `\|u1`, `\|b1`, `S`) view zero-copy
  regardless - byte order is meaningless at one byte.
- **long double** (`g`) - MSVC's 8-byte long double is IEEE double and views as `Double`; the
  extended-precision widths have no NumSharp dtype and throw with `astype(np.float64)` guidance.

<sub>See here [`Dtypes_EveryRowRoundTrips`][gate]</sub>

### How does text cross?

**Fixed-width `S` and `U` arrays are ordinary buffers and share memory both ways; a `StringDType`
array has nothing to share and crosses as an exact copy.**

```csharp
using (Py.GIL())
{
    Scope.Exec("u = np.array(['ab', 'cde'], dtype='U3')");
    using PyObject u = Scope.Get("u");
    using NDArray view = u.AsNDArray();          // U3, a view over numpy's buffer
    Scope.Exec("u[0] = 'zz'");                   // ...so numpy's write shows in view

    Scope.Exec("t = np.array(['short', 'a string longer than 15 bytes', None], " +
               "dtype=np.dtypes.StringDType(na_object=None))");
    using PyObject t = Scope.Get("t");
    using NDArray copy = t.ToNDArray();          // StringDType(na_object=None), NA kept
}
```

```text
view.GetValue(0)    "zz"
copy.dtype          StringDType(na_object=None)
copy.GetValue(2)    null  (the NA value)
```

- **`S<n>` and `U<n>`** cross like numbers: `ToNumpy` and `AsNDArray` share the bytes, NUL padding
  included, with every layout of the [export table](#what-does-numpy-see-when-i-export-a-slice-or-a-transpose)
  and every [import route](#which-python-objects-can-numsharp-view-without-copying); the copies carry
  the same bytes. `<U1` is the one-character `U1`, never `Char`: a code point is 4 bytes and a C#
  `char` 2, so `U1` holds an astral character (`'😀'`) that a `char` cannot. Values read the way
  numpy reads them - trailing NULs are padding, an embedded NUL is kept, a `U` element that is not a
  Unicode scalar value crosses as its raw 4-byte unit.
- **`np.str_` and `np.bytes_` scalars** are the 0-d `U<n>` and `S<n>` they are
  (`np.asarray(np.bytes_(b'ab'))` is `S2`, not two bytes). An empty one has no byte to share, so
  `ToNDArray` copies it to the one-character `''` numpy reads, and the view refuses.
- **`StringDType`** (`T`) keeps the text of every element longer than 15 bytes in an allocator its
  descriptor owns, outside the array buffer. `ToNDArray` and `ToNumpyCopy` copy it through numpy's
  own `NpyString` C API, element by element with `NpyString_dup`'s rules: an NA value stays NA, an
  empty string stays empty, embedded and trailing NULs and non-BMP text are kept, any layout
  (strided, reversed, broadcast, 0-d). The copy keeps the descriptor's `na_object` and `coerce`, and
  its cells are byte-for-byte the cells numpy's own copy builds. `ToNumpy`, `AsNDArray` and
  `ToMemoryView` refuse it with that copy route in the message.
- **The missing-value object** crosses when it has a C# form: `None` ↔ `null`, `bool`, `int` ↔
  `long` (`BigInteger` past it), `float` ↔ `double`, `complex` ↔ `Complex`, `str` ↔ `string`,
  `bytes` ↔ `byte[]`, `np.float16` ↔ `Half`. Any other object (`pd.NA`, a class of your own) is
  refused in both directions, rather than guessed.
- **Object arrays** of `str` - what pandas text columns hold - copy to `StringDType()`. Any other
  object array (a `None`, a NaN, a number among the strings) is refused with the first such element
  named; convert it in Python first, e.g. `np.array(arr, dtype=np.dtypes.StringDType(na_object=None))`.
- **PyTorch** has no string tensors: `ToTorch` refuses a text array before exporting anything.

**What a `StringDType` copy costs.** Microseconds per call, minimum of three runs of best-of-9
rounds, pinned to two cores, on the stack in the banner; *short* is 2–5 characters (held in the
16-byte cell), *long* 21–40 (held in the allocator's arena), *mixed* adds 10 % NA:

| Data, n | numpy `a.copy()` | Python route (`tolist()` + rebuild) | `ToNDArray` | `ToNumpyCopy` |
|---|---:|---:|---:|---:|
| short, 1 000 | 17.0 | 160 | 9.5 | 11.7 |
| short, 100 000 | 1 874 | 17 112 | 826 | 1 216 |
| short, 1 000 000 | 18 353 | 171 185 | 7 875 | 11 840 |
| long, 1 000 | 43.4 | 188 | 36.0 | 49.5 |
| long, 100 000 | 7 749 | 25 023 | 4 285 | 8 284 |
| long, 1 000 000 | 79 156 | 258 126 | 41 217 | 85 795 |
| mixed, 1 000 | 30.0 | 190 | 22.8 | 28.9 |
| mixed, 100 000 | 3 590 | 21 359 | 2 503 | 3 422 |
| mixed, 1 000 000 | 46 093 | 222 544 | 23 958 | 44 835 |

Against the Python route - one Python `str` per element, the only way to move this text without the
bridge - both copies are 3–22× faster at every size. Against numpy's own in-memory `copy()`, the
import is 1.2–2.3× faster (it reserves its arena once, where numpy's grows by 25 % steps) and the
export of short strings 1.5× (a short string moves as its 16-byte cell); the export of long and
mixed strings runs at 0.88–1.05× numpy's speed, because numpy's allocator stores each one through
`NpyString_pack` - the same per-element work its own copy does, and the only public way to write
numpy's cells. A fixed-width view does not grow with *n*: `ToNumpy` takes 5.7–6.6 µs and
`AsNDArray` 4.0–4.1 µs for `U10` / `S10` arrays from a thousand to a million elements.

<sub>See here [`TextFixedWidthCrossingTests`][gate-text-fixed], [`TextStringDTypeCrossingTests`][gate-text-vstring], [`Text_FixedWidthViews_StringDTypeCopies`][gate]</sub>

---

## Versions

### Which pythonnet do I need for my Python?

**The package floor (3.0.5) covers Python 3.7–3.13; Python 3.14 needs pythonnet 3.1.0.** Each
pythonnet release hard-caps the newest Python it can drive. The mapping below is read out of each
release's own `PythonEngine.MaxSupportedVersion` - and the package checks it once per session,
turning pythonnet's opaque symbol-load failures into an actionable error naming the version to
install:

| Your Python | Minimum pythonnet |
|---|---|
| 3.7 – 3.10 | 3.0.0 |
| 3.11 | 3.0.1 |
| 3.12 | 3.0.3 |
| 3.13 | 3.0.5 |
| 3.14 | 3.1.0 |

The floor is 3.0.5 because its range (3.7–3.13) is a strict superset of every earlier 3.0.x - the
older floors silently shipped defaults that could not run current Pythons. The `4.0.0` upper bound
keeps a future breaking pythonnet from resolving into your build unasked; the source compiles clean
against all of 3.0.0–3.1.0, so pinning 3.1.0 yourself is supported:

```xml
<PackageReference Include="pythonnet" Version="3.1.0" />  <!-- only for Python 3.14 -->
```

One version-specific repair worth knowing exists but not relying on: pythonnet 3.0.1's `PyBuffer`
is broken for shape/strides/format flags, so the package reads all buffer *metadata* through
Python's own `memoryview` and uses `PyBuffer` only with the crash-free flags - which is why every
import route works uniformly across pythonnet 3.0.x.

<sub>See here [`Versions_TableIsTheGuardsOwnMapping`][gate]</sub>

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `Python engine is not initialized` | Set `Runtime.PythonDLL` (or `PYTHONNET_PYDLL`) and call `PythonEngine.Initialize()` before any conversion |
| `Python 3.x is not supported by the loaded pythonnet` | Your pythonnet caps out below your Python. Install the version the message names (table above) |
| `'numpy.ndarray' value cannot be converted to NumSharp.NDArray` - even after `RegisterCodec()` | A decode ran before registration and pythonnet cached the miss for that Python type. Register at startup, before the first `As<NDArray>()` |
| `the exporter's buffer is read-only` | You asked for a writable view of `bytes` or a `writeable=False` array. Pass `allowReadonly: true`, or `ToNDArray()` to copy |
| `assignment destination is read-only` | You wrote through a non-writeable view - a read-only or broadcast source. Copy first if you need to mutate |
| `Existing exports of data: object cannot be re-sized` | A live NumSharp view leases that object. `Dispose()` the view to release the lock |
| `cannot resize an array that references or is referenced by another array in this way` | A live Python export (or another NumSharp view) pins the buffer. Release it, or copy first |
| `cannot resize this array: it does not own its data` | Import views never own their memory. `np.require(view, null, "O")` for an owning copy |
| `big-endian dtype '>f8' cannot be shared with a native-endian NumSharp buffer` | Only the zero-copy *view* refuses big-endian. `ToNDArray()` copies it and byte-reverses each value; or byte-swap on the Python side first: `arr.astype(arr.dtype.newbyteorder('<'))` |
| `big-endian UCS-4 text '>U2' cannot be shared with a native-endian NumSharp buffer` | Only the zero-copy *view* refuses big-endian text. `ToNDArray()` copies it and byte-reverses each code point into a native `U2` |
| `StringDType() has no buffer form` | A `StringDType` array keeps its text outside its buffer, so no view can share it. `ToNDArray()` / `ToNumpyCopy()` copy it exactly, NA values included |
| `the object array holds a NoneType at flat index 1 (None), which is not a str` | Only an object array whose every element is a `str` crosses (as a `StringDType` copy). Convert it in Python first: `np.array(arr, dtype=np.dtypes.StringDType(na_object=None))` |
| `the StringDType's na_object … has no C# form in NumSharp` | The missing-value object is `pd.NA` or another object with no C# counterpart. Rebuild the array with `None`, NaN or a string as its NA: `arr.astype(np.dtypes.StringDType(na_object=None))` |
| `PyTorch has no string tensors` | Text has no tensor form. Cross it to numpy instead (`ToNumpy` / `ToNumpyCopy`) |
| `decimal has no numpy dtype (16-byte, non-IEEE)` | Nothing in numpy describes it. `nd.astype(NPTypeCode.Double)` first |
| `the object does not export a PEP 3118 buffer` | Not every object is an exporter - a `dict`, a PIL `Image`. `np.asarray(obj)` first if numpy understands it |
| A Python write did not show up in NumSharp | You copied. `ToNumpy`/`AsNDArray` share; `ToNumpyCopy`/`ToNDArray` do not |
| Access violation around a conversion | `requireGIL: false` on a thread that does not hold the GIL - including inside a Python → .NET callback, where pythonnet releases it |
| `PythonEngine.Shutdown()` crashes on .NET 8+ | pythonnet's state stashing uses the removed `BinaryFormatter`. Set `RuntimeData.FormatterType = typeof(NoopFormatter)` first |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | After `BeginAllowThreads`, a fresh thread converts | conversion succeeds on a thread that never touched Python | [`Bootstrap_AFreshThreadConverts`][gate] |
| 2 | Export is a shared view; Python writes reach NumSharp | `x[1,2] = 99` → `nd[1,2] == 99` | [`QuickStart_OneBufferBothSides`][gate] |
| 3 | View shares later writes; copy never does - both directions | write visibility + `shares_memory == False` | [`Verbs_ViewSharesAndCopyDetaches_BothDirections`][gate] |
| 4 | `ToNumpy(copy: true)` routes to the copy | no write-through, no shared memory | [`Verbs_CopyTrueRoutesToTheCopy`][gate] |
| 5 | `To…` copies, `As…` shares, incl. read-only opt-in | write-through matrix over the aliases | [`Verbs_ToCopiesAndAsShares`][gate] |
| 6 | View verbs are flat in *n*; copy verbs are linear | 10⁴× size step moves copy time >50×, view time <5× | [`Costs_ViewVerbsAreFlat_CopyVerbsScale`][gate] |
| 7 | Every export layout keeps its strides and flags | strides/flags per table row; `shares_memory == True` | [`Export_TableRows_KeepStridesAndFlags`][gate] |
| 8 | Exports chain to a ctypes base; strided exports via `as_strided` | base types `c_char_Array_N` / `DummyArray`; `OWNDATA=False` | [`Export_BaseChain_EndsAtTheCtypesWindow`][gate] |
| 9 | All three import routes produce writable views | a write crosses back on each route | [`Import_ThreeRoutes_EachYieldsAView`][gate] |
| 10 | 50 of 55 exporter varieties view; 4 copy; 1 rejected | census constants, re-measured each run | [`Import_Census_50Of55VarietiesView`][gate] |
| 11 | Read-only sources: refused by default, non-writeable on opt-in | verbatim refusal; `IsWriteable == false` | [`Import_ReadonlyRefusedByDefault_OptInIsNonWriteable`][gate] |
| 12 | Import views own nothing; `require("O")` detaches | resize refusal text; detached write | [`ImportView_OwnsNothing_ResizeRefusesAndRequireOCopies`][gate] |
| 13 | An export outlives every managed reference | `Dispose()` + GC, then Python reads and writes | [`Lifetime_ExportOutlivesItsSource`][gate] |
| 14 | The import lease is held by the last view, incl. derived slices | counter stays up until the derived slice dies | [`Lifetime_LeaseIsHeldByTheLastView`][gate] |
| 15 | `LiveExports`/`LiveImports` track conversions exactly | counters move by exactly one per conversion | [`Lifetime_CountersTrackBothDirections`][gate] |
| 16 | A live view blocks reallocation on both sides | `BufferError` + both numpy-worded resize refusals | [`Lifetime_ALiveConversionLocksResizing_BothSides`][gate] |
| 17 | Shutdown drains imports and sweeps orphaned exports | post-shutdown counters at zero, no crash | [`OrphanExport_HeldOnlyByPython_AwaitsTheShutdownSweep`][gate-shutdown], [`OrphanImport_StillReferencedByCSharp_AwaitsTheShutdownDrain`][gate-shutdown] |
| 18 | The registered codec converts at every boundary, as views | `type(c) == ndarray`; writes cross both ways | [`Codec_RegisterOnce_ThenSetAndAsJustWork`][gate] |
| 19 | A pre-registration decode poisons that type pair for the session | verbatim `InvalidCastException`; fresh pairs still work | [`Codec_RegisterBeforeFirstConversion_OrThePairIsPoisoned`][gate] |
| 20 | Modes: `Auto` falls back, `View` declines, `Copy` detaches; `Decimal` → float64 | `TryDecode` outcomes; `d.dtype == float64` | [`Codec_Modes_AutoViewCopy_AndTheDecimalConversion`][gate] |
| 21 | One GIL acquisition can host many conversions | hot loop under a single `Py.GIL()` | [`Gil_OneAcquisitionManyConversions`][gate] |
| 22 | Python → .NET callback bodies do not hold the GIL | `PyGILState_Check() == 0` inside the body | [`PythonToNetCallbackBody_DoesNotHoldTheGil_SoTheOptOutIsWrongThere`][gate-gil] |
| 23 | Every dtype row maps as printed; the four specials convert or refuse | round-trips + verbatim errors | [`Dtypes_EveryRowRoundTrips`][gate] |
| 24 | The version table is the guard's own mapping | table compared against `MinimumPythonnetFor` | [`Versions_TableIsTheGuardsOwnMapping`][gate] |
| 25 | Every quoted Troubleshooting symptom is the message actually raised | verbatim assertion per row | [`Troubleshooting_SymptomsAreVerbatim`][gate] |
| 26 | C# tuples cross as Python tuples and back, through the registered codec | `numpy.zeros((2, 3))` takes the shape; `(long, long) = a.shape` decodes | [`Encode_ValueTupleShape_ReachesNumpyAsATuple`][gate-tuples], [`Decode_ShapeAndStrides_IntoValueTuples_ThroughDynamic`][gate-tuples] |
| 27 | An implicit encode's wrapper is released at the next encode / conversion, so a rebinding loop keeps one export live | 150 rebindings peak ≤ 2 above baseline; `sys.getrefcount` shows no lingering wrapper | [`RebindingANameThroughTheCodec_KeepsTheLiveExportCountBounded`][gate-handoff], [`TheHandedOffWrapperIsRedundant_PythonnetHoldsItsOwnReference`][gate-handoff] |
| 28 | Fixed-width `S` / `U` arrays share memory both ways in every layout; `<U1` is `U1`; big-endian `U` copies byte-swapped; text scalars are 0-d | C-order bytes, write-through both ways and flags per layout, against live numpy | [`Export_EveryLayout_IsAViewWithTheSameBytes`][gate-text-fixed], [`Import_EveryLayout_ViewsAndCopiesWithTheSameBytes`][gate-text-fixed], [`BigEndianText_CopiesByteSwapped_ViewRefused`][gate-text-fixed], [`TextScalars_AreZeroDText_EmptyOnesCopy`][gate-text-fixed] |
| 29 | A `StringDType` array crosses as an exact copy both ways: the descriptor, the values with NA identity, and numpy's own cells | every NA kind × both `coerce` × every layout against live numpy; `tobytes()` equal | [`Import_EveryNAKind_EveryLayout_SameDescriptorValuesAndCells`][gate-text-vstring], [`Export_EveryNAKind_EveryLayout_SameDescriptorValuesAndCells`][gate-text-vstring], [`RoundTrip_NumSharpToNumpyToNumSharp_IsIdentical`][gate-text-vstring] |
| 30 | An object array of `str` copies to `StringDType()`; any other object array, and an NA object with no C# / Python form, is refused | verbatim refusals naming the element | [`ObjectArrays_OfStr_Copy_AnythingElseRefused`][gate-text-vstring], [`ForeignNAObjects_AreRefusedBothWays`][gate-text-vstring] |
| 31 | A `StringDType` copy holds no lease and survives its source; an allocation failure mid-copy leaves nothing behind | counters at baseline; injected failure releases the half-built copy | [`Copies_AreIndependentOfTheSourcesLifetime`][gate-text-vstring], [`FaultInjection_DuringImport_ReleasesTheHalfBuiltCopy`][gate-text-vstring] |
| 32 | The page's text example: a `U3` import is a view, a `StringDType` import a copy with its NA | the three output lines | [`Text_FixedWidthViews_StringDTypeCopies`][gate] |

---

## See also

- [The examples](https://github.com/SciSharp/NumSharp/tree/master/examples/NumSharp.Interop.pythonnet.Examples)
  - twelve single-file `dotnet run` tutorials (bootstrap, verbs, layouts, dtypes, buffers, lifetime,
  codec, GIL, PyTorch, Pandas, a custom adapter, application scenarios), written for reading, with a
  `dynamic` Python namespace and one `NDScope` per script; every script has an asserted twin under
  [`test/NumSharp.Tests.Interop/Examples`](https://github.com/SciSharp/NumSharp/tree/master/test/NumSharp.Tests.Interop/Examples)
- [A local project](https://github.com/SciSharp/NumSharp/tree/master/examples/NumSharpPythonnetLocalExample)
  and [a GitHub Actions project](https://github.com/SciSharp/NumSharp/tree/master/examples/NumSharpPythonnetGhActionsCicd)
  - two projects to copy. The first is a console app that finds the Python on your machine, starts
  the engine inside a virtual environment and runs NumPy, pandas, SciPy, scikit-learn, Matplotlib and
  PyTorch over NumSharp arrays. The second runs the same six libraries as tests, with the workflow
  that sets up Python on Windows, Linux and macOS runners
- [Any library via np.frombuffer](np-frombuffer.md) - reaching consumers that want bytes, not numpy
  arrays: `ToMemoryView` + the buffer protocol, torch/Pillow/Arrow/OpenCV verified
- [Numpy.NET](numpy-net.md) - driving these same buffers through SciSharp's `Numpy.Bare` C# API
- [Interoperability](index.md) - the contract underneath every NumSharp bridge

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/DocExamples.PythonnetNumpyPage.cs
[gate-gil]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/GilPolicyTests.cs
[gate-shutdown]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/ShutdownLeakTests.cs
[gate-tuples]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/TupleCodecTests.cs
[gate-handoff]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/EncoderHandoffTests.cs
[gate-text-fixed]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/TextFixedWidthCrossingTests.cs
[gate-text-vstring]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/TextStringDTypeCrossingTests.cs
