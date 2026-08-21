# Any Python library, zero-copy — via `np.frombuffer`

NumSharp keeps every array in raw unmanaged memory, and CPython has a protocol for passing raw
memory between libraries that have never heard of each other: PEP 3118, the buffer protocol.
`nd.ToMemoryView()` puts a NumSharp array on that protocol as a writable Python `memoryview`, and
from there `np.frombuffer` re-types those same bytes as a numpy array without copying one of them.
Anything numpy can reach — PyTorch, Pillow, Arrow, pandas, polars, OpenCV — you can now reach, and
so can everything that reads a buffer without involving numpy at all: `struct`, `hashlib`, `zlib`,
a socket, a C extension you wrote yesterday.

This page is self-contained. You do not need the rest of the interop documentation to use it.

**On this page:** [The recipe](#the-recipe) · [What it costs](#what-it-costs) ·
[What can be exported](#what-can-be-exported) · [Reading Python's memory](#reading-pythons-memory) ·
[Lifetime](#lifetime-who-frees-what) · [The GIL](#the-gil) · [Seven libraries](#seven-libraries-verbatim) ·
[Troubleshooting](#troubleshooting) · [Claims](#claims-ledger)

> Verified on CPython 3.12.12 · numpy 2.4.2 · pythonnet 3.0.5 · net8.0/net10.0.
> Every claim below is reproduced by a test in `NumSharp.Tests.Interop`.

---

## The recipe

```bash
dotnet add package NumSharp.Interop.pythonnet
```

```csharp
using NumSharp;
using NumSharp.Interop.PythonNet;
using Python.Runtime;

Runtime.PythonDLL = @"C:\Python312\python312.dll";   // or the PYTHONNET_PYDLL env var
PythonEngine.Initialize();
PythonEngine.BeginAllowThreads();
```

Three lines do the work. Take the bytes, name the dtype, hand both to Python:

```csharp
var nd = np.arange(6).reshape(2, 3).astype(NPTypeCode.Double);

using (Py.GIL())
{
    using var scope = Py.CreateScope();
    scope.Exec("import numpy as np");

    string dtype = NDArrayPythonInterop.ToNumpyDtypeStr(nd.typecode);   // "<f8"
    string shape = string.Join(", ", nd.shape);                         // "2, 3"

    using PyObject mv = nd.ToMemoryView();                              // raw bytes, no copy
    scope.Set("mv", mv);
    scope.Exec($"a = np.frombuffer(mv, '{dtype}').reshape({shape})");

    scope.Exec("a[0, 0] = -5.0");                                       // Python writes...
}

// ...and NumSharp sees it — same memory:
Console.WriteLine(nd);        // [[-5. 1. 2.] [3. 4. 5.]]
```

`ToNumpyDtypeStr` is public and total over the dtypes numpy can express, so the recipe generalises
without a switch statement of your own. `np.frombuffer` always produces a **1-D** array — the
`reshape` is what gives it back its dimensions, and it costs nothing.

<sub>See here [`Recipe_RoundTripsThroughFrombuffer`][gate], [`DtypeStrings_AreTotalOverNumpyExpressibleDtypes`][gate]</sub>

### Why not just `ToNumpy()`?

**Because a `memoryview` reaches consumers a numpy array cannot.** There are two ways to hand
NumSharp memory to Python, and they are good at different things:

| | `nd.ToNumpy()` | `nd.ToMemoryView()` + `np.frombuffer` |
|---|---|---|
| Result | a `numpy.ndarray` the bridge built | raw bytes you type yourself |
| Layouts | **every** layout — slices, transposes, F-order, negative strides, broadcasts | C-contiguous only |
| Needs numpy importable | yes | **no** |
| Reaches non-numpy consumers | only via `np.asarray` round-trips | directly — `struct`, `hashlib`, sockets, `PIL.Image.frombuffer`, `torch.frombuffer`, `pa.py_buffer` |
| Shape/dtype | carried automatically | you supply both |

Use `ToNumpy()` when the destination wants a numpy array and your data may be strided. Use
`ToMemoryView()` when the destination wants *bytes* — which is what this page is about.

---

## What it costs

### Does the buffer route scale with array size?

**No. It is flat, and the copy route is linear.** The memoryview is a pointer, a length and a
lifetime hook; nothing is read, so the size of the array never enters the cost.

| Elements (float64) | `ToMemoryView` | `ToNumpy` | `ToNumpyCopy` |
|---:|---:|---:|---:|
| 1,000 | 0.0073 ms | 0.0066 ms | 0.0031 ms |
| 100,000 | 0.0065 ms | 0.0066 ms | 0.0147 ms |
| 10,000,000 | 0.0070 ms | 0.0067 ms | 12.8216 ms |

Best of 7, warm engine, on the stack named in the banner. At ten million elements the buffer route
is roughly **1,800× cheaper** than copying, and it will keep pulling away — the left two columns are
constants.

**Read the first row again, though: at a thousand elements the copy wins.** Sharing has a fixed
setup cost of about seven microseconds — a `ctypes` window, a `frombuffer` call, a `weakref.finalize`
registration — and copying eight kilobytes is faster than that. The crossover sits somewhere around
ten thousand elements. Below it, if you have no reason to share, don't: a copy is simpler, it locks
nothing, and it outlives the interpreter.

<sub>See here [`BufferRoute_IsFlatInN_WhileCopyIsLinear`][gate]</sub>

---

## What can be exported

### What happens to a slice or a transpose?

**`ToMemoryView` refuses it.** A flat run of bytes cannot describe a strided window, and quietly
handing one over would tell Python a lie about your data. It throws instead:

```text
InvalidOperationException: the array is not C-contiguous; a flat memoryview would
misrepresent it. Materialize first: np.ascontiguousarray(nd) or nd.copy().
```

| Layout | `ToMemoryView` | `ToNumpy` |
|---|---|---|
| C-contiguous `(4,6)` | ✅ 192 bytes | ✅ strides `(48, 8)` |
| row slice `b["1:3"]` | ✅ 96 bytes | ✅ strides `(48, 8)` |
| column slice `b[":, ::2"]` | ❌ | ✅ strides `(48, 16)` |
| transpose `b.T` | ❌ | ✅ strides `(8, 48)` |
| reversed `b["::-1"]` | ❌ | ✅ strides `(-48, 8)` |
| Fortran order | ❌ | ✅ strides `(8, 32)` |
| broadcast `(3,) → (2,3)` | ❌ | ✅ strides `(0, 8)`, read-only |
| 0-d scalar | ✅ 8 bytes | ✅ `shape=()` |
| empty `(0,3)` | ✅ 0 bytes | ✅ `shape=(0,3)` |

Row slices pass because a contiguous window at an offset is still a contiguous window — the
memoryview covers exactly that window, not the parent buffer. The two ways out of a `❌` are
`np.ascontiguousarray(nd)`, which copies once into a layout the protocol can describe, or
`ToNumpy()`, which describes the strides exactly and stays zero-copy.

<sub>See here [`MemoryView_RefusesNonContiguousLayouts`][gate], [`MemoryView_AcceptsContiguousWindowsScalarsAndEmpty`][gate]</sub>

### Dtype mapping

Fourteen of NumSharp's fifteen dtypes cross. Both maps are public, so you never hard-code a string:

| NumSharp | `ToNumpyDtypeStr` | `ToBufferFormat` | numpy calls it |
|---|---|---|---|
| `Boolean` | `\|b1` | `?` | `bool` |
| `Byte` / `SByte` | `\|u1` / `\|i1` | `B` / `b` | `uint8` / `int8` |
| `Int16` / `UInt16` | `<i2` / `<u2` | `h` / `H` | `int16` / `uint16` |
| `Int32` / `UInt32` | `<i4` / `<u4` | `i` / `I` | `int32` / `uint32` |
| `Int64` / `UInt64` | `<i8` / `<u8` | `q` / `Q` | `int64` / `uint64` |
| `Half` / `Single` / `Double` | `<f2` / `<f4` / `<f8` | `e` / `f` / `d` | `float16` / `float32` / `float64` |
| `Complex` | `<c16` | `Zd` | `complex128` |
| `Char` | `<u2` | `H` | `uint16` — a C# `char` is a UTF-16 code unit; numpy has no char dtype |
| `Decimal` | *throws* | *throws* | — |

`Decimal` is sixteen bytes of non-IEEE decimal float. Nothing in numpy or PEP 3118 describes it, so
both maps refuse rather than invent something:

```text
NotSupportedException: decimal has no PEP 3118 format (16-byte, non-IEEE).
Convert first: nd.astype(NPTypeCode.Double).
```

<sub>See here [`DtypeMaps_CoverEveryDtypeExceptDecimal`][gate], [`Decimal_ThrowsWithConversionGuidance`][gate]</sub>

---

## Reading Python's memory

The protocol runs both ways. Anything on the Python side that exports a buffer becomes a NumSharp
array — and `AsNDArray()` shares it rather than copying:

```csharp
using (Py.GIL())
{
    using PyObject buf = scope.Eval("bytearray(b'abcd')");
    using NDArray v = buf.AsNDArray();        // Byte[4], writable, shares `bytearray`'s bytes

    v[0] = (NDArray)(byte)90;                 // ...writes into the bytearray
}
```

`AsNDArray()` shares; `ToNDArray()` copies. The naming follows numpy's own `asarray` / `array`
convention, and both accept **any** exporter, not just numpy arrays:

| Python source | Result |
|---|---|
| `np.arange(4, dtype='f4')` | view · `Single[4]` · writable |
| `bytearray(b'abcd')` | view · `Byte[4]` · writable |
| `array.array('i', [1,2,3])` | view · `Int32[3]` · writable |
| `(ctypes.c_double * 3)(...)` | view · `Double[3]` · writable |
| `memoryview(bytearray(...))[::2]` | view · `Byte[4]` · writable |
| `io.BytesIO(...).getbuffer()` | view · `Byte[8]` · writable |
| `np.arange(6).reshape(2,3).T` | view · `Int64[3,2]` · writable |
| `np.broadcast_to(np.arange(3), (2,3))` | view · `Int64[2,3]` · **read-only** |
| `np.array([1+2j], dtype='c8')` | **copy** — complex64 widens to `Complex` |
| `np.array(['a','b'], dtype='U1')` | **copy** — UCS-4 text narrows to `Char` |
| `np.arange(4, dtype='>i4')` | **copy** — big-endian byte-reversed to native |

Strided sources view too, including transposes and negative strides: NumSharp reconstructs the
layout rather than flattening it. Only three things stop a zero-copy *view*, and each is a width or
byte-order mismatch that no reinterpretation can fix — complex64 is two 4-byte floats where
NumSharp's `Complex` is two 8-byte doubles, `<U1` is a 4-byte code point where `Char` is a 2-byte
UTF-16 unit, and big-endian data would byte-swap every value if read natively. But the **copy**
path (`ToNDArray`) handles all three: complex64 widens, UCS-4 narrows, and big-endian byte-reverses
each element to a value-correct native array. Only the *view* path still refuses big-endian — a
native-endian shared view is impossible — with the fix in the message:

```text
NotSupportedException: big-endian buffer format '>l' cannot be mapped onto a native-endian
NumSharp buffer. Byte-swap first: arr.astype(arr.dtype.newbyteorder('<')).
```

<sub>See here [`Import_ViewsEveryExporterVariety`][gate], [`Import_WidensComplex64AndNarrowsUcs4OnCopy`][gate], [`Import_ViewRefusesBigEndian_CopyByteSwaps`][gate]</sub>

### Can I view a read-only object?

**Yes, if you ask — and the view comes back non-writeable.** By default a read-only source is
refused outright, so you cannot accidentally take a view you have no right to write through:

```text
InvalidOperationException: the exporter's buffer is read-only; writing through a NumSharp view
would corrupt an immutable Python object. Use ToNDArray (copy), or pass allowReadonly:true to
take a NON-WRITEABLE view (guarded writes through it throw).
```

Pass `allowReadonly: true` and you get a view whose `Shape.IsWriteable` is `false` — numpy's own
`writeable=False`, carried across the boundary. Reads are free; a write raises rather than
corrupting a `bytes` object:

```csharp
using NDArray ro = pyBytes.AsNDArray(allowReadonly: true);

bool writable = ro.Shape.IsWriteable;   // false
ro[0] = (NDArray)(byte)7;               // NumSharpException: assignment destination is read-only
```

<sub>See here [`ReadonlySource_RefusedByDefault_ViewableOnOptIn`][gate], [`ReadonlyView_ThrowsOnGuardedWrite`][gate]</sub>

---

## Lifetime: who frees what

Two collectors, one buffer. The rule in both directions is the same: **memory is released when the
last reference on the far side lets go**, never when the near side happens to tidy up first.

### Can Python outlive my `NDArray`?

**Yes.** Exporting takes an independent reference on the NumSharp buffer, so the memory survives
even when every C# reference to it — the `NDArray`, the `PyObject` wrapper, all of them — is
disposed or collected. Python's own garbage collector decides when it ends:

```csharp
void ExportAndAbandon()
{
    var nd = np.arange(4).astype(NPTypeCode.Double);
    using (Py.GIL())
    {
        using PyObject mv = nd.ToMemoryView();
        scope.Set("mv", mv);
        scope.Exec("a = np.frombuffer(mv, '<f8')");
    }
    nd.Dispose();                       // the only C# reference, gone
}

ExportAndAbandon();
GC.Collect(); GC.WaitForPendingFinalizers();

// Python still reads — and writes — valid memory:
//   a           ->  [0. 1. 2. 3.]
//   a[0] = 7.0  ->  [7. 1. 2. 3.]
```

The release is a `weakref.finalize` registered on the buffer object every derived numpy view chains
back to, so `a[1:]`, `a.reshape(2,2)` and `np.asarray(a)` all extend the buffer's life. It fires
when the last of them dies. Delete them on the Python side and the NumSharp memory is freed.

### Can my `NDArray` outlive Python's object?

**Yes — that is what the lease is for.** An imported view holds the Python exporter alive through
NumSharp's own reference counting, and the lease is released when the last NumSharp view over the
memory is disposed or collected. That includes views *derived* from it: disposing the array you
imported while a slice of it lives frees nothing. The reference count decides, not the order you
disposed things in.

Two counters make the live state observable, which is what makes leak tests possible:

```csharp
int pinned = NDArrayPythonInterop.LiveExports;   // NumSharp buffers held by Python
int leased = NDArrayPythonInterop.LiveImports;   // Python buffers held by NumSharp
```

<sub>See here [`Export_SurvivesEveryManagedReferenceDying`][gate], [`Import_LeaseOutlivesTheOriginalView`][gate], [`LiveCounters_TrackBothDirections`][gate]</sub>

### What does sharing cost the Python object?

**It gets pinned: nothing may reallocate it while your view is alive.** That is CPython enforcing
the protocol, not NumSharp being cautious — a `bytearray` that moved its storage would leave your
pointer dangling:

```text
BufferError: Existing exports of data: object cannot be re-sized
```

`bytearray.append`, `numpy`'s `resize(refcheck=True)`, and anything else that would move the bytes
raise until the lease is released. `Dispose()` your view and the object is free again. This is the
strongest argument for `ToNDArray()` when you only wanted a snapshot: a copy touches the source
once and never again.

The mirror image applies to exports: while Python holds a view of a NumSharp array,
`nd.resize(...)` refuses to reallocate, exactly as NumPy protects its own exported buffers.

<sub>See here [`LiveView_LocksTheSourceAgainstResize`][gate], [`DisposingTheView_ReleasesTheLock`][gate]</sub>

---

## The GIL

Every conversion verb **acquires the GIL itself**, re-entrantly, so nesting inside your own
`Py.GIL()` block is fine and calling from a thread that has never touched Python is fine.

What you still own is your `PyObject`s: create and dispose them inside a `Py.GIL()` scope, because
pythonnet's final decref needs it.

For a hot loop you can lift the per-call acquisition:

```csharp
using (Py.GIL())                                                  // one acquisition...
    foreach (var batch in batches)
        using (PyObject mv = batch.ToMemoryView(requireGIL: false))   // ...N conversions inside
            consumer.Invoke(mv);
```

`requireGIL: false` means **you** hold the GIL. Converting without actually holding it is an
immediate access violation, the same as any raw C-API misuse. The process-wide default is
`NDArrayPythonInterop.RequireGIL`, and `null` (the parameter's default) follows it.

> **The trap.** A .NET method or delegate invoked *from* Python does **not** hold the GIL —
> pythonnet's binder releases it around managed bodies. Inside a Python → .NET callback, leave GIL
> management on.

Import views are tied to the interpreter that owns their memory: after `PythonEngine.Shutdown()`
they must not be touched, though disposing them stays safe.

<sub>See here [`Verbs_AcquireTheGilThemselves`][gate], [`RequireGilFalse_WorksUnderAnOuterAcquisition`][gate]</sub>

---

## Seven libraries, verbatim

Every row below was executed on the stack in the banner. Where a library wants a numpy array,
`ToNumpy()` is the shorter road; where it wants bytes, the memoryview goes in directly.

| Library | Entry point | Verified |
|---|---|---|
| **PyTorch** 2.12.1 | `torch.frombuffer(mv, dtype=torch.float32)` | same `data_ptr` as `np.frombuffer`; a tensor write lands in the `NDArray` |
| **Pillow** 11.3.0 | `Image.frombuffer('RGB', (w,h), mv, 'raw', 'RGB', 0, 1)` | `getpixel` reads NumSharp's bytes |
| **PyArrow** 23.0.1 | `pa.py_buffer(mv)` → `pa.Array.from_buffers` | `to_numpy(zero_copy_only=True)` returns our address |
| **pandas** 2.3.3 | `pd.DataFrame(nd.ToNumpy(), copy=False)` | `np.shares_memory(df.to_numpy(), x)` is `True` |
| **polars** 1.38.1 | `pl.Series('v', nd.ToNumpy())` | `to_numpy(allow_copy=False)` shares |
| **OpenCV** 4.13.0 | `cv2.circle(nd.ToNumpy(), ...)` | draws **into** the NumSharp buffer |
| **stdlib** | `struct` · `hashlib` · `zlib` · `memoryview.cast` | all read the memoryview directly |

### PyTorch

```csharp
var nd = np.arange(6).astype(NPTypeCode.Single);

using (Py.GIL())
{
    scope.Exec("import torch");
    using PyObject mv = nd.ToMemoryView();
    scope.Set("mv", mv);

    scope.Exec("t = torch.frombuffer(mv, dtype=torch.float32)");
    scope.Exec("t[0] = 42.0");
}

Console.WriteLine(nd);      // [42. 1. 2. 3. 4. 5.]
```

`torch.from_numpy(np.frombuffer(mv, '<f4'))` reaches the identical address — both report the same
`data_ptr`. Results come home the same way: `resultTensor.numpy()` is a numpy array, and
`AsNDArray()` leases it.

### Pillow

```csharp
var img = np.zeros(new Shape(4, 6, 3), NPTypeCode.Byte);
img[2, 3] = (NDArray)new byte[] { 10, 20, 30 };

using (Py.GIL())
{
    scope.Exec("from PIL import Image");
    using PyObject mv = img.ToMemoryView();
    scope.Set("mv", mv);
    scope.Exec("im = Image.frombuffer('RGB', (6, 4), mv, 'raw', 'RGB', 0, 1)");
    // im.getpixel((3, 2)) == (10, 20, 30)
}
```

> **Coming back from Pillow needs a step.** A PIL `Image` exports no PEP 3118 buffer, and its
> `__array_interface__["data"]` is a `bytes` object rather than the `(pointer, readonly)` tuple a
> view would need an address from. Call `np.asarray(im)` first; the resulting array imports as a
> read-only view.

### PyArrow

```csharp
using (Py.GIL())
{
    scope.Exec("import pyarrow as pa");
    using PyObject mv = nd.ToMemoryView();
    scope.Set("mv", mv);
    scope.Exec("buf = pa.py_buffer(mv)");
    scope.Exec("arr = pa.Array.from_buffers(pa.float64(), 5, [None, buf])");
    // arr.to_numpy(zero_copy_only=True).ctypes.data == buf.address
}
```

### OpenCV

```csharp
var frame = np.zeros(new Shape(480, 640, 3), NPTypeCode.Byte);

using (Py.GIL())
{
    scope.Exec("import cv2");
    using PyObject im = frame.ToNumpy();        // cv2 wants an ndarray, not bytes
    scope.Set("im", im);
    scope.Exec("cv2.circle(im, (320, 240), 40, (255, 0, 0), -1)");
}
// frame now holds the circle — cv2 rendered straight into NumSharp memory.
```

### No numpy required

Nothing above obliges the consumer to know what a numpy array is:

```csharp
var pcm = np.arange(8).astype(NPTypeCode.Int16);

using (Py.GIL())
{
    scope.Exec("import struct, hashlib, zlib");
    using PyObject mv = pcm.ToMemoryView();
    scope.Set("mv", mv);

    scope.Exec("first4 = struct.unpack_from('<4h', mv)");   // (0, 1, 2, 3)
    scope.Exec("digest = hashlib.sha256(mv).hexdigest()");  // hashes in place
    scope.Exec("packed = zlib.compress(mv)");               // compresses in place
    scope.Exec("mv.cast('h')[0] = 77");                     // and writes back
}

Console.WriteLine(pcm);     // [77 1 2 3 4 5 6 7]
```

`memoryview.cast` re-types the window without copying, which is how a raw-byte export becomes typed
elements for a consumer that wants them. `bytes(mv)`, by contrast, **is** a copy — reach for it when
you want one.

<sub>See here [`Torch_SharesTheDataPointer`][gate] †, [`Pillow_ReadsTheExportedBuffer`][gate] †, [`Pillow_ImageItselfIsNotImportable`][gate] †, [`PyArrow_WrapsTheMemoryViewZeroCopy`][gate] †, [`Pandas_DataFrameSharesMemory`][gate] †, [`Polars_SeriesSharesMemory`][gate] †, [`OpenCv_DrawsIntoNumSharpMemory`][gate] †, [`Stdlib_ConsumersReadAndWriteTheMemoryView`][gate]</sub>

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `the array is not C-contiguous; a flat memoryview would misrepresent it` | You exported a slice, transpose, F-order or broadcast view. `np.ascontiguousarray(nd)` to materialize, or `ToNumpy()` to keep the strides and stay zero-copy |
| `ValueError: buffer size must be a multiple of element size` | The dtype you passed `np.frombuffer` does not divide the byte count. Use `ToNumpyDtypeStr(nd.typecode)` rather than a literal |
| `decimal has no PEP 3118 format (16-byte, non-IEEE)` | No dtype in numpy or PEP 3118 describes it. `nd.astype(NPTypeCode.Double)` first |
| `the exporter's buffer is read-only` | You asked for a writable view of `bytes` or a `writeable=False` array. Pass `allowReadonly: true`, or `ToNDArray()` to copy |
| `assignment destination is read-only` | You wrote through a non-writeable view — a read-only or broadcast source. Copy it first if you need to mutate |
| `BufferError: Existing exports of data: object cannot be re-sized` | A live NumSharp view leases that object. `Dispose()` the view to release the lock |
| `big-endian buffer format '>l' cannot be mapped` | Only the zero-copy *view* refuses big-endian. `ToNDArray()` copies it, byte-reversing each value automatically; or byte-swap on the Python side first: `arr.astype(arr.dtype.newbyteorder('<'))` |
| `the object does not export a PEP 3118 buffer` | Not every object is an exporter — a `dict`, a PIL `Image`. `np.asarray(obj)` first if numpy understands it |
| `Python engine is not initialized` | `Runtime.PythonDLL` and `PythonEngine.Initialize()` before any conversion |
| A Python write did not show up in NumSharp | You copied. `ToMemoryView`/`ToNumpy`/`AsNDArray` share; `ToNumpyCopy`/`ToNDArray` do not |
| Access violation around a conversion | `requireGIL: false` on a thread that does not hold the GIL — including inside a Python → .NET callback, where pythonnet releases it |

---

## Claims ledger

| # | Claim | Evidence | Gate |
|---|---|---|---|
| 1 | `np.frombuffer` over an exported memoryview shares NumSharp's memory | `a[0,0] = -5.0` → `nd[0,0] == -5.0` | [`Recipe_RoundTripsThroughFrombuffer`][gate] |
| 2 | `ToNumpyDtypeStr` covers every numpy-expressible dtype | 14 dtypes map; `Decimal` throws | [`DtypeStrings_AreTotalOverNumpyExpressibleDtypes`][gate] |
| 3 | The buffer route is flat in *n*; copying is linear | 0.0073 → 0.0070 ms vs 0.0031 → 12.82 ms across 10⁴× | [`BufferRoute_IsFlatInN_WhileCopyIsLinear`][gate] |
| 4 | `ToMemoryView` refuses non-contiguous layouts | `InvalidOperationException`, verbatim text | [`MemoryView_RefusesNonContiguousLayouts`][gate] |
| 5 | Contiguous windows, scalars and empties export | 192 / 96 / 8 / 0 bytes | [`MemoryView_AcceptsContiguousWindowsScalarsAndEmpty`][gate] |
| 6 | `Decimal` is refused with conversion guidance | verbatim `NotSupportedException` | [`Decimal_ThrowsWithConversionGuidance`][gate] |
| 7 | Every buffer-exporter variety imports as a view | 8 of 12 probes view, incl. strided and transposed | [`Import_ViewsEveryExporterVariety`][gate] |
| 8 | complex64 widens, UCS-4 narrows — on copy, never as a view | `c8` → `Complex`, `<U1` → `Char` | [`Import_WidensComplex64AndNarrowsUcs4OnCopy`][gate] |
| 9 | Big-endian: the view refuses, the copy byte-reverses to native | view throws; copy round-trips value-exact | [`Import_ViewRefusesBigEndian_CopyByteSwaps`][gate] |
| 10 | Read-only sources are refused by default, viewable on opt-in | throws; then `IsWriteable == false` | [`ReadonlySource_RefusedByDefault_ViewableOnOptIn`][gate] |
| 11 | Writing a read-only view throws instead of corrupting | `assignment destination is read-only` | [`ReadonlyView_ThrowsOnGuardedWrite`][gate] |
| 12 | An export outlives every managed reference to it | `Dispose()` + GC, then Python reads and writes | [`Export_SurvivesEveryManagedReferenceDying`][gate] |
| 13 | An import lease outlives the Python object's own reference | derived view still valid after the source is deleted | [`Import_LeaseOutlivesTheOriginalView`][gate] |
| 14 | A live view locks the source against reallocation | `BufferError` on `bytearray.append`; gone after `Dispose()` | [`LiveView_LocksTheSourceAgainstResize`][gate] |
| 15 | Verbs acquire the GIL themselves, re-entrantly | conversion succeeds with and without an outer `Py.GIL()` | [`Verbs_AcquireTheGilThemselves`][gate] |
| 16 | PyTorch reaches the identical address | `t.data_ptr() == a.ctypes.data` | [`Torch_SharesTheDataPointer`][gate] † |
| 17 | Pillow reads the exported buffer | `getpixel((3,2)) == (10,20,30)` | [`Pillow_ReadsTheExportedBuffer`][gate] † |
| 18 | A PIL `Image` itself cannot be imported | `__array_interface__["data"]` is `bytes` | [`Pillow_ImageItselfIsNotImportable`][gate] † |
| 19 | PyArrow wraps the memoryview zero-copy | `to_numpy(...).ctypes.data == buf.address` | [`PyArrow_WrapsTheMemoryViewZeroCopy`][gate] † |
| 20 | pandas and polars share the exported array | `np.shares_memory(...) == True` | [`Pandas_DataFrameSharesMemory`][gate] †, [`Polars_SeriesSharesMemory`][gate] † |
| 21 | OpenCV draws into NumSharp memory | pixel written by `cv2.circle` read back in C# | [`OpenCv_DrawsIntoNumSharpMemory`][gate] † |
| 22 | Non-numpy consumers read and write the memoryview | `struct` · `hashlib` · `zlib` · `cast('h')` | [`Stdlib_ConsumersReadAndWriteTheMemoryView`][gate] |

† Self-skipping: reports Inconclusive when the Python package is absent, so the suite stays green
on an image without it.

---

## See also

- [Python & numpy (pythonnet)](pythonnet-numpy.md) — the full bridge: every layout exported with its
  strides intact, the auto-marshaling codec, and the engine lifecycle
- [Numpy.NET](numpy-net.md) — sharing these same buffers with SciSharp's `Numpy` / `Numpy.Bare`
- [Interoperability](index.md) — the contract underneath every NumSharp bridge

[gate]: https://github.com/SciSharp/NumSharp/blob/master/test/NumSharp.Tests.Interop/DocExamples.NpFrombufferPage.cs
