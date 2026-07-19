# The Zero-Copy Model — how a conversion decides *view* or *copy*

Every conversion across the Python boundary answers one question: **can both sides share the same
bytes, or must the data be duplicated?** This page explains how
[NumSharp.Interop.pythonnet](pythonnet.md) answers it — the decision tree, the three routes that
produce a view, the handful of layouts that genuinely cannot be shared, and the trade-off you accept
when you do share.

Everything here is measured, not asserted: the coverage table at the bottom comes from a census of 48
exporter varieties run against live CPython, and each behaviour maps to a test in
[`test/NumSharp.Interop.UnitTests`](https://github.com/SciSharp/NumSharp/tree/master/test/NumSharp.Interop.UnitTests).

---

## The two semantics

|  | View (share) | Copy (snapshot) |
|---|---|---|
| Memory | one buffer, both sides | two buffers |
| Later writes on either side | visible to the other | invisible |
| Cost | O(1) — a pointer and a lease | O(n) — a full blit |
| Effect on the Python object | holds a `Py_buffer` lock (blocks resize) | none |
| Works for | representable dtypes/layouts | **everything** |
| Survives `PythonEngine.Shutdown()` | no — the memory was the interpreter's | yes — it owns its bytes |

Neither is "better". A view is the faithful bridge — it *is* the same array. A copy is the total
one — it always succeeds and never touches the source. `Auto` picks the view whenever the layout
permits, so you get sharing where sharing is possible and correctness everywhere else.

---

## `Auto` — view-first, copy only when impossible

`NumpyCodecMode.Auto` is the default in both directions. The decode side is literally two attempts:

```csharp
NDArray nd = _options.DecodeMode switch
{
    NumpyCodecMode.Copy => TryDecodeCopy(pyObj),
    NumpyCodecMode.View => TryDecodeView(pyObj),
    _                    => TryDecodeView(pyObj) ?? TryDecodeCopy(pyObj),   // Auto
};
```

`TryDecodeView` returns `null` when no view is constructible — that `??` **is** the fallback. The
common case (a contiguous numpy array) succeeds on the first attempt, so the fallback costs nothing
on the hot path.

```csharp
NDArrayPythonInterop.RegisterCodec();          // Auto both ways; once per engine session

using (Py.GIL())
{
    scope.Exec("src = np.arange(6, dtype='f8')");
    using PyObject p = scope.Get("src");

    NDArray nd = p.As<NDArray>();              // Auto -> zero-copy VIEW
    nd[0] = (NDArray)42.0;                     // ...so this reaches Python
    // scope: src[0] == 42.0
}
```

On **encode** (`NDArray` → Python) a view and a copy have identical dtype coverage — both need a
numpy-expressible dtype — so `Auto` always yields a view, and only `Copy` forces a detached array.
`Decimal` has no numpy dtype at all and falls through to pythonnet's CLR-object wrapping rather than
failing the conversion.

### The three modes

```csharp
// default — share when possible, copy when not
NDArrayPythonInterop.RegisterCodec();

// never share: a detached snapshot that never locks the Python object
NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.Copy });

// always share, or fail loudly — for code that depends on shared memory
NDArrayPythonInterop.RegisterCodec(new NumpyCodecOptions { DecodeMode = NumpyCodecMode.View });
```

| Mode | When a view is possible | When it is not |
|---|---|---|
| `Auto` *(default)* | zero-copy view | independent copy |
| `View` | zero-copy view | **declines** the conversion (no silent copy) |
| `Copy` | independent copy | independent copy |

Use `View` when a silent copy would be a *bug* — e.g. you hand an array to Python expecting it to
fill your buffer in place. A copy there fails silently; `View` fails loudly.

---

## The three routes that produce a view

`ToNDArrayView` tries them in order. All metadata is read through Python's own `memoryview`, never
pythonnet's `PyBuffer` flags (broken for shape/strides/format across 3.0.x).

### Route 1 — C-contiguous exporters (any object)

The overwhelmingly common case: a contiguous numpy array, `bytes`, `bytearray`, `array.array`, a
`memoryview`, a `ctypes` array, a `BytesIO` buffer. A `Py_buffer` lease is taken and the pointer
becomes the NDArray's base address.

```csharp
PyExec("ba = bytearray(b'abcd')");
var v = py.AsNDArray();            // Byte[4], writable, shares bytes with `ba`
```

### Route 2 — non-contiguous **numpy** arrays, via `__array_interface__`

Slices, transposes, Fortran order, negative strides and broadcasts. numpy publishes the exact layout,
which is rebuilt as a NumSharp strided `Shape`:

```csharp
PyExec("base = np.arange(20, dtype='i8')");

var even = ViewOf("base[::2]");    // stride 2, shape (10,)
var rev  = ViewOf("base[::-1]");   // NEGATIVE stride
var tr   = ViewOf("np.arange(6).reshape(2,3).T");   // F-order strided
```

Negative strides need the window normalised, because the exporter's pointer addresses *element 0*
and other elements live **below** it:

```csharp
long minOffset = 0, maxOffset = 0;
for (int i = 0; i < dims.Length; i++) {
    long extent = (dims[i] - 1) * elemStrides[i];
    if (extent < 0) minOffset += extent; else maxOffset += extent;
}
spanElements = maxOffset - minOffset + 1;
basePtr      = dataPtr + minOffset * itemsize;
shape        = new Shape(dims, elemStrides, offset: -minOffset, bufferSize: spanElements);
```

### Route 3 — non-contiguous **non-numpy** exporters

A sliced, offset or reversed `memoryview` — anything with strides but no `__array_interface__`. The
base pointer comes from a `PyBUF.STRIDED` request and the exact shape/strides from the memoryview
itself, so these are true views rather than copies:

```csharp
PyExec("ba = bytearray(range(16))");

var even = ViewOf("memoryview(ba)[::2]");    // [0,2,4,...] — shared
var odd  = ViewOf("memoryview(ba)[1::2]");   // offset included in the pointer
var rev  = ViewOf("memoryview(ba)[::-1]");   // negative stride

even[3] = (NDArray)(byte)99;                 // logical 3 -> byte 6
// python: ba[6] == 99
```

---

## What cannot be viewed (and therefore copies)

Exactly three things. Each throws inside the view attempt, which in `Auto` is the signal to copy.

### complex64

numpy's `complex64` is two 4-byte floats (8 bytes); NumSharp's `Complex` is two 8-byte doubles
(16 bytes). There is no reinterpretation — the values must be widened, and widening is a copy.

```csharp
PyExec("c = np.array([1+2j, 3+4j], dtype='c8')");
NDArray nd = ImportOf("c");        // Complex (complex128), values preserved, independent
```

Viewing it as `float32[..., 2]` *would* be zero-copy, but it would silently hand back a different
dtype and shape than the caller asked for — so the copy is the correct answer, not a limitation.

### Big-endian multi-byte data

NumSharp buffers are native-endian. Viewing `>i4` on a little-endian machine would byte-swap every
value. Rejected rather than silently misread — byte-swap on the Python side first:

```python
arr.astype(arr.dtype.newbyteorder('<'))
```

Single-byte big-endian dtypes (`>i1`, `|u1`, `|b1`) **do** view — byte order is meaningless at one
byte wide.

### Sub-item strides

A stride that is not a whole multiple of the element size, so consecutive elements *overlap*:

```python
a = np.arange(4, dtype='i4')                                    # itemsize 4, normal stride 4
w = np.lib.stride_tricks.as_strided(a, shape=(2,), strides=(2,))  # stride 2 BYTES
# w -> [0, 65536]
#   elem[0] = bytes[0:4] = 00 00 00 00 = 0
#   elem[1] = bytes[2:6] = 00 00 01 00 = 65536   <- overlaps elem[0]
```

numpy strides are in **bytes**; NumSharp strides are in **elements**. `2 / 4 = 0.5` elements is not
expressible, so the guard is one line:

```csharp
if (byteStrides[i] % itemsize != 0)
    throw new NotSupportedException(
        $"stride {byteStrides[i]} bytes is not a multiple of itemsize {itemsize}; " +
        "NumSharp strides are element-based. Use ToNDArray (copy).");
```

In practice this is unreachable from ordinary code — normal slicing always yields element-multiple
strides (`a[::2]` → 8, `a[1::3]` → 12, `a[::-1]` → −4). You essentially only get one from
`as_strided`, numpy's explicitly-unsafe power tool.

---

## Read-only sources still view

A read-only exporter (`bytes`, a numpy array with `writeable=False`, a broadcast view) does not force
a copy — it produces a **non-writeable view**, mirroring numpy's own `writeable=False`:

```csharp
var ro = py.AsNDArray(allowReadonly: true);
ro.Shape.IsWriteable;                 // false
ro[0] = (NDArray)(byte)5;             // throws: assignment destination is read-only
```

By default (`allowReadonly: false`) the *verb* refuses such sources outright, so you cannot
accidentally take a view you are not allowed to write through. The codec passes `allowReadonly: true`,
because a non-writeable view is still a view and still beats copying. A useful consequence: since a
copy always owns **writable** memory, `IsWriteable == false` is a reliable signal that you got the
view path.

---

## The trade-off: a live view locks the source

This is the cost of sharing, and the main reason `Copy` mode exists. While a view is alive it holds a
`Py_buffer` lease, and CPython forbids anything that would reallocate the exporter:

```csharp
PyExec("ba = bytearray(b'abcd')");
var v = py.AsNDArray();

PyExec("ba.append(1)");    // BufferError: Existing exports of data: object cannot be re-sized

v.Dispose();               // release the lease...
PyExec("ba.append(1)");    // ...now fine
```

numpy is guarded the same way — `arr.resize(refcheck=True)` refuses while the lease exists.

The lease is released when the **last** NumSharp view over the memory — including derived slices like
`nd["2:"]` — is disposed or garbage-collected. That means **`Dispose()` is what makes the release
deterministic**; leaving it to the GC leaves Python's object locked for an unbounded window. If you
want a snapshot that never touches the Python object at all, use `DecodeMode = Copy` or `ToNDArray`.

---

## Measured coverage

A census of 48 exporter varieties through the view path (CPython 3.12, numpy 2.4.2, pythonnet 3.0.5):

| Category | Result |
|---|---|
| `bytes`, `bytearray`, `memoryview`, `BytesIO.getbuffer()` | **view** |
| `array.array` — all 12 typecodes (`b B h H i I l L q Q f d`) | **view** |
| `ctypes` arrays (`c_int`, `c_double`, `c_ubyte`, `c_int16`, ...) | **view** |
| numpy dtypes: `i1 u1 i2 u2 i4 u4 i8 u8 f2 f4 f8 c16`, big-endian `i1` | **view** |
| numpy layouts: contiguous, strided, reversed, transposed, F-order, broadcast, read-only, 0-d | **view** |
| `memoryview` casts and strided/reversed forms | **view** |
| numpy `complex64` | copy (widened) |
| sub-item stride (`as_strided`) | copy (linearised) |
| big-endian multi-byte | rejected by both paths — byte-swap first |

**45 view, 2 copy.** Both copies are the genuine floor described above, not gaps.

> **Why acquisition goes through the `memoryview`.** The lease is always taken from the exporter's
> `memoryview`, never the raw object. pythonnet 3.0.x's `obj.GetBuffer` is per-exporter buggy — on a
> raw `ctypes` array it hard-crashes the process for *every* flag, which used to take down both the
> view and the copy path. The memoryview over the very same memory leases cleanly, and it is
> retained by the `Py_buffer`, so the source stays pinned and resize-locks still hold.

---

## Controlling the GIL

Every verb manages the GIL itself by default (re-entrantly, so nesting under your own `Py.GIL()` is
fine). For hot loops under one outer acquisition you can switch that off per call or process-wide:

```csharp
using (Py.GIL())                                     // ONE acquisition...
    foreach (var batch in batches)
        using (PyObject p = batch.ToNumpy(requireGIL: false))   // ...N conversions inside
            consumer.Invoke(p);

NDArrayPythonInterop.RequireGIL = false;             // or process-wide, when EVERY call site holds it
```

`requireGIL: false` means **the caller owns the GIL** — the calling thread must already hold it.
Converting without it is undefined behaviour (an immediate access violation, exactly like any raw
C-API misuse).

> **The trap:** a .NET method or delegate invoked *from* Python does **not** hold the GIL —
> pythonnet's binder releases it around managed bodies. Keep GIL management **on** inside
> Python → .NET callbacks. Only pythonnet's argument/return marshaling (where the codec runs)
> executes under the GIL.

The interop's background machinery (deferred lease disposal, the shutdown drain) always manages the
GIL itself regardless of this setting — it runs on threads that cannot inherit a caller's GIL.

---

## Choosing, in one table

| You want | Use |
|---|---|
| Best of both, no thought required | `Auto` (default) |
| Python to fill a buffer you own | `View` — a silent copy would be a bug |
| A detached snapshot; never lock or touch the Python object | `Copy` / `ToNDArray` |
| A view, deterministically released | `AsNDArray` + `Dispose()` |
| Data that must outlive `PythonEngine.Shutdown()` | `Copy` — a view's memory dies with the interpreter |

---

## See also

- [Pythonnet — NumSharp ⇄ Python](pythonnet.md) — the package, the verbs, lifetime & engine rules
- [Numpy.NET — coexistence & migration](numpy-net.md) — sharing buffers with SciSharp's `Numpy` packages
- [Interoperability](index.md) — the contract every NumSharp bridge builds on
