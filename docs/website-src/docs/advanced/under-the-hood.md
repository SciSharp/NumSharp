# Under the hood — internals

This is a low-level look at how an `NDArray` is built, for people extending NumSharp or reasoning about its performance. It mirrors NumPy's [*Internal organization*](https://numpy.org/doc/stable/dev/underthehood.html) chapter: an array is a **raw data buffer** plus **metadata that says how to read it**, and almost everything cheap NumSharp does — slicing, transpose, reshape, broadcast — is a change to the metadata with the buffer left untouched.

For the gentle version of this, read [Introduction](../intro.md) first; for the memory-lifetime mechanics, [Buffering & Memory](../buffering.md).

---

## The two components

An `NDArray` is three parts, but two carry the data model:

```
NDArray
├── Storage   UnmanagedStorage  — the raw data buffer (unmanaged memory)
├── Shape     readonly struct   — the metadata (how to interpret the buffer)
└── TensorEngine                — the compute backend (DefaultEngine = pure C#)
```

The **data buffer** is what you'd think of as an array in C: one contiguous, fixed block of unmanaged memory holding fixed-size elements. Everything else is the **metadata** that describes how to read that block — and, exactly as in NumPy, it contains:

1. the element size in bytes (`itemsize`);
2. the start of the data within the buffer (`offset`);
3. the number of dimensions and each dimension's size (`shape`);
4. the separation between elements along each dimension (`strides`);
5. whether the buffer is writeable;
6. the dtype (how to interpret each element);
7. whether the array is C-order or Fortran-order contiguous.

Byte order is item (5)'s NumPy sibling, but NumSharp is **always host-endian** — big-endian data is byte-swapped to native on read (e.g. from a `.npy`), so there is no non-native byte-order state to carry.

---

## `UnmanagedStorage` — the buffer

NumSharp stores element data in **unmanaged memory**, not a managed `T[]` (benchmarked fastest; it avoids the GC and pins nothing). The buffer is owned by an **atomically reference-counted** memory block:

- Multiple arrays can share one block — every **view** (slice, transpose, reshape) holds the *same* block, so disposing one array frees nothing while another view lives.
- Release is deterministic when you `Dispose`, and the GC finalizer is the safety net: `~NDArray` **abandons** its reference (decrements the count) rather than freeing directly — `Release` is the eager path, `Abandon` the finalizer path.
- This refcount is what powers the resize guard and the interop lifetime contract — see [Interoperability → the contract](../interop/index.md#the-contract) and [Buffering & Memory](../buffering.md).

---

## `Shape` — the metadata

`Shape` is a `readonly struct` (immutable after construction, NumPy-aligned) that carries the metadata and precomputes what it can:

```csharp
public readonly partial struct Shape
{
    internal readonly int    _flags;      // cached ArrayFlags bitmask (O(1) property reads)
    internal readonly int    _hashCode;   // precomputed
    internal readonly long   size;        // total element count
    internal readonly long[] dimensions;  // dimension sizes
    internal readonly long[] strides;     // stride values in ELEMENTS (0 = broadcast axis)
    internal readonly long   bufferSize;  // size of the underlying buffer
    internal readonly long   offset;      // base offset into storage
}
```

### `ArrayFlags`

Cached at construction, mirroring NumPy's `ndarraytypes.h`:

| Flag | Value | Meaning |
|------|-------|---------|
| `C_CONTIGUOUS` | `0x0001` | row-major contiguous |
| `F_CONTIGUOUS` | `0x0002` | column-major contiguous |
| `OWNDATA` | `0x0004` | array owns its buffer |
| `ALIGNED` | `0x0100` | always true for managed allocations |
| `WRITEABLE` | `0x0400` | false for broadcast views |
| `BROADCASTED` | `0x1000` | has a stride-0 axis with dim > 1 |

So `Shape.IsContiguous`, `IsFContiguous`, `IsBroadcasted`, `IsWriteable`, `IsSliced`, `IsSimpleSlice` are all **O(1)** flag reads, not walks.

### Strides: elements internally, bytes publicly ⚠️

This is the one internals detail that bites: the **internal `Shape.strides` are in ELEMENTS**, but the **public `ndarray.strides` property is in BYTES** — matching NumPy's `ndarray.strides`:

```csharp
var a = np.arange(24).reshape(4, 6).astype(np.float64);
a.strides;          // [48, 8]   ← BYTES  (public, NumPy-parity)
a.Shape.strides;    // [6, 1]    ← ELEMENTS (internal)
```

When you write a backend or a kernel and read `a.Shape.Strides`, they are element counts — multiply by `itemsize` to get bytes (NumPy's C-API gives you bytes directly). Getting this wrong is the classic off-by-`itemsize` bug.

---

## Views: metadata-only reinterpretation

Because the buffer and the metadata are separate, a new "way of looking at" the same bytes is just a new `Shape` (and a shared `Storage`). Slicing, transpose, reshape (where possible), and broadcast all do exactly this — **no data moves**:

```csharp
var a = np.arange(12).reshape(3, 4);
var t = a.T;               // transpose: strides reversed, same buffer
var s = a["1:3, ::2"];     // slice: new strides + offset, same buffer
```

- **Transpose** reverses the stride order; the data doesn't move, only the index→address mapping changes.
- **Slicing** sets a new `offset` and strides into the same buffer.
- **Broadcast** sets a stride of **0** on a stretched axis — one stored element is read for every logical position along it — which is why broadcast views are marked `WRITEABLE = false` (writing would corrupt every aliasing position).

A view increments the buffer's reference count, so the buffer outlives the original array if any view remains. Forcing an independent buffer requires `.copy()`. This is the full story in [Copies and views](../fundamentals/copies-and-views.md).

---

## Multidimensional array indexing order

NumSharp uses **C-order (row-major)** indexing: the **last index varies fastest** in memory. `a[i, j]` selects row `i`, column `j`; `a[0]` is the first *row* (a contiguous sub-array), and iterating the first index steps by the largest stride.

The confusion NumPy documents is real and worth restating: matrix convention (first index = row) and image convention (first index = x/column) disagree, and each language's storage order (C row-major vs Fortran column-major) then decides which index is cheap to iterate. NumSharp resolves this the way NumPy does:

- The default is **C-order**, so `a[0]` (a row) is contiguous and cheap, while `a[:, 0]` (a column) is strided.
- `Shape` also tracks **F-contiguity**, and APIs with an `order` parameter resolve NumPy's `C`/`F`/`A`/`K` modes through `OrderResolver`, so column-major layouts are first-class where you ask for them.
- Kernels don't care about the *label* order — `NDIter` determines which axis is most rapidly varying in memory and makes that the inner loop, so a ufunc over a transposed view is as efficient as over a contiguous one.

The practical consequence: iterate with the memory order in mind. `a.flat` / the typed iterators walk in **logical C-order**; a fast per-chunk kernel should follow the array's actual strides (which `NDIter` and `np.nditer_chunks<T>` do for you). See [Indexing](../fundamentals/indexing.md) and [Iterating & Enumerating](../iterating-and-enumerating.md).

---

## How kernels traverse it

Two managed subsystems turn this metadata into computation:

- **`NDIter`** — the multi-operand iterator (NumSharp's `NpyIter`): C/F/A/K order, broadcasting, buffering, casting, external loops, reductions. It picks the inner-loop axis from the strides. See [NDIter](../NDIter.md).
- **`ILKernelGenerator` / `DirectILKernelGenerator`** — runtime IL emission with SIMD (V128/V256/V512), so the inner loops run at native speed with no C. See [IL Generation](../il-generation.md).

Everything the [Extending NumSharp](extending-numsharp.md) seams expose is built on these two over the buffer/metadata split described here.

---

## Inspecting the internals

From a `dotnet run` script (with the internals-visible signing directives) you can read the raw metadata:

```csharp
a.Shape.dimensions;   // long[] dimension sizes
a.Shape.strides;      // long[] strides in ELEMENTS (0 = broadcast)
a.Shape.offset;       // base offset into storage
a.Shape.bufferSize;   // underlying buffer element count
a.Shape.IsContiguous; // O(1) flag
a.Storage.IsView;     // is this a view?
a.@base;              // owning array, or null

// public (no internals access needed):
a.shape; a.strides;   // strides in BYTES (NumPy parity)
a.ndim; a.size; a.dtype;
```

---

## Troubleshooting

### "My strided kernel read the wrong elements"
`a.Shape.Strides` are in **elements**; `a.strides` are in **bytes**. Backends read element strides and scale by `itemsize` themselves.

### "Writing to a broadcast view threw"
Broadcast axes have stride 0 and are `WRITEABLE = false` by design. Copy first — see [Copies and views](../fundamentals/copies-and-views.md#broadcast-views-are-read-only).

### "The buffer wasn't freed when I disposed the array"
A view still holds the shared, refcounted block. The buffer frees on the *last* reference (or the GC finalizer as backstop). See [Buffering & Memory](../buffering.md).

---

## API reference

| Item | Meaning |
|------|---------|
| `NDArray.Storage` (`UnmanagedStorage`) | the unmanaged data buffer + refcounted block |
| `NDArray.Shape` (`readonly struct`) | dimensions, strides (elements), offset, flags, size |
| `NDArray.strides` | strides in **bytes** (public, NumPy parity) |
| `Shape.strides` | strides in **elements** (internal) |
| `Shape.{IsContiguous,IsFContiguous,IsBroadcasted,IsWriteable,IsSliced}` | O(1) `ArrayFlags` reads |
| `ArrayFlags` | `C_CONTIGUOUS`/`F_CONTIGUOUS`/`OWNDATA`/`ALIGNED`/`WRITEABLE`/`BROADCASTED` |
| `NDArray.@base` / `Storage.IsView` | view detection |

---

## Related reading

- [Introduction](../intro.md) — the same model, gently.
- [Buffering & Memory](../buffering.md) — the ARC block, slices, and lifetime.
- [Copies and views](../fundamentals/copies-and-views.md) — metadata-only reinterpretation in practice.
- [NDIter](../NDIter.md) · [IL Generation](../il-generation.md) — how kernels traverse and compile.
- [Extending NumSharp](extending-numsharp.md) · [Native code & backends](native-backends.md).
- [NumPy internals](https://numpy.org/doc/stable/dev/underthehood.html) — the upstream article this converts.
