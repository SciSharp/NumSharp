# Interoperability

NumSharp arrays are raw, unmanaged, densely-typed buffers — exactly the shape the rest of the scientific-computing world speaks. The interop packages turn that into practice: **zero-copy bridges** that let other runtimes and libraries read and write the same memory NumSharp computes over, with lifetimes coupled safely across the boundary.

## Official Bridges

| Package | Bridge | Transport |
|---------|--------|-----------|
| [NumSharp.Interop.pythonnet](pythonnet.md) | `NDArray` ⇄ Python / numpy (and ANY PEP 3118 buffer exporter) | [Python.NET](https://github.com/pythonnet/pythonnet) + the buffer protocol |

Guides — every sample maps 1:1 to a test in `NumSharp.Interop.UnitTests`:

- [The Zero-Copy Model](zero-copy-model.md): how a conversion decides **view or copy** — the `Auto` decision tree, the three routes that produce a view, the three layouts that genuinely cannot be shared (`complex64`, big-endian, sub-item strides), the `Py_buffer` lock you accept when you share, and the measured coverage (45 view / 2 copy across 48 exporter varieties).
- [Numpy.NET — coexistence & migration](numpy-net.md): zero-copy interop with SciSharp's `Numpy`/`Numpy.Bare` packages — wrap NumSharp buffers in their `NDarray`, lease their arrays into NumSharp, one shared engine, and the GIL rule their library needs.

## The Interop Contract

Every bridge builds on the same three NumSharp capabilities:

1. **Raw layout access** — an `NDArray` exposes its base address, element strides, offset and dtype, so any strided-array convention (numpy views, buffer protocol shapes, DLPack-style descriptors) can be expressed without copying.
2. **External memory wrapping** — `UnmanagedMemoryBlock<T>` can wrap foreign pointers with a custom release hook, so foreign buffers become first-class `NDArray`s that every NumSharp kernel runs over directly.
3. **Atomic reference counting** — the memory block behind every array is refcounted. A bridge takes its own reference while the other side can still see the buffer and releases it when that side lets go — memory is never freed early and never leaks, regardless of which side's garbage collector fires first. Guards like `ndarray.resize(refcheck: true)` see those references too, mirroring NumPy's own protection for exported buffers.

```csharp
// The primitive every bridge is made of: wrap foreign memory with a release hook.
var nd = new NDArray(new UnmanagedStorage(
    new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>(ptr, length, onLastReferenceReleased)),
    new Shape(rows, cols)));
```

Building a bridge to another ecosystem (Arrow, DLPack, a GPU runtime)? The pythonnet package is the reference implementation of the lifetime model — [open a PR](https://github.com/SciSharp/NumSharp) to add yours to this list.
