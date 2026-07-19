# Interoperability

NumSharp arrays are raw, unmanaged, densely-typed buffers — exactly the shape the rest of the scientific-computing world speaks. The interop packages turn that into practice: **zero-copy bridges** that let other runtimes and libraries read and write the same memory NumSharp computes over, with lifetimes coupled safely across the boundary.

## Official Bridges

| Package | Bridge | Transport |
|---------|--------|-----------|
| [NumSharp.Interop.pythonnet](pythonnet.md) | `NDArray` ⇄ Python / numpy (and ANY PEP 3118 buffer exporter) | [Python.NET](https://github.com/pythonnet/pythonnet) + the buffer protocol |

Guides — every sample maps 1:1 to a test in `NumSharp.Interop.UnitTests`:

- [The Zero-Copy Model](zero-copy-model.md): how a conversion decides **view or copy** — the `Auto` decision tree, the three routes that produce a view, the three layouts that genuinely cannot be shared (`complex64`, big-endian, sub-item strides), the `Py_buffer` lock you accept when you share, and the measured coverage (47 view / 2 copy / 1 rejected across 50 exporter varieties).
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

The release hook is the whole trick: it fires when the **last** NumSharp reference to that block — the original array *or any view derived from it* — goes away, whether by `Dispose()` or by the GC. A bridge puts "tell the other side we're done" in that hook and the refcount takes care of ordering, so neither side has to know about the other's lifetime rules.

> **Alias it, or it can detach.** Written exactly as above, the array believes it **owns** the block, so a size-changing `nd.resize(...)` succeeds: it allocates fresh NumSharp memory, releases the foreign one and leaves your bridge pointing at nothing. Pass the storage through `Alias` to give it view semantics instead — numpy's `owndata == False` — and the same resize refuses with `cannot resize this array: it does not own its data`:
>
> ```csharp
> var storage = new UnmanagedStorage(slice, Shape.Vector(length)).Alias(new Shape(rows, cols));
> ```
>
> This is what the pythonnet bridge's import path does, and it is what makes an imported view behave like `np.frombuffer(...)` on both sides of the boundary.

## Three questions every bridge must answer

The pythonnet package is the reference implementation; if you build another, these are the decisions it had to make:

1. **Share or duplicate?** Not every foreign layout is representable — dtype widths, byte order and stride granularity all have to line up. Prefer sharing, fall back to copying, and be explicit about which you did. The pythonnet bridge's answer is written up in [The Zero-Copy Model](zero-copy-model.md).
2. **Who releases, and when?** Both runtimes have their own collector. Sharing means each side must hold a reference the *other* side's collector respects, and the release must be safe to run from a finalizer thread, on a foreign thread, or during interpreter teardown.
3. **What does sharing cost the other side?** A live view is not free: it pins the source. Python's `bytearray` refuses to resize and numpy's `resize(refcheck=True)` refuses to reallocate while a lease exists — correct behaviour, but it must be documented, not discovered.

Building a bridge to another ecosystem (Arrow, DLPack, a GPU runtime)? [Open a PR](https://github.com/SciSharp/NumSharp) to add yours to this list.
