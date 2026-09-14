# Advanced usage and interoperability

These pages are the NumSharp counterparts of NumPy's [*Advanced usage and interoperability*](https://numpy.org/doc/stable/user/index.html) section — one NumSharp article per NumPy article. Where NumPy's advanced story is dominated by dropping into **C** (the C-API, F2PY's Fortran bridge, the CPython internals), NumSharp's is the opposite: `NumSharp.Core` is **100% managed C# with no native dependency and no P/Invoke**, so "advanced" here means the managed seams you extend through, the internals that make views cheap, the one place native code *can* plug in, and the buffer contract that lets any .NET ecosystem share an array.

If the [Fundamentals](../fundamentals/index.md) pages are about *using* `NDArray`, these are about *extending* it, *understanding* it, and *connecting* it.

---

## The articles

| Article | What it covers | NumPy original |
|---------|----------------|----------------|
| [Extending NumSharp](extending-numsharp.md) | The managed extension seams — custom kernels via `NDIter`/typed iterators, fused ops via `np.evaluate`/`NDExpr`, the `TensorEngine` seam, runtime IL generation, the `NumSharp.Build` weaver | [Using NumPy C-API](https://numpy.org/doc/stable/user/c-info.html) |
| [Native code & backends](native-backends.md) | Why Core has no native dependency, how a native BLAS/LAPACK plugs in through `IBlasBackend`, and how to write your own backend | [F2PY](https://numpy.org/doc/stable/f2py/) |
| [Under the hood — internals](under-the-hood.md) | The buffer + metadata split, `UnmanagedStorage`/`Shape`/strides/offset/flags, the ARC memory model, and C-order indexing | [Under-the-hood for developers](https://numpy.org/doc/stable/dev/underthehood.html) |
| [Interoperability](../interop/index.md) | The one-buffer contract every bridge builds on, and the bridges themselves (pythonnet/numpy, PyTorch, Pandas, ONNX, ML.NET, System.Numerics.Tensors, `np.frombuffer`) | [Interoperability with NumPy](https://numpy.org/doc/stable/user/basics.interoperability.html) |

> **Interoperability** already has a dedicated, in-depth NumSharp hub ([Interoperability](../interop/index.md) and its per-bridge pages); it *is* the NumSharp conversion of NumPy's interoperability article — the one-buffer contract plus every bridge — so it is linked here rather than duplicated.

---

## The one big difference from NumPy: no C-API

NumPy's advanced section is largely about escaping Python into C — writing C extension modules, custom ufuncs in C, subtyping the ndarray in C, and F2PY's Fortran bridge. NumSharp does not have, and does not need, any of that:

- **There is no C-API.** You extend NumSharp in **managed C#**, through the seams documented in [Extending NumSharp](extending-numsharp.md) — the same performance is reached with runtime **IL generation + SIMD**, not hand-written C. And with **C# 14 extension members** (in the `NumSharp` namespace) you can grow the `np.*` / `NDArray` surface itself — adding static `np.your_func(...)` functions and `NDArray` methods/properties that read like built-ins, something NumPy's `np` namespace can't cleanly do.
- **Native code is opt-in and curated, not hand-rolled.** The only native path is an optional package implementing the [`IBlasBackend`](native-backends.md) seam (e.g. `NumSharp.Interop.OpenBLAS`, which binds the very OpenBLAS/LAPACK binary NumPy ships). You reference a package; you do not write P/Invoke into Core.
- **The internals are all managed too** — `UnmanagedStorage` is raw unmanaged memory managed by an atomic reference count, `Shape` is a `readonly struct`, and views are metadata-only reinterpretations, exactly as NumPy's are, but with no CPython object machinery.

---

## Where these fit among the other guides

Several existing NumSharp guides are the deep-dives these advanced pages point into:

- [NDArray](../NDArray.md) and [Buffering & Memory](../buffering.md) — the storage, shape, and reference-counting mechanics.
- [NDIter (Kerneling NDArray)](../NDIter.md) — the iterator you drive to write custom kernels.
- [IL Generation](../il-generation.md) — how the SIMD kernels are emitted at runtime.
- [NumSharp.Build Compiler](../numsharp-build-compiler.md) — the source weaver.
- [NumPy Compliance & Compatibility](../compliance.md) — the parity story these internals serve.

---

## Related reading

- [Fundamentals and usage](../fundamentals/index.md) — the *using* half of the guide.
- [NumPy user guide — Advanced usage and interoperability](https://numpy.org/doc/stable/user/index.html) — the upstream section these pages track.
