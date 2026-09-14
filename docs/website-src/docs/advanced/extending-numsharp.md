# Extending NumSharp

NumPy is extended through its **C-API**: you write a C extension module to create and consume `ndarray`s, hand-code a custom ufunc in C, or subtype the array in C. NumSharp has **no C-API** — `NumSharp.Core` is 100% managed C# — and it does not need one. You extend it in managed code, and you still reach SIMD speed, because the hot loops are generated as IL at runtime rather than compiled from C ahead of time.

This page maps the extension seams, from "compute over an array in a tight loop" to "replace the whole compute backend". Each seam has a dedicated deep-dive; this is the map.

---

## The seams at a glance

| You want to… | Use | NumPy C-API analog |
|--------------|-----|--------------------|
| Loop over elements yourself, fast, unboxed | `np.nditer<T>` / `np.nditer_chunks<T>` / `nd.Unsafe` | iterate a buffer via `NpyIter` / `PyArray_DATA` |
| Compose a custom elementwise/reduction op with no temporaries | `np.evaluate` / `NDExpr` | write a custom ufunc |
| Drive multi-operand, broadcasting, buffered traversal | `NDIter` | the `NpyIter` C iterator |
| Emit a specialized SIMD kernel at runtime | `ILKernelGenerator` / `DirectILKernelGenerator` | a hand-written C inner loop |
| Swap the compute backend (e.g. native BLAS) | `TensorEngine.Blas` (`IBlasBackend`) | link a different C/Fortran library |
| Weave ownership/lifetime at build time | `NumSharp.Build` | — (no analog) |
| Turn on threaded kernels | `np.multithreading` | OpenMP in a C loop |

---

## 1. Write your own element loop

The lowest-friction extension point: iterate the array's elements in C# and do whatever you like. Prefer the **typed, unboxed** iterators — they yield `ref T` (or a `Span<T>` per chunk) straight into the array's memory, write-through, with no allocation:

```csharp
// element-wise, by reference — writes land in the array
foreach (ref double x in np.nditer<double>(a, writeable: true))
    x = MyTransform(x);

// chunk-wise — a Span<T> per inner loop, ideal for TensorPrimitives / your own SIMD
foreach (Span<double> chunk in np.nditer_chunks<double>(a, writeable: true))
    System.Numerics.Tensors.TensorPrimitives.Multiply(chunk, 2.0, chunk);
```

These walk **any layout** (contiguous, strided, transposed, reversed, broadcast) in memory order, and are 40–250× faster than a boxed per-element loop. For the raw pointer (your own unsafe kernel), use `nd.Unsafe`:

```csharp
Span<double> s = a.Unsafe.Span<double>();     // zero-copy, C-contiguous
double* p      = a.Unsafe.Pointer<double>();  // element 0; walk with a.strides
```

See [Iterating & Enumerating](../iterating-and-enumerating.md) for the full set (`flat`, `ndindex`, `ndenumerate`, `.NET` interop) and the lifetime rules for `nd.Unsafe`.

---

## 2. Compose a custom op with `np.evaluate` / `NDExpr`

The managed analog of "write a custom ufunc" is to build an **expression tree** and let NumSharp compile it into one fused pass — every elementwise node runs inside a single inner loop, reading each operand once and allocating no intermediates:

```csharp
// a fused (a*b + c) with no temporary arrays:
NDArray r = np.evaluate((NDExpr)a * b + c);

// a fused reduction — sum(a*b) in one pass, no temp:
NDArray s = np.evaluate(NDExpr.Sum((NDExpr)a * b));

// call your own scalar function per element inside the fused loop:
NDArray t = np.evaluate(NDExpr.Call(a, x => MyScalarFn(x)));
```

`np.evaluate` compiles **once per (tree structure, input dtype signature)** and caches, so building the tree inside a loop is cheap. Per-node dtypes follow the same NEP 50 rules as the built-in ufuncs. This is NumSharp's equivalent of `numexpr` and the closest thing to authoring a ufunc — see the fused-expressions section of [Universal functions](../fundamentals/ufuncs.md).

> Hold `Call` delegates in a field, not a per-call closure — a closure allocated per call is a new delegate identity and forces a fresh kernel JIT. See the `NDExpr` notes in the API reference.

---

## 3. Drive `NDIter` for a full custom kernel

For a kernel that needs broadcasting, multiple operands, buffered casting, or reductions — the jobs NumPy solves with the `NpyIter` C-API — NumSharp exposes `NDIter`, the same iterator its own engine uses. It handles C/F/A/K order, broadcasting, external loops, buffering, casting, masks, and synchronized traversal. This is the seam to reach for when the typed iterators above are too simple:

```csharp
using var it = np.nditer(new[] { a, b, null }, ...);   // allocate an output operand
// drive the inner loop; write into operand 2
```

See [NDIter (Kerneling NDArray)](../NDIter.md) for the full contract, and note the disposal rules (`NDIter` owns unmanaged state — use `using`).

---

## 4. Emit a specialized SIMD kernel (IL generation)

NumSharp reaches C-speed without C by generating kernels as IL at runtime via `System.Reflection.Emit`, with SIMD (V128/V256/V512) chosen at startup. Two generators split by contract:

- **`DirectILKernelGenerator`** — whole-array kernels (the kernel walks dimensions/strides itself). Carries most elementwise, reduction, scan, cast, and selection kernels.
- **`ILKernelGenerator`** — per-chunk kernels driven as an `NDIter` inner loop (NumPy's `PyUFuncGenericFunction` model).

You rarely emit kernels by hand — the engine does it for the built-in ops — but understanding which generator owns a kernel is how you add one to the engine. See [IL Generation](../il-generation.md).

---

## 5. Replace the compute backend (`TensorEngine.Blas`)

The heaviest extension: change *which* implementation computes an operation. NumSharp exposes exactly one settable seam on the engine — `TensorEngine.Blas`, an `IBlasBackend` — so an optional package can route products and factorisations through a native library while everything else stays managed. This is how `NumSharp.Interop.OpenBLAS` makes `np.dot`/`np.matmul` byte-identical to NumPy. Writing your own backend is a matter of implementing `IBlasBackend` and assigning the property. This is covered in full in [Native code & backends](native-backends.md).

---

## 6. Weave lifetime at build time (`NumSharp.Build`)

`NumSharp.Build` is a Roslyn-based source weaver/analyzer package that enforces `NDArray` ownership and disposal at compile time (the `[NDScoped]` family of rules). It is the tooling seam for authoring NumSharp-consuming code correctly — see [NumSharp.Build Compiler](../numsharp-build-compiler.md).

---

## 7. Turn on threaded kernels

NumSharp kernels are **single-threaded by default** (deterministic, no oversubscription). Opt into threaded kernels explicitly:

```csharp
np.multithreading(enabled: true, max_threads: 8);
```

Use this for large, embarrassingly-parallel workloads; leave it off for latency-sensitive or already-parallel callers.

---

## Common patterns

### A custom elementwise transform, fastest first

```csharp
// Preferred: fused, no temporaries, cached kernel
var y = np.evaluate(NDExpr.Call((NDExpr)x * 2.0, v => Math.Tanh(v)));

// Or: unboxed loop when the logic doesn't fit an expression
foreach (ref double v in np.nditer<double>(x, writeable: true))
    v = CustomActivation(v);
```

### Hand the buffer to `TensorPrimitives` (BCL SIMD)

```csharp
foreach (Span<float> c in np.nditer_chunks<float>(x, writeable: true))
    TensorPrimitives.Sigmoid(c, c);
```

### A drop-in native backend

```csharp
OpenBlasEngine.Enable();     // reference NumSharp.Interop.OpenBLAS; products now route to OpenBLAS
```

---

## Troubleshooting

### "`np.frompyfunc` / `np.vectorize` doesn't exist"
Correct — NumSharp has no Python-function-wrapping ufunc factory. Use `np.evaluate` with `NDExpr.Call` for a fused per-element function, or an `np.nditer<T>` loop.

### "My `np.evaluate` recompiles every call"
A `Call` delegate captured in a per-call closure is a new identity each time. Hold the delegate in a field. Also prefer a 0-d `NDArray` over a literal for a runtime-varying scalar (a literal bakes one kernel per value).

### "`nd.Unsafe.Span<T>()` threw"
The Span/Memory/Bytes views require C-contiguity, the exact dtype, and ≤ `int.MaxValue` elements. Use `Pointer<T>()` + `nd.strides` for non-contiguous or huge arrays. See [Iterating & Enumerating → .NET interop](../iterating-and-enumerating.md).

---

## API reference

| Seam | Entry points |
|------|--------------|
| Typed iteration | `np.nditer<T>`, `np.nditer_chunks<T>`, `np.flat<T>`, `nd.Unsafe.{Span,Memory,Bytes,Pointer}<T>` |
| Fused expressions | `np.evaluate`, `NDExpr.{Arr,Sum,Prod,Min,Max,Mean,Call}`, `expr.Compile()` |
| General iterator | `np.nditer`, `NDIter` |
| IL kernels | `ILKernelGenerator`, `DirectILKernelGenerator` |
| Backend seam | `TensorEngine.Blas` (`IBlasBackend`), `ISlidingDotBackend` |
| Build weaver | `NumSharp.Build` (`[NDScoped]`) |
| Threading | `np.multithreading(enabled, max_threads)` |

---

## Related reading

- [Native code & backends](native-backends.md) — the `IBlasBackend` seam in full.
- [Under the hood — internals](under-the-hood.md) — the memory model your kernels run over.
- [NDIter (Kerneling NDArray)](../NDIter.md) · [IL Generation](../il-generation.md) · [NumSharp.Build Compiler](../numsharp-build-compiler.md) · [Iterating & Enumerating](../iterating-and-enumerating.md).
- [Universal functions](../fundamentals/ufuncs.md) — the ufunc model these seams extend.
- [Using NumPy C-API](https://numpy.org/doc/stable/user/c-info.html) — the upstream article this converts.
