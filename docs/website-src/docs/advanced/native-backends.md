# Native code & backends

NumPy's advanced section includes **F2PY**, the tool that bridges Python to compiled Fortran and C so a NumPy program can call into a native library. NumSharp's story is deliberately different: **`NumSharp.Core` has no native dependency and no P/Invoke** — every kernel is its own managed C#, and that is the default a user gets. Native code enters through exactly one curated seam, and it is *opt-in*: you reference a backend package (or write one), you do not hand-roll bindings into Core.

This page explains that seam — `IBlasBackend` on `TensorEngine.Blas` — the flagship backend that uses it, and how to write your own.

<!-- Tests: NumSharp.Tests.Documentation.AdvancedNativeBackendsDocTests — the backend-INDEPENDENT code examples on this page are asserted in test/NumSharp.Tests/Documentation/AdvancedNativeBackendsDocTests.cs (integer products compute bit-exact; the LU family computes). The OpenBLAS-specific claims (byte-identical products with a backend; a factorisation raising NotSupportedException without one) depend on the global TensorEngine.Blas state and are gated by LinAlgEngineSeamTests + the NumSharp.Interop.OpenBLAS test project — see interop/openblas.md. -->

---

## The default: 100% managed, no native anything

There is no Fortran to bridge and no C library to link. NumSharp's matrix products, factorisations, transcendentals, and every other kernel are managed C# with runtime-generated SIMD ([IL Generation](../il-generation.md)). For most workloads this is the whole story — and for integer/boolean products it is even *exactly* NumPy's answer, because modular integer arithmetic is associative regardless of order.

The one place a portable managed algorithm cannot match NumPy **bit-for-bit** is float matrix products: NumPy always routes float32/float64/complex128 `@` through cblas, whose arch-specific multi-accumulator scheme no portable loop reproduces. That — and raw speed on large matrices — is why a native backend seam exists.

---

## The seam: `TensorEngine.Blas` (`IBlasBackend`)

Core's entire knowledge of a native backend is **one interface plus one settable property**:

```csharp
public interface IBlasBackend                       // Backends/IBlasBackend.cs
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;  // optional
    // + 15 more Try* members (products + factorisations), all defaulting to false
}

public abstract partial class TensorEngine
{
    public IBlasBackend Blas { get; set; }   // null (the default) = NumSharp's own managed kernels
}
```

Every member is **`Try`-shaped**: a backend answers only for the operands it implements and returns `false` for the rest, so a backend can change *which* implementation computes a product, never *whether* one can be computed. Assigning `null` takes the backend back out. The engine is **not** subclassed or replaced — it is one cached singleton whose `Blas` property is read per call — because an `NDArray` binds its engine at construction, so a property on the shared instance reaches arrays that already exist.

There is a companion seam, `ISlidingDotBackend`, that routes `np.correlate`/`np.convolve`'s reductions through the same native `?dot` for byte-parity.

---

## The flagship backend: `NumSharp.Interop.OpenBLAS`

Referencing this package **is the whole opt-in** — a `[ModuleInitializer]` assigns `TensorEngine.Blas`. It implements `IBlasBackend` over **the exact OpenBLAS/LAPACK binary NumPy itself pins** (scipy-openblas, verified byte-identical to NumPy's copy, bundled as per-RID runtime assets so parity needs no Python installed).

With it referenced:

- `np.dot`, `np.matmul`, `np.inner`, `np.vdot`, `np.vecdot`, `np.matvec`, `np.vecmat` for **float32/float64/complex128** route through cblas — **byte-identical to NumPy**, and faster on large matrices (1.7–13×).
- The `np.linalg` factorisations — `cholesky`, `eig`/`eigh`, `svd`, `qr`, `solve`, `inv`, `det`, `lstsq`, … — route through LAPACK (`getrf`/`gesv`/`potrf`/`geev`/`gesdd`/`geqrf`/`gelsd`), byte-identical to NumPy. (Without a backend, the LU family has a managed fallback; the eigen/SVD/QR family raises `NotSupportedException` naming the LAPACK routine to install.)

```csharp
using NumSharp;

OpenBlasEngine.Enable();                 // load + install the bundled library
np.dot(a, b);                            // now byte-identical to NumPy's np.dot

OpenBlasEngine.Enabled;                  // true
OpenBlasEngine.Info;                     // library + core-type description
OpenBlasEngine.LibraryPath;              // which .dll/.so/.dylib was loaded
OpenBlasEngine.Disable();                // back to managed kernels
```

Three parity levers matter: the result bits depend on the BLAS **thread count** and the **DYNAMIC_ARCH kernel** the CPU dispatches, and a **named library is binding** (never silently substituted). Pin them with `OpenBlasEngine.Enable(library, threads, coreType)` when you need to match a specific local NumPy exactly. Full detail: [OpenBLAS interop](../interop/openblas.md).

> **Trap:** `Environment.SetEnvironmentVariable` does **not** reach a native `getenv` — .NET keeps its own table. To set `OPENBLAS_CORETYPE`/thread env for the native library, go through the CRT (`_putenv_s`/`setenv`), or use the `OpenBlasEngine.Enable(...)` parameters.

---

## Writing your own backend

Implement `IBlasBackend` and assign it — typically from a `[ModuleInitializer]` so referencing your package is the opt-in:

```csharp
public sealed class MyBackend : IBlasBackend
{
    public string Info => "MyBackend (MKL 2024)";
    public bool TryDot(NDArray a, NDArray b, out NDArray result) { /* … */ }
    public bool TryMatMul2D(NDArray a, NDArray b, NDArray result) { /* … */ }
    // return false for anything you don't implement — Core falls back to managed
}

static class Init
{
    [ModuleInitializer]
    internal static void Install() =>
        BackendFactory.GetEngine().Blas = new MyBackend();
}
```

Read the operands as `(T*)a.GetData().Address + a.Shape.Offset` with `a.Shape.Strides` (all public; **strides are in elements**, so multiply by the caller's byte convention yourself — NumPy's are bytes). Three rules the review pinned, each with a reproduction:

1. **Read `Blas` into a local before calling it** — a test-then-call on a settable property was ~2% NREs under a concurrent `Disable()`.
2. **A failed `Enable` must be a no-op** — do not unload the working library first, or `Enabled` reports true while products silently revert to managed kernels.
3. **No `InternalsVisibleTo` is needed** — everything a backend reads is public. Do **not** pair `a.GetData<T>().Address` (which densifies a non-contiguous view) with `a.Shape.Strides`.

---

## When native, when managed

| Operation class | Without a backend | With `NumSharp.Interop.OpenBLAS` |
|-----------------|-------------------|-----------------------------------|
| float32/float64/complex128 `dot`/`matmul` + product gufuncs | managed GEMM (fast, ~ULP off NumPy) | **cblas, byte-identical to NumPy, faster** |
| integer / bool products | managed (bit-exact by construction) | managed (associativity makes native pointless) |
| LU family (`det`/`slogdet`/`solve`/`inv`) | **managed fallback** (`allclose`) | LAPACK, byte-identical |
| eigen / SVD / QR / Cholesky / `lstsq` | **`NotSupportedException`** (no managed impl) | LAPACK, byte-identical |
| everything elementwise / reductions / sorts | managed SIMD | managed SIMD (unchanged) |

The dividing line is what a portable managed algorithm *costs*: a GEMM and an unblocked LU are a few hundred lines each (so they exist in Core); a portable `gesdd`/`geev`/`gelsd` is a library in its own right (so it waits for the backend).

---

## Troubleshooting

### "`np.linalg.svd` threw NotSupportedException"
Core ships no managed SVD/eig/QR. Reference `NumSharp.Interop.OpenBLAS` — the message names the LAPACK routine and the `IBlasBackend.Try*` member the backend supplies.

### "My product results changed after referencing OpenBLAS"
That's expected — they became byte-identical to NumPy (and the exact bits depend on thread count + dispatched kernel). Pin `threads`/`coreType` via `OpenBlasEngine.Enable(...)` to match a specific NumPy.

### "`Enable()` said it loaded but products are still managed"
A named/pinned library that fails to load leaves the backend uninstalled (a required-override miss is reported to stderr, never silently substituted). Check `OpenBlasEngine.LibraryPath` / `OpenBlasEngine.Info`.

---

## API reference

| Member | Purpose |
|--------|---------|
| `TensorEngine.Blas` (`IBlasBackend`) | the single settable backend seam; `null` = managed |
| `IBlasBackend.Try{Dot,MatMul2D,MatMulBatched,Inner,Vdot,Vecdot,Matvec,Vecmat}` | product routes |
| `IBlasBackend.Try{Cholesky,Det,Slogdet,Eig,Eigh,Inv,Lstsq,Qr,Solve,Svd}` | factorisation routes |
| `ISlidingDotBackend` | optional `?dot` seam for `correlate`/`convolve` |
| `OpenBlasEngine.Enable(library, threads, coreType)` / `Disable()` / `Enabled` | toggle the OpenBLAS backend |
| `OpenBlasEngine.Info` / `LibraryPath` / `LoadedImagePath` / `CoreName` / `CoreType` / `IsBundledLibrary` | introspection + parity pins |

---

## Related reading

- [OpenBLAS interop](../interop/openblas.md) — the full delivery, bundling, and parity story.
- [Extending NumSharp](extending-numsharp.md) — the managed extension seams.
- [Interoperability](../interop/index.md) — sharing *buffers* (a different kind of native boundary).
- [NumPy Compliance & Compatibility](../compliance.md) — why byte-parity needs the native path.
- [F2PY user guide](https://numpy.org/doc/stable/f2py/) — the upstream article this converts.
