# NumSharp.Interop.BLAS

Optional external-CBLAS matrix-product backend for [NumSharp](https://github.com/SciSharp/NumSharp).

**NumSharp.Core is 100 % managed C# with no native dependency.** Its own SIMD GEMM computes every
matrix product out of the box. This package is the *alternative*: reference it and `np.dot` /
`np.matmul` compute `float32` / `float64` products through a CBLAS library you already have — which
is both **faster on large matrices** and, when that library is the one your reference NumPy calls,
**bit-identical to NumPy**.

```powershell
dotnet add package NumSharp.Interop.BLAS
```

```csharp
using NumSharp;
using NumSharp.Interop.Blas;

// Referencing the package is the opt-in: a module initializer installs the backend if a CBLAS
// library can be found, and silently does nothing if one cannot.
var c = np.dot(a, b);

// For BYTE parity with a specific NumPy, name that wheel's own BLAS and pin the thread count —
// the result bits depend on both.
Blas.Enable(@"…\site-packages\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);
Console.WriteLine(Blas.Info);
// C:\...\libscipy_openblas64_-74a4….dll [symbols scipy_*64_, ILP64, threads 1] OpenBLAS 0.3.31.dev …

Blas.Disable();   // clears engine.Blas — back to NumSharp's managed SIMD GEMM
```

## What it does

`OpenBlasBackend` implements NumSharp's `IBlasBackend` — `Try`-shaped members only — and the package
assigns it to the engine's `TensorEngine.Blas` property. The engine itself is NOT replaced or
subclassed: every other operation, dtype, kernel and iterator is untouched, and anything the backend
does not serve (any dtype other than float32/float64, or a shape outside the ported surface) is
computed by the built-in managed kernels as before.

```csharp
// The entire Core-side surface:
public interface IBlasBackend {
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;  // optional
}
public abstract class TensorEngine { public IBlasBackend Blas { get; set; } }
```

`TryMatMulBatched` takes a whole stacked product at once. It is a **default interface method**, so a
backend that only implements the first two members compiles and runs unchanged — the engine falls
back to calling `TryMatMul2D` once per batch element. Implementing it lets a backend amortise
per-product setup across the stack (the trailing strides never vary within one), which is what NumPy
does in its matmul gufunc's outer loop; measured on 2000 stacked 8×8 float32 products, doing that
setup per element instead made this backend **0.80×** — *slower* than NumSharp's own managed GEMM —
against **6.5×** with the work hoisted.

Writing your own backend needs no access to NumSharp's internals: read operands as
`(T*)a.GetData().Address + a.Shape.Offset` with `a.Shape.Strides` (element units). Do **not** use
`a.GetData<T>().Address` — it densifies a non-contiguous view, so pairing it with `Shape.Strides`
reads outside the buffer.

The two entry points are a **route-for-route port of NumPy 2.4.2's two matrix-product dispatchers** —
`cblas_matrixproduct` (behind `np.dot`) and the `@TYPE@_matmul` gufunc (behind `np.matmul` and
`@`). They are genuinely different C code and disagree on some inputs, so both are mirrored:
gemm (with the `a @ a.T` **syrk** shortcut and NumPy's copy-if-not-blasable rule), both gemv
directions, the row·column `?dot` with NumPy's double accumulator, the portable no-blas loop,
stacked/batched products, and the N-D `np.dot` route that is not gemm at all.

## Why a port, and not just "call gemm"

Both NumSharp's answer and NumPy's are correct floating-point matrix products; they sum in
different orders. NumSharp's managed GEMM differed from NumPy on **94.5 %** of the elements of a
`(128,784)@(784,128)` float32 product (max ~976 000 ULP) — enough that training the same network in
both stacks diverges. NumPy's bits *are* its BLAS's bits, and OpenBLAS' multi-accumulator
micro-kernels match neither a sequential `mul + add` chain nor a sequential FMA chain. Reproducing
them means calling the same binary through the same route with the same flags.

Three things are load-bearing:

- **The thread count is part of the answer** — 1, 2, 4 and 24 threads give four different results.
- **A named library is binding** — an explicit path (or `NUMSHARP_PARITY_BLAS`) is used as given and
  never silently replaced, because parity is a claim about one specific binary. A failed
  `Blas.Enable` is a **no-op**: it throws having changed nothing, so a mistyped path cannot demote a
  working setup back to the managed kernels behind your back.
- **No native asset ships with this package** — the BLAS is whichever one you already have.

## Supported providers

Any CBLAS export set is bound, in this order: numpy's renamed ILP64 wheels
(`scipy_cblas_sgemm64_`), plain ILP64 (`cblas_sgemm64_` / `cblas_sgemm_64`), and stock LP64
(`cblas_sgemm`). Integer widths are marshalled per call. OpenBLAS' thread and build-info entry
points are used when present; a reference CBLAS without them works fine.

## Environment

| Variable | Effect |
|---|---|
| `NUMSHARP_PARITY_BLAS` | Path (or directory) of the CBLAS library to load. Binding, like an explicit argument. |
| `NUMSHARP_BLAS_AUTOINSTALL=0` | Skip the module-load auto-install; `Blas.Enable(...)` still works. |

Full design record, probe results and traps: [`docs/GEMM_PARITY.md`](https://github.com/SciSharp/NumSharp/blob/master/docs/GEMM_PARITY.md).
