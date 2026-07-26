# `NumSharp.Interop.BLAS` — byte-identical `np.dot` / `np.matmul`

> **✅ IMPLEMENTED (2026-07-25).** Optional package `src/NumSharp.Interop.BLAS/`
> (`OpenBlasBackend : IBlasBackend`, `Blas`, `CBlasNative`, `BlasParity{,.Matmul,.Dot,.Entry}`),
> installed through the one seam Core exposes — the engine's settable `TensorEngine.Blas`
> property — gate `FuzzCorpusTests.MatmulParity` over the 342-case `matmul_parity` corpus tier.
> Measured: **342/342 bit-exact with the backend installed, 294/342 divergent without it.**
>
> **NumSharp.Core stays 100 % managed C#** — it contains no BLAS code, no P/Invoke and no native
> dependency. With this package absent every matrix product is NumSharp's own SIMD GEMM.

Status date: 2026-07-25 · Branch: `journey3` · Issue:
[SciSharp/NumSharp#626](https://github.com/SciSharp/NumSharp/issues/626)

---

## 1. The problem

NumSharp's `np.dot` disagreed with NumPy's on **94.5 % of the elements** of a
`(128,784) @ (784,128)` float32 product (max ~976 000 ULP), and on **45 %** at `K = 10`. Both
answers are correct floating-point matrix products; they differ because they sum in different
orders. That is fatal for the one thing #626 asks for — training a network in both stacks and
byte-comparing the weights — because the divergence compounds: 50 Adam steps spread it to 73 % of
`w1`'s bytes.

The committed `matmul` fuzz tier never saw this. Its operands are small integers and its deepest
contraction is `k = 4`, where every summation order agrees exactly.

## 2. Why no portable algorithm can fix it

Read `numpy/_core/src/umath/matmul.c.src` (the clone is in-repo at `src/numpy/`). For float32 and
float64 **matrix @ matrix, NumPy always calls cblas**: since gh-23588 it copies a non-blasable
operand into a contiguous temp and calls BLAS anyway, so the portable C loop
`matmul_inner_noblas` (`*(typ*)op += val1 * val2`, ascending k) is unreachable in a stock wheel
except for zero-sized or `> BLAS_MAXSIZE` dimensions.

So NumPy's bits are scipy-openblas' bits, and those were probed to match **neither** a sequential
`mul + add` chain **nor** a sequential FMA chain (8 per-element references each; float32 FMA is
exactly emulable in float64 as `f32(f64(a)*f64(b)+f64(acc))`). OpenBLAS' `sgemm` micro-kernel uses
an arch-specific multi-accumulator register scheme. Worse, the bits also depend on:

| Knob | Measured effect |
|---|---|
| BLAS worker threads | 1, 2, 4 and 24 threads give **four different results** (35 540 / 39 055 / 45 091 elements differ from the 24-thread answer on a `(512,784)@(784,512)` f32 product) |
| CPU kernel (`DYNAMIC_ARCH`) | a different micro-architecture dispatches a different micro-kernel |
| The library build itself | different OpenBLAS versions re-tune blocking |

Reimplementing that per CPU in managed code was rejected: brittle, per-arch, and broken by every
wheel or CPU change.

## 3. What was built (strategy S-A)

An **optional package** that P/Invokes the *very same BLAS binary* the target NumPy calls, driven by
a **route-for-route port of NumPy's two matrix-product dispatchers**. Same binary + same route +
same flags ⇒ same bits by construction.

```powershell
dotnet add package NumSharp.Interop.BLAS
```

```csharp
using NumSharp.Interop.Blas;

// Referencing the package IS the opt-in: a [ModuleInitializer] installs the backend if a CBLAS
// library can be found, and silently does nothing if one cannot.
Blas.Enable(@"…\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);  // pin, for byte parity
Blas.Enabled;                                            // bool
Blas.Info;                                               // path + symbol scheme + int width + threads + build string
Blas.Disable();                                          // back to NumSharp's managed SIMD GEMM
```

Discovery order: explicit path/directory → `NUMSHARP_PARITY_BLAS` → the `numpy.libs` folder of any
python on `PATH` (plus `VIRTUAL_ENV` / `CONDA_PREFIX` / `PYTHONHOME`) → bare loader names.
NumPy's wheels rename their symbols `scipy_cblas_sgemm64_`-style and use **64-bit BLAS integers**;
a stock LP64 `cblas_sgemm` is also bound, with the integer width marshalled per call
(`CBlasNative.IsIlp64`).

### The seam: one property on the engine

NumSharp's GEBP/FMA SIMD GEMM is untouched and remains the default. Core gained a single
abstraction and a single settable property — and nothing else BLAS-shaped:

```csharp
namespace NumSharp.Backends;

public interface IBlasBackend                                    // the whole Core-side surface
{
    string Info { get; }
    bool TryDot(NDArray left, NDArray right, out NDArray result);
    bool TryMatMul2D(NDArray left, NDArray right, NDArray result);
    bool TryMatMulBatched(NDArray left, NDArray right, NDArray result) => false;   // optional
}

public abstract class TensorEngine
{
    public IBlasBackend Blas { get; set; }   // null (default) = NumSharp's own managed kernels
}
```

The package implements `IBlasBackend` and assigns it from a `[ModuleInitializer]`. `DefaultEngine`
consults the property at the top of `Dot`, inside `MultiplyMatrix`, and at the top of
`BatchedMatmul`, and computes the product itself whenever it is null or the backend returns false:

| Entry point | Mirrors |
|---|---|
| `TryDot` | `PyArray_MatrixProduct2` → `cblas_matrixproduct` (ndim ≤ 2) + the `dotfunc` iterator tail (ndim > 2) |
| `TryMatMul2D` | `@TYPE@_matmul`, the gufunc behind `np.matmul` and `@` |
| `TryMatMulBatched` | that gufunc's OUTER loop — the whole stack in one call, so the route decision and scratch are built once, as NumPy builds them once from `steps[]` |

All three are `Try`-shaped because a backend answers only for what it implements (an external CBLAS:
float32/float64), so **installing one can change WHICH implementation runs, never WHETHER NumSharp
can compute the product**. That invariant is load-bearing enough to cost coverage: `TryMatMulBatched`
declines zero-sized extents, both because NumPy's own dispatcher excludes them from its blas routes
and because NumSharp's per-element loop currently throws on them — answering would make `np.matmul`
succeed or fail depending on whether this package happens to be referenced.

`TryMatMulBatched` is a **default interface method**, so a backend written against the original two
members keeps compiling and running with no edit; the engine just falls back to per-element calls.
Adding a whole new dtype family needs no interface change at all — a third-party Complex128 backend
was written against this exact surface, batched path included, without touching Core.

**A backend does not need access to NumSharp's internals.** Operands are read as
`(T*)a.GetData().Address + a.Shape.Offset` with `a.Shape.Strides` — all public, all element units.
(`a.GetData<T>().Address` is the trap: for a non-contiguous view it hands back a *densified copy*,
so pairing it with `Shape.Strides` reads outside the buffer and quietly returns wrong numbers on
exactly the sliced and transposed operands a matrix product sees most.)

**Why a property and not an engine subclass** (which is what the first cut did): an `NDArray` binds
its engine when it is CONSTRUCTED, and `np.dot(a, b)` dispatches on `a.TensorEngine`. Swapping the
engine therefore only reaches arrays created afterwards — a silent footgun. Engines are cached
singletons, so setting a property on one takes effect for arrays that already exist. The only other
Core change is mechanical: `MultiplyMatrix` and `BatchedMatmul` went from `protected static` to
instance methods, because a static cannot read the engine's property.

## 4. Why BOTH dispatchers had to be ported

`np.dot` and `np.matmul` are **not the same C code**, and they do not always agree. Probed on
2.4.2, they are bit-identical across every layout tried (C, F, negative stride, strided, sliced,
transposed, syrk) — **except** when an operand is not blasable in a special-shape route: a
stride-2 matrix times a vector gives `np.dot` a gemv-on-a-copy and `np.matmul` the portable loop,
and 278/300 elements differ. NumSharp funnels both APIs into one engine, so both dispatchers are
mirrored and each API keeps its own.

### The five routes of `@TYPE@_matmul` (all ported)

| Condition | Route |
|---|---|
| any zero dim, or a dim > `BLAS_MAXSIZE` | `matmul_inner_noblas` (portable C loop) |
| `dm == 1 && dp == 1` | `@TYPE@_dot` → chunked `cblas_?dot` summed in a **double** accumulator |
| `dn == 1 && (dm == 1 \|\| dp == 1)` | `matmul_inner_noblas` |
| `dm == 1` (+ blasable) / `dp == 1` (+ blasable) | `@TYPE@_gemv`, operands swapped for the vector·matrix direction |
| otherwise | `cblas_?gemm`, **or `cblas_?syrk`** when an operand is multiplied by its own transpose (same data pointer, mirrored triangle), copying any non-blasable operand into a temp first (with NumPy's `i1_transpose`/`i2_transpose` choice by `\|stride\|` comparison, and the `matmul(a,b,o) == matmul(bᵀ,aᵀ,oᵀ)` equivalence when the output is transposed) |

`cblas_matrixproduct` adds its own shape classification (`_scalar` / `_column` / `_row` /
`_matrix`), its `_bad_strides` up-front copy, the `?axpy` scalar-multiply branch, and its own
syrk test. The N-D `np.dot` tail is the one route that is **not** gemm at all: NumPy walks every
(row, column-plane) pair and calls the dtype's `dotfunc`, so the port does too.

### Units

NumPy's strides are in **bytes**, NumSharp's in **elements**. Every ported predicate works in
elements — the same logic with `itemsize == 1`, which makes NumPy's `stride % itemsize` guards
vacuous (an element stride is aligned by construction).

## 5. Scope

**Covered:** `float32` and `float64`; every route above; 1-D promotion; stacked/batched `matmul`;
N-D `dot`; mixed `float32 × float64` (cast to the common type first, exactly as NumPy's
`PyArray_FromAny` does); zero-sized extents.

**Not covered (falls through to the managed kernel):** every other dtype. Integer and bool products are
bit-exact by construction anyway — modular integer addition is associative, so summation order
cannot change the result. `Complex` (`cblas_zgemm`) and `Half` are the only real gaps.

## 6. Results

Acceptance harness: 110 hand-built cases (NumPy saves backing arrays + a view recipe both stacks
apply; NumSharp replays and bit-compares) and the 342-case committed corpus tier.

| Gate | Managed kernels | `IBlasBackend` installed |
|---|---|---|
| 110-case acceptance harness | 20 bit-exact | **110 bit-exact** |
| `matmul_parity` corpus tier (342) | 294 divergent | **342 bit-exact** |
| MLP twin, Gate 2 (each op recomputed on NumPy's own inputs) | 4 `dot` rows divergent | **all 4 bit-exact** |
| MLP twin, Gate 4 (50 full training steps) | 73 % of `w1` bytes differ | **0 % — byte-identical weights** |

Gate 4 needs `np.exp` parity too (`NDFloatMath.Exp`, the parallel session's port); with both in
place a 50-step float32 MLP train produces byte-identical `w1`/`w2`/`b1`/`b2`.

### Performance (Release, best-of-rounds, ms)

| shape | dtype | native | parity | parity (1 thread) |
|---|---|---|---|---|
| 128×784×128 | f32 | 0.436 | **0.233** | 0.227 |
| 256³ | f32 | 0.604 | **0.296** | 0.291 |
| 1024³ | f32 | 38.77 | **15.97** | 16.79 |
| 2048³ | f32 | 282.2 | **118.3** | 124.3 |
| 128×784×128 | f64 | 0.966 | **0.435** | 0.426 |
| 1024³ | f64 | 134.5 | **33.8** | 32.3 |
| 2048³ | f64 | 3860 | **292.6** | 310.6 |

The BLAS path is 1.7×–13× *faster* than NumSharp's own GEMM (it is OpenBLAS, after all). NumSharp
ships no native binaries: the package binds whatever BLAS the consumer already has, because bundling
one would make results depend on which copy won the loader race, and parity is a claim about a
specific binary.

### Stacked products, and why the seam is per-STACK and not per-matrix

The granularity of an entry point is a performance decision, and this one was wrong at first. With
only `TryMatMul2D`, a stacked matmul called it once per batch element and the backend rebuilt
NumPy's `MatmulPlan` — route decision plus, when an operand is not blasable, a native scratch
`malloc` — every time. NumPy does that work once, outside its gufunc's outer loop, because the
trailing strides cannot vary within a stack. Measured (NPY/NS convention, higher = BLAS faster,
Release, best-of-9):

| workload | managed ms | BLAS, per element | BLAS, hoisted | ratio before → after |
|---|---|---|---|---|
| 2000 × (8,8) f32 | 2.34 | 2.71 | **0.361** | **0.80× → 6.47×** |
| 500 × (32,32) f32 | 1.93 | 1.39 | **0.850** | 1.28× → **2.27×** |
| 8 × (256,256) f32 | 4.78 | 2.29 | **2.33** | 1.58× → **2.05×** |
| single (512,784)@(784,512) f32 | 6.50 | 2.89 | 2.89 | 2.25× (unaffected) |
| 2000 × (8,8) **int32** (declined) | 2.51 | 2.65 | 2.65 | 0.95× — the seam itself is free |

So per-element setup did not merely erode the advantage on many-small-matrix workloads, it inverted
it: the external OpenBLAS was **20 % slower than NumSharp's own managed GEMM**. `TryMatMulBatched`
removes that, and it is a pure performance change — the plan is a function of the trailing strides
and dims alone, so both routes reach the same `MatmulCore` call with the same pointers. Verified by
A/B-ing them through a decorator that declines the batched entry point: **25/25 stacked products
bit-identical** across broadcast, transposed, sliced, negative-stride, 1-D-promoted, degenerate-core
(`dm`/`dn`/`dp` = 1), 4-D and 5-D cases, f32 and f64.

## 7. The host pin

This is the first corpus tier whose expected bytes are **not** reproducible everywhere. It ships
`Fuzz/corpus/matmul_parity.host.jsonl` recording the numpy version, the BLAS file name, the
OpenBLAS build string, the dispatched core name and the thread count; `MatmulParityPin` compares
before judging and reports `Assert.Inconclusive` — never red — when the host differs. A machine
without that wheel has nothing to be wrong about. Same precedent as the MSVC-pinned float→uint
cast kernels (see `Fuzz/README.md` → "Host-dependent values").

Regenerate for your own host with:

```bash
python test/oracle/gen_oracle.py matmul_parity   # needs numpy==2.4.2
dotnet build                                     # copies the corpus to the test output
dotnet test --filter "FullyQualifiedName~FuzzCorpusTests.MatmulParity"
```

The test assembly suppresses the package's module-load auto-install
(`Fuzz/BlasEngineAutoInstallGuard.cs` sets `NUMSHARP_BLAS_AUTOINSTALL=0`) and installs the backend
per-test instead: every other tier asserts NumSharp's OWN kernels, and an install triggered at
whatever moment the first parity type is touched would silently change what the tests around it
measured.

## 8. Traps

- **The thread count is part of the answer.** `Blas.Enable(..., threads: n)` is not a
  performance knob, it is a correctness knob. The gate pins it to the value recorded at generation
  time so a host whose default differs still matches.
- **`a @ a.T` is not a gemm.** Both dispatchers detect the shared data pointer and call `?syrk`,
  computing the upper triangle and mirroring it. The corpus cannot express two operands sharing a
  buffer, so those cases carry ONE operand and the op names `dot_aat` / `dot_ata` /
  `matmul_aat` / `matmul_ata` form the product in `OpRegistry`.
- **Layout does not change the bits for mat @ mat, but it does for the special routes.** For the
  same values, a C, F, strided or sliced operand gives gemm the same answer — but a `(1,500)`
  vector with stride 2 changes gemv's `incX` and the result with it. That is exactly why the port
  passes real strides through instead of normalizing operands.
- **`?dot` accumulates in double.** NumPy's `FLOAT_dot` sums the chunked `cblas_sdot` results into
  a `double` "for stability" before casting back to float — the row·column route is therefore
  *more* accurate than the gemm route, and a parity implementation must reproduce that too.

## 9. What the design review changed (2026-07-26)

The seam itself survived every attempt to break it; three defects in the plumbing around it did not.
Each is now pinned by a test in `MatmulParityBackendTests`.

- **Never test-then-call a settable property.** Both hook sites read `Blas` twice —
  `if (Blas != null && Blas.TryDot(...))`. `Blas` is settable from any thread, so a concurrent
  `Blas.Disable()` landing between the two reads is a `NullReferenceException` inside `np.dot`:
  measured **31 252 of 1 562 291 products (~2 %)** under a thread flipping the property while eight
  workers multiplied. Reading it into a local first fixes it (re-measured 0 of 1 553 695). A
  `volatile` field would not have helped — reference reads are already atomic; the bug was the
  *second* read, not a torn one.
- **A failed `Enable` must change nothing.** `CBlasNative.Load` used to `Unload()` the working
  library before it knew the new candidate was good, and `Blas.Enable` left the backend installed
  when the load then threw. The result was the worst failure mode a parity feature has: `Blas.Enabled`
  reporting **true** while every product had silently reverted to the managed kernel — no exception
  at the call site, and an answer that still looks like a matrix product. Load now binds the
  replacement fully before touching the incumbent, and `Enabled` means installed **and** bound.
  The same commit-before-validate pattern in `Bind` was worse and is fixed the same way: it wrote
  `_handle` before resolving the remaining nine symbols, so a library exporting `cblas_sgemm` but
  missing, say, `cblas_dsyrk` would leave `IsLoaded` true over a handle the caller had just freed —
  a later `Enable()` would skip loading and call into it, and a later `Load` would free it twice.
- **`TryEnable` catches everything, deliberately.** It runs from a `[ModuleInitializer]`, where an
  escaping exception is not a failed opt-in but a `TypeInitializationException` on every later touch
  of the assembly — i.e. merely *referencing* the package would break the app, the one thing the
  design promises it cannot do. Discovery walks `PATH` and the filesystem, so the reachable failures
  were never limited to the three loader exceptions it used to catch; an unreadable directory alone
  raises `UnauthorizedAccessException` (now also guarded at the `Directory.GetFiles` call).

**Verified sound, with the probe that proves it** — do not re-derive:

| Claim | How it was probed | Result |
|---|---|---|
| Every `NDArray` resolves to the one cached engine, so the property reaches arrays that already exist | 38 construction / operation / IO paths asserted `ReferenceEquals(arr.TensorEngine, BackendFactory.GetEngine())` | 38/38. The only escape is the explicit public `arr.TensorEngine = new DefaultEngine()`; even then `a+b` results land back on the singleton |
| Installing a backend changes bits and speed, nothing else | 52 calls (valid shapes, every layout, mixed dtypes, and 14 invalid ones) compared shape + dtype + exception type + exception message, backend on vs off | 52/52 identical. The `Dot` hook sitting *above* Core's own validation is safe because every misaligned route returns false |
| A backend needs `InternalsVisibleTo` | wrote a working `IBlasBackend` in an assembly that is **not** a Core friend | Refuted — it compiles, installs and computes correctly, strided views included |
| The interface would need widening for the uncovered dtypes | added a Complex128 backend (what NumPy sends to `zgemm`), non-friend, batched path included | Zero change to `IBlasBackend`, `TensorEngine` or any hook |

**Found while probing, NOT a BLAS defect and still open:** `np.matmul` throws on any ≥3-D operand
with a zero-sized dimension — `matmul(zeros((0,3,4)), zeros((0,4,5)))` raises
`InvalidOperationException` where NumPy gives shape `(0,3,5)`, and the zero-K/M/N variants raise
`IndexOutOfRangeException`. `np.dot` and 2-D `np.matmul` are both correct, so it is specific to
`BatchedMatmul` (its `ValueCoordinatesIncrementor` rejects an empty iteration shape, and
`BroadcastStackDims` broadcasts `0` against `1` to **1** instead of `0`).
