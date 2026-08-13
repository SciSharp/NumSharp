# `NumSharp.Interop.OpenBLAS` — byte-identical `np.dot` / `np.matmul`

> **✅ IMPLEMENTED (2026-07-25).** Optional package `src/NumSharp.Interop.OpenBLAS/`
> (`OpenBlasBackend : IBlasBackend`, `Blas`, `CBlasNative`, `BlasParity{,.Matmul,.Dot,.Entry}`),
> installed through the one seam Core exposes — the engine's settable `TensorEngine.Blas`
> property — gate `FuzzCorpusTests.MatmulParity` over the 342-case `matmul_parity` corpus tier.
> Measured: **342/342 bit-exact with the backend installed, 294/342 divergent without it.**
>
> **NumSharp.Core stays 100 % managed C#** — it contains no BLAS code, no P/Invoke and no native
> dependency. With this package absent every matrix product is NumSharp's own SIMD GEMM.
>
> **UPDATE (2026-07-26): the package now BUNDLES the binary.** It ships the prebuilt OpenBLAS
> artifacts NumPy itself pins (scipy-openblas 0.3.31.22.0 — verified byte-identical to numpy 2.4.2's)
> as per-RID NuGet runtime assets for 8 platforms, so bit-parity no longer requires a Python
> installation. §6 explains why this is not a reversal of the "ships no native binaries" decision it
> replaces, and §6.1 documents the discovery order.

Status date: 2026-07-26 · Branch: `journey3` · Issue:
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
dotnet add package NumSharp.Interop.OpenBLAS
```

```csharp
using NumSharp.Interop.OpenBLAS;

// Referencing the package IS the opt-in: a [ModuleInitializer] installs the backend if a CBLAS
// library can be found, and silently does nothing if one cannot.
Blas.Enable(@"…\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);  // pin, for byte parity
Blas.Enabled;                                            // bool
Blas.Info;                                               // path + symbol scheme + int width + threads + build string
Blas.Disable();                                          // back to NumSharp's managed SIMD GEMM
```

Discovery order (this is the initial cut; the current full order — the bundled asset,
`NUMSHARP_OPENBLAS_PATH` and the system scan — is in §6.1): explicit path/directory →
`NUMSHARP_PARITY_BLAS` → the `numpy.libs` folder of any python on `PATH`
(plus `VIRTUAL_ENV` / `CONDA_PREFIX` / `PYTHONHOME`) → bare loader names.
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
declines zero-sized extents, because NumPy's own dispatcher excludes them from its blas routes (the
`any_zero_dim` arm of `@TYPE@_matmul` forces `matmul_inner_noblas`) — answering would make the port
unfaithful. The engine settles them itself, in the same place NumPy does; see §9.

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

### 5.1 The rest of the surface NumPy routes through this binary (2026-07-27)

`np.dot`/`np.matmul` are not everything NumPy sends to the library this package bundles. The API
shell for the remainder now exists, so the seam it plugs into is fixed even where the numerics are
not — `IBlasBackend.LinearAlgebra.cs` adds fifteen `Try*` members, all of them DEFAULT
implementations returning false so `OpenBlasBackend` (and any third-party backend) compiles
untouched. `TensorEngine.LinearAlgebra.cs` reads `Blas` into a local, tries it, and then does one of
two different things:

| Group | Members | No backend installed |
|---|---|---|
| Products (CBLAS) | `TryInner`, `TryVdot`, `TryVecdot`, `TryMatvec`, `TryVecmat` | **Managed kernels compute it.** The invariant this document states elsewhere still holds: a backend changes WHICH implementation runs, never WHETHER an answer exists. |
| Factorisations (LAPACK) | `TryCholesky`, `TryDet`, `TrySlogdet`, `TryEig`, `TryEigh`, `TryInv`, `TryLstsq`, `TryQr`, `TrySolve`, `TrySvd` | **`NotSupportedException`.** NumSharp.Core ships no managed LU, QR, SVD or eigensolver, so there is nothing to fall back to. |

That second row is the one genuine exception to the fallback invariant, and it is a property of the
problem rather than of the design: a portable GEMM is a hundred lines, a portable `gesdd` is not.
The exception message names the NumPy API, the LAPACK routine NumPy uses and the seam member to
implement, so it reads as an install instruction rather than a dead end.

LAPACK is worth routing through `TensorEngine.Blas` rather than a second property because it lives
**inside the very binary already bundled** — scipy-openblas ships `gesv`/`getrf`/`potrf`/`geev`/
`syevd`/`heevd`/`gesdd`/`geqrf`/`orgqr`/`ungqr`/`gelsd` alongside the BLAS symbols, so a backend that
has loaded the library for `sgemm` already has the handle for everything else.

**Signatures are gated by reflection, not by review.** `LinAlgSignatureParityTests` asserts every
function has an overload whose parameter NAMES appear in NumPy's ORDER, plus the 18 defaults NumPy
states concretely, plus that no `np.linalg` member is missing — a renamed or reordered parameter
compiles and passes every value test while silently breaking named-argument calls ported from
Python. The gufuncs carry NumPy's `out`/`axes`/`axis`/`keepdims`/`dtype`; `casting`, `order`,
`subok` and `signature` are deliberately absent, since NumSharp models none of them anywhere in its
ufunc surface.

**Validation is NOT deferred.** Every entry point runs NumPy's own rank, squareness, dtype and
argument checks BEFORE reaching the seam, with verbatim messages
(`LinAlgError("Last 2 dimensions of the array must be square")`,
`TypeError("array type float16 is unsupported in linalg")`, the two `solve` gufunc signatures, …), so
the error contract is gated today by `LinAlgErrorParityTests` and does not move when the numerics
land. Anything reachable without a factorisation is IMPLEMENTED rather than stubbed: `np.inner`,
`np.vdot`, `np.vecdot`, `np.matvec`, `np.vecmat`, `np.tensordot`, `np.linalg.multi_dot`,
`matrix_power` at a non-negative exponent, the Array-API forms, and every `np.linalg.norm` order
except the three defined by singular values (matrix `2`, `-2`, `'nuc'`).

**`np.einsum` is the one entry here that is NOT waiting on a backend**, and its shell goes further
than the rest because its validation IS a language. `EinsumSubscripts.cs` ports
`PyArray_EinsteinSum`'s parsing block plus `parse_operand_subscripts`/`parse_output_subscripts` from
`einsum.cpp`: both spellings (subscripts string and sublist), ellipsis broadcasting, repeated-label
diagonals, implicit-output inference, and the resolved output SHAPE — which the
`NotSupportedException` then carries, so the parser could be differential-tested against NumPy across
75 expressions without a contraction kernel existing. What is missing is that kernel and a
contraction-path planner, both NumSharp's own work, which is why the seam has no `TryEinsum` and the
message points at `np.tensordot` rather than at a package to install.

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

The BLAS path is 1.7×–13× *faster* than NumSharp's own GEMM (it is OpenBLAS, after all).

### The package now SHIPS the binary (2026-07-26) — and why that is not a reversal

This document used to argue the opposite: *"NumSharp ships no native binaries: the package binds
whatever BLAS the consumer already has, because bundling one would make results depend on which copy
won the loader race, and parity is a claim about a specific binary."*

The objection was to bundling **an arbitrary** OpenBLAS. It dissolves once the bundled one is
**NumPy's own**, which is checkable rather than hopeful:

- NumPy 2.x does not build an OpenBLAS. Its wheels bundle the prebuilt `scipy-openblas32` /
  `scipy-openblas64` artifacts from PyPI (openblas-libs, BSD-3-Clause-Attribution), pinned in
  `requirements/ci_requirements.txt` → `scipy-openblas64==0.3.31.22.0`, consumed by
  `tools/wheels/cibw_before_build.sh`.
- The DLL inside that wheel is **byte-identical** to the one in numpy 2.4.2's `numpy.libs/`:
  both sha256 `74a408729250596b0973e69fdd954eea07a70ff527a1dbaccf9ae21247b80377`. NumPy's mangled
  file name *is* that hash's first 32 hex chars — delvewheel names the file by its content.
- So "bundle a binary" and "call the same binary NumPy calls" became the same act. Verified
  end-to-end: 12/12 products bit-exact against numpy 2.4.2 through the bundled asset (the same 12
  were 8-divergent on the managed kernels).

There is no loader race, because the order is fixed and documented (§6.1). And the "specific binary"
requirement is now *stronger* than before: the corpus host pin compares the loaded library by
**content hash**, not by file name.

#### 6.1 Discovery order

| # | Source | Notes |
|---|---|---|
| 1 | explicit path / `NUMSHARP_PARITY_BLAS` | **binding** — never substituted, not even by the bundled copy |
| 2 | **bundled `runtimes/<rid>/native/`** | `NUMSHARP_BLAS_BUNDLED=0` drops it |
| 3 | `NUMSHARP_OPENBLAS_PATH` | **additive** file(s)/dir(s) — "also look here", non-binding (the opposite of row 1) |
| 4 | `numpy.libs` of any python on `PATH` | plus `VIRTUAL_ENV` / `CONDA_PREFIX` / `PYTHONHOME` |
| 5 | machine-wide OpenBLAS (apt / brew / MacPorts / conda / vcpkg / source) | **not** byte-parity with NumPy — a different build — so it ranks below the parity sources; honours `OPENBLAS_HOME` / `VCPKG_ROOT` / `CONDA_PREFIX` |
| 6 | bare loader names | `libscipy_openblas64_`, `libscipy_openblas`, `scipy_openblas`, `libopenblas`, `libblas`, … |

Bundled beats a discovered `numpy.libs` **on purpose**: a library's numeric output must not change
because an unrelated `pip install` ran, and an app with no python must not get a different backend
from one that has it. Matching some *other* numpy stays possible and is now an explicit act (name it).

##### Discovery widened: additive path + system scan (2026-08-13)

Two tiers were inserted between "bundled" and "bare names" (rows 3 and 5 above), and neither can
disturb parity: both rank *below* the bundled asset and `numpy.libs`, and the binding row 1 still
bypasses discovery entirely. Confirmed by 30/30 `MatmulParityBackendTests` green after the change.

- **`NUMSHARP_OPENBLAS_PATH` (row 3)** — the additive, non-binding sibling of `NUMSHARP_PARITY_BLAS`.
  Point it at a local build or an unpacked release to *add* a search location without displacing the
  bundled default; to *force* one specific binary, `NUMSHARP_PARITY_BLAS` remains the binding knob.
- **System scan (row 5)** — the conventional install prefixes of the package managers OpenBLAS's own
  install docs list: apt multiarch (`/usr/lib/<triplet>`), a Homebrew keg
  (`/opt/homebrew/opt/openblas/lib`), MacPorts, a conda tree's `lib` / `Library\bin`, a vcpkg
  triplet's `bin`, the `make PREFIX=/opt/OpenBLAS` default, `/usr/lib64` — plus the `OPENBLAS_HOME` /
  `OPENBLAS_ROOT` / `VCPKG_ROOT` / `CONDA_PREFIX` roots. It finds a machine-wide OpenBLAS the bare
  names miss (a versioned soname like `libopenblas.so.0`, or a lib off the default loader path), but
  it is **not** NumPy-parity, so it is the last resort before bare names.

Both scipy distributions bind: **scipy-openblas64** (ILP64, `scipy_cblas_*64_`) and
**scipy-openblas32** (LP64, plain `scipy_cblas_*`). The bare-name list gained the 32-bit spellings
(`libscipy_openblas`, `scipy_openblas`) alongside the 64-bit ones — and since scipy-openblas32 is the
only build PyPI publishes for 32-bit x86 (`win32` / `i686`), a 32-bit target can only be served by it.

RIDs: `win-x64`, `win-arm64`, `linux-{x64,arm64,musl-x64,musl-arm64}`, `osx-{x64,arm64}` — 62 MB
packed. Binaries are gitignored; `tools/openblas-manifest.json` is the checked-in pin and
`tools/fetch_openblas.py` verifies **two** hashes per RID (the wheel's, then the extracted library's).

**win-arm64 is `scipy-openblas32`, not 64** — because NumPy's own build selects the LP64 distribution
there. It exports `scipy_cblas_sgemm` (prefix, no suffix, 32-bit BLAS int), a scheme `Bind` did
**not** have: binding that library would have failed outright with "exports no recognizable
cblas_sgemm symbol". Added.

### The third variable: DYNAMIC_ARCH (probed 2026-07-26)

Bundling pins the library build; it does **not** pin the answer. OpenBLAS is built `DYNAMIC_ARCH`
and selects a micro-kernel from the running CPU. Measured by forcing the choice on one host, same
binary, `OPENBLAS_NUM_THREADS=1`, `(512,784)@(784,512)` f32:

| `OPENBLAS_CORETYPE` | dispatched | result |
|---|---|---|
| *(unset)* | Haswell | `b9ea5057dc59fd41…` |
| `Haswell` | Haswell | `b9ea5057dc59fd41…` — identical to unset |
| `Nehalem` | Nehalem | `b6bb0f5873bc4721…` |
| `Sandybridge` | Sandybridge | `1903e357791df3c9…` |
| `Core2` | Katmai | `0da2143835f74a56…` |
| `SkylakeX` | — | **process killed: Illegal instruction** |

Four facts fall out, all load-bearing:

1. **The variable works**, and pinning it to the corpus's kernel reproduces the corpus's bits on any
   CPU that can run that kernel — `Blas.ParityCoreType` ("Haswell") covers every x86-64 with AVX2+FMA.
2. **It is not safe to set blindly.** It *overrides* OpenBLAS' capability detection, so naming a
   kernel above the CPU's ISA executes an unsupported instruction. That is not catchable, so
   `Enable` refuses the combinations it can recognise up front.
3. **It must be set before the library loads.** OpenBLAS reads it once while initialising and never
   again; setting it afterwards is a silent no-op (verified: corename unchanged).
4. **Thread count is independent of it** — with the core pinned to Haswell, 1/2/4/24 threads still
   give four different answers. Both must be pinned, or neither.

It is therefore **opt-in, not the default**. The default must match the NumPy on the same machine,
and that NumPy dispatches by CPU as well — pinning unilaterally would *create* divergence on any host
newer than Haswell.

#### `Environment.SetEnvironmentVariable` does not reach a native `getenv`

The first implementation appeared to work and did nothing. .NET keeps its own environment table, so
the managed getter reads the new value back while the C runtime the native library links against
never sees it: setting `OPENBLAS_CORETYPE=Nehalem` that way and then loading OpenBLAS still
dispatched **Haswell**. Going through the CRT (`ucrtbase!_putenv_s`, `libc!setenv`) dispatched
Nehalem as asked. Windows' own `SetEnvironmentVariableW` does **not** work either — it updates the
process environment block, not the CRT copy `getenv` reads.

#### A silently-ignored pin, found by its own test

`Blas.Enable` skips `CBlasNative.Load` when a library is already loaded — so a `coreType` passed on
any call after the first was **discarded without a word**, leaving the caller believing the kernel
was pinned while the bits were whatever the CPU chose. Now it either already holds (idempotent) or
throws. Same failure class as the two the previous review caught: the parity feature reporting
success while quietly not delivering parity.

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
`Fuzz/corpus/matmul_parity.host.jsonl` recording the numpy version, the BLAS file name **and its
SHA-256**, the OpenBLAS build string, the dispatched core name and the thread count;
`MatmulParityPin` compares before judging and reports `Assert.Inconclusive` — never red — when the
host differs. A machine without that wheel has nothing to be wrong about. Same precedent as the
MSVC-pinned float→uint cast kernels (see `Fuzz/README.md` → "Host-dependent values").

**The library is identified by CONTENT, not by file name** (2026-07-26). Bundling forced this and
made it strictly better. A file name is a poor proxy for a binary in both directions: pip's
`delvewheel` mangles it per build (`libscipy_openblas64_-74a4….dll`), while the bundled copy of the
very same bytes is plainly `libscipy_openblas64_.dll`. Matching on the name alone therefore skipped
the whole 342-case tier on a host that was genuinely bit-identical — observed the moment bundled
discovery landed — and would equally have *accepted* a differently-built library that happened to
share a name. The pin now carries `blas_library_sha256`, and the gate hashes what it actually
loaded; the name check remains only as a fallback for corpora generated before the field existed.
Note this is the opposite of weakening the pin to widen coverage: the excuse got narrower, the
assertion stronger.

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

**Found while probing, NOT a BLAS defect — since FIXED:** `np.matmul` threw on any ≥3-D operand
with a zero-sized dimension. `np.dot` and 2-D `np.matmul` were both already correct, so it was
specific to `BatchedMatmul`, and it took **three** independent fixes, one per failure mode:

| Cause | Symptom | Fix |
|---|---|---|
| `BroadcastStackDims` used `Math.Max`, so a stack `0` broadcast against `1` gave **1** | `matmul((0,3,4), (4,5))` → `IncorrectShapeException`, because a `(0,3,4)` operand was then asked to broadcast to `(1,3,4)` | take the extent that **isn't 1** — `np.broadcast_shapes((0,), (1,))` is `(0,)` |
| the batch odometer is constructed before the loop and rejects an empty iteration space | `matmul((0,3,4), (0,4,5))` → `InvalidOperationException` | short-circuit a degenerate stack before building it |
| a sub-view of a zero-sized operand cannot be taken — `Shape.GetSubshape` bounds-checks the offset against a buffer size of **0**, so even a valid index throws | the zero-K/M/N variants → `IndexOutOfRangeException` | same short-circuit; the loop never runs for these |

The shortcut is the port of NumPy's own structure: `any_zero_dim` forces `@TYPE@_matmul` off every
blas route into `matmul_inner_noblas`, where the answer falls out of the loop bounds. Two outcomes,
neither needing the batch loop — an **empty result** (a zero stack dim, or `n == 0` / `m == 0`)
writes nothing, and `k == 0` leaves a **non-empty** result whose every entry is an **empty sum**.
That sum is exactly zero because NumPy stores `0` into the output cell *before* its zero-trip
accumulation loop, so `nan(2,3,0) @ inf(2,0,5)` is `+0.0`, not `nan` — probed, and pinned by a test
that repeats the case 400 times against deliberately poisoned memory (nothing else writes that
buffer, so the result is only as correct as the allocation is zeroed).

`TryMatMulBatched` still **declines** zero-sized extents, and should: NumPy excludes them from its
own blas routes, so answering would make the port unfaithful. The seam test no longer merely asserts
that the outcome is *identical* backend-on vs backend-off — that held while both sides threw — it
now pins NumPy's shape and, for `k == 0`, that every element is bit-exact `0.0f`.
Gate: 14 tests in `LinearAlgebra/np.matmul.ZeroSized.Test.cs` plus 280 oracle cases in the `matmul`
tier (`gen_oracle.gen_matmul_zerodim`, 35 shape pairs × 4 dtypes × C/F).

## 10. `float16` accumulates in float32 — and which dtypes NumPy actually sends to BLAS (2026-07-26)

A second differential pass (301 byte-exact cases: Python records the operand bytes and NumPy's
answer, C# replays and bit-compares) turned up **one portable wrong answer** and made the scope of
§2's "no portable algorithm can fix it" claim precise. The dividing line is one row of
`matmul.c.src`:

```
 * #TYPE = FLOAT, DOUBLE, LONGDOUBLE, HALF, CFLOAT, CDOUBLE, CLONGDOUBLE, …
 * #USEBLAS =  1,      1,          0,    0,      1,       1,           0, 0*13#
```

**float32, float64, complex64 and complex128 go to cblas. `float16` does not** — so for half,
`matmul_inner_noblas` *is* NumPy's shipping implementation, and it is reproducible exactly on any
host with no native dependency.

### Fixed: half products accumulated in half

NumPy's half loop declares `float sum = 0`, accumulates every product in **float32**, and narrows
once with `npy_float_to_half(sum)`; `HALF_dot` does the same with `float tmp = 0.`. NumSharp
accumulated in `Half`, which does not merely lose a ulp — it **saturates**, because half's spacing
above 2048 is 2, so `2048 + 1 == 2048`:

| | NumSharp (before) | NumPy |
|---|---|---|
| `ones(3000)·ones(3000)` | **2048** | 3000 |
| `ones(4096)·ones(4096)` | **2048** | 4096 |
| `2s(3000)·2s(3000)` | **8190** | 12000 |
| `[1, 2⁻¹¹×9] @ ones(10)` | `0x3c00` | `0x3c04` |

Visible from `k = 10` with ordinary values, and unconditional — no BLAS backend serves half, so
nothing masked it. Three kernels needed the wider accumulator: `MatMulStridedSame<Half>` (matmul
2-D/1-D/batched and 2-D dot), `DotGeneric<Half>` (the fused 1-D inner product), and
`MatMulStridedMixed<Half>` (half against bool/int8/uint8, which still promote to float16). The
accumulator is a float **row** rather than NumPy's scalar, purely to keep the cache-friendly
j-inner loop — for a given output cell the additions still run over `k` ascending, which is all
that fixes the bits. `np.sum(float16)` was already correct and is untouched.

The `matmul` tier could never have caught this: it carries float16, but every shape in it has
`k ≤ 4`, where a half accumulator and a float32 one agree exactly. Gate: 288 new oracle cases
(`gen_oracle.gen_matmul_half_depth` — `k ∈ {8, 64, 256, 1024, 3000, 4096}` × 4 value classes ×
C/F layouts × matmul/dot/1-D/batched/mixed) plus 8 tests in
`LinearAlgebra/np.matmul.HalfAccumulator.Test.cs`.

### Not fixed, and not fixable here: complex64 / complex128

`CFLOAT` and `CDOUBLE` are BLAS dtypes, so NumPy's complex matrix products are **zgemm's bits**,
exactly as float32/float64 are sgemm's. Measured: **40/40** complex128 cases diverge at every
contraction depth including `k = 4` (they agree only on small exact integers, where summation order
cannot matter), and the inf edge differs too — `[-inf, 1] @ [[1],[1]]` is `(-inf, nan)` in NumSharp
against NumPy's `(nan, nan)`.

This is §2's class, not a defect in the managed kernel — but §3's table scopes the backend to
float32/float64 and justifies the fall-through only for integers ("modular addition is
associative"), which leaves complex silently in the divergent bucket. `NumSharp.Interop.OpenBLAS`
serves no complex dtype, so installing it does not close this. The seam already supports it: §8's
probe wrote a third-party Complex128 backend against `IBlasBackend` untouched, batched path
included. Closing it is a package-side job — add `cblas_zgemm`/`cblas_cgemm` routes — not a Core one.

### Withdrawn: the `np.dot(0-d, …)` signed zero

Recorded here because it looked like a third bug and is not. `np.dot(-2.5, B)` where `B` holds a
`+0.0` returns `+0.0` in NumPy but `-0.0` from a plain multiply. NumPy's 0-d branch of
`PyArray_MatrixProduct2` **is** a plain multiply — it just sits *below* the `HAVE_CBLAS` shortcut,
so only the four cblas dtypes reach `cblas_matrixproduct` and come back `+0.0`. Probed: NumPy gives
`-0.0` for `float16` and `longdouble`, which is exactly what NumSharp gives for every dtype. So the
managed answer is NumPy's own non-BLAS answer, and with the package installed NumSharp already
returns `+0.0` for float32/float64 (its `TryDot2D` ports the scalar branch). Only the complex cell
remains, and it folds into the paragraph above. "Fixing" it by adding zero would have *broken*
float16 parity.
