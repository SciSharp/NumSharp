# `np.parity_matmul` — byte-identical `np.dot` / `np.matmul`

> **✅ IMPLEMENTED (2026-07-25).** Opt-in backend in
> `src/NumSharp.Core/Backends/Kernels/Blas/` (`CBlasNative`, `BlasParity{,.Matmul,.Dot,.Entry}`),
> public switch `np.parity_matmul(...)` in `APIs/np.parity_matmul.cs`, hooks in
> `DefaultEngine.Dot` and `DefaultEngine.MultiplyMatrix`, gate
> `FuzzCorpusTests.MatmulParity` over the 342-case `matmul_parity` corpus tier.
> Measured: **342/342 bit-exact with the backend on, 294/342 divergent with it off.**

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

An **opt-in** backend that P/Invokes the *very same BLAS binary* the target NumPy calls, driven by
a **route-for-route port of NumPy's two matrix-product dispatchers**. Same binary + same route +
same flags ⇒ same bits by construction.

```csharp
np.parity_matmul(true);                                  // auto-discover numpy's OpenBLAS
np.parity_matmul(true, @"…\numpy.libs\libscipy_openblas64_-74a4….dll", threads: 1);
np.parity_matmul(false);                                 // back to NumSharp's SIMD GEMM (default)
np.parity_matmul_enabled;                                // bool
np.parity_matmul_info();                                 // path + symbol scheme + int width + threads + build string
```

Discovery order: explicit path/directory → `NUMSHARP_PARITY_BLAS` → the `numpy.libs` folder of any
python on `PATH` (plus `VIRTUAL_ENV` / `CONDA_PREFIX` / `PYTHONHOME`) → bare loader names.
NumPy's wheels rename their symbols `scipy_cblas_sgemm64_`-style and use **64-bit BLAS integers**;
a stock LP64 `cblas_sgemm` is also bound, with the integer width marshalled per call
(`CBlasNative.IsIlp64`).

### Nothing else changes

NumSharp's GEBP/FMA SIMD GEMM is untouched and remains the default. The two hooks are single
`if (…TryX(…)) return …;` lines that only fire when the switch is on:

| Hook | Mirrors |
|---|---|
| `DefaultEngine.Dot` (top) | `PyArray_MatrixProduct2` → `cblas_matrixproduct` (ndim ≤ 2) + the `dotfunc` iterator tail (ndim > 2) |
| `DefaultEngine.MultiplyMatrix` (before the SIMD paths) | `@TYPE@_matmul`, the gufunc behind `np.matmul` and `@` — also every batch element of a stacked matmul |

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

**Not covered (silently keeps the native path):** every other dtype. Integer and bool products are
bit-exact by construction anyway — modular integer addition is associative, so summation order
cannot change the result. `Complex` (`cblas_zgemm`) and `Half` are the only real gaps.

## 6. Results

Acceptance harness: 110 hand-built cases (NumPy saves backing arrays + a view recipe both stacks
apply; NumSharp replays and bit-compares) and the 342-case committed corpus tier.

| Gate | Backend off | Backend on |
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

The parity path is 1.7×–13× *faster* than NumSharp's own GEMM (it is OpenBLAS, after all), so
turning it on costs nothing but a native dependency. It stays opt-in regardless: NumSharp ships no
native binaries, and a parity backend that silently loaded whatever BLAS it found would make
results depend on the machine's Python installation.

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

## 8. Traps

- **The thread count is part of the answer.** `parity_matmul(..., threads: n)` is not a
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
