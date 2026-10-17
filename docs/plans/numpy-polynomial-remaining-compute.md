# numpy.polynomial — compute routes for the 28 remaining names (discovery, 2026-10-17)

> **Status: discovery only, nothing implemented.** This document decides HOW each name that `numpy.polynomial` still
> lacks in NumSharp should compute: which of the house engines carries its loops, what NumPy's statements are, which
> alternatives were measured and rejected, where the bit-parity hazards are, and what speed is physically reachable.
> It refines units U3 (`polyvalfromroots`), U6, U8, U9, U10 and U11 of [numpy-polynomial.md](numpy-polynomial.md),
> whose §9/§10 already settled the engine for the delivered units. Branch `strings` at `1799d162`.

## 0. The question and the answer

**Question.** For each missing name, which implementation should carry the compute: a fused `np.evaluate` (NDExpr)
pass, a Direct IL whole-array kernel, an NDIter-driven per-chunk kernel (and which NDIter mode), the polynomial
package's own engines, a composition of existing `np.*` functions, the LAPACK backend, or something else?

**Answer, in one line per family** (the full table is §4):

| Family | Names | Route |
|---|---|---|
| Root product | `polyvalfromroots` | U3's step-table IL engine with a third skeleton (an ascending product), NDIter-driven |
| Weights | `chebweight` `legweight` `lagweight` `hermweight` `hermeweight` | one fused `np.evaluate` tree per weight; float16 through the eager house ufuncs |
| Chebyshev points | `chebpts1` `chebpts2` `chebgauss` | `np.arange`/`np.linspace` + one fused or eager unary pass |
| Gauss (eigen) | `leggauss` `laggauss` `hermgauss` `hermegauss` | facade-level composition around `eigvalsh`; the two normed-Hermite recurrences as U3 step-table programs |
| Interpolation | `chebinterpolate` | composition (`chebpts1` → delegate → `chebvander` → `np.dot`) |
| Fits | `polyfit` `chebfit` `legfit` `lagfit` `hermfit` `hermefit` | composition of `pu._fit` around `lstsq`, with a fused streaming norm and C-ordered scaling |
| Classes | `Polynomial` `Chebyshev` `Legendre` `Laguerre` `Hermite` `HermiteE` | an object layer over the unit entry points; no kernels; U3 step tables interpreted over a series algebra for `convert`/`cast` |
| Printing | `set_default_printstyle` `polyutils.format_float` | a process-wide flag; the existing Dragon4 port |

No name needs a new struct kernel, a per-dtype function or a hand-written C# element loop. Every element loop runs
in IL (U3's emitter, NDExpr, the house ufunc kernels) or in LAPACK through `TensorEngine.Blas`.

## 1. The mechanisms on the table

What each one is, and when it wins, from §9/§10 of the main plan and from this discovery:

| Mechanism | What it is | Wins when | Loses when |
|---|---|---|---|
| **Composition** (Pattern 1) | NumPy's statements as `np.*` calls on NDArrays | the statements are few, each one already a NumSharp kernel, and the call is dominated by a kernel or by LAPACK | a statement repeats per coefficient or per step (§9: per-op overhead and temporaries) |
| **Fused `np.evaluate`** (NDExpr) | one NDIter pass, per-node NEP 50 typing, structural cache, 0-d parameters hoisted, axis reductions streamed | an elementwise expression of a few nodes over one or two operands | a node is reused (NDExpr re-emits shared subtrees: Clenshaw grows exponentially, §9 finding 4); float16 (NDExpr's half path is 2–5× slower than the house float16 kernels, §3.2) |
| **Direct IL whole-array kernel** (`DirectILKernelGenerator`) | one call walks the whole array itself | a fresh output in one pass with a structure NDIter adds nothing to (Vandermonde, calculus) | the operands need broadcasting or buffering |
| **NDIter-driven per-chunk IL kernel** (Tier 3A raw / 3B shell / 3C expression; `ForEach`, `ExecuteElementWise`, buffered casts, masked runs, 2-D blocks, `NumPyReduceIterator`) | the iterator walks, the kernel does one chunk | any layout or broadcast must be supported with one kernel | per-call setup (~0.2–1 µs) matters more than the work |
| **U3 step-table emitter** (`ILKernelGenerator.Polynomial*`) | a recurrence per point, written as data in NumPy's source order, typed per node, 4 interleaved chains, lane kinds for every dtype pair, three complex products, weak Python-scalar x, NDIter-driven | a per-point recurrence over a short list of coefficients/roots/steps | the work is not per point |
| **U2 arena** (`NDPolyAlgebra`) | NumPy's Python statements over raw short series in a per-thread bump arena, each statement on its house kernel | many statements over SHORT series (6–51 terms) | long arrays (an NDArray is then cheap relative to the work) |
| **Backend seam** (`TensorEngine.Blas`, `NumSharp.Interop.OpenBLAS`) | LAPACK/BLAS through the bundled scipy-openblas | the math is a factorisation or a product NumPy itself sends to BLAS | — (it is the parity route; there is no managed alternative for `eigvalsh`/`lstsq`) |
| **Scalar replay** (`PolyNumber`/`PyScalar`) | CPython / NumPy-scalar arithmetic for Python-scalar operands | NumPy computes with Python numbers before any ufunc | — |
| **Dragon4 port** (`Backends/Printing`) | NumPy's float formatter | printing | — |

## 2. Method

- **NumPy side:** NumPy 2.4.2 source read for every name (`refs/numpy`, docstrings stripped by `ast` to read the
  statements), plus NumPy 2.4.2 probes: boundary facts (argument kinds, dtypes, verbatim errors), result layouts, and
  the complex product forms of `polyvalfromroots` in exact rational arithmetic.
- **NumSharp side:** every candidate route written against the existing building blocks and run:
  - **routes**: the candidates for one name interleaved round by round in ONE process (relative costs hold under
    load), each byte-compared with the eager composition, which is NumPy's own statement sequence;
  - **parity**: every route's SHA-256 against NumPy's on inputs both sides build bit-identically
    (`sin(arange(n) * 1.2345678901) * amp`: an exact multiply and the same CRT `sin`);
  - **head-to-head**: the recommended route (or the existing kernel standing in for a kernel not yet written, marked
    *proxy*) against NumPy's call, NumPy and NumSharp ALTERNATED twice, min over 25 batches of ≥ 5 ms, both pinned to
    CPUs 2 and 4 (`0x14`), OpenBLAS at one thread on both sides;
  - **floors**: `np.exp`/`np.sin`/`np.cos` float64, `lstsq`, `eigvalsh` and `np.dot` alone, the same way.
- **Host caveat.** Another session's `testhost` held 7–14 cores and Rider 3–5 during most of the run (total CPU
  ~1,850–3,000 % of one core). Absolute microseconds are inflated (by 2–14× in a first NumPy run that was discarded),
  so read the RATIOS: both sides ran alternately under the same load. The route tables (interleaved in one process)
  are from the loaded host; the head-to-head ratios quoted here are the final, correctly pinned run. Re-measure on a
  quiet host inside the benchmark harness before quoting any number in a release.
- **Pinning trap, hit and fixed here.** The first NumPy timing scripts called `SetProcessAffinityMask` through `ctypes`
  without declared signatures, which fails silently (benchmark/CLAUDE.md, host stability), so NumPy ran UNPINNED
  against a pinned NumSharp. The committed `polynomial_rest_h2h_numpy.py` declares the signatures and reads the mask
  back; every head-to-head number below comes from it.
- **Probes** (committed, reproducible): `benchmark/polynomial/probes/polynomial_rest_{routes.cs, parity.cs,
  parity_numpy.py, h2h.cs, h2h_numpy.py, numpy_facts.py}` — commands in §9.

## 3. Per family

### 3.1 `polyvalfromroots(x, r, tensor=True)` — U3

**NumPy 2.4.2.**
```python
r = np.array(r, ndmin=1, copy=None)
if r.dtype.char in '?bBhHiIlLqQpP': r = r.astype(np.double)
if isinstance(x, (tuple, list)): x = np.asarray(x)
if isinstance(x, np.ndarray):
    if tensor: r = r.reshape(r.shape + (1,) * x.ndim)
    elif x.ndim >= r.ndim: raise ValueError('x.ndim must be < r.ndim when tensor == False')
return np.prod(x - r, axis=0)
```
So per point: `acc = 1` (the multiply identity, which `PyUFunc_ReduceWrapper` assigns), then `acc = acc * (x - r[k])`
for k = 0, 1, … in ROOT order, in the dtype `T = result_type(x, r)` (a Python-scalar x is weak). np.prod keeps `T`
(float16 runs HALF_multiply, float32/float64 the sequential reduce or the slab multiply — the same per-point sequence).

**The complex product depends on NumPy's iteration, not on values** (probe, exact rational fma, seeded with `1+0j`):

| Call shape | NumPy's product | Matches |
|---|---|---|
| Python-scalar x, 0-d x, a one-point x (one output element) | `LoopScalar` | 400/400 |
| many points, the roots axis outermost in the `x - r` temporary | `Simd` | 2,000/2,000 |
| `tensor=False`, F-ordered 2-D r (the roots axis innermost in the temporary) | `LoopScalar` | 400/400 |
| the same r, C-ordered | `Simd` | 400/400 |

Rule: NumPy's complex multiply loop (`loops_arithm_fp.dispatch.c.src`) has no reduce branch; it takes `simd_cmul`
unless the output stride is 0 (`b_sdst == 0` → `loop_scalar`, MSVC-contracted). The output stride is 0 exactly when
np.prod's inner loop walks ROOTS: one output element, or the roots axis is the innermost axis of the `x - r`
temporary's NpyIter layout (`Shape.NpyIterOutputShape` decides it; an exact in-place alias is not an overlap for
`is_mem_overlap`). `PolyComplexProduct.{Simd, LoopScalar}` already reproduce both forms.

**Result layout** (probed): the reduction output of the temporary, i.e. x's memory order (an F or transposed x gives an
F result), owning; an empty x has strides `(0,)`; an empty r gives ones.

**Routes measured** (float64, µs; routes interleaved; U3 proxy = `polyval` with nr+1 coefficients, same per-point op
count; its values differ by construction):

| N, roots | composition `np.prod(x - r, 0)` | `np.evaluate(Prod(x - r, 0))` | U3 kernel proxy |
|---|---:|---:|---:|
| 1K, 10 | 5.58 | 12.27 | 1.95 |
| 100K, 10 | 749 | 1,133 | 96.7 |
| 1M, 3 | 4,799 | 6,586 | 628 |
| 1M, 10 | 35,177 | 37,059 | 1,060 |
| 1M, 30 | 107,587 | 121,202 | 3,675 |

Head-to-head (proxy): 7.2× NumPy at 1K, 30.9× at 100K, 35.6× at 1M, complex 9.4×, scalar x 4.2×. Parity: the float64
composition is byte-exact; the complex composition is NOT (NumSharp's np.prod, finding F3); `np.evaluate`'s axis Prod
matched NumPy's complex array case.

**Recommendation: a third U3 skeleton, "root product".**
- `PolyEvalProgram` gets a skeleton beside Horner and Clenshaw: ascending over r, `Init = identity(T)`,
  `Step = Mul(Acc, Sub(X, Rk))`. The typing pass gives `Sub` NumPy's `result_type(x, r)` (strong r, weak or strong x)
  and `Mul` `T`; no peel is needed (the whole product runs in `T` from the first step).
- The real identity multiply `1*v` is exact (v is a subtraction result, never a signalling NaN), so real kernels may
  start at `x - r[0]`. Complex ones must emit the identity multiply in the chosen product form.
- Operand forms are U3's: a 1-D r in auxdata (pre-converted to `T` — exact widening), an N-D r through the
  `(x, r[0], y)` NDIter form, tensor reshapes as in `{p}val`.
- The driver adds NumPy's prologue (above), the complex-form rule, and the reduction-output layout.
- Rejected:
  - the composition: an nr×N temporary, 3–33× the kernel, and not exact for complex;
  - `np.evaluate`: it materializes the child, slower than the composition;
  - an NDIter REDUCE driver (`NumPyReduceIterator` with a fused subtract-multiply): exact, but nr passes over the
    output instead of one;
  - a standalone Direct IL kernel: it would duplicate U3's lanes, typing, weak-x folding and complex forms.

### 3.2 The five weight functions — U8

**NumPy 2.4.2:**

| Name | Statement |
|---|---|
| `chebweight(x)` | `1.0/(np.sqrt(1.0 + x)*np.sqrt(1.0 - x))` |
| `legweight(x)` | `x*0.0 + 1.0` |
| `lagweight(x)` | `np.exp(-x)` |
| `hermweight(x)` | `np.exp(-x**2)` |
| `hermeweight(x)` | `np.exp(-.5*x**2)` |

**`x**2` is `np.square(x)`, not `np.power(x, 2)` (finding F1).** `ndarray.__pow__` → `fast_scalar_power` (`number.c`)
maps an exact Python int 2 to `np.square` for EVERY array dtype (gh-29388); -1 → reciprocal and 0.5 → sqrt only for
float/complex. So bool x squares to int8 (`np.power(bool, 2)` is int64): NumPy's `hermweight(bool array)` is float16,
`hermeweight(bool array)` float64, `hermweight(uint8 [200])` is float16 inf (200² wraps).

**Boundary facts** (probed): Python-scalar x is Python arithmetic until the first `np.*` call (`legweight(0.5)` is a
Python `float`, `chebweight(2)` is nan, `hermeweight(2**70)` is 0.0, `hermweight(2**70)` and `lagweight(2**70)` raise
`TypeError: loop of ufunc does not support argument 0 of type int which has no callable exp method`); a list or tuple x
raises Python's own operator TypeErrors (`unsupported operand type(s) for +: 'float' and 'list'`,
`bad operand type for unary -: 'list'`, `unsupported operand type(s) for ** or pow(): 'list' and 'int'`,
`can't multiply sequence by non-int of type 'float'`); `lagweight(bool array)` raises NumPy's boolean-negative TypeError.

**Routes measured** (µs, interleaved; every route byte-identical to the eager composition unless marked):

| Cell | eager | fused | fused with `Power(x, 2)` |
|---|---:|---:|---:|
| chebweight f64 1M | 11,883 | 2,813 | — |
| legweight f64 1M | 1,332 | 547 | — |
| lagweight f64 1M | 5,907 | 5,011 | — |
| hermweight f64 1M | 7,507 | 5,632 | 28,868, 262 of 1M differ |
| chebweight f32 1M | 2,309 | 751 | — |
| hermweight f32 1M | 1,530 | 1,035 | 17,943, 39 differ |
| chebweight f16 1M | 17,273 | 85,067 | — |
| legweight f16 1M | 2,126 | 3,985 | — |
| hermweight f16 1M | 22,637 | 23,345 | 52,889 |

- **Parity:** eager and fused (with `Square`) are byte-exact with NumPy for float64/float32/float16 (hash parity, 100K).
  NDExpr's `Power(x, 2)` runs `pow` per element (open item X23) and differs; it is the wrong node for `x**2` anyway.
- **NDExpr's typing reproduces NumPy's per node** (checked): `Square(bool)` int8, `Square(int8)` wraps, `Exp(int8)`
  float16, `-0.5*Square(uint8)` float64, a negated bool raises NumPy's TypeError text.
- **Head-to-head:**

  | Cell | NPY/NS |
  |---|---:|
  | chebweight f64 1K / 1M | 2.39× / 5.28× |
  | legweight f64 1M | 8.86× |
  | chebweight f32 1M | 9.67× |
  | hermweight f32 1M | 4.11× |
  | chebweight f16 100K (eager) | 2.10× |
  | lagweight f64 1M | 1.75× |
  | hermweight f64 1K / 1M | 1.20× / 1.95× |
- **The float64 exp-based weights sit near a physical floor.** Both sides call the same CRT `exp` per element (NumPy's
  float64 exp is SIMD only on AVX-512; this host is AVX2). NumSharp's `np.exp` float64 alone runs 0.95× NumPy at 1K and
  1.23× at 1M (finding F4), so a 1K exp-based weight reaches ~1.2× and fusion only adds the saved passes at 1M
  (1.75–1.95×). Core is 100 % managed, so calling the CRT directly is not an option, and no other `exp` gives the same
  bits.

**Recommendation.**
- One fused `np.evaluate` tree per weight, NumPy's expression node for node: `1.0/(Sqrt(1.0 + X)*Sqrt(1.0 - X))`,
  `X*0.0 + 1.0`, `Exp(-X)`, `Exp(-Square(X))`, `Exp(-0.5*Square(X))`. Literals are weak, as in NumPy.
- float16 goes through the same statements on the eager house ufuncs. NDExpr's float16 path is 2–5× slower than the
  house widen-compute-narrow kernels; `np.var`/`np.std` route float16 the same way. The clean fix is lane-matched float16
  in NDExpr (§5 lever 6), which would remove this split.
- A Python-scalar x is replayed by `PolyNumber`/`PyScalar` (CPython arithmetic, then the first ufunc on a 0-d value).
  List/tuple x and the huge Python int raise NumPy's texts.
- Rejected:
  - `Power(x, 2)`: slower, and not what NumPy runs;
  - per-weight Direct IL kernels: no gain over the fused tree, which is already one SIMD pass.

### 3.3 `chebpts1`, `chebpts2`, `chebgauss` — U8 / U9

**NumPy 2.4.2.**
- `chebpts1(npts)`: `_npts = int(npts)`; `_npts != npts` → `ValueError('npts must be integer')`; `< 1` → `npts must be >= 1`;
  `x = 0.5*np.pi/_npts * np.arange(-_npts+1, _npts+1, 2)`; `np.sin(x)`.
- `chebpts2(npts)`: the same checks with `>= 2`; `np.cos(np.linspace(-np.pi, 0, _npts))`.
- `chebgauss(deg)`: `ideg = pu._as_int(deg, 'deg')` (`TypeError: deg must be an integer, received 3.0` — a float is
  refused, unlike `int()`); `<= 0` → `ValueError('deg must be a positive integer')`;
  `x = np.cos(np.pi*np.arange(1, 2*ideg, 2)/(2.0*ideg))`; `w = np.ones(ideg)*(np.pi/ideg)`.
- Probed: `chebpts1(3.0)` works, `chebpts1('3')` → `npts must be integer`, `chebpts1(None)` → int()'s TypeError text,
  `chebpts1(True)` → one point, `chebgauss(np.array(3))` works (`operator.index`).

**Routes** (µs, interleaved, all byte-exact with NumPy):

| Cell | eager | arange/linspace + fused |
|---|---:|---:|
| chebpts1 10 / 100 | 2.12 / 2.95 | 1.23 / 1.91 |
| chebpts1 1M | 6,330 | 5,956 |
| chebpts2 100 / 1M | 1.96 / 7,275 | 2.10 / 7,158 |
| chebgauss.x 10 / 100 | 2.76 / 3.70 | 1.27 / 2.09 |
| chebgauss.x 1M | 9,025 | 8,066 |

`w = np.full(deg, π/deg)` is byte-identical to `np.ones(deg)*(π/deg)`: `1.0*c` is exact.

**Head-to-head:**

| Cell | NPY/NS |
|---|---:|
| chebpts1 100 / 1M | 2.04× / 1.80× |
| chebpts2 100 / 1M | 4.06× / 1.39× |
| chebgauss 50 / 100K | 2.96× / 2.09× |

Large n is `sin`/`cos`-bound: `np.sin`/`np.cos` float64 alone run 0.95–1.25× NumPy (the same CRT). That is the ceiling
there; `chebpts2` also pays linspace's pass.

**Recommendation.**
- `chebpts1`/`chebgauss`: `np.arange` (the house int64 ramp) + one fused `Sin(c·k)` / `Cos((π·k)/(2.0·deg))`.
  `c = 0.5*π/n` is computed the way Python computes it, `(0.5*π)/n`.
- `chebpts2`: `np.cos(np.linspace(-π, 0, n))`, two house kernels, exact by construction.
- Optional lever (§5 lever 5): an NDExpr index leaf would remove the ramp temporary. Expected ~3× at n ≤ 100 (where
  chebpts1 sits at 2.04×) and nothing at large n; `chebpts2` would also drop the linspace pass, the one lever for its
  1.39× at 1M.

### 3.4 `leggauss`, `laggauss`, `hermgauss`, `hermegauss` — U8

**NumPy 2.4.2 (leggauss).**
```python
ideg = pu._as_int(deg, 'deg');  if ideg <= 0: raise ValueError('deg must be a positive integer')
c = np.array([0] * deg + [1]);  m = legcompanion(c);  x = np.linalg.eigvalsh(m)
dy = legval(x, c);  df = legval(x, legder(c));  x -= dy / df
fm = legval(x, c[1:]);  fm /= np.abs(fm).max();  df /= np.abs(df).max();  w = 1 / (fm * df)
w = (w + w[::-1]) / 2;  x = (x - x[::-1]) / 2;  w *= 2.0 / w.sum()
```
- `laggauss` has no symmetrisation and ends `w /= w.sum()`.
- `hermgauss`/`hermegauss` replace `legval`/`legder` with the normalised recurrences `_normed_hermite_n` /
  `_normed_hermite_e_n`, and end `w *= np.sqrt(np.pi) / w.sum()` / `np.sqrt(2*np.pi)`. Each runs the recurrence three
  times:
  ```python
  c0 = 0.;  c1 = 1./np.sqrt(np.sqrt(np.pi))            # hermite_e: np.sqrt(2*np.pi)
  for i in range(n - 1):
      tmp = c0;  c0 = -c1*np.sqrt((nd - 1.)/nd);  c1 = tmp + c1*x*np.sqrt(2./nd)    # hermite_e: 1./nd
      nd = nd - 1.0
  return c0 + c1*x*np.sqrt(2)                          # hermite_e: c0 + c1*x
  ```
- Probed: `leggauss(3.0)` → `TypeError: deg must be an integer, received 3.0`; `leggauss(np.array(3))` → the
  companion's `Series must have maximum degree of at least 1.`, because `[0] * np.array(3)` is not a list repeat.

**Measured.**
- Parts: `eigvalsh` is 2.0 / 12.4 / 92 / 367 µs at degree 5 / 20 / 50 / 100, parity with NumPy (1.0× at 100); the
  companion is 0.7–3.4 µs.
- Compositions (route table, loaded host): `leggauss` 23.9 / 32.3 / 116 / 402 µs, `hermgauss` 57 / 214 / 598 / 1,391 µs.
- `_normed_hermite_n` alone, composed from array statements: 14.6 / 59.1 / 311 µs at degree 5 / 20 / 100 (pinned run).
  The U3 Clenshaw kernel over the same points (`hermval`, the same step shape) takes 0.90 / 0.97 / 2.95 µs.
- Head-to-head:

  | Cell | NPY/NS |
  |---|---:|
  | leggauss 5 / 20 / 100 | 4.36× / 10.2× / 4.27× |
  | laggauss 20 | 18.9× |
  | hermgauss 5 / 20 / 100 | 1.51× / 1.33× / 1.14× |
  | hermegauss 20 | 1.38× |
- Estimated with the recurrence as a kernel (3 calls): `hermgauss` ~4.5× / ~12× / ~3.8× at degree 5 / 20 / 100.
- Parity: `leggauss` and `hermgauss` compositions are byte-exact with NumPy at degree 5 and 20 (OpenBLAS `syevd` at one
  thread).

**Recommendation.**
- Compose at the facade level in NumPy's statement order: U7 `{p}companion`, backend `np.linalg.eigvalsh`, U3
  `{p}val`, U4 `{p}der`, the house ufuncs, and the exact max (`NumPyMinMaxReduce`) and sum (`NumPyAddReduce`)
  reductions.
- `_normed_hermite_n` and `_normed_hermite_e_n` become U3 step-table programs: Clenshaw-shaped, two accumulators, no
  coefficient term. The program needs:
  - `StepC0 = Mul(Neg(C1), S1(nd))` (a unary negation node: `-c1` flips a NaN's sign, `0 - c1` does not);
  - `StepC1 = Add(Tmp, Mul(Mul(C1, X), S2(nd)))`;
  - final `C0 + (C1·X)·√2` (herm) or `C0 + C1·X` (herme);
  - init `C0 = 0.0` (a Python float, kept: `0.0 + (-0.0)` is `+0.0`), `C1 = 1/√√π` (herm) or `1/√√(2π)` (herme);
  - row constants `np.sqrt((nd-1)/nd)` and `np.sqrt(2/nd)` (or `1/nd`) as np.float64 (identical to Python floats here,
    because x is float64).
- From degree ~50 the call is `eigvalsh`-bound, and parity is the ceiling there.
- The U2 arena is not needed: the statements run on ≤ deg elements, and the measured composition already leaves only
  `eigvalsh` and the recurrence.

### 3.5 `chebinterpolate(func, deg, args=())` — U9

**NumPy 2.4.2.**
```python
deg = np.asarray(deg)
if deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0: raise TypeError("deg must be an int")
if deg < 0: raise ValueError("expected deg >= 0")
order = deg + 1;  xcheb = chebpts1(order);  yfunc = func(xcheb, *args)
m = chebvander(xcheb, deg);  c = np.dot(m.T, yfunc);  c[0] /= order;  c[1:] /= 0.5*order
```
Probed:
- `deg=3.0`, `True` and `[3]` raise `deg must be an int`; `deg=0` gives one coefficient.
- `func` may return `(n, k)` (k series at once) or any dtype `np.dot` promotes (float32 → float64).

**Measured.** Composition 5.6 / 6.4 / 14.1 µs at degree 5 / 20 / 100, 4.8× / 8.0× / 12.7× NumPy. Byte-exact (5, 20)
with `np.dot` on the OpenBLAS backend (gemv; 1.00–1.12× NumPy alone).

**Recommendation: composition.** Overloads `chebinterpolate(Func<NDArray, NDArray> func, object deg)` and
`(Func<NDArray, object[], NDArray> func, object deg, object[] args)`. Without the backend, `np.dot` computes managed
and is bounded-ULP, so the oracle tier is host-pinned, as the main plan already records.

### 3.6 The six fits — U6

**NumPy 2.4.2:** `{p}fit = pu._fit({p}vander, x, y, deg, rcond, full, w)`:
1. `x = np.asarray(x) + 0.0`, `y = …`; deg is an int or a non-empty 1-D int array (sorted; `vander[:, deg]`);
   NumPy's TypeErrors for the shapes and lengths.
2. `lhs = van.T`, `rhs = y.T`; the weights multiply both; `rcond = len(x)*finfo(x.dtype).eps`.
3. `scl = np.sqrt(np.square(lhs).sum(1))` (complex: `square(real) + square(imag)`); `scl[scl == 0] = 1`.
4. `c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)`; `c = (c.T / scl).T`; re-expanded into zeros
   for a deg list; `full` → `(c, [resids, rank, s, rcond])`.

`RankWarning` is not emitted (house convention). Probed:
- float16 x → lstsq's `TypeError: array type float16 is unsupported in linalg`, raised AFTER the Vandermonde;
- float32 x with float32 y → a float32 result; with float64 y → float64.

**Measured.**
- Norm: the fused streaming `np.evaluate(Sum(Square(lhs), axis: 1))` is byte-exact and 1.7–3.2× the eager norm
  (3.46 vs 6.40 µs at n=1000 d=10; 359 vs 1,069 µs at n=100K).
- Scaling: NumPy's `lhs.T / scl` (an F-ordered result with a row-broadcast operand) is slow in NumSharp:
  - 33.5 µs at 11×1,000 and 3,287 µs at 11×100K, i.e. 0.33× / 0.73× NumPy's 10.9 µs / 2,400 µs;
  - the C-ordered `lhs / scl[:, None]`, whose transpose has the same bytes, takes 9.0 µs and 825 µs (1.21× / 2.9×
    NumPy's spelling) — finding F2.
- Floor: `lstsq` alone is 1.02–1.50× NumPy (the same gelsd).
- Parity: the `chebfit` composition is byte-exact with NumPy (n=1000, d=10, OpenBLAS at one thread).
- Head-to-head (the recommended composition): `chebfit` 2.58× (n=100 d=3), 1.51× (n=1000 d=10), 2.60× (n=100K d=10);
  `polyfit` n=1000 d=10 1.41×; `Chebyshev.fit` 1.63×.
- Ceiling: at n=1000 d=10 gelsd is 78 of NumSharp's ~98 µs and of NumPy's ~138 µs, so the ceiling is ~1.77× with no
  wrapper at all. The measured 1.41–1.63× is that cell's honest range; it falls to parity as gelsd grows (d ≥ 30).

**Recommendation: composition of `_fit` in NumPy's order.**
- Two bit-neutral savings:
  - the norm as one fused streaming pass;
  - the column scaling done in the C-ordered form (or in place on the fresh Vandermonde buffer, which IS C-ordered
    under its F view), with its transpose handed to `lstsq`.
- `lstsq` runs through the backend; the oracle tier is host-pinned (`linalg_parity` model).
- Not worth a kernel: everything outside `lstsq` is O(n·d) with 3–4 passes.

### 3.7 The six classes — U10

NumPy's `ABCPolyBase` holds no loops of its own. Every member binds a module function (`_add = chebadd`,
`_val = chebval`, `_fit = chebfit`, …) or a `polyutils` helper, plus small Python logic:
- the domain/window/symbol checks (`_get_coefficients`, verbatim TypeErrors);
- `fromroots`: `domain=[]` vs `None`, `/ scl**deg`;
- `integ`/`deriv` domain scaling, `roots` mapped back through the domain, `linspace`, `cutdeg`/`truncate`/`trim`;
- printing.

`convert`/`cast` evaluate the series AT a series: `self(kind.identity(domain, window))` runs Clenshaw with the
class operators, `x*0` included (main plan §3 U10).

Measured proxies (the unit functions the members call; the class adds an object and a few checks):

| Member | NPY/NS |
|---|---:|
| `p + q` | 38× |
| `p * q` | 31× |
| `ch(x)` 1K / 100K (mapdomain + chebval) | 6.3× / 45× |
| `Chebyshev.fit` (LAPACK-bound) | 1.63× |
| `convert` | far above NumPy's 0.6 ms: deg × a few series ops instead of a Python loop of object ops |

**Recommendation.**
- An object layer, `NDPolyBase` + the six classes (Appendix B of the main plan), whose members call the unit entry
  points. No kernels.
- The one new engine piece: **U3's step tables interpreted over a series algebra**. Each value is a series object and
  each op a class operator over U1/U2, like U4's `PyVal1D` interpreting weak-x trees with `PolyNumber`.
- Optional lever: fuse `mapdomain` into the U3 kernel's prologue for `p(x)`. `off + scl*x` per point, so even the
  identity map's `-0.0 → +0.0` is kept. Measured 310 → ~120 µs at 100K, but `p(x)` is already 55× without it.

### 3.8 Printing — U11

- **`set_default_printstyle(style)`**: a process-wide flag on the base class. `'unicode'`/`'ascii'`; anything else
  raises `ValueError("Unsupported format string '<s>'. Valid options are 'ascii' and 'unicode'")`. The default is
  NumPy's `os.name != 'nt'`, i.e. `!OperatingSystem.IsWindows()`. No compute.
- **`polyutils.format_float(x, parens=False)`**:
  - a non-floating x is `str(x)`; NaN/inf are the print options' `nanstr`/`infstr`;
  - scientific when `|x| >= 1e8` or `|x| < 10**min(0, -(precision-1)//2)` (1e-4 at precision 8), else positional;
  - `floatmode == 'fixed'` → `trim='k', unique=False`, else `trim='0', unique=True`; `sign == '+'` honoured;
  - `parens` wraps the scientific form only: `1e-5` → `(1.0e-05)`;
  - route: the existing Dragon4 port (`Backends/Printing/Dragon4.cs`, already taking a `FloatKind`). The prerequisite
    is float32/float16 entry points for `np.format_float_positional`/`_scientific` (main plan U11). The Dragon4 call
    alone is 0.148 µs against NumPy's 4.59 µs `format_float` (31×).
- No IL or NDIter: string building over a handful of coefficients, read through the typed accessors as
  `ArrayFormatter` does.

## 4. The list, with the recommendation

| # | NumPy name | Module(s) | Recommended compute route | Rejected | Parity tier | Measured / expected NPY/NS |
|---|---|---|---|---|---|---|
| 1 | `polyvalfromroots` | polynomial | **U3 step-table IL kernel, new "root product" skeleton, NDIter-driven**; NumPy's prologue; complex form by reduction mode | composition (temp, 3–33× slower, complex inexact); `np.evaluate` Prod; NDIter REDUCE; standalone Direct IL | portable (complex via `PolyComplexProduct`) | 4.2–35.6× (proxy) |
| 2 | `chebweight` | chebyshev | **fused `np.evaluate`**; f16 eager | Direct IL | portable (sqrt exact) | 2.1–9.7× |
| 3 | `legweight` | legendre | **fused `np.evaluate`**; f16 eager | — | portable | 8.9× (f64 1M) |
| 4 | `lagweight` | laguerre | **fused `np.evaluate`**; f16 eager | — | host-libm (exp) | 1.75× f64 1M (CRT exp floor) |
| 5 | `hermweight` | hermite | **fused `np.evaluate` with `Square`** (NumPy's `x**2`); f16 eager | `Power(x, 2)` (pow: slower, not NumPy's op) | host-libm | 1.20× / 1.95× f64 1K / 1M (exp floor), 4.1× f32 |
| 6 | `hermeweight` | hermite_e | **fused `np.evaluate` with `Square`**; f16 eager | `Power(x, 2)` | host-libm | as `hermweight` |
| 7 | `chebgauss` | chebyshev | **`np.arange` + fused `Cos`; `np.full` for w** | eager (2× the fixed cost) | host-libm (cos) | 2.1–3.0× |
| 8 | `leggauss` | legendre | **composition** (U7 companion → `eigvalsh` → U3/U4 → house ufuncs/reductions) | arena (not needed) | host-pinned (syevd) | 4.3–10.2× |
| 9 | `laggauss` | laguerre | **composition** (as leggauss, no symmetrisation) | — | host-pinned | 18.9× (deg 20) |
| 10 | `hermgauss` | hermite | **composition + `_normed_hermite_n` as a U3 step-table program** | array-statement loop (1.14–1.51×) | host-pinned | est. 3.8–12× |
| 11 | `hermegauss` | hermite_e | **composition + `_normed_hermite_e_n` as a U3 step-table program** | array-statement loop (1.38×) | host-pinned | est. as hermgauss |
| 12 | `chebpts1` | chebyshev | **`np.arange` + fused `Sin`** (lever: NDExpr index leaf) | eager | host-libm (sin) | 1.8–2.0× |
| 13 | `chebpts2` | chebyshev | **`np.cos(np.linspace(...))`** (lever: index leaf) | — | host-libm (cos) | 1.39× (1M, cos-bound) – 4.1× |
| 14 | `chebinterpolate` | chebyshev | **composition** (`chebpts1` → delegate → `chebvander` → `np.dot` → 2 in-place divides) | a fused DCT (breaks gemv parity) | host-pinned (gemv) | 4.8–12.7× |
| 15 | `polyfit` | polynomial | **composition of `_fit`** + fused streaming norm + C-ordered scaling, `lstsq` via backend | a fit kernel (LAPACK-bound) | host-pinned (gelsd) | 1.41× (n=1000 d=10; ceiling ~1.77×) |
| 16 | `chebfit` | chebyshev | as `polyfit` | as `polyfit` | host-pinned | 1.51–2.6× |
| 17 | `legfit` | legendre | as `polyfit` | as `polyfit` | host-pinned | as chebfit |
| 18 | `lagfit` | laguerre | as `polyfit` | as `polyfit` | host-pinned | as chebfit |
| 19 | `hermfit` | hermite | as `polyfit` | as `polyfit` | host-pinned | as chebfit |
| 20 | `hermefit` | hermite_e | as `polyfit` | as `polyfit` | host-pinned | as chebfit |
| 21 | `Polynomial` | package, polynomial | **object layer over U1–U9**; series-algebra step-table interpretation for `convert`/`cast` | — | portable (fit/roots host-pinned) | 6–45× (proxies; fit 1.6×) |
| 22 | `Chebyshev` | package, chebyshev | as `Polynomial` | — | as `Polynomial` | as `Polynomial` |
| 23 | `Legendre` | package, legendre | as `Polynomial` | — | as `Polynomial` | as `Polynomial` |
| 24 | `Laguerre` | package, laguerre | as `Polynomial` | — | as `Polynomial` | as `Polynomial` |
| 25 | `Hermite` | package, hermite | as `Polynomial` | — | as `Polynomial` | as `Polynomial` |
| 26 | `HermiteE` | package, hermite_e | as `Polynomial` | — | as `Polynomial` | as `Polynomial` |
| 27 | `set_default_printstyle` | package | **a process-wide flag** (default: ASCII on Windows) | — | portable | — |
| 28 | `format_float` | polyutils | **Dragon4 port** + NumPy's branch; float32/float16 entry points first | — | portable | ~31× (proxy) |

## 5. Prerequisites and levers, ranked

1. **U3 emitter, three additions** (needed by #1, #10, #11):
   - a skeleton with an ascending index and a product accumulator;
   - a unary `Neg` node;
   - programs without a coefficient term (the normed Hermite recurrences).
2. **The complex-form rule for `polyvalfromroots`** (#1): LoopScalar where np.prod's inner loop walks roots, Simd
   otherwise. Decided from `Shape.NpyIterOutputShape` of the `x - r` temporary.
3. **Dragon4 float32/float16 entry points** (#28 and the class printing).
4. **U3 over a series algebra** (#21–#26: `convert`/`cast`/`identity`), reusing the U4 `PyVal1D` interpreter pattern.
5. *Optional:* an **NDExpr index leaf** (iota / linspace ramp): chebpts1 ~3× at small n, chebpts2 one pass less at
   large n; also np.hanning & co.
6. *Optional:* **lane-matched float16 in NDExpr**, removing the float16 split of the weights (and of np.var/np.std).
7. *Optional:* fusing `mapdomain` into the U3 kernel prologue for the classes' `__call__`.
8. *Not needed:* X23 (NDExpr `Power` fast path). The weights use `Square`, which is what NumPy's `x**2` runs; X23
   remains open for np.kaiser.

## 6. Findings on the way

- **F1 — `x**2` is `np.square(x)` for every array dtype.** `fast_scalar_power` (`number.c`, gh-29388): a Python int 2
  → square for any dtype; -1 → reciprocal and 0.5 → sqrt for float/complex only. `np.power(bool, 2)` is int64, but
  `bool_array**2` is int8. Every NumSharp port of a NumPy expression `a**2` with an array `a` should emit
  `np.square`/`NDExpr.Square`, not `power`.
- **F2 — `x.T / v` (an F-ordered result with a row-broadcast operand) is a slow binary-op path in NumSharp.** At
  11×1,000 it takes 33.5 µs, against 9.0 µs for the C-ordered spelling with the same bytes; at 11×100K, 3.3 against
  0.82 ms. NumPy's own spelling is 10.9 µs / 2.4 ms, so the NumSharp spelling runs 0.33× / 0.73× NumPy (the C
  spelling 1.21× / 2.9×). This is library-wide; it belongs to the binary-op layout work, not to this package.
- **F3 — NumSharp's `np.prod` over complex does not reproduce NumPy's complex products.** On the `x - r` temporary,
  NumPy uses `Simd` for a slab and `LoopScalar` for one output, both seeded with `1+0j`. `np.evaluate`'s axis Prod
  matched NumPy for the slab case. Related pins: `OpenBugs.Reductions.cs` (complex one-element prod skips the identity).
- **F4 — the float64 transcendental floor.** NumSharp's `np.exp`/`np.sin`/`np.cos` float64 run 0.95× NumPy at 1K and
  1.17–1.25× at 1M (the same CRT on both sides). An unpinned NumPy run had read exp at 0.85×, which the pinned run
  does not reproduce. Any weight or point generator dominated by one of them inherits this floor.
- **F5 — measurement trap:** repeatedly dividing one buffer in place by values ≥ 1 drives it into subnormals; the
  timing then measures denormal arithmetic, not the kernel. Time in-place routes on a fresh buffer per call.

## 7. Oracle and test hooks per family

| Family | Tier | Notes |
|---|---|---|
| `polyvalfromroots` | the `polyeval` tier (portable), new section | all layouts × dtypes × tensor; the four complex modes of §3.1; Python/NumPy-scalar x; list/tuple x; empty r; int/bool r; the ValueError; the result-layout facet |
| weights | `chebweight`/`legweight` portable; `lagweight`/`hermweight`/`hermeweight` host-libm (`RunHostLibmCorpus`) | every dtype incl. bool/int8/uint8 wrap-arounds, Python-scalar kinds, list/tuple TypeErrors, the huge-Python-int TypeError |
| `chebpts1`/`chebpts2`/`chebgauss` | host-libm | the `int()`/`_as_int` argument matrix of §3.3 |
| `leg/lag/herm/hermegauss` | host-pinned (`linalg_parity` model, syevd at one thread) | degrees 1…200; Hermite past 207 (where the plain recurrence overflows and NumPy's normed one does not) |
| `chebinterpolate` | host-pinned (gemv) | (n,) and (n, k) results, every dtype the delegate returns, `args` |
| fits | host-pinned (gelsd) | the `_fit` TypeError/ValueError matrix, deg lists, weights, 2-D y, `full`, float16 refusal, float32 lite |
| classes | portable (fit/roots inherit their tiers) | `test_classes.py` × 6, `test_symbol.py`, `convert`/`cast` bytes (the series-object Clenshaw) |
| printing | portable | `test_printing.py`; `format_float` over print options |

## 8. Open questions for the owner

- **`chebinterpolate`'s delegate shape.** `Func<NDArray, NDArray>` plus an `object[] args` overload, as the main plan
  proposes, or `Func<NDArray, object[], NDArray>` only?
- **F2.** Fix the F-result broadcast binary path library-wide (it also affects any `x.T op v`), or only spell the fit's
  scaling in C order?
- **The float16 weights.** Accept the eager route (the `np.var` precedent), or wait for lever 6?

## 9. Reproduce

From the repository root on branch `strings` (OpenBLAS staged in `src/NumSharp.Interop.OpenBLAS/runtimes`):

```bash
# NumPy facts: boundary kinds/dtypes/errors and the polyvalfromroots complex forms
python benchmark/polynomial/probes/polynomial_rest_numpy_facts.py
# NumSharp routes (interleaved, byte-compared with the eager composition); sections W P R F G I C
DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/polynomial/probes/polynomial_rest_routes.cs WPRFGIC
# parity: compare the two hash listings line by line
python benchmark/polynomial/probes/polynomial_rest_parity_numpy.py
DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/polynomial/probes/polynomial_rest_parity.cs
# head-to-head (run the pair alternately at least twice; take each cell's minimum)
OPENBLAS_NUM_THREADS=1 python benchmark/polynomial/probes/polynomial_rest_h2h_numpy.py
DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/polynomial/probes/polynomial_rest_h2h.cs
```
