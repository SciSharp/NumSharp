# numpy.polynomial — delivery plan in backend-sharing units

> Plan for porting NumPy's **polynomial package** (`numpy.polynomial`: six basis modules, `polyutils`, the
> six convenience classes) to NumSharp, split into **delivery units — each unit a family of functions
> that share one backend driver**, so every driver is written once and instantiated for all six bases.
> The legacy half of the same docs page (`np.poly1d`, `np.polyval`, `np.polyfit`, …) already ships.
>
> Source of truth: `refs/numpy/` (**2.4.2**) — `numpy/polynomial/{polyutils,_polybase,polynomial,chebyshev,
> legendre,laguerre,hermite,hermite_e,__init__}.py` and `numpy/polynomial/tests/`. The linked page
> <https://numpy.org/devdocs/reference/routines.polynomials.html> documents numpy **main (2.5.dev)**; its
> source was diffed against 2.4.2 (§0.3). Every behaviour, dtype, error text and timing below was probed
> live against `numpy==2.4.2` (win-amd64, `OPENBLAS_NUM_THREADS=1`) on 2026-09-24.

---

## 0. Where we stand (the check)

### 0.1 The page, section by section

| Page section | NumPy API | NumSharp today |
|---|---|---|
| Legacy "polynomial module" (`numpy.lib.polynomial`) | `poly1d`, `polyval`, `poly`, `roots`, `polyfit`, `polyder`, `polyint`, `polyadd`, `polydiv`, `polymul`, `polysub` | **Done** — all 11 functions and `poly1d(c_or_r, r, variable)`, byte-exact. Oracle: `poly.jsonl` (portable); `roots`, `polyfit` and `poly`-of-a-matrix are in host-pinned `linalg_parity`. Unit tests + live-parity tests. Only `RankWarning` is absent: NumSharp emits no warnings anywhere. |
| "Polynomial package" (`numpy.polynomial`) | 6 modules × ~31 names, `polyutils`, 6 classes, `set_default_printstyle` | **Not started** — 0 of 193 functions, 0 of 6 classes, no `np.polynomial` facade. `coverage/generate_coverage.py` already catalogues all seven `numpy.polynomial.*` submodules as out-of-headline **"candidate"** surfaces, every member currently "missing". |
| "Transition guide" | the reversed coefficient order; `Polynomial.fit(...).convert()` | Documentation only. It is a real hazard for us, though, because the new package **reuses the legacy names with the opposite coefficient order** (§2 D5). |

User demand on record: issue **#496** "Can NumSharp fit polynomial surface equations?" — that is exactly
`polyvander2d` + `lstsq` (units U5 + U6).

### 0.2 Size of the package surface (2.4.2 `__all__`, machine-counted)

| | Count |
|---|---|
| Basis-module functions and constants (`polynomial` 27, `chebyshev` 33, `legendre`/`laguerre`/`hermite`/`hermite_e` 30 each, classes excluded) | 180 |
| `*valnd(pts, c)`, new in devdocs (one per basis) | 6 |
| `polyutils` (`as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapdomain`, `mapparms`, `format_float`) | 7 |
| **Functions total** | **193** |
| Classes (`Polynomial`, `Chebyshev`, `Legendre`, `Laguerre`, `Hermite`, `HermiteE` over `ABCPolyBase`) | 6 + base, ≈40 members each (shared) |
| Package level | `set_default_printstyle` |

NumPy's own suite for it: `numpy/polynomial/tests/` — 11 files, **339** test functions (many parameterised
over all six classes). Their layout already follows the unit boundaries below: every `test_<basis>.py`
has `TestConstants/TestArithmetic/TestEvaluation/TestIntegral/TestDerivative/TestVander/TestFitting/
TestCompanion/TestGauss/TestMisc`.

### 0.3 devdocs (numpy main) vs the 2.4.2 oracle

`polyutils.py`, `_polybase.py` and `__init__.py` are byte-identical (after CRLF normalisation). The six basis
modules differ in two ways only:

1. **`{p}valnd(pts, c)`** added to every basis — literally `return pu._valnd({p}val, c, *pts)`. Free once
   `_valnd` exists, and **oracle-testable against 2.4.2** through that same private helper.
2. `{p}roots` now ends with `_to_real_if_imag_zero(r, m)`. main's `eigvals` changed to always return complex, and this
   shim restores the old result. **Observable behaviour = 2.4.2's** (real iff every imaginary part is 0), which is
   what NumSharp's `np.linalg.eigvals` already returns. Nothing to port.

---

## 1. How NumPy builds it — three layers

```
Layer C  ABCPolyBase (+ 6 ~30-line subclasses)     domain/window mapping, operators, call, convert, fit, printing
           │  12 "virtual functions" bound per class: _add _sub _mul _div _pow _val _int _der _fit _line _roots _fromroots
Layer B  six basis modules                          the same ~31-name template; the ONLY real per-basis code is a
           │                                         handful of recurrence steps (mulx, Clenshaw, vander, der/int)
Layer A  polyutils                                  basis-agnostic kernels, parameterised by the basis's own functions:
                                                     as_series trimseq trimcoef getdomain mapdomain mapparms
                                                     _add _sub _div(mul) _pow(mul) _fromroots(line,mul)
                                                     _valnd(val) _gridnd(val) _vander_nd(vander) _fit(vander) _as_int
```

What is shared vs what varies, per operation (read from the 2.4.2 source):

| Operation | Shared driver (written once) | Per-basis variance |
|---|---|---|
| add / sub / trim | `pu._add`, `pu._sub`, `pu.trimcoef` — identical for all 6 | none |
| line / constants | — | 2–3 numbers per basis (`lagline = [off+scl, -scl]`, `hermline = [off, scl/2]`, `lagdomain = [0,1]`, `hermx = [0, 0.5]`, `lagx = [1, -1]`) |
| mulx | — | one 3-term step per basis |
| mul | **power:** `np.convolve`. **Chebyshev:** z-series (`_cseries_to_zseries` → `np.convolve` → back). **Legendre, Laguerre, Hermite, HermiteE:** the same backward-recurrence skeleton over *series* built from `{p}mulx`/`{p}add`/`{p}sub` | only the two step expressions inside that skeleton |
| div | `pu._div(mul)`, repeated subtraction of `mul([0]*i+[1], c2)`; **power and Chebyshev** have their own faster loops (synthetic division / `_zseries_div`) | which driver is used |
| pow | `pu._pow(mul)`, repeated multiplication (power uses `np.convolve` directly; Chebyshev convolves z-series) | `maxpower` default: `None` for power, 16 for the other five |
| fromroots | `pu._fromroots(line, mul)`: sort the roots, then a pairwise product tree | `line`, `mul` |
| val | **power:** Horner (`c0 = c[-i] + c0*x`, seeded with `c[-1] + x*0`). **Others:** the same 2-accumulator Clenshaw skeleton | the two step expressions (`legval` uses `c1*((nd-1)/nd)`, `lagval` uses `(c1*(nd-1))/nd` — ORDER differs, and it matters, §2 D3) |
| val2d / 3d / nd, grid2d / 3d | `pu._valnd(val)`, `pu._gridnd(val)` | none |
| der / int | the same ~25-line prologue (copy, int→f64, `_as_int` for `m`/`axis`, `moveaxis`, `k` padding, `lbnd` via `{p}val`) | a 2–4 line recurrence body |
| vander (+2d/3d) | `pu._vander_nd` / `_vander_nd_flat` (broadcast product) | forward-recurrence step |
| fit | `pu._fit(vander)`: validation, column-norm scaling, **`np.linalg.lstsq`**, deg-list expansion | none |
| companion / roots | roots = companion, **rotated `[::-1, ::-1]`** for the 5 non-power bases, **`np.linalg.eigvals`**, then sort | companion matrix construction; the degree-1 closed form |
| gauss | **Chebyshev:** closed form (`cos`). **Others:** companion, **`np.linalg.eigvalsh`**, one Newton step (`{p}val`/`{p}der` or `_normed_herm(e)_n`), symmetrise (except Laguerre), normalise | the weight normaliser (`2`, `1`, `√π`, `√(2π)`) |
| weight | — | one elementwise formula (`1/(√(1+x)·√(1-x))`, `1`, `e^{-x}`, `e^{-x²}`, `e^{-x²/2}`) |
| X2poly / poly2X | poly2X = Horner in basis X (`Xadd(Xmulx(res), pol[i])`); X2poly = recurrence over `polyadd/polysub/polymulx` | step expressions |

So the **natural unit of delivery is a row of that table** (one shared driver, six step tables). Delivering
by basis instead ("do all of Chebyshev first") would build each driver six times or refactor it five times,
and it cuts across NumPy's own test layout.

---

## 2. Cross-cutting decisions (settled in U1, before any family lands)

**D1 — Facade shape: instance-property modules, not nested static classes.**

```csharp
var P = np.polynomial.polynomial;      // from numpy.polynomial import polynomial as P
var C = np.polynomial.chebyshev;       // from numpy.polynomial import chebyshev as C
NDArray y = P.polyval(x, c);           // NOTE: low→high coefficient order (the package's convention)
NDArray k = C.chebfit(x, y, 5);
np.polynomial.polyutils.mapdomain(x, oldDom, newDom);
np.polynomial.set_default_printstyle("unicode");
```

`np.polynomial` is a lowercase static property returning `PolynomialModule`, annotated
`[ModuleName("np.polynomial")]`. Each submodule is a lowercase property on it, returning its own class
annotated `[ModuleName("np.polynomial.<mod>")]`: `polynomial`, `chebyshev`, `legendre`, `laguerre`,
`hermite`, `hermite_e`, `polyutils`. This is the `np.fft`/`np.emath`/`np.ma` shape, and it is forced.
A nested static class like `np.linalg` cannot work because C# forbids a member named `polynomial` inside a
type named `polynomial` (CS0542). As a bonus, the module object is a first-class value, so Python's
`import … as P` becomes `var P = …`. The coverage tool credits a member only when the facade annotated for
that exact submodule has it, never via the top-level `np.polyval` namesake (`generate_coverage.py`
`extended_surface_rows`). Annotating the eight hosts is all it takes to flip the seven "candidate" surfaces.

**D2 — Classes: `ND*` types with NumPy-named aliases.** Following the house rule, the types are `NDPolyBase`
(≙ `ABCPolyBase`), `NDPolynomial`, `NDChebyshev`, `NDLegendre`, `NDLaguerre`, `NDHermite` and `NDHermiteE`.
The NumPy names exist as **factory aliases** on the facades (`np.polynomial.Polynomial(coef, domain, window,
symbol)`, and again on the defining submodule, e.g. `np.polynomial.chebyshev.Chebyshev(...)`). That is how
`numpy.random.RandomState` is credited today. Proposed mechanics:
- A CRTP generic `NDPolyBase<TSelf> : NDPolyBase` whose inherited statics are reachable through the derived
  name, returning `TSelf`: `NDChebyshev.fit(x, y, 3)`, `.fromroots`, `.identity`, `.basis`, `.cast`.
- A `static abstract` factory for constructing `TSelf` (net8+).
- Operators declared on `NDPolyBase<TSelf>`.

What C# cannot express, all registered in §6:
- **One member that is both callable and dotted.** `np.polynomial.Polynomial.fit(...)` does not port
  verbatim. The idiomatic port of `from numpy.polynomial import Polynomial` is
  `using Polynomial = NumSharp.NDPolynomial;`, after which `new Polynomial(c)` and `Polynomial.fit(x, y, 3)`
  both read like the Python.
- **A static and an instance member with one name** (CS0102). The class-level defaults `Polynomial.domain`
  and `Polynomial.window` therefore become `NDPolynomial.default_domain` and `NDPolynomial.default_window`;
  the per-instance `p.domain`, `p.window`, `p.coef`, `p.symbol`, `p.maxpower` and `p.basis_name` keep NumPy's
  names.
- **Printing keeps NumPy's names.** `repr`/`str` print NumPy's class names (`Polynomial(`, `T_1`), never the
  `ND*` names; these are NumPy-verbatim strings.

**D3 — Typed span kernels under one per-dtype ops contract (the load-bearing correctness decision).**
NumPy's package is pure Python over array ops. Porting each op to an `NDArray` call would be correct but
construction-bound: `legmul` at degree 50 runs thousands of small array and scalar ops (≈2 ms in NumPy, most
of it in `legmulx`'s per-coefficient loop). The alternative is to port the
per-basis steps to typed span loops over one `ISeriesOps<T>` per coefficient dtype (`Half`, `float`, `double`,
`Complex`, `decimal`) that reproduces **NumPy's array-loop semantics op by op**. Traps it must encode, most
of which have already bitten NumSharp:
- **complex:** NumPy's *array* multiply is the fused `simd_cmul`; *numpy-scalar × numpy-scalar* is naive
  scalarmath (the legacy `polydiv` hit exactly this and carries `NaiveComplexScalarMultiply`). Array divide is
  Smith with `scl = 1/denom` (`ComplexDivideNumPy`). Every NumPy expression must be classified as
  scalar∘scalar or array∘(scalar|array). For example, `_div`'s `q = rem[-1] / p[-1]` is scalar/scalar, while
  `rem[:-1] - q * p[:-1]` is array.
- **float16:** every op widens to f32, computes, and narrows (NumPy's HALF loops). A weak Python-int factor is
  first *rounded to float16* (`(2*j) * c[j]` → `half(2j)`), which is inexact above 2048.
- **float32** stays float32; Python int/float factors are weak.
- **Evaluation order is the source's, token for token:** `(c1 * (nd-1)) / nd` (lagval) is not
  `c1 * ((nd-1) / nd)` (legval). There is no FMA contraction: RyuJIT never contracts, and NumPy's MSVC build
  doesn't either.
- **N-D coefficient arrays** (the `axis` of der/int, multi-series `c` in val) run the same kernels per column
  over strided spans. There is never a second implementation.
- **Proof before use:** U1 ships a differential "ops probe" (every step expression × all 5 dtypes × random,
  special and adversarial values, vs NumPy) that U2+ build on.

**D4 — Dtype policy (probed 2.4.2).**
- **Coefficients** go through `np.common_type`:
  - f16, f32, f64 and c128 are preserved;
  - every int/uint becomes f64;
  - **bool raises `ValueError("Coefficient arrays have no common type")`**, because common_type rejects bool
    and there is no object dtype to fall back to;
  - Char follows NumSharp's `common_type`;
  - **Decimal stays native decimal**, a NumSharp extension: NumPy only reaches Decimal through object arrays,
    so it gets the independent C# decimal oracle instead of a byte oracle;
  - **complex64 comes back complex128** (F1, #569).
- **Evaluation dtype** is `result_type(c, x)` with an NEP 50 weak `x`: `polyval(0.5, f16)` returns f16.
  NumPy's *strong* `np.float64(0.5)` has no C# spelling, because 0-d operands are weak at NumSharp's ufunc
  entry points (house-wide).

**D5 — The legacy-name collision.** The package reuses legacy names with the **opposite** coefficient order
and argument order: legacy `np.polyval(p, x)` takes high→low coefficients with `x` second, while
`np.polynomial.polynomial.polyval(x, c, tensor=true)` takes low→high with `x` first. The same goes for
`polyadd/polysub/polymul/polydiv/polyder/polyint/polyfit`. They live on different C# classes, so nothing
mis-binds. **Every name-keyed gate would silently credit the new function with the legacy corpus**, though. So:
- the new `OpRegistry` keys are **module-qualified** (`polynomial.polyval`, `chebyshev.chebval`, …);
- the new facade surface gate keys on qualified names;
- each pair's XML docs cross-reference each other and state the coefficient order.

**D6 — Constants are read-only singletons, dtype-exact.** `polydomain`, `chebzero` and the rest are mutable
ndarrays in NumPy. Expose them as non-writeable process-wide singletons **detached from every `NDScope`** (the
`np.ma.nomask` `Own(...)` pattern: a singleton first built inside a caller's scope would be disposed with it).
Writing to one raises read-only (`[Misaligned]`). Their dtypes are NumPy's literal ones, probed from the
source:
- every `{p}domain` is float64 (`[-1., 1.]`; Laguerre `[0., 1.]`);
- `{p}zero`/`{p}one`/`{p}x` are **int64** (`[0]`, `[1]`, `[0, 1]`; Laguerre `lagx = [1, -1]`);
- **`hermx = [0, 0.5]` is float64**, because its literal is `1/2`.

**D7 — Lifetime.**
- Every function is `[NDScoped]`.
- Classes own `coef`, `domain` and `window`, so they are `IDisposable`, like `poly1d`.
- The eight `[ModuleName]` facades are **auto-enrolled** in `LeakSurfaceCoverageTests`. The class types must
  be added to its hardcoded `OperatorOwners`/`ObjectOwners` lists.
- Each unit adds **measured** `LeakCatalogue` entries; the completeness gate credits only measured ids.

**D8 — Backend-bound members ride the existing seam.** These route through the existing `np.linalg` or
product functions, adding no new `IBlasBackend` surface, exactly as the legacy `roots`/`polyfit` do:

| Members | Route | Without `NumSharp.Interop.OpenBLAS` |
|---|---|---|
| `*fit` | `np.linalg.lstsq` (gelsd) | NumPy-exact validation first, then the existing `NotSupportedException` |
| `*roots` | `np.linalg.eigvals` (geev) | same |
| `leggauss`/`laggauss`/`hermgauss`/`hermegauss` | `np.linalg.eigvalsh` (syevd) | same |
| `chebinterpolate` | `np.dot` (gemv) | computes (managed product), but byte parity needs the backend |

Portable escape hatches: a degree-≤1 `roots` is closed-form, and `chebgauss` is closed-form. The
`np.convolve`-based products are covered in U2.

**D9 — Layout parity.** Observable through `.flags`/`.strides`, so reproduce it:
- `*vander` returns NumPy's **`moveaxis` view** of a C-contiguous `(deg+1, *x.shape)` buffer. For 1-D `x`
  that is F-contiguous with strides `(itemsize, itemsize·x.size)` (probed: `chebvander(arange(4.), 2)` →
  `(8, 32)`).
- `*vander2d`/`*vander3d` are F-contiguous.
- `*der`/`*int` with `axis≠0` come back in the `moveaxis`-back layout (F-contiguous for 2-D).
- `*val` with a 2-D `c` returns C-contiguous `c.shape[1:] + x.shape`.

**D10 — Errors verbatim** (Appendix C). The house types apply (ValueError, TypeError, AxisError). NumPy's bare
`raise ZeroDivisionError` has an **empty** message, so pass `""` explicitly: .NET's default text would
otherwise leak.

**D11 — Operator resolution (the `poly1d` lesson, inverted).** NumPy's classes set `__array_ufunc__ = None`
and have no `__array__`. So `ndarray + Polynomial` is **series** addition via `__radd__`: the opposite of
`poly1d`, where the ndarray wins elementwise. Rules:
- Give `NDPolyBase` **no** implicit conversion to `NDArray`.
- Declare exact `(NDArray, NDPolyBase<TSelf>)` and `(NDPolyBase<TSelf>, NDArray)` overloads. Without them,
  `nd + p` is CS0034-ambiguous against NDArray's `(NDArray, object)` operators.
- An `(object, NDPolyBase<TSelf>)` overload will also catch `"str" + p`, so it must raise TypeError, never
  build a char series.

---

## 3. The delivery units

Every unit ships **for all six bases at once**: one shared driver plus six step tables. Names are
machine-partitioned; the counts sum to 193 with no overlap (Appendix A).

### U1 — Substrate, facades and the additive family (54 names)

- **Scope:**
  - facades: `np.polynomial` + 7 submodules, `[ModuleName]` ×8;
  - `polyutils`: `as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapdomain`, `mapparms`;
  - constants `{p}domain/{p}zero/{p}one/{p}x` (24), `{p}line` (6);
  - `{p}add`/`{p}sub` (12) and `{p}trim` (6, NumPy aliases of `trimcoef`).
- **Shared backend:** `as_series` (`np.common_type` + `trimseq`), `_add`/`_sub`. The **`ISeriesOps<T>`
  contract + ops probe** (D3) and the **`SeriesBasis` step-table abstraction** that every later unit fills in.
- **Infrastructure landed here:**
  - the oracle generator mode + module-qualified `OpRegistry` keys;
  - the new facade surface gate `PolynomialSurface_IsCoveredOrExplicitlyClassified`;
  - `LeakCatalogue` fixtures;
  - the coverage-artifact regeneration.
- **Parity:** portable, bit-exact (pure `+`/`-`, compares, copies).
- **Tests to port:** `test_polyutils.py`; `TestConstants`; the add/sub/trim/line cases of
  `TestArithmetic`/`TestMisc` in all six files.

### U2 — Series algebra: products, division, powers, roots → series, basis conversion (40 names)

- **Scope:** `{p}mulx`, `{p}mul`, `{p}div`, `{p}pow`, `{p}fromroots` (30) + `X2poly`/`poly2X` for the five
  non-power bases (10).
- **Shared backend:** three engines behind one "series product" interface:
  - (a) **convolution engine**: power (`np.convolve`, synthetic division) and Chebyshev (z-series
    `_cseries_to_zseries`/`_zseries_mul`/`_zseries_div`);
  - (b) **three-term-recurrence engine**: Legendre, Laguerre, Hermite and HermiteE (mul = backward recurrence
    over series via `mulx`);
  - (c) the generic `_div`/`_pow` drivers.
  - `fromroots` = `_fromroots(line, mul)`; conversions compose `mulx`/`add`/`sub` of both bases.
- **Natural sub-deliveries:** U2a (convolution bases) and U2b (recurrence bases).
- **Parity:**
  - **Recurrence bases:** portable and bit-exact at every size (no reductions).
  - **Convolution bases:** inherit **`np.convolve`'s switch to BLAS `?dot`**. Probed: real `polymul`/`chebmul`
    match a sequential sum while the operand (or z-series) length is ≲ 12–15, then diverge (16 → 34/50
    random cases, 32+ → 50/50; `chebmul` degree ≥ 8 → 35–40/40). **Complex products differ from a naive sum
    at every length** (`zdotu` association). So: byte-exact everywhere for short real series; long real and
    all complex products are byte-exact **with the backend installed** (the existing `ISlidingDotBackend`
    seam) and a bounded-ULP `allclose` without it. That is the documented `np.correlate` split, and it needs
    no new code.
- **Perf headroom (NumPy µs):** `legmul` deg 50 **1,996**, `legdiv` 50/10 **7,831**, `legfromroots` 20
  **517**, `cheb2poly` deg 20 **186** — interpreter-bound loops. A span port of the *same step sequence*
  should land two or more orders of magnitude faster.
- **Tests to port:** `TestArithmetic` (mul/div/pow/mulx) and `TestMisc` (fromroots/2poly) × 6.
  `test_polynomial.TestFraction` (Fraction object arrays) → `[Misaligned]` (no object dtype).

### U3 — Evaluation (37 names)

- **Scope:** `{p}val`, `{p}val2d`, `{p}val3d`, `{p}grid2d`, `{p}grid3d`, `{p}valnd` (devdocs), and
  `polyvalfromroots`.
- **Shared backend:** a **Clenshaw/Horner driver** (6 step tables) + `_valnd`/`_gridnd` + the `tensor`
  reshape rule (result = `c.shape[1:] + x.shape`). The driver is **generic over the "x algebra"** from day one
  (§7 R5): `NDArray` x now, and *series-valued* x later, which U10's `convert`/`cast` need.
- **Specialized path:** a fused per-element Clenshaw, SIMD across `x` lanes with broadcast scalar
  coefficients. Every lane executes the source's exact op sequence, so the result is bit-exact.
  - NumPy runs about three full-array ops per coefficient, each allocating: `chebval` deg 10 costs
    **4,205 µs @100K** and **467 ms @10M**; `polyval` deg 10 **356 ms @10M**; `chebval2d` (5×5) **14,718 µs
    @100K**.
  - The fused kernel reads `x` once and writes `y` once. The target is ≥10× at 100K, memory-bound at 10M.
    This is the legacy `polyval` `out=`-Horner precedent: 2.4–8.5× there.
- **Parity:** portable and bit-exact (`+ - * /` only). `polyvalfromroots` = `np.prod(x - r, axis=0)`, an
  axis-0 reduction whose order NumSharp's `prod` must match; the oracle checks it.
- **Tests to port:** `TestEvaluation` × 6.

### U4 — Calculus (12 names)

- **Scope:** `{p}der`, `{p}int`.
- **Shared backend:** one prologue driver:
  - copy with `ndmin=1`; ints → f64;
  - `_as_int(m, "the order of derivation")`; the verbatim errors;
  - `normalize_axis_index`; `moveaxis`;
  - `cnt >= n → c[:1]*0`;
  - `k` padding with "Too many integration constants";
  - the `lbnd`/`scl` scalar checks;
  - `tmp[0] += k[i] - {p}val(lbnd, tmp)` (from U3).
  - Per basis: a 2–4 line recurrence body.
- **Depends on:** U3.
- **Parity:** portable, bit-exact.
- **Perf:** NumPy `chebder` deg 50 **23 µs**, `legint` deg 50 **34 µs**.
- **Tests to port:** `TestIntegral`, `TestDerivative` × 6.

### U5 — Vandermonde (18 names)

- **Scope:** `{p}vander`, `{p}vander2d`, `{p}vander3d`.
- **Shared backend:** a forward-recurrence driver (the same step constants as U3's table, used forward),
  followed by `_vander_nd` (broadcast product over `_nth_slice` axes) and `_vander_nd_flat`.
- **Output layout per D9:** a `moveaxis` view.
- **Parity:** portable, bit-exact.
- **Specialized path:** write straight into the final layout.
- **Perf:** NumPy `chebvander` deg 10 **4,203 µs @100K**, `polyvander2d` (3,3) **5,191 µs @100K**.
- **Closes half of #496.**
- **Tests to port:** `TestVander` × 6.

### U6 — Least-squares fit (6 names)

- **Scope:** `{p}fit`. The class `.fit` is in U10.
- **Shared backend:** `pu._fit`:
  - validation, whose texts are TypeError-heavy (Appendix C);
  - `x+0.0`, `y+0.0`; deg as int or as a list of degrees (sorted, column-selected, re-expanded with zeros);
  - weights; `rcond = len(x)·eps`;
  - column norms `sqrt(square(lhs).sum(1))`, a contiguous-row reduction whose pairwise order must match;
  - **`np.linalg.lstsq`**;
  - `full` → `(c, [resids, rank, s, rcond])`.
  - `RankWarning` is not emitted (house convention).
- **Depends on:** U5.
- **Parity:** **host-pinned**: rows go into `linalg_parity` (like legacy `polyfit`). Without the backend it
  raises NotSupported after NumPy-exact validation.
- **Perf:** NumPy `chebfit` deg 10 @1K **80 µs** (BLAS-bound; NumSharp can only shave the Python wrapper).
- **Closes #496** (together with U5).
- **Tests to port:** `TestFitting` × 6.

### U7 — Companion and roots (12 names)

- **Scope:** `{p}companion`, `{p}roots`.
- **Shared backend:**
  - the per-basis companion builder: pure arithmetic + `sqrt`, IEEE-exact; `hermcompanion`'s scale is
    `np.multiply.accumulate`, i.e. sequential cumprod;
  - the roots driver:
    - `len < 2` → an empty array of `c.dtype`;
    - `len == 2` → closed form (basis-specific: `-c0/c1`, `1 + c0/c1`, `-0.5·c0/c1`);
    - otherwise the rotated companion (non-power bases) → **`np.linalg.eigvals`** → complex-lexicographic sort.
- **Parity:** `companion` is a portable, bit-exact tier; `roots` of degree ≥ 2 is host-pinned; `roots` of
  degree ≤ 1 is portable.
- **Perf:** NumPy `chebroots` deg 20 **45 µs**, `polyroots` deg 50 **204 µs** (LAPACK-bound).
- **Tests to port:** `TestCompanion` + the roots cases of `TestMisc` × 6.

### U8 — Gauss quadrature and weights (10 names)

- **Scope:** `{p}gauss` and `{p}weight` for the five non-power bases.
- **Shared backend:**
  - the gauss driver: companion (U7) → **`np.linalg.eigvalsh`** → one Newton step → `max|·|` scaling →
    symmetrisation (not for Laguerre) → normalisation;
  - the Newton step uses `{p}val`/`{p}der` (U3/U4) for Legendre/Laguerre and the `_normed_hermite(_e)_n`
    recurrences for Hermite/HermiteE;
  - `chebgauss` is closed form (`cos`);
  - weights are elementwise.
- **Depends on:** U3, U4, U7.
- **Parity:**
  - `leg/lag/herm/hermegauss`: host-pinned (eigvalsh).
  - `chebgauss` and the `exp`-based weights (`herm`, `herme`, `lag`): **host-libm** tier (float64 `cos`/`exp`
    are the CRT on both sides → bit-exact on win-amd64; `RunHostLibmCorpus`, like `windows`/`sinc`).
  - `cheb`/`leg` weights: portable (`sqrt` is IEEE-exact).
- **Perf:** NumPy `leggauss(50)` **478 µs**, `chebgauss(50)` **3.4 µs**.
- **Tests to port:** `TestGauss` + the weight cases of `TestMisc` × 5.

### U9 — Chebyshev sampling and interpolation (3 names)

- **Scope:** `chebpts1`, `chebpts2`, `chebinterpolate`. `Chebyshev.interpolate` is in U10.
- **Backend:**
  - `chebpts1`: `sin` of a scaled `arange`;
  - `chebpts2`: `cos(linspace(-π, 0, n))`;
  - `chebinterpolate`: `chebpts1`, a user callable (`Func<NDArray, NDArray>` plus an `args` overload),
    `chebvander` (U5), `np.dot(m.T, y)`, and two in-place divides.
- **Parity:** `pts1`/`pts2` are host-libm. `chebinterpolate` is host-pinned (gemv) and computes managed.
- **Perf:** NumPy `chebpts1(100)` **2.6 µs**, `chebinterpolate(sin, 20)` **59 µs**.
- **Tests to port:** `TestInterpolate` + the `chebpts*` cases of `TestMisc`.

### U10 — Convenience classes (6 classes + base)

- **Scope:** `NDPolyBase`/`NDPolyBase<TSelf>` + the six classes and their facade aliases:
  - **construction:** `as_series` (untrimmed), the two-element domain/window check, `symbol` validation (a
    Python-identifier rule);
  - **properties:** `coef`, `domain`, `window`, `symbol`, `maxpower` (100), `basis_name` (`None`/`T`/`P`/`L`/
    `H`/`He`);
  - **comparison helpers:** `has_samecoef`/`samedomain`/`samewindow`/`sametype`; `_get_coefficients` with its
    verbatim TypeErrors;
  - **operators:** `+ - *`, `/` (numbers only, verbatim TypeError), `%`, unary `-`/`+`, `==`/`!=` (with
    symbol); methods for `//`, `divmod` and `**` (`pow`, honouring `maxpower`);
  - **call and sequence protocol:** `Call(x)` (= `mapdomain` then `_val`), iteration/`__len__`, `degree`;
  - **trimming and conversion:** `copy`, `cutdeg`, `trim`, `truncate`, `mapparms`, `convert`/`cast`;
  - **calculus and roots:** `integ`/`deriv` (with domain scaling), `roots` (mapped back through the domain),
    `linspace`;
  - **class methods:** `fit`, `fromroots` (`domain=[]` vs `None` semantics), `identity`, `basis`,
    `Chebyshev.interpolate`.
- **Shared backend:** everything in U1–U9, bound through the 12 virtual functions.
- **`convert`/`cast` is the subtle one:** NumPy evaluates the series *at the identity series of the target
  kind*: `self(kind.identity(...))`. Horner/Clenshaw therefore run with a **series as `x`**, going through the
  class operators, `x * 0` included. Byte parity needs that exact sequence, which is why U3's driver is generic
  over the x algebra.
- **Depends on:** U1–U9.
- **Parity:** portable except `fit`/`roots` (inherited from U6/U7).
- **Perf:** NumPy `p + q` **12 µs**, `p * q` **15 µs**, `ch(x100k)` **6,122 µs**, `ch.convert(kind=Polynomial)`
  **337 µs**, `Chebyshev.fit` **98 µs**.
- **Tests to port:** `test_classes.py` (33 tests × 6 classes), `test_symbol.py` (31).

### U11 — Printing (2 names + class rendering)

- **Scope:**
  - `polyutils.format_float`;
  - `__str__` (ASCII terms ` x**2`, ` T_1(x)`; Unicode terms `·x²`, `·T₁(x)`; scaled-argument parentheses from
    `mapparms`; line-wrapping at `printoptions.linewidth`);
  - `__format__`, as `IFormattable` with `"ascii"`/`"unicode"`/`""`, so `$"{p:unicode}"` works;
  - `__repr__` (`repr(coef)[6:-1]` = `coef.ToString(true)`, the `poly1d` trick);
  - `_repr_latex_`;
  - `set_default_printstyle`.
- **Platform default:** NumPy's is `os.name != 'nt'`, i.e. **ASCII on Windows** (probed: `_use_unicode ==
  False` here) → `!OperatingSystem.IsWindows()`.
- **Prerequisite:** `np.format_float_positional`/`format_float_scientific` take only `double` today. They need
  `float`/`Half` overloads (the internal `Dragon4` already takes a `FloatKind`); otherwise float32
  coefficients print as their double expansion (`0.10000000149011612` instead of `0.1`).
- **Perf:** NumPy `str(ch)` **38 µs**, `repr(ch)` **89 µs**.
- **Tests to port:** `test_printing.py` (44).

---

## 4. Order, dependencies, parallel lanes

```
U1 ──┬──> U3 ──> U4 ──────────┐
     ├──> U2 ─────────────────┤
     ├──> U5 ──┬──> U6        ├──> U8 ──> U10 ──> U11
     │         └──> U9 ───────┤
     └──> U7 ─────────────────┘
```

(U8 needs U3 + U4 + U7; U10 needs every unit; U9 needs U5.)

**Recommended sequence: U1 → U3 → U2 → U4 → U5 → U6 → U7 → U8 → U9 → U10 → U11.**
- U3 comes right after U1: it needs nothing else, it is the most-used surface, and it is the biggest perf win.
- U2 then unblocks `fromroots`/`pow` and the class layer's `_mul`/`_div`/`_pow`.
- U5 + U6 close #496.

After U1 there are four independent lanes: {U2}, {U3 → U4}, {U5 → U6, U9}, {U7}. Their only shared files are
the append-only registries: `gen_oracle.py`, `OpRegistry.cs`, `MisalignedRegistry.cs`, `LeakCatalogue.cs` and
the surface gate. Keep each unit's edits to those files append-only, and land them as index-only commits
(`git apply --cached` of just the unit's hunks, diffed against the live `HEAD`) so a concurrent session's
edits to the same files survive.

---

## 5. Definition of Done — per unit

1. **API:** every name of the unit on its facade(s), with NumPy's parameter names, order and defaults
   (`tensor=true`, `m=1`, `scl=1`, `axis=0`, `k=[]`, `lbnd=0`, `rcond=null`, `full=false`, `w=null`;
   `polypow maxpower=null`, the other five `=16`).
2. **Docs:** full XML docs on every member (house rule): consequence/when-to-use summaries, params, returns
   and exceptions, the coefficient-order warning where D5 applies, and inline WHY comments on the ported
   step expressions ("order is the NumPy source's; do not refactor").
3. **Unit tests:** port the unit's NumPy test classes (§3) and add the probed dtype (f16/f32/f64/c128 +
   int→f64 + bool ValueError), layout (C/F/strided/reversed/offset/broadcast `x`) and edge
   (empty/zero series/NaN/±inf/−0.0) cases.
4. **Oracle:**
   - new tier file(s) with module-qualified op keys; complex128 in strict tiers;
   - the **teeth-check**: flip 1 ULP and see it fail. The byte-exact single-operand ops must be carved out of
     `MisalignedRegistry`'s "unary ~ULP" blanket excuse via `ByteExactArithmeticUnaryOps`, or a drift passes
     silently;
   - backend-bound rows go to `linalg_parity`, libm-bound rows to a host-libm tier;
   - Decimal through the independent C# decimal oracle.
5. **Surface gate:** `PolynomialSurface_IsCoveredOrExplicitlyClassified` grows with the unit (no unclassified
   public member).
6. **Leaks:** measured `LeakCatalogue` entries for every new member (auto-enrolled via `[ModuleName]`); class
   types listed in `OperatorOwners`/`ObjectOwners` (U10).
7. **Coverage:** `python coverage/generate_coverage.py` regenerated; `--check` green.
8. **Benchmarks:** BenchmarkDotNet cells + NumPy twins for the unit's hot members (benchmark skill), reported
   as NPY/NS, with correctness checked before timing.
9. **Docs site:** CLAUDE.md "Supported np.* APIs" + a `docs/website-src/docs/` page (grows per unit, finalised
   with U10/U11).

---

## 6. Pre-registered divergences (`[Misaligned]`, documented up front)

| # | Divergence | Why |
|---|---|---|
| M1 | complex64 coefficients / `x` → complex128 results | NumSharp has one complex width (F1, #569) |
| M2 | no object-dtype fallback in `as_series`: `Fraction`/mixed exotic coefficients → `ValueError("Coefficient arrays have no common type")`; Decimal handled natively | no object dtype |
| M3 | `RankWarning` not emitted by `*fit`/`.fit` | NumSharp emits no warnings (same as legacy `polyfit`) |
| M4 | `p(x)` → `p.Call(x)`; `p // q` → `floordiv`; `divmod(p, q)` → `p.divmod(q)`; `p ** n` → `p.pow(n)`; `__format__` → `IFormattable`; `_repr_latex_` → method; pickle n/a | C# operator set |
| M5 | class attrs `Polynomial.domain/window` → `NDPolynomial.default_domain/default_window`; `np.polynomial.Polynomial.fit(...)` → `NDPolynomial.fit(...)` (or a `using` alias) | CS0102; a member cannot be both invocable and dotted |
| M6 | module constants are read-only | shared singletons (D6) |
| M7 | NumPy-strong scalar `x` (`np.float64(0.5)`) behaves weak | house-wide 0-d-is-weak rule |
| M8 | `symbol` identifier validation approximates Python's `str.isidentifier` (XID_Start/XID_Continue + NFKC) with .NET Unicode categories | no NFKC-identifier API in the BCL |
| M9 | `__hash__ = None` (unhashable) vs .NET `GetHashCode` | decide in U10 (follow the `poly1d` precedent; never silently hash mutable coefficients by value) |

---

## 7. Risks

- **R1 — Step-level bit-exactness of the typed kernels** (complex scalar-vs-array multiply, float16 weak-int
  rounding, operand order). *Mitigation:* the D3 ops probe gates U1; each unit's oracle tier plus the
  teeth-check.
- **R2 — `_fit`'s column norms.** They are `np.sqrt(np.square(lhs).sum(1))` over `van.T`, which is
  C-contiguous rows, i.e. NumPy's pairwise summation. The fit's bits depend on NumSharp's `sum` reproducing it
  for that layout. *Verify first thing in U6* with a probe on the exact `van.T` view.
- **R3 — `np.convolve`'s BLAS switch** for long and complex products (U2). *Mitigation:* the existing
  `ISlidingDotBackend` seam; managed = `allclose`, documented; the oracle's long/complex product rows are
  host-pinned with the backend.
- **R4 — Parallel-session collisions** in the shared registries. *Mitigation:* append-only edits and
  index-only commits; diff against LIVE `HEAD` before landing.
- **R5 — Series-valued evaluation** for `convert`/`cast` (U10). *Mitigation:* U3's driver is generic over the
  x algebra from the start, so U10 plugs a series algebra into the same step code instead of re-deriving it.
- **R6 — Printing needs precision-aware Dragon4 overloads.** *Mitigation:* land the `float`/`Half`
  `format_float_*` overloads as U11's first commit, oracle-tested through the existing printing tier.
- **R7 — `chebinterpolate` takes a Python callable.** *Mitigation:* `Func<NDArray, NDArray>` (+ an
  `args`-forwarding overload). The callable may return any dtype, and NumPy's `np.dot` promotes; mirror that.

---

## 8. NumPy baselines (µs, best-of-7, 2.4.2, `OPENBLAS_NUM_THREADS=1`) and what they imply

| Unit | Call | NumPy | Regime → expectation for NumSharp |
|---|---|---|---|
| U1 | `polyadd` deg5+deg10 | 2.84 | Python-overhead-bound; NumSharp floor ≈ one NDArray allocation → ~2–3× |
| U2 | `polymul` 10×10 / `chebmul` 10×10 | 3.46 / 7.70 | convolve-bound; small |
| U2 | `legmul` deg 10 / deg 50 | 165 / 1,996 | interpreter loop over array ops → span kernel, ≫10× |
| U2 | `legdiv` 50/10, `legfromroots` 20, `cheb2poly` 20 | 7,831 / 517 / 186 | same, ≫10× |
| U3 | `chebval` deg 10 @1K / @100K / @10M | 19 / 4,205 / 467,314 | allocation-bound array recurrence → fused SIMD Clenshaw, ≥10× at 100K |
| U3 | `polyval` deg 10 @10M; `legval` deg 10 @100K | 355,886 / 2,544 | same |
| U3 | `chebval2d` (5×5) @100K | 14,718 | nested `_valnd` → fused, ≥10× |
| U4 | `chebder` / `legint` deg 50 | 22.7 / 33.9 | small-series loops → span kernel |
| U5 | `chebvander` deg 10 @100K; `polyvander2d` (3,3) @100K | 4,203 / 5,191 | write-bound output → direct-layout fill, several × |
| U6 | `chebfit` deg 10 @1K | 80 | LAPACK-bound → parity plus wrapper savings |
| U7 | `chebroots` deg 20 / `polyroots` deg 50 | 45 / 204 | LAPACK-bound → parity plus wrapper savings |
| U8 | `leggauss(50)` / `chebgauss(50)` | 478 / 3.4 | eigvalsh-bound / trivial |
| U9 | `chebpts1(100)` / `chebinterpolate(sin, 20)` | 2.6 / 59 | trivial / gemv-bound |
| U10 | construct / `p+q` / `p*q` / `ch(x@100K)` / `convert` / `.fit` | 1.4 / 12 / 15 / 6,122 / 337 / 98 | object overhead + the U3/U2 kernels |
| U11 | `str(ch)` / `repr(ch)` | 38 / 89 | string building |

Method: `timeit` autorange, best of 7 repeats; coefficients `default_rng(0).standard_normal(deg+1)`,
`x = uniform(-1, 1, N)`; `p = Polynomial(c10)`, `q = Polynomial(c5)`, `ch = Chebyshev(c10)`; the fit
data are `x = uniform(-1, 1, 1000)`, `y = sin(3x)`. These are planning baselines only: re-measure inside the
benchmark harness (BenchmarkDotNet + warm NumPy twin) before quoting any NPY/NS ratio.

---

## Appendix A — operation × basis matrix (the 186 basis-module names; + 7 `polyutils` = 193)

| Operation | Unit | polynomial | chebyshev | legendre | laguerre | hermite | hermite_e |
|---|---|---|---|---|---|---|---|
| domain / zero / one / x | U1 | `polydomain` `polyzero` `polyone` `polyx` | `chebdomain` `chebzero` `chebone` `chebx` | `legdomain` `legzero` `legone` `legx` | `lagdomain` `lagzero` `lagone` `lagx` | `hermdomain` `hermzero` `hermone` `hermx` | `hermedomain` `hermezero` `hermeone` `hermex` |
| line | U1 | `polyline` | `chebline` | `legline` | `lagline` | `hermline` | `hermeline` |
| add | U1 | `polyadd` | `chebadd` | `legadd` | `lagadd` | `hermadd` | `hermeadd` |
| sub | U1 | `polysub` | `chebsub` | `legsub` | `lagsub` | `hermsub` | `hermesub` |
| trim | U1 | `polytrim` | `chebtrim` | `legtrim` | `lagtrim` | `hermtrim` | `hermetrim` |
| mulx | U2 | `polymulx` | `chebmulx` | `legmulx` | `lagmulx` | `hermmulx` | `hermemulx` |
| mul | U2 | `polymul` | `chebmul` | `legmul` | `lagmul` | `hermmul` | `hermemul` |
| div | U2 | `polydiv` | `chebdiv` | `legdiv` | `lagdiv` | `hermdiv` | `hermediv` |
| pow | U2 | `polypow` | `chebpow` | `legpow` | `lagpow` | `hermpow` | `hermepow` |
| fromroots | U2 | `polyfromroots` | `chebfromroots` | `legfromroots` | `lagfromroots` | `hermfromroots` | `hermefromroots` |
| X2poly / poly2X | U2 | — | `cheb2poly` `poly2cheb` | `leg2poly` `poly2leg` | `lag2poly` `poly2lag` | `herm2poly` `poly2herm` | `herme2poly` `poly2herme` |
| val | U3 | `polyval` | `chebval` | `legval` | `lagval` | `hermval` | `hermeval` |
| val2d / val3d | U3 | `polyval2d` `polyval3d` | `chebval2d` `chebval3d` | `legval2d` `legval3d` | `lagval2d` `lagval3d` | `hermval2d` `hermval3d` | `hermeval2d` `hermeval3d` |
| grid2d / grid3d | U3 | `polygrid2d` `polygrid3d` | `chebgrid2d` `chebgrid3d` | `leggrid2d` `leggrid3d` | `laggrid2d` `laggrid3d` | `hermgrid2d` `hermgrid3d` | `hermegrid2d` `hermegrid3d` |
| valnd (devdocs) | U3 | `polyvalnd` | `chebvalnd` | `legvalnd` | `lagvalnd` | `hermvalnd` | `hermevalnd` |
| valfromroots | U3 | `polyvalfromroots` | — | — | — | — | — |
| der | U4 | `polyder` | `chebder` | `legder` | `lagder` | `hermder` | `hermeder` |
| int | U4 | `polyint` | `chebint` | `legint` | `lagint` | `hermint` | `hermeint` |
| vander | U5 | `polyvander` | `chebvander` | `legvander` | `lagvander` | `hermvander` | `hermevander` |
| vander2d / vander3d | U5 | `polyvander2d` `polyvander3d` | `chebvander2d` `chebvander3d` | `legvander2d` `legvander3d` | `lagvander2d` `lagvander3d` | `hermvander2d` `hermvander3d` | `hermevander2d` `hermevander3d` |
| fit | U6 | `polyfit` | `chebfit` | `legfit` | `lagfit` | `hermfit` | `hermefit` |
| companion | U7 | `polycompanion` | `chebcompanion` | `legcompanion` | `lagcompanion` | `hermcompanion` | `hermecompanion` |
| roots | U7 | `polyroots` | `chebroots` | `legroots` | `lagroots` | `hermroots` | `hermeroots` |
| gauss | U8 | — | `chebgauss` | `leggauss` | `laggauss` | `hermgauss` | `hermegauss` |
| weight | U8 | — | `chebweight` | `legweight` | `lagweight` | `hermweight` | `hermeweight` |
| pts1 / pts2 / interpolate | U9 | — | `chebpts1` `chebpts2` `chebinterpolate` | — | — | — | — |

`polyutils`: `as_series` `trimseq` `trimcoef` `getdomain` `mapdomain` `mapparms` → **U1**; `format_float` →
**U11**. Package level: `set_default_printstyle` → **U11**.

Unit totals: U1 54 · U2 40 · U3 37 · U4 12 · U5 18 · U6 6 · U7 12 · U8 10 · U9 3 · U11 1 (+`set_default_printstyle`)
= 193 functions; U10 = the class layer.

## Appendix B — `ABCPolyBase` → C# surface (U10/U11)

| NumPy | C# (proposed) |
|---|---|
| `Poly(coef, domain=None, window=None, symbol='x')` | `new NDPoly(coef, domain, window, symbol)`; alias `np.polynomial.Poly(...)` |
| `p.coef / p.domain / p.window / p.symbol` | same-named instance properties |
| `Poly.domain / Poly.window` (class defaults) | `NDPoly.default_domain / default_window` (M5) |
| `maxpower` (100), `basis_name` | instance read-only properties (+ static defaults) |
| `p(x)` | `p.Call(x)` (the `poly1d` precedent) |
| `+ - *`, `/` (numbers only), `%`, unary `-`/`+`, `==`/`!=` | C# operators on `NDPolyBase<TSelf>` (D11 overload set) |
| `//`, `divmod`, `**` | `floordiv(other)`, `divmod(other)` → `(q, r)`, `pow(n)` (M4) |
| `len(p)`, `iter(p)` | `__len__()`/`Count`, `IEnumerable` over coefficients |
| `copy degree cutdeg trim truncate mapparms` | same-named methods |
| `convert(domain, kind, window)`, `Poly.cast(series, domain, window)` | `convert<TKind>(...)` / `convert(kind: …)`; static `NDPoly.cast(series, …)` |
| `integ(m, k, lbnd)`, `deriv(m)`, `roots()`, `linspace(n, domain)` | same-named methods |
| `Poly.fit(...)`, `Poly.fromroots(...)`, `Poly.identity(...)`, `Poly.basis(...)`, `Chebyshev.interpolate(...)` | static members on the `ND*` type (inherited CRTP statics) |
| `has_samecoef/samedomain/samewindow/sametype` | same-named methods |
| `str(p)`, `format(p, 'ascii'|'unicode')`, `repr(p)`, `p._repr_latex_()` | `ToString()`, `ToString("ascii"|"unicode")` (IFormattable), `repr`/`ToString(true)`-style, `_repr_latex_()` |

## Appendix C — verbatim error texts (probed 2.4.2)

| Call | Exception |
|---|---|
| `polyder(c, m=1.5)` | `TypeError: the order of derivation must be an integer, received 1.5` |
| `polyint(c, k=[1,2])` with m=1 | `ValueError: Too many integration constants` |
| `polyint(c, lbnd=[0,1])` | `ValueError: lbnd must be a scalar.` |
| `polyvander(x, -1)` | `ValueError: deg must be non-negative` |
| `polyfit(x, y, -1)` | `ValueError: expected deg >= 0` |
| `polyfit(x[3], y[2], 1)` | `TypeError: expected x and y to have same length` |
| `chebcompanion([1])` | `ValueError: Series must have maximum degree of at least 1.` |
| `polypow(c, 1.5)` | `ValueError: Power must be a non-negative integer.` |
| `chebpow(c, 17)` / `Polynomial(c) ** 101` | `ValueError: Power is too large` |
| `polydiv(c, [0])` | `ZeroDivisionError` with an **empty** message |
| `polyval2d([1,2], [1], c)` | `ValueError: x, y are incompatible` (3-D: `x, y, z are incompatible`; N-D: `ordinates are incompatible`) |
| `chebpts1(0)` / `chebpts2(1)` | `ValueError: npts must be >= 1` / `npts must be >= 2` |
| `leggauss(0)` | `ValueError: deg must be a positive integer` |
| `polyadd([], [1])` / `polyadd([[1,2],[3,4]], [1])` | `ValueError: Coefficient array is empty` / `Coefficient array is not 1-d` |
| `polyadd([True], [1.0])` | `ValueError: Coefficient arrays have no common type` |
| `Polynomial(c, domain=[1,2,3])` | `ValueError: Domain has wrong number of elements.` |
| `Polynomial(c, symbol='1x')` | `ValueError: Symbol string must be a valid Python identifier` |
| `Polynomial(c) + Chebyshev(c)` | `TypeError: Polynomial types differ` (also `Domains differ`, `Windows differ`; `ValueError: Polynomial symbols differ`) |
| `Polynomial(c) / Polynomial(c)` | `TypeError: unsupported types for true division: '<class 'numpy.polynomial.polynomial.Polynomial'>', '<class …>'` |
| `pu.trimcoef(c, tol=-1)` | `ValueError: tol must be non-negative` |
