# numpy.polynomial — delivery plan in backend-sharing units

> Plan for porting NumPy's **polynomial package** (`numpy.polynomial`: six basis modules, `polyutils`, the
> six convenience classes) to NumSharp, split into **delivery units — each unit a family of functions
> that share one backend driver**, so every driver is written once and instantiated for all six bases.
> The legacy half of the same docs page (`np.poly1d`, `np.polyval`, `np.polyfit`, …) already ships.
> §9 measured the candidate engines. §10 is the engine the house rules require: loops in IL or NDIter,
> no struct kernels, no per-dtype C#. It uses Tier-3A IL kernels driven by NDIter, emitted by one typed
> emitter from per-basis step tables, plus IL primitives over an arena for the small-series algebra. It is
> measured across bases × dtypes × layouts × tensor modes × Vandermonde × series algebra: byte-exact on
> every cell and 2.8–168× NumPy.
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
| "Polynomial package" (`numpy.polynomial`) | 6 modules × ~31 names, `polyutils`, 6 classes, `set_default_printstyle` | **U3 + U1 + U4 delivered** — 102 of 193 names: the evaluation family (36, `polyvalfromroots` open), the additive family with `polyutils` (54: `{p}add/sub/trim/line`, the 24 constants, `as_series`/`trimseq`/`trimcoef`/`getdomain`/`mapparms`/`mapdomain`) and the calculus family (12: `{p}der`/`{p}int`), all bit-exact (`polyeval.jsonl` 16,606 + `polyseries.jsonl` 18,437 + `polycalc.jsonl` 21,646 cases). Facade `np.polynomial.{polynomial,chebyshev,legendre,laguerre,hermite,hermite_e,polyutils}`; 0 of 6 classes. `coverage/generate_coverage.py` catalogues all seven `numpy.polynomial.*` submodules as out-of-headline surfaces. |
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

**D3 — NumPy's per-op semantics, emitted through the house IL emitters (the load-bearing correctness
decision).** NumPy's package is pure Python over array ops. Porting each op to an `NDArray` call would be
correct but construction-bound: `legmul` at degree 50 runs thousands of small array and scalar ops (≈2 ms in
NumPy, most of it in `legmulx`'s per-coefficient loop). The house route instead emits every recurrence as IL.
One **typed emitter** (§10.1) lowers each node of a basis's step table through `EmitScalarOperation` /
`EmitConvertTo` (scalars) or the lane-matched vector operators, so each dtype gets **NumPy's array-loop
semantics op by op** from the same emitters the ufuncs use. There is no per-dtype C# and no struct kernel.
Traps it must encode, most of which have already bitten NumSharp:
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
  doesn't either. The step tables are data, written in source order and never re-associated.
- **The dtype of each op is NumPy's, not the result's** (§10.1 "peeling"). NumPy runs Clenshaw's first one
  or two steps in the *coefficient* dtype. Casting float32 coefficients to float64 up front changes the result
  on 100% of points for leg/lag/herm/herme (probed, §10.3 C'). x-only subexpressions (`2*x`, `(2*nd-1) - x`,
  `1 - x`) run in *x's* dtype: an int8 x wraps, and a Python int that does not fit raises NumPy's
  `OverflowError`.
- **N-D coefficient arrays** (the `axis` of der/int, multi-series `c` in val) run the same kernels through
  NDIter's operand form (§10.1). There is never a second implementation.
- **Proof before use:** U1 ships a differential "ops probe" (every step expression × all 5 dtypes × random,
  special and adversarial values, vs NumPy) that U2+ build on.
- **Measured (§10.3):** every evaluation, Vandermonde and series cell is byte-exact against NumPy, at 2.8–168×
  its speed. For contrast, the 1-to-1 `NDArray` port of §9 reaches only **0.61–10.6×**, and **0.26×** when
  element access keeps NumPy's literal `c[i]` (0-d view) spelling.

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

**D6 — Constants are shared, WRITEABLE singletons, dtype-exact** (as built in U1, 2026-09-27; the original plan
made them read-only, which would have been a needless divergence). `polydomain`, `chebzero` and the rest are
mutable module-level ndarrays in NumPy, the SAME object on every attribute access, so a write through one
persists. NumSharp reproduces exactly that: one process-wide instance each (`NDPolySeries.Constant`), writeable,
**detached from every `NDScope`** (the `np.ma.nomask` `Own(...)` pattern: a singleton first built inside a caller's
scope would be disposed with it) and holding one extra buffer reference, so a caller's `Dispose()` of the shared
object cannot free the memory every later read still uses. The corpus pins four facets of each (value, identity,
writeable, owndata). Their dtypes are NumPy's literal ones, probed from the source:
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

**D12 — Engine: Tier-3A IL kernels driven by NDIter, one typed emitter over per-basis step tables
(measured, §10).** This decision replaces the first engine verdict (fused plain-C# kernels generic over a
per-basis step struct), which the house rules forbid: no struct kernels, no per-dtype C#, every loop in IL or
NDIter.
- **Keep NumPy's algorithm structure and operation order exactly.** Every strategy measured is byte-exact
  against NumPy *because* it keeps that order (§9 finding 1); the engine only decides speed and memory.
- **Evaluation, `_valnd`/`_gridnd`, Vandermonde and `_vander_nd`** (U3/U5, class `__call__`): Tier-3A raw
  inner loops (`DirectILKernelGenerator.CompileRawInnerLoop`) driven by `NDIterRef.ForEach` with auxdata.
  - One typed emitter builds them from the six step tables, using NumPy's per-op dtypes, peeling to a dtype
    fixpoint and a weak-constant pool.
  - Each block runs four interleaved chains; a point's recurrence is one serial dependency chain, and the
    interleave is 1.2–1.9× over a single chain.
  - Details and measurements: §10.1 and §10.3.
- **Every dtype takes the same emitter; there is no homogeneous-dtype gate.**
  - float32 and float64 run vector chains, including MIXED x/c dtypes, via exact lane-matched conversions
    (float32/int32/int64 x → float64 lanes, float16 x → float32 lanes). Coefficient-only math stays a shared
    scalar in NumPy's dtype.
  - float16 runs vector chains through the house AVX2 widen/narrow primitives.
  - Complex, decimal and the pairs with no lane conversion run scalar chains, still 4-wide interleaved.
- **Operand forms and buffering.**
  - A 1-D `c` rides in auxdata as operands `(x, y)`. A non-contiguous `x` then runs BUFFERED+CONTIG, 2.7–3.0×
    faster than strided scalar chains.
  - An N-D `c` becomes operands `(x, c[0], y)` and is **never buffered**, because the kernel reads past the
    `c[0]` operand.
- **Small-series algebra** (U1/U2/U4, companion builders, class operators) is built from four pieces:
  - dtype-generic IL **primitive** kernels;
  - **whole-routine** IL kernels where the routine is one flat recurrence (`chebder`);
  - dtype-agnostic orchestration mirroring NumPy's Python line for line, whose only loops are NumPy's own
    loops over series, never over elements;
  - a bump **arena** for intermediates.
  Never `NDArray` ops per coefficient, never 0-d `c[i]` element views.
- **Do not write a loop over full-size arrays as a plain `[NDScoped]` port.** A scope keeps every temporary
  alive until it closes: 1,733 MB at 10M points / degree 10, against 80 MB for the out= port, and it collapses
  to 0.21× NumPy at degree 30 / 100K points.
- **Not used:**
  - **NDExpr** cannot express Clenshaw (five of the six bases); the tree grows exponentially (§9 finding 4).
  - **The Tier-3B shell** wraps a one-chain body in the house 4×-unrolled loop. That unrolls one dependency
    chain instead of interleaving independent ones, and it is 1.15–1.46× slower than Tier 3A (§10.3 B).
  - **Struct kernels / per-dtype C#** are forbidden by the house rules. §9's `Span` variant is kept only as
    the speed reference: Tier 3A lands within 10–20% of it.
- **Without dynamic code** (NativeAOT) the IL route is unavailable, as for every house IL kernel. The package
  then falls back to the out= composition over NumSharp's own ufuncs: exact by construction, 1–3× NumPy
  (§9 `P1out`).
- **Glue stays a plain composition:** validation, the `_fit`/`_fromroots` drivers, the class layer and the
  BLAS-bound members. Their cost is the leaf kernels or LAPACK.

---

## 3. The delivery units

Every unit ships **for all six bases at once**: one shared driver plus six step tables. Names are
machine-partitioned; the counts sum to 193 with no overlap (Appendix A).

### U1 — Substrate, facades and the additive family (54 names) — DELIVERED 2026-09-27

**As built** (the planned items below are kept for the record; where the build diverged, this block wins):
- **Engine:** `Polynomial/Package/NDPolyNumber.cs` (`PolyNumber` — a Python scalar, a NumPy scalar or an ndarray,
  with NumPy's operator dispatch between the three arithmetics: CPython, scalarmath, ufunc) and
  `Polynomial/Package/NDPolySeries.cs` (the functions). Element loops are IL kernels in
  `Backends/Kernels/Direct/DirectILKernelGenerator.PolySeries.cs` (trim scan, NumPy's in-place `_add`/`_sub`
  combine, tolerance scan, cast, one-element scalarmath ops, and a fused `mapparms` for same-dtype ndarray domains).
  CPython's arithmetic lives on `PyScalar` (`IntTrueDivide` = `long_true_divide`, `ComplexQuotient` = 3.12
  `_Py_c_quot`, and NaN-priority `Float*`/`Complex*` helpers — CPython's per-operator NaN operand order, which a C#
  operator does not pin). The "series primitives + arena", the "ops probe" and the step-table format planned here
  were not needed by the additive family (U3 had already landed the typed emitter and step tables); they move to
  U2, the first unit that needs a series arena.
- **Fast paths, each proven against the exact general lane:** generic tuple overloads of `mapparms`/`mapdomain`
  (one per x kind — `double`, `Complex`, `NDArray` — or a complex x becomes CS0121-ambiguous); a CPython
  machine-number lane (`PyNum`: int within `long` / float / complex, bailing to exact big integers on overflow) for
  tuples and lists; the fused scalarmath kernel for two 1-D ndarray domains of one dtype; per-thread reusable 0-d
  parameters for `mapdomain`'s one `np.evaluate` pass.
- **Oracle:** `polyseries.jsonl` (15,950 cases at delivery, 18,437 after the audit below; `gen_oracle.py polyseries`,
  0 excused). The registry replays every `mapparms`/`mapdomain` case through three routes — the object overload, the
  generic overload (tuples rebuilt element-typed by reflection) and `NDPolySeries.MapParmsGeneral` (the exact
  reference) — which must agree to the byte. Unit tests: `Polynomial/PolynomialSeriesTests.cs` (38, incl. a
  22,400-comparison lane-vs-reference property test and, since the audit, the affine-vs-fused mapdomain route check
  and the blocked-vs-copy getdomain checks).
- **Perf (NPY/NS, pinned, best-of-7, 867 cells after the audit below):** add/sub 1.70–24.3× (geo 6.7), trim
  5.5–515×, trimseq 3.8–490×, getdomain 2.13–104× (geo 8.3), mapdomain 1.96–33× (geo 4.9), as_series 1.11–10.3×,
  mapparms 1.18–8.7×; overall geomean 8.47×. `{p}line` is 0.80–1.03× — bound by the ~230 ns NDArray allocation
  (NumPy's whole `np.array([off, scl])` is 260–430 ns), not by the algorithm.
- **Parity + perf audit (2026-09-27/28).** The first corpus stopped at a dozen coefficients, so no cell reached the
  routes a long series takes. Added: corpus sections K (1,458 long-series cases across the 64-element house threshold,
  every dtype pair and layout, specials in both the updated prefix and the untouched tail), L (552 complex64 ARRAY
  loops of mapdomain, value-compared — NumSharp emulates complex64 on its float32 kernels with NumPy's product forms)
  M (45 cases on the 1,024-element block boundaries and the strided-staging threshold) and N (72 getdomain cases
  across the blocked pass's 8 KB windows, incl. NumPy's lane-order ±0 tie and canonical-vs-payload NaN) — 18,437
  cases, 0 excused.
  The long routes: `{p}add`/`{p}sub` through the house SIMD kernels block by block (`CombineViaHouseKernels`), the
  conversion copy through the house casts / SIMD strided copies (`CopyInto`), `getdomain` in ONE blocked pass (NumPy's
  copy taken 8 KB at a time — in place, or packed into L1 — and folded for min AND max by the engine's exact
  schedule, bit-identical to reducing the whole copy; an integer x read in any order, uint64 packed as float64), and
  mapdomain's `MapDomainAffine` (points converted 8 KB at a time into an L1 scratch, then a
  multiply-then-add kernel — never an FMA — or `simd_cmul`; the fused pass keeps the shapes it runs at vector speed).
  New house kernels, library-wide and bit-exact: AVX2 int → float casts incl. int64/uint64 → float64 by the
  exponent-bias splice (one RNE rounding), float16 → float64 as NumPy's `ToDoubleBits`, the 4/8-byte SIMD strided
  copy, and packed float32 ↔ float64 in `EmitConvertTo` (the scalar `cvtss2sd` carries a false dependency). Full
  benchmark matrix (867 cells, NumPy re-measured back to back): see the CLAUDE.md U1 section; the 26 cells still
  under 1.5× are `{p}line` (21, the allocation floor), 100K `as_series` of two arrays (4, memcpy: two plain copies
  into reused buffers take the whole call's time), and `mapparms` of a mixed ndarray/tuple domain (1, 1.18× — the
  exact general lane's per-operation dispatch; a typed program per argument-kind signature is the lever).

- **Scope:**
  - facades: `np.polynomial` + 7 submodules, `[ModuleName]` ×8;
  - `polyutils`: `as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapdomain`, `mapparms`;
  - constants `{p}domain/{p}zero/{p}one/{p}x` (24), `{p}line` (6);
  - `{p}add`/`{p}sub` (12) and `{p}trim` (6, NumPy aliases of `trimcoef`).
- **Shared backend:** `as_series` (`np.common_type` + `trimseq`), `_add`/`_sub`. Also landed here:
  - the **typed IL emitter** and the **series primitives + arena** (D3, D12, §10.1);
  - the **ops probe**;
  - the **step-table** format that every later unit fills in.
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
  **517**, `cheb2poly` deg 20 **186** — interpreter-bound loops.
- **Engine (D12, §10.1):** IL primitive kernels plus NumPy-mirroring orchestration over a bump arena.
- **Measured (§10.3 S), all byte-exact:**
  - `legmul` 10×10: **114×** (float64), **127×** (float32), **82×** (complex128), **18×** (float16).
  - `legmul` 50×50: **97×**.
  - `legdiv` 50/10: **168×** (float64), **82×** (complex128).
  - For contrast, the 1-to-1 `NDArray` port of §9 manages only **1.9–3.5×**, and `chebmul` as a port is
    *slower* than NumPy (**0.61×**).
- **Dtype quirk to encode (probed):** the `len(c) == 1` branch of `legmul`/`hermmul`/`lagmul` sets `c1 = 0`,
  a Python int. That int flows through `as_series` → `common_type`, so the **result is float64 for float16 or
  float32 series**. `chebmul` has no such branch and keeps the dtype. The orchestration must therefore carry
  a dtype per series.
- **Tests to port:** `TestArithmetic` (mul/div/pow/mulx) and `TestMisc` (fromroots/2poly) × 6.
  `test_polynomial.TestFraction` (Fraction object arrays) → `[Misaligned]` (no object dtype).

### U3 — Evaluation (37 names) — DELIVERED 2026-09-25 (36 of 37; `polyvalfromroots` open)

- **Scope delivered:** `{p}val`, `{p}val2d`, `{p}val3d`, `{p}grid2d`, `{p}grid3d` and `{p}valnd` (devdocs)
  for all six bases, under the new facade `np.polynomial.{polynomial,chebyshev,legendre,laguerre,hermite,
  hermite_e}` (instance-property modules, §2). **Still open:** `polyvalfromroots` (`np.prod(x - r, axis=0)`,
  an axis-0 reduction whose order NumSharp's `prod` must match).
- **Engine as built (D12, §10.1):** Tier-3A per-chunk IL kernels driven by `NDIterRef.ForEach`
  (`src/NumSharp.Core/Backends/Kernels/ILKernelGenerator.Polynomial*.cs`, driver
  `src/NumSharp.Core/Polynomial/Package/NDPolyEval.cs`):
  - **Step tables as data** (`ILKernelGenerator.Polynomial.cs`): NumPy's source-order expression trees per
    basis, never re-associated, typed node by node with NumPy's NEP 50 rules (`...Typing.cs`) — strong
    `promote_types`, weak Python int/float/complex, int true-divide → float64. NumPy runs the first 1–2
    Clenshaw steps in the COEFFICIENT dtype, so the emitter **peels** to a dtype fixpoint; kernel class =
    `min(nc, P+3)` for Clenshaw, 1 for Horner.
  - **Weak (Python-scalar) x** folds into weak constants evaluated with CPython 3.12 arithmetic
    (`PyScalar`, BigInteger ints); a 0-d NDArray x is STRONG. Weak values reach the kernel through a
    per-(constant, dtype) region pool (`...ConstPool.cs`), converted by astype's own cast path, with NumPy's
    `OverflowError` for a Python int that does not fit an integer x.
  - **Complex has THREE NumPy multiplies** (a third one was found by the 2026-09-26 review):
    - An ARRAY op is the fused `simd_cmul`.
    - A 0-d result is scalar math (naive product), except where an operand is the raw 0-d x array.
    - An N-D series at a per-point x with a ONE-element result, where `c[k]` and x differ in ndim, is
      NpyIter's stride-0 iteration. That runs `CDOUBLE_multiply`'s MSVC-contracted fallback:
      im = `fma(ai, br, ar*bi)`.

    Reproduced per op (`PolyComplexProduct`, `PolyUnitBroadcast`: values carry NumPy's ndim, and the peel
    reaches the joint dtype+ndim fixpoint). See [numpy-polynomial-review.md](numpy-polynomial-review.md).
  - **Vector lanes for every dtype pair** (`...Lanes.cs`): 256-bit chains on AVX2 hosts (float64 4 lanes,
    float32/float16 8, complex128 2), and every other per-point dtype held at the SAME lane count —
    int32 containers re-wrapped to int8/uint8/int16/uint16/char width after every op, packed 2-lane
    float32/int32/uint32, float16 as 8 float32 lanes with 2/4/8 live, complex128 `[re, im]×2` with NumPy's
    `simd_cmul` (vfmaddsub) and Smith division (a shared divisor prepared once per block) — each conversion
    exact or rounded once as NumPy's cast is (uint64/int64 via .NET's ConvertToDouble — the scalar tail's
    EmitConvertTo rounds once too since ebe18b7c; gated at the ties above 2^63 by the review probe). A lane table
    miss falls back to scalar chains (same bits), counted by `ILKernelGenerator.PolyVectorFallbacks` — zero
    across the whole dtype matrix (unit-tested).
  - **One DynamicMethod per STAGE** (dispatcher → part → U-chain stage → remainder stage): the JIT stops
    inlining in a method once its inline budget (local-variable limit, time budget) is spent, so one method
    holding every block left the lane helpers as calls — measured 340 un-inlined calls in a 23 KB complex kernel (3.3× slower); one method per part
    still left 95 in its float16-series twin (0.93× NumPy). Per stage: 0–2 calls.
  - `_valnd`/`_gridnd` stay NumPy's two-pass compositions over the N-D coefficient operand form (never
    buffered: the kernel reads past `c[0]`).
- **Parity:** `test/NumSharp.Tests.Oracle/Fuzz/corpus/polyeval.jsonl` (`gen_oracle.py polyeval`, **16,606
  cases, 0 excused**; 4,392 of them are section (I), the single-element broadcasts added by the review):
  the x-dtype × series-dtype × count-class matrix, x layouts, N-D series (tensor
  True/False, series layouts), weak and 0-d scalars incl. 2**64-1 / NaN / ±inf / complex, special
  coefficients, errors, 2-D/3-D/N-D, the lagval int8/uint8 OverflowError, and — section (H) — every x dtype
  × every inexact series dtype at 45 points (U-chain stage + 1-chain stage + scalar tail at every lane width)
  as a 1-D and a per-point series, at the dtype extremes (integer wrap-around, NaN/±inf/subnormal/overflow),
  plus column-strided per-point series (the scalar part). `MisalignedRegistry`'s generic unary/complex
  ULP excuses are carved out for these ops. Unit tests `test/NumSharp.Tests/Polynomial/
  PolynomialEvaluationTests.cs` (17): NumPy bytes per basis, peeled float32, weak vs strong, complex
  scalar-math/ufunc mix, the single-element-broadcast fallback product (series deeper and x deeper),
  float16 NaN priority (vector AND scalar path), int8 wrap, shapes/errors/layouts,
  Decimal/Char, thread safety, zero vector fallbacks over the dtype matrix, vector part ≡ scalar part byte
  for byte. Mutation-checked (below).
- **Measured (NPY/NS, pinned to 4 logical CPUs on both sides, Release, drained JIT warm-up, every cell
  SHA-256-checked against NumPy's result BEFORE it is timed — `benchmark/polynomial/polyeval_{numpy.py,
  bench.cs}`):**

  | Section | Cells | min | geomean | What it covers |
  |---|---|---|---|---|
  | E | 48 | 2.53 | 11.7 | six bases × degree 3/10/30 × 1K/100K/10M, float64 |
  | T | 162 | 2.52 | 9.7 | 21 dtype pairs @100K + float32/float16/complex128 @1K/10M |
  | L | 54 | 2.43 | 7.8 | strided/reversed/transposed/F/broadcast/2-D/3-D x |
  | N | 36 | 2.24 | 10.1 | multi-series tensor=True, per-point series, series layouts, complex per point |
  | S | 42 | 2.73 | 6.5 | Python-scalar and 0-d x against 1-D and N-D series |
  | D | 42 | 3.01 | 14.9 | val2d/val3d/grid2d/grid3d/valnd |
  | M | 936 | 1.62 | 6.7 | every x dtype × every inexact series dtype @1K/10K, 1-D and per point |
  | **all** | **1,320** | **1.62** | **7.5** | 0 byte mismatches |

  Representative: `chebval` d10 float64 **16.9× / 11.2× / 36.5×** (1K/100K/10M); `lagval` (two divisions
  per step, divider-bound) **7.2× / 2.7× / 14.1×**; complex128 `chebval` @100K **23.7×**; float16 @10M
  **3.3×**; `chebval2d` @100K **41.5×**; `polyval(0.37, c)` **4.8×** (0.44 µs vs 2.10 µs).
- **What the first measurement got wrong (keep for the next family):** (1) the 100K-only dtype section
  hid 227 sub-1.5× cells — NumPy's own loops are cheapest when L1/L2-resident (1K–10K), which is why the M
  section exists; (2) a single warm-up pass only QUEUES methods for tier-1, so the first-timed cells read
  4–5× slow — the harness now warms, sleeps 250 ms for the background JIT, and repeats; (3) `dotnet run
  file.cs` reuses its cached build — referenced NumSharp.Core included — when the script itself did not
  change, so a whole 936-cell run silently timed the previous kernels: `--no-cache` is mandatory.
- **Mutation check (every one killed):** 7 engine mutations — no peeling, no naive scalar-math complex
  product, Horner operand order, weak x treated as strong, Legendre re-association, truncating Python
  int → double (oracle), and the float16 NaN-priority blend removed (unit test only: the oracle tokenizes NaN
  payloads) — plus 8 lane mutations — int8 wrap width, uint16 wrap mask, complex multiply fused/addend swap,
  shared complex divide scale, uint32→float64 bias, float16 gang-load lane, float64→complex imaginary lane,
  float16 4-lane load width — all killed by the oracle tier, 7 of 8 also by the unit tests.
- **Tests to port:** `TestEvaluation` × 6 (covered by the oracle tier + unit tests above).

### U4 — Calculus (12 names) — DELIVERED 2026-09-29

**As built.** Where the build diverges from the plan below, this block wins; the plan is kept for the record.
- **Engine:** the driver is `Polynomial/Package/NDPolyCalc.cs`: NumPy's prologue in NumPy's statement order,
  which is also its error order. The kernel is `Backends/Kernels/Direct/DirectILKernelGenerator.PolyCalculus.cs`:
  - the six recurrences are data (`PolyCalcRoutines`: head steps, a j loop, tail steps; each statement is NumPy's
    source expression token for token);
  - one whole-array kernel is emitted per (basis, der/int, dtype, source dtype, 1-D scalarmath, fused scale);
  - each stage (load/convert/scale and recurrence) is its own DynamicMethod, so the JIT's inline budget covers one
    stage's lane helpers.
- **One buffer, in place, for any order:**
  - a derivative writes `der[q]` one row below `c[q]`, so the result is rows [m, n);
  - an integral writes `tmp[q]` one row above `c[q]`, with m spare rows above the loaded series.

  Columns (positions in `c.shape[1:]`) never meet, so the kernel takes them in ~512 KB blocks through every order
  of a derivative. It is vector over columns (U3's lane kinds), with a scalar tail. The load stage converts
  int/bool series to float64 and applies `c *= scl` in the same pass. It reads the caller's layout in place when a
  row's columns sit at one stride; otherwise NDIter copies first.
- **Three arithmetics, as planned (R10):**
  - `c *= scl` is always the ufunc (`simd_cmul`, operands (c, scl));
  - a 1-D series' recurrence is scalarmath (the naive product);
  - an N-D series' recurrence is ufuncs.

  The integral's `tmp[0] += k[i] - {p}val(lbnd, tmp)` runs through U3's `NDPolyEval.Val` plus U1's `PolyNumber`.
  One gap needed its own route: a 1-D float16/float32 series with a Python-complex lbnd, where NumPy's `{p}val` is
  complex64 SCALARMATH. U3's kernel computes complex in complex128, so `NDPolyCalc.PyVal1D` interprets U3's weak-x
  step trees with `PolyNumber` instead, which emulates complex64 scalars exactly.
- **Result objects are NumPy's:** moveaxis views of fresh C buffers; the m == 0 K-order copy; `c[:1]*0` from
  NpyIter's KEEPORDER vote, left unmoved by hermeder (a NumPy quirk); and the n == 1 zero branch's view of the
  moved copy.
- **Oracle:** `polycalc.jsonl` (`gen_oracle.py polycalc`, 21,646 cases, sections A–L, 0 excused), described in
  `test/NumSharp.Tests.Oracle/Fuzz/README.md`.
  - Planted-bug check: 10 mutants, 10 killed, 8 to 7,981 red cases each:
    - the 1-D scale product;
    - both loop bounds;
    - the block width;
    - the strided load;
    - vector negation;
    - the correction's sign;
    - later-order scaling;
    - the complex64 error name;
    - the complex64 `{p}val`.
  - Unit tests: `Polynomial/PolynomialCalculusTests.cs` (20).
- **Perf:** `benchmark/polynomial/polycalc_{numpy.py,bench.cs,report.py}`, committed summary `polycalc_results.md`;
  606 cells, all bit-exact.

  | Section | min NPY/NS | geomean NPY/NS |
  |---|---:|---:|
  | 1-D | 3.69 | 25.3 |
  | parameters | 2.47 | 7.7 |
  | N-D | 2.02 | 4.8 |
  | dtypes | 2.14 | 8.0 |
  | layouts | 2.91 | 7.3 |
  | orders | 3.10 | 6.6 |
  | small N-D | 3.87 | 12.4 |
  | **all** | **2.02** | **8.71** |
- **Open lever, measured:** float16 N-D runs ~30× slower than float32 (2.1–2.6× NumPy). U3's float16 lane kind rounds
  every op back onto the f16 grid (narrow + widen) and the store narrows again. An in-float round would buy only
  ~1.25×. A float32 working buffer, narrowed once at the end, is the real fix.

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
- **Engine (D12):** each body is one flat recurrence, so the whole routine is ONE IL kernel.
  - **Measured (§10.3 S):** `chebder` 50 runs **50–55×** (float64/float32/complex128) and **16×** (float16),
    byte-exact.
  - `*int`'s `tmp[0] += k[i] - {p}val(lbnd, tmp)` evaluates at a Python-scalar `lbnd`. The evaluation
    kernel's weak-x typing handles it: ops run in the coefficient dtype, and for complex they are
    scalar∘scalar, so they need the naive multiply (R10).
- **Tests to port:** `TestIntegral`, `TestDerivative` × 6.

### U5 — Vandermonde (18 names)

- **Scope:** `{p}vander`, `{p}vander2d`, `{p}vander3d`.
- **Shared backend:** a forward-recurrence driver (the same step constants as U3's table, used forward),
  followed by `_vander_nd` (broadcast product over `_nth_slice` axes) and `_vander_nd_flat`.
- **Output layout per D9:** a `moveaxis` view.
- **Parity:** portable, bit-exact.
- **Engine (D12, §10.1):**
  - `{p}vander` is a Tier-3A kernel over `(x, v[0])` of a C-contiguous `(deg+1, *x.shape)` buffer. NumPy's
    `x + 0.0` is fused into it, so `-0.0` becomes `+0.0` as in NumPy, and the result is returned as NumPy's
    `moveaxis` view.
  - `vander2d`/`vander3d` build their 1-D Vandermondes, then run ONE row-product kernel. It computes
    `vx_i * vy_j` (and `(vx_i*vy_j)*vz_k` in 3-D) in NumPy's operand order and writes straight into the final
    F-contiguous layout, tiled so the input rows stay in L1.
- **Perf:** NumPy `chebvander` deg 10 **4,203 µs @100K**, `polyvander2d` (3,3) **5,191 µs @100K**.
- **Measured (§10.3 G), all byte-exact and in NumPy's layout:**
  - `{p}vander` d10: **4.6–10×** @1K, **9.7–17×** @100K, **2.8–7.7×** @1M. The 1M cells are bound by page
    faults on the fresh 88 MB output, so they are noisy.
  - `vander2d` (3,3) @100K: the IL product kernel runs **9.2–9.7×**; the `np.multiply` broadcast composition
    only **1.4–1.8×** across runs. The inner dimensions are 4 wide, so per-chunk overhead dominates it.
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
| ~~M6~~ | ~~module constants are read-only~~ — RETIRED in U1: the constants are writeable shared singletons exactly as NumPy's (D6) | — |
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
- **R8 — Allocating from a view's `Shape` overruns the buffer (a live NumSharp bug, found by the §9 probe).**
  `np.empty(Shape, …)`, `np.zeros(Shape, …)`, `np.full(Shape, …)` and `new NDArray(dtype, Shape, fillZeros)`
  install the passed `Shape` as-is. `UnmanagedStorage.Allocate(Shape, DType, bool)` allocates `shape.size`
  elements but keeps the caller's strides. Handing it a strided view's `x.Shape` therefore returns a
  "fresh" array whose last element lies past its buffer: for `arange(20.)[::2]`, 10 elements at stride 2
  reach offset 18. The probe process died with a CLR fatal error this way.
  *Mitigation:* until it is fixed, every output buffer in the package is allocated from the dimensions
  (`new Shape(x.shape)`), never from `x.Shape`. The fix (rebuild the layout from `dimensions` unless the shape
  is already a clean C/F-contiguous, offset-0 layout) is a separate Core change with its own tests.
- **R9 — `[NDScoped]` retains loop temporaries.** A scope frees nothing until it closes. A recurrence over
  full-size arrays written as a plain port therefore keeps `deg × (2–5)` arrays alive: 1,733 MB vs 80 MB at
  10M points. *Mitigation:* D12. The hot recurrences are IL kernels that allocate only their result; the
  NativeAOT fallback composition reuses out= buffers.
- **R10 — NumPy dtype edges the emitter must reproduce** (all probed on 2.4.2; the IL probe byte-proves (a),
  (c) and int64 x; the int8 cases in (b) are NumPy-only probes so far):
  - **(a) Peeling.** NumPy's first Clenshaw steps run in the coefficient dtype (§10.1). Casting up front
    diverges on 100% of points (§10.3 C').
  - **(b) Int x arithmetic.** Integer x ops wrap in x's dtype: `chebval(int8 [100], [0, 0, 1])` is `-5601.0`,
    because `2*x` wraps to -56. A Python int that does not fit x's dtype raises: `lagval` on an int8 x with
    70 coefficients gives `OverflowError: Python integer 137 out of bounds for int8`, raised at pool fill.
  - **(c) Mixed dtypes.** float32 coefficients at a float64 x run steps 1–2 in float32. float16 coefficients
    at a float32 x run them in float16.
  - **(d) `hermval`'s `x2 = x*2` runs for every length.** So with `len(c) == 1`, an x above DBL_MAX/2 yields
    `0*inf = NaN`. `chebval` computes its x2 only when `len(c) >= 3`. The step tables encode both.
  - **(e) 0-d / Python-scalar x with complex coefficients.** NumPy's ops are then scalar∘scalar (scalarmath).
    Its complex multiply is the NAIVE product, not the array loop's, while its complex divide calls the
    CDOUBLE_divide loop. Real dtypes are unaffected.
  - **(f) The `len(c) == 1` product branch** (`legmul`/`hermmul`/`lagmul`) returns float64 for
    float16/float32 series (U2).
  - **(g) Integer and bool coefficients.** `chebval`-family `c` converts them with `astype(double)`, so
    `chebval(x, [True])` works. `as_series`-based routines raise `ValueError` for bool.
  - **(h) N-D x Vandermonde layout.** The layout of `vander2d`/`vander3d` for an N-D x is unverified; the
    probe checked 1-D x only.
  - *Mitigation:* each is a unit-test + oracle row in the unit that owns it.

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

This table was measured **unpinned**. On this hybrid P/E-core host that inflates some cells by up to 2.6×
(`chebval` deg 10 @100K read 4,205 µs here and 1,631 µs pinned). Where the two overlap, §9's pinned numbers
supersede these.

---

## 9. Engine choice — measured (2026-09-24)

> **Superseded as the engine DECISION by §10.** This section's measurements stand: they compare the candidate
> engines and they are why fusion and op order matter. But its verdict, plain C# span code and struct-generic
> kernels, violates the house rules for np.* functions: every loop in IL or NDIter, no struct kernels, no
> per-dtype C#. §10 re-derives the engine under those rules. Findings 5 and 6 below are answered there:
> - Tier 3A lands within 10–20% of the `Span` kernel.
> - NDIter is the driver, not an optional add-on. Buffering makes non-contiguous x 2–3× faster, and NDIter's
>   operand form carries N-D coefficients.

**Question.** Should NDExpr (`np.evaluate`) or NDIter + IL kernels power the package, or does a simple
1-to-1 port of NumPy's Python do the job?

**Method.** The probe pair `benchmark/polynomial/probes/` (`polynomial_engine_numpy.py` writes seeded inputs
and NumPy's outputs; `polynomial_engine_probe.cs` implements the SAME NumPy algorithm five ways):

| Variant | What it is |
|---|---|
| **P1** | A 1-to-1 port: each NumPy array expression becomes the matching `np.*`/`NDArray` op, element access goes through typed accessors, and temporaries are reclaimed by an `NDScope` (what `[NDScoped]` weaves in). |
| **P1lit** | P1 with NumPy's literal `c[i]` element spelling (0-d `NDArray` views). |
| **P1out** | The port rewritten to reuse buffers through out= (no per-step allocation). The best a composition gets. |
| **Expr** | One NDExpr tree through `np.evaluate`. Only Horner is expressible (finding 4). |
| **Span** | Typed C#: span loops for coefficient algebra; fused `Vector256` kernels (4 independent chains per iteration) for evaluation and Vandermonde; the same per-element op sequence as NumPy. |

Every variant was byte-compared with NumPy 2.4.2 before timing. Timing is best-of-7, Release,
`PublishAot=false`, `DOTNET_TC_CallCountingDelayMs=0`, with both processes pinned to logical CPUs 0–7 (the
P-cores of the i9-13900K) through `NS_PROBE_AFFINITY=0xFF`. Ratios are NPY/NS: **higher = NumSharp faster**.
`≠` = not byte-exact.

| Cell | NumPy µs | P1lit | P1 | P1out | Expr | Span |
|---|---|---|---|---|---|---|
| `polyadd 5+10` | 3.2 | — | 1.8 (1.77×) | — | — | 0.2 (13.79×) |
| `chebmul 10x10` | 7.9 | — | 12.9 (0.61×) ≠ | — | — | 0.6 (13.99×) ≠ |
| `legmul 10x10` | 174.9 | 674.8 (0.26×) | 88.2 (1.98×) | — | — | 1.7 (104.81×) |
| `legmul 50x50` | 2,045.4 | — | 587.9 (3.48×) | — | — | 31.6 (64.74×) |
| `legdiv 50/10` | 8,208.5 | — | 4,414.9 (1.86×) | — | — | 91.1 (90.06×) |
| `chebder 50` | 23.9 | — | 3.2 (7.51×) | — | — | 0.4 (55.96×) |
| `legint 50` | 34.9 | — | 3.3 (10.55×) | — | — | 0.6 (59.75×) |
| `polyval d3 n1000` | 6.4 | — | 10.9 (0.59×) | 8.2 (0.78×) | 0.8 (7.60×) | 0.6 (9.98×) |
| `chebval d3 n1000` | 7.0 | — | 5.4 (1.28×) | 6.4 (1.09×) | — | 0.4 (16.99×) |
| `legval d3 n1000` | 8.6 | — | 6.0 (1.43×) | 10.5 (0.82×) | — | 0.5 (18.56×) |
| `polyval d10 n1000` | 15.9 | — | 11.6 (1.37×) | 12.1 (1.32×) | 1.4 (11.15×) | 0.7 (22.76×) |
| `chebval d10 n1000` | 19.7 | — | 15.3 (1.29×) | 19.0 (1.04×) | — | 0.8 (24.10×) |
| `legval d10 n1000` | 31.4 | — | 26.0 (1.21×) | 40.5 (0.78×) | — | 1.4 (22.61×) |
| `polyval d30 n1000` | 43.0 | — | 38.0 (1.13×) | 49.3 (0.87×) | 5.2 (8.35×) | 2.1 (20.59×) |
| `chebval d30 n1000` | 55.9 | — | 53.9 (1.04×) | 62.2 (0.90×) | — | 2.6 (21.55×) |
| `legval d30 n1000` | 95.2 | — | 90.3 (1.05×) | 120.4 (0.79×) | — | 4.5 (21.36×) |
| `polyval d3 n100000` | 253.9 | — | 232.5 (1.09×) | 119.6 (2.12×) | 22.6 (11.23×) | 22.1 (11.51×) |
| `chebval d3 n100000` | 317.8 | — | 259.7 (1.22×) | 248.2 (1.28×) | — | 29.8 (10.65×) |
| `legval d3 n100000` | 336.5 | — | 299.2 (1.12×) | 229.1 (1.47×) | — | 21.5 (15.65×) |
| `polyval d10 n100000` | 761.2 | — | 748.0 (1.02×) | 260.3 (2.92×) | 59.4 (12.81×) | 48.9 (15.57×) |
| `chebval d10 n100000` | 1,631.3 | — | 1,840.8 (0.89×) | 592.4 (2.75×) | — | 57.8 (28.24×) |
| `legval d10 n100000` | 1,899.0 | — | 5,049.7 (0.38×) | 777.6 (2.44×) | — | 111.5 (17.02×) |
| `polyval d30 n100000` | 2,222.5 | — | 10,289.0 (0.22×) | 877.5 (2.53×) | 273.7 (8.12×) | 196.8 (11.29×) |
| `chebval d30 n100000` | 4,243.7 | — | 16,718.5 (0.25×) | 1,822.5 (2.33×) | — | 237.2 (17.89×) |
| `legval d30 n100000` | 5,206.1 | — | 24,332.1 (0.21×) | 2,270.5 (2.29×) | — | 386.1 (13.48×) |
| `polyval d10 n10000000` | 381,605.2 | — | 401,390.1 (0.95×) | 129,953.3 (2.94×) | 15,554.1 (24.53×) | 16,203.2 (23.55×) |
| `chebval d10 n10000000` | 479,030.9 | — | 508,591.5 (0.94×) | 238,318.3 (2.01×) | — | 15,161.8 (31.59×) |
| `legval d10 n10000000` | 778,630.0 | — | 766,544.2 (1.02×) | 292,696.8 (2.66×) | — | 18,975.3 (41.03×) |
| `chebval2d d5x5 n100000` | 15,659.0 | — | 10,698.3 (1.46×) | — | — | 367.5 (42.61×) |
| `chebvander d10 n100000` | 2,938.2 | — | 1,270.0 (2.31×) | 761.5 (3.86×) | — | 551.4 (5.33×) |
| `chebval d10 strided 50K` | 587.5 | — | — | 232.4 (2.53×) | — | 42.3 (13.88×) |
| `chebval d10 strided 5M` | 334,218.2 | — | — | 106,491.9 (3.14×) | — | 9,077.5 (36.82×) |

Memory (section M): the scoped 1-to-1 `polyval` at 10M points / degree 10 holds **1,733 MB** when its
scope closes; the out= port holds **80 MB** (x alone is 80 MB). A repeat run reproduced every row within
host noise.

**Findings.**
1. **Bit-exactness comes from keeping NumPy's operation order, not from the engine.** Every variant of every
   cell is byte-identical to NumPy. The one exception is `chebmul` 10×10: NumPy convolves the 21-long z-series
   through cblas `ddot`, which no managed engine reproduces (U2 parity rule).
2. **Coefficient algebra: typed spans win by one to two orders of magnitude; NDExpr/NDIter cannot help.**
   - Span is 14–105×. P1 is 0.61–10.6×: slower than NumPy for `chebmul`, where NumPy's per-op overhead is
     already low. P1lit is 0.26×: a 0-d `NDArray` per element read is 4× NumPy's cost.
   - On a 6–51-element series the whole span computation (0.2–90 µs) is at or below what an iterator
     construction plus kernel dispatch costs *per op*, so no iterator-driven engine can win there.
3. **Evaluation: fusion is the whole win.**
   - Span is 10–41× at every size.
   - P1out stays at ≤3.1×: every coefficient still costs 2–5 full memory passes, and fixed per-op overhead
     pushes it below NumPy at 1K.
   - P1 collapses as degree × size grows (0.21× at d30/100K) because the scope keeps every temporary alive
     (the memory row).
4. **NDExpr cannot express five of the six bases.**
   - `BinaryNode.EmitScalar` re-emits both children on every use. There is no interior-node sharing, only
     array-leaf dedup.
   - A Clenshaw step reads `c1` twice, so the tree roughly doubles per degree:

     | Degree | 10 | 20 | 30 | 50 |
     |---|---|---|---|---|
     | Nodes | 1,033 | 128,149 | 15.8 M | 2.4 × 10¹¹ |

     Horner stays linear (45 nodes at degree 10).
   - Where NDExpr works (power basis), it equals the hand kernel from 100K up (12.8–24.5× vs 15.6–23.6×). It
     is 1.3–2× slower at 1K (per-call tree/cache/iterator overhead) and 1.4× slower at degree 30 (one deep
     serial chain per loop body is latency-bound; the kernel interleaves four).
5. **IL emission buys nothing here.** A DynamicMethod compiles to the same machine code as the equivalent C#
   (measured 2026-09-24), and the IL/NDExpr routes throw `PlatformNotSupportedException` without dynamic code
   (NativeAOT), where plain C# kernels run.
6. **NDIter is optional.** Copying a strided `x` to contiguous and then running the fused kernel is 13.9×
   (50K) and 36.8× (5M) NumPy. The any-layout out= composition gets 2.5–3.1×. The copy is the only thing an
   NDIter-buffered driver could remove: 5.5 of 9.1 ms at 5M, i.e. ≤1.6× on strided input only. Worth doing
   after v1, not before.
7. **Vandermonde is write-bound.** Fusion adds only 1.4× over the out= composition (5.3× vs 3.9×). The
   composition would be an acceptable v1 there.
8. **Fused kernels need a homogeneous dtype** (the float32/float64 `inf` example in D12). Mixed and integer
   inputs go to the composition.

**Verdict at the time:** keep NumPy's structure 1-to-1, use typed span code for coefficient algebra, and use
fused plain-C# kernels for evaluation, `_valnd` and Vandermonde with an out= fallback. Two parts of that
survive: the structure/op-order rule and the rejection of NDExpr. The vehicle is replaced by §10's IL + NDIter
engine (D12). §10 also removes the homogeneous-dtype gate: mixed dtypes vectorize through lane-matched
conversions instead of falling back to the composition.

---

## 10. Engine — the house route: Tier-3A IL kernels driven by NDIter (measured, 2026-09-24)

**Question.** Under the house rules for np.* functions, what is the fastest engine? The rules: every loop in
IL (`DirectILKernelGenerator`) or NDIter, no struct kernels, no per-dtype C#, NumPy 2.4.2 parity for every
dtype, and at least 1.5× NumPy on every variation.

**Answer.** One typed IL emitter turns six per-basis step tables into Tier-3A raw inner loops. NDIter drives
them, and IL primitives over an arena handle the small-series algebra. Every measured cell is byte-exact
against NumPy, and the chosen route is ≥ 2.8× NumPy on every timed cell (§10.3). Tier 3A lands within 10–20%
of hand-written C# `Vector256` code.

### 10.1 Design

**Step tables are data.** Each basis's NumPy 2.4.2 step expressions are small expression trees, written in
source order and never re-associated. For example:
- `chebval`: `x2 = 2*x; c0 = c[k] - c1; c1 = tmp + c1*x2; return c0 + c1*x`
- `lagval`: `c0 = c[k] - (c1*(nd-1))/nd; c1 = tmp + (c1*((2*nd-1) - x))/nd; return c0 + c1*(1 - x)`
- `hermvander`: `x2 = x*2; v[1] = x2; v[i] = v[i-1]*x2 - v[i-2]*(2*(i-1))`

The emitter is basis-agnostic; a basis differs only by its table.

**NumPy's per-op typing, emitted.** Every node gets NumPy's dtype:
- strong operands promote through `np.promote_types`;
- Python literals follow NEP 50 weak rules: a Python int adopts an integer partner's dtype and raises
  NumPy's `OverflowError` when it does not fit, a Python float turns an integer partner into float64, and
  `true_divide` of an integer loop is float64.

Each op is lowered through the house `EmitScalarOperation`/`EmitConvertTo`, or through the vector operators
at a lane-matched width. NumPy's dtype semantics therefore come from the same emitters the ufuncs use: the
complex array-loop multiply, float16's per-op round trip, and integer wrap-around.

**Peeling to a dtype fixpoint.** NumPy starts Clenshaw's `(c0, c1)` in the coefficient dtype. `c1` absorbs
x's dtype in the first step, and `c0` absorbs `c1`'s in the second. The emitter simulates these carried dtypes
and emits the first P steps straight-line, then loops at the fixpoint. P is 0 for a homogeneous call and 2
for float32 coefficients at a float64 x. The kernel key includes the coefficient-count class `min(nc, P+3)`,
which also covers NumPy's `len(c) == 1` and `len(c) == 2` branches. Casting the coefficients up front instead
diverges on 100% of points (§10.3 C').

**Weak constants from a pool.** Python literals and Python-computed values (`2`, `(nd-1)/nd`, `2*(nd-1)`, …)
live in a constant pool that the kernel reads through auxdata:
- one region per (constant, dtype), indexed by `nd`;
- filled once per (kernel, coefficient count) with the house `astype`, so a float16 kernel reads `(nd-1)/nd`
  rounded to float16 exactly as NumPy rounds the Python float;
- cached, so the per-call cost is one dictionary lookup.

**Shared scalars, lane-matched vectors.**
- A value that does not depend on the point is a SHARED scalar in its own dtype, computed once per block for
  every chain: a broadcast coefficient, a constant, or math on those, such as the peeled coefficient-only
  steps. This is also NumPy's semantics for coefficient-only math. It is converted and broadcast only where it
  meets a per-point value.
- Per-point values are vectors with the loop dtype's lane count W:
  - a float32 or int32 x at a float64 loop is a `Vector128` widened by `vcvtps2pd`/`vcvtdq2pd`;
  - an int64 x converts through `Vector256.ConvertToDouble`;
  - a float16 x already rides float32 lanes.
- MIXED dtypes therefore vectorize too. The scalar-chain fallback measured 0.69× on `legval(f32 x, f16 c)`;
  lane-matched it is 5.7×.
- For a 1-D `c` of another dtype, the steady-state steps read a copy pre-converted to the loop dtype. This is
  exact: NumPy's promotion in those steps is a widening conversion. It saves a scalar conversion per
  coefficient per block, and a float16 conversion is a software call.
- Scalar chains remain only for complex, decimal and the pairs with no lane conversion (narrow ints, float16 x
  at a float64 loop). They are still 4-wide interleaved.

**Four interleaved chains.** Each point's recurrence is one serial dependency chain, so a block runs U = 4
independent chains. That is 1.2–1.9× faster than one chain, and it is why the kernel is a Tier-3A raw loop
rather than the Tier-3B shell: the shell unrolls one body, i.e. one chain (1.15–1.46× slower, §10.3 B).

**NDIter drives the kernel, in two operand forms.**
- **1-D `c`** (the common case, and class `__call__`): operands `(x, y)`, with the coefficient base in
  auxdata.
  - A non-contiguous x runs BUFFERED + CONTIG. The kernel then sees contiguous chunks and stays on vector
    chains, 2.7–3.0× faster than strided scalar chains.
  - A contiguous x runs unbuffered, because buffering it costs ~20%.
- **N-D `c`** (`tensor=True` multi-series, `tensor=False`, `_valnd`, `_gridnd`): NumPy's tensor reshape, then
  operands `(x, c[0], y)` with NDIter broadcasting. Per chunk, the kernel reads `c[0] + k·stride`:
  - a stride-0 chunk is one broadcast series;
  - a contiguous chunk is one series per point, read with lane loads.

  This form is NEVER buffered. The kernel reads past the `c[0]` operand, and NumSharp's NDIter buffers any
  operand whose walk is not linear (a multi-series `c[0]` beside a non-contiguous x). Its extra iterator setup
  costs ~1 µs at 1K, which is why a 1-D `c` keeps the auxdata form.

**Vandermonde.** A Tier-3A kernel over `(x, v[0])` of a C-contiguous `(deg+1, *x.shape)` buffer:
- NumPy's `x + 0.0` is fused, so `-0.0` becomes `+0.0` as in NumPy;
- each point is read once and its deg+1 rows are written once;
- the result is `np.moveaxis(v, 0, -1)`, NumPy's layout;
- a non-contiguous x is copied first (NumPy's `x + 0.0` copies it anyway), which beats strided chains by
  1.1–1.3×.

`vander2d`/`vander3d` build their 1-D Vandermondes, then run ONE row-product kernel. It computes NumPy's
single multiply `vx_i * vy_j` in its operand order (3-D: `(vx_i*vy_j)*vz_k`) and writes straight into the
final F-contiguous layout. It works in tiles of 512 points, so the input rows stay in L1 across all output
rows.

**Small-series algebra.** Routines over 6–51 coefficients are dominated by NumPy's per-array-op overhead, and
an NDIter per op would cost about as much. The engine instead has four parts:
- dtype-generic IL **primitives**: trim length, `k*v`, `(v*k)/d`, in-place add/sub/negate, `legmulx`, and
  scalar÷scalar;
- **whole-routine** IL kernels where the routine is one flat recurrence (`chebder`);
- dtype-agnostic **orchestration** mirroring NumPy's Python line for line over byte pointers. Its only loops
  are NumPy's own Python-level loops over SERIES (legmul's i-loop), never over elements;
- a bump **arena** (one reused block per thread) for every intermediate series, so an intermediate costs a
  pointer bump instead of an NDArray.

**Not used:**
- NDExpr: a Clenshaw tree grows exponentially (§9 finding 4).
- Struct kernels / per-dtype C#: house rule. §9's `Span` variant serves only as the speed reference.
- The Tier-3B shell: one chain per body.
- A plain `[NDScoped]` port: it holds every temporary (§9).
- Without dynamic code (NativeAOT) the package falls back to the out= composition over NumSharp's ufuncs.

### 10.2 Method

The probe pair lives in `benchmark/polynomial/probes/`:
- `polynomial_il_eval_probe.cs`: the emitter, the evaluation, Vandermonde and product kernels, and the NDIter
  drivers.
- `polynomial_il_series_probe.cs`: primitives, orchestration and arena.
- `polynomial_il_numpy.py`: the NumPy twin. It writes seeded inputs and NumPy's outputs to `data/il/` and
  prints NumPy's timings under the same cell labels.

Protocol:
- Every row is byte-compared (NaN payloads and signed zeros included) before it is timed.
- Timing is best-of-7 after a 300 ms warm-up, in Release, with `PublishAot=false` and
  `DOTNET_TC_CallCountingDelayMs=0`. Both sides are pinned to logical CPUs 0–7 (`NS_PROBE_AFFINITY=0xFF`).
- The host was shared with other sessions: a `python` job held one core, and Zoom and agents took ~30% of the
  CPU. Every figure is therefore the per-cell **minimum over two full runs on each side**.
- Absolute NumPy times vary 1.5–2× between states of this host (§8's note), so read ratios, not microseconds.
- Ratios are NPY/NS: higher = NumSharp faster.

### 10.3 Measurements

#### A. Evaluation, float64, 1-D coefficients: NPY/NS for Tier 3A with 4 interleaved chains

| Basis | d3 @1K | d10 @1K | d30 @1K | d3 @100K | d10 @100K | d30 @100K | d10 @10M |
|---|---|---|---|---|---|---|---|
| `polyval` | 11.1 | 16.8 | 17.3 | 16.2 | 15.7 | 12.6 | 24.5 |
| `chebval` | 11.6 | 19.4 | 20.1 | 30.3 | 23.0 | 18.3 | 31.5 |
| `legval` | 13.5 | 19.6 | 20.5 | 12.0 | 16.0 | 12.1 | 36.3 |
| `lagval` | 6.5 | 6.9 | 7.4 | 3.1 | 3.9 | 3.4 | 13.8 |
| `hermval` | 13.5 | 21.2 | 24.1 | 14.0 | 27.8 | 27.7 | 32.0 |
| `hermeval` | 12.5 | 21.6 | 23.8 | 10.3 | 16.7 | 14.8 | 34.9 |

Absolute time for the d10 @100K column (µs, NumPy → NumSharp):
`polyval` 860.6 → 54.7, `chebval` 1,580 → 68.8, `legval` 1,902 → 119.2, `lagval` 2,751 → 706.3, `hermval` 2,708 → 97.3, `hermeval` 1,447 → 86.4.

#### A'. Interleaving and the operand form (µs; ratio NPY/NS)

| Cell | NumPy | 4 chains | 1 chain | 3-operand form (c via NDIter) |
|---|---|---|---|---|
| `polyval` d10 @1K | 16.5 | 1.0 (16.8×) | 1.2 (13.7×) | 2.0 (8.3×) |
| `polyval` d10 @100K | 860.6 | 54.7 (15.7×) | 77.1 (11.2×) | 57.8 (14.9×) |
| `polyval` d30 @100K | 2,569 | 204.2 (12.6×) | 333.2 (7.7×) | - |
| `chebval` d10 @1K | 21.0 | 1.1 (19.4×) | 1.4 (15.5×) | 2.2 (9.7×) |
| `chebval` d10 @100K | 1,580 | 68.8 (23.0×) | 90.5 (17.5×) | 78.4 (20.1×) |
| `chebval` d30 @100K | 4,346 | 237.6 (18.3×) | 377.7 (11.5×) | - |
| `legval` d10 @1K | 31.6 | 1.6 (19.6×) | 2.1 (15.2×) | 2.8 (11.1×) |
| `legval` d10 @100K | 1,902 | 119.2 (16.0×) | 169.1 (11.2×) | 122.1 (15.6×) |
| `legval` d30 @100K | 5,288 | 438.8 (12.1×) | 828.2 (6.4×) | - |
| `lagval` d10 @1K | 52.9 | 7.6 (6.9×) | 7.4 (7.2×) | 8.2 (6.5×) |
| `lagval` d10 @100K | 2,751 | 706.3 (3.9×) | 694.6 (4.0×) | 742.4 (3.7×) |
| `lagval` d30 @100K | 7,513 | 2,188 (3.4×) | 2,326 (3.2×) | - |

#### B. Tier 3A vs the Tier-3B shell vs hand-written C# (`chebval` d10, µs; ratio NPY/NS)

| Size | NumPy | Tier 3A (4 chains) | Tier 3B shell | plain C# `Vector256`, no NDIter |
|---|---|---|---|---|
| 1K | 21.0 | 1.1 (19.4×) | 1.3 (15.6×) | 0.8 (25.8×) |
| 100K | 1,580 | 68.8 (23.0×) | 100.6 (15.7×) | 56.9 (27.8×) |
| 10M | 522,375 | 16,561 (31.5×) | 19,015 (27.5×) | 15,151 (34.5×) |

#### C. Dtypes and mixed dtypes (d10 / 11 coefficients @100K, µs; ratio NPY/NS; every row byte-exact)

| Cell | NumPy | NumSharp | NPY/NS | Path |
|---|---|---|---|---|
| `polyval f32 d10 n100000` | 238.1 | 29.1 | 8.2 | vector chains |
| `polyval c128 d10 n100000` | 7,057 | 736.4 | 9.6 | scalar chains (complex) |
| `chebval f32 d10 n100000` | 505.8 | 35.0 | 14.4 | vector chains |
| `chebval c128 d10 n100000` | 9,812 | 835.5 | 11.7 | scalar chains (complex) |
| `legval f32 d10 n100000` | 666.3 | 58.6 | 11.4 | vector chains |
| `legval c128 d10 n100000` | 15,332 | 1,446 | 10.6 | scalar chains (complex) |
| `lagval f32 d10 n100000` | 1,089 | 217.1 | 5.0 | vector chains |
| `lagval c128 d10 n100000` | 22,214 | 3,651 | 6.1 | scalar chains (complex) |
| `hermval f32 d10 n100000` | 648.1 | 44.5 | 14.6 | vector chains |
| `hermval c128 d10 n100000` | 12,693 | 1,118 | 11.4 | scalar chains (complex) |
| `hermeval f32 d10 n100000` | 586.1 | 44.6 | 13.2 | vector chains |
| `hermeval c128 d10 n100000` | 11,950 | 1,109 | 10.8 | scalar chains (complex) |
| `polyval f16 d10 n100000` | 6,636 | 1,356 | 4.9 | float16 vector chains (widen-op-narrow) |
| `chebval f16 d10 n100000` | 8,724 | 1,835 | 4.8 | float16 vector chains (widen-op-narrow) |
| `legval f16 d10 n100000` | 13,507 | 2,976 | 4.5 | float16 vector chains (widen-op-narrow) |
| `hermval f16 d10 n100000` | 10,998 | 2,782 | 4.0 | float16 vector chains (widen-op-narrow) |
| `chebval x32/c64 d10 n100000` | 1,981 | 96.5 | 20.5 | vector, float32 lanes → float64 |
| `legval x32/c64 d10 n100000` | 2,258 | 128.4 | 17.6 | vector, float32 lanes → float64 |
| `polyval int64x d10 n100000` | 915.4 | 127.4 | 7.2 | vector, int64 lanes → float64 |
| `chebval int64x d10 n100000` | 1,970 | 154.7 | 12.7 | vector, int64 lanes → float64 |
| `lagval int64x d10 n100000` | 3,364 | 687.3 | 4.9 | vector, int64 lanes → float64 |
| `chebval c32/x64 nc11 n100000` | 1,716 | 77.1 | 22.3 | vector + 2 peeled steps in float32 |
| `legval c32/x64 nc11 n100000` | 2,036 | 125.3 | 16.3 | vector + 2 peeled steps in float32 |
| `lagval c32/x64 nc11 n100000` | 2,829 | 658.6 | 4.3 | vector + 2 peeled steps in float32 |
| `hermval c32/x64 nc11 n100000` | 3,034 | 98.1 | 30.9 | vector + 2 peeled steps in float32 |
| `hermeval c32/x64 nc11 n100000` | 1,589 | 90.0 | 17.7 | vector + 2 peeled steps in float32 |
| `legval c16/x32 nc11 n100000` | 676.2 | 119.2 | 5.7 | vector + 2 peeled steps in float16 |

#### C'. Why the peeling is load-bearing (float32 coefficients at a float64 x, 100K points)

| Basis | nc=3 | nc=4 | nc=5 | nc=11 |
|---|---|---|---|---|
| `chebval` | exact / 100% differ | exact / exact | exact / 100% differ | exact / 100% differ |
| `legval` | exact / 100% differ | exact / 100% differ | exact / 100% differ | exact / 100% differ |
| `lagval` | exact / 100% differ | exact / 100% differ | exact / 100% differ | exact / 100% differ |
| `hermval` | exact / 100% differ | exact / 100% differ | exact / 100% differ | exact / 100% differ |
| `hermeval` | exact / 100% differ | exact / exact | exact / 100% differ | exact / 100% differ |

Each cell: peeled kernel / the same kernel on coefficients pre-cast to float64.

#### D. Non-contiguous x (d10, µs; ratio NPY/NS)

| Cell | NumPy | strided scalar chains | BUFFERED+CONTIG → vector chains |
|---|---|---|---|
| `chebval` strided (50K) | 538.3 | 158.8 (3.4×) | 56.3 (9.6×) |
| `chebval` reversed (100K) | 1,644 | 305.5 (5.4×) | 107.6 (15.3×) |
| `chebval` 2dT (100K) | 1,634 | 321.2 (5.1×) | 119.1 (13.7×) |
| `legval` strided (50K) | 748.5 | 243.0 (3.1×) | 83.2 (9.0×) |
| `legval` reversed (100K) | 1,949 | 470.6 (4.1×) | 156.3 (12.5×) |
| `legval` 2dT (100K) | 1,898 | 502.6 (3.8×) | 168.2 (11.3×) |

- BUFFERED on an already-contiguous x, `chebval` d10 @1K: 1.1 → 1.3 µs.
- BUFFERED on an already-contiguous x, `chebval` d10 @100K: 68.8 → 84.2 µs.

#### F. N-D coefficients (µs; ratio NPY/NS)

| Cell | NumPy | NumSharp | NPY/NS | Route |
|---|---|---|---|---|
| `polyval multi8 d10 n100000` | 27,000 | 468.3 | 57.7 | (x, c[0], y) operands, broadcast coefficients |
| `polyval tensorF d10 n100000` | 931.9 | 129.6 | 7.2 | (x, c[0], y), one series per point |
| `polyval2d d5x5 n100000` | 10,905 | 555.0 | 19.6 | two kernel passes, NumPy's intermediate |
| `polygrid2d d5x5 300x300` | 461.4 | 34.3 | 13.5 | two kernel passes |
| `chebval multi8 d10 n100000` | 37,932 | 551.7 | 68.8 | (x, c[0], y) operands, broadcast coefficients |
| `chebval tensorF d10 n100000` | 3,510 | 132.7 | 26.4 | (x, c[0], y), one series per point |
| `chebval2d d5x5 n100000` | 14,836 | 588.6 | 25.2 | two kernel passes, NumPy's intermediate |
| `chebgrid2d d5x5 300x300` | 1,010 | 39.6 | 25.5 | two kernel passes |
| `legval multi8 d10 n100000` | 60,882 | 937.8 | 64.9 | (x, c[0], y) operands, broadcast coefficients |
| `legval tensorF d10 n100000` | 2,253 | 133.1 | 16.9 | (x, c[0], y), one series per point |
| `legval2d d5x5 n100000` | 20,564 | 610.8 | 33.7 | two kernel passes, NumPy's intermediate |
| `leggrid2d d5x5 300x300` | 943.9 | 53.2 | 17.7 | two kernel passes |
| `chebval tensorF strided d10 n50000` | 1,884 | 203.1 | 9.3 | strided lanes → scalar chains |

#### G. Vandermonde (d10; NPY/NS; every result byte-exact and in NumPy's `moveaxis` layout)

| Basis | @1K | @100K | @1M | @100K, 1 chain |
|---|---|---|---|---|
| `polyvander` | 4.6 | 11.5 | 2.8 | 11.2 |
| `chebvander` | 6.5 | 14.5 | 4.2 | 13.7 |
| `legvander` | 9.7 | 9.7 | 7.7 | 9.0 |
| `lagvander` | 10.1 | 11.3 | 7.4 | 11.0 |
| `hermvander` | 8.9 | 17.1 | 5.9 | 16.5 |
| `hermevander` | 9.0 | 16.1 | 6.6 | 15.8 |

| Cell | NumPy | NumSharp | NPY/NS |
|---|---|---|---|
| `chebvander f32 d10 n100000` (IL3A-x4) | 1,316 | 92.2 | 14.3 |
| `chebvander f16 d10 n100000` (IL3A-x4) | 7,169 | 1,789 | 4.0 |
| `chebvander c128 d10 n100000` (IL3A-x4) | 10,830 | 1,235 | 8.8 |
| `chebvander int64x d10 n100000` (IL3A-x4) | 2,938 | 188.5 | 15.6 |
| `chebvander strided d10 n50000` (IL3A-x4 strided) | 1,310 | 124.5 | 10.5 |
| `chebvander strided d10 n50000` (IL3A-x4 copy-first) | 1,310 | 111.2 | 11.8 |
| `chebvander 2dT d10 n100000` (IL3A-x4 strided) | 3,218 | 303.7 | 10.6 |
| `chebvander 2dT d10 n100000` (IL3A-x4 copy-first) | 3,218 | 239.4 | 13.4 |
| `polyvander2d d3x3 n100000` (_vander_nd np.multiply) | 5,405 | 2,990 | 1.8 |
| `polyvander2d d3x3 n100000` (_vander_nd IL product) | 5,405 | 587.0 | 9.2 |
| `chebvander2d d3x3 n100000` (_vander_nd np.multiply) | 5,775 | 3,357 | 1.7 |
| `chebvander2d d3x3 n100000` (_vander_nd IL product) | 5,775 | 597.8 | 9.7 |

#### S. Small-series algebra: IL primitives + orchestration + arena (µs; every result byte-exact)

| Cell | NumPy | NumSharp | NPY/NS |
|---|---|---|---|
| `polyadd 5+10 f64` | 3.2 | 0.8 | 4.2 |
| `legmul 10x10 f64` | 175.6 | 1.5 | 113.5 |
| `legmul 10x10 f32` | 189.0 | 1.5 | 127.4 |
| `legmul 10x10 f16` | 202.7 | 11.1 | 18.3 |
| `legmul 10x10 c128` | 189.6 | 2.3 | 81.6 |
| `legmul 50x50 f64` | 2,144 | 22.2 | 96.7 |
| `legdiv 50/10 f64` | 8,476 | 50.6 | 167.5 |
| `legdiv 50/10 c128` | 8,962 | 109.6 | 81.8 |
| `chebder 50 f64` | 24.2 | 0.5 | 51.9 |
| `chebder 50 f32` | 26.1 | 0.5 | 55.2 |
| `chebder 50 f16` | 28.8 | 1.8 | 15.8 |
| `chebder 50 c128` | 26.1 | 0.5 | 50.0 |

Lowest NPY/NS over the chosen route of every timed cell: `polyvander d10 n1000000` 2.82×; `lagval d3 n100000` 3.13×; `lagval d30 n100000` 3.43×; `lagval d10 n100000` 3.89×; `hermval f16 d10 n100000` 3.95×; `chebvander f16 d10 n100000` 4.01×.

Non-exact rows (excluding the deliberate pre-cast control): 0

### 10.4 Findings

1. **The IL route costs little against hand C#.** A hand-written C# `Vector256` Clenshaw (no NDIter) is
   1.2× faster at 100K (56.9 vs 68.8 µs) and 1.09× at 10M. In a side-by-side rerun the gap at 100K was 1.10×
   (59.8 vs 65.6 µs). At 1K the difference is the ~0.2–0.3 µs NDIter setup that the direct call skips; that
   setup is what buys every layout and dtype.
2. **Interleaving is the one kernel-shape decision that matters.**
   - Four independent chains are 1.2–1.9× over one, because the recurrence is latency-bound.
   - The Tier-3B shell cannot interleave, and it is the slower Tier.
   - Division-bound `lagval` is flat across chain counts (the divider is the bottleneck), which makes it the
     lowest basis at 3.1–3.9× @100K.
3. **Peeling is correctness, not tuning.** Pre-casting float32 coefficients diverged on 100% of points for
   every basis and count, except where the peeled subtraction happened to be exact. The peeled kernel matched
   everywhere.
4. **Lane-matched vectors turn every real dtype pair into vector chains.** The one sub-1.5× variation of the
   scalar fallback, `legval(f32 x, f16 c)` at 0.69×, is 5.7× lane-matched. Mixed float32/float64 moved from
   2.3–8.5× to 4.3–31×, and int64 x from 2.4–4.4× to 4.9–12.7×.
5. **Composition over NumSharp ufuncs is not good enough for `_vander_nd`.** The broadcast `np.multiply`
   over 4-wide inner dims ran 1.4–1.8×; the row-product kernel runs 9.2–9.7×. `_valnd`/`_gridnd`, by
   contrast, compose fine as two kernel passes: each pass is itself a kernel (13–34×).
6. **Small series: the arena is the win.** IL primitives with arena intermediates run 82–168× NumPy for
   `legmul`/`legdiv` in float64 and complex128. float16 is 16–18× because each scalar Half op is software.
   `polyadd` sits at 4.2×: one output NDArray is the whole cost there.

**Open levers for the implementation** (none is needed to clear 1.5×):
- **Hoist the shared peeled math out of the block loop.** Coefficient-only work is loop-invariant across
  blocks. It dominates `legval(f32 x, f16 c)` (5.7×, vs 11× for homogeneous float32).
- **AVX-512 lane conversions.** A 512-bit host needs the Avx512F twins of the three conversions; until then it
  runs scalar chains for mixed pairs.
- **Narrow-int x lane kinds** (int8/int16/uint*). They run scalar chains today; each needs a widen-then-wrap
  lane form.
- **R10 (e):** 0-d complex x must use the naive scalar multiply.
- **R10 (h):** verify the N-D x layout of `vander2d`/`vander3d`, and write the 3-D product kernel.
- **Zero-size x:** needs NDIter's zero-size flag or an early return; the probe does not cover it.
- **Decimal** rides scalar chains through the house decimal ops and is gated by the independent decimal
  oracle.

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
