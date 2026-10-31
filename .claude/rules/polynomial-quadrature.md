---
paths:
  - "src/NumSharp.Core/Polynomial/Package/NDPoly{Weight,ChebPoints,Gauss,Interpolate,Fit,FitResult}.cs"
  - "src/NumSharp.Core/Backends/Kernels/Direct/DirectILKernelGenerator.Poly{ChebPoints,NormedHermite}.cs"
  - "test/NumSharp.Tests/Polynomial/Polynomial{Weight,ChebPoints,Gauss,Interpolate,Fit}Tests.cs"
  - "test/NumSharp.Tests.Oracle/Fuzz/corpus/poly{weight,pts,gauss,interp,fit}*.jsonl"
---

# The polynomial package (3/4): weights, nodes, quadrature, interpolation and fits

Every function here is bit-exact with NumPy 2.4.2, zero excuses. The LAPACK-bound ones (the Gauss quadratures, the
interpolation and the fits) are exact with the OpenBLAS NumPy loads, at one thread, and their value tiers are host-pinned
like `polyroots_parity` (the BLAS identity in the tier's `.host.jsonl`); without a LAPACK backend a valid call raises
`MissingBackendException` after NumPy-exact validation.

## Weight functions: `chebweight`, `legweight`, `lagweight`, `hermweight`, `hermeweight`

- Engine (`Polynomial/Package/NDPolyWeight.cs`): NumPy's one statement per weight, run on whatever x is:
  `1.0/(np.sqrt(1.0 + x)*np.sqrt(1.0 - x))`, `x*0.0 + 1.0`, `np.exp(-x)`, `np.exp(-x**2)`, `np.exp(-.5*x**2)`.
  - A Python scalar is CPython arithmetic until the first ufunc (`legweight(0.5)` is a Python float, `chebweight(2)`
    NaN, `hermweight(2**70)` NumPy's object-loop TypeError); the statements are replayed over `PolyNumber`. A Python
    list / tuple / None / str meets Python's own operator, which raises before NumPy sees it.
  - `x**2` of an ndarray is `np.square`, not `np.power` (`fast_scalar_power`): a bool array squares to int8.
  - `x**2` of a scalar is the C runtime's `pow` (`PyScalar.PowTwo`, `PolyNumber.PowTwo`): CPython's `float_pow` and
    NumPy's scalarmath call `pow` / `powf`, and ucrtbase's `pow` is not the correctly rounded square (`powf` flushes
    subnormal squares: `np.float32(2.9465066e-23)**2` is 0). A complex Python value is `c_powu`'s two naive products; an
    infinite square raises CPython's OverflowError texts.
  - Scalar-sized ufunc calls run the house kernel on one stack element (`PolyNumber.FloatUfunc`, the loop dtype from
    `PolyNumber.FloatLoop`, which a unit test holds equal to `ResolveUnaryFloatReturnType`): no 0-d arrays but the result.
- Routes for an ndarray x, each byte-identical to the others and to NumPy: one fused `np.evaluate` tree, node for node
  (`Square` for `x**2`; the literals stay weak); float16, decimal and the string dtypes run the statements on the house
  ufuncs (decimal has no NumPy dtype, so the house routines define it; a string array gets NumPy's no-loop text); a
  broadcast x is computed once on its un-broadcast core and copied into NumPy's layout; an integer or bool x computes its
  integer part (`-x`, `x**2`, which wrap) in x's dtype, then ONE cast into the float loop (NumPy's buffered input cast);
  an N-D x that is not C-contiguous is computed over its C-ordered permutation and relabelled with NumPy's layout
  (`NDArray.AdoptLayout`).
- Not bit-exact, so not in the byte tiers: complex `chebweight` inherits `np.sqrt`'s complex envelope (unit-tested within
  4 ULP). A `numpy.matrix` x is computed as its plain array, elementwise (`[Misaligned]`: NumPy's operators would make
  `x**2` a matrix power).
- Gates: `polyweight.jsonl` (portable) and `polyweight_host.jsonl` (`RunHostLibmCorpus`),
  `Polynomial/PolynomialWeightTests.cs`.

## Chebyshev points and Gauss-Chebyshev quadrature: `chebpts1`, `chebpts2`, `chebgauss`

- Engine (`Polynomial/Package/NDPolyChebPoints.cs`), NumPy's statements:
  - npts goes through Python's `int()` and `_npts != npts` (`PolyPowerArgument`, shared with `{p}pow`): a fraction, a
    str (`int('3')` succeeds, then `'3' != 3`) and a 0-d array holding 3.5 are `npts must be integer`. deg goes through
    `operator.index` (`PolyIndexArgument`): a float, even 3.0, is `deg must be an integer, received 3.0`.
  - The nodes are NumPy's statement chains on the exact integer ramp: chebpts1 `sin(c·(1 − n + 2k))` with
    `c = (0.5·π)/float(n)`, chebpts2 `cos(k·(π/(n − 1)) + (−π))` with linspace's endpoint stored as 0, chebgauss
    `cos((π·(2k + 1))/(2.0·deg))`, and `w = π/deg` in every slot. chebpts1's `0.5*np.pi / n` converts n to a float
    BEFORE np.arange runs, so a count past the float range raises CPython's `OverflowError: int too large to convert to
    float` there.
- One emitted fill per formula (`DirectILKernelGenerator.PolyChebPoints.cs`, `GetPolyChebNodesKernel`): it writes only
  the result, the ramp kept as a running double (exact below 2^53, every count that can be allocated); `Math.Sin` /
  `Math.Cos` are the C runtime functions NumPy's float64 loops call on win-amd64.
- chebpts1 computes half its sines: its ramp is antisymmetric and the C runtime's sin is odd bit for bit, so each sine's
  negation fills the mirrored slot. `Nodes_AreNumPysStatementChain` compares every element with `Math.Sin` of its own
  argument, so a host whose sin is not odd fails CI instead of differing silently.
- np.arange's length rule is ported (`NDPolyChebPoints.ArangeLength`): `ceil((stop − start)/step)` as a correctly
  rounded Python int division, checked against `(double)NPY_MAX_INTP` (2^63), every OverflowError re-raised as
  `ValueError: Maximum allowed size exceeded`; a length that rounds to exactly 2^63 casts to INT64_MIN on x86 and
  np.arange returns an EMPTY array. So from 2^60 points: `array is too big; …`; counts in [2^63 − 512, 2^63 + 1024]:
  chebpts1 returns an empty float64 array (an empty OBJECT array from 2^63 + 2: `NotSupportedException`), chebpts2 raises
  linspace's `IndexError: index -1 is out of bounds for axis 0 with size 0`, chebgauss np.ones' `Maximum allowed
  dimension exceeded` (or `array is too big` below 2^63); past the band `Maximum allowed size exceeded`; a count below
  the byte limit the machine cannot hold is NumPy's MemoryError, `OutOfMemoryException`.
- Gates: `polypts.jsonl` (errors, empty results, flags, Char) and `polypts_host.jsonl` (values, `RunHostLibmCorpus`),
  `Polynomial/PolynomialChebPointsTests.cs`.

## Gauss quadrature: `leggauss`, `laggauss`, `hermgauss`, `hermegauss`

- leggauss / laggauss (`Polynomial/Package/NDPolyGauss.cs`): NumPy's statements composed from bit-exact engines: the
  companion of `[0]*deg + [1]`, `np.linalg.eigvalsh` (LAPACK `syevd`), one Newton step through `{p}val` and `{p}der`,
  the weights scaled by their largest magnitude (`np.max`, NumPy's schedule), leggauss symmetrising both and normalising
  the weights to 2, laggauss normalising them to 1 (`np.sum`, NumPy's pairwise add.reduce).
- `c = np.array([0] * deg + [1])` uses `deg` itself, so NumPy's sequence rules apply: a Python int, a bool (`[0] * True`
  is one zero) or an np.uint16 (a C# char) repeats the list; a 0-d integer ARRAY does not (`[0] * arr` is
  `np.multiply`), so c is `[1]` and the companion raises `Series must have maximum degree of at least 1.`; a count past
  2^63 − 1 is CPython's `OverflowError: cannot fit 'int' into an index-sized integer`, and one of 2^60 − 1 or more its
  list size limit (an empty MemoryError raised before any allocation: `OutOfMemoryException("")`).
- hermgauss / hermegauss (`HermiteGauss`): `_as_int`, `c = np.array([0] * deg + [1], dtype=np.float64)` (the dtype is
  unobservable: the companion's as_series turns an int64 series into float64 anyway), the companion, eigvalsh, the
  normalised recurrence three times (`dy`, then `df` scaled by `sqrt(2·deg)` / `sqrt(deg)`, then `fm` at the
  Newton-corrected nodes), the in-place ufunc statements, `np.max(np.abs(fm))`, the symmetrisation of both, and
  `w *= sqrt(pi) / w.sum()` (hermegauss `sqrt(2·pi)`; np.sum's pairwise schedule).
- NumPy's `_normed_hermite_n` / `_normed_hermite_e_n` is ONE IL kernel
  (`Backends/Kernels/Direct/DirectILKernelGenerator.PolyNormedHermite.cs`), not a series-evaluation step program: that
  float64 emitter does not pin which NaN an add / multiply keeps, and NumPy's results reach NaN. Per point, NumPy's
  statements in NumPy's order: `c0 = (-c1)·s1` (np.negative as a sign-bit XOR), `c1 = tmp + (c1·x)·s2`, the final
  `c0 + (c1·x)·√2` (hermegauss `c0 + c1·x`), with the ufunc NaN rule (an array-array add / multiply keeps the SECOND
  operand's NaN) through the house emitters (`NaNOperandOrder.NumPyUfunc`). Every lane starts at `c0 = +0.0`, `c1 = C`,
  the bits of NumPy's scalar first steps (`0. + array` turns -0.0 into +0.0). Blocks of 4 vectors per step (6 spill
  AVX2's registers), then at most one block of 2, single vectors and a scalar tail; a second IL kernel writes NumPy's
  factors `sqrt((nd-1)/nd)`, `sqrt(2/nd)` (`sqrt(1/nd)`) once per call, and the degree-(deg − 1) recurrence reads the
  same table from its second row. The kernel has a scalar path (bit-exact with AVX2 or all hardware intrinsics off),
  but the compositions need SIMD support: the companion takes its sqrt through a house unary kernel that emits vector
  code unconditionally.
- NumPy's own results turn special (identical for both Hermite functions): from degree 371 some weights are 0 (the final
  scale underflows), from 372 some are NaN (`w.sum()` overflows to inf, `inf · 0`), from 741 some nodes are NaN (the
  recurrence overflows: `inf - inf`); the kernel's NaN payloads match. The oracle compares a NaN as a token, so the NaN
  operand order is held by the raw-byte unit tests (`SpecialPoints_AreNumPysBits`) and the benchmark's SHA-256 check:
  today's RyuJIT register allocation happens to keep NumPy's NaN even without the explicit order, and the explicit
  `NumPyUfunc` emitters keep it that way when the JIT allocates differently.
- From degree ~100 both libraries wait on the same `syevd`, so the speed ratio falls toward the eigvalsh ceiling; a
  cheaper tridiagonal solver would give other bits.
- Gates: `polygauss.jsonl` (portable errors) and `polygauss_parity.jsonl` (host-pinned values / flags),
  `Polynomial/PolynomialGaussTests.cs`, `Backends/Kernels/PolyNormedHermiteKernelTests.cs`.

## Chebyshev interpolation: `chebinterpolate`

- One overload (the one-signature rule): `chebinterpolate(Func<NDArray, object[], object> func, object deg, object args =
  null)`. func receives the points (chebpts1's new float64 array of `deg + 1` elements) and the unpacked args, and
  returns anything np.dot accepts: an NDArray of any dtype (a 2-D `(deg + 1, k)` array interpolates k functions; an N-D
  one contracts its second-to-last axis), a Python list / tuple, a typed C# array, or a scalar (np.dot of a 0-d operand is
  a product: the result is then the `(deg + 1, deg + 1)` matrix). A null args is NumPy's `()`.
- Engine (`Polynomial/Package/NDPolyInterpolate.cs`): chebpts1's emitted fill, chebvander (its `m.T` is the vander's
  C-contiguous buffer, as in NumPy), np.dot (gemv / gemm / the N-D dotfunc through the backend), then `c[0] /= order` and
  `c[1:] /= 0.5*order` as in-place np.divide on views (complex: CDOUBLE_divide, a multiplication by the reciprocal of
  the real divisor, which NumPy's scalar `c[0]` path also calls).
- deg is `np.asarray(deg)`: a 0-d integer array is required (a C# integer / BigInteger is a Python int: int64, uint64
  from 2**63, an OBJECT array past 2**64; a char is np.uint16), else `deg must be an int` (after np.asarray's
  inhomogeneous ValueError for a ragged list). `deg + 1` WRAPS in deg's dtype (a ufunc add on a 0-d array):
  np.uint8(255), np.int8(127), char 65535, 2**63 − 1 and 2**64 − 1 give chebpts1's `npts must be >= 1`; 2**63 lands in
  chebpts1's 2**63 band, so func is called with an EMPTY array and chebvander raises `Maximum allowed dimension
  exceeded`. args is Python's `*args`: a tuple, a list, an NDArray (first-axis items), a str (one-character strs), a
  dict (keys); a 0-d array is `iteration over a 0-d array`, anything else `Value after * must be an iterable, not
  {type}` (CPython 3.12), raised before a null func's `'NoneType' object is not callable`.
- Ownership (`[NDScoped]`): the points handed to func and every array func creates are reclaimed when the call returns;
  `NDScope.Detach` keeps one inside func. A func result created before the call stays the caller's.
- Not reproduced: a func result NumPy computes with as Python objects (None, a dict, a list holding None or a Python int
  past uint64) is NotSupportedException; a numpy.matrix result is computed plain (NumPy returns a matrix), `[Misaligned]`.
  Without the backend np.dot is the managed product: the same coefficients to rounding.
- Gates: `polyinterp.jsonl` (portable errors / flags) and `polyinterp_parity.jsonl` (host-pinned values; the callables
  are a menu of named functions both sides implement, `gen_oracle.py`'s `_PI_FUNCS` and `OpRegistry.PolyInterp.cs`),
  `Polynomial/PolynomialInterpolateTests.cs`.

## Least-squares fits: `polyfit`, `chebfit`, `legfit`, `lagfit`, `hermfit`, `hermefit`

- One overload per basis, `{p}fit(object x, object y, object deg, object rcond = null, bool full = false, object w =
  null)` → `NDPolyFitResult`, an `NDArray` that IS the coefficient array (it indexes, evaluates, takes operators). With
  `full: true` it also carries `residuals`, `rank`, `singular_values`, `rcond` and deconstructs as NumPy's
  `(c, [resids, rank, s, rcond])`: `var (c, (res, rank, s, rc)) = np.polynomial.chebyshev.chebfit(x, y, 3, full: true);`
  (without full the deconstruction is a ValueError). The statistics are detached from the call's scope: whoever holds
  the result disposes them with it.
- Engine (`Polynomial/Package/NDPolyFit.cs`): `polyutils._fit`'s statements in NumPy's order: `np.asarray(a) + 0.0`
  (PolyNumber's weak Python float; skipped for a numeric NDArray x, whose `+ 0.0` the Vandermonde kernel repeats in its
  load stage), `{p}vander` (lhs is the Vandermonde buffer's C-ordered `(k, n)` view), `np.take(van.T, sorted, axis=0)` for
  a degree list (NumPy's `van[:, deg]` is a transposed MapIter view whose `.T` is C-ordered), the norm as ONE fused
  `np.evaluate(Sum(Square(lhs), axis 1))` (complex: `Square(Real) + Square(Imag)`; np.sum's pairwise schedule, no
  squared temporary), `scl[scl == 0] = 1` only when `count_nonzero` finds a zero, the scaling `lhs / scl[:, None]`
  written over lhs (NumPy's `lhs.T / scl` has the same quotients), `np.linalg.lstsq`, `c / scl`, and the zeros
  re-expansion `cc[deg] = c` for a degree list (a duplicate's last row wins). The OpenBLAS backend's `Linearize` and
  lstsq's b copy move a column that is contiguous in the source as one block, as NumPy's `linearize_matrix` does with
  BLAS `?copy`.
- Arguments: x / y / w through np.asarray's coercion (`UnmanagedStorage.FromSequence`) and `+ 0.0`: None is Python's
  `unsupported operand type(s) for +: 'NoneType' and 'float'`, a str / str list / str ndarray the ufunc add's no-loop
  UFuncTypeError, an object list NotSupported. deg: a Python int (int64, uint64 from 2**63), a char (np.uint16), a 0-d
  or non-empty 1-D integer array_like; anything else (a bool, a float, a list of floats, an EMPTY int array) is `deg
  must be an int or non-empty 1-D array of int`. rcond: None → `len(x) * finfo(x.dtype).eps` (NEP 50: float32 for a
  float32 x); a number or 0-d array is converted to double for lstsq; a Python complex is CPython's `float() argument
  must be a string or a real number, not 'complex'` (0.5+0j too), a complex / str array lstsq's same_kind cast text, an
  rcond array with dimensions NotSupported (NumPy broadcasts the gufunc over it), `[Misaligned]`.
- Dtypes: float32 x and y fit float32 (lstsq computes in double, NumPy's "lite" rule; the statistics float32 too); a
  float16 x reaches lstsq's `array type float16 is unsupported in linalg` after the Vandermonde, as in NumPy; complex
  points, y or w fit complex; an int / bool x is float64.
- Ownership: c is NumPy's object: for a scalar degree a VIEW of the quotient (OWNDATA false, `(c.T / scl).T`), for a
  degree list an owner (`cc`). RankWarning is not emitted (house convention).
- lstsq's gelsd is most of the call on both sides, so each benchmark cell's ceiling is NumPy's whole call over
  NumSharp's lstsq alone.
- Gates: `polyfit.jsonl` (portable errors in NumPy's order, argument kinds, flags) and `polyfit_parity.jsonl` (host-pinned
  values / statistics / flags), `Polynomial/PolynomialFitTests.cs`.
