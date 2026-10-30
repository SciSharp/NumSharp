---
paths:
  - "src/NumSharp.Core/Polynomial/**"
  - "src/NumSharp.Core/Backends/Kernels/**/*Poly*.cs"
  - "src/NumSharp.Core/Math/NDArray.SlidingDot*.cs"
  - "test/NumSharp.Tests/Polynomial/**"
  - "test/NumSharp.Tests.Oracle/Fuzz/OpRegistry.Poly*.cs"
  - "test/NumSharp.Tests.Oracle/Fuzz/corpus/poly*.jsonl"
  - "benchmark/polynomial/**"
---

# The polynomial package (2/2): series algebra, Vandermonde matrices, companion matrices and roots

Evaluation, the additive family, polyutils, calculus and the shared argument coercion: `polynomial-series.md`.

## Series algebra

`{p}mulx`, `{p}mul`, `{p}div`, `{p}pow`, `{p}fromroots` for the six bases, and `X2poly`/`poly2X` for the five non-power
bases.
- NumPy runs these as Python loops over SHORT series, every statement one of: `as_series`, `_add`/`_sub`, a `{p}mulx`, an
  ARRAY op with a Python int or NumPy scalar (`c[-i] * xs`, `(c1 * (nd - 1)) / nd`, `c1[i:j] -= c2 * c1[j]`),
  `np.convolve`, `trimseq`, or scalarmath on one element. The engine (`Polynomial/Package/NDPolyAlgebra.cs` + the
  per-basis `NDPolyAlgebra.Bases.cs`) replays that Python statement for statement, in NumPy's order and dtypes, over raw
  series (`PolySer`: pointer, length, dtype, and whether NumPy returns it as a view) in a per-thread bump arena
  (`PolyArena`). Each statement runs through the kernel NumPy's statement runs through in NumSharp:
  - array ops: the ufunc kernels (`GetPolyHouseBinaryKernel`, flat slot fronts over `GetMixedTypeKernel`: simd_cmul,
    CDOUBLE_divide's Smith, the HALF loop);
  - as_series / `_add` / `_sub`: the additive family's substrate (`TrimLength`, `CopyInto`, `CombineInto`);
  - `{p}mulx`: an integral-layout calculus kernel (`PolyCalcKey.Mulx`); chebmulx of ≥ 2 terms the fused one-pass kernel;
  - `np.convolve`: `NDArray.SlidingCorrelateInto`, the engine `np.convolve` itself runs, its OpenBLAS route included;
  - scalarmath: the one-element kernels (the naive complex product, NumPy's scalar division).

  So the only C# is NumPy's own Python: its loops are over SERIES, never elements.
- Bit-exact with NumPy 2.4.2, except the convolution bases' BLAS-bound products without the backend (below). Oracle:
  `polyalgebra.jsonl` (portable, zero excused) and the host-pinned `polyalgebra_parity.jsonl` (the BLAS-bound products,
  byte-exact with `NumSharp.Interop.OpenBLAS` at threads = 1, Inconclusive off the pinned host). Unit tests
  `Polynomial/PolynomialAlgebraTests.cs` (NumPy's `TestArithmetic`/`TestMisc` ported, dtype-quirk byte dumps, decimal and
  char, the arena, the fused chebmulx kernel, the non-finite complex product, the ±0 divergence) and
  `PolynomialAlgebraArgumentKindsTests.cs`.
- Dtypes follow NumPy's Python statement by statement, not a promotion rule:
  - a recurrence-basis product (leg/lag/herm/herme) whose SHORTER factor has one term binds the Python int `c1 = 0`,
    which as_series makes float64 `[0.]`, so a float16/float32 product turns float64;
  - `_div`'s remainder is computed against the int unit series `[0]*i + [1]`, so a float32 division's remainder is
    float64 while its quotient is float32 (polydiv, which skips `_div`, keeps float32 in both; its remainder is a VIEW);
  - poly2X starts from `res = 0` (float64); fromroots' lines are `np.array([-r, 1])` (strong array coercion: float16/32
    roots give float64); `{p}pow(c, 0)` is `[1]` in c's dtype.
- The arena: a series costs a pointer bump, not an NDArray. Every internal function returns its result through `Keep`,
  which moves it down to the function's entry mark, so memory is bounded by the live series. Blocks come from
  `SizeBucketedBufferPool`, and growth blocks (powers of two) go back to it when the outermost call exits (fresh OS
  memory would cost first-touch page faults on every call). Only the final result becomes an NDArray, with NumPy's
  ownership: a trimseq slice is a VIEW (`flags.owndata` false).
- Products and np.convolve's BLAS switch: NumPy's power- and Chebyshev-basis products are `np.convolve`, whose dotfunc
  switches to OpenBLAS's VECTOR kernels once a dot is long enough: ddot at 16 terms, sdot at 32 (below that float32
  products are summed in a DOUBLE), zdotu at 8 (below that four separate double sums); float16's HALF_dot is a
  sequential float32 sum at every length. Without the backend NumSharp reproduces every scalar regime byte for byte
  (`DotSimd`, `SdotManaged`, `ZdotuManaged`) and sums the vector-regime positions in its own order, bounded-ULP; with
  `NumSharp.Interop.OpenBLAS` those positions run the same `?dot` NumPy calls, byte-exact. That split is why the corpus
  has two tiers: the generator records every `np.convolve` a case makes (`_PAConvRecorder`) and routes a case with a
  BLAS-bound product to the host-pinned file; `BlasBackendDelta` replays the affected ordinary cases with the backend on
  (the outcome flips are byte-checked against NumPy).
- The long managed products (`Math/NDArray.SlidingDot.Long.cs`) are blocked kernels: a block of B consecutive output
  positions shares ONE loop over the kernel; each step is one broadcast plus one load per vector, with fused multiply-adds
  into eight independent chains; the core reads the data directly, and the ≤ B − 1 head/tail steps read tiny zero-padded
  edge windows. A padding lane adds `0·k[s]`, which is 0 only for a FINITE kernel, so a kernel holding inf/NaN declines to
  the per-position kernels. The positions shorter than the vector block keep their exact scalar-regime sums. float16 has
  an exact blocked kernel (sequential per lane, separate multiply and add), byte-exact at every position.
- The complex128 ceiling: a complex product is four real multiply-adds, one 256-bit FMA per complex MAC on both sides
  (zdotu's AVX2 microkernel does exactly that), so the blocked kernel wins only NumPy's per-position call overhead.
  Rejected: Gauss's three-multiply trick (it loses componentwise accuracy on real × complex imaginary parts) and FFT
  convolution (its error is normwise, not per coefficient).
- The fused chebmulx kernel (`DirectILKernelGenerator.PolyAlgebra.cs`, `GetPolyChebMulxKernel`): chebmulx is NumPy's one
  mulx that is three array statements (`tmp = c[1:]/2; prd[2:] = tmp; prd[0:-2] += tmp`), whose combined effect
  `prd[j] = c[j-1]/2 + c[j+1]/2` is one IL pass:
  - the vector stage uses the evaluation family's lane kinds and halves by a MULTIPLY by 0.5 (`x/2 ≡ x·0.5` for every
    binary float: the exact half, rounded once);
  - float16 halves in float32, then `PolyLaneOps.HalfHalveOnGrid` rounds the half to the f16 grid (the magic-number
    trick: `0.75 + |y|` has float32 ulp 2^-24, the f16 subnormal step), then a PLAIN float32 add, rounded once by the
    store (HALF_add);
  - complex uses Smith's branch with the divisor 2+0j prepared once (`CDivPrep`/`CDivBy`: rat = +0, scl = 0.5);
  - the head, remainder, tail and decimal run the literal scalar ops; a contiguous same-dtype series is read in place
    (no as_series copy).

  Gates: `ChebMulx_FusedKernel_VectorStage_MatchesScalarEmission`, `ChebMulx_Float16_EveryBitPattern_IsTheHalfLoop`, and
  corpus section L (special values through the vector stage, every float16 pattern below 2^-12).
- `[Misaligned]` (unit-pinned, kept out of the corpus): fromroots sorts its roots, and where +0.0 and −0.0 meet, NumPy's
  x86-simd-sort keeps an order that depends on the CPU; NumSharp's sort puts −0.0 first, which decides a zero
  coefficient's sign in the recurrence bases. A NaN root is not a divergence: the sort turns it into the positive quiet
  NaN and `-r` negates it, as in NumPy.
- Argument kinds:
  - `{p}pow(object c, object pow)` (the default limit: 16, None for polypow) and `{p}pow(object c, object pow, object
    maxpower)` (null is None), ported in `Polynomial/Package/NDPolyPowerArgument.cs` from NumPy's three statements.
    `power = int(pow)` is CPython's `int()`: bool 0/1, np.float16 (Half) / decimal truncate, a str parses (`"3"`, `" 3 "`,
    `"3_0"`) and then fails `power != pow`, NaN / inf / complex / None / list / tuple raise CPython's texts, an ndarray of
    one or more dims raises `only 0-dimensional arrays can be converted to Python scalars` whatever its size, a 0-d one
    converts its element. `power > maxpower` is exact for Python numbers (NaN never exceeds, -inf always), decimal, char,
    BigInteger and integer ndarrays, but NEP 50 for np.float16 and float/complex/bool ndarrays: the weak int goes to a
    double first (`int too large to convert to float` at 2**1024), then to the dtype (2049 rounds to float16 2048: not
    exceeded; float32 double-rounds), complex compares by NumPy's CGT, bool in int64 (`int too big to convert` past
    int64); an ndarray's result must have exactly one element (NumPy's two truth-value texts), its conversion error
    first. C# binds `pow` to the int / double overloads first (a `long`/`ulong` converts to double), so only kinds they
    cannot take reach the object overloads; the corpus replays them as C# source would bind them
    (`OpRegistry.PolyAlgebra.cs` `PolyPow`).
  - A None / str series: `polyutils.as_series` converts every argument, checks sizes and dims, and only THEN meets the
    common type: a str array (a list holding a str) fails it (`Coefficient arrays have no common type`), an object array
    (None, a non-numeric object, an oversized int) makes NumPy compute with Python objects. So
    `PolySequence.ToArrayOrNonNumeric` reports such a sequence as NumPy's array SHAPE (`PolyNonNumericArray`: dims,
    object-or-str, the refusal), `AsCoefficientArray` makes it a `PolySeriesView.NonNumericArray` (None and an
    unconvertible scalar are one-element object arrays), and the refusal is raised by `NDPolySeries.CommonType`, or later
    where NumPy still has checks to run: div's `c2[-1] == 0` (on the object copies: `polydiv([None], [False])` is
    ZeroDivisionError) and pow's power checks. Evaluation and calculus compute at once and refuse on conversion.
  - Complex products with infinities or NaNs below the vector block are managed-exact through np.convolve's zdotu result
    model (`float-math.md`); `_PAConvRecorder.blas_bound` does not count non-finite values.
  - Not supported: an OBJECT array's computation (NumPy returns object arrays of Python ints / None, or raises its object
    arithmetic's TypeError) raises `NotSupportedException`; a vector-regime product can overflow to inf in one summation
    order and not the other.
  - Corpus (generator functions of `test/oracle/gen_oracle.py`): section M of `gen_polyalgebra` (pow/maxpower kinds; the deferred refusal filtered to outcomes NumPy reaches
    before any object arithmetic, `_pa_object_land`; the non-finite complex products), section Q of `gen_polyseries`
    (the deferred refusal in the additive family), and the `groupa` tier's complex convolve/correlate block.
- Traps:
  - A float16 lane kind's generic `Bin` rounds every op to the f16 grid (narrow + widen + NaN blends): round only where
    NumPy's float16 array holds the value (the halves), and let the store's narrow be the add's rounding.
  - Emit halving as a multiply, not `Bin(Divide)`: `vdivpd` bounds a streaming pass at one division per element (the
    equivalence is exact only for a power-of-two divisor).
  - A corpus of random operands never reaches a vector stage's rare branches (the subnormal float16 halves appear in no
    random series): build the edge set explicitly.
  - Slow SIMD with no calls is instruction count, not inlining: check with `DOTNET_JitDisasm="NDPolyChebMulx_*"` +
    `DOTNET_JitStdOutFile`.
  - A one-element array's stride is observable: NumPy's dot passes a size-1 operand through with whatever stride it has
    (it is C-contiguous regardless), and cblas refuses a non-positive one, so `polymul(c, [x])` and `polymul(c, [x, y])`
    reach different complex dot code. Materializing operands (`MaterializeForSliding`) loses that stride: decide
    `DotOperandBlasable` from the caller's array first.
  - A corpus case NumPy only reaches through object arithmetic cannot be gated: filter the generator's candidates by
    NumPy's outcome (`_pa_object_land`), not by hand.

## Vandermonde matrices: `{p}vander`, `{p}vander2d`, `{p}vander3d`

Bit-exact with NumPy 2.4.2: `polyvander.jsonl` (zero excused), `Polynomial/PolynomialVanderTests.cs`,
`PolynomialVanderArgumentKindsTests.cs`.
- Driver (`Polynomial/Package/NDPolyVander.cs`), NumPy's statements in NumPy's order, which is also its error order:
  - 1-D: `_as_int(deg)` → `deg must be non-negative` → `np.array(x, ndmin=1)` → `np.empty`'s dimension conversion
    (`Maximum allowed dimension exceeded` at ≥ 2^63 − 1 rows, AllocationGuard's `array is too big` below it, MemoryError
    past the allocator) → the recurrence → `np.moveaxis`;
  - 2-D / 3-D run `_vander_nd`'s order: `len(deg)` (`object of type 'int' has no len()`, `len() of unsized object`) →
    `Expected N dimensions of degrees, got K` → `np.asarray(points) + 0.0` (strong promotion; a ragged stack is the
    inhomogeneous ValueError) → per dimension its checks and allocation, the product's from the second dimension on → an
    empty stack's reshape error;
  - the result is always a VIEW (OWNDATA false): F-contiguous for 1-D points, both C and F for a scalar / degree 0 /
    empty, neither for N-D points.
- The degree text is `format(deg, '')`, not `str()` (`PolyIndexArgument` in `NDPolyIndexArgument.cs`): a 0-d array formats
  through its scalar (`np.float16(1000)` → `1000.0`), an ndim ≥ 1 array through `array_str`, a list/tuple through its
  items' REPR (`[np.float16(1e+03), array([1.5]), (2,)]`). `operator.index` takes bool, every integer width,
  `BigInteger`, char and a 0-d integer array, not a 0-d bool. A `char` degree container reads `numpy.uint16` in the
  `len()` text (`NDPolySeries.PythonTypeName`).
- Kernel (`Backends/Kernels/Direct/DirectILKernelGenerator.PolyVander.cs`): the six recurrences are DATA
  (`PolyVanderRoutines`) in the calculus kernel's step language, run forward, NumPy's source expressions token for token
  (`legvander`'s `v[i-1]*x*(2i-1)` is `(v[i-1]*x)*(2i-1)`). One whole-array kernel per (basis, dtype, 1/2/3-D, source
  dtypes, streamed), as four DynamicMethod stages: LOAD (fusing the conversion with NumPy's `+ 0.0`: -0.0 → +0.0, a sNaN
  quieted with its payload kept), RECURRENCE over a block of points, PRODUCT (writing each output row
  `(a*(dy+1)+b)[*(dz+1)+c]` straight into the final layout; 3-D through a scratch row holding NumPy's rounded
  `V_x[a]*V_y[b]` temporary), ROOT (walking the blocks). Blocks are sized so every per-dimension matrix stays in L1/L2.
  A 2-D / 3-D stack of same-shape NDArrays is read in place per coordinate (each loaded from its own dtype: the two-step
  `stack + 0.0` rounds at most once, the same value).
- Results of 32 MiB or more stream (`NDPolyVander.NonTemporalMinBytes`, `PolyVanderKey.NonTemporal`): each product row
  runs a scalar head up to the store alignment, then aligned non-temporal vector stores (`vmovntdq`; 16-byte for
  float16), then the scalar tail; the stage ends with `sfence`; the bytes are identical. Test hooks
  `NDPolyVander.NonTemporalOverride` (force on/off) and `BlockOverride` (tiny blocks).
- Scalar points that stack into an OBJECT array are computed: a Python int past uint64 (`BigInteger` 2**70) among
  vander2d/3d scalar points makes NumPy's stack object, and its per-element `+ 0.0` then hands every dimension its OWN
  number: `float(int)`, correctly rounded, with CPython's OverflowError past the float range; a float / complex; NEP 50's
  dtype for an np.float16 or a 0-d array (`NDPolyVander.ObjectScalarPoints`, via `PolyNumber.Binary`). One dtype → the
  kernel path (`StackScalars`); mixed dtypes → `MixedDtypeVander`, each dimension's `{p}vander` in its own dtype,
  multiplied through np.multiply's promotion. A None / str element stays refused.
- Matched: typed arrays, `Memory<T>`, `object[,]`, jagged and `NDArray[]` points; `short`/`byte`/`char` degrees binding
  the int overload; 5-D/6-D transposed and reversed points; result writeability; kernel first-compiles and calls under
  contention.
- Traps:
  - NumPy's float multiply keeps the SECOND operand's NaN when both are NaN (MSVC `FLOAT/DOUBLE_multiply`, every loop
    length); x86 `mulpd` keeps the first, and RyuJIT may swap a commutative multiply's operands. The outer product is the
    one place two independently sourced NaNs meet: `PolyVanderOps.Mul*` blends explicitly, only when the product holds a
    NaN lane. A random corpus never pairs two NaNs: it takes special values in BOTH coordinates.
  - A valid huge degree over an object x: NumPy computes `x + 0.0` with Python objects BEFORE the dimension error, where
    NumSharp refuses the object stack; only degree errors that come first are corpus cases.
  - NumPy hangs on a huge degree over EMPTY points whose byte count does not overflow (its Python loop runs 2^50 times
    over nothing); NumSharp returns at once. Never put one in a generator.
  - A lone `object[]` argument to a `params object[]` parameter IS the params array (`L(L(1.0, 2.0))` is `L(1.0, 2.0)`):
    spell nesting out with `new object[] { … }`.
  - `np.multiply` of two same-shape float64 arrays keeps the FIRST operand's NaN where NumPy keeps the second's
    (library-wide; the oracle compares NaNs as tokens, so only byte tests see it); the vander kernel imposes NumPy's
    priority itself, and only the mixed-dtype object-stack path goes through np.multiply.
  - A page-fault-bound benchmark cell measures the host: fresh > 64 MiB results swing with the OS's zeroed-page list, so
    the biggest cells run best-of-9 on both sides; A/B such a kernel change in ONE process, interleaving the
    configurations round by round.

## Companion matrices and roots: `{p}companion`, `{p}roots`

Bit-exact with NumPy 2.4.2: `polyroots.jsonl` (portable: every companion, plus the roots that never reach LAPACK) and the
host-pinned `polyroots_parity.jsonl` (the roots that run geev; byte-exact with `NumSharp.Interop.OpenBLAS` at threads = 1,
Inconclusive off the pinned host), zero excused; `Polynomial/PolynomialRootsTests.cs`.
- Engine (`Polynomial/Package/NDPolyAlgebra.Roots.cs`, on the series-algebra arena), NumPy's statements in NumPy's order:
  as_series, the length checks, and for two terms the linear root's 1×1 matrix (scalarmath: `-c0/c1`, lag `1 + c0/c1`,
  herm `-.5*c0/c1`); `np.zeros((n, n))`, then the diagonals through `mat.reshape(-1)[k::n+1]` (`GetDiagWriteKernel`; a
  stride-0 source broadcasts a scalar), then ONE in-place update of the last column. Every element operation runs
  through the kernel NumPy's statement runs through: the binary ufunc loops, the conversions, np.sqrt's unary loop,
  np.multiply.accumulate's scan (herm/herme's `scl`), and np.arange through an int64 ramp IL kernel
  (`DirectILKernelGenerator.PolyRoots.cs`). Roots = `np.linalg.eigvals` of the companion, ROTATED for every basis but
  the power series (`np.flip` over both axes, NumPy's `[::-1, ::-1]` view), sorted in place. Real roots are a VIEW of
  eigvals' complex result (OWNDATA false, stride 16 bytes), as NumPy's.
- The dtype rule decides the bits: the helper vectors (`scl`, `top`, `mid`) are float64, so wherever one meets the series
  the loop is `result_type(c.dtype, float64)`. A float16/float32 series' cheb/leg/herm/herme last column is therefore
  computed in float64 and rounded ONCE into the matrix; poly/lag stay in the series dtype.
- The zero matrix is a write-once allocation following `np.tri`'s policy (`np.PrefersWriteOnce`): up to 64 MiB a pooled
  buffer plus `np.ZeroBytes`, above that OS-zeroed pages (a calloc'd matrix would fault every fresh page on the diagonal
  writes).
- `np.linalg.eig`/`eigvals` of a float32 operand with complex eigenvalues round each component to float32 (NumPy returns
  complex64, geev in double then `astype(complex64)`; `CollapseEig`: one cast and copy through a float64 view of a
  contiguous result, the per-lane form otherwise). `AssertFinite` (eig/eigvals) uses the fused `FiniteScan.IsAllFinite`
  kernel.
- Object series: a Python int past uint64, None or another object among the terms makes as_series continue in the OBJECT
  dtype, the items keeping their kinds (Python numbers, NumPy scalars, 0-d arrays):
  - the length check runs on the TRIMMED object array (`ObjectTrimLength` scans Python's `item != 0` from the end; a 0-d
    array item compares elementwise, so a trailing 0-d zero is trimmed too); `[None, 0]` and `[2**70, 0]` raise NumPy's
    ValueError;
  - two terms: the linear root computed on the ITEMS (`ObjectLinearRoot`): CPython for Python numbers, scalarmath for
    NumPy scalars, ufuncs for 0-d arrays; the result is a numeric array of that number's dtype;
  - three or more: NumPy builds an object companion matrix. `ObjectCompanionArithmetic` replays the element operations
    of its last-column statement that can raise and raises NumPy's first error, always a Python int too large for a
    float (`integer division result too large for a float` for Python int / Python int, `int too large to convert to
    float` for any other operand; the divisor is cast to the object dtype first, so a 0-d integer divisor is a Python int
    there; herm/herme multiply by their float64 helper first). When nothing raises, `{p}companion` refuses (an object
    matrix), and `{p}roots` raises eigvals' `ufunc 'isfinite' not supported…` TypeError before LAPACK, so no backend is
    needed;
  - not supported (`NotSupportedException`): a one-term object series' roots (NumPy's `np.array([], dtype=object)`,
    `[Misaligned]`), the object companion matrix itself, CPython's TypeErrors from None/str arithmetic.
  - The shared coercion classifies a Python int past uint64 without throwing (`PolyNumber.TryDiscoveredDtype` +
    `ObjectArrayRefusal`: the refusal is built, never thrown; a thrown and caught exception costs ~1.4 µs per leaf).
- Errors in NumPy's order: as_series' texts first, then `Series must have maximum degree of at least 1.`. Roots of degree
  ≥ 2 then meet eigvals' checks in eigvals' order: finiteness (LinAlgError) before dtype (the float16 / decimal
  TypeError). All of these fire before LAPACK, so they need no backend; with no backend, a root of degree ≥ 2 raises
  `MissingBackendException`.
- Speed floors: LAPACK `geev` for roots of degree ≥ 50, and zeroing the companion matrix (a hot buffer clears at the same
  rate whatever the method: `Span.Clear`, `InitBlock` and an AVX2 loop are equal, non-temporal stores slower).
- Traps:
  - Decide the host split by recording, not predicting: the generator replaces `np.linalg.eigvals` per case (the
    polynomial modules look it up at call time); geev ran exactly when eigvals got a finite float32/float64/complex128
    matrix.
  - NumPy's SIMD sort orders value-tied ±0 elements by CPU: the generator skips such results.
  - Never put a `?:` with an `object[]` arm and an `NDArray` arm in a benchmark: it is typed NDArray through the implicit
    Array → NDArray conversion. Cast both arms to `object`.
  - cmd's `%1` splits on commas: `bench.cmd C/poly/,T/poly/` passes only `C/poly/`.
