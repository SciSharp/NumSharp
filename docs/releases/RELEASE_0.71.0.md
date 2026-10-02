# NumSharp 0.71.0

<!-- The owner's intro paragraph goes here (0.70.0 opened with one in the owner's own voice). -->

### 🧭 Overview
- NumPy 2.x API coverage **87.1% -> 95.5%** (488 -> 535 of the 560 headline APIs), and **494 of the 560 are oracle-verified** against committed NumPy 2.4.2 contracts - every figure from the [Supported Features Dashboard](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html).
  - `np.polynomial.*` - the `numpy.polynomial` package, **172 of its 193 functions** for all six bases, bit-exact.
  - `np.*` - **+48 functions**.
  - `np.random.*` - every NumPy bit generator, the whole `Generator` distribution surface, and a legacy `RandomState` that is NumPy's own sampler code line for line, all byte-identical streams.
  - `np.ma.*` - the masked-array module; `np.emath.*` - the automatic-domain math module.
- **`DType` is NumPy's two-level dtype model** - `NDArray.dtype` returns a `numpy.dtype`-shaped descriptor, with `np.dtypes`, byte order and `datetime64` / `timedelta64` descriptors.
- **Reductions on NumPy's own schedules** - `max`, `min`, `ptp`, `nanmax`, `nanmin`, `argmax` and the fused `np.evaluate` reductions match NumPy bit for bit, down to the sign of a ±0 tie and a NaN's payload, and most of them run faster.
- **~60 NumPy-parity and memory-safety fixes**, **14 measured speedups** and **14 breaking changes** - every line traced to one of 138 cited commits out of the release's 348.
- **1 new NuGet package** - **NumSharp.Interop.System.Numerics.Tensors**, a zero-copy bridge to .NET's `Tensor<T>` / `TensorSpan<T>`.
- **407,913 NumPy 2.4.2 oracle test cases** (0.70.0 shipped with 116K+) and 16,818 test methods.

## Detailed Breakdown
<details>
  <summary>Show All</summary>

### 📦 New NuGet Packages

One optional companion package ships for the first time, co-versioned with **NumSharp** 0.71.0 (it depends on **NumSharp**).

- **NumSharp.Interop.System.Numerics.Tensors** - zero-copy interop between `NDArray` and [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors) (`Tensor<T>`, `TensorSpan<T>`, `ReadOnlyTensorSpan<T>`); all 15 NumSharp dtypes cross as their own CLR type, `Half`, `Complex`, `decimal` and `char` included.
  - Export - `AsTensorSpan<T>()` views an `NDArray` of any non-negative-stride layout (contiguous, sliced, transposed, strided, broadcast) as a `TensorSpan<T>` over the same unmanaged buffer, through a disposable handle that keeps the buffer pinned; `ToTensor<T>()` copies into a dense `Tensor<T>`.
  - Import - `AsNDArray()` views a `Tensor<T>` as an `NDArray` without copying (writes go through to the tensor); `ToNDArray()` copies a `Tensor<T>`, `TensorSpan<T>` or `ReadOnlyTensorSpan<T>`.
  - The exported spans feed the BCL's SIMD `TensorPrimitives` directly, with no fill loop.
  - Limits - a negative-stride view is refused, because the BCL forbids negative strides (copy it first); the copies stop at the managed-array size limit; live export and import counters help leak checks.
  - Depends - [System.Numerics.Tensors](https://www.nuget.org/packages/System.Numerics.Tensors) 10.x only, a managed BCL package. Its tensor types are experimental, so the bridge marks its own API experimental too and you opt in once. No engine backend and no native asset: Core stays 100% managed.

### 📊 Dashboards & Docs

- **[Supported Features Dashboard](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html)** - now reports, for every API, whether its behavior is **oracle-verified** (committed NumPy 2.4.2 contracts resolve to it, directly or through a reviewed alias) next to whether it is available. Headline 95.5% available (535/560), 88.2% oracle-verified (494/560).
  - An oracle filter in the API explorer and an evidence pane per API: contract and error-contract counts, dtypes, layouts, corpus files and host pins.
  - The catalog also covers what lies beyond the headline - NumPy's submodules (`numpy.ma`, `numpy.emath`, `numpy.polynomial`, `numpy.char`, `numpy.strings`, ...) and the members of its objects (`nditer`, `dtype`, `finfo`, the random classes): 1,137 of 2,519 APIs available in that expanded scope, the dashboard's default view.
- **[Tests & Oracle Dashboard](https://scisharp.github.io/NumSharp/docs/tests-oracle-dashboard.html)** - 407,913 committed NumPy 2.4.2 oracle test cases over 793 operations (0.70.0: 116K+), and 16,818 test declarations.
- **[Benchmark Dashboard](https://scisharp.github.io/NumSharp/docs/benchmarks-dashboard.html)** - benchmark evidence for 452 of 518 benchmarkable APIs (the denominator grew with this release's new APIs).
- **[Documentation site](https://scisharp.github.io/NumSharp/)** - new guides, whose code samples run as tests:
  - [The absolute basics](https://scisharp.github.io/NumSharp/docs/absolute-basics.html), [Quickstart](https://scisharp.github.io/NumSharp/docs/quickstart.html), [Iterating & Enumerating](https://scisharp.github.io/NumSharp/docs/iterating-and-enumerating.html) and [System.Numerics.Tensors](https://scisharp.github.io/NumSharp/docs/interop/system-numerics-tensors.html).
  - [Fundamentals and usage](https://scisharp.github.io/NumSharp/docs/fundamentals/index.html) - array creation, indexing, getting & setting values, I/O, copies and views, strings and bytes, structured arrays, ufunc basics.
  - [Advanced usage](https://scisharp.github.io/NumSharp/docs/advanced/index.html) - extending NumSharp, native code & backends, under the hood.

### ✨ New APIs & Modules
- `np.polynomial` - the `numpy.polynomial` package for all six bases (`polynomial`, `chebyshev`, `legendre`, `laguerre`, `hermite`, `hermite_e`) plus `polyutils`: 172 of its 193 functions, bit-exact with NumPy 2.4.2 against generated oracles (139,882 cases, 0 excused); `{p}` below is `poly` / `cheb` / `leg` / `lag` / `herm` / `herme` - `57545200`, `38ed8da7`, `ac8608d9`, `38526155`, `e2ff9477`, `0c02740c` (+ `0896cbf8`, `d494e83d`).
  - `{p}val`, `{p}val2d`, `{p}val3d`, `{p}grid2d`, `{p}grid3d`, `{p}valnd` - evaluation.
  - `{p}add`, `{p}sub`, `{p}mulx`, `{p}mul`, `{p}div`, `{p}pow`, `{p}fromroots` - series arithmetic.
  - `{p}der`, `{p}int` - calculus, with every `m` / `k` / `lbnd` / `scl` / `axis` form.
  - `{p}vander`, `{p}vander2d`, `{p}vander3d` - Vandermonde matrices.
  - `{p}companion`, `{p}roots` - companion matrices and roots (the roots compute through the OpenBLAS backend's `eigvals`).
  - `{p}trim`, `{p}line`, `{p}domain`, `{p}zero`, `{p}one`, `{p}x`, the five `X2poly` / `poly2X` pairs - helpers, module constants and basis conversions.
  - `as_series`, `trimseq`, `trimcoef`, `getdomain`, `mapparms`, `mapdomain` - `polyutils`.
  - Speed - x4.65-x37.8 NumPy's speed by benchmark geomean per family (evaluation x7.5, additive x8.5, calculus x7.6, series algebra x37.8, Vandermonde x10.2, roots x4.65).
  - Still to come: fitting, Gauss quadrature, `chebpts` / `chebinterpolate`, `polyvalfromroots`, `format_float` and the six series classes.
- `np.random` bit-generator family - NumPy's engines, seeding and stream splitting, byte-identical to NumPy 2.4.2 and replayed overload by overload (493 members, 5 engines x 10 seeds) - `c1e0feb5`, `65bdea7b` (+ `0e39714b`, `eebe85a6`).
  - `PCG64DXSM`, `Philox`, `SFC64` - new bit generators; `PCG64.advance`, `jumped`, `random_raw`, `state` - engine controls; `MT19937` is now a NumPy `BitGenerator` too.
  - `SeedSequence.spawn`, `spawn_key`, `pool_size`, `state`, `BitGenerator.spawn`, `Generator.spawn`, `ISeedSequence` - seeding and independent streams.
  - `default_rng` - every NumPy seed form (integers, `BigInteger`, arrays, `SeedSequence`, a bit generator, a `Generator`, a `RandomState`, `null` for fresh entropy).
  - `np.random.PCG64(42)`, `np.random.Generator(...)`, `np.random.SeedSequence(...)` and the other classes - NumPy's module spellings; `get_bit_generator`, `set_bit_generator`.
- `Generator` distributions - NumPy's whole `numpy.random.Generator` surface on NumPy's modern algorithms, 29 samplers byte-identical to `default_rng` - `c0613b2f`.
  - `beta`, `chisquare`, `f`, `noncentral_chisquare`, `noncentral_f`, `standard_t`, `standard_cauchy`, `vonmises` - the gamma family and friends.
  - `pareto`, `weibull`, `power`, `laplace`, `gumbel`, `logistic`, `lognormal`, `rayleigh`, `wald`, `triangular` - one transform per draw.
  - `binomial`, `negative_binomial`, `poisson`, `zipf`, `geometric`, `hypergeometric`, `logseries` - discrete.
  - `multinomial`, `dirichlet`, `multivariate_normal` (`svd` / `eigh` / `cholesky`), `multivariate_hypergeometric` (`marginals` / `count`) - multivariate.
  - `pareto` / `power` stay within 2 ULP / `ceil(2/a)+3` ULP of NumPy, whose C runtime `expm1` has no portable twin; `multivariate_normal` is byte-identical with the OpenBLAS backend and may pick the other eigenvector sign without it.
- Array-valued distribution parameters - the 29 parameterized samplers of both `RandomState` and `Generator` take `NDArray` parameters, broadcast against `size` as NumPy does, byte-identical streams - `39d446e0` (+ `5f204884`).
  - `integers`, `randint` - array bounds and `BigInteger` bounds, so `g.integers(0, BigInteger.Pow(2, 64), dtype: np.uint64)` ports verbatim - `0e39714b`.
- Legacy `RandomState` gains NumPy's form over any bit generator (`np.random.RandomState(new PCG64(42))`), the dict `get_state(legacy: false)` / `set_state`, `tomaxint`, `ranf` and `sample` - `f87b4d04`.
- `np.ma` - NumPy's masked-array module on a new `NDMaskedArray` type, bit-exact vs NumPy 2.4.2 and gated by a 68,860-case differential corpus - `5f1476f5` (+ `ba15bad4`, `bf66bc0f`, `b33b6515`, `3fb1d897`, `e0d6a81c`, `67f8f96e`, `7a262afc`).
  - The ufunc wrappers, masked reductions, operators, creation, extrema and manipulation - the core.
  - `average`, `median`, `dot`, the set operations, `sort`, `unique`, `cov`, `corrcoef`, `convolve`, `correlate`, `polyfit`, `apply_along_axis`, `apply_over_axes` - the extras.
  - The masked indexer, hard masks and byte-exact masked `str` / `repr`; only the structured/record-dtype functions are missing.
- `np.emath` - automatic-domain `sqrt`, `log`, `log2`, `log10`, `logn`, `power`, `arccos`, `arcsin`, `arctanh` (an out-of-domain input gives the complex result instead of NaN) - `a3666a5f`.
- 48 new `np.*` functions, NumPy 2.4.2 parity, each gated by the differential-fuzz oracle -
  - `hypot` (correctly rounded: float32/float16 bit-exact, float64 bit-exact on ~91% of inputs and 1 ULP more accurate on the rest), `divmod`, `fmod`, `remainder`, `float_power`, `heaviside`, `spacing`, `frexp`, `ldexp`, `gcd`, `lcm`, `fabs`, `signbit`, `fix` - elementwise math ufuncs - `5c667b09`, `df232348`, `44636747`, `6e0e5ea2`, `ade8d6b5`, `83e0e43e`, `a15f0ee4`, `68a84d3c`, `46f5f651`.
  - `sinc`, `i0`, `real_if_close`, `unwrap` - special functions and signal helpers - `c3be0b7f`, `a0c6e2fe`, `877a36b2`, `b1937c51`.
  - `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser` - window functions, bit-identical - `0e9dbc42`.
  - `trapezoid`, `gradient` - numerical integration and differentiation, bit-exact - `22d5f4d3` (+ `f6037bc8`, `4898b40a`).
  - `histogram`, `histogram_bin_edges`, `histogramdd`, `histogram2d` - histograms - `2693d83e` (+ `0e80bc89`).
  - `packbits`, `unpackbits`, `bitwise_count` - bit packing and population count - `3936fd0d`, `bab51edb`.
  - `piecewise`, `putmask`, `put_along_axis` - indexed and conditional assignment - `76b2ab9d`, `86cc28b8`, `f417827e`.
  - `vectorize`, `frompyfunc`, `apply_along_axis`, `apply_over_axes`, `broadcast_shapes` - functional and shape helpers - `b1841830`, `53b7a149`.
  - `array_equiv`, `may_share_memory`, `shares_memory` (with `TooHardError`) - array predicates - `38f9a14e`, `4cd3b72f`.
  - `logspace`, `geomspace` - log-scale ranges, bit-exact - `d48737ec` (+ `3a367520`).
  - `binary_repr`, `base_repr`, `typename` - string helpers - `c2f129b5`, `7cd38a97`.
- Existing APIs gain NumPy's parameters -
  - `maximum`, `minimum`, `fmax`, `fmin` - `out=` / `where=` - `e5afe84c`.
  - `left_shift`, `right_shift`, `logical_and`, `logical_or`, `logical_xor`, `logical_not`, `modf` - `out=` / `where=` / `dtype=` - `3384a0b5`.
  - `array_equal` - `equal_nan=` - `38f9a14e`.
  - `indices`, `indices_sparse` - `long[]` / `Shape` dimensions (`np.indices(arr.shape)`), and `mgrid` past 2^31 elements - `07c7b94c`.
- `np.evaluate` / `NDExpr` - the fused-expression DSL covers most of the ufunc surface and every reduction, NumPy-typed and bit-exact (29,563-case oracle tier) - `6175a017`, `bd0091fb`, `e1400974`, `2f02d65d`, `eb1921e1`, `fbef1c4e`, `c5422cfc`, `dcbe7079`.
  - `Maximum`, `Minimum`, `FMax`, `FMin`, `Fmod`, `CopySign`, `NextAfter`, `LogAddExp`, `LogAddExp2`, `Hypot`, `Heaviside`, `Gcd`, `Lcm`, `LeftShift`, `RightShift` - binary nodes.
  - `Positive`, `Conjugate`, `Fabs`, `Spacing`, `SignBit`, `IsPosInf`, `IsNegInf`, `BitwiseCount`, `Rint`, `Real`, `Imag`, `Angle`, `Cast`, `Round(decimals)` - unary nodes.
  - `LogicalAnd`, `LogicalOr`, `LogicalXor`, `Select`, `Clip`, and the `<` / `>` / `<=` / `>=` operators on `NDExpr` - logic and selection.
  - `Any`, `All`, `CountNonzero`, `NanSum`, `NanProd`, `Ptp`, `NanMin`, `NanMax`, `ArgMax`, `ArgMin`, `NanMean`, `Var`, `Std` (`ddof`), the weighted `Average`, axis forms and `keepdims` - reductions.
  - `dtype=`, `casting=`, `order=`, `where=` - keywords; `expr.Compile()` returns a reusable `CompiledExpression`.
  - `If`, `Switch`, `Mux`, `Saturate`, `NanTo`, `Coalesce`, `Relu`, `LeakyRelu`, `Elu`, `Sigmoid`, `Swish`, `Softplus`, `Gelu`, `Step`, `Nand`, `Implies`, `IsClose`, `Between`, `Bucketize`, `Lerp` and 18 more - 38 decision and activation combinators that fuse like any node - `d3057b39` (+ `7aba9b4a`).
- `DType` system - NumPy's NEP 40-50 two-level dtype model (dtype classes and dtype instances), the descriptor every dtype parameter and `NDArray.dtype` now use - `896f02b2`, `c916c579`, `72e4d29c`.
  - `np.dtypes` (`Float64DType` and the other dtype classes), `np.datetime_data` - the class module.
  - `num`, `kind`, `char`, `itemsize`, `str`, `descr`, `byteorder`, `newbyteorder`, `isbuiltin`, `isnative`, `flags`, `type_num` (`NPY_TYPES`) - the full `numpy.dtype` surface.
  - Non-native byte orders (`np.dtype(">i4")`) and `datetime64` / `timedelta64` descriptors that parse, print, promote and cast-check (no datetime storage yet).
  - `promote_types`, `result_type`, `can_cast`, `issubdtype`, `isdtype` - `DType` overloads with NEP 50 weak scalars.
- Typed iteration and .NET interop -
  - `np.flat<T>(a)` / `a.flatiter.AsTyped<T>()` - typed by-`ref` C-order flat iteration - `fa2e8521`.
  - `np.ndenumerate<T>(a).AsRef()`, `np.ndindex(...).AsSpans()` - allocation-free enumeration - `afb0d5ce`.
  - `nd.Unsafe.Span<T>()`, `Memory<T>()`, `Bytes()`, `Pointer<T>()`, `TryGetSpan<T>()`, `TryGetMemory<T>()` - zero-copy BCL views of the buffer (they do not keep it alive: hold the `NDArray`); `nd.AsEnumerable<T>()` and the typed iterators' `ToArray()` / `CopyTo(Span<T>)` - safe copies - `37a112be`.
- `Py.RegisterNumSharpCodec()` (**NumSharp.Interop.pythonnet**) - the one-call codec registration, spelled on pythonnet's own `Py` (a C# 14 extension member) - `f18aaa23` (+ `b648ee53`).
- `NumSharp.Collections` - `ConcurrentOrderedDictionary<TKey,TValue>`, `ConcurrentOrderedCompactDictionary<TKey,TValue>` (about a third of the memory) and the lock-free `OrderedDictionary<TKey,TValue>`: insertion-ordered, index-addressable dictionaries, plus a vendored `NumSharp.Collections.Concurrent.ConcurrentDictionary<TKey,TValue>` with by-ref access - `39e5ba17`, `47c12604`, `ba5c347d` (+ `04677203`, `873900b1`, `478b5310`).

### 🧩 ndarray surface
- `ToArray<T>()` - a `T` other than the dtype converts like `astype` (it threw `ArrayTypeMismatchException`), and every layout copies in one pass (~5.8x faster than before on the measured set) - `73514657`.
- `ToMuliDimArray<T>()` / `(Array)nd` - one pass over any layout; float16, complex128, a `T` other than the dtype (converted like `astype`), 0-d and empty decimal arrays now work (they threw) - `c243201a`.
- `ToJaggedArray<T>()` - any rank (rank 6 came back wrong, deeper ranks threw), any layout (a transposed view gave wrong values, a column slice read out of bounds) and a `T` other than the dtype; 10.7x faster with 7.7x less memory - `4231d2d0` (+ `6f2ec473`, `ac89847a`).

### ⚡ Performance
*Ratios are NumPy ÷ NumSharp - higher is better (`x2` = twice NumPy's speed); `xLOW->xHIGH` spans the worst→best measured cell across sizes and dtypes.*
- `x0.70->x3.1` - `np.max` / `np.min` / `np.ptp` on NumPy's exact reduction schedules, slab reductions folding eight reduced indices per pass (axis cells on non-C layouts were x0.28-x1.0; L3-bound 4M cells of 1- and 4-byte lanes still trail) - `1d79ce61`, `f8f91b3d`.
- `x0.41->x4.7` - `np.nanmax` / `np.nanmin` (float32/float64 axis cells were x0.15-x0.50, flat C now x2.89 / float32 x3.33; float16/complex128 non-C axes run NumPy's scalar loop at x0.41-x0.69) - `de6333dd`, `268d91fd`.
- `x0.77->x106` - `np.argmax` / `np.argmin` along an axis walk the strides instead of a per-output div/mod walk (axis 0: float64 x4.4-x8.8, integers x10-x16, bool x95-x106; memory-bound contiguous rows at 4M x0.77-x1.04), and the 64-bit flat argmax runs NumPy's single-pass tournament - `ceac36f2`.
- `x0.89->x2.65` - `np.mean` / `np.sum` along an axis with a large output: the identity seed and the mean's division run as typed block passes (`np.mean(a, axis=0)` of 3x1M float32 was x0.29; `mean(axis=1)` over 3-wide rows stays below NumPy) - `3f22d30a`.
- `x1.43->x1.83` - `np.convolve` long products without the OpenBLAS backend (1000x1000 float64 was ~x0.9, complex128 x0.25) - `38526155`.
- `x1.15->x5.9` - complex128 `divide`, `multiply` and integer-exponent `power`, now on NumPy's own formulas - `dc9caff5`, `338d260a`, `3e7d6b33`.
- `x1.26->x4.0` - `np.modf` plain, `out=` and promoted calls (`out=` at 10M was x0.40) - `3384a0b5`.
- `x1.0->x16` - `np.array_equal` / `np.array_equiv` fused early-exit compare (x15-x16 at N=1000, parity at 100K, instant on an early mismatch) - `44604193`.
- `x1.05->x6.4` - every `np.random` bit generator fills through a bulk layer: all 105 engine x op x size cells beat NumPy - `aa9905b7`.
- `x1.08->x5.58` - legacy `RandomState` samplers (hypergeometric was x0.96, zipf x1.12, multinomial x0.97; the samplers bound by the C runtime's `pow` / `log` / `exp` stay at x1.17-x1.49) - `910a6480`.
- `x0.65->x17.6` - `np.evaluate` comparison, logical, `where` and bool trees vectorize instead of running the whole tree scalar (`a > 0.5` was x0.16) - `939d0636`.
- `x1.0->x22.2` - `np.evaluate` mixed-dtype trees vectorize through lane groups (`i4*2+f8` x14.2 and `f4*f8+f4` x22.2 at 100K; 1K cells sit at the per-call floor) - `3631fcc7`.
- `x0.86->x53` - `np.evaluate` reductions stream block by block instead of materializing their child, and `Any` / `All` stop at the deciding block (`any(a > b, axis=0)` was x0.37; `count_nonzero(a > b)` at 100K sits at the read-bandwidth ceiling) - `ee639862`, `0f8819b7`, `129fbaba`, `78692d91`, `ceac36f2`, `0f78f6e2`.
- `x2.06` - a small `np.evaluate` call skips the iterator like NumPy's trivial loop: `a*b+c` at n=8 runs in 255 ns vs NumPy's 526 (`out=`: 75 ns) - `30dbfe0d` (+ `df799381`).

### 🎯 Parity & Fixes
- `np.max` / `np.min` / `np.ptp`, flat and along an axis, run NumPy's own reduction schedules - the sign of a ±0 tie and a canonical-vs-payload NaN now match on every non-broadcast layout - `1d79ce61` (+ `129fbaba`, `f8f91b3d`).
- `np.nanmax` / `np.nanmin` are NumPy's `fmax.reduce` / `fmin.reduce` bit for bit: complex128 skips NaN (it propagated it), float16 on non-C layouts read the wrong elements, and an all-NaN float16 slice returns its first NaN - `de6333dd` (+ `268d91fd`).
- complex128 arithmetic bit-exact with NumPy - `divide` (1 ULP off on ~33% of operands), `multiply` (NumPy's fused `simd_cmul`), `power` with an integer exponent (up to ~1e14 ULP off) and `abs` (NumPy's fused `simd_cabsolute`; 35.5% of values were off) - `dc9caff5`, `338d260a`, `3e7d6b33`, `d1b89c11`.
- `percentile` / `quantile` / `median` bit-exact - NumPy's two-branch `_lerp` in each dtype's arithmetic, `median` as the mean of the middle pair, and a view at a non-zero offset no longer reads the wrong data - `d2c8d26e`.
- `np.round` / `np.around` with `decimals != 0` port `PyArray_Round` - negative decimals threw, complex values stayed unrounded, float32 rounded in double, and the result dtypes follow NumPy - `6db8238a`.
- `sum` / `prod` of a size-0 or size-1 array widen the accumulator by NEP 50 like any other size - `23ea2ad9`.
- `np.array_equal(a, a)` with a NaN in `a` returned `True`; NumPy returns `False` unless `equal_nan` - `38f9a14e`.
- int64-vs-uint64 comparisons are exact on every path (they went through float64) - `53602da9`.
- float16 `mod` / `floor_divide` / `fmod` by zero give ±inf like NumPy (they gave NaN), and float16 `add` / `subtract` / `multiply` / `divide` on the `out=` / `where=` / strided paths compute in float32 like NumPy's HALF loops (they double-rounded through double) - `df232348`, `b53a0cc1`.
- float16 signalling NaNs stay signalling across every float16 cast route, uint64 at or above 2^63 converts to float64/float32 with one rounding on .NET 8 (it rounded twice, inside mixed uint64/float arithmetic and comparisons too), and `maximum` / `minimum` of float16 with float32/float64 keep the NaN's bits - `ac89847a`.
- `np.linalg.eig` / `eigvals` of a float32 matrix with complex eigenvalues return NumPy's complex64 values (they kept double precision) - `0c02740c`.
- Use-after-free fixed: `np.real` / `np.imag` / `.real` / `.imag`, `view(dtype)` and `getfield` did not keep their owner's buffer alive, so after the owner was disposed (as inside `eigvals`) the next allocation overwrote them; `resize(refcheck: true)` now sees such views - `59890501`.
- `np.convolve` / `np.correlate` without the backend are byte-exact on float32 positions under 32 terms and complex128 positions under 8 (NumPy's scalar `sdot` / `zdotu` regimes), and complex products with inf/NaN follow NumPy (`convolve([1+0j], [inf+0j])` is `nan+nanj`) - `38526155`, `058f4f27`.
- `np.arctan2(complex)` corrupted the heap and crashed the process; it raises NumPy's `TypeError` - `dc2fc075`.
- Complex input to `cbrt`, `floor`, `ceil`, `trunc`, `deg2rad`, `rad2deg`, `floor_divide` and `mod` raises NumPy's `TypeError` (it raised `NotSupportedException`, and a zero-size complex input returned an empty array); `IncorrectTypeException` now derives from `TypeError` - `9a1eeb7e`, `2e205a1f`.
- `np.modf` accepts int, float16 and bool input like NumPy (it rejected them), float16 byte-exact - `3384a0b5`.
- `np.trace` of uint8/uint16/char accumulates in uint64 like NumPy; `ndarray.flat` of a 0-d array has shape `(1,)`; `np.take`, `ndarray.item()` and `len()` of a 0-d array raise NumPy's verbatim texts - `e84fd2f1`.
- `np.argsort` of an empty array with sort axis >= 2 overran the heap - `74656f13`.
- A boolean-mask selection owns its result (`flags.owndata`), like NumPy - `32162b64`.
- `!` (logical not) on a non-contiguous array returned wrong values - `76b2ab9d`.
- Memory leaks fixed in `np.choose`, `tofile`, `tobytes`, `ogrid`, `poly1d`, `polyfit`, `multivariate_normal`, `apply_along_axis` / `apply_over_axes`, `flatiter` slice/fancy access, cross-dtype `SetData` and the float16 `sum` / `mean` temporaries; `NDArray.ReplaceData` could free another array's buffer - `1dd23bbe`, `6f2ec473`, `7e60327d`, `e84fd2f1`.
- `Generator` parity (it shipped in 0.70.0) - `1855ea9e`.
  - `integers` with a `high` past the dtype returned constant data (`integers(0, 257, dtype: np.uint8)` was all zeros); it raises NumPy's error.
  - Concurrent draws on one bit generator tore its state (72-99% of 4-thread values belonged to no seed's stream); draws hold the generator's `lock`, as in NumPy.
  - `choice` without a `size` returns one element (it returned shape `(1,)`); `permuted` follows memory order and casts into `out=` safely; a `-0.0` scale, shape or range is rejected; a `()` size keeps float32.
- Legacy `RandomState` parity - `c1e0feb5`, `f87b4d04`.
  - An unseeded `RandomState()` took its seed from `Environment.TickCount`, so instances made in the same millisecond were identical; it uses OS entropy.
  - `choice` ignored `replace=False` and searched the probability CDF from the wrong side; `randint(dtype: bool)` crashed.
  - Every sampler is NumPy's `legacy-distributions.c` line for line: `gamma(shape<1)`, `binomial`, `negative_binomial`, `poisson`, `f`, `pareto`, `power`, `standard_cauchy`, `multinomial`, `hypergeometric`, `zipf`, `geometric` and `vonmises` used other algorithms.
- `SeedSequence` - a typed `null` seed array seeded the empty sequence (the same stream on every run; now fresh entropy, NumPy's `None`), and `SeedSequence(new[] { -1 })` silently seeded `0xFFFFFFFF` instead of raising - `0e39714b`, `65bdea7b`.
- `result_type` of four or more operands dropped every other pair (`result_type(int8, int8, float64, int8)` was int8; it is float64) - `896f02b2`.
- `np.evaluate` - flat and axis `Sum` / `Mean` / `Prod` / `Min` / `Max` follow NumPy's own schedules bit for bit (they matched the value only), a flat `Mean` of an integer child is NumPy's buffered float64 sum, NEP 50 typing gaps are closed, and two closures over one lambda no longer share a kernel - `53602da9`, `e1400974`, `129fbaba`, `df799381`.
- .NET 8: string slicing (`nd["1:3"]`) re-parsed its pattern on every call because the static `Regex` cache thrashed - 9-18 µs -> 0.6-2.8 µs per parse - `28fa9d79`.

### 🧰 Testing & Tooling
- Differential-fuzz oracle 116K+ -> 406,172 corpus cases: new tiers for every `numpy.polynomial` unit (139,882 cases), `np.ma` (68,860), `np.evaluate` (29,563), `np.emath`, `ndarray` instance members, and complex128 / Decimal coverage, with a dtype-spread gate (at least 4 dtypes per op) and an applicability gate - `57545200`, `67f8f96e`, `7a262afc`, `e84fd2f1`, `55a2a6a7`, `192f456c`, `1dd23bbe`.
- The random-API oracle replays every public `np.random` overload against NumPy 2.4.2 by its exact C# signature (493 members, 5 engines x 10 seeds, the full post-call state), with coverage gates and a nightly soak under 10 fresh seeds - `20a5b60d`, `1c7ac693`, `8f70af7c`, `ecdab2fb`.
- The coverage join credits each API from the oracle contracts that name it, strictly: a new op key or corpus file that resolves to nothing fails the docs build - `fd058f03`.

### 💥 Breaking Changes
- `DType` replaces `System.Type` as the dtype currency - `c916c579`, `68a02258`.
  - `NDArray.dtype` returns `DType` (was `Type`), and `np.float64`, `np.int32` and the rest are `DType` descriptors (were `Type`); `DType` -> `Type` is explicit (`a.dtype.type`), and `a.dtype.Name` is `a.dtype.name`.
  - Each dtype parameter takes one `DType`: the `Type` / `NPTypeCode` twins are gone (a `Type`, `NPTypeCode` or dtype string still converts implicitly), and `linspace` / `nancumsum`'s `typeCode:` is `dtype:`.
  - The promotion family (`result_type`, `promote_types`, `find_common_type`, `common_type`, ...) returns `DType`, and `isdtype` rejects `issubdtype`'s vocabulary (`"floating"`), as NumPy does.
- `finfo.dtype` / `iinfo.dtype` return `DType` (were `NPTypeCode`), and the pythonnet dtype maps take `DType` - `ea080d8e`.
- Casting-rule strings are case-sensitive like NumPy (`"Safe"` raises `ValueError`) - `896f02b2`.
- On Windows, `np.int32` / `np.int64` report the local NumPy's LLP64 type numbers and chars (`7` / `'l'` and `9` / `'q'`; they were hard-coded to Linux's `5` / `'i'` and `7` / `'l'`) - `72e4d29c`.
- A write to a read-only array raises NumPy's `ValueError` (was `NumSharpException`); it derives from `ArgumentException`, so `catch (NumSharpException)` no longer catches it - `f319b657`.
- `np.maximum` / `np.minimum` / `np.fmax` / `np.fmin` take `out` as the third positional argument (NumPy's order), not `dtype`; pass `dtype:` by name - `e5afe84c`.
- `np.logical_and` / `logical_or` / `logical_xor` / `logical_not` return `NDArray` (was `NDArray<bool>`), and `np.modf`'s tuple field `Intergral` is `Integral` - `3384a0b5`.
- `ndarray.shape = ...` raises NumPy's `AttributeError` when the layout cannot be reshaped in place (it silently copied); use `reshape` - `6f2ec473`.
- `np.nanmax` / `np.nanmin` of an empty reduction raise NumPy's "zero-size array to reduction operation fmax which has no identity" (they returned a 0-d NaN) - `de6333dd`.
- `new MT19937(42)` seeds like NumPy's `np.random.MT19937(42)` (through `SeedSequence`), not like `RandomState(42)`; `mt._legacy_seeding(42)` is the old stream. `MT19937`'s non-NumPy members (`NextUInt32`, `NextDouble`, `NextInt`, `NextLong`, `Next`, `NextBytes`, `Seed`, `SeedByArray`, `SetState`, `Key`, `Pos`, `Clone`) are gone, and `Pcg64StateData` is `PCG64.State` - `c1e0feb5`.
- Seeded legacy `RandomState` streams of `gamma(shape<1)`, `binomial`, `negative_binomial`, `poisson`, `f`, `pareto`, `power`, `standard_cauchy`, `multinomial`, `hypergeometric`, `zipf`, `geometric`, `vonmises` (and `gumbel` / `logistic` / `rayleigh` at scale 0) change to NumPy's values; `binomial(int n)` is `binomial(long n)`; inputs NumPy accepts (`dirichlet([])`, a NaN alpha, `negative_binomial(p: 0)`) no longer throw - `f87b4d04`.
- Legacy integers are NumPy's LP64 C `long`: `randint` defaults to int64 (was int32), `random_integers` / `permutation` / `choice` / `multinomial` return int64, `NumPyRandom.Seed` is a `uint` (was `int`), and `SeedSequence.generate_state` returns an `NDArray` (was `Array`) - `eebe85a6`.
- NumPy's parameter names: `Generator(bit_generator)` (was `bitGenerator`), `PCG64(seed)` and the other bit generators (was `seedSeq`, now any `ISeedSequence`), and `default_rng(seed)` (was `bitGenerator` / `generator`) - named-argument calls must be renamed - `eebe85a6`, `1c7ac693`.
- The non-NumPy `NumPyRandom.uniform(NDArray low, NDArray high, DType dtype)` is removed; two-array calls bind NumPy's broadcasting overload (float64; `.astype()` the result for another dtype) - `39d446e0`.

</details>
