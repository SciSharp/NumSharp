# NumSharp 0.70.0

*Deduplicated highlights from the journey3 branch - one line per feature; iteration commits folded in.*

### 📦 New NuGet Packages

Three optional companion packages ship for the first time, co-versioned with **NumSharp** 0.70.0 (the two interop packages depend on **NumSharp.Core**; **NumSharp.Build** is a build-time development dependency that never enters your dependency graph).

- **NumSharp.Interop.OpenBLAS** - new TensorEngine.Blas BLAS+LAPACK backend (NumPy's own dependency): powered by [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), byte-identical to [NumPy](https://numpy.org) 2.4.2; Core stays 100% managed without it but lacks support for most of the functions.
  - Delivery - bundles the exact binaries NumPy 2.4.2 pinned dependency version (the [scipy-openblas64](https://pypi.org/project/scipy-openblas64/) / [scipy-openblas32](https://pypi.org/project/scipy-openblas32/) PyPI packages), per-RID for 8 platforms; enable/disable at runtime. Supports PyPI version pin and build-time download with auto-install at runtime.
  - Products - `dot`, `matmul`, `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `multi_dot`, `matrix_power`.
  - Linear systems & inverses - `solve`, `inv`, `det`, `slogdet`, `tensorsolve`, `tensorinv`.
  - Decompositions - `cholesky`, `qr`, `svd`, `svdvals`.
  - Eigenproblems - `eig`, `eigvals`, `eigh`, `eigvalsh`.
  - Least-squares & SVD-derived - `lstsq`, `pinv`, `matrix_rank`, `cond`, `norm`.
  - Sliding dot - `correlate`, `convolve`.
- **NumSharp.Interop.pythonnet** - zero-copy NumSharp ↔ Python via [Python.NET](https://github.com/pythonnet/pythonnet); any numpy / any Python, no Numpy.NET dependency.
  - Explicit - `arr.ToNumpy()` / `arr.ToPython()` out; `pyObj.AsNDArray()` / `pyObj.FromArrayLike()` in.
  - Implicit - `RegisterCodec()` once, then pythonnet's own `obj.ToPython()` / `pyObj.As<NDArray>()` round-trip transparently.
  - Mode Copy vs view - `Auto` (view when possible, else copy), `View` (share or decline), `Copy` (always independent).
  - Buffer protocol - `FromArrayLike` imports any PEP 3118 exporter, strided/offset/reversed included; read-only stays non-writeable:
    - numpy arrays ([numpy](https://numpy.org));
    - memoryview, bytes, bytearray, array.array, ctypes arrays ([Python stdlib](https://docs.python.org/3/library/index.html));
    - PIL images ([Pillow](https://python-pillow.org));
    - tensors ([PyTorch](https://pytorch.org));
    - plus anything exposing __array_interface__ (e.g. [pandas](https://pandas.pydata.org)) - and plain list/tuple/nested sequences via numpy's asarray.
  - Lifetime & GIL - GC-safe leases, optional GIL control, live export/import counters for leak checks.
  - Dependency - pythonnet 3.0.5+ (Python 3.7-3.13, and future 3.x).
- **NumSharp.Build** - build-time IL weaver for `[NDScoped]` / `[NDScopedAsync]` deterministic memory reclamation: mark a method and the `NDArray` temporaries it drops return to NumSharp's buffer pool the moment it exits, instead of waiting on the finalizer - the source keeps its 100% original body; the scope is woven post-compile into the intermediate assembly.
  - Not a dependency - MSBuild targets + a tool only (no `lib/`, no dependency entries); `dotnet add package NumSharp.Build` writes `PrivateAssets="all"` by itself, so installing it changes your **build**, never your package's dependency graph.
  - Coverage - synchronous methods, `async` methods, iterators, and non-`async` `Task`/`ValueTask` returns are woven through their compiler state machines; incremental per-TFM, idempotent (double-weaving impossible), and a strong-named consumer is re-signed with its own key.
  - Compile-time safety ships with **NumSharp itself**, not this package: the Roslyn analyzer rides the NumSharp nupkg's `analyzers/dotnet/cs/`, so referencing NumSharp alone reports a wrong or unsupported `[NDScoped]` target as a build **error** (NDW002-NDW011, NDW015), nudges on leaked `NDArray` temporaries (NDW012), and holds **types** to the same ownership contract: a class/struct that stores NDArrays (a field or auto-property holding an `NDArray`, an array/tuple/collection/generic of them, a carrier struct, or another NDArray-owning type - ownership is contagious) must be `IDisposable`/`IAsyncDisposable` (NDW016) and must dispose every such member on its `Dispose` path (NDW017); an instance of such a disposable is then an owned value for NDW012 (`new Holder(a + b);` or a never-closed `np.nditer(a)` warns), and `foreach` over a produced `NDArray` or such an instance is flagged (C# disposes only the enumerator). The runtime-inert `[NDBorrowed]` attribute (field / property / class / struct) states "this references an array owned elsewhere" and opts out. NumSharp also carries the NDW013 build warning for `[NDScoped]` used **without** the weaver installed (the attributes are then inert).
  - Escape hatches - `-p:SkipNDScopeWeave=true` builds without weaving (nothing else changes); `-p:NDScopeWeaveILVerify=true` additionally runs `dotnet-ilverify` on the woven output.
  - Gate - `tools/verify_build_package.sh`, an 18-step real-consumer nupkg flow (package shapes, weave + incrementality, transitive isolation, re-signing, state machines, the analyzer/weaver error layers, NDW012/NDW013, NDW016/NDW017 + `[NDBorrowed]`, and analyzer-via-NumSharp-alone).

### 📊 Dashboards & Docs

Three living dashboards ship on the [documentation site](https://scisharp.github.io/NumSharp/), each generated from the same CI artifacts the release gates run on.

- **[Supported Features Dashboard](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html)** - NumPy 2.x API coverage & support: every public NumPy API in scope, its NumSharp equivalent, known limitations and C# overloads, and the coverage-score math. Headline ~85% (478/560), with np.random / np.fft / np.linalg at 100%.
  - Surface scoreboard (top-level · ndarray · random · linalg · fft), a deterministic capability map, and a searchable API explorer.
- **[Benchmark Dashboard](https://scisharp.github.io/NumSharp/docs/benchmarks-dashboard.html)** - the NumSharp-vs-NumPy performance lab: 18 op suites × all dtypes × three cache tiers (1K/100K/10M), plus six scans (iterator, layout, operand, cast, fusion, native OpenBLAS/LAPACK); 456/456 benchmarkable APIs have evidence.
  - NumPy÷NumSharp heatmaps with drill-down, published as release-tracked history snapshots (not scratch output).
- **[Tests & Oracle Dashboard](https://scisharp.github.io/NumSharp/docs/tests-oracle-dashboard.html)** - the correctness/verification lab: reflected MSTest inventory (net8.0 + net10.0), the committed NumPy 2.4.2 differential-fuzz corpus (116K+ cases, bit-exact, no Python in CI), independent Decimal evidence, format/index oracles, known-bug gates, and live interop suites.

### ✨ New APIs & Modules
- `np.random.default_rng` - the full modern PCG64 `Generator`, byte-identical streams to NumPy 2.4.2 - `e868d8ae` (+ `754b7476`, `febfbbdd`, `f491c499`).
  - `default_rng` - entry point (seed / `SeedSequence` / `BitGenerator` / `PCG64` overloads).
  - `random`, `integers`, `standard_normal`, `normal`, `exponential`, `uniform`, `standard_gamma`, `gamma`, `choice`, `shuffle`, `permutation`, `permuted` - the Generator draw surface.
  - `random_integers`, `bytes` - the legacy RandomState helpers.
- `np.fft.*` - the whole 18-function Fourier module, a pure-managed pocketfft port, bit-exact incl. float32/float16 values - `3b9d5cfb`, `a525e355`, `4cb91898`.
  - `fft`, `ifft`, `fft2`, `ifft2`, `fftn`, `ifftn` - complex forward/inverse (1-D/2-D/N-D).
  - `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn` - real-input transforms.
  - `hfft`, `ihfft` - Hermitian-symmetric transforms.
  - `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift` - sample-frequency & shift helpers.
- `np.einsum` - Einstein summation, now computing and planning - `7d2d7a2f` (+ `d78e07db`, `b61b0998`), `bb63ba48`.
  - `einsum` - contracts via the matrix products (rides OpenBLAS when the package is referenced).
  - `einsum_path` - greedy/optimal contraction planner, byte-exact info string.
- `np.r_` / `np.c_` / `np.ix_` / `np.s_` / `np.index_exp` - the grid & slice-expression DSL, 131/131 bit-exact vs NumPy 2.4.2 - `00dfe402` (+ `3c63734d`, `7eea4f7f`, `c4e27523`).
- `np.ogrid` / `np.mgrid` / `np.meshgrid` - open-mesh / dense-mesh / coordinate-matrix grid constructors, differential bit-exact vs NumPy 2.4.2 - `19feaed2`, `7f558d05`, `4e8c3925`.
- Iteration objects - NumPy 2.4.2 parity over the `NDIterRef` engine (37 cases probed side-by-side, all identical) - `8bd882b3`, `7112cbe4`.
  - `np.nditer`, `np.ndindex`, `np.ndenumerate` - the boxed iterators, full flag/error parity.
  - `np.nested_iters`, `ndarray.flatiter` - nested-loop iterators + a write-through flat iterator.
- The issue #623 sorting/searching six - NumPy 2.4.2 parity, ~2,170 fuzz cases bit-exact - `b3505398` (+ `8cad3025`).
  - `partition`, `argpartition` - kth-element partial sort (value + index).
  - `lexsort` - indirect stable multi-key sort; `sort_complex` - real-then-imag complex sort.
  - `nanargmax`, `nanargmin` - NaN-aware argmax/argmin.
- `np.take_along_axis` - the per-slice gather (the `argsort`/`argmax` inverse), NumPy 2.4.2 parity, 24,000+ fuzz cases bit-exact, ≥1.5× faster on every measured variation - `f351600a` (+ `88550d13`, `7091a3c9`).
- `np.select` - pick each element from the first choice whose condition is true, NumPy 2.4.2 parity (fused single-pass kernel on the contiguous path) - `fc10404d` (+ `42d96a14`).
- `np.isin` + `intersect1d` / `union1d` / `setxor1d` / `setdiff1d` - element-wise membership + sorted set algebra, NumPy 2.4.2 parity (1.9-13.5× faster) - `bfe952d5` (+ `27632ed5`).
- Array-API unique family - `unique_values`, `unique_counts`, `unique_inverse`, `unique_all`, 102/102 bit-exact vs NumPy 2.4.2 across 13 dtypes - `bec1c497`.
- The `np.diag` family + triangular ops - 13 functions, 165/165 side-by-side parity with NumPy 2.4.2 - `27b9b012` (+ `a7782984`).
  - `diag`, `diagflat`, `fill_diagonal` - diagonal build & in-place fill.
  - `tri`, `tril`, `triu` - triangular masks & extraction.
  - `diag_indices`, `diag_indices_from`, `tril_indices`, `tril_indices_from`, `triu_indices`, `triu_indices_from`, `mask_indices` - index generators.
- Linear-algebra product family - new managed `np.*` products; byte-parity via the OpenBLAS backend when referenced - `53d7764f` (+ `81509766`, `297f883f`, `74aa5d5a`).
  - `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `multi_dot`, `matrix_power`.
- Polynomial family - NumPy 2.4.2 parity, pure functions bit-exact - `956f3392` (+ `2628c921`, `d6a50593`).
  - `poly`, `roots`, `polyfit`, `polyval` - construction / fitting / evaluation.
  - `polyadd`, `polysub`, `polymul`, `polydiv`, `polyder`, `polyint` - arithmetic & calculus.
  - `poly1d` - the polynomial object; `vander` - Vandermonde matrix.
- Text I/O - byte-exact with NumPy, `savetxt`→`loadtxt` round-trips - `a1920a4a`, `17a1ff8a` (+ `80a0ed50`, `d39ff824`).
  - `np.savetxt`, `np.loadtxt`, `np.fromstring`.
- Inverse-hyperbolic trig - byte-exact vs NumPy 2.4.2 - `9fa48041` (+ `615f1ee5`).
  - `arcsinh`, `arccosh`, `arctanh` - primary ufuncs; `asinh`, `acosh`, `atanh` - Array-API aliases.
- Array-API `device` conformance (CPU shim) - `ebba2cbf`.
  - `ndarray.device`, `ndarray.to_device`, and `device=` on `array` / `zeros` / `ones` / `empty` / `arange` / ….
- Additional array, linalg & stats functions -
  - `np.kron`, `np.cross` - Kronecker & cross products - `7bcad845`, `73019dce`.
  - `np.cov`, `np.corrcoef` - covariance & Pearson correlation - `92dc537b`, `aaf731b2`.
  - `np.choose` - index-into-choices gather - `aaa41ef2`.
  - `np.nancumsum`, `np.nancumprod` - NaN-aware cumulative scans - `0370c0aa`.
  - `np.digitize`, `np.bincount` - bin-index + integer histogram, bit-exact vs NumPy 2.4.2 - `f2cefba2`, `12f484c3`.
  - `np.correlate` - sliding cross-correlation (managed SIMD; OpenBLAS byte-parity below) - `12f484c3`.
  - `np.bmat` - block-matrix assembly - `6ba24752` (+ `d5621d57`).
  - `np.real`, `np.imag`, `np.angle`, `np.conjugate` / `np.conj` - complex component / phase accessors (post-FFT spectrum extractors) - `8b0ac701`, `d0081b6d`.
  - `np.iterable` - NumPy's pure iterability predicate - `ce560796` (+ `8cf54d35`).
  - `np.isfortran` - F-contiguity predicate (`a.flags.fnc`) - `30453696`.
- The `np.linalg` factorisation surface and complex128 `dot`/`matmul` are listed under **New NuGet Packages** above (they compute via the OpenBLAS backend) - `dc448acc`, `f5ec6276`, `d09e4376`, `6ee562da`.

### 🧩 ndarray surface
- `ndarray` member parity with NumPy 2.4.2 -
  - `data` - the memoryview buffer object (`np.MemoryView`); accepted zero-copy by `array` / `asarray` / `frombuffer` / … - `25ae7053` (+ `4072577d`, `bc544403`).
  - `byteswap` - width-dispatched endian byte-swap - `67994cbc`.
  - `getfield`, `setfield` - byte-field views - `4b07b71d`.
  - `real`, `imag`, `conj`, `conjugate` - complex accessors - `7765ce50`.
  - `itemsize`, `nbytes`, `fill`, `flags` - metadata members - `792a9f14` (+ `aee7cbab`, `27a19ae4`); `setflags` - write/align control - `275f089c`.
  - +14 instance methods (`all`, `any`, `clip`, `take`, `repeat`, `squeeze`, `trace`, …) - `06869352`.

### ⚡ Performance
*Ratios are NumPy ÷ NumSharp - higher is better (`x2` = twice NumPy's speed); `xLOW->xHIGH` spans the worst→best measured cell across sizes and dtypes.*
- `x0.98->x74` - `np.unique` family routed through the radix sort core - `5df10897` (+ `35d12699`).
- `x1.6->x5.2` - `percentile` / `median` / `quantile` pivot-stack block-partition quickselect - `8a1376ff`.
- `x0.4->x2.75` - `np.argpartition` on the same block/pivot-stack path - `75a1d873`.
- `x1.35->x11` - `np.isin` hash-set membership replaces sort+searchsorted - `bd96d541`.
- `x15.6` - blocked GEBP double GEMM for transposed-B `dot` (2.9→43 GFLOP/s) - `97e9e82a` (+ `7d680eb1`).
- `x40->x249` - typed `np.nditer<T>` / `nditer_chunks<T>`, allocation-free iteration (chunks + `Vector<T>` hits 249×) - `d58f3728`.
- `x1.0->x4.5` - `take` / `put` / `place` element-copy specialization + gather prefetch (`take` went from x0.68 losing to winning everywhere) - `88550d13`.
- `x1.04->x11.7` - float32 `exp` / `log` / `sin` / `cos` / `tanh` + `rad2deg` / `deg2rad` reimplemented as bit-exact NumPy kernel ports (`tanh` also replaces the float64 loop) - `ecdb4581`, `6bab5754`, `f5f21ff3`.

### 🎯 Parity & Fixes
- `ndarray.flags` / `setflags` - full NumPy 2.4.2 parity across the whole layout/producer space, hardened by a 1104-case differential oracle (owndata/writeable/contiguity, squeeze-as-view, split-child contiguity, read-only reduction scalars) - `275f089c`, `53b5d82e`, `ca1b0fac`.
- `searchsorted` - complex lexicographic order + `result_type` key promotion (no more silent key down-cast) + NaN-as-largest total order - `93abe13d`, `cc676ea8`, `f2cefba2`.
- `np.take` / `np.put` index validation matches NumPy - a negative index under `mode='raise'` normalizes once (`np.take(a, [-1])` addresses the last element instead of throwing), and a non-castable float/complex index raises the verbatim `TypeError` instead of silently truncating - `fc10404d`, `88550d13`.
- `np.correlate` / `np.convolve` - OpenBLAS byte-parity via the new sliding-dot seam - `d0be3132`.
- Broadcast write semantics - `broadcast_to` is read-only, `broadcast_arrays` is writeable, and writing a non-writeable view now raises NumPy's verbatim message instead of silently corrupting the shared source - `1eadb83b`, `6fb518c0`, `1cc67d47` (+ `baf41c89`).
- Allocation & reshape guards - `size×itemsize` overflow, `reshape(-1, …)`, and `expand_dims` axis now raise NumPy's verbatim texts instead of silent wrong-size allocations or raw .NET exceptions - `c2552d6a`.
- Empty / zero-sized array and float16-matmul edge cases now match NumPy - float16 products accumulate in float32 (no more `ones(3000)@ones(3000)`=2048 saturation), stacked/fancy indexing into zero-sized arrays, and the 0-d boolean setter - `03d0f0c8`, `7636100a`, `f6e258c0`.
- Fancy-set into a non-contiguous destination no longer silently corrupts the view (`SetIndicesNDNonLinear`), bit-exact vs NumPy 2.4.2 across all 15 dtypes - `ff68bf14`.
- `astype(copy: false)` never mutates the caller's array on a dtype conversion, matching NumPy - `e5274cdc`.
- `ndarray.view(dtype)` of a different-itemsize dtype now follows NumPy 2.x's last-axis-contiguous rule, so `arr[::2].view(int32)` works instead of throwing - `970ee7f1`.
- `np.matmul` gains the full ufunc keyword surface (`out=`/`axes=`/`axis=`/`keepdims=`/`dtype=`/`casting=`/`order=`), and `np.dot`/`np.outer` gain `out=` - `73019dce` (+ `87ff5797`).
- Three engine `argmax` / `argmin` bugs the sort audit exposed - the Decimal and Char flat paths and a NaN-tie ordering - now match NumPy 2.4.2 - `8cad3025`.
- `np.unique` full-parameter parity with NumPy 2.4.2 - the axis path's slab equality is corrected so each NaN sub-array is distinct and signed-zero sub-arrays collapse (a real unique-row-count bug for floats/complex), `sorted=` / `equal_nan=` are accepted, an out-of-range axis raises the verbatim `AxisError`, the bare-return overloads (`np.unique(ar, axis: 0)`) and `intersect1d(return_indices:)` now port verbatim, and `UniqueResult` fields are case-identical to NumPy - `9f573dd5`, `262eefd7`, `0151a832`.
- `np.linalg` factorisations without a backend now raise a typed `OpenBlasMissingBackendException` - derives from `NotSupportedException` so existing catches still work, and names the `NumSharp.Interop.OpenBLAS` package to install (was a bare `NotSupportedException`) - `d1347c36`.
- Six creation/math/linalg/stats functions brought to NumPy 2.4.2 parity - `b2a8374b`:
  - `np.ascontiguousarray` / `np.asfortranarray` - a 0-D input returns a length-1 view (shares storage), matching NumPy's ndim≥1 contract.
  - `np.eye` / `np.ones` - `Char` fills numeric one U+0001, not the character `'1'`.
  - `np.full_like` - preserves the source array's dtype; `fill_value`'s CLR type no longer selects the result dtype.
  - `np.linspace` - floors inexact values before an integer-dtype cast and pins the endpoint to `stop` exactly.
  - `np.einsum` - a scalar (`ndim==0`) contraction keeps its `()` shape instead of promoting to `(1,)`.
  - `np.angle(deg: true)` - a 0-D `Half` / `Single` result keeps its float tier instead of promoting to `Double`.
- Removed the last 64-dimension caps - axis reductions (`var`/`std`/`cumsum`/`cumprod`/`all`/`any`) and `ndarray.fill` now run at unlimited ndim like the rest of NumSharp - `8f34e8ff`, `7fa96750`.

### 🧰 Testing & Tooling
- The NumPy differential-fuzz oracle gained new tiers - FFT transforms, ufunc `out=`/`where=` (3,727 cases over out × mask layouts), result-kinds + verbatim-error + iterator-trace, IEEE special-values (nan/±inf/±0/subnormal), and a truthful-vs-precise precision channel - `bc91dd25`, `6cd1de9b`, `0882edbb`, `359e9d3c`, `76f0c918`.
- np.random byte-parity + CBLAS product-value + axis-precision oracle tiers - the seeded-stream gate **surfaced 8 `np.random` sampler byte-parity divergences** (`f`/`pareto`/`standard_cauchy`/`binomial`/`negative_binomial`/`multinomial`/`multivariate_normal`/`gamma(shape<1)`), now pinned as known `[OpenBugs]` issues (not yet fixed) - `31a178f2`.
- First host-pinned differential-fuzz coverage for the OpenBLAS-backed LAPACK factorisations (eigen/SVD/QR/Cholesky + the LU family, 366 cases byte-exact) plus the polynomial / einsum / cross / cov families - `cf559a1a`, `03415ec9`, `5ff54a72`.

### 💥 Breaking Changes
- `np.random.bytes` / `Generator.bytes` now return `NDArray<byte>` instead of `byte[]`, so draws **>2 GiB** succeed (NumPy `npy_intp` parity) - `44d2e7d9`.
- `ndarray.strides` now reports **bytes** per axis (was elements), matching NumPy's `PyArray_STRIDES` - `6ef30215`.
- `np.unique(ar)` now returns a `UniqueResult` struct instead of a bare `NDArray`, so `np.unique(ar)[k]` selects the k-th **output** (use `.values[k]` for the k-th value); it converts implicitly to `NDArray` / `NDArray[]` so most call-sites are unchanged - `17f571ef`.
- `np.mgrid` / `np.meshgrid` drop their legacy non-NumPy signatures: `mgrid[...]` is now an indexer (was a 2-arg method) and `meshgrid` is variadic returning `MeshgridResult` (was a fixed 2-tuple + `Kwargs`) - `7f558d05`, `4e8c3925`.
- Every NumSharp assembly is now strong-named - `PublicKeyToken` changes from `null` to `cc7b13ffcd2ddd51` (published NumSharp had shipped unsigned since 2019); every consumer (TensorFlow.NET, Pandas.NET, Gym.NET) must recompile - `478d550d`.
- Environment variables were hard-renamed to a consistent `NUMSHARP_<AREA>_<SETTING>` scheme with no back-compat aliases - `NUMSHARP_GUARD_PAGES` (shipped in 0.60.0) becomes `NUMSHARP_DEBUG_GUARD_PAGES`, and the OpenBLAS/pythonnet knobs take `_LIBRARY` / `_SEARCH_PATH` / `_USE_BUNDLED` / `_PYPI_FEED_URL` / `_REQUIRE_ENGINE` names - `079d1859`.
- `NDArray.Normalize()` (a non-NumPy extension) is marked `[Obsolete]` in favour of `np.clip()` - `b701843e`.
- `np.poly1d` is now `IDisposable` - it owns the coefficient array its constructor yields into it, so `Dispose()` releases it, and the copy-constructor `new poly1d(p)` now **copies** the coefficients instead of sharing one array between two owners (NumPy shares by refcount; two NumSharp owners would double-dispose). Found by the new NDW016 ownership analyzer, which also made `np.Broadcast.Dispose()` (what `foreach` calls when a loop ends) release the `broadcast_to` views it had built (they are rebuilt lazily on the next `iters`/enumeration, so the object stays re-enumerable) and `IndexCollector` an `IDisposable` that no longer strands its outgrown buffer.
