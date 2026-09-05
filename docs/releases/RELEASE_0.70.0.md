# NumSharp 0.70.0

It took a while but we are at 85% NumPy API coverage.
This is a huge milestone for the .NET ecosystem as NumSharp grows to matureness.
This branch also delivers integration with NumPy's OpenBLAS backend and full integration with Python giving a new angle of use cases for NumSharp to a point NumSharp is an unmanaged memory and math interop with the python ecosystem. I believe in integration rather than competition thus the large scale of support from PyTorch to Pillow.
OpenBLAS is a rapidly developed ecosystem NumSharp will eventually replace with a simpler version but that requires porting of 100k-300k lines of code to achieve complete mathematical parity. OpenBLAS roughly powers 30% of NumPy's backend.

### 🧭 Overview
- NumPy 2.x API coverage **60.4% -> 87.1%** - **150 NumPy APIs newly available**, every figure from the [Supported Features Dashboard](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html).
  - `np.*` - **+88 functions**
  - `np.fft.*` - **fully ported +18 functions**
  - `np.linalg.*` - **fully ported +31 functions**
  - `ndarray.*` - **+16 functions**
  - `np.random.*` - **fully ported**: the modern `default_rng` / PCG64 `Generator`, byte-identical streams.
- **~40 NumPy-parity fixes**, **25 measured speedups** and **10 breaking changes** - every line traced to one of 186 cited commits out of the branch's 540.
- **3 new NuGet packages**, all published signed:
  - **NumSharp.Interop.OpenBLAS** - the full BLAS/LAPACK `np.linalg` surface, byte-identical to NumPy 2.4.2 (bundles NumPy's own pinned OpenBLAS for 8 RIDs).
  - **NumSharp.Interop.pythonnet** - zero-copy interop with numpy, PyTorch, pandas, Pillow and any PEP 3118 buffer exporter via Python.NET.
  - **NumSharp.Build** - the `[NDScoped]` IL weaver for deterministic memory reclamation; its Roslyn leak/ownership analyzer ships inside NumSharp.
- **3 living dashboards** on the docs site - Supported Features, Benchmarks (456 benchmarkable APIs) and Tests & Oracle (a 116K+ test-case NumPy 2.4.2 differential-fuzz corpus, bit-exact).
- **Deterministic memory reclamation** - NDArray is now IDisposable, `NDScope` and `[NDScoped]` for automated memory disposal and buffer caching reclamation.
- `NDArray` object base size reduction `1088 B -> 192 B` (896 B smaller).
- 116K Oracle unit tests and 14,000 unit tests + integration tests.
- **Performance vs a warm NumPy** - NDIter **1.46x geomean**.

## Detailed Breakdown
<details>
  <summary>Show All</summary>

### 📦 New NuGet Packages

Three optional companion packages ship for the first time, co-versioned with **NumSharp** 0.70.0 (the two interop packages depend on **NumSharp.Core**; **NumSharp.Build** is a build-time development dependency that never enters your dependency graph).
All packages are now published as signed NuGet packages.

- **NumSharp.Interop.OpenBLAS** - new TensorEngine.Blas BLAS+LAPACK backend (NumPy's own dependency): powered by [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), byte-identical to [NumPy](https://numpy.org) 2.4.2; Core stays 100% managed without it but lacks support for most of the functions.
  - Delivery - bundles the exact binaries NumPy 2.4.2 pinned dependency version (the [scipy-openblas64](https://pypi.org/project/scipy-openblas64/) / [scipy-openblas32](https://pypi.org/project/scipy-openblas32/) PyPI packages), per-RID for 8 platforms; enable/disable at runtime. Supports PyPI version pin and build-time download with auto-install at runtime. The Linux/macOS wheels' vendored Fortran runtime (`libgfortran`/`libquadmath`/`libgcc_s`) is co-staged, and on macOS materialized at load (in place, or in a per-user cache for read-only / single-file layouts) so a plain PackageReference restore loads on every platform.
  - Products - `dot`, `matmul`, `inner`, `vdot`, `vecdot`, `matvec`, `vecmat`, `tensordot`, `multi_dot`, `matrix_power`.
  - Linear systems & inverses - `solve`, `inv`, `det`, `slogdet`, `tensorsolve`, `tensorinv`.
  - Decompositions - `cholesky`, `qr`, `svd`, `svdvals`.
  - Eigenproblems - `eig`, `eigvals`, `eigh`, `eigvalsh`.
  - Least-squares & SVD-derived - `lstsq`, `pinv`, `matrix_rank`, `cond`, `norm`.
  - Sliding dot - `correlate`, `convolve`.
- **NumSharp.Interop.pythonnet** - zero-copy NumSharp ↔ Python via [Python.NET](https://github.com/pythonnet/pythonnet); any numpy / any Python, no Numpy.NET dependency.
  - Explicit - `arr.ToNumpy()` / `arr.ToPython()` out; `pyObj.AsNDArray()` / `pyObj.FromArrayLike()` in.
  - Implicit - `RegisterCodec()` once, then pythonnet's own `obj.ToPython()` / `pyObj.As<NDArray>()` round-trip transparently.
  - Tuples - C# `ValueTuple`/`Tuple` cross as Python tuples (any arity, nested; an `NDArray` element becomes a numpy view) and a Python tuple / namedtuple / `torch.Size` decodes back into a same-arity C# tuple, so `(long, long) shape = py.a.shape` just works.
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

### 📊 Dashboards & Docs

Three living dashboards ship on the [documentation site](https://scisharp.github.io/NumSharp/), each generated from the same CI artifacts the release gates run on.

- **[Supported Features Dashboard](https://scisharp.github.io/NumSharp/docs/coverage-support-dashboard.html)** - NumPy 2.x API coverage & support: every public NumPy API in scope, its NumSharp equivalent, known limitations and C# overloads, and the coverage-score math. Headline ~87% (488/560), with np.random / np.fft / np.linalg at 100%.
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
- `np.einsum` - Einstein summation, now computing and planning - `7d2d7a2f` (+ `d78e07db`, `b61b0998`), `bb63ba48` (+ `5a55d065`).
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
  - `np.logaddexp`, `np.logaddexp2`, `np.nextafter`, `np.copysign` - IEEE binary ufuncs (full `out=`/`where=`/`dtype=` surface); `nextafter`/`copysign` bit-exact, `logaddexp` ≤2 ULP vs NumPy 2.4.2 - `043370e0`, `26d014ed`.
  - `np.interp` - 1-D linear interpolation (incl. `period` + complex `fp`), bit-exact vs NumPy 2.4.2 - `043370e0`.
  - `np.nan_to_num`, `np.isposinf`, `np.isneginf` - NaN/±inf replacement + signed-infinity predicates, byte-identical to NumPy 2.4.2 - `480c2786`.
  - `np.getbufsize`, `np.setbufsize` - thread-local ufunc buffer size with NumPy's verbatim validation, byte-exact - `6d471cf3`.
- The `np.linalg` factorisation surface and complex128 `dot`/`matmul` are listed under **New NuGet Packages** above (they compute via the OpenBLAS backend) - `dc448acc`, `f5ec6276`, `d09e4376`, `6ee562da`.
- `NDScope` - deterministic buffer reclamation: `using (var s = NDScope.Open())` returns the `NDArray` temporaries built inside the scope to the pool at exit (via `s.Returns(result)`) instead of waiting on the finalizer; Core weaves ~265 `np.*` methods with `[NDScoped]` so their transients are reclaimed eagerly, and the **NumSharp.Build** weaver applies the same to your own methods - `1b4e776b` (+ `99583e25`, `726ec48b`).
- `DType` / `np.dtype` - a unified dtype descriptor (NumPy's `numpy.dtype` analog) that folds the three historical spellings - `System.Type`, `NPTypeCode`, and a NumPy dtype string (`"float32"` / `"f4"` / `"<f8"`, case-sensitive) - behind implicit conversions, plus 15 static spellings (`DType.Int32`, `DType.Single`, …); the ufunc, reduction and logic overloads migrated to it first (Creation/IO to follow) - `25a35f45` (+ `c4ebbfb4`, `8b13234e`).
- `TensorEngine.Threading` - one process-wide registry for every threading knob NumSharp and the native BLAS/OpenMP ecosystem expose (`NUMSHARP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `OMP`/`MKL`/`BLIS`/`NumExpr`/`vecLib`): `Register` / `Get` / `SetThreads` / `SetAll`; a variable already set in the environment is the source of truth a module default never overrides, every write stays process-scoped, and `np.multithreading` now routes through it (the OpenBLAS package plugs in a live reader/applier) - `659ed82a`.

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
- `x1.8->x60` - the whole float16 family on bit-level AVX2 / widen-compute-narrow kernels: `min`/`max`/`ptp`, `maximum`/`minimum`/`fmax`/`fmin`, the six comparisons, `nanmin`/`nanmax`, `clip`, `add`/`subtract`/`multiply`/`divide`, `floor`/`ceil`/`trunc`/`rint` - `a05b5b1e`, `f3405659`, `3a46cfec`, `c9e141c5`, `498a68f0`, `c8b0573d`, `c8babf28`.
- `x2.0->x3.4` - the bool dtype family on byte-lane SIMD (bitwise/logical/comparisons; was x0.38-0.62); `sum(bool)` is a popcount (x23.6) and `argmax(bool)` a find-first scan (up to ~60,000x on sparse input) - `46dbb9c6`, `d967d99b` (+ `bd655743`).
- `x0.9->x1.9` - SIMD `isnan` / `isinf` / `isfinite` for float32/float64 (was ~x0.10; 1K stays at the small-N alloc floor) - `7bd4f380` (+ `4dfe7619`).
- `x1.0->x1.65` - float32/float64 `argmax` / `argmin` single-pass SIMD tournament (was ~x0.15) - `48f894ea`.
- `x1.82->x3.12` - float32 `exp2` SIMD kernel (hybrid double-`2^r` + float scale; was ~x0.19) - `87a8bb8f`.
- `x1.4->x6.5` - NDIter 2-D block kernel for narrow strided rows + NumPy-style axis coalescing (narrow rows were x0.4-0.82; a contiguous `(250000,4)` array times a scalar dropped 1.7 ms->251 µs) - `af25a746`, `ccadeef4`.
- `x1.1->x10.5` - fancy indexing (`a[idx]`, `a[idx]=v`, `m[ridx]`) routed to the take/put kernels and `where=` masked ops scanned with SIMD (`a[idx64]` was x0.22, `a[idx32]` x0.81; masked all-false 27.7->1.5 µs) - `09d1fc59`.
- `x1.06->x9.3` - NDIter fixed-cost cut (recycled state block, packed kernel key, direct external-loop advance, SIMD comparison `out=`, eager overlap-temp dispose): construction geomean x2.73->x5.8, every `out=` ufunc now beats NumPy at n=1 - `15154b00`.
- `x2.88->x3.00` - `cov` / `corrcoef` via a managed symmetric-Gram (syrk) path at ≤16 variables (100K; was ~x0.56) - `d64f3df5`.
- `x0.7->x1.85` - managed `matvec` / `vecmat` / matrix-vector `dot` gemv/gevm paths, no backend (was ~x0.1-0.2) - `4f666fc8`.
- `x6` - `diff` / `ediff1d` fused adjacent-difference stencil (1K; ~4x fewer allocations at 100K) - `6e94dbcc`.
- `x1.46->x4.8` - `fill_diagonal` / `diag` / `diagflat` diagonal-write IL kernel (`fill_diagonal` 10M was x0.40) - `a13238ac`.
- `x1.4->x1.7` - streamed int64/uint64 `mean` axis reductions + unrolled flat `nanmin` / `nanmax` (were x0.16 / x0.26) - `8c09dc15`.
- `x2.0->x2.3` - pre-state `cpblk` fast path for trivial same-layout `copyto` / `copy` / `clone` at small N - `7bfc27a2`.
- `x1.8->x2.3` - small same-dtype single-broadcast ops routed to the direct SimdChunk kernel (1K; was ~x0.55) - `a3819869`.
- `x0.6->x1.35` - buffer-pool GC pacing + burst-sized buckets lift the small-N elementwise floor for undisposed results (1K float32 `abs` was x0.29) - `160ecbba`.
- `1088 B -> 192 B` per-`NDArray` object base size (896 B smaller) - `UnmanagedStorage`'s 15 per-dtype slice fields collapsed into one `StructLayout.Explicit` union - `8306fa63`.

### 🎯 Parity & Fixes
- Release-candidate consumer QA (the packed 0.70.0 nupkgs consumed from real net8.0/net9.0/net10.0 projects against NumPy 2.4.2 output) - seven fixes - `3893b41a`:
  - `np.frombuffer(bytes, "float64")` - NumPy dtype **names** (`"float64"`, `"int32"`, `"bool"`, `"complex128"`, `"float16"`) now parse like the sized codes (`"<f8"`); they threw `NotSupportedException`.
  - `np.clip(float32, 1, 3)` / `ndarray.clip` and the arctan2-template ufuncs (`arctan2`, `copysign`, `logaddexp`, `logaddexp2`, `nextafter`) keep the array's float dtype for a weak C# int literal (NEP50), instead of promoting float32/float16 to float64.
  - `np.dtype("f4").name` / `ToString()` render NumPy's name (`float32`), not the CLR type name (`Single`).
  - `np.cov` / `np.corrcoef` take the BLAS product whenever the OpenBLAS backend is installed (the managed ≤16-variable Gram fast path is now no-backend only), so they are byte-identical to NumPy with the package as documented.
  - `NumSharp.Interop.OpenBLAS` discovery probes the runtime's native search directories, so a single-file publish with `IncludeNativeLibrariesForSelfExtract=true` still finds the bundled binary (the backend silently vanished there).
  - `poly1d.Call(x)` - the evaluation member NumPy spells `p(x)`.
  - Package layout: `build/NumSharp.targets` and the OpenBLAS `buildTransitive/*` files ship under a `net8.0/` TFM folder, so a `netstandard2.0` / `net6.0` / `net7.0` consumer gets NuGet's `NU1202` at restore instead of a "compatible" restore with no `lib/` and a bare `CS0246` at compile.
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
- The LU-based `np.linalg` factorisations - `det`, `slogdet`, `solve`, `inv` (+ `tensorinv`/`tensorsolve`/`matrix_power(n<0)`) - now compute in a backend-free Core via a managed LU (`allclose` to NumPy, faster for small matrices) instead of raising; an installed OpenBLAS backend still wins the seam for byte-parity - `48b00e00` (+ `03884ee9`).
- `np.isclose` / `np.allclose` now compute in NumPy's exact `result_type` - fixes a complex128 correctness bug (the imaginary part was dropped, so `isclose([1+0j],[1+100j])` wrongly returned `True`) and evaluates float32 pairs in float32 (100K x0.18->x3.3) - `fbbda5f0`.
- `np.sum(float16)` accumulates in a float32 shadow and narrows per orientation like NumPy's `HALF_add` - an axis sum now saturates (`sum(ones((4096,3),f16),axis=0)` = `[2048,2048,2048]`, was `[4096,...]`, a ~3.5% error) while a flat sum still reaches 4096 - `32732a0f`.
- `np.power(x, negative_int)` - a strided/2-D/broadcast integer exponent no longer reads out of bounds (a Release memory-safety bug), and a bool base now raises NumPy's verbatim `ValueError` instead of silently computing - `02e6929f`.
- Seven array-creation/ctor fixes match NumPy 2.4.2 - `new NDArray(buffer, shape, 'F')` lays out column-major, `np.arange(dtype=bool)` raises past length 2, `np.frombuffer` rejects complex64/`'c8'`, and `np.array`'s default `ndmin` is 0 - `5c7e3ad8`.
- `np.isreal` / `np.iscomplex` now inspect the imaginary part (they returned all-`True` / all-`False` for complex regardless of value) and no longer emit garbage bytes on a strided real input - `fa491573`.
- Axis reductions (`sum`/`mean`/`prod`/`min`/`max`/`std`/`var` + all `nan*`) preserve an F-contiguous input's layout (KEEPORDER allocation) instead of flipping it to C, matching NumPy - `0ae977d9` (issue #610).
- `np.clip` on a non-contiguous Boolean array (strided/transposed/F-order/reversed) now clips instead of throwing `NotSupportedException` - `f6f5b657`.
- Strided/broadcast/negative-stride float16 `add` / `subtract` / `multiply` / `divide` (and `kron`) no longer read the wrong elements - a stride-coalescer bug (merged adjacent axes by value, not magnitude) plus a bit-exact odometer kernel - `7f2c09a3`.
- The float16 `maximum`/`minimum`/`nanmin`/`nanmax`/`clip` now return NaN operands verbatim (payload + sign preserved, was canonical `Half.NaN`), and a `clip` NaN-max-bound precedence bug that returned float32 fills (any dtype) is fixed - `f3405659`, `498a68f0`, `c9e141c5`.
- complex128 unary ufuncs are now byte-identical to NumPy 2.4.2 on the **NaN sign** - `sqrt`/`log`/`exp`/`expm1`/`square`/`reciprocal`/`sin`/`cos`/`tan`/`sinh`/`cosh`/`tanh`/the `arc*` family/`abs`/`sign` emit NumPy's positive quiet NaN where it canonicalizes and propagate the operand's NaN sign where it does (was .NET's negative NaN); `square` is additionally made arch-consistent via a portable FMA off x86 (fixing a thousands-of-ULP Apple-silicon divergence), all gated by a new `nan` differential-fuzz tier - `760bb5bb` (+ `298e7c60`, `8e1f3cd9`, `2fd9d785`, `56a55712`).
- The sort/select core is 64-bit: `sort`, `argsort`, `partition`, `argpartition` (and the flat `cumsum`/`cumprod` output shapes) now handle arrays above `int.MaxValue` elements - `np.sort` silently returned **unsorted** data and `np.partition` threw `OverflowException` on a 2.1-billion-element array (int-indexed radix/introselect over managed scratch), caught by a new >int.MaxValue byte oracle; the Half/Complex/Decimal comparison path documents a clear `NotSupportedException` there instead of truncating - `ab15b165`, `85891751` (+ `7ea70490`).
- NumPy-style errors now chain the underlying exception - every catch that re-wraps a low-level failure into a verbatim `ValueError` / `IndexError` / `TypeError` / `IncorrectShapeException` (broadcast, index bounds, dtype/format, latin-1 header) passes it through as `InnerException`, so the root cause and stack survive (message text unchanged) - `fe42db9b`.
- Honestly.. many many more.

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
- A `bool` array combined with a weak integer literal now promotes to **int64** (was int32), matching NumPy's NEP50 - `np.left_shift(boolArr, 2)`, `boolArr + 2`, `boolArr & 2` etc. change result dtype; a narrower strong spelling like `(short)2` keeps its own kind - `46dbb9c6`.
- `np.poly1d` is now `IDisposable` - it owns the coefficient array its constructor yields into it, so `Dispose()` releases it, and the copy-constructor `new poly1d(p)` now **copies** the coefficients instead of sharing one array between two owners (NumPy shares by refcount; two NumSharp owners would double-dispose). Found by the new NDW016 ownership analyzer, which also made `np.Broadcast.Dispose()` (what `foreach` calls when a loop ends) release the `broadcast_to` views it had built (they are rebuilt lazily on the next `iters`/enumeration, so the object stays re-enumerable) and `IndexCollector` an `IDisposable` that no longer strands its outgrown buffer - `4cc9cac5`.
- The `dtype` argument is now **keyword-only** on the ufunc/reduction/logic surface (the Math ufuncs, the reductions, and the comparison/`isnan`/`isinf`/`isfinite` predicates) - `np.sqrt(x, typeof(float))` becomes `np.sqrt(x, dtype: ...)` (NumPy-faithful); `System.Type` / `NPTypeCode` / a dtype string all still bind via the new `DType` implicit conversion, and `power`/`floor_divide` drop their object-scalar-rhs-plus-`dtype` convenience overload - `c4ebbfb4`, `8b13234e`.

</details>