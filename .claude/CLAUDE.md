# NumSharp Project Instructions

NumSharp is a .NET port of Python's NumPy targeting 1-to-1 API and behavioral compatibility with NumPy 2.x; the oracle
pins NumPy 2.4.2. `refs/numpy/` is a full clone of NumPy: the authority on API behavior, edge cases and implementation.

Area rules live in `.claude/rules/*.md` and load when Claude reads or edits a file in their area. The API index below
names the rule for each area: read it before answering a question about that area without opening its files.

## Principles

1. Match NumPy exactly: run the NumPy code first, observe, replicate in C#.
2. Match NumPy's implementation structure, not only its behavior: where NumSharp's code differs from a clean NumPy
   design, refactor it to NumPy's.
3. Edge cases are behavior: NaN handling, empty arrays, type promotion, broadcasting, negative axes.
4. Breaking changes that align with NumPy 2.x are acceptable.
5. Tests come from NumPy output.

When fixing a bug, read how `refs/numpy/` implements the function and port that structure instead of patching the
symptom.

## Implementing an `np.*` function (definition of done)

1. Read NumPy's docs and source, then run NumPy over the input space (dtypes, layouts, edge cases) to build a behavior
   matrix.
2. Check similar NumSharp implementations. Compose existing `np.*` functions, or delegate to the TensorEngine when a
   kernel is needed:
   ```csharp
   public static NDArray std(NDArray a, int? axis = null, ...)            // compose
       => np.sqrt(np.mean(np.power(a - np.mean(a, axis, keepdims: true), 2), axis));
   public static NDArray sum(NDArray a, int? axis = null, ...)            // delegate
       => a.TensorEngine.Sum(a, axis, typeCode, keepdims);
   ```
3. Layouts: C-contiguous (the SIMD path), strided / transposed views, broadcast views (stride 0, read-only), sliced views
   (`Shape.offset` in the base address).
4. Dtypes: all 15 through the type switch, or documented as unsupported:
   ```csharp
   switch (nd.typecode)
   {
       case NPTypeCode.Boolean: return Process<bool>(nd);    case NPTypeCode.Byte:    return Process<byte>(nd);
       case NPTypeCode.SByte:   return Process<sbyte>(nd);   case NPTypeCode.Int16:   return Process<short>(nd);
       case NPTypeCode.UInt16:  return Process<ushort>(nd);  case NPTypeCode.Int32:   return Process<int>(nd);
       case NPTypeCode.UInt32:  return Process<uint>(nd);    case NPTypeCode.Int64:   return Process<long>(nd);
       case NPTypeCode.UInt64:  return Process<ulong>(nd);   case NPTypeCode.Char:    return Process<char>(nd);
       case NPTypeCode.Half:    return Process<Half>(nd);    case NPTypeCode.Single:  return Process<float>(nd);
       case NPTypeCode.Double:  return Process<double>(nd);  case NPTypeCode.Decimal: return Process<decimal>(nd);
       case NPTypeCode.Complex: return Process<Complex>(nd);
       default: throw new NotSupportedException();
   }
   ```
5. NumPy parity: the signature (parameter names, order, defaults), NEP 50 promotion, edge cases, the exact result dtype,
   the error type and text.
6. Tests from NumPy output: edge cases (empty, scalar, broadcast, strided) and dtype coverage, plus oracle coverage
   (the `oracle` skill; `oracle.md`).

## Supported types (15)

| NPTypeCode | C# Type | NPTypeCode | C# Type |
|------------|---------|------------|---------|
| Boolean | bool | UInt64 | ulong |
| Byte | byte | Char | char |
| SByte | sbyte | Half | System.Half |
| Int16 | short | Single | float |
| UInt16 | ushort | Double | double |
| Int32 | int | Decimal | decimal |
| UInt32 | uint | Complex | System.Numerics.Complex |
| Int64 | long |  |  |

- The integers, float32 and float64 run full SIMD through the mixed-type kernel's `SimdFull` path (V128/V256/V512
  detected at startup). float16 has no BCL vector support and uses bit-level AVX2, SIMD widen-compute-narrow and scalar
  F16C widening (`kernels.md`). complex128 runs scalar through `System.Numerics.Complex` (about 2× slower than NumPy);
  its unary ufuncs match NumPy bit for bit, NaN sign included (`float-math.md`). Decimal runs scalar (`DecimalMath`):
  the highest precision and the slowest.
- Text: one more lane, `NPTypeCode.String`, stores NumPy's `bytes_` (`S`), `str_` (`U`) and NEP 55 `StringDType` (`T`);
  the descriptor carries the class and the width, and every computation NumPy runs on text computes (`strings.md`).
- NEP 50 result dtypes: `sum`/`prod`/`cumsum` of int32 → int64; `abs`/`sign` keep the dtype; `power(int32, float)` →
  float64.

## Architecture

```
NDArray           Main class (like numpy.ndarray)
├── Storage       UnmanagedStorage (raw pointers, not managed arrays)
├── Shape         Dimensions, strides, offset calculation
└── TensorEngine  Computation backend (DefaultEngine = pure C#)

np                Static API class (like `import numpy as np`)
├── np.random     NumPyRandom (1-to-1 seed/state with NumPy)
└── np.*          Functions in Creation/, Math/, Statistics/, Logic/, etc.
```

| Decision | Rationale |
|----------|-----------|
| Unmanaged memory | The fastest option measured; an NDArray manages its own allocation. |
| Buffers of one page or more start on a cache line | `SizeBucketedBufferPool.AlignedMinBytes` (4096) / `BufferAlignment` (64): a buffer at 16 mod 32 makes every second 32-byte vector load cross a cache line, and only the allocator can align every stream of a kernel (`memory-views.md`). |
| Order-aware layout | C order is the default; `Shape` also tracks F-contiguity, and `order` parameters resolve NumPy's `C`/`F`/`A`/`K` through `OrderResolver`. |
| Abstract TensorEngine | One computation contract; `DefaultEngine` is pure C#. `NumSharp.Core` has no native dependency and no P/Invoke: a native backend comes only as a separate package behind the settable `TensorEngine.Blas` property (`openblas.md`). |
| View semantics | Slicing returns views (shared memory), not copies. |
| `Shape` is a readonly struct | Immutable after construction (NumPy-aligned), with cached `ArrayFlags` for O(1) checks. |
| Broadcast write protection | Broadcast views are read-only (`IsWriteable = false`), as in NumPy. |
| Two IL kernel generators | Runtime IL with SIMD: `DirectILKernelGenerator` (whole-array kernels, `Backends/Kernels/Direct/`) and `ILKernelGenerator` (per-chunk kernels an NDIter drives, NumPy's ufunc inner-loop model) (`kernels.md`). |

- Where the main pieces live (under `src/NumSharp.Core/`): `NDArray` `Backends/NDArray.cs`; `UnmanagedStorage`
  `Backends/Unmanaged/UnmanagedStorage.cs`; `Shape` `View/Shape.cs`; `Slice` `View/Slice.cs`; `TensorEngine`
  `Backends/TensorEngine.cs`; `DefaultEngine` `Backends/Default/DefaultEngine.*.cs`; `np` `APIs/np.cs`; NDIter
  `Backends/Iterators/NDIter.cs`; kernels `Backends/Kernels/`; `InfoOf<T>` `Utilities/InfoOf.cs` (cached type facts:
  `Size`, `NPTypeCode`, `Zero`, `MaxValue`/`MinValue`, instead of reflection); `NDArray<T>` ``Generics/NDArray`1.cs`` (a
  typed wrapper whose indexer returns `T`, with a typed `Address` (`T*`) and `Array` (`ArraySlice<T>`)).
- `np.<name>` lives in `src/NumSharp.Core/<Area>/np.<name>.cs` (`Math`, `Creation`, `RandomSampling`, `Manipulation`,
  `Indexing`, `APIs`, `Logic`, `Statistics`, `Sorting_Searching_Counting`, `LinearAlgebra`, `LinearAlgebra/linalg`,
  `Polynomial`, `Fourier`, …: glob `**/np.<name>*.cs`), its engine half in `Backends/Default/**/Default.<Name>.cs`, its
  kernels in `Backends/Kernels/Direct/DirectILKernelGenerator.<Name>.cs`, its tests beside the same area in
  `test/NumSharp.Tests/`.
- Kernels are single-threaded and vectorized; `np.multithreading(enabled, max_threads)` is an opt-in for threaded
  kernels.
- `Shape`: dimensions, strides, offset, buffer size and cached `ArrayFlags` (`C_CONTIGUOUS`, `F_CONTIGUOUS`, `OWNDATA`,
  `ALIGNED`, `WRITEABLE`, `BROADCASTED`); O(1) `IsContiguous`, `IsFContiguous`, `IsBroadcasted`, `IsWriteable`,
  `IsSliced`, `IsSimpleSlice`. Shapes and arrays share dimension and stride arrays, so nothing writes into them: inside
  Core read `x.Shape.dimensions` / `x.Shape.strides` (not `x.shape`, which allocates a copy) and build a shape over a
  Core-created array with `Shape.Adopt(arr)`, never `new Shape(arr)` (`memory-views.md`).
- Views: `original["2:5"]` shares memory (`view[0] = 999` writes `original[2]`); `.copy()` copies. A view, and a
  byte-reinterpreting alias (`.real`/`.imag`, `view(dtype)`, `getfield`), keeps its owner's buffer alive; an internal
  alias must be disposed or created inside an `[NDScoped]` member. Allocating from a view's Shape is safe:
  `Shape.ForFreshAllocation` keeps a layout that already describes a fresh buffer (dense C/F, a KEEPORDER permutation)
  and lays any view layout out again in C order (`memory-views.md`).
- Slicing syntax:
  ```csharp
  nd[":"]           // all elements
  nd["1:5"]         // elements 1-4 (stop exclusive)
  nd["::2"]         // every 2nd element
  nd["-1"]          // last element (reduces dimension)
  nd["::-1"]        // reversed
  nd[":, 0"]        // all rows, first column
  nd["..., -1"]     // ellipsis fills dimensions
  ```

## API index

Tested against NumPy 2.4.2. Each line: the `np.*` names, then the rule that holds the area's details.
- Array creation: `arange`, `array`, `asanyarray`, `asarray`, `asarray_chkfinite`, `ascontiguousarray`, `asfortranarray`,
  `asmatrix`, `bmat`, `copy`, `empty`, `empty_like`, `eye`, `frombuffer`, `full`, `full_like`, `geomspace`, `identity`,
  `linspace`, `logspace`, `matrix`, `meshgrid`, `mgrid`, `ndarray`, `ogrid`, `ones`, `ones_like`, `require`, `tri`,
  `zeros`, `zeros_like` → `creation-manipulation.md` (`matrix.md` for `matrix`/`asmatrix`/`bmat`, `index-tricks.md` for
  the grids, `selection-indexing.md` for `tri`).
- Shape manipulation: `append`, `array_split`, `atleast_1d`, `atleast_2d`, `atleast_3d`, `block`, `c_`, `column_stack`,
  `concat`, `concatenate`, `delete`, `dsplit`, `dstack`, `expand_dims`, `flatten`, `flip`, `fliplr`, `flipud`, `hsplit`,
  `hstack`, `insert`, `intersect1d`, `matrix_transpose`, `moveaxis`, `pad`, `permute_dims`, `r_`, `ravel`, `repeat`,
  `reshape`, `resize`, `roll`, `rollaxis`, `rot90`, `setdiff1d`, `setxor1d`, `split`, `squeeze`, `stack`, `swapaxes`,
  `tile`, `transpose`, `trim_zeros`, `union1d`, `unique`, `unique_all`, `unique_counts`, `unique_inverse`,
  `unique_values`, `unstack`, `vsplit`, `vstack` → `creation-manipulation.md` (`unique.md` for `unique` and `unique_*`;
  every axis argument: `axis-arguments.md`).
- Broadcasting: `are_broadcastable`, `broadcast`, `broadcast_arrays`, `broadcast_to`.
- Math, arithmetic: `abs`, `absolute`, `acosh`, `add`, `arccos`, `arccosh`, `arcsin`, `arcsinh`, `arctan`, `arctan2`,
  `arctanh`, `asinh`, `atanh`, `cbrt`, `ceil`, `clip`, `convolve`, `correlate`, `cos`, `cosh`, `deg2rad`, `degrees`,
  `divide`, `divmod`, `exp`, `exp2`, `expm1`, `fabs`, `fix`, `float_power`, `floor`, `floor_divide`, `fmod`, `gcd`, `lcm`,
  `log`, `log10`, `log1p`, `log2`, `mod`, `modf`, `multiply`, `negative`, `positive`, `power`, `rad2deg`, `radians`,
  `reciprocal`, `remainder`, `rint`, `sign`, `sin`, `sinc`, `sinh`, `spacing`, `sqrt`, `square`, `subtract`, `tan`, `tanh`,
  `true_divide`, `trunc` → `ufuncs.md` and `float-math.md`.
- Math, reductions: `all`, `amax`, `amin`, `any`, `argmax`, `argmin`, `average`, `average_returned`, `count_nonzero`,
  `cov`, `cumprod`, `cumsum`, `diff`, `ediff1d`, `max`, `mean`, `median`, `min`, `percentile`, `prod`, `ptp`, `quantile`,
  `std`, `sum`, `var` → `reductions.md` (median / percentile / quantile: `quantile.md`).
- Math, NaN-aware: `nanmax`, `nanmean`, `nanmedian`, `nanmin`, `nanpercentile`, `nanprod`, `nanquantile`, `nanstd`,
  `nansum`, `nanvar` → `reductions.md` (the nan quantile forms: `quantile.md`).
- Bitwise: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert`, `left_shift`, `right_shift` → `ufuncs.md`.
- Comparison and logic: `all`, `allclose`, `any`, `array_equal`, `equal`, `fmax`, `fmin`, `greater`, `greater_equal`,
  `isclose`, `iscomplex`, `iscomplexobj`, `isfinite`, `isin`, `isinf`, `isnan`, `isreal`, `isrealobj`, `isscalar`,
  `iterable`, `less`, `less_equal`, `logical_and`, `logical_not`, `logical_or`, `logical_xor`, `maximum`, `minimum`,
  `not_equal`, `signbit` → `comparison.md`.
- Type promotion and dtypes: `bytes_`, `can_cast`, `character`, `common_type`, `complexfloating`, `datetime_data`,
  `dtype`, `dtypes` (the `np.dtypes` class module), `False_`, `find_common_type`, `finfo`, `flexible`, `floating`,
  `generic`, `iinfo`, `inexact`, `integer`, `isdtype`, `issctype`, `issubdtype`, `longdouble`, `min_scalar_type`,
  `mintypecode`, `number`, `promote_types`, `result_type`, `ScalarType`, `sctypeDict`, `signedinteger`, `str_`, `True_`,
  `typecodes`, `typing` (the `np.typing` module), `unsignedinteger` → `dtypes.md` (the classes: `scalar-types.md`).
- Text: `np.strings.*` (NumPy's 46 names) → `strings-methods.md`; text comparisons, arithmetic, reductions and ordering
  → `strings-compute.md`; `np.char.*` and `np.char.chararray` → `char-chararray.md`. Masked arrays: `np.ma.*`
  (`NDMaskedArray`) → `masked-arrays.md`.
- Selection: `choose`, `compress`, `extract`, `index_exp`, `indices`, `ix_`, `place`, `put`, `put_along_axis`, `putmask`,
  `ravel_multi_index`, `s_`, `select`, `take`, `take_along_axis`, `unravel_index`, `where` → `selection-indexing.md`
  (`index-tricks.md` for `index_exp`, `indices`, `ix_`, `s_`, `r_`, `c_`).
- Iteration: `nditer`, `ndenumerate`, `ndindex`, `nested_iters`, `flatiter`, `broadcast`, and the typed `nditer<T>` /
  `nditer_chunks<T>` / `flat<T>` → `iteration.md`. Fused expressions: `evaluate` → `evaluate.md`.
- Sorting and searching: `argmax`, `argmin`, `argpartition`, `argsort`, `argwhere`, `flatnonzero`, `lexsort`,
  `nanargmax`, `nanargmin`, `nonzero`, `partition`, `searchsorted`, `sort`, `sort_complex` → `sorting.md`.
- Linear algebra: `diag`, `diagflat`, `diag_indices`, `diag_indices_from`, `diagonal`, `dot`, `einsum`, `einsum_path`,
  `fill_diagonal`, `inner`, `mask_indices`, `matmul`, `matvec`, `outer`, `tensordot`, `trace`, `tril`, `tril_indices`,
  `tril_indices_from`, `triu`, `triu_indices`, `triu_indices_from`, `vdot`, `vecdot`, `vecmat`; `np.linalg.*`:
  `cholesky`, `cond`, `cross`, `det`, `diagonal`, `eig`, `eigh`, `eigvals`, `eigvalsh`, `inv`, `lstsq`, `matmul`,
  `matrix_norm`, `matrix_power`, `matrix_rank`, `matrix_transpose`, `multi_dot`, `norm`, `outer`, `pinv`, `qr`, `slogdet`,
  `solve`, `svd`, `svdvals`, `tensordot`, `tensorinv`, `tensorsolve`, `trace`, `vecdot`, `vector_norm`, `LinAlgError` →
  `linalg-einsum.md` (the diagonal/triangular family: `selection-indexing.md`; the OpenBLAS backend: `openblas.md`).
  Without a backend the products and the LU family (`det`, `slogdet`, `solve`, `inv`) compute managed; `cholesky`, `qr`,
  `svd`, the eigensolvers, `lstsq`, `pinv`, `matrix_rank`, `cond` and the SVD norms raise `NotSupportedException`.
- Window functions: `bartlett`, `blackman`, `hamming`, `hanning`, `kaiser`; Fourier: `np.fft.fft`, `ifft`, `fft2`,
  `ifft2`, `fftn`, `ifftn`, `rfft`, `irfft`, `rfft2`, `irfft2`, `rfftn`, `irfftn`, `hfft`, `ihfft`, `fftfreq`, `rfftfreq`,
  `fftshift`, `ifftshift`, plus `conjugate`/`conj`, `real`, `imag`, `angle` → `fft-windows.md`.
- `np.polynomial.*` (the six bases): `{p}val*`/`{p}grid*`, `polyvalfromroots`, `{p}add`/`{p}sub`/`{p}trim`/`{p}line`,
  the constants, `polyutils`, `{p}der`/`{p}int` → `polynomial-series.md`; `{p}mulx`/`{p}mul`/`{p}div`/`{p}pow`/
  `{p}fromroots`, `X2poly`/`poly2X`, `{p}vander*`, `{p}companion`/`{p}roots` → `polynomial-algebra.md`; `{p}weight`,
  `chebpts1/2`, `{p}gauss`, `chebinterpolate`, `{p}fit` → `polynomial-quadrature.md`; the six series classes,
  `format_float`, `set_default_printstyle` → `polynomial-classes.md`.
- Random (`np.random.*`): `bernoulli`, `beta`, `binomial`, `chisquare`, `choice`, `dirichlet`, `exponential`, `f`,
  `gamma`, `geometric`, `get_bit_generator`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `logseries`,
  `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`,
  `permutation`, `poisson`, `power`, `rand`, `randint`, `randn`, `random_sample`, `rayleigh`, `seed`, `set_bit_generator`,
  `shuffle`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`,
  `uniform`, `vonmises`, `wald`, `weibull`, `zipf`, plus `Generator`/`default_rng` → `random.md`.
- File I/O: `fromfile`, `fromstring`, `load`, `load_npy`, `load_npz`, `loadtxt`, `save`, `savetxt`, `savez`,
  `savez_compressed`, `tofile`; printing: `array2string`, `array_repr`, `array_str`, `format_float_positional`,
  `format_float_scientific`, `get_printoptions`, `printoptions`, `set_printoptions` → `io-printing.md` (the text
  functions: `text-io.md`).
- Other: `around`, `asscalar`, `copyto` (`creation-manipulation.md`), `round_`, `size`, `multithreading`.
- Operators: arithmetic `+`, `-`, `*`, `/`, `%`, unary `-`; comparison `==`, `!=`, `<`, `>`, `<=`, `>=` (they return
  `NDArray<bool>`); logical `&`, `|`, `!`.
- Indexing: integers and slice strings (`nd[0]`, `nd["1:3"]`, `nd["::-1"]`), boolean masks (`nd[mask]`, and
  `nd[mask] = v`), fancy indexing (`nd[indices]`). A raw `int[]`/`long[]` as the ONLY index is fancy (`nd[new int[]{0,2}]` selects rows 0 and 2); `nd[0,2]`
  is a coordinate, and coordinate access by array is `nd.GetData(int[]/long[])` → `selection-indexing.md`.
- Interop packages (OpenBLAS, ONNX Runtime, ML.NET, System.Numerics.Tensors, pythonnet) → `openblas.md`,
  `interop-packages.md`. ONNX Runtime, ML.NET and ParquetNet are UNRELEASED: no package, no docs, no mention in release
  notes.

## Build and test

The library and the tests multi-target net8.0 and net10.0; NumSharp.Core has no external runtime dependencies. Tests use
MSTest v3 with source-generated discovery. Test projects: `test/NumSharp.Tests` (the unit suite), `test/NumSharp.Tests.Oracle`
(the NumPy oracle replay), `test/NumSharp.Tests.Interop*` (the interop packages), `test/NumSharp.Tests.Build.Analyzer`
(the build analyzers).

```bash
dotnet build -v q --nologo "-clp:NoSummary;ErrorsOnly" -p:WarningLevel=0       # build, errors only

cd test/NumSharp.Tests
dotnet test --no-build --filter "TestCategory!=OpenBugs&TestCategory!=HighMemory"   # what CI runs (per framework)
dotnet test --no-build                                                        # everything, known bugs included
dotnet test --no-build --filter "TestCategory=OpenBugs"                       # only the known bugs
dotnet test --no-build --filter "ClassName~BinaryOpTests"                     # one class
dotnet test --no-build --filter "Name~Add_Int32_SameType"                     # one method
dotnet test --no-build -v normal                                              # passed tests too
dotnet test ../NumSharp.Tests.Oracle --filter "TestCategory=FuzzMatrix"       # the NumPy oracle gate
dotnet test --filter "TestCategory=NpyOracle"                                 # the .npy/.npz format gate

dotnet test --no-build 2>&1 | grep -E "^(failed|skipped|Test run|  total:|  failed:|  succeeded:|  skipped:|  duration:)"
dotnet test --no-build 2>&1 | grep -v "^    at " | grep -v "^     at " | grep -v "^    ---" | grep -v "^  from K:" | sed 's/AssertFailedException: //'
```

Test categories (typed attributes in `TestCategory.cs`; CI excludes by filter, so a new bug reproduction needs only the
attribute):

| Category | Attribute | Purpose | CI |
|----------|-----------|---------|----|
| `OpenBugs` | `[OpenBugs]` | known-failing bug reproductions; remove the attribute when fixed | excluded |
| `Misaligned` | `[Misaligned]` | a documented NumSharp vs NumPy behavioral difference | runs (passes) |
| `WindowsOnly` | `[WindowsOnly]` | needs GDI+/System.Drawing.Common | excluded off Windows |
| `LongIndexing` | `[LongIndexing]` | arrays with more than int.MaxValue elements | runs unless also HighMemory |
| `HighMemory` | `[HighMemory]` | needs 8 GB+ RAM | excluded |
| `LargeMemoryTest` | `[LargeMemoryTest]` | memory-heavy non-bugs (OpenBugs + HighMemory) | excluded |
| `FuzzMatrix` | `[TestCategory("FuzzMatrix")]` | the NumPy differential gate (`oracle.md`) | runs |

Suite speed, the parallel rules, guard pages and the OpenBugs inventory: `tests.md`. CI, packaging and strong naming:
`ci-release.md`.

## Performance convention

- Ratios are NPY/NS = NumPy_ms / NumSharp_ms: > 1 means NumSharp is faster, < 1 slower; higher is better. Icons:
  ✅ ≥ 1.0 · 🟡 ≥ 0.5 · 🟠 ≥ 0.2 · 🔴 < 0.2.
- Time Release builds only (`dotnet run -c Release - < script.cs`; Debug taints hand-written kernels about 2×), pin a C#
  timing process to TWO CPUs (`start /affinity 14`: two P-cores here; a single CPU starves the background tier-1
  compiler, so hot paths stay at tier 0), pin the NumPy twin the same way, take the best of several rounds with warm-up
  excluded, and check correctness before every timed row.
- The harness, its subsystems, the BenchmarkDotNet configuration, the snapshots (local `benchmark/history/`, published to
  the `data` branch through the `refs/data` submodule) and the report/UI surfaces: `benchmark/CLAUDE.md` and the
  `benchmark` skill.

## Scripting with `dotnet run` (.NET 10 file-based apps)

```csharp
#:project path/to/src/NumSharp.Core
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property SignAssembly=true
#:property AssemblyOriginatorKeyFile=K:/source/NumSharp/Open.snk
```

- `AssemblyName=NumSharp.DotNetRunScript` matches NumSharp's `[assembly: InternalsVisibleTo("NumSharp.DotNetRunScript,
  PublicKey=…")]`, which grants the script every `internal` member. NumSharp is strong-named, so a keyed
  `InternalsVisibleTo` matches only an assembly signed with that key: a script outside the repo root needs the two
  signing properties (with an absolute key path); a script under the repo root inherits them from
  `Directory.Build.props`. NumSharp uses unsafe code throughout, hence `AllowUnsafeBlocks`.
- Useful internals: `shape.dimensions`, `shape.strides`, `shape.size`, `shape.offset`, `shape.bufferSize`,
  `shape._flags`, `shape.IsWriteable`, `shape.IsBroadcasted`, `shape.IsSimpleSlice`, `shape.OriginalSize`,
  `arr.Storage`, `arr.GetTypeCode`, `arr.Array` (`IArraySlice`), `np._FindCommonType(...)`, `np.powerOrder`,
  `NPTypeCode.GetGroup()` / `GetPriority()` / `AsNumpyDtypeName()`, `Shape.NewScalar()`. Public: `nd.shape`, `nd.ndim`,
  `nd.size`, `nd.dtype` (a `DType`; `nd.dtype.type` is the CLR `Type`), `nd.typecode`, `nd.T`, `nd.flat`.
- A file-based app reuses its cached build, NumSharp.Core included, when the script did not change: pass `--no-cache`
  after editing Core.

## GitHub issues

Create issues on `SciSharp/NumSharp` with `gh issue create` (`GH_TOKEN` through the `env-tokens` skill).
- Feature / enhancement: Overview (what and why, 1–2 sentences); Problem; Proposal with a task checklist (`- [ ]`);
  Evidence (data, benchmarks, references); Scope / Non-goals; Benchmark / Performance when applicable (before/after,
  methodology); a Breaking changes table when any (Change | Impact | Migration); Related issues.
- Bug report: Overview (the bug and its impact); Reproduction (minimal code); Expected (NumPy's output as the source of
  truth); Actual (the error, wrong output or crash); Workaround (if any); Root cause (file, line, why, if known); Related
  issues.

## Library-wide traps

- RyuJIT may swap the operands of a commutative operation, so the IL operand order proves nothing about which NaN wins:
  encode NaN priority explicitly (`kernels.md`). RyuJIT also rewrites `(-a) + b` into `b - a`, which changes a NaN's
  sign.
- `Environment.SetEnvironmentVariable` does not reach a native `getenv` (.NET keeps its own table): go through the CRT
  (`ucrtbase!_putenv_s` / `libc!setenv`).
- `/` on an integer NDArray is TRUE division: `(ar / 64 % 2) == 0` is not a run-of-64 mask; use `np.floor_divide`.
- `NDArray.SetAtIndex<T>(T, long)` checks `T` against the dtype only with a `Debug.Assert`: in Release a wrong `T` writes
  its own size (a `double` into a float32 array overwrites two elements, past the buffer for the last one). Write each
  dtype with its own type.
- A C# scalar literal operand (`a * 2.0`, `m > 0.75`) allocates a 0-d temporary left to the finalizer; in a hot loop hold
  the 0-d operand (`NDArray s = 0.75`) or dispose it. Build a computed scalar with `NDArray.Scalar(x)`, never
  `np.array(x)` (an explicit 0-d array changes `NDMatrix *`).
- C# overload resolution prefers an `NDArray` overload over `object` for any C# array argument (the implicit Array →
  NDArray conversion), so an `object[]` (a Python list) needs its own overload. A `?:` or switch expression with an
  `object[]` arm and an `NDArray` arm is typed `NDArray`; a switch expression takes its arms' natural type (cast arms to
  `object`). A lone `object[]` passed to `params object[]` IS the params array.
- `stackalloc` is zero-filled unless the method carries `[SkipLocalsInit]`.
- Normalize an axis with `Shape.check_and_adjust_axis` / `Shape.normalize_axis_tuple` (test as given, then wrap): a
  `while (axis < 0) axis += ndim` loop never ends on a 0-d input (`axis-arguments.md`).
- A DynamicMethod whose JIT inline budget is spent inlines nothing more, `AggressiveInlining` included: split large
  emitted kernels into stages (one DynamicMethod each) and check with `DOTNET_JitDisasm` + `DOTNET_JitStdOutFile`.
- Read `TensorEngine.Blas` into a local before calling it; a test-then-call on the settable property races
  `OpenBlasEngine.Disable()`.
- MSTest's `[TestClass]` is not inherited: a class with `[TestMethod]`s and no attribute of its own never runs.
- New pointer code runs under `NUMSHARP_DEBUG_GUARD_PAGES=1`, and a stride kernel is tested on a view that ENDS at its
  buffer's last element (`tests.md`). A test whose two sides share a wrong operand proves nothing.
- Measuring: one warm-up pass only queues tier-1 (warm, pause, warm again); time two routes in one process with
  interleaved rounds; never A/B the live tree (a chain of `dotnet run` benches compiles the tree as it is when each run
  starts: put both sides in detached worktrees); measure the host clock before an argument in cycles.
- A copy-ready example project is the regular form: exactly what `dotnet new` plus `dotnet add package` write (one plain
  `PackageReference`; no custom MSBuild property, no ProjectReference switch, no `SignAssembly`/`IsPackable` line), so it
  builds against the published package and uses only its API.

## Where else to read

- The rules (`.claude/rules/`), and what each holds:

| Rule | Holds |
|------|-------|
| `openblas.md` | the `IBlasBackend` seam; `NumSharp.Interop.OpenBLAS`: why it exists, the bundled binary and discovery order, the macOS loader, the build-time override, what decides the result bits, API, gates, examples |
| `interop-packages.md` | the unreleased-package policy; ONNX Runtime, ML.NET, System.Numerics.Tensors (sources, verbs, dtypes, lifetime, gates); the pythonnet examples |
| `kernels.md` | the two IL generators' contracts, shared infrastructure, coverage, kernel rules, float16 techniques, NaN operand order, the 2-D block kernel, broadcast pattern stages |
| `memory-views.md` | `Shape` (fields, flags, sharing), Slice / broadcasting / operator internals, views and aliases, allocation from a view's Shape, the buffer pool |
| `creation-manipulation.md` | `np.arange`, `np.ndarray`, the `as*` family, object-list coercion, full / ones / eye, `np.frombuffer`, linspace / logspace / geomspace, set routines, flips / transposes, stacking, resize / tolist, allocation and reshape guards, `np.copyto` |
| `unique.md` | `np.unique` (`UniqueResult`, the axis path, NumPy's flat routes) and the Array-API unique family, the hash path |
| `axis-arguments.md` | `Shape.check_and_adjust_axis` / `normalize_axis_tuple`, the rules every axis argument follows, NumPy's errors around axes |
| `matrix.md` | `NDMatrix`: static type versus object (the matrix boundary), scalars, indexing, operators and scaling, reductions, construction, divergences, gates |
| `ufuncs.md` | the ufunc surface (`dtype=`, `out=`, `where=`), fix, fabs, clip, divmod, float_power, power's float loops, gcd / lcm, bitwise |
| `float-math.md` | correlate / convolve, inverse hyperbolic, sinc, the float32 kernels ported from NumPy, exp2, expm1 / log1p, complex128 NaN signs |
| `reductions.md` | the reduce prologue, sum / mean, var / std, numpy.ma mean / var, max / min / ptp schedules and keywords, float16 HALF loops, cov, the NaN-aware family |
| `quantile.md` | the quantile family: `q`'s dtype and the lerp, `out=`, the nan forms, `weights=`, the partition |
| `masked-arrays.md` | numpy.ma: semantics by result kind, the fill value, the set routines, `ma.put`, text and `StringDType` |
| `comparison.md` | signbit, isin, iterable, the comparison ufuncs (NEP 50 weak scalars), the logical ufuncs, the comparison engine |
| `dtypes.md` | the DType model, the C-type classes, promotion and casting, the `np.dtype(string)` parser, the element slots, traps, NumPy's names, C enums and find_common_type |
| `scalar-types.md` | `NDScalarType` and Python's built-in classes, issubdtype / isdtype / issctype / finfo / iinfo of a class, `np.typing` |
| `strings.md` | the string dtypes: the refuse-or-compute rule, the C# boundary, movement, `StringDType` arrays, text ↔ number, gates |
| `strings-compute.md` | the text kernels, `StringDType` computation and NA rules, text sorts / searches / set routines, ordering `T` |
| `strings-methods.md` | the 46 `np.strings` functions: search / strip / pad, predicates, case maps, translate, mod, encode / decode |
| `char-chararray.md` | `numpy.char`, `NDCharArray` and its boundary, the object path (`NDListArray`) |
| `selection-indexing.md` | the one indexer and its index forms, storing a C# literal, take / put / place / putmask, take_along_axis / put_along_axis / select, unravel_index / ravel_multi_index, the diagonal and triangular family |
| `index-tricks.md` | the `r_` / `c_` / `ix_` / `s_` / `index_exp` DSL, `mgrid` / `ogrid` / `meshgrid`, `np.indices` |
| `iteration.md` | NDIter, ndindex / ndenumerate / nditer and their traps, nested_iters, flatiter, typed iteration, `nd.Unsafe` and the .NET bridges |
| `evaluate.md` | `np.evaluate`: typing, root reductions, execution and streaming, programs, 0-d parameters, the cache, `Compile()`, Call nodes |
| `sorting.md` | axis argmax / argmin, partition / argpartition / lexsort / nanargmax / nanargmin / sort_complex |
| `linalg-einsum.md` | what computes with and without a backend, the seam, NumPy behaviours to keep, einsum and einsum_path, signature parity |
| `fft-windows.md` | the window functions; `np.fft`: the pocketfft port, layer-3 semantics, the single-precision engine, the F1 divergence |
| `polynomial-series.md` | numpy.polynomial evaluation, the additive family and polyutils, calculus, the array_like coercion |
| `polynomial-algebra.md` | series algebra, Vandermonde matrices, companion matrices and roots |
| `polynomial-quadrature.md` | weights, Chebyshev points, Gauss quadrature, Chebyshev interpolation, least-squares fits |
| `polynomial-classes.md` | the six series classes (protocol, lifetimes, printing) and the printing helpers |
| `random.md` | integer types, NumPy's call spellings, the kind of array a sampler returns |
| `io-printing.md` | `.npy` / `.npz` (the `T` pickle), array printing, the Dragon4 float formatters |
| `text-io.md` | savetxt / loadtxt / fromstring / fromfile / tofile, Python's codecs, the `%` operator |
| `tests.md` | keeping the suite fast, the memory-safety gates, the OpenBugs files and audit |
| `oracle.md` | the differential-fuzz pipeline: layout, gate semantics, every tier, the `.npy` format oracle, traps |
| `ci-release.md` | the workflows, no embedded package README, strong naming |

- Skills: `oracle` (the differential-fuzz gate), `benchmark`, `changelog`, `pythonic`, `specialized-path`; the command
  `np-function`.
- Nested guides: `benchmark/CLAUDE.md`, `test/NumSharp.Tests.Interop/CLAUDE.md`,
  `examples/NeuralNetwork.NumSharp/.claude/CLAUDE.md` (the Keras-style example framework and its MNIST MLP demo); the
  divergence ledger `test/NumSharp.Tests.Oracle/Fuzz/README.md`.
- At the repository root: `ARCHITECTURE.md` (technical details) and `CONTRIBUTING.md` (development workflow).
