# NumSharp benchmark API coverage

Generated from the compiled NumPy 2.4.2 API inventory and the official benchmark sources.
A covered alias shares the exact compiled NumSharp target with a directly measured API.

Headline benchmark coverage: **98.7%** (455 covered of 461 benchmarkable APIs; 19 reviewed exclusions).

| Surface | Covered | Blocked | Excluded | Missing | Benchmarkable | Coverage |
|---|---:|---:|---:|---:|---:|---:|
| fft | 18 | 0 | 0 | 0 | 18 | 100.0% |
| linalg | 31 | 0 | 0 | 0 | 31 | 100.0% |
| ndarray | 52 | 0 | 15 | 0 | 52 | 100.0% |
| np | 309 | 0 | 1 | 3 | 312 | 99.0% |
| random | 45 | 0 | 3 | 3 | 48 | 93.8% |

## Execution routes

| Route | APIs | Meaning |
|---|---:|---|
| managed | 441 | NumSharp.Core managed code only. |
| openblas_optional | 10 | Direct product hook with managed fallback. |
| openblas_optional_composed | 4 | Composition can reach an accelerated product. |
| openblas_optional_sliding_dot | 2 | Long eligible correlate/convolve regions only. |
| openblas_required | 13 | LAPACK operation; no managed fallback. |
| openblas_required_composed | 7 | Top-level API composed from a required LAPACK factorisation. |
| openblas_hybrid | 3 | Backend requirement depends on parameters/composed path. |

## Native/OpenBLAS API map

Every API not listed below runs entirely in managed NumSharp.Core. Optional routes always
retain a managed fallback; required routes throw when no suitable LAPACK backend is installed.

| Route | APIs |
|---|---|
| openblas_optional | `numpy.dot`, `numpy.inner`, `numpy.linalg.matmul`, `numpy.linalg.vecdot`, `numpy.matmul`, `numpy.matvec`, `numpy.ndarray.dot`, `numpy.vdot`, `numpy.vecdot`, `numpy.vecmat` |
| openblas_optional_composed | `numpy.einsum`, `numpy.linalg.multi_dot`, `numpy.linalg.tensordot`, `numpy.tensordot` |
| openblas_optional_sliding_dot | `numpy.convolve`, `numpy.correlate` |
| openblas_required | `numpy.linalg.cholesky`, `numpy.linalg.det`, `numpy.linalg.eig`, `numpy.linalg.eigh`, `numpy.linalg.eigvals`, `numpy.linalg.eigvalsh`, `numpy.linalg.inv`, `numpy.linalg.lstsq`, `numpy.linalg.qr`, `numpy.linalg.slogdet`, `numpy.linalg.solve`, `numpy.linalg.svd`, `numpy.linalg.svdvals` |
| openblas_required_composed | `numpy.linalg.cond`, `numpy.linalg.matrix_rank`, `numpy.linalg.pinv`, `numpy.linalg.tensorinv`, `numpy.linalg.tensorsolve`, `numpy.polyfit`, `numpy.roots` |
| openblas_hybrid | `numpy.linalg.matrix_norm`, `numpy.linalg.matrix_power`, `numpy.linalg.norm` |

## Missing benchmark coverage

| API | Surface | Kind | Category | Backend route |
|---|---|---|---|---|
| `numpy.bmat` | np | function | Linear algebra | managed |
| `numpy.nancumprod` | np | function | Reductions | managed |
| `numpy.nancumsum` | np | function | Reductions | managed |
| `numpy.random.bytes` | random | function | Random | managed |
| `numpy.random.default_rng` | random | function | Random | managed |
| `numpy.random.random_integers` | random | function | Random | managed |

## Benchmark-blocking open bugs

These APIs remain in the benchmark denominator. They are not timed until the referenced
repeated-execution hazard is fixed; the reason is explicit rather than a silent omission.

| API | Reason | Evidence |
|---|---|---|

## Reviewed exclusions

| API | Reason |
|---|---|
| `numpy.ndarray.base` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.data` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.device` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.dtype` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.flags` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.flat` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.imag` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.itemsize` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.mT` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.nbytes` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.ndim` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.real` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.shape` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.size` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.ndarray.strides` | Attribute access has no size-dependent array kernel; cross-language timing is dispatch noise. |
| `numpy.random.get_state` | State snapshot API; no array kernel or size-dependent throughput. |
| `numpy.random.seed` | Global state mutation; timing is dominated by synchronization and is not an array-throughput comparison. |
| `numpy.random.set_state` | Global state mutation; timing is dominated by synchronization and is not an array-throughput comparison. |
| `numpy.shape` | Partial cross-surface property mapping; there is no comparable NumSharp np.shape call. |

## Unmatched benchmark descriptions

These official benchmark labels do not identify a current API-inventory row. Operator and
scenario-only benchmarks are expected here; API-looking stale labels should be corrected.

| Description | Source |
|---|---|
| `a + 5 (literal)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/AddBenchmarks.cs` |
| `a + b (element-wise)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/AddBenchmarks.cs` |
| `a + scalar` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/AddBenchmarks.cs` |
| `a / b (element-wise)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/DivideBenchmarks.cs` |
| `a / scalar` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/DivideBenchmarks.cs` |
| `scalar / a` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/DivideBenchmarks.cs` |
| `a % 7 (literal)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/ModuloBenchmarks.cs` |
| `a % b (element-wise)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/ModuloBenchmarks.cs` |
| `a * 2 (literal)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/MultiplyBenchmarks.cs` |
| `a * a (square)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/MultiplyBenchmarks.cs` |
| `a * b (element-wise)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/MultiplyBenchmarks.cs` |
| `a * scalar` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/MultiplyBenchmarks.cs` |
| `a - b (element-wise)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/SubtractBenchmarks.cs` |
| `a - scalar` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/SubtractBenchmarks.cs` |
| `scalar - a` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Arithmetic/SubtractBenchmarks.cs` |
| `a & b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Bitwise/BitwiseBenchmarks.cs` |
| `a ^ b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Bitwise/BitwiseBenchmarks.cs` |
| `a | b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Bitwise/BitwiseBenchmarks.cs` |
| `matrix * 2.0` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `matrix * col_vector (N,M)*(N,1)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `matrix * row_vector (N,M)*(M,)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `matrix + col_vector (N,M)+(N,1)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `matrix + row_vector (N,M)+(M,)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `matrix + scalar` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `tensor3D + matrix2D (D,D,D)+(D,D)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Broadcasting/BroadcastBenchmarks.cs` |
| `a != b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `a < b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `a <= b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `a == b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `a > b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `a >= b` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Comparison/ComparisonBenchmarks.cs` |
| `np.byteswap(a)` (`numpy.byteswap`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/ByteswapBenchmarks.cs` |
| `np.c_(a, b)` (`numpy.c_`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/IndexTricksBenchmarks.cs` |
| `np.r_(0:1:Nj)` (`numpy.r_`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/IndexTricksBenchmarks.cs` |
| `np.r_(0:N)` (`numpy.r_`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/IndexTricksBenchmarks.cs` |
| `np.r_(a, 0, 0, b)` (`numpy.r_`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/IndexTricksBenchmarks.cs` |
| `np.r_(a, b)` (`numpy.r_`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/IndexTricksBenchmarks.cs` |
| `reshape 1D -> 2D` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/ReshapeBenchmarks.cs` |
| `reshape 1D -> 3D` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/ReshapeBenchmarks.cs` |
| `reshape 2D -> 1D` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Manipulation/ReshapeBenchmarks.cs` |
| `a.amax() [method]` (`numpy.ndarray.amax`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Reduction/MinMaxBenchmarks.cs` |
| `a.amin() [method]` (`numpy.ndarray.amin`) | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Reduction/MinMaxBenchmarks.cs` |
| `a[100:1000] (contiguous slice)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[10:100, :] (row slice 2D)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[:, 10:100] (col slice 2D)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[::-1] (reversed)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[::2] (strided slice)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[::2] * 2` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[N/10:9N/10] * 2` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a[N/10:9N/10].copy()` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Slicing/SliceBenchmarks.cs` |
| `a * a (square via multiply)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Unary/PowerBenchmarks.cs` |
| `a * a * a (cube via multiply)` | `benchmark/NumSharp.Benchmark.CSharp/Benchmarks/Unary/PowerBenchmarks.cs` |
