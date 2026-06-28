# NumSharp Benchmark Suite - Development Guide

This document provides comprehensive guidance for working with the NumSharp benchmark infrastructure. It covers architecture, patterns, API usage, extending benchmarks, and troubleshooting.

---

## ⚠️ CRITICAL: ad-hoc `dotnet run` scripts build DEBUG by default

`dotnet run file.cs` / `dotnet_run <<'EOF'` (file-based apps) compile **both the
script AND any `#:project`-referenced NumSharp.Core in Debug** with
`DebuggableAttribute(DisableOptimizations)` — the JIT honors it even over
`[MethodImpl(AggressiveOptimization)]`. Measured effect: hand-written C# hot
loops run ~2× slower; `NDIterRef` construction ~40% slower.

**Rule: every timing script must run as `dotnet run -c Release - < script.cs`.**
- `#:property Optimize=true` fixes only the *script* assembly — NumSharp.Core stays Debug.
- `#:property Configuration=Release` changes output paths but the binaries remain unoptimized.
- Only the command-line `-c Release` produces optimized script + optimized NumSharp.Core.

**Diagnostic signature of a Debug-tainted measurement**: paths through
`DynamicMethod`-emitted kernels (ILKernelGenerator, `ExecuteExpression`) look
normal — emitted IL is always JIT-optimized — while hand-written C# kernels and
NumSharp.Core C# glue inflate ~2×. If strided/custom-kernel numbers look ~2×
worse than contiguous/IL-kernel numbers suggest, check
`Assembly.GetCustomAttribute<DebuggableAttribute>().IsJITOptimizerDisabled`
for both assemblies.

The BenchmarkDotNet projects below are unaffected (they already mandate `-c Release`).

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Architecture](#architecture)
4. [Infrastructure Classes](#infrastructure-classes)
5. [Benchmark Categories](#benchmark-categories)
6. [Python Benchmark System](#python-benchmark-system)
7. [Report Generation](#report-generation)
8. [Running Benchmarks](#running-benchmarks)
9. [Adding New Benchmarks](#adding-new-benchmarks)
10. [Type System](#type-system)
11. [NumSharp API Patterns](#numsharp-api-patterns)
12. [Common Issues & Solutions](#common-issues--solutions)
13. [Performance Interpretation](#performance-interpretation)
14. [CI Integration](#ci-integration)

---

## Overview

The benchmark suite provides fair, reproducible performance comparisons between NumSharp and NumPy. It was designed with these principles:

- **Matching methodology**: Same operations, same array sizes, same random seeds
- **Comprehensive type coverage**: All 15 NumSharp-supported data types
- **Categorical organization**: Operations grouped by type (arithmetic, unary, reduction, etc.)
- **Automated reporting**: JSON export and Markdown report generation
- **Cross-platform**: Runs on Windows, Linux, macOS

### Key Metrics

| Metric | Current Coverage |
|--------|-----------------|
| Operations | ~615 per size (**1,851** op×dtype×N cells in the official op matrix) |
| Data Types | **15** (all NumSharp types) |
| Comparison suites | **14** (arithmetic, unary, reduction, broadcast, creation, manipulation, slicing, comparison, bitwise, logic, statistics, sorting, linalg, selection) |
| Matrix subsystems | **5** appended to every official run (nditer, layout, operand, cast, fusion) |
| Array Sizes | **3** cache tiers in the official run (1K / 100K / 10M); the C# project also defines Scalar/100 |
| Convention | **NPY/NS** — ratio = NumPy_ms / NumSharp_ms, **>1 = NumSharp faster** |

---

## Directory Structure

```
benchmark/
├── CLAUDE.md                              # This file (development guide)
├── run-benchmarks.ps1                     # PowerShell benchmark runner
├── README.md                              # Benchmark results (== benchmark-report.md)
├── benchmark-report.md                    # Generated report (after running)
├── benchmark-report.json                  # JSON results (after running)
│
├── run_benchmark.py                       # THE entry point (orchestrates everything below)
│
├── history/                               # TRACKED snapshots — *what we commit and reference*
│   ├── latest -> <date>_<sha>             # symlink (git mode 120000) → the newest snapshot
│   └── <date>_<sha>/                      # MANIFEST.md + benchmark-report.{md,json,csv}
│                                          #   + numpy-results.json + every subsystem
│                                          #   *_results.{md,tsv} + cards/  (full provenance)
├── results/                               # GITIGNORED raw per-run scratch (results/<timestamp>/)
│
├── scripts/                               # Helper scripts
│   ├── merge-results.py                   # Merges NumPy and NumSharp op-matrix results
│   ├── bench_common.py                    # Shared driver for the matrix subsystems (build/run/parse)
│   └── snapshot_history.py                # Builds history/<date>_<sha>/ + latest (the publish step)
│
├── nditer/                               # Subsystem: iterator machinery (aspect × tier)
├── layout/                                # Subsystem: reduction/copy/elementwise × memory layout × dtype
│   ├── {reduce_layout,copy_path,elementwise_layout}_bench.{cs,py}
│   └── layout_sheet.py → layout_results.md (+ .tsv)
├── operand/                               # Subsystem: 1-D / scalar / mixed-operand / broadcast
│   ├── operand_bench.{cs,py}
│   └── operand_sheet.py → operand_results.md (+ .tsv)
├── cast/                                  # Subsystem: astype src→dst × layout × dtype
│   ├── cast_matrix_bench.{cs,py}
│   └── cast_sheet.py → cast_results.md (+ .tsv)
├── fusion/                                # Subsystem: np.evaluate fused vs unfused
│   ├── evaluate_bench.{cs,py}
│   └── fusion_sheet.py → fusion_results.md
│
├── NumSharp.Benchmark.Python/             # Python/NumPy benchmarks
│   └── numpy_benchmark.py                 # NumPy benchmark implementation
│
└── NumSharp.Benchmark.CSharp/        # C# BenchmarkDotNet project
    ├── README.md                          # C# benchmark documentation
    ├── Program.cs                         # Entry point with interactive menu
    ├── NumSharp.Benchmark.CSharp.csproj
    │
    ├── Infrastructure/                    # Base classes and configuration
    │   ├── BenchmarkConfig.cs             # BenchmarkDotNet configurations
    │   ├── BenchmarkBase.cs               # Base class for all benchmarks
    │   ├── TypeParameterSource.cs         # Type parameter sources (NPTypeCode)
    │   └── ArraySizeSource.cs             # Standard array size constants
    │
    └── Benchmarks/                        # Benchmark implementations
        │   # Experimental (no NumPy counterpart — NOT run by run_benchmark.py):
        ├── DispatchBenchmarks.cs          # DynamicMethod dispatch
        ├── FusionBenchmarks.cs            # Kernel fusion patterns
        ├── NumSharpBenchmarks.cs          # NumSharp baseline
        ├── DynamicEmissionBenchmarks.cs   # Per-op DynMethod
        ├── SimdVsScalarBenchmarks.cs      # SIMD vs scalar kernels
        ├── Allocation/                    # Alloc/zero-init micro + size sweeps
        │   ├── AllocationMicroBenchmarks.cs
        │   ├── AllocationSizeBenchmarks.cs
        │   ├── NumSharpAllocationBenchmarks.cs
        │   └── ZeroInitBenchmarks.cs
        │
        │   # Comparison suites (the 14 run by run_benchmark.py, each with a NumPy twin):
        ├── Arithmetic/                    # +, -, *, /, % (Add/Subtract/Multiply/Divide/Modulo)
        ├── Unary/                         # MathBenchmarks, ExpLogBenchmarks, TrigBenchmarks,
        │                                  #   PowerBenchmarks, UnaryExtraBenchmarks (cbrt/reciprocal/
        │                                  #   square/negative/positive/trunc)
        ├── Reduction/                     # Sum, Mean, VarStd, MinMax, Prod,
        │                                  #   CumulativeBenchmarks (cumsum/cumprod), NanReductionBenchmarks
        ├── Broadcasting/                  # BroadcastBenchmarks (scalar/row/column/3D/broadcast_to)
        ├── Creation/                      # CreationBenchmarks (zeros/ones/empty/full/*_like)
        ├── Manipulation/                  # Reshape, Stack, Dims
        ├── Slicing/                       # SliceBenchmarks (contiguous/strided/reversed/copy)
        ├── Comparison/                    # ComparisonBenchmarks (==, !=, <, >, <=, >=)
        ├── Bitwise/                       # BitwiseBenchmarks (and/or/xor/invert/shifts)
        ├── Logic/                         # LogicBenchmarks (logical_and/or/not/xor, all/any, isnan/…)
        ├── Statistics/                    # StatisticsBenchmarks (median/percentile/quantile/average/ptp)
        ├── Sorting/                       # SortingBenchmarks (sort/argsort/searchsorted)
        ├── LinearAlgebra/                 # LinAlgBenchmarks (dot/matmul/outer/trace)
        ├── Selection/                     # WhereBenchmarks (np.where + fancy/boolean selection)
        └── MultiDim/                      # MultiDimBenchmarks (1D vs 2D vs 3D — experimental)
```

> The **14 comparison suites** are wired into `run_benchmark.py`'s `SUITES` map (namespace filters like `*Benchmarks.Statistics.*`); the experimental/allocation classes are excluded because they have no NumPy twin. See `run_benchmark.py` for the authoritative list.

---

## Architecture

### Three-Tier Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    Report Generation Layer                       │
│  run_benchmark.py → merge-results.py + subsystem *_sheet.py       │
│    → benchmark-report.md + history/<date>_<sha>/  (legacy: ps1)   │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Benchmark Execution Layer                     │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │  C# / BenchmarkDotNet │    │  Python / NumPy                │ │
│  │  NumSharp.Benchmark   │    │  numpy_benchmark.py            │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  BenchmarkBase → TypeParameterSource → ArraySizeSource           │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
1. Setup Phase:
   - CreateRandomArray(N, dtype, seed) → NDArray
   - Same seed ensures reproducibility across C#/Python

2. Benchmark Phase:
   - BenchmarkDotNet / Python benchmark() wrapper
   - Warmup iterations excluded
   - Statistical analysis (mean, stddev, min, max)

3. Export Phase:
   - C#: JSON via BenchmarkDotNet exporters
   - Python: JSON via --output flag

4. Report Phase:
   - PowerShell merges results
   - Generates Markdown report with tables
```

---

## Infrastructure Classes

### BenchmarkBase

Base class providing array creation helpers.

```csharp
public abstract class BenchmarkBase
{
    // Override in derived class
    public virtual int N { get; set; } = ArraySizeSource.Large;

    // Reproducible random seed
    protected const int Seed = 42;

    // Create typed random arrays
    protected static NDArray CreateRandomArray(int n, NPTypeCode dtype, int seed = Seed);
    protected static NDArray CreatePositiveArray(int n, NPTypeCode dtype, int seed = Seed);
    protected static NDArray CreateRandomArray2D(int rows, int cols, NPTypeCode dtype, int seed = Seed);
    protected static NDArray CreateRandomArray3D(int d1, int d2, int d3, NPTypeCode dtype, int seed = Seed);

    // Create scalar values for each type
    protected static object GetScalar(NPTypeCode dtype, double value = 42.0);
}
```

### TypedBenchmarkBase

Extension that adds dtype parameterization.

```csharp
public abstract class TypedBenchmarkBase : BenchmarkBase
{
    [ParamsSource(nameof(Types))]
    public NPTypeCode DType { get; set; }

    // Override to customize available types
    public virtual IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;
}
```

### TypeParameterSource

Static class providing type collections.

```csharp
public static class TypeParameterSource
{
    // All 15 NumSharp types
    public static IEnumerable<NPTypeCode> AllNumericTypes;

    // Fast subset: int32, int64, float32, float64
    public static IEnumerable<NPTypeCode> CommonTypes;

    // All except bool and char
    public static IEnumerable<NPTypeCode> ArithmeticTypes;

    // float32, float64, decimal (for sqrt, log, trig)
    public static IEnumerable<NPTypeCode> TranscendentalTypes;

    // Minimal: int32, float64
    public static IEnumerable<NPTypeCode> MinimalTypes;

    // Integer types only
    public static IEnumerable<NPTypeCode> IntegerTypes;

    // Floating types only
    public static IEnumerable<NPTypeCode> FloatingTypes;

    // Helpers
    public static string GetDtypeName(NPTypeCode code);   // NumPy dtype name
    public static string GetShortName(NPTypeCode code);   // Display name
}
```

### ArraySizeSource

Standard array sizes for consistency.

```csharp
public static class ArraySizeSource
{
    public const int Small = 1_000;        // L1 cache, per-element overhead
    public const int Medium = 100_000;     // L2/L3 cache, typical use
    public const int Large = 10_000_000;   // Memory-bound throughput

    public static IEnumerable<int> StandardSizes;  // All three
    public static IEnumerable<int> QuickSizes;     // Large only

    // 2D/3D size tuples
    public static IEnumerable<(int, int)> Matrix2DSizes;
    public static IEnumerable<(int, int, int)> Tensor3DSizes;
}
```

---

## Benchmark Categories

### Arithmetic (`Benchmarks/Arithmetic/`)

Binary arithmetic operations between arrays and scalars.

| File | Operations | Notes |
|------|------------|-------|
| `AddBenchmarks.cs` | `+`, `np.add`, scalar add, row/col broadcast | Tests both operator and function syntax |
| `SubtractBenchmarks.cs` | `-`, scalar subtract | Tests both directions (a-b, scalar-a) |
| `MultiplyBenchmarks.cs` | `*`, square, scalar multiply | Tests self-multiplication |
| `DivideBenchmarks.cs` | `/`, scalar divide | Uses positive arrays to avoid div-by-zero |
| `ModuloBenchmarks.cs` | `%`, scalar modulo | Limited to types supporting modulo |

**Type coverage**: `ArithmeticTypes` (excludes bool, char)

### Unary (`Benchmarks/Unary/`)

Single-input mathematical functions.

| File | Operations | Notes |
|------|------------|-------|
| `MathBenchmarks.cs` | sqrt, abs, sign, floor, ceil, around, clip | Basic math operations |
| `ExpLogBenchmarks.cs` | exp, exp2, expm1, log, log2, log10, log1p | Exponential/logarithmic |
| `TrigBenchmarks.cs` | sin, cos, tan | Trigonometric (expensive!) |
| `PowerBenchmarks.cs` | power(a, 2), power(a, 3), power(a, 0.5) | Scalar exponents only |

**Type coverage**: `TranscendentalTypes` (float32, float64, decimal)

**Note**: NumSharp `np.power` only supports scalar exponents (`ValueType`), not element-wise NDArray exponents.

### Reduction (`Benchmarks/Reduction/`)

Operations that reduce array dimensionality.

| File | Operations | Notes |
|------|------------|-------|
| `SumBenchmarks.cs` | sum, sum(axis=0), sum(axis=1), cumsum | Full and axis reductions |
| `MeanBenchmarks.cs` | mean, mean(axis) | Returns floating-point |
| `VarStdBenchmarks.cs` | var, std (full and axis) | Multiple passes required |
| `MinMaxBenchmarks.cs` | amin, amax, argmin, argmax | Value and index operations |
| `ProdBenchmarks.cs` | prod (full and axis) | Uses small arrays to avoid overflow |

**Type coverage**: `ArithmeticTypes` for sum/minmax, `TranscendentalTypes` for var/std

### Broadcasting (`Benchmarks/Broadcasting/`)

Operations involving shape broadcasting.

| File | Operations | Notes |
|------|------------|-------|
| `BroadcastBenchmarks.cs` | scalar, row, column, 3D, broadcast_to | All broadcast patterns |

**Broadcasting patterns**:
- Scalar: `(N,M) + scalar`
- Row: `(N,M) + (M,)` → broadcasts row across all rows
- Column: `(N,M) + (N,1)` → broadcasts column across all columns
- 3D: `(D,D,D) + (D,D)` → broadcasts 2D across first dimension

### Creation (`Benchmarks/Creation/`)

Array allocation and initialization.

| File | Operations | Notes |
|------|------------|-------|
| `CreationBenchmarks.cs` | zeros, ones, empty, full, arange, linspace, copy, *_like | All creation patterns |

**Key insight**: `zeros` and `empty` are O(1) lazy allocation (~0.01ms), while `ones` and `full` require initialization (~8-16ms for 10M elements).

### Manipulation (`Benchmarks/Manipulation/`)

Shape and dimension manipulation.

| File | Operations | Notes |
|------|------------|-------|
| `ReshapeBenchmarks.cs` | reshape, transpose, ravel, flatten | View vs copy semantics |
| `StackBenchmarks.cs` | concatenate, stack, hstack, vstack, dstack | Combining arrays |
| `DimsBenchmarks.cs` | squeeze, expand_dims, swapaxes, moveaxis, rollaxis | Dimension manipulation |

**Key insight**: View operations (reshape, transpose, ravel) are O(1), while copy operations (flatten, concatenate) are O(N).

### Slicing (`Benchmarks/Slicing/`)

Array slicing and view operations.

| File | Operations | Notes |
|------|------------|-------|
| `SliceBenchmarks.cs` | contiguous, strided, reversed, row/col slices, copy | View creation and operations |

**Key insight**: Strided slices are slower than contiguous due to non-sequential memory access.

### MultiDim (`Benchmarks/MultiDim/`)

Comparing 1D, 2D, and 3D array performance.

| File | Operations | Notes |
|------|------------|-------|
| `MultiDimBenchmarks.cs` | add, sum, sqrt on 1D/2D/3D | Same total elements, different shapes |

### Added comparison categories

These suites were added on top of the original set; each has a NumPy twin in `numpy_benchmark.py` and is wired into `run_benchmark.py`.

| Category (dir) | File | Operations |
|---|---|---|
| Comparison | `ComparisonBenchmarks.cs` | `equal`/`not_equal`/`less`/`greater`/`less_equal`/`greater_equal` |
| Bitwise | `BitwiseBenchmarks.cs` | `bitwise_and`/`or`/`xor`, `invert`, `left_shift`/`right_shift` |
| Logic | `LogicBenchmarks.cs` | `logical_and`/`or`/`not`/`xor`, `all`/`any`, `isnan`/`isfinite`/`isinf` |
| Statistics | `StatisticsBenchmarks.cs` | `median`, `percentile`, `quantile`, `average`, `ptp` |
| Sorting | `SortingBenchmarks.cs` | `sort`, `argsort`, `searchsorted` |
| LinearAlgebra | `LinAlgBenchmarks.cs` | `dot`, `matmul`, `outer`, `trace` |
| Selection | `WhereBenchmarks.cs` | `np.where`, fancy / boolean selection |
| Reduction (+) | `CumulativeBenchmarks.cs`, `NanReductionBenchmarks.cs` | `cumsum`/`cumprod`; `nansum`/`nanmean`/`nanmax`/… |
| Unary (+) | `UnaryExtraBenchmarks.cs` | `cbrt`, `reciprocal`, `square`, `negative`, `positive`, `trunc` |
| Allocation* | `Allocation/*.cs` | alloc / zero-init micro + size sweeps (*no NumPy twin — not in the official run) |

### Matrix subsystems (appended, not categories)

Beyond the op-matrix categories, the official run appends five subsystems with their own result models (see **Running Benchmarks → Official run**): `nditer` (iterator machinery × tier), `layout` (op × 8 memory layouts × dtype), `operand` (1-D/scalar/mixed/broadcast), `cast` (full 15×15 `astype` × layout), and `fusion` (`np.evaluate`).

---

## Python Benchmark System

### Structure

```python
# Configuration
ARRAY_SIZES = {'small': 1_000, 'medium': 100_000, 'large': 10_000_000}
DTYPES = {'int32': np.int32, 'float64': np.float64, ...}
COMMON_DTYPES = ['int32', 'int64', 'float32', 'float64']

# Benchmark function
def benchmark(func, n, warmup=10, iterations=50) -> BenchmarkResult

# Suite functions (each emits BenchmarkResult rows keyed (op, dtype, n) for the merge)
def run_arithmetic_benchmarks(n, dtype_name, iterations) -> List[BenchmarkResult]
def run_unary_benchmarks(n, dtype_name, iterations) -> List[BenchmarkResult]
def run_unary_extra_benchmarks(n, dtype_name, iterations) -> ...   # cbrt/reciprocal/square/negative/positive/trunc
def run_reduction_benchmarks(n, dtype_name, iterations) -> List[BenchmarkResult]
def run_cumulative_benchmarks(n, dtype_name, iterations) -> ...    # cumsum/cumprod
def run_prod_benchmarks(n, dtype_name, iterations) -> ...
def run_nan_reduction_benchmarks(n, dtype_name, iterations) -> ... # nansum/nanmean/nanmax/…
def run_broadcast_benchmarks(n, iterations) -> List[BenchmarkResult]
def run_creation_benchmarks(n, dtype_name, iterations) -> List[BenchmarkResult]
def run_manipulation_benchmarks(n, iterations) -> List[BenchmarkResult]
def run_slicing_benchmarks(n, iterations) -> List[BenchmarkResult]
def run_comparison_benchmarks(n, dtype_name, iterations) -> ...
def run_bitwise_benchmarks(n, dtype_name, iterations) -> ...
def run_logic_benchmarks(n, dtype_name, iterations) -> ...         # + run_bool_logic_benchmarks(n, iterations)
def run_statistics_benchmarks(n, dtype_name, iterations) -> ...
def run_sorting_benchmarks(n, dtype_name, iterations) -> ...
def run_linalg_benchmarks(n, iterations) -> ...
def run_where_benchmarks(n, iterations) -> ...
# Experimental (no C# comparison wired into run_benchmark.py):
def run_dispatch_benchmarks(n, iterations) -> List[BenchmarkResult]
def run_fusion_benchmarks(n, iterations) -> List[BenchmarkResult]
```

### Command-Line Interface

```bash
cd NumSharp.Benchmark.Python
python numpy_benchmark.py                       # All benchmarks
python numpy_benchmark.py --suite statistics    # Specific suite
python numpy_benchmark.py --cache-sizes         # Sweep 1K/100K/10M in one run (what run_benchmark.py uses)
python numpy_benchmark.py --quick               # 10 iterations
python numpy_benchmark.py --type int32          # Specific dtype
python numpy_benchmark.py --size large          # 10M elements
python numpy_benchmark.py --output results.json # JSON export
```

> `run_benchmark.py` invokes this per suite as `--suite <s> --cache-sizes --output <tmp>`; each result carries its own `n`, which the merge keys on.

### Result Format

```python
@dataclass
class BenchmarkResult:
    name: str          # "np.sum (float64)"
    category: str      # "Sum"
    suite: str         # "Reduction"
    dtype: str         # "float64"
    n: int             # 10000000
    mean_ms: float     # 5.248
    stddev_ms: float   # 0.395
    min_ms: float      # 4.821
    max_ms: float      # 6.134
    iterations: int    # 50
    ops_per_sec: float # 190.55
```

---

## Report Generation

> **The official report is generated by `run_benchmark.py`** (see **Running Benchmarks → Official run**): it merges the C# + NumPy op matrix via `scripts/merge-results.py`, appends the five subsystem sheets, and writes the committable `history/<date>_<sha>/` snapshot. The `run-benchmarks.ps1` flow below is the **legacy Windows-only** path (NumPy-baseline tables, the inverse NS/NPY icon convention) — kept for reference; prefer `run_benchmark.py` for anything new.

### PowerShell Script (`run-benchmarks.ps1`, legacy)

```powershell
# Parameters
-Quick           # Fewer iterations
-Suite           # Specific suite (all, arithmetic, reduction, etc.)
-OutputPath      # Report file path
-SkipCSharp      # Skip C# benchmarks
-SkipPython      # Skip Python benchmarks
-Type            # Specific dtype
-Size            # Array size preset
```

### Report Sections

1. **Environment** - .NET SDK, Python, NumPy versions, CPU
2. **Executive Summary** - Operations tested, suites, array size
3. **NumPy Baseline Performance** - Grouped by suite with tables
4. **NumSharp Results** - BenchmarkDotNet tables (when C# runs)
5. **Performance Comparison Guide** - Legend and interpretation
6. **Key Insights** - Improvement areas and advantages
7. **Reproduction** - Command examples
8. **Type Coverage Matrix** - NumPy dtype to C# type mapping

### Status Icons

> ⚠️ **Convention note.** The **canonical Performance Convention** (project `.claude/CLAUDE.md`)
> is **NPY/NS**: `ratio = NumPy_ms / NumSharp_ms`, **`>1` = NumSharp faster** (higher is better) —
> used by the `nditer` sheet. The legacy `run-benchmarks.ps1`
> table BELOW is the **inverse** (NS/NPY, lower is better). Prefer the canonical NPY/NS direction
> for any new reporting.

Legacy `run-benchmarks.ps1` icons (NS/NPY — NumSharp_ms / NumPy_ms, **lower is better**):

| Ratio (NS/NPY) | Icon | Meaning |
|-------|------|---------|
| ≤ 1.0 | ✅ | NumSharp faster or equal |
| 1.0 - 2.0 | 🟡 | Close to NumPy |
| 2.0 - 5.0 | 🟠 | Slower |
| > 5.0 | 🔴 | Much slower |

---

## Running Benchmarks

### Official run — `run_benchmark.py` (cross-platform, recommended)

`run_benchmark.py` is the single reusable entry point for the official NumSharp-vs-NumPy
comparison. It builds the C# suite, runs each suite through BenchmarkDotNet (per-class JSON,
so it is resumable), sweeps NumPy across the three cache-tier sizes (1K / 100K / 10M), merges,
archives the raw run to `results/<timestamp>/` (gitignored), and finally writes the committable
`history/<date>_<sha>/` snapshot + repoints `history/latest` (see **History snapshots & the
publish ritual** below; `--no-history` opts out).

```bash
python run_benchmark.py                      # full official run, all comparison suites
python run_benchmark.py --suites arithmetic unary
python run_benchmark.py --skip-build         # reuse the existing Release build
python run_benchmark.py --skip-csharp        # NumPy only
python run_benchmark.py --quick              # dev: fewer NumPy iterations
```

The C# side runs under `OfficialBenchmarkConfig` (Infrastructure/BenchmarkConfig.cs):

- **InProcessEmit toolchain** — avoids BenchmarkDotNet's out-of-process project search, which
  fails here ("project names need to be unique") because sibling git worktrees under
  `.claude/worktrees/` contain same-named copies of the benchmark project. In-process also
  matches the warm long-lived Python/NumPy process, so the cross-language ratio is fair.
- **Iteration time capped at 25 ms** with 50 measured iterations. BDN's default Throughput
  strategy ramps to ~8192 invocations/iteration for nanosecond microbenchmarks; for µs–ms
  array ops that made a single 10M case take ~25 s and the full matrix take days. Capping the
  iteration time lets the pilot pick a per-op invocation count that fits 25 ms — fast ops
  still get hundreds–thousands of invocations, slow ops drop to 1/iteration. (~15× faster,
  all 50 iterations preserved.)

The merge keys the join on `(op, dtype, N)` and emits a per-size geomean summary plus the full
per-(op, dtype, N) ratio matrix in `benchmark-report.md`.

**Op coverage** spans comparison, bitwise, logic, NaN-aware reductions, statistics,
sorting/searching, linear algebra, selection (`where`), and unary extras (cbrt/reciprocal/
square/negative/positive/trunc) in addition to the original arithmetic/unary/reduction/
broadcast/creation/manipulation/slicing suites.

After the op matrix, the orchestrator runs the **NDIter iterator benchmark**
(`benchmark/nditer/`, via `nditer_sheet.py` + `nditer_cards.py`) and appends its sheet to
`benchmark-report.md` as its own section (`--skip-nditer` opts out). That harness has a
different result model — *aspect × cache-tier* (construction, traversal, reductions, selection,
dtypes, pathologies, dividends) rather than op/dtype/N — so it is **appended, not merged**: it
measures the iterator machinery the op matrix cannot isolate. It is file-based and
section-isolated (each section runs in its own subprocess); a section that hits NumSharp's known
intermittent AccessViolation across all retries is reported **NA / IGNORED** with a header rather
than crashing the run. See `benchmark/nditer/README.md` for the harness internals. Both the
`.github/workflows/benchmark.yml` post-release workflow and this entry point produce the same
unified report + the two README cards (`cards/ops.png`, `cards/cat.png`).

After NDIter, the orchestrator runs four more **matrix subsystems** that fill axes the
op/dtype/N matrix cannot express, each appended as its own report section (and `--skip-layout`
/ `--skip-operand` / `--skip-cast` / `--skip-fusion` opt out):

| Subsystem | Dir | What it adds | Result model |
|-----------|-----|--------------|--------------|
| **Layout** | `benchmark/layout/` | reduction / copy / elementwise across the 8 layouts (C/F/T/sliced/strided/negrow/negcol/bcast) — the op matrix is C-contiguous only | op × layout × dtype ratio matrix |
| **Operand** | `benchmark/operand/` | 1-D (contig/strided/reversed), scalar operand, mixed operand layouts (C+F, C+T), binary broadcast (row/col) — classes the per-operand grid can't express | case × dtype ratio table |
| **Cast** | `benchmark/cast/` | full `astype` src→dst × 8 layouts at 1M — no op-matrix coverage at all | 15×15 per-layout ratio matrices |
| **Fusion** | `benchmark/fusion/` | `np.evaluate` fused vs unfused np.* chains (+ NumPy context) | fixed-expression report (fenced) |

Each subsystem mirrors NDIter's shape: a NumSharp `*_bench.cs` (fed on stdin via
`dotnet run -c Release -`, the author's absolute `#:project` path rewritten to the running
checkout) + a NumPy `*_bench.py` twin emitting identical keys, merged and rendered by a
`*_sheet.py` to a committed `*_results.md`. The shared build/run/parse plumbing lives in
`benchmark/scripts/bench_common.py`. The convention is **NPY/NS** throughout
(ratio = NumPy_ms / NumSharp_ms, **>1.0 = NumSharp faster**).

### History snapshots & the publish ritual

A full run ends by writing a **committable snapshot** — this is *what we commit and
reference*, distinct from the gitignored raw scratch:

| Path | Tracked? | Contents |
|------|----------|----------|
| `benchmark/results/<timestamp>/` | ❌ gitignored | raw per-run scratch: per-suite NumPy JSONs, BenchmarkDotNet per-class reports, the merged json/csv. Ephemeral. |
| `benchmark/history/<date>_<sha>/` | ✅ tracked | the snapshot: `MANIFEST.md` + `benchmark-report.{md,json,csv}` + `numpy-results.json` + every subsystem `*_results.{md,tsv}` + `cards/`. The json/csv/numpy-results are **gitignored at the benchmark root**, so the snapshot is their only committed home. |
| `benchmark/history/latest` | ✅ tracked symlink | relative symlink (git mode 120000) → the newest snapshot. The stable path for docs/CI: `benchmark/history/latest/benchmark-report.md`. |

`benchmark/scripts/snapshot_history.py` assembles the snapshot, repoints `latest`, and
auto-generates `MANIFEST.md` (provenance, env, methodology, headline geomeans, NDIter/Cast
headlines). `run_benchmark.py` invokes it at the end of every run (skip with `--no-history`):

```bash
python run_benchmark.py                                   # run + write history/<date>_<sha>/ + latest
python benchmark/scripts/snapshot_history.py              # (re)build from newest results/ at HEAD
python benchmark/scripts/snapshot_history.py --commit     # also git-commit the snapshot + latest
python benchmark/scripts/snapshot_history.py \
    --results-dir benchmark/results/<ts> --snap-name <date>_<sha> --head <sha>   # after-the-fact
```

The folder is `<date>_<HEAD-short-sha at snapshot time>` (the benchmarked commit; the MANIFEST
records the dirty/WIP state when the tree isn't clean). Raw BenchmarkDotNet per-class JSON
(~tens of MB) is **not** persisted — regenerable. **Publish ritual:** run → review → commit
`benchmark/history/` together with the rendered root reports. The post-release
`.github/workflows/benchmark.yml` does exactly this (`git add benchmark/history/`) and redeploys
the docs.

### Quick Start (PowerShell, Windows)

```powershell
# Full suite with report
.\run-benchmarks.ps1

# Quick NumPy-only
.\run-benchmarks.ps1 -Quick -SkipCSharp

# Specific suite
.\run-benchmarks.ps1 -Suite arithmetic -Type float64
```

### C# Interactive Menu

```bash
cd NumSharp.Benchmark.CSharp
dotnet run -c Release -f net10.0
```

Menu options:
1. Dispatch Mechanism Comparison
2. Fusion Pattern Benchmarks
3. NumSharp Current Performance
4. DynamicMethod Emission
5. Arithmetic Operations
6. Unary Operations
7. Reduction Operations
8. Broadcasting Operations
9. Array Creation Operations
10. Shape Manipulation
11. Slicing Operations
12. Multi-dimensional Arrays
A. All Benchmarks
Q. Quick smoke test

### C# Command-Line

```bash
# Filter by name pattern
dotnet run -c Release -- --filter "*Add*"

# Specific job type
dotnet run -c Release -- --job Short --filter "*"

# JSON export
dotnet run -c Release -- --exporters json

# Multiple filters
dotnet run -c Release -- --filter "*Arithmetic*,*Reduction*"
```

### Python Standalone

```bash
cd NumSharp.Benchmark.Python

# All with JSON
python numpy_benchmark.py --output ../benchmark-report.json

# Quick arithmetic only
python numpy_benchmark.py --quick --suite arithmetic

# Specific type and size
python numpy_benchmark.py --type float32 --size medium
```

---

## Adding New Benchmarks

### C# Benchmark Template

```csharp
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.YourCategory;

/// <summary>
/// Brief description of what this benchmarks.
/// </summary>
[BenchmarkCategory("YourCategory", "SubCategory")]
public class YourBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    // Array sizes to test
    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    // Types to test (override for custom selection)
    public override IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);  // Different seed
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        GC.Collect();
    }

    [Benchmark(Description = "Operation description")]
    [BenchmarkCategory("SubCategory")]
    public NDArray YourOperation() => np.your_operation(_a);

    [Benchmark(Description = "Another operation")]
    [BenchmarkCategory("AnotherSubCategory")]
    public NDArray AnotherOperation() => _a + _b;
}
```

### Python Benchmark Template

```python
def run_your_benchmarks(n: int, dtype_name: str, iterations: int) -> List[BenchmarkResult]:
    """Benchmark your operations."""
    results = []
    dtype = DTYPES[dtype_name]

    # Setup
    np.random.seed(42)
    a = create_random_array(n, dtype_name, seed=42)
    b = create_random_array(n, dtype_name, seed=43)

    # Benchmark 1
    def your_operation(): return np.your_operation(a)
    r = benchmark(your_operation, n, iterations=iterations)
    r.name = f"np.your_operation ({dtype_name})"
    r.category = "YourCategory"
    r.suite = "YourSuite"
    r.dtype = dtype_name
    results.append(r)

    # Benchmark 2
    def another_operation(): return a + b
    r = benchmark(another_operation, n, iterations=iterations)
    r.name = f"a + b ({dtype_name})"
    r.category = "AnotherCategory"
    r.suite = "YourSuite"
    r.dtype = dtype_name
    results.append(r)

    return results
```

### Adding to main()

```python
# In main():
if args.suite in ["yoursuite", "all"]:
    for dtype in dtypes_to_run:
        results = run_your_benchmarks(args.n, dtype, args.iterations)
        all_results.extend(results)
```

### Updating Program.cs Menu

```csharp
// Add to menu:
Console.WriteLine("13. Your New Benchmarks");

// Add to switch:
"13" => ["--filter", "*YourBenchmarks*"],
```

### Updating run-benchmarks.ps1

```powershell
# Add to $filter switch:
'yoursuite' { "*YourBenchmarks*" }

# Add to ValidateSet:
[ValidateSet('all', ..., 'yoursuite')]
```

---

## Type System

### NumSharp NPTypeCode to NumPy dtype Mapping

| NPTypeCode | C# Type | NumPy dtype | Size (bytes) |
|------------|---------|-------------|--------------|
| Boolean | bool | bool | 1 |
| Byte | byte | uint8 | 1 |
| SByte | sbyte | int8 | 1 |
| Int16 | short | int16 | 2 |
| UInt16 | ushort | uint16 | 2 |
| Int32 | int | int32 | 4 |
| UInt32 | uint | uint32 | 4 |
| Int64 | long | int64 | 8 |
| UInt64 | ulong | uint64 | 8 |
| Char | char | uint16 | 2 |
| Half | System.Half | float16 | 2 |
| Single | float | float32 | 4 |
| Double | double | float64 | 8 |
| Decimal | decimal | float128* | 16 |
| Complex | System.Numerics.Complex | complex128 | 16 |

All **15** NumSharp dtypes are swept by the official op matrix and the cast subsystem.

*Note: NumPy's float128 is platform-dependent and not exactly equivalent to C# decimal; the cast subsystem reports `—` for Decimal cells (no NumPy counterpart).*

### Type Selection Guidelines

| Operation Type | Use Types |
|----------------|-----------|
| Arithmetic (+, -, *, /) | `ArithmeticTypes` |
| Transcendental (sqrt, exp, log, trig) | `TranscendentalTypes` |
| Reduction (sum, mean) | `ArithmeticTypes` |
| Variance/StdDev | `TranscendentalTypes` |
| Comparison | `AllNumericTypes` |
| Quick benchmarks | `CommonTypes` or `MinimalTypes` |

---

## NumSharp API Patterns

### Array Creation

```csharp
// NumSharp requires Shape, not int
np.zeros(new Shape(N), typeCode);           // NOT np.zeros(N, typeCode)
np.ones(new Shape(N), typeCode);
np.empty(new Shape(N), typeCode);
np.full(new Shape(N), value, typeCode);

// Scalar creation
NDArray.Scalar(value, typeCode);            // NOT np.array(value)
```

### Rounding

```csharp
// NumSharp uses np.around or np.round_, NOT np.round
np.around(_a);
np.round_(_a);
```

### Power

```csharp
// NumSharp np.power only accepts ValueType exponents
np.power(_a, 2);                            // OK
np.power(_a, 0.5);                          // OK
np.power(_a, _b);                           // NOT SUPPORTED - use _a * _b
```

### Type Conversion

```csharp
// astype for conversion
_a.astype(NPTypeCode.Float64);
_a.astype(np.float64);

// copy vs view
_a.copy();                                  // Always creates copy
np.copy(_a);                                // Same as above
```

### Slicing Syntax

```csharp
// String-based slicing
_arr["100:1000"];                           // Contiguous slice
_arr["::2"];                                // Every 2nd element
_arr["::-1"];                               // Reversed
_arr["10:100, :"];                          // 2D row slice
_arr[":, 10:100"];                          // 2D column slice (strided)
```

---

## Common Issues & Solutions

### Build Error: Type object has no attribute 'type'

**Python error**: `AttributeError: type object 'numpy.int32' has no attribute 'type'`

**Cause**: Using `dtype.type(5)` instead of `dtype(5)`

**Fix**:
```python
# Wrong
scalar = dtype.type(5)

# Correct
scalar = dtype(5)
```

### Build Error: Cannot reshape array

**Python error**: `ValueError: cannot reshape array of size X into shape (Y,Z)`

**Cause**: Integer division means `rows * cols != n`

**Fix**:
```python
rows = int(np.sqrt(n))
cols = n // rows
actual_n = rows * cols  # May be slightly less than n
arr_1d = np.random.random(actual_n)  # Use actual_n
```

### Build Error: CS8377 - Type must be non-nullable value type

**C# error**: `np.array<T>(params T[])` fails when T is object

**Cause**: `GetScalar` returns `object`, not `ValueType`

**Fix**:
```csharp
// Wrong
_scalar = np.array(GetScalar(DType, 5.0)).astype(DType);

// Correct
_scalar = NDArray.Scalar(GetScalar(DType, 5.0), DType);
```

### Build Error: CS0117 - 'np' does not contain definition for 'round'

**Cause**: NumSharp uses `np.around` or `np.round_`, not `np.round`

**Fix**:
```csharp
// Wrong
np.round(_a);

// Correct
np.around(_a);
// or
np.round_(_a);
```

### Build Error: File is locked

**Error**: `The file is locked by: "NumSharp.Benchmark.CSharp (PID)"`

**Cause**: Previous benchmark run is still executing

**Fix**: Wait for process to complete or kill it manually

### PowerShell Variable Conflict

**Error**: `The value Microsoft.PowerShell.Commands.GroupInfo is not a valid value for the Suite variable`

**Cause**: Local variable `$suite` conflicts with parameter `$Suite`

**Fix**: Rename local variable to `$suiteGroup` or similar

---

## Performance Interpretation

### What the Numbers Mean

| Time Range | Interpretation |
|------------|----------------|
| < 0.1 ms | O(1) operation (view creation, metadata) |
| 0.1 - 1 ms | Small overhead + minimal computation |
| 1 - 10 ms | Cache-efficient operations |
| 10 - 50 ms | Memory-bound operations |
| 50 - 200 ms | Complex operations (trig, var/std) |
| > 200 ms | Multi-pass or unoptimized |

### Memory Bandwidth Expectations

For 10M elements:
- float32 (40 MB) read: ~5 ms theoretical minimum
- float64 (80 MB) read: ~10 ms theoretical minimum
- Operations should be within 2-5x of theoretical minimum

### What Affects Performance

1. **Memory layout**: Contiguous > strided > random access
2. **Cache utilization**: L1 > L2 > L3 > RAM
3. **SIMD**: Vectorized operations are 4-8x faster
4. **Type size**: Smaller types = more elements per cache line
5. **Complexity**: Arithmetic < transcendental < trig

### NumPy Advantages (Why It's Fast)

1. **BLAS/LAPACK**: Highly optimized linear algebra
2. **AVX/SSE**: SIMD vectorization throughout
3. **C implementation**: No GC overhead
4. **Memory pools**: Reduced allocation overhead
5. **Expression templates**: Some kernel fusion

### NumSharp Advantages

1. **.NET integration**: Type safety, tooling, ecosystem
2. **Unmanaged memory**: No GC pressure for large arrays
3. **View semantics**: Zero-copy slicing
4. **Future potential**: SIMD via DynamicMethod emission

---

## CI Integration

### JSON Export

```bash
# Python (from benchmark/ directory)
python NumSharp.Benchmark.Python/numpy_benchmark.py --output benchmark-report.json

# C#
cd NumSharp.Benchmark.CSharp
dotnet run -c Release -- --exporters json
# Results in BenchmarkDotNet.Artifacts/results/*.json
```

### CI Workflow Example

```yaml
name: Benchmarks

on:
  push:
    branches: [master]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: '10.0.x'

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install NumPy
        run: pip install numpy tabulate

      - name: Run NumPy benchmarks
        run: |
          cd benchmark
          python NumSharp.Benchmark.Python/numpy_benchmark.py --quick --output benchmark-report.json

      - name: Run NumSharp benchmarks
        run: |
          cd benchmark/NumSharp.Benchmark.CSharp
          dotnet run -c Release -- --job Short --exporters json

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: |
            benchmark/benchmark-report.json
            benchmark/NumSharp.Benchmark.CSharp/BenchmarkDotNet.Artifacts/
```

### Regression Detection

Compare current results against baseline:

```python
import json

def check_regression(current_file, baseline_file, threshold=1.2):
    """Alert if any operation is >threshold slower than baseline."""
    current = json.load(open(current_file))
    baseline = json.load(open(baseline_file))

    baseline_map = {r['name']: r['mean_ms'] for r in baseline}

    regressions = []
    for r in current:
        if r['name'] in baseline_map:
            ratio = r['mean_ms'] / baseline_map[r['name']]
            if ratio > threshold:
                regressions.append((r['name'], ratio))

    return regressions
```

---

## Quick Reference

### Common Commands

```powershell
# Full benchmark with report
.\run-benchmarks.ps1

# Quick NumPy only
.\run-benchmarks.ps1 -Quick -SkipCSharp

# Specific suite
.\run-benchmarks.ps1 -Suite arithmetic

# C# interactive
cd NumSharp.Benchmark.CSharp && dotnet run -c Release

# C# specific filter
dotnet run -c Release -- --filter "*Sum*" --job Short

# Python specific
python NumSharp.Benchmark.Python/numpy_benchmark.py --suite reduction --type float64 --quick
```

### File Locations

| Item | Location |
|------|----------|
| **Official orchestrator** | `run_benchmark.py` |
| C# benchmarks | `NumSharp.Benchmark.CSharp/Benchmarks/` |
| Infrastructure | `NumSharp.Benchmark.CSharp/Infrastructure/` |
| Python benchmarks | `NumSharp.Benchmark.Python/numpy_benchmark.py` |
| Subsystem sheets | `{nditer,layout,operand,cast,fusion}/*_sheet.py` |
| Helper scripts | `scripts/{merge-results,bench_common,snapshot_history,render_dashboard}.py` |
| Generated report | `benchmark-report.md` (also `README.md`) |
| Committed snapshots | `history/<date>_<sha>/` + `history/latest` (symlink) |
| Results JSON / scratch | `benchmark-report.json`; raw per-run under `results/<ts>/` (gitignored) |
| C# JSON | `NumSharp.Benchmark.CSharp/BenchmarkDotNet.Artifacts/results/` |
| Report generator (legacy) | `run-benchmarks.ps1` |

### Type Shorthand

| Short | Full | NumPy |
|-------|------|-------|
| i32 | Int32 | int32 |
| i64 | Int64 | int64 |
| f32 | Single | float32 |
| f64 | Double | float64 |
| u8 | Byte | uint8 |
| bool | Boolean | bool |

---

*Last updated: 2026-06-27*
*Benchmark suite version: 3.0 — `run_benchmark.py` orchestrator: 14-suite op matrix (15 dtypes × 1K/100K/10M) + five appended subsystems (nditer/layout/operand/cast/fusion) + committable `history/` snapshots; NPY/NS convention.*
