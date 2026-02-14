# NumSharp Benchmark Suite

[![BenchmarkDotNet](https://img.shields.io/badge/benchmark-BenchmarkDotNet%200.14-blue)](https://benchmarkdotnet.org/)
[![.NET](https://img.shields.io/badge/.NET-8.0%20%7C%2010.0-purple)](https://dotnet.microsoft.com/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](../../LICENSE)

Industry-standard performance benchmarks for [NumSharp](https://github.com/SciSharp/NumSharp) using [BenchmarkDotNet](https://benchmarkdotnet.org/). Compare NumSharp operations against NumPy baselines with reproducible, statistically rigorous measurements.

## Key Metrics

| Metric | Coverage |
|--------|----------|
| **Operations** | 130+ benchmarked operations |
| **Data Types** | All 12 NumSharp types |
| **Suites** | 12 benchmark categories |
| **Array Sizes** | Scalar, 100, 1K, 100K, 10M elements |

## Quick Start

```bash
# Navigate to benchmark directory
cd benchmark/NumSharp.Benchmark.GraphEngine

# Build in Release mode (required for accurate benchmarks)
dotnet build -c Release

# Run interactive benchmark menu
dotnet run -c Release -f net10.0

# Or run all benchmarks directly
dotnet run -c Release -f net10.0 -- --filter "*"
```

## Benchmark Suites

### Core Operation Benchmarks

| Suite | Operations | Description |
|-------|------------|-------------|
| **Arithmetic** | `+`, `-`, `*`, `/`, `%` | Binary arithmetic with element-wise and scalar variants |
| **Unary** | `sqrt`, `abs`, `exp`, `log`, `sin`, `cos`, `tan` | Single-input mathematical functions |
| **Reduction** | `sum`, `mean`, `var`, `std`, `min`, `max`, `prod` | Aggregation operations with axis support |
| **Broadcasting** | Scalar, row, column, 3D patterns | Shape broadcasting performance |

### Memory & Layout Benchmarks

| Suite | Operations | Description |
|-------|------------|-------------|
| **Creation** | `zeros`, `ones`, `empty`, `full`, `arange`, `copy` | Array allocation and initialization |
| **Manipulation** | `reshape`, `transpose`, `ravel`, `flatten` | Shape transformation (view vs copy) |
| **Stacking** | `concatenate`, `stack`, `hstack`, `vstack` | Combining multiple arrays |
| **Slicing** | Contiguous, strided, reversed slices | View creation and iteration |

### Research Benchmarks

| Suite | Operations | Description |
|-------|------------|-------------|
| **Dispatch** | Raw ptr, Static, Struct, DynamicMethod | Compares dispatch mechanisms for binary operations |
| **Fusion** | Multi-pass vs fused kernels | Evaluates kernel fusion optimization potential |
| **MultiDim** | 1D vs 2D vs 3D | Same operations across different dimensionalities |
| **DynamicEmission** | Per-operation IL emission | Proposed SIMD optimization via DynamicMethod |

## Running Benchmarks

### Interactive Menu

```bash
dotnet run -c Release -f net10.0
```

```
NumSharp Performance Benchmarks
================================

=== Original Benchmarks ===
1. Dispatch Mechanism Comparison
2. Fusion Pattern Benchmarks
3. NumSharp Current Performance
4. DynamicMethod Emission (#544)

=== Comprehensive Benchmarks ===
5. Arithmetic Operations
6. Unary Operations
7. Reduction Operations
8. Broadcasting Operations
9. Array Creation Operations
10. Shape Manipulation
11. Slicing Operations
12. Multi-dimensional Arrays

=== Meta Options ===
A. All Benchmarks
Q. Quick smoke test (dry run)
```

### Command-Line Options

```bash
# Run specific suite by filter
dotnet run -c Release -- --filter "*Dispatch*"
dotnet run -c Release -- --filter "*Arithmetic*"
dotnet run -c Release -- --filter "*Reduction*,*Sum*"

# Quick smoke test (single iteration)
dotnet run -c Release -- --job Dry

# Short benchmarks (faster, less statistical rigor)
dotnet run -c Release -- --job Short

# Filter by array size
dotnet run -c Release -- --filter "*10000000*"

# Export results
dotnet run -c Release -- --exporters json html markdown

# Compare .NET versions
dotnet run -c Release -f net10.0 -- --runtimes net8.0 net10.0
```

### Common Filter Patterns

| Pattern | Benchmarks |
|---------|------------|
| `*Add*` | All addition benchmarks |
| `*Sum*` | Sum reduction benchmarks |
| `*Arithmetic*` | Add, Subtract, Multiply, Divide, Modulo |
| `*Unary*,*Math*,*Trig*` | All unary operations |
| `*N: 1*` | Scalar benchmarks only (overhead analysis) |
| `*N: 100*` | Tiny array benchmarks only |
| `*10000000*` | Large array (10M elements) benchmarks only |

## Interpreting Results

### Output Columns

| Column | Meaning |
|--------|---------|
| **Mean** | Average execution time |
| **Error** | Half of 99.9% confidence interval |
| **StdDev** | Standard deviation across iterations |
| **Ratio** | Time relative to baseline (1.00 = same speed) |
| **Rank** | Performance ranking (1 = fastest) |
| **Allocated** | Memory allocated per operation |

### Performance Guidelines

| Time Range | Interpretation |
|------------|----------------|
| < 1 μs | Pure metadata (view creation, shape queries) |
| 1 - 10 μs | Scalar operations, minimal dispatch overhead |
| 10 - 100 μs | Small array operations (100 elements) |
| 0.1 - 1 ms | Small overhead + minimal computation |
| 1 - 10 ms | Cache-efficient operations |
| 10 - 50 ms | Memory-bound operations |
| 50 - 200 ms | Complex operations (trig, var/std) |
| > 200 ms | Multi-pass or unoptimized |

### Overhead Analysis (Scalar Benchmarks)

Scalar (N=1) benchmarks isolate fixed costs:
- **Method dispatch**: Virtual calls, interface dispatch
- **Array allocation**: NDArray/Storage creation overhead
- **Safety checks**: Bounds checking, null checks
- **Type resolution**: NPTypeCode switching

A well-optimized operation should show scalar overhead < 10 μs.

### Memory Bandwidth Reference

For 10M elements:
- **float32** (40 MB): ~5 ms theoretical minimum
- **float64** (80 MB): ~10 ms theoretical minimum
- Well-optimized operations should be within 2-5x of theoretical minimum

## Architecture

```
NumSharp.Benchmark.GraphEngine/
├── Program.cs                         # Entry point with interactive menu
├── Infrastructure/
│   ├── BenchmarkBase.cs               # Base class with array creation
│   ├── BenchmarkConfig.cs             # BenchmarkDotNet configurations
│   ├── TypeParameterSource.cs         # NPTypeCode collections
│   └── ArraySizeSource.cs             # Standard array sizes
└── Benchmarks/
    ├── DispatchBenchmarks.cs          # Dispatch mechanism comparison
    ├── FusionBenchmarks.cs            # Kernel fusion patterns
    ├── NumSharpBenchmarks.cs          # NumSharp baseline
    ├── DynamicEmissionBenchmarks.cs   # DynamicMethod per-op
    ├── Arithmetic/                    # +, -, *, /, %
    ├── Unary/                         # sqrt, exp, log, trig
    ├── Reduction/                     # sum, mean, var, min/max
    ├── Broadcasting/                  # Broadcast patterns
    ├── Creation/                      # zeros, ones, empty, full
    ├── Manipulation/                  # reshape, transpose, stack
    ├── Slicing/                       # View and slice operations
    └── MultiDim/                      # 1D vs 2D vs 3D
```

### Type Collections

```csharp
TypeParameterSource.CommonTypes        // int32, int64, float32, float64 (fast benchmarks)
TypeParameterSource.ArithmeticTypes    // All types except bool, char
TypeParameterSource.TranscendentalTypes // float32, float64, decimal (for sqrt, log, trig)
TypeParameterSource.AllNumericTypes    // All 12 NumSharp types
```

### Array Sizes

```csharp
ArraySizeSource.Scalar = 1            // Pure overhead measurement (dispatch, allocation, no loop)
ArraySizeSource.Tiny   = 100          // Common small collections (configs, batches, embeddings)
ArraySizeSource.Small  = 1,000        // L1 cache, per-element overhead
ArraySizeSource.Medium = 100,000      // L2/L3 cache, typical use case
ArraySizeSource.Large  = 10,000,000   // Memory-bound, throughput measurement
```

**Why these sizes matter:**
- **Scalar (1)**: Reveals fixed costs (method dispatch, array allocation, safety checks) with zero loop iterations
- **Tiny (100)**: Represents real-world small data (config arrays, feature vectors, small batches)
- **Small-Large**: Traditional cache-tier progression for throughput analysis

## Adding New Benchmarks

### 1. Create Benchmark Class

```csharp
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.YourCategory;

[BenchmarkCategory("YourCategory")]
public class YourBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;

    // Use StandardSizes for comprehensive coverage (scalar through large)
    // Or pick specific sizes: Scalar, Tiny, Small, Medium, Large
    [Params(ArraySizeSource.Scalar, ArraySizeSource.Tiny, ArraySizeSource.Small,
            ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.CommonTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        GC.Collect();
    }

    [Benchmark(Description = "Your operation")]
    public NDArray YourOperation() => np.your_operation(_a);
}
```

### 2. Add Menu Entry (Program.cs)

```csharp
Console.WriteLine("13. Your New Benchmarks");
// ...
"13" => ["--filter", "*YourBenchmarks*"],
```

### 3. Best Practices

- Use `[GlobalSetup]` for array allocation (not measured)
- Use `[GlobalCleanup]` to release unmanaged memory
- Return results from benchmark methods (prevents dead code elimination)
- Use `CreateRandomArray()` with fixed seed for reproducibility
- Mark baseline with `[Benchmark(Baseline = true)]` for relative comparisons

## Comparison with NumPy

The companion Python benchmarks in `../numpy_benchmark.py` provide NumPy baselines using the same:
- Array sizes (1K, 100K, 10M)
- Random seeds (42)
- Operations

```bash
# Run NumPy benchmarks
python ../NumSharp.Benchmark.Python/numpy_benchmark.py --output ../benchmark-report.json

# Run full comparison suite
pwsh ../run-benchmarks.ps1
```

See `../benchmark-report.md` for the generated comparison report.

## Results Location

After running benchmarks, results are saved to:

```
BenchmarkDotNet.Artifacts/
└── results/
    ├── *-report.html       # Interactive HTML report
    ├── *-report.md         # Markdown tables
    └── *-report.json       # Machine-readable JSON
```

## Contributing

When adding benchmarks:

1. Match existing code style and patterns
2. Use appropriate `TypeParameterSource` for your operation
3. Include both element-wise and scalar variants where applicable
4. Add corresponding Python benchmark in `numpy_benchmark.py`
5. Document any NumSharp API quirks (see `../CLAUDE.md`)

## License

This benchmark suite is part of NumSharp, licensed under [Apache 2.0](../../LICENSE).
