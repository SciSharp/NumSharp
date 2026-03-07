using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Order;
using NumSharp;
using NumSharp.Backends.Kernels;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks;

/// <summary>
/// Benchmarks for SIMD kernel performance analysis.
///
/// Tests four categories:
/// 1. Binary operations (Add, Sub, Mul, Div) - current SIMD performance
/// 2. Full reductions (Sum, Max, Min, Prod, ArgMax, ArgMin, All, Any) - current SIMD performance
/// 3. Axis reductions (sum axis=0, axis=1) - currently scalar, identifies optimization targets
/// 4. Boolean helpers (NonZero, CountTrue, CopyMasked) - SIMD-accelerated boolean operations
///
/// Array sizes: 100 (Tiny), 10K, 1M per task requirements.
/// Dtypes: float64 default; see SimdReductionTypeBenchmarks for int32/float32/float64.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
[GroupBenchmarksBy(BenchmarkDotNet.Configs.BenchmarkLogicalGroupRule.ByCategory)]
public class SimdVsScalarBenchmarks : BenchmarkBase
{
    // Task sizes: 100, 10K, 1M
    private const int Size100 = 100;
    private const int Size10K = 10_000;
    private const int Size1M = 1_000_000;

    private NDArray _a1D = null!;
    private NDArray _b1D = null!;
    private NDArray _a2D = null!;
    private NDArray _boolArray = null!;
    private NDArray _boolMask = null!;  // For CopyMasked benchmark
    private NDArray _prodArray = null!;

    [Params(Size100, Size10K, Size1M)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        // Ensure SIMD is enabled
        ILKernelGenerator.Enabled = true;

        // Create test arrays with float64 (best SIMD support)
        np.random.seed(Seed);
        _a1D = np.random.rand(N) * 100 - 50;
        _b1D = np.random.rand(N) * 100 - 50 + 1;  // +1 to avoid div-by-zero

        // 2D array for axis reductions
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _a2D = np.random.rand(rows * cols).reshape(rows, cols);

        // Boolean array for All/Any tests (mix of true/false, ~50% true)
        _boolArray = (np.random.rand(N) > 0.5).astype(np.@bool);

        // Boolean mask for CopyMasked test (~30% true for realistic filtering)
        _boolMask = (np.random.rand(N) > 0.7).astype(np.@bool);

        // Small values for Prod to avoid overflow
        _prodArray = (np.random.rand(Math.Min(N, 1000)) * 0.5 + 0.5); // Values 0.5-1.0
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a1D = null!;
        _b1D = null!;
        _a2D = null!;
        _boolArray = null!;
        _boolMask = null!;
        _prodArray = null!;
        GC.Collect();
    }

    // ========================================================================
    // BINARY OPERATIONS (SIMD)
    // ========================================================================

    [Benchmark(Description = "Add (SIMD)")]
    [BenchmarkCategory("Binary")]
    public NDArray Add() => _a1D + _b1D;

    [Benchmark(Description = "Subtract (SIMD)")]
    [BenchmarkCategory("Binary")]
    public NDArray Subtract() => _a1D - _b1D;

    [Benchmark(Description = "Multiply (SIMD)")]
    [BenchmarkCategory("Binary")]
    public NDArray Multiply() => _a1D * _b1D;

    [Benchmark(Description = "Divide (SIMD)")]
    [BenchmarkCategory("Binary")]
    public NDArray Divide() => _a1D / _b1D;

    // ========================================================================
    // FULL REDUCTIONS (SIMD)
    // ========================================================================

    [Benchmark(Description = "Sum (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public NDArray Sum() => np.sum(_a1D);

    [Benchmark(Description = "Max (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public NDArray Max() => np.amax(_a1D);

    [Benchmark(Description = "Min (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public NDArray Min() => np.amin(_a1D);

    [Benchmark(Description = "ArgMax (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public int ArgMax() => np.argmax(_a1D);

    [Benchmark(Description = "ArgMin (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public int ArgMin() => np.argmin(_a1D);

    [Benchmark(Description = "All (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public bool All() => (bool)np.all(_boolArray);

    [Benchmark(Description = "Any (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public bool Any() => (bool)np.any(_boolArray);

    [Benchmark(Description = "Prod (SIMD)")]
    [BenchmarkCategory("Reduction_Full")]
    public NDArray Prod() => _prodArray.prod();

    // ========================================================================
    // BOOLEAN HELPERS (SIMD)
    // NonZero, CountTrue (via sum of bool array), CopyMasked (via boolean indexing)
    // ========================================================================

    [Benchmark(Description = "NonZero (SIMD)")]
    [BenchmarkCategory("Boolean_Helpers")]
    public NDArray[] NonZero() => np.nonzero(_boolArray);

    [Benchmark(Description = "CountTrue (sum of bool)")]
    [BenchmarkCategory("Boolean_Helpers")]
    public int CountTrue() => (int)np.sum(_boolArray.astype(np.int32));

    [Benchmark(Description = "CopyMasked (boolean indexing)")]
    [BenchmarkCategory("Boolean_Helpers")]
    public NDArray CopyMasked() => _a1D[_boolMask];

    // ========================================================================
    // AXIS REDUCTIONS: Optimization Targets
    // (Currently scalar - measuring baseline for future SIMD optimization)
    // ========================================================================

    [Benchmark(Description = "Sum axis=0 (scalar)")]
    [BenchmarkCategory("Reduction_Axis")]
    public NDArray Sum_Axis0() => np.sum(_a2D, axis: 0);

    [Benchmark(Description = "Sum axis=1 (scalar)")]
    [BenchmarkCategory("Reduction_Axis")]
    public NDArray Sum_Axis1() => np.sum(_a2D, axis: 1);

    [Benchmark(Description = "Mean axis=0 (scalar)")]
    [BenchmarkCategory("Reduction_Axis")]
    public NDArray Mean_Axis0() => np.mean(_a2D, axis: 0);

    [Benchmark(Description = "Max axis=0 (scalar)")]
    [BenchmarkCategory("Reduction_Axis")]
    public NDArray Max_Axis0() => np.amax(_a2D, axis: 0);
}

/// <summary>
/// Benchmarks for SIMD reduction operations across multiple data types.
/// Tests int32, float32, and float64 to identify type-specific performance characteristics.
/// Array sizes: 100, 10K, 1M per task requirements.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
[GroupBenchmarksBy(BenchmarkDotNet.Configs.BenchmarkLogicalGroupRule.ByCategory)]
public class SimdReductionTypeBenchmarks : BenchmarkBase
{
    // Task sizes: 100, 10K, 1M
    private const int Size100 = 100;
    private const int Size10K = 10_000;
    private const int Size1M = 1_000_000;

    public enum DType { Int32, Float32, Float64 }

    private NDArray _array = null!;

    [Params(Size100, Size10K, Size1M)]
    public override int N { get; set; }

    [Params(DType.Int32, DType.Float32, DType.Float64)]
    public DType Type { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        ILKernelGenerator.Enabled = true;
        np.random.seed(Seed);

        _array = Type switch
        {
            DType.Int32 => np.random.randint(-1000, 1000, new Shape(N)),
            DType.Float32 => (np.random.rand(N) * 100 - 50).astype(np.float32),
            DType.Float64 => np.random.rand(N) * 100 - 50,
            _ => throw new ArgumentException($"Unknown type: {Type}")
        };
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _array = null!;
        GC.Collect();
    }

    [Benchmark(Description = "Sum")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum() => np.sum(_array);

    [Benchmark(Description = "Max")]
    [BenchmarkCategory("Max")]
    public NDArray Max() => np.amax(_array);

    [Benchmark(Description = "Min")]
    [BenchmarkCategory("Min")]
    public NDArray Min() => np.amin(_array);

    [Benchmark(Description = "ArgMax")]
    [BenchmarkCategory("ArgMax")]
    public int ArgMax() => np.argmax(_array);

    [Benchmark(Description = "ArgMin")]
    [BenchmarkCategory("ArgMin")]
    public int ArgMin() => np.argmin(_array);
}

/// <summary>
/// Benchmarks comparing contiguous vs non-contiguous array operations.
/// This shows the SIMD fast-path vs general iterator path difference.
/// Array sizes: 100, 10K, 1M per task requirements.
/// </summary>
[MemoryDiagnoser]
[Orderer(SummaryOrderPolicy.FastestToSlowest)]
[RankColumn]
[CategoriesColumn]
[GroupBenchmarksBy(BenchmarkDotNet.Configs.BenchmarkLogicalGroupRule.ByCategory)]
public class SimdContiguousVsSlicedBenchmarks : BenchmarkBase
{
    // Task sizes: 100, 10K, 1M
    private const int Size100 = 100;
    private const int Size10K = 10_000;
    private const int Size1M = 1_000_000;

    private NDArray _contiguous = null!;
    private NDArray _sliced = null!;       // Every other element
    private NDArray _transposed = null!;   // Column-major access

    [Params(Size100, Size10K, Size1M)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        ILKernelGenerator.Enabled = true;
        np.random.seed(Seed);

        // Contiguous array
        _contiguous = np.random.rand(N);

        // Sliced array (every other element = non-contiguous)
        var full = np.random.rand(N * 2);
        _sliced = full["::2"];

        // Transposed 2D array (column-major access pattern)
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        var matrix = np.random.rand(rows * cols).reshape(rows, cols);
        _transposed = matrix.T;  // Transpose makes it non-contiguous
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _contiguous = null!;
        _sliced = null!;
        _transposed = null!;
        GC.Collect();
    }

    // ========================================================================
    // SUM: Contiguous vs Non-Contiguous
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Sum contiguous (SIMD)")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum_Contiguous() => np.sum(_contiguous);

    [Benchmark(Description = "Sum sliced (iterator)")]
    [BenchmarkCategory("Sum")]
    public NDArray Sum_Sliced() => np.sum(_sliced);

    // ========================================================================
    // ADD: Contiguous vs Non-Contiguous
    // ========================================================================

    [Benchmark(Baseline = true, Description = "Add contiguous (SIMD)")]
    [BenchmarkCategory("Add")]
    public NDArray Add_Contiguous() => _contiguous + _contiguous;

    [Benchmark(Description = "Add sliced (iterator)")]
    [BenchmarkCategory("Add")]
    public NDArray Add_Sliced() => _sliced + _sliced;
}
