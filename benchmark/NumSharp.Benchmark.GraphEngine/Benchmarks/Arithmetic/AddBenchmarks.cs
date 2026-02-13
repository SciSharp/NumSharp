using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.GraphEngine.Infrastructure;

namespace NumSharp.Benchmark.GraphEngine.Benchmarks.Arithmetic;

/// <summary>
/// Benchmarks for addition operations: a + b, a + scalar, broadcasting.
/// Tests all numeric types and standard array sizes.
/// </summary>
[BenchmarkCategory("Arithmetic", "Add")]
public class AddBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _scalar = null!;
    private NDArray _rowVector = null!;
    private NDArray _colVector = null!;
    private NDArray _matrix = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => TypeParameterSource.ArithmeticTypes;

    [GlobalSetup]
    public void Setup()
    {
        _a = CreateRandomArray(N, DType);
        _b = CreateRandomArray(N, DType, seed: 43);  // Different seed for variety

        // For broadcasting tests
        _scalar = NDArray.Scalar(GetScalar(DType, 5.0), DType);

        // For 2D broadcasting (use sqrt of N for square-ish dimensions)
        var rows = (int)Math.Sqrt(N);
        var cols = N / rows;
        _matrix = CreateRandomArray(rows * cols, DType).reshape(rows, cols);
        _rowVector = CreateRandomArray(cols, DType);
        _colVector = CreateRandomArray(rows, DType).reshape(rows, 1);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        // NDArray uses unmanaged memory, explicit cleanup helps
        _a = null!;
        _b = null!;
        _scalar = null!;
        _rowVector = null!;
        _colVector = null!;
        _matrix = null!;
        GC.Collect();
    }

    // ========================================================================
    // Element-wise Addition
    // ========================================================================

    [Benchmark(Description = "a + b (element-wise)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray Add_Elementwise() => _a + _b;

    [Benchmark(Description = "np.add(a, b)")]
    [BenchmarkCategory("Elementwise")]
    public NDArray NpAdd() => np.add(_a, _b);

    // ========================================================================
    // Scalar Addition (broadcasting)
    // ========================================================================

    [Benchmark(Description = "a + scalar")]
    [BenchmarkCategory("Scalar")]
    public NDArray Add_Scalar() => _a + _scalar;

    [Benchmark(Description = "a + 5 (literal)")]
    [BenchmarkCategory("Scalar")]
    public NDArray Add_ScalarLiteral() => _a + 5;

    // ========================================================================
    // Broadcasting Addition
    // ========================================================================

    [Benchmark(Description = "matrix + row_vector (N,M)+(M,)")]
    [BenchmarkCategory("Broadcast")]
    public NDArray Add_RowBroadcast() => _matrix + _rowVector;

    [Benchmark(Description = "matrix + col_vector (N,M)+(N,1)")]
    [BenchmarkCategory("Broadcast")]
    public NDArray Add_ColBroadcast() => _matrix + _colVector;
}
