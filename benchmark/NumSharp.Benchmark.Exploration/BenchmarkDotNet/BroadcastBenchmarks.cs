using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Diagnosers;
using NumSharp.Benchmark.Exploration.Infrastructure;

namespace NumSharp.Benchmark.Exploration.BenchmarkDotNet;

/// <summary>
/// BenchmarkDotNet validation benchmarks for broadcasting scenarios.
/// Provides statistically rigorous validation of Stopwatch results.
/// </summary>
[MemoryDiagnoser]
[DisassemblyDiagnoser(maxDepth: 3)]
public unsafe class BroadcastBenchmarks
{
    private double* _lhs = null!;
    private double* _rhs = null!;
    private double* _result = null!;

    [Params(1_000, 100_000, 1_000_000, 10_000_000)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _lhs = SimdImplementations.AllocateAligned<double>(Size);
        _rhs = SimdImplementations.AllocateAligned<double>(Size);
        _result = SimdImplementations.AllocateAligned<double>(Size);

        SimdImplementations.FillRandom(_lhs, Size, 42);
        SimdImplementations.FillRandom(_rhs, Size, 43);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        SimdImplementations.FreeAligned(_lhs);
        SimdImplementations.FreeAligned(_rhs);
        SimdImplementations.FreeAligned(_result);
    }

    [Benchmark(Baseline = true, Description = "Scalar loop")]
    public void ScalarLoop()
    {
        SimdImplementations.AddScalarLoop_Float64(_lhs, _rhs, _result, Size);
    }

    [Benchmark(Description = "SIMD Vector256")]
    public void SimdFull()
    {
        SimdImplementations.AddFull_Float64(_lhs, _rhs, _result, Size);
    }
}

/// <summary>
/// BenchmarkDotNet benchmarks for scalar broadcast (S2).
/// </summary>
[MemoryDiagnoser]
public unsafe class ScalarBroadcastBenchmarks
{
    private double* _lhs = null!;
    private double* _result = null!;
    private double _scalar = 42.5;

    [Params(1_000, 100_000, 1_000_000, 10_000_000)]
    public int Size { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _lhs = SimdImplementations.AllocateAligned<double>(Size);
        _result = SimdImplementations.AllocateAligned<double>(Size);
        SimdImplementations.FillRandom(_lhs, Size, 42);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        SimdImplementations.FreeAligned(_lhs);
        SimdImplementations.FreeAligned(_result);
    }

    [Benchmark(Baseline = true, Description = "Scalar loop")]
    public void ScalarLoop()
    {
        for (int i = 0; i < Size; i++)
        {
            _result[i] = _lhs[i] + _scalar;
        }
    }

    [Benchmark(Description = "SIMD Vector256")]
    public void SimdScalar()
    {
        SimdImplementations.AddScalar_Float64(_lhs, _scalar, _result, Size);
    }
}

/// <summary>
/// BenchmarkDotNet benchmarks for row broadcast (S4).
/// </summary>
[MemoryDiagnoser]
public unsafe class RowBroadcastBenchmarks
{
    private double* _matrix = null!;
    private double* _row = null!;
    private double* _result = null!;
    private int _rows;
    private int _cols;

    [Params(100, 316, 1000, 3162)]
    public int MatrixDim { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _rows = MatrixDim;
        _cols = MatrixDim;
        int size = _rows * _cols;

        _matrix = SimdImplementations.AllocateAligned<double>(size);
        _row = SimdImplementations.AllocateAligned<double>(_cols);
        _result = SimdImplementations.AllocateAligned<double>(size);

        SimdImplementations.FillRandom(_matrix, size, 42);
        SimdImplementations.FillRandom(_row, _cols, 43);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        SimdImplementations.FreeAligned(_matrix);
        SimdImplementations.FreeAligned(_row);
        SimdImplementations.FreeAligned(_result);
    }

    [Benchmark(Baseline = true, Description = "Scalar nested loops")]
    public void ScalarLoop()
    {
        SimdImplementations.AddRowBroadcastScalar_Float64(_matrix, _row, _result, _rows, _cols);
    }

    [Benchmark(Description = "SIMD-CHUNK (SIMD inner)")]
    public void SimdChunk()
    {
        SimdImplementations.AddRowBroadcast_Float64(_matrix, _row, _result, _rows, _cols);
    }
}

/// <summary>
/// BenchmarkDotNet benchmarks for different dtypes.
/// </summary>
[MemoryDiagnoser]
public unsafe class DtypeBenchmarks
{
    private const int Size = 1_000_000;

    private byte* _byteLhs = null!;
    private byte* _byteRhs = null!;
    private byte* _byteResult = null!;

    private short* _shortLhs = null!;
    private short* _shortRhs = null!;
    private short* _shortResult = null!;

    private int* _intLhs = null!;
    private int* _intRhs = null!;
    private int* _intResult = null!;

    private long* _longLhs = null!;
    private long* _longRhs = null!;
    private long* _longResult = null!;

    private float* _floatLhs = null!;
    private float* _floatRhs = null!;
    private float* _floatResult = null!;

    private double* _doubleLhs = null!;
    private double* _doubleRhs = null!;
    private double* _doubleResult = null!;

    [GlobalSetup]
    public void Setup()
    {
        _byteLhs = SimdImplementations.AllocateAligned<byte>(Size);
        _byteRhs = SimdImplementations.AllocateAligned<byte>(Size);
        _byteResult = SimdImplementations.AllocateAligned<byte>(Size);
        SimdImplementations.FillRandom(_byteLhs, Size, 42);
        SimdImplementations.FillRandom(_byteRhs, Size, 43);

        _shortLhs = SimdImplementations.AllocateAligned<short>(Size);
        _shortRhs = SimdImplementations.AllocateAligned<short>(Size);
        _shortResult = SimdImplementations.AllocateAligned<short>(Size);
        SimdImplementations.FillRandom(_shortLhs, Size, 42);
        SimdImplementations.FillRandom(_shortRhs, Size, 43);

        _intLhs = SimdImplementations.AllocateAligned<int>(Size);
        _intRhs = SimdImplementations.AllocateAligned<int>(Size);
        _intResult = SimdImplementations.AllocateAligned<int>(Size);
        SimdImplementations.FillRandom(_intLhs, Size, 42);
        SimdImplementations.FillRandom(_intRhs, Size, 43);

        _longLhs = SimdImplementations.AllocateAligned<long>(Size);
        _longRhs = SimdImplementations.AllocateAligned<long>(Size);
        _longResult = SimdImplementations.AllocateAligned<long>(Size);
        SimdImplementations.FillRandom(_longLhs, Size, 42);
        SimdImplementations.FillRandom(_longRhs, Size, 43);

        _floatLhs = SimdImplementations.AllocateAligned<float>(Size);
        _floatRhs = SimdImplementations.AllocateAligned<float>(Size);
        _floatResult = SimdImplementations.AllocateAligned<float>(Size);
        SimdImplementations.FillRandom(_floatLhs, Size, 42);
        SimdImplementations.FillRandom(_floatRhs, Size, 43);

        _doubleLhs = SimdImplementations.AllocateAligned<double>(Size);
        _doubleRhs = SimdImplementations.AllocateAligned<double>(Size);
        _doubleResult = SimdImplementations.AllocateAligned<double>(Size);
        SimdImplementations.FillRandom(_doubleLhs, Size, 42);
        SimdImplementations.FillRandom(_doubleRhs, Size, 43);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        SimdImplementations.FreeAligned(_byteLhs);
        SimdImplementations.FreeAligned(_byteRhs);
        SimdImplementations.FreeAligned(_byteResult);
        SimdImplementations.FreeAligned(_shortLhs);
        SimdImplementations.FreeAligned(_shortRhs);
        SimdImplementations.FreeAligned(_shortResult);
        SimdImplementations.FreeAligned(_intLhs);
        SimdImplementations.FreeAligned(_intRhs);
        SimdImplementations.FreeAligned(_intResult);
        SimdImplementations.FreeAligned(_longLhs);
        SimdImplementations.FreeAligned(_longRhs);
        SimdImplementations.FreeAligned(_longResult);
        SimdImplementations.FreeAligned(_floatLhs);
        SimdImplementations.FreeAligned(_floatRhs);
        SimdImplementations.FreeAligned(_floatResult);
        SimdImplementations.FreeAligned(_doubleLhs);
        SimdImplementations.FreeAligned(_doubleRhs);
        SimdImplementations.FreeAligned(_doubleResult);
    }

    [Benchmark(Description = "byte SIMD")]
    public void ByteSimd() => SimdImplementations.AddFull_Byte(_byteLhs, _byteRhs, _byteResult, Size);

    [Benchmark(Description = "int16 SIMD")]
    public void Int16Simd() => SimdImplementations.AddFull_Int16(_shortLhs, _shortRhs, _shortResult, Size);

    [Benchmark(Description = "int32 SIMD")]
    public void Int32Simd() => SimdImplementations.AddFull_Int32(_intLhs, _intRhs, _intResult, Size);

    [Benchmark(Description = "int64 SIMD")]
    public void Int64Simd() => SimdImplementations.AddFull_Int64(_longLhs, _longRhs, _longResult, Size);

    [Benchmark(Description = "float32 SIMD")]
    public void Float32Simd() => SimdImplementations.AddFull_Float32(_floatLhs, _floatRhs, _floatResult, Size);

    [Benchmark(Description = "float64 SIMD")]
    public void Float64Simd() => SimdImplementations.AddFull_Float64(_doubleLhs, _doubleRhs, _doubleResult, Size);
}
