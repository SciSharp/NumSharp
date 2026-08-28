using System.Globalization;
using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Creation;

/// <summary>
/// Array conversion and structured creation APIs. Uses float64 across the universal
/// 1K / 100K / 10M workload tiers.
/// </summary>
[BenchmarkCategory("Creation", "Conversion")]
public class ConversionCreationBenchmarks : BenchmarkBase
{
    private NDArray _source = null!;
    private NDArray _matrix = null!;
    private NDArray _strided = null!;
    private NDArray _axis = null!;
    private byte[] _bytes = null!;
    private string _text = null!;
    private int _side;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _source = np.random.rand(WorkN) * 100;
        _side = (int)Math.Sqrt(WorkN);
        _matrix = np.random.rand(_side * _side).reshape(_side, _side);
        _strided = _source["::2"];
        _axis = np.arange(_side).astype(np.float64);
        _bytes = new byte[checked(WorkN * sizeof(double))];
        _text = string.Join(" ", Enumerable.Range(0, WorkN)
            .Select(i => (i % 100).ToString(CultureInfo.InvariantCulture)));
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _source = null!;
        _matrix = null!;
        _strided = null!;
        _axis = null!;
        _bytes = null!;
        _text = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.array(a)")] public NDArray Array() => np.array(_source);
    [Benchmark(Description = "np.asarray(a)")] public NDArray AsArray() => np.asarray(_source);
    [Benchmark(Description = "np.asanyarray(a)")] public NDArray AsAnyArray() => np.asanyarray(_source);
    [Benchmark(Description = "np.asarray_chkfinite(a)")] public NDArray AsArrayCheckFinite() => np.asarray_chkfinite(_source);
    [Benchmark(Description = "np.ascontiguousarray(a)")] public NDArray AsContiguousArray() => np.ascontiguousarray(_strided);
    [Benchmark(Description = "np.asfortranarray(a)")] public NDArray AsFortranArray() => np.asfortranarray(_matrix);
    [Benchmark(Description = "np.asmatrix(a)")] public NDArray AsMatrix() => np.asmatrix(_source);
    [Benchmark(Description = "np.require(a, requirements=C)")] public NDArray Require() => np.require(_strided, requirements: new[] { "C" });
    [Benchmark(Description = "np.frombuffer(buffer)")] public NDArray FromBuffer() => np.frombuffer(_bytes, np.float64);
    [Benchmark(Description = "np.fromstring(text)")] public NDArray FromString() => np.fromstring(_text, np.float64, sep: " ");
    [Benchmark(Description = "np.eye(n)")] public NDArray Eye() => np.eye(_side);
    [Benchmark(Description = "np.identity(n)")] public NDArray Identity() => np.identity(_side);
    [Benchmark(Description = "np.meshgrid(x, y)")] public np.MeshgridResult Meshgrid() => np.meshgrid(_axis, _axis);
    [Benchmark(Description = "np.vander(x)")] public NDArray Vander() => np.vander(_axis);
}
