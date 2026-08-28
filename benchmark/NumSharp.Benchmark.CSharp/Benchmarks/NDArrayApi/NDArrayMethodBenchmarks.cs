using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.NDArrayApi;

/// <summary>
/// Direct ndarray method surface. These deliberately call the instance entry points instead of
/// crediting their np.* twins: wrapper/dispatch and in-place behavior are part of the public API.
/// </summary>
[BenchmarkCategory("NDArray", "Methods")]
public class NDArrayMethodBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _matrix = null!;
    private NDArray _withSingletons = null!;
    private NDArray _mask = null!;
    private NDArray _indices = null!;
    private NDArray _selector = null!;
    private NDArray _sorted = null!;
    private NDArray _putTarget = null!;
    private NDArray _fillTarget = null!;
    private NDArray _fieldTarget = null!;
    private NDArray _clipMin = null!;
    private NDArray _clipMax = null!;
    private string _toFilePath = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.rand(WorkN) * 100 - 50;
        _b = np.random.rand(WorkN) * 100 - 50;
        int rows = 10;
        _matrix = _a.reshape(rows, WorkN / rows);
        _withSingletons = _a.reshape(1, WorkN, 1);
        _mask = _a > 0.0;
        _indices = np.arange(0, WorkN, 2);
        _selector = np.random.randint(0, 2, new Shape(WorkN));
        _sorted = np.arange(WorkN).astype(np.float64);
        _putTarget = _a.copy();
        _fillTarget = _a.copy();
        _fieldTarget = _a.copy();
        _clipMin = NDArray.Scalar(-10.0);
        _clipMax = NDArray.Scalar(10.0);
        _toFilePath = Path.GetTempFileName();
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _matrix = null!;
        _withSingletons = null!;
        _mask = null!;
        _indices = null!;
        _selector = null!;
        _sorted = null!;
        _putTarget = null!;
        _fillTarget = null!;
        _fieldTarget = null!;
        _clipMin = null!;
        _clipMax = null!;
        if (_toFilePath != null && File.Exists(_toFilePath)) File.Delete(_toFilePath);
        _toFilePath = null!;
        GC.Collect();
    }

    [Benchmark(Description = "a.all() [ndarray]")] public NDArray All() => _a.all();
    [Benchmark(Description = "a.any() [ndarray]")] public NDArray Any() => _a.any();
    [Benchmark(Description = "a.argmax() [ndarray]")] public long ArgMax() => _a.argmax();
    [Benchmark(Description = "a.argmin() [ndarray]")] public long ArgMin() => _a.argmin();
    [Benchmark(Description = "a.cumprod() [ndarray]")] public NDArray CumProd() => _a.cumprod();
    [Benchmark(Description = "a.cumsum() [ndarray]")] public NDArray CumSum() => _a.cumsum();
    [Benchmark(Description = "a.max() [ndarray]")] public NDArray Max() => _a.max(dtype: null);
    [Benchmark(Description = "a.min() [ndarray]")] public NDArray Min() => _a.min(dtype: null);
    [Benchmark(Description = "a.prod() [ndarray]")] public NDArray Prod() => _a.prod();
    [Benchmark(Description = "a.choose(choices) [ndarray]")] public NDArray Choose() => _selector.choose(new[] { _a, _b });
    [Benchmark(Description = "a.clip(min, max) [ndarray]")] public NDArray Clip() => _a.clip(_clipMin, _clipMax);
    [Benchmark(Description = "a.compress(mask) [ndarray]")] public NDArray Compress() => _a.compress(_mask);
    [Benchmark(Description = "a.conj() [ndarray]")] public NDArray Conj() => _a.conj();
    [Benchmark(Description = "a.conjugate() [ndarray]")] public NDArray Conjugate() => _a.conjugate();
    [Benchmark(Description = "a.diagonal() [ndarray]")] public NDArray Diagonal() => _matrix.diagonal();
    [Benchmark(Description = "a.dot(b) [ndarray]")] public NDArray Dot() => _a.dot(_b);
    [Benchmark(Description = "a.fill(value) [ndarray]")] public void Fill() => _fillTarget.fill(3.25);
    [Benchmark(Description = "a.getfield(dtype) [ndarray]")] public NDArray GetField() => _a.getfield(np.float64);
    [Benchmark(Description = "a.item(index) [ndarray]")] public object Item() => _a.item(WorkN / 2);
    [Benchmark(Description = "a.put(indices, values) [ndarray]")] public void Put() => _putTarget.put(_indices, _b["::2"]);
    [Benchmark(Description = "a.repeat(repeats) [ndarray]")] public NDArray Repeat() => _a.repeat(2);
    [Benchmark(Description = "a.reshape(shape) [ndarray]")] public NDArray Reshape() => _a.reshape(10, WorkN / 10);
    [Benchmark(Description = "a.resize(shape) [ndarray]")] public NDArray Resize()
    {
        var copy = _a.copy();
        copy.resize(new Shape(WorkN + 1L), refcheck: false);
        return copy;
    }
    [Benchmark(Description = "a.round() [ndarray]")] public NDArray Round() => _a.round();
    [Benchmark(Description = "a.setfield(value, dtype) [ndarray]")] public void SetField() => _fieldTarget.setfield(1.0, np.float64);
    [Benchmark(Description = "a.setflags(write=True) [ndarray]")] public void SetFlags() => _a.setflags(write: true);
    [Benchmark(Description = "a.squeeze() [ndarray]")] public NDArray Squeeze() => _withSingletons.squeeze();
    [Benchmark(Description = "a.swapaxes(0, 1) [ndarray]")] public NDArray SwapAxes() => _matrix.swapaxes(0, 1);
    [Benchmark(Description = "a.take(indices) [ndarray]")] public NDArray Take() => _a.take(_indices);
    [Benchmark(Description = "a.to_device(cpu) [ndarray]")] public NDArray ToDevice() => _a.to_device("cpu");
    [Benchmark(Description = "a.tobytes() [ndarray]")] public byte[] ToBytes() => _a.tobytes();
    [Benchmark(Description = "a.tofile(path) [ndarray]")] public void ToFile() => _a.tofile(_toFilePath);
    [Benchmark(Description = "a.tolist() [ndarray]")] public object ToList() => _a.tolist();
    [Benchmark(Description = "a.trace() [ndarray]")] public NDArray Trace() => _matrix.trace();
    [Benchmark(Description = "a.transpose() [ndarray]")] public NDArray Transpose() => _matrix.transpose();
    [Benchmark(Description = "a.view() [ndarray]")] public NDArray View() => _a.view();
    [Benchmark(Description = "a.argpartition(kth) [ndarray]")] public NDArray ArgPartition() => _a.argpartition(WorkN / 2);
    [Benchmark(Description = "a.argsort() [ndarray]")] public NDArray ArgSort() => _a.argsort();
    [Benchmark(Description = "a.nonzero() [ndarray]")] public object NonZero() => _a.nonzero();
    [Benchmark(Description = "a.partition(kth) [ndarray]")] public NDArray Partition()
    {
        var copy = _a.copy();
        copy.partition(WorkN / 2);
        return copy;
    }
    [Benchmark(Description = "a.searchsorted(v) [ndarray]")] public NDArray SearchSorted() => _sorted.searchsorted(_a);
    [Benchmark(Description = "a.sort() [ndarray]")] public NDArray Sort()
    {
        var copy = _a.copy();
        copy.sort();
        return copy;
    }
}
