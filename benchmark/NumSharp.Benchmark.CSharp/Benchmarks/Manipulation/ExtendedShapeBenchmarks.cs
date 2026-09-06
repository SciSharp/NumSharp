using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Manipulation;

/// <summary>Remaining public joining, splitting, repetition, padding, and rolling APIs.</summary>
[BenchmarkCategory("Manipulation", "ExtendedShape")]
public class ExtendedShapeBenchmarks : OfficialBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _matrix = null!;
    private NDArray _cube = null!;
    private NDArray _row = null!;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = CreateRandomArray(WorkN, DType);
        _b = CreateRandomArray(WorkN, DType, seed: 43);
        _matrix = CreateRandomArray(WorkN, DType).reshape(10, WorkN / 10);
        _cube = CreateRandomArray(WorkN, DType).reshape(10, 10, WorkN / 100);
        _row = np.arange(WorkN / 10).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _matrix = null!;
        _cube = null!;
        _row = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.append(a, b)")] public NDArray Append() => np.append(_a, _b);
    [Benchmark(Description = "np.array_split(a, sections)")] public object ArraySplit() => np.array_split(_a, 7);
    [Benchmark(Description = "np.atleast_1d(a)")] public NDArray AtLeast1D() => np.atleast_1d(_a);
    [Benchmark(Description = "np.atleast_2d(a)")] public NDArray AtLeast2D() => np.atleast_2d(_a);
    [Benchmark(Description = "np.atleast_3d(a)")] public NDArray AtLeast3D() => np.atleast_3d(_a);
    [Benchmark(Description = "np.block([a, b])")] public NDArray Block() => np.block(new object[] { _a, _b });
    [Benchmark(Description = "np.broadcast_arrays(a, b)")] public object BroadcastArrays() => np.broadcast_arrays(_matrix, _row);
    [Benchmark(Description = "np.column_stack((a, b))")] public NDArray ColumnStack() => np.column_stack(_a, _b);
    [Benchmark(Description = "np.concat((a, b))")] public NDArray Concat() => np.concat((_a, _b));
    [Benchmark(Description = "np.delete(a, index)")] public NDArray Delete() => np.delete(_a, WorkN / 2);
    [Benchmark(Description = "np.insert(a, index, value)")] public NDArray Insert() => np.insert(_a, WorkN / 2, 0.0);
    [Benchmark(Description = "np.dsplit(a, sections)")] public object DSplit() => np.dsplit(_cube, 10);
    [Benchmark(Description = "np.hsplit(a, sections)")] public object HSplit() => np.hsplit(_matrix, 10);
    [Benchmark(Description = "np.vsplit(a, sections)")] public object VSplit() => np.vsplit(_matrix, 10);
    [Benchmark(Description = "np.split(a, sections)")] public object Split() => np.split(_a, 10);
    [Benchmark(Description = "np.pad(a, width)")] public NDArray Pad() => np.pad(_a, 16);
    [Benchmark(Description = "np.repeat(a, repeats)")] public NDArray Repeat() => np.repeat(_a, 2);
    [Benchmark(Description = "np.resize(a, shape)")] public NDArray Resize() => np.resize(_a, new Shape(WorkN * 2L));
    [Benchmark(Description = "np.roll(a, shift)")] public NDArray Roll() => np.roll(_a, WorkN / 2);
    [Benchmark(Description = "np.size(a)")] public long Size() => np.size(_a);
    [Benchmark(Description = "np.tile(a, reps)")] public NDArray Tile() => np.tile(_a, 2);
    [Benchmark(Description = "np.unstack(a)")] public object Unstack() => np.unstack(_matrix);
}
