using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Selection;

/// <summary>
/// The public indexing/selection family beyond where(), measured at all universal workload tiers.
/// </summary>
[BenchmarkCategory("Selection", "Indexing")]
public class IndexingSelectionBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;
    private NDArray _mask = null!;
    private NDArray _negativeMask = null!;
    private NDArray _indices = null!;
    private NDArray _selector = null!;
    private NDArray _matrix = null!;
    private NDArray _putTarget = null!;
    private NDArray _placeTarget = null!;
    private NDArray _placeValues = null!;
    private int _side;
    private int WorkN => ArraySizeSource.ResolveMemoryHeavyWorkload(N);

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = np.random.rand(WorkN) * 100 - 50;
        _b = np.random.rand(WorkN) * 100 - 50;
        _mask = _a > 0.0;
        _negativeMask = _a < 0.0;
        _indices = np.arange(0, WorkN, 2);
        _selector = np.random.randint(0, 2, new Shape(WorkN));
        _side = (int)Math.Sqrt(WorkN);
        _matrix = np.arange(_side * _side).reshape(_side, _side);
        _putTarget = _a.copy();
        _placeTarget = _a.copy();
        _placeValues = np.array(new[] { 1.0, 2.0, 3.0 });
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _b = null!;
        _mask = null!;
        _negativeMask = null!;
        _indices = null!;
        _selector = null!;
        _matrix = null!;
        _putTarget = null!;
        _placeTarget = null!;
        _placeValues = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.choose(selector, choices)")] public NDArray Choose() => np.choose(_selector, new[] { _a, _b });
    [Benchmark(Description = "np.compress(mask, a)")] public NDArray Compress() => np.compress(_mask, _a);
    [Benchmark(Description = "np.extract(mask, a)")] public NDArray Extract() => np.extract(_mask, _a);
    [Benchmark(Description = "np.take(a, indices)")] public NDArray Take() => np.take(_a, _indices);
    [Benchmark(Description = "np.take_along_axis(a, indices)")] public NDArray TakeAlongAxis() => np.take_along_axis(_a, _indices, 0);
    [Benchmark(Description = "np.select(condlist, choicelist)")] public NDArray Select() => np.select(new[] { _negativeMask, _mask }, new object[] { _a, _b }, 0.0);
    [Benchmark(Description = "np.put(a, indices, values)")] public void Put() => np.put(_putTarget, _indices, _b["::2"]);
    [Benchmark(Description = "np.place(a, mask, values)")] public void Place() => np.place(_placeTarget, _mask, _placeValues);
    [Benchmark(Description = "np.diag_indices(n)")] public object DiagIndices() => np.diag_indices(_side);
    [Benchmark(Description = "np.diag_indices_from(a)")] public object DiagIndicesFrom() => np.diag_indices_from(_matrix);
    [Benchmark(Description = "np.tril_indices_from(a)")] public object TrilIndicesFrom() => np.tril_indices_from(_matrix);
    [Benchmark(Description = "np.triu_indices_from(a)")] public object TriuIndicesFrom() => np.triu_indices_from(_matrix);
    [Benchmark(Description = "np.mask_indices(n, triu)")] public object MaskIndices() => np.mask_indices(_side, (x, k) => np.triu(x, k));
    [Benchmark(Description = "np.indices(shape)")] public NDArray Indices() => np.indices(new[] { _side, _side });
    [Benchmark(Description = "np.ravel_multi_index(coords, dims)")] public NDArray RavelMultiIndex() => np.ravel_multi_index(new[] { _indices }, new[] { WorkN });
    [Benchmark(Description = "np.unravel_index(indices, shape)")] public object UnravelIndex() => np.unravel_index(_indices, new[] { WorkN });
}
