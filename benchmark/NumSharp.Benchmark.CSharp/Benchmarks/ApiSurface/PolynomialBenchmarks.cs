using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.ApiSurface;

[BenchmarkCategory("ApiSurface", "Polynomial")]
public class PolynomialBenchmarks : OfficialBenchmarkBase
{
    private NDArray _p = null!;
    private NDArray _q = null!;
    private NDArray _roots = null!;
    private NDArray _x = null!;

    [Params(ArraySizeSource.Small)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _p = CreateRandomArray(17, DType);
        _q = CreateRandomArray(9, DType, seed: 43);
        _roots = np.linspace(-2, 2, 16).astype(DType);
        _x = np.linspace(-1, 1, N).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup() { _p = null!; _q = null!; _roots = null!; _x = null!; GC.Collect(); }

    [Benchmark(Description = "np.poly(roots)")] public NDArray Poly() => np.poly(_roots);
    [Benchmark(Description = "np.polyadd(p, q)")] public NDArray PolyAdd() => np.polyadd(_p, _q);
    [Benchmark(Description = "np.polyder(p)")] public NDArray PolyDer() => np.polyder(_p);
    [Benchmark(Description = "np.polydiv(p, q)")] public object PolyDiv() => np.polydiv(_p, _q);
    [Benchmark(Description = "np.polyint(p)")] public NDArray PolyInt() => np.polyint(_p);
    [Benchmark(Description = "np.polymul(p, q)")] public NDArray PolyMul() => np.polymul(_p, _q);
    [Benchmark(Description = "np.polysub(p, q)")] public object PolySub() => DType != NPTypeCode.Boolean
        ? np.polysub(_p, _q)
        : VerifyUnsupportedDtype(() => np.polysub(_p, _q));
    [Benchmark(Description = "np.polyval(p, x)")] public NDArray PolyVal() => np.polyval(_p, _x);
}
