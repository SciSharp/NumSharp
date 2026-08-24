using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.ApiSurface;

[BenchmarkCategory("ApiSurface", "DType")]
public class DTypeUtilityBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _b = null!;

    // Scalar (N=1) is the pure dispatch/call-overhead tier ("scalar x scalar"); Small is the
    // standard size. These ops are dtype/metadata only, so the two sizes isolate call cost.
    [Params(ArraySizeSource.Scalar, ArraySizeSource.Small)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _a = np.arange(N).astype(np.int32);
        _b = np.arange(N).astype(np.float64);
    }

    [GlobalCleanup]
    public void Cleanup() { _a = null!; _b = null!; GC.Collect(); }

    [Benchmark(Description = "np.can_cast(from, to)")] public bool CanCast() => np.can_cast(np.int32, np.float64);
    [Benchmark(Description = "np.common_type(a, b)")] public Type CommonType() => np.common_type(_a, _b);
    [Benchmark(Description = "np.isdtype(dtype, kind)")] public bool IsDType() => np.isdtype(np.float64, "real floating");
    [Benchmark(Description = "np.issubdtype(dtype, kind)")] public bool IsSubDType() => np.issubdtype(np.float64, "floating");
    [Benchmark(Description = "np.min_scalar_type(value)")] public NPTypeCode MinScalarType() => np.min_scalar_type(127);
    [Benchmark(Description = "np.mintypecode(typechars)")] public char MinTypeCode() => np.mintypecode("fd");
    [Benchmark(Description = "np.promote_types(a, b)")] public NPTypeCode PromoteTypes() => np.promote_types(np.int32, np.float64);
    [Benchmark(Description = "np.result_type(a, b)")] public NPTypeCode ResultType() => np.result_type(_a, _b);
    [Benchmark(Description = "np.iterable(a)")] public bool Iterable() => np.iterable(_a);
    [Benchmark(Description = "np.copyto(dst, src)")] public void CopyTo() => np.copyto(_b, _a, casting: "unsafe");
    [Benchmark(Description = "np.nested_iters(a, axes)")] public void NestedIters()
    {
        int mrows = N >= 10 ? 10 : 1;   // keep the reshape valid at the N=1 tier
        var iterators = np.nested_iters(_b.reshape(mrows, N / mrows), new[] { new[] { 0 }, new[] { 1 } });
        foreach (var iterator in iterators) iterator.Dispose();
    }
}

[BenchmarkCategory("ApiSurface", "Text")]
public class TextFormattingBenchmarks : BenchmarkBase
{
    private NDArray _a = null!;

    // Scalar (N=1) = pure dispatch overhead for the format/print ops; Small is the standard size.
    [Params(ArraySizeSource.Scalar, ArraySizeSource.Small)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup() => _a = np.linspace(-100, 100, N);

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.array_repr(a)")] public string ArrayRepr() => np.array_repr(_a);
    [Benchmark(Description = "np.array_str(a)")] public string ArrayStr() => np.array_str(_a);
    [Benchmark(Description = "np.array2string(a)")] public string Array2String() => np.array2string(_a);
    [Benchmark(Description = "np.format_float_positional(x)")] public string FormatPositional() => np.format_float_positional(Math.PI, precision: 12);
    [Benchmark(Description = "np.format_float_scientific(x)")] public string FormatScientific() => np.format_float_scientific(Math.PI, precision: 12);
    [Benchmark(Description = "np.get_printoptions()")] public object GetPrintOptions() => np.get_printoptions();
    [Benchmark(Description = "np.printoptions()")] public void PrintOptions()
    {
        using var _ = np.printoptions(precision: 6);
    }
    [Benchmark(Description = "np.set_printoptions()")] public void SetPrintOptions() => np.set_printoptions(precision: 6);
}
