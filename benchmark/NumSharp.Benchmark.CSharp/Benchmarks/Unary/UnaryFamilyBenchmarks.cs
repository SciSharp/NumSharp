using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.Unary;

/// <summary>
/// Remaining public unary families: inverse trig, hyperbolic, angle conversion, conjugation,
/// component extraction, and rint/modf. Inputs are domain-shaped exactly like the NumPy twin.
/// </summary>
[BenchmarkCategory("Unary", "Families")]
public class UnaryFamilyBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;
    private NDArray _unit = null!;
    private NDArray _acosh = null!;
    private NDArray _atanh = null!;
    private NDArray _other = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[]
    {
        NPTypeCode.Half,
        NPTypeCode.Single,
        NPTypeCode.Double
    };

    [GlobalSetup]
    public void Setup()
    {
        np.random.seed(Seed);
        _a = ((np.random.rand(N) * 8) - 4).astype(DType);
        _unit = ((np.random.rand(N) * 2) - 1).astype(DType);
        _acosh = ((np.random.rand(N) * 8) + 1).astype(DType);
        _atanh = ((np.random.rand(N) * 1.8) - 0.9).astype(DType);
        _other = ((np.random.rand(N) * 8) - 4).astype(DType);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        _unit = null!;
        _acosh = null!;
        _atanh = null!;
        _other = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.absolute(a)")] public NDArray Absolute() => np.absolute(_a);
    [Benchmark(Description = "np.arcsin(a)")] public NDArray Arcsin() => np.arcsin(_unit);
    [Benchmark(Description = "np.arccos(a)")] public NDArray Arccos() => np.arccos(_unit);
    [Benchmark(Description = "np.arctan(a)")] public NDArray Arctan() => np.arctan(_a);
    [Benchmark(Description = "np.arctan2(a, b)")] public NDArray Arctan2() => np.arctan2(_a, _other);
    [Benchmark(Description = "np.sinh(a)")] public NDArray Sinh() => np.sinh(_unit);
    [Benchmark(Description = "np.cosh(a)")] public NDArray Cosh() => np.cosh(_unit);
    [Benchmark(Description = "np.tanh(a)")] public NDArray Tanh() => np.tanh(_a);
    [Benchmark(Description = "np.arcsinh(a)")] public NDArray Arcsinh() => np.arcsinh(_a);
    [Benchmark(Description = "np.asinh(a)")] public NDArray Asinh() => np.asinh(_a);
    [Benchmark(Description = "np.arccosh(a)")] public NDArray Arccosh() => np.arccosh(_acosh);
    [Benchmark(Description = "np.acosh(a)")] public NDArray Acosh() => np.acosh(_acosh);
    [Benchmark(Description = "np.arctanh(a)")] public NDArray Arctanh() => np.arctanh(_atanh);
    [Benchmark(Description = "np.atanh(a)")] public NDArray Atanh() => np.atanh(_atanh);
    [Benchmark(Description = "np.deg2rad(a)")] public NDArray Deg2Rad() => np.deg2rad(_a);
    [Benchmark(Description = "np.radians(a)")] public NDArray Radians() => np.radians(_a);
    [Benchmark(Description = "np.rad2deg(a)")] public NDArray Rad2Deg() => np.rad2deg(_a);
    [Benchmark(Description = "np.degrees(a)")] public NDArray Degrees() => np.degrees(_a);
    [Benchmark(Description = "np.angle(a)")] public NDArray Angle() => np.angle(_a);
    [Benchmark(Description = "np.conjugate(a)")] public NDArray Conjugate() => np.conjugate(_a);
    [Benchmark(Description = "np.conj(a)")] public NDArray Conj() => np.conj(_a);
    [Benchmark(Description = "np.real(a)")] public NDArray Real() => np.real(_a);
    [Benchmark(Description = "np.imag(a)")] public NDArray Imag() => np.imag(_a);
    [Benchmark(Description = "np.rint(a)")] public NDArray Rint() => np.rint(_a);
    [Benchmark(Description = "np.round(a)")] public NDArray Round() => np.round_(_a);
}

[BenchmarkCategory("Unary", "Modf")]
public class ModfBenchmarks : TypedBenchmarkBase
{
    private NDArray _a = null!;

    [Params(ArraySizeSource.Small, ArraySizeSource.Medium, ArraySizeSource.Large)]
    public override int N { get; set; }

    [ParamsSource(nameof(Types))]
    public new NPTypeCode DType { get; set; }

    public static IEnumerable<NPTypeCode> Types => new[] { NPTypeCode.Single, NPTypeCode.Double };

    [GlobalSetup]
    public void Setup() => _a = CreateRandomArray(N, DType);

    [GlobalCleanup]
    public void Cleanup() { _a = null!; GC.Collect(); }

    [Benchmark(Description = "np.modf(a)")] public (NDArray Fractional, NDArray Integral) Modf() => np.modf(_a);
}
