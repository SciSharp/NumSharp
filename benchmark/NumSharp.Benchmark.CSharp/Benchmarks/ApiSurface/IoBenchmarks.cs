using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.ApiSurface;

[BenchmarkCategory("ApiSurface", "IO")]
public class IoBenchmarks : OfficialBenchmarkBase
{
    private NDArray _a = null!;
    private string _directory = null!;
    private string _npy = null!;
    private string _raw = null!;
    private string _text = null!;
    private string _save = null!;
    private string _saveText = null!;
    private string _saveZip = null!;
    private string _saveZipCompressed = null!;

    [Params(ArraySizeSource.Small)]
    public override int N { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        _a = np.linspace(-100, 100, N).astype(DType);
        _directory = Path.Combine(Path.GetTempPath(), "numsharp-benchmark-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        _npy = Path.Combine(_directory, "input.npy");
        _raw = Path.Combine(_directory, "input.bin");
        _text = Path.Combine(_directory, "input.txt");
        _save = Path.Combine(_directory, "save.npy");
        _saveText = Path.Combine(_directory, "save.txt");
        _saveZip = Path.Combine(_directory, "save.npz");
        _saveZipCompressed = Path.Combine(_directory, "save-compressed.npz");
        if (DType != NPTypeCode.Decimal)
            np.save(_npy, _a);
        _a.tofile(_raw);
        var loadFixtureFormat = DType is
            NPTypeCode.Boolean or NPTypeCode.Byte or NPTypeCode.SByte or NPTypeCode.Int16 or
            NPTypeCode.UInt16 or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or
            NPTypeCode.UInt64 or NPTypeCode.Char
            ? "%d"
            : "%.18e";
        np.savetxt(_text, _a, fmt: loadFixtureFormat);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        if (_directory != null && Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        _directory = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.fromfile(path)")] public NDArray FromFile() => np.fromfile(_raw, DType);
    [Benchmark(Description = "np.loadtxt(path)")] public NDArray LoadTxt() => np.loadtxt(_text, DType);
    [Benchmark(Description = "np.load(path)")] public object Load() => DType != NPTypeCode.Decimal
        ? np.load(_npy)
        : RecordUnsafeUnsupportedDtype();
    [Benchmark(Description = "np.save(path, a)")] public object Save()
    {
        if (DType != NPTypeCode.Decimal)
        {
            np.save(_save, _a);
            return DType;
        }
        return VerifyUnsupportedDtype(() => { np.save(_save, _a); return DType; });
    }
    [Benchmark(Description = "np.savetxt(path, a)")] public void SaveTxt() => np.savetxt(_saveText, _a);
    [Benchmark(Description = "np.savez(path, a)")] public object SaveZ()
    {
        if (DType != NPTypeCode.Decimal)
        {
            np.savez(_saveZip, _a);
            return DType;
        }
        return VerifyUnsupportedDtype(() => { np.savez(_saveZip, _a); return DType; });
    }
    [Benchmark(Description = "np.savez_compressed(path, a)")] public object SaveZCompressed()
    {
        if (DType != NPTypeCode.Decimal)
        {
            np.savez_compressed(_saveZipCompressed, _a);
            return DType;
        }
        return VerifyUnsupportedDtype(() => { np.savez_compressed(_saveZipCompressed, _a); return DType; });
    }
}
