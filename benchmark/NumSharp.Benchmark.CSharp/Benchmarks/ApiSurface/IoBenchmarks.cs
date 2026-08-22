using BenchmarkDotNet.Attributes;
using NumSharp;
using NumSharp.Benchmark.CSharp.Infrastructure;

namespace NumSharp.Benchmark.CSharp.Benchmarks.ApiSurface;

[BenchmarkCategory("ApiSurface", "IO")]
public class IoBenchmarks : BenchmarkBase
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
        _a = np.linspace(-100, 100, N);
        _directory = Path.Combine(Path.GetTempPath(), "numsharp-benchmark-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_directory);
        _npy = Path.Combine(_directory, "input.npy");
        _raw = Path.Combine(_directory, "input.bin");
        _text = Path.Combine(_directory, "input.txt");
        _save = Path.Combine(_directory, "save.npy");
        _saveText = Path.Combine(_directory, "save.txt");
        _saveZip = Path.Combine(_directory, "save.npz");
        _saveZipCompressed = Path.Combine(_directory, "save-compressed.npz");
        np.save(_npy, _a);
        _a.tofile(_raw);
        np.savetxt(_text, _a);
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _a = null!;
        if (_directory != null && Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        _directory = null!;
        GC.Collect();
    }

    [Benchmark(Description = "np.fromfile(path)")] public NDArray FromFile() => np.fromfile(_raw, np.float64);
    [Benchmark(Description = "np.loadtxt(path)")] public NDArray LoadTxt() => np.loadtxt(_text);
    [Benchmark(Description = "np.load(path)")] public object Load() => np.load(_npy);
    [Benchmark(Description = "np.save(path, a)")] public void Save() => np.save(_save, _a);
    [Benchmark(Description = "np.savetxt(path, a)")] public void SaveTxt() => np.savetxt(_saveText, _a);
    [Benchmark(Description = "np.savez(path, a)")] public void SaveZ() => np.savez(_saveZip, _a);
    [Benchmark(Description = "np.savez_compressed(path, a)")] public void SaveZCompressed() => np.savez_compressed(_saveZipCompressed, _a);
}
