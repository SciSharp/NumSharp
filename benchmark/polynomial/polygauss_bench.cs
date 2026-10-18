#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polygauss_bench.cs — numpy.polynomial Gauss-Legendre / Gauss-Laguerre quadrature (U8: leggauss / laggauss):
// NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polygauss_numpy.py wrote (data/polygauss/manifest.json), checks dtype, shape and the SHA-256 of
// NumSharp's two result slots against NumPy's BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0x14
//   set OPENBLAS_NUM_THREADS=1
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polygauss_numpy.py > numpy_gauss_times.tsv
//   dotnet run -c Release --no-cache polygauss_bench.cs -- data/polygauss numpy_gauss_times.tsv > ns_gauss.tsv
//   python polygauss_report.py ns_gauss.tsv > polygauss_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit, and PublishAot=false is
// load-bearing (the polynomial kernels are DynamicMethods). The eigenvalues come from NumSharp.Interop.OpenBLAS at ONE
// thread — the scipy-openblas NumPy itself loads — so the byte check holds on a host whose NumPy loads the same build.
// =============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

// One thread, like NumPy's run under OPENBLAS_NUM_THREADS=1: eigvalsh's bits and speed both depend on it.
OpenBlasEngine.Enable(threads: 1);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polygauss");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

// The one facade overload of each function (an object argument), returning its two arrays.
var funcs = new Dictionary<string, Func<object, NDArray[]>>
{
    ["leggauss"] = v =>
    {
        var (x, w) = np.polynomial.legendre.leggauss(v);
        return new[] { x, w };
    },
    ["laggauss"] = v =>
    {
        var (x, w) = np.polynomial.laguerre.laggauss(v);
        return new[] { x, w };
    },
};

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the composition's engines (companion, eigvalsh, {p}val, {p}der, the ufuncs and
// reductions, the NDScope weaving) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1;
// the background compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDArray[]>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("calls").GetInt32() >= 1000)
        warmCalls.Add(MakeCall(cell));
for (int pass = 0; pass < 3; pass++)
{
    foreach (var f in warmCalls)
        for (int w = 0; w < 60; w++) Dispose(f());
    System.Threading.Thread.Sleep(250);   // the tier-1 background compiler drains its queue
}

Console.WriteLine("cell\tnumsharp_us\tnumpy_us\tNPY/NS\tcheck");
foreach (var cell in manifest.RootElement.EnumerateArray())
{
    string label = cell.GetProperty("label").GetString();
    int calls = cell.GetProperty("calls").GetInt32(), rounds = cell.GetProperty("rounds").GetInt32();
    Func<NDArray[]> call = MakeCall(cell);

    var first = call();
    string check = Check(first, cell.GetProperty("expect"));
    Dispose(first);
    for (int w = 0; w < 2; w++) Dispose(call());   // warm: kernel compile + tier-up

    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < rounds; round++)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) Dispose(call());
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / calls);
    }
    double npv = numpyUs.TryGetValue(label, out var v) ? v : double.NaN;
    rows.Add((label, best, npv, check));
    Console.WriteLine($"{label}\t{best:F3}\t{npv:F3}\t{npv / best:F2}\t{check}");
}

// Summaries: per section min and geomean of NPY/NS, then every cell under 1.5x.
Console.WriteLine();
Console.WriteLine("section\tcells\tmin NPY/NS\tgeomean NPY/NS\tmismatches");
foreach (var g in rows.GroupBy(x => x.Label.Split('/')[0]).OrderBy(g => g.Key))
{
    var ratios = g.Where(x => !double.IsNaN(x.Np)).Select(x => x.Np / x.Ns).ToArray();
    double gm = ratios.Length == 0 ? double.NaN : Math.Exp(ratios.Average(Math.Log));
    Console.WriteLine($"{g.Key}\t{g.Count()}\t{(ratios.Length == 0 ? double.NaN : ratios.Min()):F2}\t{gm:F2}\t{g.Count(x => x.Check != "exact")}");
}
var all = rows.Where(x => !double.IsNaN(x.Np)).Select(x => x.Np / x.Ns).ToArray();
Console.WriteLine($"ALL\t{rows.Count}\t{all.Min():F2}\t{Math.Exp(all.Average(Math.Log)):F2}\t{rows.Count(x => x.Check != "exact")}");
Console.WriteLine();
Console.WriteLine("cells below 1.5x:");
foreach (var x in rows.Where(x => x.Np / x.Ns < 1.5).OrderBy(x => x.Np / x.Ns))
    Console.WriteLine($"  {x.Label}\t{x.Np / x.Ns:F2}\t(ns {x.Ns:F2} us, numpy {x.Np:F2} us)");

/// <summary>Builds a cell's argument (once — the timed call only runs the facade) and returns the timed call.</summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call.</returns>
/// <exception cref="NotSupportedException">An unknown argument form.</exception>
Func<NDArray[]> MakeCall(JsonElement cell)
{
    string fn = cell.GetProperty("fn").GetString();
    long deg = cell.GetProperty("deg").GetInt64();
    string form = cell.TryGetProperty("form", out var f) ? f.GetString() : null;
    // The degree in the C# boundary map's spelling of the NumPy twin's: a Python int (a long), an np.uint16 scalar (a
    // char) or a Python bool. Every arm is cast to object.
    object x = form switch
    {
        null => (object)deg,
        "u16" => (object)(char)deg,
        "bool" => (object)(deg != 0),
        var other => throw new NotSupportedException(other),
    };
    var call = funcs[fn];
    return () => call(x);
}

/// <summary>Disposes every array of one call's result.</summary>
/// <param name="arrays">The result arrays.</param>
static void Dispose(NDArray[] arrays)
{
    foreach (var a in arrays)
        a.Dispose();
}

/// <summary>"exact" when every slot's dtype, shape and SHA-256 of the bytes (logical C order) match NumPy's.</summary>
/// <param name="got">NumSharp's result arrays.</param>
/// <param name="expect">NumPy's digests, one per slot (dtype, shape, sha256).</param>
/// <returns>"exact", or what differs.</returns>
static unsafe string Check(NDArray[] got, JsonElement expect)
{
    var slots = expect.EnumerateArray().ToArray();
    if (slots.Length != got.Length) return $"ARITY {got.Length} vs {slots.Length}";
    for (int s = 0; s < slots.Length; s++)
    {
        var e = slots[s];
        var a = got[s];
        if (a.dtype.name != e.GetProperty("dtype").GetString()) return $"DTYPE[{s}] {a.dtype.name}";
        var shape = e.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
        if (!a.shape.SequenceEqual(shape)) return $"SHAPE[{s}] ({string.Join(",", a.shape)}) vs ({string.Join(",", shape)})";
        var span = new ReadOnlySpan<byte>((byte*)a.Storage.Address + a.Shape.offset * a.dtypesize, checked((int)(a.size * a.dtypesize)));
        string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
        if (hex != e.GetProperty("sha256").GetString()) return $"BYTES[{s}]";
    }
    return "exact";
}
