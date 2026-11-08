#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyinterp_bench.cs — numpy.polynomial chebinterpolate (U9): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyinterp_numpy.py wrote (data/polyinterp/manifest.json), checks dtype, shape and the SHA-256 of
// NumSharp's result against NumPy's BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0x14
//   set OPENBLAS_NUM_THREADS=1
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyinterp_numpy.py > numpy_interp_times.tsv
//   dotnet run -c Release --no-cache polyinterp_bench.cs -- data/polyinterp numpy_interp_times.tsv > ns_interp.tsv
//   python polyinterp_report.py ns_interp.tsv > polyinterp_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit, and PublishAot=false is
// load-bearing (the polynomial kernels are DynamicMethods). np.dot runs through NumSharp.Interop.OpenBLAS at ONE thread —
// the scipy-openblas NumPy itself loads — so the byte check holds on a host whose NumPy loads the same build. The named
// functions below are the twins of polyinterp_numpy.py's FUNCS; both sides time the whole call, the function included.
// =============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text.Json;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

// One thread, like NumPy's run under OPENBLAS_NUM_THREADS=1: np.dot's bits and speed both depend on it.
OpenBlasEngine.Enable(threads: 1);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyinterp");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

// The named callables (polyinterp_numpy.py's FUNCS): the same operations in the same operand order.
var funcs = new Dictionary<string, Func<NDArray, object[], object>>
{
    ["tanh_half"] = (x, _) => np.tanh(x) + 0.5,
    ["stack2"] = (x, _) => np.stack(new[] { np.sin(x), np.cos(x) }, axis: 1),
    // np.exp(1j * x): the complex scalar is the multiply's first operand, as in Python.
    ["expi"] = (x, _) => np.exp(np.multiply(NDArray.Scalar(new Complex(0, 1)), x)),
    ["list"] = (x, _) => Array.ConvertAll(np.sin(x).ToArray<double>(), v => (object)v),
    ["pyfloat"] = (x, _) => 2.0,
    ["f32"] = (x, _) => np.sin(x).astype(np.float32),
    ["i64"] = (x, _) => (x * 10).astype(np.int64),
    // np.sin(a * x) + b over args the runner builds as boxed doubles (a tuple / list) or an ndarray's elements.
    ["sin_ab"] = (x, a) => np.add(np.sin(np.multiply(NDArray.Scalar(Convert.ToDouble(a[0])), x)), NDArray.Scalar(Convert.ToDouble(a[1]))),
};

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the composition's engines (chebpts1, chebvander, np.dot, the ufuncs, the NDScope
// weaving) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the background compiler
// needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDArray>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("calls").GetInt32() >= 2000)
        warmCalls.Add(MakeCall(cell));
for (int pass = 0; pass < 3; pass++)
{
    foreach (var f in warmCalls)
        for (int w = 0; w < 60; w++) f().Dispose();
    System.Threading.Thread.Sleep(250);   // the tier-1 background compiler drains its queue
}

Console.WriteLine("cell\tnumsharp_us\tnumpy_us\tNPY/NS\tcheck");
foreach (var cell in manifest.RootElement.EnumerateArray())
{
    string label = cell.GetProperty("label").GetString();
    int calls = cell.GetProperty("calls").GetInt32(), rounds = cell.GetProperty("rounds").GetInt32();
    Func<NDArray> call = MakeCall(cell);

    var first = call();
    string check = Check(first, cell.GetProperty("expect")[0]);
    first.Dispose();
    for (int w = 0; w < 2; w++) call().Dispose();   // warm: kernel compile + tier-up

    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < rounds; round++)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) call().Dispose();
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

/// <summary>
///     Builds a cell's arguments (once — the timed call only runs the facade and the function) and returns the timed call.
/// </summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call.</returns>
/// <exception cref="NotSupportedException">An unknown argument form.</exception>
Func<NDArray> MakeCall(JsonElement cell)
{
    var f = funcs[cell.GetProperty("fn").GetString()];
    long deg = cell.GetProperty("deg").GetInt64();
    string form = cell.TryGetProperty("deg_form", out var df) ? df.GetString() : null;
    // The degree in the C# boundary map's spelling of the NumPy twin's: a Python int (a long), an np.uint16 scalar (a
    // char) or a 0-d int64 array. Every arm is cast to object.
    object d = form switch
    {
        null => (object)deg,
        "u16" => (object)(char)deg,
        "nd0" => (object)np.array(deg),
        var other => throw new NotSupportedException(other),
    };
    if (!cell.TryGetProperty("args", out var argsSpec))
        return () => np.polynomial.chebyshev.chebinterpolate(f, d);
    double[] values = argsSpec.GetProperty("values").EnumerateArray().Select(e => e.GetDouble()).ToArray();
    // args as the twin built it: a tuple (a ValueTuple), a list (an object[]) or an ndarray.
    object a = argsSpec.GetProperty("kind").GetString() switch
    {
        "tuple" => (object)(values[0], values[1]),
        "list" => (object)Array.ConvertAll(values, v => (object)v),
        "ndarray" => (object)np.array(values),
        var other => throw new NotSupportedException(other),
    };
    return () => np.polynomial.chebyshev.chebinterpolate(f, d, a);
}

/// <summary>"exact" when the result's dtype, shape and SHA-256 of the bytes (logical C order) match NumPy's.</summary>
/// <param name="got">NumSharp's result.</param>
/// <param name="e">NumPy's digest (dtype, shape, sha256).</param>
/// <returns>"exact", or what differs.</returns>
static unsafe string Check(NDArray got, JsonElement e)
{
    if (got.dtype.name != e.GetProperty("dtype").GetString()) return $"DTYPE {got.dtype.name}";
    var shape = e.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
    if (!got.shape.SequenceEqual(shape)) return $"SHAPE ({string.Join(",", got.shape)}) vs ({string.Join(",", shape)})";
    // np.dot's result is a fresh C-contiguous array; any other layout is hashed through a C-order copy.
    NDArray c = got.Shape.IsContiguous ? got : np.ascontiguousarray(got);
    var span = new ReadOnlySpan<byte>((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize, checked((int)(c.size * c.dtypesize)));
    string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
    if (!ReferenceEquals(c, got))
        c.Dispose();
    return hex == e.GetProperty("sha256").GetString() ? "exact" : "BYTES";
}
