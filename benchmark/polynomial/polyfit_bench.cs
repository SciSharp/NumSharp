#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyfit_bench.cs — numpy.polynomial least-squares fits (U6: {p}fit): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyfit_numpy.py wrote (data/polyfit/manifest.json over data/polyfit/inputs.npz), checks dtype,
// shape and the SHA-256 of every result slot against NumPy's BEFORE timing it, then prints a TSV joined with NumPy's
// timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), NumSharp lstsq us, NumPy lstsq us, ceiling, byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x. "lstsq" is the cell's
// np.linalg.lstsq call alone, on the operands NumPy's _fit hands it (<key>_A / <key>_b, the manifest's rcond): both
// libraries run the same gelsd there, so NumPy's whole call over NumSharp's lstsq time — the ceiling — bounds NPY/NS.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0x14
//   set OPENBLAS_NUM_THREADS=1
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyfit_numpy.py > numpy_fit_times.tsv
//   dotnet run -c Release --no-cache polyfit_bench.cs -- data/polyfit numpy_fit_times.tsv > ns_fit.tsv
//   python polyfit_report.py ns_fit.tsv > polyfit_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit, and PublishAot=false is
// load-bearing (the polynomial kernels are DynamicMethods). lstsq runs through NumSharp.Interop.OpenBLAS at ONE thread —
// the scipy-openblas NumPy itself loads — so the byte check holds on a host whose NumPy loads the same build.
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

// One thread, like NumPy's run under OPENBLAS_NUM_THREADS=1: gelsd's bits and speed both depend on it.
OpenBlasEngine.Enable(threads: 1);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyfit");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
var numpyLstsqUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 3 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us)
            && double.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var lus))
        {
            numpyUs[parts[0]] = us;
            numpyLstsqUs[parts[0]] = lus;
        }
    }

// The six fits through their one facade overload.
var fits = new Dictionary<string, Func<object, object, object, object, bool, object, NDPolyFitResult>>
{
    ["poly"] = (x, y, d, r, f, w) => np.polynomial.polynomial.polyfit(x, y, d, r, f, w),
    ["cheb"] = (x, y, d, r, f, w) => np.polynomial.chebyshev.chebfit(x, y, d, r, f, w),
    ["leg"] = (x, y, d, r, f, w) => np.polynomial.legendre.legfit(x, y, d, r, f, w),
    ["lag"] = (x, y, d, r, f, w) => np.polynomial.laguerre.lagfit(x, y, d, r, f, w),
    ["herm"] = (x, y, d, r, f, w) => np.polynomial.hermite.hermfit(x, y, d, r, f, w),
    ["herme"] = (x, y, d, r, f, w) => np.polynomial.hermite_e.hermefit(x, y, d, r, f, w),
};

using var inputs = np.load_npz(Path.Combine(data, "inputs.npz"));
using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, double NsLstsq, double NpLstsq, string Check)>();

// Warm-up over the small cells: the composition's engines (the Vandermonde kernel, the ufuncs, np.sum, lstsq, the
// NDScope weaving) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the background
// compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDPolyFitResult>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("calls").GetInt32() >= 500)
        warmCalls.Add(MakeCall(cell));
for (int pass = 0; pass < 3; pass++)
{
    foreach (var f in warmCalls)
        for (int w = 0; w < 60; w++) Release(f());
    System.Threading.Thread.Sleep(250);   // the tier-1 background compiler drains its queue
}

Console.WriteLine("cell\tnumsharp_us\tnumpy_us\tNPY/NS\tnumsharp_lstsq_us\tnumpy_lstsq_us\tceiling\tcheck");
foreach (var cell in manifest.RootElement.EnumerateArray())
{
    string label = cell.GetProperty("label").GetString();
    int calls = cell.GetProperty("calls").GetInt32(), rounds = cell.GetProperty("rounds").GetInt32();
    Func<NDPolyFitResult> call = MakeCall(cell);

    var first = call();
    string check = Check(first, cell.GetProperty("expect"));
    Release(first);
    for (int w = 0; w < 2; w++) Release(call());   // warm: kernel compile + tier-up
    double best = Best(() => Release(call()), calls, rounds);

    // The lstsq call alone, on NumPy's operands for this fit.
    string key = cell.GetProperty("key").GetString();
    NDArray la = inputs[key + "_A"], lb = inputs[key + "_b"];
    double lrcond = cell.GetProperty("rcond").GetDouble();
    Action lcall = () =>
    {
        var (c, r, k, s) = np.linalg.lstsq(la, lb, lrcond);
        c.Dispose(); r.Dispose(); k.Dispose(); s.Dispose();
    };
    for (int w = 0; w < 3; w++) lcall();
    double bestL = Best(lcall, calls, rounds);

    double npv = numpyUs.TryGetValue(label, out var v) ? v : double.NaN;
    double npl = numpyLstsqUs.TryGetValue(label, out var lv) ? lv : double.NaN;
    rows.Add((label, best, npv, bestL, npl, check));
    Console.WriteLine($"{label}\t{best:F3}\t{npv:F3}\t{npv / best:F2}\t{bestL:F3}\t{npl:F3}\t{npv / bestL:F2}\t{check}");
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
Console.WriteLine("cells below 1.5x (ceiling = NumPy's call / NumSharp's lstsq alone):");
foreach (var x in rows.Where(x => x.Np / x.Ns < 1.5).OrderBy(x => x.Np / x.Ns))
    Console.WriteLine($"  {x.Label}\t{x.Np / x.Ns:F2}\t(ns {x.Ns:F2} us, numpy {x.Np:F2} us, lstsq ns {x.NsLstsq:F2} / numpy {x.NpLstsq:F2} us, ceiling {x.Np / x.NsLstsq:F2})");

/// <summary>The best per-call time of <paramref name="f"/> over <paramref name="rounds"/> rounds of
///     <paramref name="calls"/> calls (the house best-of-rounds).</summary>
/// <param name="f">The timed call.</param>
/// <param name="calls">Calls per round.</param>
/// <param name="rounds">Rounds.</param>
/// <returns>Microseconds per call.</returns>
static double Best(Action f, int calls, int rounds)
{
    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < rounds; round++)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) f();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / calls);
    }
    return best;
}

/// <summary>Builds a cell's arguments (once — the timed call only runs the facade) and returns the timed call.</summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call.</returns>
Func<NDPolyFitResult> MakeCall(JsonElement cell)
{
    var f = fits[cell.GetProperty("fn").GetString()];
    string key = cell.GetProperty("key").GetString();
    NDArray x = inputs[key + "_x"], y = inputs[key + "_y"], w = inputs[key + "_w"];
    var degEl = cell.GetProperty("deg");
    // A degree list is a Python list of Python ints (an object[] of longs); a scalar degree a Python int.
    object deg = degEl.ValueKind == JsonValueKind.Array
        ? degEl.EnumerateArray().Select(e => (object)e.GetInt64()).ToArray()
        : degEl.GetInt64();
    bool weights = cell.GetProperty("weights").GetBoolean(), full = cell.GetProperty("full").GetBoolean();
    object xa = x, ya = y;
    if (cell.GetProperty("lists").GetBoolean())
    {
        // x.tolist() / y.tolist(): Python lists of Python floats.
        xa = Array.ConvertAll(x.ToArray<double>(), v => (object)v);
        ya = Array.ConvertAll(y.ToArray<double>(), v => (object)v);
    }
    object wa = weights ? w : null;
    return () => f(xa, ya, deg, null, full, wa);
}

/// <summary>Disposes a result and, for a full fit, its statistics (which the result owns).</summary>
/// <param name="r">The result.</param>
static void Release(NDPolyFitResult r)
{
    r.residuals?.Dispose();
    r.rank?.Dispose();
    r.singular_values?.Dispose();
    r.rcond?.Dispose();
    r.Dispose();
}

/// <summary>"exact" when every slot's dtype, shape and SHA-256 of the bytes (logical C order) match NumPy's.</summary>
/// <param name="got">NumSharp's result.</param>
/// <param name="expect">NumPy's digests, one per slot (the coefficients, then full=True's four statistics).</param>
/// <returns>"exact", or what differs.</returns>
static unsafe string Check(NDPolyFitResult got, JsonElement expect)
{
    var slots = expect.EnumerateArray().ToArray();
    NDArray[] arrays = slots.Length == 1 ? new NDArray[] { got } : new[] { got, got.residuals, got.rank, got.singular_values, got.rcond };
    for (int s = 0; s < slots.Length; s++)
    {
        var e = slots[s];
        var a = arrays[s];
        if (a.dtype.name != e.GetProperty("dtype").GetString()) return $"DTYPE[{s}] {a.dtype.name}";
        var shape = e.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray();
        if (!a.shape.SequenceEqual(shape)) return $"SHAPE[{s}] ({string.Join(",", a.shape)}) vs ({string.Join(",", shape)})";
        NDArray c = a.Shape.IsContiguous ? a : np.ascontiguousarray(a);
        var span = new ReadOnlySpan<byte>((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize, checked((int)(c.size * c.dtypesize)));
        string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
        if (!ReferenceEquals(c, a))
            c.Dispose();
        if (hex != e.GetProperty("sha256").GetString()) return $"BYTES[{s}]";
    }
    return "exact";
}
