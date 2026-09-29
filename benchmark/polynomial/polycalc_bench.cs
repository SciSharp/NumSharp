#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polycalc_bench.cs — numpy.polynomial calculus family (U4: {p}der / {p}int): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polycalc_numpy.py wrote (data/polycalc/manifest.json) — inputs regenerated from the shared
// exact formula of polyeval_numpy.py — checks dtype, shape and the SHA-256 of NumSharp's result bytes against
// NumPy's BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0xF0
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polycalc_numpy.py > numpy_calc_times.tsv
//   dotnet run -c Release --no-cache polycalc_bench.cs -- data/polycalc numpy_calc_times.tsv > ns_calc.tsv
//   python polycalc_report.py ns_calc.tsv > polycalc_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit (a file-based app whose own
// source did not change reuses its cached build, referenced library included), and PublishAot=false is
// load-bearing (the calculus kernels are DynamicMethods).
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

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polycalc");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

var P = np.polynomial;
var der = new Dictionary<string, Func<object, int, object, int, NDArray>>
{
    ["poly"] = P.polynomial.polyder, ["cheb"] = P.chebyshev.chebder, ["leg"] = P.legendre.legder,
    ["lag"] = P.laguerre.lagder, ["herm"] = P.hermite.hermder, ["herme"] = P.hermite_e.hermeder,
};
var integ = new Dictionary<string, Func<object, int, object, object, object, int, NDArray>>
{
    ["poly"] = P.polynomial.polyint, ["cheb"] = P.chebyshev.chebint, ["leg"] = P.legendre.legint,
    ["lag"] = P.laguerre.lagint, ["herm"] = P.hermite.hermint, ["herme"] = P.hermite_e.hermeint,
};

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the driver code every cell shares (NDPolyCalc, NDPolyEval's lbnd correction,
// the NDScope weaving) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the
// background compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDArray>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("calls").GetInt32() >= 20)   // big cells: their per-cell warm-up suffices
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

    string check;
    using (var first = call())
        check = Check(first, cell.GetProperty("expect"));
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

/// <summary>Builds a cell's arguments (once — the timed call only runs the facade) and returns the timed call.</summary>
Func<NDArray> MakeCall(JsonElement cell)
{
    string b = cell.GetProperty("base").GetString();
    string kind = cell.GetProperty("kind").GetString();
    int m = cell.GetProperty("m").GetInt32(), axis = cell.GetProperty("axis").GetInt32();
    object Opt(string name) => cell.TryGetProperty(name, out var s) ? BuildArg(s) : null;
    NDArray c = Gen(cell.GetProperty("c"));
    object scl = Opt("scl"), k = Opt("k"), lbnd = Opt("lbnd");
    return kind switch
    {
        "der" => () => der[b](c, m, scl, axis),
        "int" => () => integ[b](c, m, k, lbnd, scl, axis),
        _ => throw new NotSupportedException(kind),
    };
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

/// <summary>Rebuilds one argument: a generated array (optionally re-viewed), a Python scalar as its C# primitive, or
///     a Python list (object[]) of those.</summary>
static object BuildArg(JsonElement spec)
{
    if (spec.TryGetProperty("gen", out var gen)) return Gen(gen);
    if (spec.TryGetProperty("list", out var list)) return list.EnumerateArray().Select(BuildArg).ToArray();
    string kind = spec.GetProperty("weak").GetString();
    return kind switch
    {
        "float" => (object)spec.GetProperty("value").GetDouble(),
        "int" => spec.GetProperty("value").GetInt64(),
        "bool" => spec.GetProperty("value").GetBoolean(),
        "complex" => new Complex(spec.GetProperty("re").GetDouble(), spec.GetProperty("im").GetDouble()),
        _ => throw new NotSupportedException(kind),
    };
}

/// <summary>
///     polyeval_numpy.py's gen() in C#: t_i = (i*7919 + seed*104729) mod 20011, value = t_i/10005.5 - 1.0 —
///     integer arithmetic, one correctly rounded division and one subtraction, so both languages produce the
///     same doubles; narrower dtypes go through the house astype (NumPy's own conversions).
/// </summary>
static NDArray Gen(JsonElement spec)
{
    var shape = spec.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    string dtype = spec.GetProperty("dtype").GetString();
    long seed = spec.GetProperty("seed").GetInt64();
    long n = shape.Aggregate(1L, (a, b) => a * b);
    long T(long i, long s) => (i * 7919 + s * 104729) % 20011;
    NDArray flat;
    switch (dtype)
    {
        case "bool":
        {
            var v = new bool[n];
            for (long i = 0; i < n; i++) v[i] = T(i, seed) % 2 == 1;
            flat = np.array(v);
            break;
        }
        case "uint8": case "uint16": case "uint32": case "uint64":
        {
            var v = new long[n];
            for (long i = 0; i < n; i++) v[i] = T(i, seed) % 4;
            flat = np.array(v).astype(DType.From(dtype));
            break;
        }
        case "int8": case "int16": case "int32": case "int64":
        {
            var v = new long[n];
            for (long i = 0; i < n; i++) v[i] = T(i, seed) % 7 - 3;
            flat = np.array(v).astype(DType.From(dtype));
            break;
        }
        case "complex128":
        {
            var v = new Complex[n];
            for (long i = 0; i < n; i++) v[i] = new Complex(T(i, seed) / 10005.5 - 1.0, T(i, seed + 1000) / 10005.5 - 1.0);
            flat = np.array(v);
            break;
        }
        default:
        {
            var v = new double[n];
            for (long i = 0; i < n; i++) v[i] = T(i, seed) / 10005.5 - 1.0;
            flat = np.array(v).astype(DType.From(dtype));
            break;
        }
    }
    // A () shape is a 0-d array (NumPy's np.asarray(v).reshape(())), which Shape.NewScalar spells.
    NDArray a = shape.Length == 0 ? flat.reshape(Shape.NewScalar()) : flat.reshape(new Shape(shape));
    if (!spec.TryGetProperty("view", out var view)) return a;
    string t = view.GetString();
    if (t == "T") return a.T;
    if (t == "F") return np.asfortranarray(a);
    if (t.StartsWith("slice:", StringComparison.Ordinal)) return a[t.Substring("slice:".Length)];
    if (t.StartsWith("bcast:", StringComparison.Ordinal))
        return np.broadcast_to(a, new Shape(t.Substring("bcast:".Length).Split(',').Select(long.Parse).ToArray()));
    throw new NotSupportedException(t);
}

/// <summary>"exact" when dtype, shape and the SHA-256 of the bytes (logical C order) match NumPy's.</summary>
static unsafe string Check(NDArray got, JsonElement expect)
{
    if (got.dtype.name != expect.GetProperty("dtype").GetString()) return $"DTYPE {got.dtype.name}";
    var shape = expect.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    if (!got.shape.SequenceEqual(shape)) return $"SHAPE ({string.Join(",", got.shape)}) vs ({string.Join(",", shape)})";
    using var g = np.ascontiguousarray(got);
    long bytes = g.size * g.dtypesize;
    var span = new ReadOnlySpan<byte>((byte*)g.Storage.Address + g.Shape.offset * g.dtypesize, checked((int)bytes));
    string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
    return hex == expect.GetProperty("sha256").GetString() ? "exact" : "BYTES";
}
