#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyprint_bench.cs — numpy.polynomial's printing helpers and the scalar float formatters (U11: polyutils.format_float,
// set_default_printstyle; np.format_float_positional / np.format_float_scientific, the port of NumPy's dragon4.c):
// NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyprint_numpy.py wrote (data/polyprint/manifest.json), compares NumSharp's text with NumPy's
// BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), text check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently; three alternating
// pairs, the report keeping each cell's best window of each side):
//   set NS_PROBE_AFFINITY=0x14
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyprint_numpy.py > numpy_print_times_1.tsv
//   dotnet run -c Release --no-cache polyprint_bench.cs -- data/polyprint numpy_print_times_1.tsv > ns_print_1.tsv
//   (the same for _2 and _3)
//   python polyprint_report.py ns_print_1.tsv ns_print_2.tsv ns_print_3.tsv > polyprint_results.md
// Release is mandatory (benchmark/CLAUDE.md), and --no-cache after any NumSharp.Core edit.
// =============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using NumSharp;

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyprint");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up: every cell's call a few dozen times, twice, with a pause for the tier-1 background compiler to drain its
// queue (one pass only QUEUES the methods).
var calls0 = manifest.RootElement.EnumerateArray().Select(MakeCall).ToList();
for (int pass = 0; pass < 2; pass++)
{
    foreach (var f in calls0)
        for (int w = 0; w < 40; w++) f();
    System.Threading.Thread.Sleep(250);
}

Console.WriteLine("cell\tnumsharp_us\tnumpy_us\tNPY/NS\tcheck");
foreach (var cell in manifest.RootElement.EnumerateArray())
{
    string label = cell.GetProperty("label").GetString();
    int calls = cell.GetProperty("calls").GetInt32(), rounds = cell.GetProperty("rounds").GetInt32();
    Func<string> call = MakeCall(cell);
    string got = call() ?? "";
    string check = got == cell.GetProperty("expect").GetString() ? "exact" : "TEXT";

    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < rounds; round++)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) call();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / calls);
    }
    double npv = numpyUs.TryGetValue(label, out var v) ? v : double.NaN;
    rows.Add((label, best, npv, check));
    Console.WriteLine($"{label}\t{best:F3}\t{npv:F3}\t{npv / best:F2}\t{check}");
}
// set_default_printstyle changed a process-wide flag: leave NumPy's default behind (nothing reads it after this).
np.polynomial.set_default_printstyle(OperatingSystem.IsWindows() ? "ascii" : "unicode");

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
    Console.WriteLine($"  {x.Label}\t{x.Np / x.Ns:F2}\t(ns {x.Ns:F3} us, numpy {x.Np:F3} us)");

/// <summary>Builds a cell's argument and options (once — the timed call only runs the facade) and returns the call.</summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call, returning the text (null for set_default_printstyle).</returns>
/// <exception cref="NotSupportedException">An unknown formatter or argument kind.</exception>
Func<string> MakeCall(JsonElement cell)
{
    string fn = cell.GetProperty("fn").GetString();
    var xs = cell.GetProperty("x");
    // The argument in the C# boundary map's spelling of the NumPy twin's: a width's scalar is a double / float / Half
    // (np.float64 / np.float32 / np.float16), a Python int a long, a 0-d or 1-D array an NDArray, a str a string. Every
    // arm is cast to object, so the conditional does not unify through NDArray's implicit conversions.
    object x = xs.GetProperty("kind").GetString() switch
    {
        "f64" => (object)BitConverter.UInt64BitsToDouble(Convert.ToUInt64(xs.GetProperty("bits").GetString(), 16)),
        "f32" => (object)BitConverter.UInt32BitsToSingle(Convert.ToUInt32(xs.GetProperty("bits").GetString(), 16)),
        "f16" => (object)BitConverter.UInt16BitsToHalf(Convert.ToUInt16(xs.GetProperty("bits").GetString(), 16)),
        "int" => (object)xs.GetProperty("v").GetInt64(),
        "str" => (object)xs.GetProperty("v").GetString(),
        "array" => xs.TryGetProperty("zerod", out var z) && z.GetBoolean()
            ? (object)np.array(xs.GetProperty("values")[0].GetDouble())
            : (object)np.array(xs.GetProperty("values").EnumerateArray().Select(e => e.GetDouble()).ToArray()),
        var other => throw new NotSupportedException(other),
    };
    var kw = cell.GetProperty("kwargs");
    int? I(string n) => kw.TryGetProperty(n, out var e) ? e.GetInt32() : null;
    bool B(string n, bool d) => kw.TryGetProperty(n, out var e) ? e.GetBoolean() : d;
    string trim = kw.TryGetProperty("trim", out var t) ? t.GetString() : "k";
    switch (fn)
    {
        case "pos":
        {
            int? p = I("precision"), pl = I("pad_left"), pr = I("pad_right"), md = I("min_digits");
            bool u = B("unique", true), fr = B("fractional", true), sg = B("sign", false);
            return () => np.format_float_positional(x, p, u, fr, trim, sg, pl, pr, md);
        }
        case "sci":
        {
            int? p = I("precision"), pl = I("pad_left"), ed = I("exp_digits"), md = I("min_digits");
            bool u = B("unique", true), sg = B("sign", false);
            return () => np.format_float_scientific(x, p, u, trim, sg, pl, ed, md);
        }
        case "ff":
        {
            var u = np.polynomial.polyutils;
            if (cell.TryGetProperty("opts", out var o) && o.TryGetProperty("floatmode", out var fm))
            {
                // NumPy's twin times the call inside np.printoptions(**opts): so does this one.
                string mode = fm.GetString();
                return () =>
                {
                    using (np.printoptions(floatmode: mode))
                        return u.format_float(x);
                };
            }
            return () => u.format_float(x);
        }
        case "style":
            return () =>
            {
                np.polynomial.set_default_printstyle(x);
                return "";
            };
        default:
            throw new NotSupportedException(fn);
    }
}
