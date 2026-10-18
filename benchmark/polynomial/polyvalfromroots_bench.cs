#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyvalfromroots_bench.cs — numpy.polynomial.polynomial.polyvalfromroots (unit 6): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyvalfromroots_numpy.py wrote (data/polyvalfromroots/manifest.json) — inputs regenerated from the
// shared exact formula of polyeval_numpy.py — checks dtype, shape and the SHA-256 of NumSharp's result against NumPy's
// BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0x14
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyvalfromroots_numpy.py > numpy_vfr_times.tsv
//   dotnet run -c Release --no-cache polyvalfromroots_bench.cs -- data/polyvalfromroots numpy_vfr_times.tsv > ns_vfr.tsv
//   python polyvalfromroots_report.py ns_vfr.tsv > polyvalfromroots_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit (a file-based app reuses its cached
// build, referenced library included), and PublishAot=false is load-bearing (the root-product kernels are DynamicMethods).
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

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyvalfromroots");
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

// Warm-up over the small cells: the driver code every cell shares (the conversions, the layout vote, the constant pool,
// NDIter setup) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the background
// compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDArray>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("k").GetInt32() >= 20)
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
    int k = cell.GetProperty("k").GetInt32(), r = cell.GetProperty("r_").GetInt32();
    Func<NDArray> call = MakeCall(cell);

    string check;
    using (var first = call())
        check = Check(first, cell.GetProperty("expect"));
    for (int w = 0; w < 2; w++) call().Dispose();   // warm: kernel compile + tier-up

    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < r; round++)
    {
        sw.Restart();
        for (int i = 0; i < k; i++) call().Dispose();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / k);
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
///     Builds a cell's arguments once (the timed call runs only the facade) and returns the timed call.
/// </summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call.</returns>
Func<NDArray> MakeCall(JsonElement cell)
{
    object x = BuildArg(cell.GetProperty("x"));
    object r = BuildArg(cell.GetProperty("r"));
    bool tensor = cell.GetProperty("tensor").GetBoolean();
    return () => np.polynomial.polynomial.polyvalfromroots(x, r, tensor);
}

/// <summary>
///     Rebuilds one argument: a generated array (optionally re-viewed), a Python scalar as its C# primitive, a Python list
///     of a generated array's values (an <c>object[]</c> of boxed doubles), or the Python list <c>[1, …, n]</c> (boxed longs).
/// </summary>
/// <param name="spec">The twin's argument spec.</param>
/// <returns>The argument.</returns>
/// <exception cref="NotSupportedException">An unknown spec kind.</exception>
static object BuildArg(JsonElement spec)
{
    if (spec.TryGetProperty("gen", out var gen)) return Gen(gen);
    if (spec.TryGetProperty("list", out var list))
        return Gen(list).ToArray<double>().Select(v => (object)v).ToArray();
    if (spec.TryGetProperty("intlist", out var n))
        return Enumerable.Range(1, n.GetInt32()).Select(v => (object)(long)v).ToArray();
    string kind = spec.GetProperty("weak").GetString();
    return kind switch
    {
        "float" => (object)spec.GetProperty("value").GetDouble(),
        "complex" => new Complex(spec.GetProperty("re").GetDouble(), spec.GetProperty("im").GetDouble()),
        _ => throw new NotSupportedException(kind),
    };
}

/// <summary>
///     polyeval_numpy.py's gen() in C#: t_i = (i*7919 + seed*104729) mod 20011, value = t_i/10005.5 - 1.0 — integer
///     arithmetic, one correctly rounded division and one subtraction, so both languages produce the same doubles;
///     narrower dtypes go through the house astype (NumPy's own conversions).
/// </summary>
/// <param name="spec">The generated array's spec (shape, dtype, seed, optional view).</param>
/// <returns>The array (or its view).</returns>
/// <exception cref="NotSupportedException">An unknown view transform.</exception>
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
/// <param name="got">NumSharp's result.</param>
/// <param name="expect">NumPy's digest (dtype, shape, sha256).</param>
/// <returns>"exact", or what differs.</returns>
static unsafe string Check(NDArray got, JsonElement expect)
{
    if (got.dtype.name != expect.GetProperty("dtype").GetString()) return $"DTYPE {got.dtype.name}";
    var shape = expect.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    if (!got.shape.SequenceEqual(shape)) return $"SHAPE ({string.Join(",", got.shape)}) vs ({string.Join(",", shape)})";
    // A non-C result is hashed through a C-order copy (NumPy's tobytes() order).
    NDArray c = got.Shape.IsContiguous ? got : np.ascontiguousarray(got);
    long bytes = c.size * c.dtypesize;
    var span = new ReadOnlySpan<byte>((byte*)c.Storage.Address + c.Shape.offset * c.dtypesize, checked((int)bytes));
    string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
    if (!ReferenceEquals(c, got))
        c.Dispose();
    return hex == expect.GetProperty("sha256").GetString() ? "exact" : "BYTES";
}
