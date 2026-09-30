#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyvander_bench.cs — numpy.polynomial Vandermonde family (U5: {p}vander / {p}vander2d / {p}vander3d): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyvander_numpy.py wrote (data/polyvander/manifest.json) — inputs regenerated from the shared
// exact formula of polyeval_numpy.py — checks dtype, shape and the SHA-256 of NumSharp's result bytes against
// NumPy's BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0xF0
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyvander_numpy.py > numpy_vander_times.tsv
//   dotnet run -c Release --no-cache polyvander_bench.cs -- data/polyvander numpy_vander_times.tsv > ns_vander.tsv
//   python polyvander_report.py ns_vander.tsv > polyvander_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit (a file-based app whose own
// source did not change reuses its cached build, referenced library included), and PublishAot=false is
// load-bearing (the Vandermonde kernels are DynamicMethods).
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

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyvander");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

var P = np.polynomial;
// The int overload a C# caller binds for an int degree (vander) and the object overloads (2-D / 3-D degree containers).
var vander = new Dictionary<string, Func<object, int, NDArray>>
{
    ["poly"] = P.polynomial.polyvander, ["cheb"] = P.chebyshev.chebvander, ["leg"] = P.legendre.legvander,
    ["lag"] = P.laguerre.lagvander, ["herm"] = P.hermite.hermvander, ["herme"] = P.hermite_e.hermevander,
};
// The object overload of {p}vander (a degree of any kind — the 0-d array degree cells).
var vanderObj = new Dictionary<string, Func<object, object, NDArray>>
{
    ["poly"] = P.polynomial.polyvander, ["cheb"] = P.chebyshev.chebvander, ["leg"] = P.legendre.legvander,
    ["lag"] = P.laguerre.lagvander, ["herm"] = P.hermite.hermvander, ["herme"] = P.hermite_e.hermevander,
};
var vander2d = new Dictionary<string, Func<object, object, object, NDArray>>
{
    ["poly"] = P.polynomial.polyvander2d, ["cheb"] = P.chebyshev.chebvander2d, ["leg"] = P.legendre.legvander2d,
    ["lag"] = P.laguerre.lagvander2d, ["herm"] = P.hermite.hermvander2d, ["herme"] = P.hermite_e.hermevander2d,
};
var vander3d = new Dictionary<string, Func<object, object, object, object, NDArray>>
{
    ["poly"] = P.polynomial.polyvander3d, ["cheb"] = P.chebyshev.chebvander3d, ["leg"] = P.legendre.legvander3d,
    ["lag"] = P.laguerre.lagvander3d, ["herm"] = P.hermite.hermvander3d, ["herme"] = P.hermite_e.hermevander3d,
};

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the driver code every cell shares (NDPolyVander, PolyIndexArgument, the NDScope weaving)
// must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the background compiler needs
// real time to drain the queue — so: pass, let the JIT drain, pass again.
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
    string form = cell.GetProperty("form").GetString();
    if (form == "special")
    {
        // The object stack of scalars (polyvander_numpy.py's special()): a BigInteger past uint64 among scalar points.
        var big = BigInteger.Pow(2, 70);
        switch (cell.GetProperty("special").GetString())
        {
            case "objstack2d":
            {
                var d2 = new object[] { 3, 3 };
                return () => vander2d[b](big, 1.5, d2);
            }
            case "objstack3d_mixed":
            {
                var d3 = new object[] { 2, 2, 2 };
                var f32 = NDArray.Scalar(1.5f);
                return () => vander3d[b](big, (Half)0.5, f32, d3);
            }
            case var k:
                throw new NotSupportedException(k);
        }
    }
    string pform = cell.TryGetProperty("pform", out var pf) ? pf.GetString() : null;
    // The points' argument form (built once, like NumPy's list exists before its call): the array, a Python list (an
    // object[] of boxed doubles — NumPy's a.tolist()) or a Python float (a 0-d array's value). Every arm is cast to object:
    // otherwise the conditional unifies object[] with NDArray through NDArray's implicit conversion from Array.
    object[] points = cell.GetProperty("points").EnumerateArray().Select(spec =>
    {
        NDArray arr = Gen(spec);
        return pform switch
        {
            "list" => (object)ToPythonList(arr),
            "scalar" => (object)arr.GetDouble(),
            "typed" => (object)arr.ToArray<double>(),                    // a typed C# array: an ndarray (converted per call)
            "glist" => (object)new List<double>(arr.ToArray<double>()),  // a List<double>: a Python list of floats
            null => (object)arr,
            var f => throw new NotSupportedException(f),
        };
    }).ToArray();
    var degEl = cell.GetProperty("deg");
    if (form == "1d")
    {
        int d = degEl.GetInt32();
        if (cell.TryGetProperty("degform", out var df1) && df1.GetString() == "nd0")
        {
            // A 0-d int64 array degree (NumPy's np.array(5)): the object overload, operator.index of the array.
            var dnd = NDArray.Scalar((long)d);
            return () => vanderObj[b](points[0], dnd);
        }
        return () => vander[b](points[0], d);
    }
    int[] ds = degEl.EnumerateArray().Select(e => e.GetInt32()).ToArray();
    // The degree container in the C# boundary map's spelling of the NumPy twin's argument: a Python list of ints is an
    // object[] of boxed ints (the default), a tuple a ValueTuple, an ndarray a typed int[] (which the facade reads as an
    // ndarray, iterating NumPy int scalars). All three are a sequence of length n to _vander_nd's len(deg).
    string degform = cell.TryGetProperty("degform", out var dfEl) ? dfEl.GetString() : null;
    object deg = degform switch
    {
        "tuple" => ds.Length == 2 ? (object)(ds[0], ds[1]) : (object)(ds[0], ds[1], ds[2]),
        "array" => (object)ds,
        null => (object)ds.Select(d => (object)d).ToArray(),
        var f => throw new NotSupportedException(f),
    };
    return form switch
    {
        "2d" => () => vander2d[b](points[0], points[1], deg),
        "3d" => () => vander3d[b](points[0], points[1], points[2], deg),
        _ => throw new NotSupportedException(form),
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

/// <summary>NumPy's <c>a.tolist()</c> of a float64 array as the C# spelling of a Python list: nested <c>object[]</c>
///     whose leaves are boxed doubles (Python floats).</summary>
/// <param name="a">A float64 array of rank ≥ 1.</param>
/// <returns>The nested list.</returns>
static object[] ToPythonList(NDArray a)
{
    if (a.ndim == 1)
        return a.ToArray<double>().Select(v => (object)v).ToArray();
    return Enumerable.Range(0, (int)a.shape[0]).Select(i => (object)ToPythonList(a[i])).ToArray();
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
