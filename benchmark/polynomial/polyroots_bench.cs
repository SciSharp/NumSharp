#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyroots_bench.cs — numpy.polynomial companion matrices and roots (U7: {p}companion / {p}roots): NumSharp vs
// NumPy 2.4.2
// =============================================================================
// Replays every cell polyroots_numpy.py wrote (data/polyroots/manifest.json) — inputs regenerated from the shared exact
// formula of polyeval_numpy.py — checks dtype, shape and the SHA-256 of NumSharp's result bytes against NumPy's BEFORE
// timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// The roots of degree 2 or more are np.linalg.eigvals of the companion matrix: NumSharp.Core ships no eigensolver, so
// this runner enables NumSharp.Interop.OpenBLAS at ONE thread — the bundled scipy-openblas NumPy itself loads, which
// makes them byte-identical ("exact") on a host that dispatches the same kernels. A "lapack" cell that differs in its
// bytes is accepted as "close" when every value is within 1e-12 of NumPy's (stored in the manifest for those cells),
// relative to the result's largest magnitude: another OpenBLAS build or kernel sums in another order.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0xF0
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyroots_numpy.py > numpy_roots_times.tsv
//   dotnet run -c Release --no-cache polyroots_bench.cs -- data/polyroots numpy_roots_times.tsv > ns_roots.tsv
//   python polyroots_report.py ns_roots.tsv > polyroots_results.md      (the committed summary)
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit (a file-based app whose own
// source did not change reuses its cached build, referenced library included), and PublishAot=false is
// load-bearing (the series kernels are DynamicMethods).
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

// NumPy's eigvals runs scipy-openblas' geev at the thread count OPENBLAS_NUM_THREADS sets (1 on the NumPy side).
OpenBlasEngine.Enable(threads: 1);
Console.Error.WriteLine($"backend: {OpenBlasEngine.Info}");

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyroots");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

var P = np.polynomial;
// Every facade member the manifest names.
var fns = new Dictionary<string, Func<object, NDArray>>(StringComparer.Ordinal)
{
    ["polycompanion"] = P.polynomial.polycompanion, ["polyroots"] = P.polynomial.polyroots,
    ["chebcompanion"] = P.chebyshev.chebcompanion, ["chebroots"] = P.chebyshev.chebroots,
    ["legcompanion"] = P.legendre.legcompanion, ["legroots"] = P.legendre.legroots,
    ["lagcompanion"] = P.laguerre.lagcompanion, ["lagroots"] = P.laguerre.lagroots,
    ["hermcompanion"] = P.hermite.hermcompanion, ["hermroots"] = P.hermite.hermroots,
    ["hermecompanion"] = P.hermite_e.hermecompanion, ["hermeroots"] = P.hermite_e.hermeroots,
};

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the driver code every cell shares (NDPolyAlgebra, the arena, the house kernels, the
// NDScope weaving, eigvals' Python layer) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for
// tier-1; the background compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
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
    {
        var first = call();
        check = Check(first, cell.GetProperty("expect"), cell.GetProperty("lapack").GetBoolean());
        first.Dispose();
    }
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

/// <summary>Builds a cell's argument (once — the timed call only runs the facade) and returns the timed call.</summary>
Func<NDArray> MakeCall(JsonElement cell)
{
    var f = fns[cell.GetProperty("fn").GetString()];
    var spec = cell.GetProperty("arg");
    NDArray arr = Gen(spec.GetProperty("gen"));
    // The argument as NumPy passed it: the generated array, or (form "list") a Python list — an object[] of boxed
    // doubles, NumPy's c.tolist(), built once like NumPy's list exists before its call. Both arms are cast to object:
    // a conditional whose arms are object[] and NDArray is typed NDArray through NumSharp's implicit Array -> NDArray
    // conversion, which cannot interpret object elements.
    object argv = spec.TryGetProperty("form", out var form) && form.GetString() == "list"
        ? (object)arr.ToArray<double>().Select(x => (object)x).ToArray()
        : (object)arr;
    return () => f(argv);
}

// Summaries: per section min and geomean of NPY/NS, then every cell under 1.5x.
Console.WriteLine();
Console.WriteLine("section\tcells\tmin NPY/NS\tgeomean NPY/NS\tmismatches");
foreach (var g in rows.GroupBy(x => x.Label.Split('/')[0]).OrderBy(g => g.Key))
{
    var ratios = g.Where(x => !double.IsNaN(x.Np)).Select(x => x.Np / x.Ns).ToArray();
    double gm = ratios.Length == 0 ? double.NaN : Math.Exp(ratios.Average(Math.Log));
    Console.WriteLine($"{g.Key}\t{g.Count()}\t{(ratios.Length == 0 ? double.NaN : ratios.Min()):F2}\t{gm:F2}\t{g.Count(x => x.Check != "exact" && x.Check != "close")}");
}
var all = rows.Where(x => !double.IsNaN(x.Np)).Select(x => x.Np / x.Ns).ToArray();
Console.WriteLine($"ALL\t{rows.Count}\t{all.Min():F2}\t{Math.Exp(all.Average(Math.Log)):F2}\t{rows.Count(x => x.Check != "exact" && x.Check != "close")}");
Console.WriteLine();
Console.WriteLine("cells below 1.5x:");
foreach (var x in rows.Where(x => x.Np / x.Ns < 1.5).OrderBy(x => x.Np / x.Ns))
    Console.WriteLine($"  {x.Label}\t{x.Np / x.Ns:F2}\t(ns {x.Ns:F2} us, numpy {x.Np:F2} us)");

/// <summary>
///     polyeval_numpy.py's gen() in C#: t_i = (i*7919 + seed*104729) mod 20011, value = t_i/10005.5 - 1.0 —
///     integer arithmetic, one correctly rounded division and one subtraction, so both languages produce the
///     same doubles; narrower dtypes go through the house astype (NumPy's own conversions) — then the view grammar
///     (<c>slice:</c> / <c>T</c> / <c>F</c> / <c>bcast:</c>) the NumPy side applied.
/// </summary>
/// <param name="spec">The manifest's <c>gen</c> object (shape, dtype, seed, optional view).</param>
/// <returns>The input array.</returns>
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
    NDArray a = flat.reshape(new Shape(shape));
    if (!spec.TryGetProperty("view", out var view)) return a;
    string t = view.GetString();
    if (t == "T") return a.T;
    if (t == "F") return np.asfortranarray(a);
    if (t.StartsWith("slice:", StringComparison.Ordinal)) return a[t.Substring("slice:".Length)];
    if (t.StartsWith("bcast:", StringComparison.Ordinal))
        return np.broadcast_to(a, new Shape(t.Substring("bcast:".Length).Split(',').Select(long.Parse).ToArray()));
    throw new NotSupportedException(t);
}

/// <summary>
///     "exact" when dtype, shape and the SHA-256 of the bytes (logical C order) match NumPy's (a complex64 result was
///     digested widened to complex128, the values NumSharp carries). For a "lapack" cell whose bytes differ, "close"
///     when every value is within 1e-12 of NumPy's relative to the largest magnitude (another OpenBLAS build or kernel).
/// </summary>
/// <param name="got">The result.</param>
/// <param name="expect">NumPy's digest (and, for a lapack cell, its values as hex).</param>
/// <param name="lapack">Whether NumPy's call ran LAPACK geev.</param>
/// <returns>"exact", "close", or the first mismatch.</returns>
static unsafe string Check(NDArray got, JsonElement expect, bool lapack)
{
    if (got.dtype.name != expect.GetProperty("dtype").GetString()) return $"DTYPE {got.dtype.name}";
    var shape = expect.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
    if (!got.shape.SequenceEqual(shape)) return $"SHAPE ({string.Join(",", got.shape)}) vs ({string.Join(",", shape)})";
    using var g = np.ascontiguousarray(got);
    long bytes = g.size * g.dtypesize;
    var span = new ReadOnlySpan<byte>((byte*)g.Storage.Address + g.Shape.offset * g.dtypesize, checked((int)bytes));
    string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(span)).ToLowerInvariant();
    if (hex == expect.GetProperty("sha256").GetString())
        return "exact";
    if (!lapack || !expect.TryGetProperty("hex", out var want)
        || (g.typecode != NPTypeCode.Single && g.typecode != NPTypeCode.Double && g.typecode != NPTypeCode.Complex))
        return "BYTES";
    double[] Doubles(byte[] b) => g.typecode == NPTypeCode.Single
        ? Enumerable.Range(0, b.Length / 4).Select(i => (double)BitConverter.ToSingle(b, i * 4)).ToArray()
        : Enumerable.Range(0, b.Length / 8).Select(i => BitConverter.ToDouble(b, i * 8)).ToArray();
    double[] mine = Doubles(span.ToArray()), theirs = Doubles(Convert.FromHexString(want.GetString()));
    double scale = theirs.Where(double.IsFinite).Select(Math.Abs).DefaultIfEmpty(0).Max();
    double tol = g.typecode == NPTypeCode.Single ? 1e-5 : 1e-12;
    for (int i = 0; i < mine.Length; i++)
        if (Math.Abs(mine[i] - theirs[i]) > tol * scale)
            return "FAR";
    return "close";
}
