#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyalg_bench.cs — numpy.polynomial series algebra (U2: {p}mulx / {p}mul / {p}div / {p}pow / {p}fromroots,
// X2poly / poly2X): NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyalg_numpy.py wrote (data/polyalg/manifest.json) — inputs regenerated from the shared exact
// formula of polyeval_numpy.py — checks dtype, shape and the SHA-256 of NumSharp's result bytes (both slots of a
// {p}div) against NumPy's BEFORE timing it, then prints a TSV joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// The check is "exact" (bytes equal) for every cell NumPy computes without OpenBLAS's vector dot kernels. A cell
// the manifest flags "blas" — a product whose np.convolve reaches those kernels (float64 dots of 16+ terms, float32
// 32+, complex128 8+) — is timed on NumSharp's DEFAULT managed kernels, whose summation order no longer matches
// (byte parity for those needs NumSharp.Interop.OpenBLAS, gated by the host-pinned polyalgebra_parity oracle tier);
// it is checked "close" instead: every element within 1e-12 of NumPy's relative to the result's largest magnitude.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently):
//   set NS_PROBE_AFFINITY=0xF0
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyalg_numpy.py > numpy_alg_times.tsv
//   dotnet run -c Release --no-cache polyalg_bench.cs -- data/polyalg numpy_alg_times.tsv > ns_alg.tsv
//   python polyalg_report.py ns_alg.tsv > polyalg_results.md      (the committed summary)
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

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyalg");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

var P = np.polynomial;
// Every facade member the manifest names, as a call taking the built arguments (+ the manifest's extra: pow).
// A {p}div returns its two slots as an array of two.
var fns = new Dictionary<string, Func<object[], JsonElement?, NDArray[]>>(StringComparer.Ordinal);
void One(string name, Func<object[], JsonElement?, NDArray> f) => fns[name] = (a, x) => new[] { f(a, x) };
void Two(string name, Func<object, object, (NDArray quo, NDArray rem)> f) => fns[name] = (a, _) => { var (q, r) = f(a[0], a[1]); return new[] { q, r }; };
int Pow(JsonElement? x) => x.Value.GetProperty("pow").GetInt32();
One("polymulx", (a, _) => P.polynomial.polymulx(a[0])); One("polymul", (a, _) => P.polynomial.polymul(a[0], a[1]));
Two("polydiv", P.polynomial.polydiv); One("polypow", (a, x) => P.polynomial.polypow(a[0], Pow(x)));
One("polyfromroots", (a, _) => P.polynomial.polyfromroots(a[0]));
One("chebmulx", (a, _) => P.chebyshev.chebmulx(a[0])); One("chebmul", (a, _) => P.chebyshev.chebmul(a[0], a[1]));
Two("chebdiv", P.chebyshev.chebdiv); One("chebpow", (a, x) => P.chebyshev.chebpow(a[0], Pow(x)));
One("chebfromroots", (a, _) => P.chebyshev.chebfromroots(a[0]));
One("cheb2poly", (a, _) => P.chebyshev.cheb2poly(a[0])); One("poly2cheb", (a, _) => P.chebyshev.poly2cheb(a[0]));
One("legmulx", (a, _) => P.legendre.legmulx(a[0])); One("legmul", (a, _) => P.legendre.legmul(a[0], a[1]));
Two("legdiv", P.legendre.legdiv); One("legpow", (a, x) => P.legendre.legpow(a[0], Pow(x)));
One("legfromroots", (a, _) => P.legendre.legfromroots(a[0]));
One("leg2poly", (a, _) => P.legendre.leg2poly(a[0])); One("poly2leg", (a, _) => P.legendre.poly2leg(a[0]));
One("lagmulx", (a, _) => P.laguerre.lagmulx(a[0])); One("lagmul", (a, _) => P.laguerre.lagmul(a[0], a[1]));
Two("lagdiv", P.laguerre.lagdiv); One("lagpow", (a, x) => P.laguerre.lagpow(a[0], Pow(x)));
One("lagfromroots", (a, _) => P.laguerre.lagfromroots(a[0]));
One("lag2poly", (a, _) => P.laguerre.lag2poly(a[0])); One("poly2lag", (a, _) => P.laguerre.poly2lag(a[0]));
One("hermmulx", (a, _) => P.hermite.hermmulx(a[0])); One("hermmul", (a, _) => P.hermite.hermmul(a[0], a[1]));
Two("hermdiv", P.hermite.hermdiv); One("hermpow", (a, x) => P.hermite.hermpow(a[0], Pow(x)));
One("hermfromroots", (a, _) => P.hermite.hermfromroots(a[0]));
One("herm2poly", (a, _) => P.hermite.herm2poly(a[0])); One("poly2herm", (a, _) => P.hermite.poly2herm(a[0]));
One("hermemulx", (a, _) => P.hermite_e.hermemulx(a[0])); One("hermemul", (a, _) => P.hermite_e.hermemul(a[0], a[1]));
Two("hermediv", P.hermite_e.hermediv); One("hermepow", (a, x) => P.hermite_e.hermepow(a[0], Pow(x)));
One("hermefromroots", (a, _) => P.hermite_e.hermefromroots(a[0]));
One("herme2poly", (a, _) => P.hermite_e.herme2poly(a[0])); One("poly2herme", (a, _) => P.hermite_e.poly2herme(a[0]));

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var rows = new List<(string Label, double Ns, double Np, string Check)>();

// Warm-up over the small cells: the driver code every cell shares (NDPolyAlgebra, the arena, the house kernels, the
// NDScope weaving) must reach tier-1 before ANY cell is timed. One pass only QUEUES methods for tier-1; the
// background compiler needs real time to drain the queue — so: pass, let the JIT drain, pass again.
var warmCalls = new List<Func<NDArray[]>>();
foreach (var cell in manifest.RootElement.EnumerateArray())
    if (cell.GetProperty("calls").GetInt32() >= 20)   // big cells: their per-cell warm-up suffices
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

    string check;
    {
        var first = call();
        check = Check(first, cell.GetProperty("expect"), cell.GetProperty("blas").GetBoolean());
        Dispose(first);
    }
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

/// <summary>Builds a cell's arguments (once — the timed call only runs the facade) and returns the timed call.</summary>
Func<NDArray[]> MakeCall(JsonElement cell)
{
    var f = fns[cell.GetProperty("fn").GetString()];
    // Each argument as NumPy passed it: the generated array, or (form "list") a Python list — an object[] of boxed
    // doubles, NumPy's c.tolist(), built once like NumPy's list exists before its call.
    object[] argv = cell.GetProperty("args").EnumerateArray().Select(s =>
    {
        NDArray arr = Gen(s.GetProperty("gen"));
        return s.TryGetProperty("form", out var form) && form.GetString() == "list"
            ? (object)arr.ToArray<double>().Select(x => (object)x).ToArray()
            : (object)arr;
    }).ToArray();
    JsonElement? extra = cell.TryGetProperty("extra", out var e) ? e : null;
    return () => f(argv, extra);
}

/// <summary>Disposes every slot of a result.</summary>
static void Dispose(NDArray[] r)
{
    foreach (var x in r) x.Dispose();
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
    return shape.Length == 0 ? flat.reshape(Shape.NewScalar()) : flat.reshape(new Shape(shape));
}

/// <summary>
///     "exact" when every slot's dtype, shape and SHA-256 of its bytes (logical C order) match NumPy's. For a cell the
///     manifest flags "blas", "close" when dtype and shape match and every element is within 1e-12 of NumPy's value
///     (stored in the manifest for those cells), relative to the slot's largest magnitude — the managed product's
///     summation order differs from OpenBLAS's vector kernels, so its bytes cannot match without the backend.
/// </summary>
/// <param name="got">The result slots.</param>
/// <param name="expect">NumPy's digest (one object, or one per {p}div slot).</param>
/// <param name="blas">Whether NumPy's product ran OpenBLAS's vector dot kernels.</param>
/// <returns>"exact", "close", or the first mismatch.</returns>
static unsafe string Check(NDArray[] got, JsonElement expect, bool blas)
{
    var slots = expect.ValueKind == JsonValueKind.Array ? expect.EnumerateArray().ToArray() : new[] { expect };
    if (slots.Length != got.Length) return $"ARITY {got.Length}";
    bool allExact = true, allClose = true;
    for (int s = 0; s < slots.Length; s++)
    {
        var g0 = got[s];
        if (g0.dtype.name != slots[s].GetProperty("dtype").GetString()) return $"DTYPE {g0.dtype.name}";
        var shape = slots[s].GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
        if (!g0.shape.SequenceEqual(shape)) return $"SHAPE ({string.Join(",", g0.shape)}) vs ({string.Join(",", shape)})";
        using var g = np.ascontiguousarray(g0);
        long bytes = g.size * g.dtypesize;
        var span = new ReadOnlySpan<byte>((byte*)g.Storage.Address + g.Shape.offset * g.dtypesize, checked((int)bytes));
        string hex = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(CanonicalNaN(span, g.typecode))).ToLowerInvariant();
        if (hex == slots[s].GetProperty("sha256").GetString())
            continue;
        allExact = false;
        if (!blas || !slots[s].TryGetProperty("hex", out var want))
            return "BYTES";
        // The values as doubles (a complex slot is its interleaved re/im parts), compared to NumPy's relative to the
        // largest magnitude — a NaN must be NaN on both sides.
        double[] Doubles(byte[] b) => g.typecode == NPTypeCode.Single
            ? Enumerable.Range(0, b.Length / 4).Select(i => (double)BitConverter.ToSingle(b, i * 4)).ToArray()
            : Enumerable.Range(0, b.Length / 8).Select(i => BitConverter.ToDouble(b, i * 8)).ToArray();
        if (g.typecode != NPTypeCode.Single && g.typecode != NPTypeCode.Double && g.typecode != NPTypeCode.Complex)
            return "BYTES";
        double[] mine = Doubles(span.ToArray()), theirs = Doubles(Convert.FromHexString(want.GetString()));
        double scale = theirs.Where(double.IsFinite).Select(Math.Abs).DefaultIfEmpty(0).Max();
        // float32 products accumulate in float32 on both sides (OpenBLAS's sdot vector kernel, the managed FMA kernel):
        // a 1000-term dot keeps ~1e-5 of the largest coefficient; float64 / complex128 keep ~1e-13.
        double tol = g.typecode == NPTypeCode.Single ? 1e-4 : 1e-12;
        for (int i = 0; i < mine.Length; i++)
        {
            bool ok = double.IsNaN(theirs[i]) ? double.IsNaN(mine[i])
                : double.IsInfinity(theirs[i]) ? mine[i] == theirs[i]
                : Math.Abs(mine[i] - theirs[i]) <= tol * scale;
            if (!ok) { allClose = false; break; }
        }
    }
    return allExact ? "exact" : allClose ? "close" : "FAR";
}

/// <summary>
///     The bytes with every NaN rewritten to the canonical quiet NaN of its lane width (per component for complex) — the
///     oracle's NaN tokenization, mirrored from polyalg_numpy.py's canonical_nan.
/// </summary>
/// <param name="bytes">A C-order element buffer.</param>
/// <param name="t">Its dtype.</param>
/// <returns>A canonicalized copy.</returns>
static byte[] CanonicalNaN(ReadOnlySpan<byte> bytes, NPTypeCode t)
{
    var b = bytes.ToArray();
    switch (t)
    {
        case NPTypeCode.Double:
        case NPTypeCode.Complex:
            for (int i = 0; i + 8 <= b.Length; i += 8)
                if (double.IsNaN(BitConverter.ToDouble(b, i)))
                    BitConverter.TryWriteBytes(b.AsSpan(i), BitConverter.Int64BitsToDouble(0x7FF8000000000000));
            break;
        case NPTypeCode.Single:
            for (int i = 0; i + 4 <= b.Length; i += 4)
                if (float.IsNaN(BitConverter.ToSingle(b, i)))
                    BitConverter.TryWriteBytes(b.AsSpan(i), BitConverter.Int32BitsToSingle(0x7FC00000));
            break;
        case NPTypeCode.Half:
            for (int i = 0; i + 2 <= b.Length; i += 2)
                if (Half.IsNaN(BitConverter.ToHalf(b, i)))
                    BitConverter.TryWriteBytes(b.AsSpan(i), (ushort)0x7E00);
            break;
    }
    return b;
}
