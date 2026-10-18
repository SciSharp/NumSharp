#:project ../../src/NumSharp.Core/NumSharp.Core.csproj
#:project ../../src/NumSharp.Interop.OpenBLAS/NumSharp.Interop.OpenBLAS.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// polyclass_bench.cs — numpy.polynomial's six series classes (U10: NDPolynomial, NDChebyshev, NDLegendre, NDLaguerre,
// NDHermite, NDHermiteE over NDPolyBase, the port of ABCPolyBase) and their printers (U11's class half):
// NumSharp vs NumPy 2.4.2
// =============================================================================
// Replays every cell polyclass_numpy.py wrote (data/polyclass/manifest.json), rebuilds the cell's inputs from the shared
// formula (polyeval_numpy.gen), checks the SHA-256 of its result against NumPy's BEFORE timing it, then prints a TSV
// joined with NumPy's timings:
//   cell, NumSharp us, NumPy us, NPY/NS (higher = NumSharp faster), byte check
// followed by per-section summaries (min / geomean NPY/NS) and every cell below 1.5x.
//
// Usage (from benchmark/polynomial; pin both sides with the same mask, never run them concurrently; three alternating
// pairs, the report keeping each cell's best window of each side):
//   set NS_PROBE_AFFINITY=0x14
//   set OPENBLAS_NUM_THREADS=1
//   set DOTNET_TC_CallCountingDelayMs=0
//   python polyclass_numpy.py > numpy_class_times_1.tsv
//   dotnet run -c Release --no-cache polyclass_bench.cs -- data/polyclass numpy_class_times_1.tsv > ns_class_1.tsv
//   (the same for _2 and _3)
//   python polyclass_report.py ns_class_1.tsv ns_class_2.tsv ns_class_3.tsv > polyclass_results.md
// Release is mandatory (benchmark/CLAUDE.md), --no-cache after any NumSharp.Core edit, and PublishAot=false is
// load-bearing (the polynomial kernels are DynamicMethods). The OpenBLAS backend is installed at ONE thread for the whole
// run: roots (eigvals) and fit (lstsq) need LAPACK, and the products of the power and Chebyshev series run np.convolve,
// whose long dots NumPy computes through OpenBLAS's ?dot — byte parity with NumPy needs the same library on both sides.
// =============================================================================
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NumSharp;
using NumSharp.Interop.OpenBLAS;

if (Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY") is { Length: > 0 } a1)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a1, 16);
else if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } a2)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(a2, 16);

OpenBlasEngine.Enable(threads: 1);
// The str cells are NumPy's str() under the ascii style (polyclass_numpy._ascii_str): the class flag, set once.
np.polynomial.set_default_printstyle("ascii");

string data = args.Length > 0 ? args[0] : Path.Combine("data", "polyclass");
var numpyUs = new Dictionary<string, double>(StringComparer.Ordinal);
if (args.Length > 1)
    foreach (var line in File.ReadLines(args[1]))
    {
        var parts = line.Split('\t');
        if (parts.Length == 2 && double.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var us))
            numpyUs[parts[0]] = us;
    }

using var manifest = JsonDocument.Parse(File.ReadAllText(Path.Combine(data, "manifest.json")));
var cells = manifest.RootElement.EnumerateArray().ToList();
var calls0 = cells.Select(MakeCall).ToList();

// Warm-up: every cell's call a few dozen times, twice, with a pause for the tier-1 background compiler to drain its
// queue (one pass only QUEUES the methods). Each result is disposed: the timed loops below must not run against a heap
// full of finalizable arrays left by the warm-up.
for (int pass = 0; pass < 2; pass++)
{
    foreach (var f in calls0)
        for (int w = 0; w < 20; w++) Release(f());
    System.Threading.Thread.Sleep(250);
}

var rows = new List<(string Label, double Ns, double Np, string Check)>();
Console.WriteLine("cell\tnumsharp_us\tnumpy_us\tNPY/NS\tcheck");
for (int ci = 0; ci < cells.Count; ci++)
{
    var cell = cells[ci];
    string label = cell.GetProperty("label").GetString();
    int calls = cell.GetProperty("calls").GetInt32(), rounds = cell.GetProperty("rounds").GetInt32();
    var call = calls0[ci];
    object first = call();
    string check = Digest(first) == cell.GetProperty("sha256").GetString() ? "exact" : "BYTES";
    Release(first);

    double best = double.MaxValue;
    var sw = new Stopwatch();
    for (int round = 0; round < rounds; round++)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) Release(call());
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000.0 / calls);
    }
    double npv = numpyUs.TryGetValue(label, out var v) ? v : double.NaN;
    rows.Add((label, best, npv, check));
    Console.WriteLine($"{label}\t{best:F3}\t{npv:F3}\t{npv / best:F2}\t{check}");
}
np.polynomial.set_default_printstyle(OperatingSystem.IsWindows() ? "ascii" : "unicode");
OpenBlasEngine.Disable();

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

// ---------------------------------------------------------------------------------------------
//  Inputs, calls and digests
// ---------------------------------------------------------------------------------------------

/// <summary>The shared float64 input formula (polyeval_numpy.gen): t = (i*7919 + seed*104729) mod 20011, t/10005.5 - 1.</summary>
/// <param name="n">The length.</param><param name="seed">The seed.</param>
/// <returns>The array (every step exact or one correctly rounded operation, as in NumPy).</returns>
static NDArray Vec(long n, long seed)
{
    var v = new double[n];
    for (long i = 0; i < n; i++)
        v[i] = ((i * 7919 + seed * 104729) % 20011) / 10005.5 - 1.0;
    return np.array(v);
}

/// <summary>The domain every series of a cell has (polyclass_numpy.DOMAIN), as NumPy's Python list.</summary>
static object[] Domain() => new object[] { -2.0, 3.0 };

/// <summary>A series of a class: its (coef, domain) constructor.</summary>
/// <param name="cls">The class name.</param><param name="coef">The coefficients.</param><param name="domain">The domain, or null.</param>
/// <returns>The series.</returns>
/// <exception cref="NotSupportedException">An unknown class.</exception>
static NDPolyBase New(string cls, object coef, object domain) => cls switch
{
    "Polynomial" => new NDPolynomial(coef, domain),
    "Chebyshev" => new NDChebyshev(coef, domain),
    "Legendre" => new NDLegendre(coef, domain),
    "Laguerre" => new NDLaguerre(coef, domain),
    "Hermite" => new NDHermite(coef, domain),
    "HermiteE" => new NDHermiteE(coef, domain),
    _ => throw new NotSupportedException(cls),
};

/// <summary>A class by name.</summary>
/// <param name="cls">The class name.</param><returns>Its CLR type.</returns>
static Type ClassType(string cls) => New(cls, new object[] { 1.0 }, null).GetType();

/// <summary>Builds a cell's inputs once (NumPy's closures capture theirs) and returns the timed call.</summary>
/// <param name="cell">The manifest entry.</param>
/// <returns>The call, returning the result the digest reads (and <see cref="Release"/> disposes).</returns>
/// <exception cref="NotSupportedException">An unknown operation.</exception>
static Func<object> MakeCall(JsonElement cell)
{
    string cls = cell.GetProperty("cls").GetString(), op = cell.GetProperty("op").GetString();
    var a = cell.GetProperty("args");
    int I(string n) => a.GetProperty(n).GetInt32();
    double D(string n) => a.GetProperty(n).GetDouble();
    NDPolyBase S(long n, long seed) => New(cls, Vec(n, seed), Domain());
    switch (op)
    {
        case "ctor":
        {
            var c = Vec(I("n"), 1);
            return () => New(cls, c, Domain());
        }
        case "identity": return () => Static(cls, "identity");
        case "basis": return () => Static(cls, "basis", I("deg"));
        case "fromroots":
        {
            var r = Vec(I("n"), 3);
            return () => Static(cls, "fromroots", r);
        }
        case "cast":
        {
            var src = New(a.GetProperty("from").GetString(), Vec(I("n"), 1), Domain());
            return () => Static(cls, "cast", src);
        }
        case "add": { var p = S(I("n"), 1); var q = S(I("n"), 2); return () => p + q; }
        case "mul": { var p = S(I("n"), 1); var q = S(I("n"), 2); return () => p * q; }
        case "add_float": { var p = S(I("n"), 1); double x = D("v"); return () => p + x; }
        case "mul_float": { var p = S(I("n"), 1); double x = D("v"); return () => p * x; }
        case "truediv_float": { var p = S(I("n"), 1); double x = D("v"); return () => p / x; }
        case "floordiv": { var p = S(I("n"), 1); var q = S(I("m"), 2); return () => p.__floordiv__(q); }
        case "divmod": { var p = S(I("n"), 1); var q = S(I("m"), 2); return () => p.__divmod__(q); }
        case "pow": { var p = S(I("n"), 1); object e = (long)I("e"); return () => p.__pow__(e); }
        case "neg": { var p = S(I("n"), 1); return () => -p; }
        case "eq": { var p = S(I("n"), 1); var q = S(I("n"), 1); return () => p == q; }
        case "call": { var p = S(I("n"), 1); var x = Vec(I("m"), 5); return () => p.Call(x); }
        case "call_float": { var p = S(I("n"), 1); double x = D("v"); return () => p.Call(x); }
        case "compose": { var p = S(I("n"), 1); var q = S(I("m"), 2); return () => p.Call(q); }
        case "deriv": { var p = S(I("n"), 1); return () => p.deriv(); }
        case "integ": { var p = S(I("n"), 1); return () => p.integ(); }
        case "integ_k_lbnd": { var p = S(I("n"), 1); object k = new object[] { 1.0, -1.0 }; return () => p.integ(2, k, 0.5); }
        case "convert": { var p = S(I("n"), 1); var kind = ClassType(a.GetProperty("kind").GetString()); return () => p.convert(kind: kind); }
        case "linspace": { var p = S(I("n"), 1); long m = I("m"); return () => p.linspace(m); }
        case "copy": { var p = S(I("n"), 1); return () => p.copy(); }
        case "trim": { var p = S(I("n"), 1); return () => p.trim(); }
        case "cutdeg": { var p = S(I("n"), 1); object d = (long)I("deg"); return () => p.cutdeg(d); }
        case "truncate": { var p = S(I("n"), 1); object z = (long)I("size"); return () => p.truncate(z); }
        case "mapparms": { var p = S(I("n"), 1); return () => p.mapparms(); }
        case "degree": { var p = S(I("n"), 1); return () => p.degree(); }
        case "str": { var p = S(I("n"), 1); return () => p.__str__(); }
        case "repr": { var p = S(I("n"), 1); return () => p.__repr__(); }
        case "latex": { var p = S(I("n"), 1); return () => p._repr_latex_(); }
        case "format_unicode": { var p = S(I("n"), 1); return () => p.__format__("unicode"); }
        case "getstate": { var p = S(I("n"), 1); return () => p.__getstate__(); }
        case "setstate":
        {
            var p = S(I("n"), 1);
            var state = p.__getstate__();
            var target = S(3, 9);
            // The target is the result each call returns; Release must not dispose it (it is reused by the next call).
            return () => { target.__setstate__(state); return new Held(target); };
        }
        case "roots": { var p = S(I("n"), 1); return () => p.roots(); }
        case "fit":
        {
            var x = Vec(I("m"), 6);
            var y = Vec(I("m"), 7);
            object deg = (long)I("deg");
            return () => Static(cls, "fit", x, y, deg);
        }
        default: throw new NotSupportedException(op);
    }
}

/// <summary>A class method through the concrete class's own static member.</summary>
/// <param name="cls">The class.</param><param name="name">The method.</param><param name="a">Its arguments.</param>
/// <returns>The series.</returns>
/// <exception cref="NotSupportedException">An unknown class or method.</exception>
static NDPolyBase Static(string cls, string name, params object[] a)
{
    object d = Domain();
    return (cls, name) switch
    {
        ("Polynomial", "identity") => NDPolynomial.identity(d), ("Chebyshev", "identity") => NDChebyshev.identity(d),
        ("Legendre", "identity") => NDLegendre.identity(d), ("Laguerre", "identity") => NDLaguerre.identity(d),
        ("Hermite", "identity") => NDHermite.identity(d), ("HermiteE", "identity") => NDHermiteE.identity(d),
        ("Polynomial", "basis") => NDPolynomial.basis(a[0], d), ("Chebyshev", "basis") => NDChebyshev.basis(a[0], d),
        ("Legendre", "basis") => NDLegendre.basis(a[0], d), ("Laguerre", "basis") => NDLaguerre.basis(a[0], d),
        ("Hermite", "basis") => NDHermite.basis(a[0], d), ("HermiteE", "basis") => NDHermiteE.basis(a[0], d),
        ("Polynomial", "fromroots") => NDPolynomial.fromroots(a[0], d), ("Chebyshev", "fromroots") => NDChebyshev.fromroots(a[0], d),
        ("Legendre", "fromroots") => NDLegendre.fromroots(a[0], d), ("Laguerre", "fromroots") => NDLaguerre.fromroots(a[0], d),
        ("Hermite", "fromroots") => NDHermite.fromroots(a[0], d), ("HermiteE", "fromroots") => NDHermiteE.fromroots(a[0], d),
        ("Polynomial", "cast") => NDPolynomial.cast((NDPolyBase)a[0], d), ("Chebyshev", "cast") => NDChebyshev.cast((NDPolyBase)a[0], d),
        ("Legendre", "cast") => NDLegendre.cast((NDPolyBase)a[0], d), ("Laguerre", "cast") => NDLaguerre.cast((NDPolyBase)a[0], d),
        ("Hermite", "cast") => NDHermite.cast((NDPolyBase)a[0], d), ("HermiteE", "cast") => NDHermiteE.cast((NDPolyBase)a[0], d),
        ("Polynomial", "fit") => NDPolynomial.fit(a[0], a[1], a[2]), ("Chebyshev", "fit") => NDChebyshev.fit(a[0], a[1], a[2]),
        ("Legendre", "fit") => NDLegendre.fit(a[0], a[1], a[2]), ("Laguerre", "fit") => NDLaguerre.fit(a[0], a[1], a[2]),
        ("Hermite", "fit") => NDHermite.fit(a[0], a[1], a[2]), ("HermiteE", "fit") => NDHermiteE.fit(a[0], a[1], a[2]),
        _ => throw new NotSupportedException($"{cls}.{name}"),
    };
}

/// <summary>
///     The SHA-256 polyclass_numpy.digest computes: a series by its coef, domain and window bytes, an array by its C-order
///     bytes, a str by its UTF-8, a bool as one byte, an integer as 8 little-endian bytes, a double as its 8 bytes, a tuple
///     or a state dictionary by its parts in order (a dictionary key by its UTF-8).
/// </summary>
/// <param name="r">The result.</param>
/// <returns>The hex digest.</returns>
static string Digest(object r)
{
    using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
    void Feed(object x)
    {
        switch (x)
        {
            case Held held: Feed(held.Series); break;
            case NDPolyBase s: Feed(s.coef); Feed(s.domain); Feed(s.window); break;
            case NDArray nd: h.AppendData(nd.tobytes()); break;
            case string t: h.AppendData(Encoding.UTF8.GetBytes(t)); break;
            case bool b: h.AppendData(new[] { b ? (byte)1 : (byte)0 }); break;
            case long l: h.AppendData(BitConverter.GetBytes(l)); break;
            case int i: h.AppendData(BitConverter.GetBytes((long)i)); break;
            case double d: h.AppendData(BitConverter.GetBytes(d)); break;
            case OrderedDictionary od:
                foreach (DictionaryEntry e in od) { h.AppendData(Encoding.UTF8.GetBytes((string)e.Key)); Feed(e.Value); }
                break;
            case System.Runtime.CompilerServices.ITuple tup:
                for (int k = 0; k < tup.Length; k++) Feed(tup[k]);
                break;
            default: throw new NotSupportedException($"no digest for {x?.GetType().Name ?? "null"}");
        }
    }
    Feed(r);
    return Convert.ToHexString(h.GetHashAndReset()).ToLowerInvariant();
}

/// <summary>Disposes a call's result: a series, an array, a tuple of them, a state dictionary's arrays.</summary>
/// <param name="r">The result.</param>
static void Release(object r)
{
    switch (r)
    {
        case Held: break;   // the setstate target, reused by the next call
        case IDisposable d: d.Dispose(); break;
        case OrderedDictionary od:
            foreach (DictionaryEntry e in od) Release(e.Value);
            break;
        case System.Runtime.CompilerServices.ITuple tup:
            for (int k = 0; k < tup.Length; k++) Release(tup[k]);
            break;
    }
}

/// <summary>A result the call does not own (the setstate cell's reused target): digested, never disposed.</summary>
/// <param name="Series">The series.</param>
sealed record Held(NDPolyBase Series);
