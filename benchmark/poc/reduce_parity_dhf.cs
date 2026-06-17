#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using NumSharp;
using NumSharp.Backends.Iteration;

// Decimal / Half / Fused parity (complex covered by reduce_parity_full).
// Prod-safe data (values ~1.0) so products don't overflow. Mirrors reduce_parity_dhf.py.

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.Error.WriteLine($"[opt] core={optCore}");

static double Bench(Action f, int it)
{
    for (int i = 0; i < 3; i++) f();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var ts = new double[it];
    for (int i = 0; i < it; i++)
    { var sw = System.Diagnostics.Stopwatch.StartNew(); f(); sw.Stop(); ts[i] = sw.Elapsed.TotalMilliseconds; }
    Array.Sort(ts);
    return ts[it / 2];
}
static NDArray Reduce(string op, NDArray a, int ax) => op switch
{
    "sum" => np.sum(a, axis: ax), "prod" => np.prod(a, axis: ax),
    "min" => np.amin(a, axis: ax), "max" => np.amax(a, axis: ax),
    "mean" => np.mean(a, axis: ax), _ => throw new Exception()
};

var sizes = new (int r, int c, string lbl)[] { (316,316,"100K"), (1000,1000,"1M") };

foreach (var (r, c, lbl) in sizes)
{
    long n = (long)r * c;
    var baseD = ((np.arange(n) % 11).astype(NPTypeCode.Double) - 5.0) * 0.01 + 1.0; // 0.95..1.05
    baseD = baseD.reshape(r, c);
    int it = n <= 1_000_000 ? 25 : 10;

    var aDec = baseD.astype(NPTypeCode.Decimal);
    foreach (var op in new[] { "sum", "prod", "min", "max", "mean" })
        for (int ax = 0; ax < 2; ax++)
            Console.WriteLine($"decimal|{op}|axis{ax}|C|{lbl}|{Bench(() => { using var rr = Reduce(op, aDec, ax); }, it):F4}");

    var aHalf = baseD.astype(NPTypeCode.Half);
    foreach (var op in new[] { "sum", "prod", "min", "max", "mean" })
        for (int ax = 0; ax < 2; ax++)
            Console.WriteLine($"half|{op}|axis{ax}|C|{lbl}|{Bench(() => { using var rr = Reduce(op, aHalf, ax); }, it):F4}");
}

// Fused (Phase 5a): evaluate(reduce(a*b, axis)) vs NumPy reduce(a*b, axis)
var fsizes = new (int r, int c, string lbl)[] { (316,316,"100K"), (1000,1000,"1M"), (3162,3162,"10M") };
foreach (var (r, c, lbl) in fsizes)
{
    long n = (long)r * c;
    var a = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7;
    var b = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0006 + 0.3;
    int it = n <= 1_000_000 ? 40 : 12;
    foreach (var op in new[] { "sum", "mean", "max" })
        for (int ax = 0; ax < 2; ax++)
        {
            int axc = ax;
            NpyExpr Mk() { NpyExpr e = (NpyExpr)a * b; return op switch {
                "sum" => NpyExpr.Sum(e, axc), "mean" => NpyExpr.Mean(e, axc), "max" => NpyExpr.Max(e, axc), _ => throw new Exception() }; }
            Console.WriteLine($"fused|{op}|axis{ax}|C|{lbl}|{Bench(() => { using var rr = np.evaluate(Mk()); }, it):F4}");
        }
}
