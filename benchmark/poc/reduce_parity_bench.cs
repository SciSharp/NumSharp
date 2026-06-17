#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Numerics;
using NumSharp;

// Live np.* complex axis-reduction parity bench (mirrors reduce_parity_bench.py).
// Prints "size|op|axisN|layout|ms" lines for join against the NumPy output.

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.Error.WriteLine($"[opt] core={optCore}");

static double Bench(Action f, int it)
{
    for (int i = 0; i < 3; i++) f();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var ts = new double[it];
    for (int i = 0; i < it; i++)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        f();
        sw.Stop();
        ts[i] = sw.Elapsed.TotalMilliseconds;
    }
    Array.Sort(ts);
    return ts[it / 2];
}

static NDArray Run(string op, NDArray a, int ax) => op switch
{
    "sum"  => np.sum(a, axis: ax),
    "prod" => np.prod(a, axis: ax),
    "min"  => np.amin(a, axis: ax),
    "max"  => np.amax(a, axis: ax),
    _ => throw new Exception()
};

var sizes = new (int r, int c, string lbl)[] { (316,316,"100K"), (1000,1000,"1M"), (3162,3162,"10M") };
foreach (var (r, c, lbl) in sizes)
{
    long n = (long)r * c;
    var re = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7;
    var im = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0006 + 0.3;
    var aC = (re.astype(NPTypeCode.Complex)) + (im.astype(NPTypeCode.Complex) * new Complex(0, 1));
    var aT = aC.T;
    int it = n <= 1_000_000 ? 50 : 15;
    foreach (var op in new[]{"sum","prod","min","max"})
    {
        for (int ax = 0; ax < 2; ax++)
        {
            double ms = Bench(() => { using var rr = Run(op, aC, ax); }, it);
            Console.WriteLine($"{lbl}|{op}|axis{ax}|C|{ms:F4}");
        }
        for (int ax = 0; ax < 2; ax++)
        {
            double ms = Bench(() => { using var rr = Run(op, aT, ax); }, it);
            Console.WriteLine($"{lbl}|{op}|axis{ax}|T|{ms:F4}");
        }
    }
}
