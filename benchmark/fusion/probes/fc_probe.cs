#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true

// =============================================================================
// fc_probe — np.evaluate fixed cost + the library-consumer shapes (sinc / windows), measured.
//
// Compiles against ANY NumSharp.Core that has np.evaluate (master's old NDExpr, the exprs merge,
// and the structural-cache build), so the same rows can be compared across the three states.
//
// Run (pin to one P-core, kill the tier-0 quiet window):
//   NS_PROBE_AFFINITY=4 DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release fc_probe_<target>.cs [DGC]
//
// Sections:
//   D. fixed cost per call at n = 8 (ns + managed B/call): rebuilt vs prebuilt tree, out=, reductions,
//      where, the 0-d parameter form, and the unfused chains they replace.
//   G. the library consumers' trees (np.sinc / np.hanning / np.kaiser build their tree INLINE on every
//      call): the inline tree vs the unfused chain at 1K / 100K, plus the real np.hanning / np.kaiser.
//   C. the 1K "fused >= unfused" rows with the tree REBUILT per call (what user code naturally writes).
// =============================================================================
using System.Diagnostics;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

if ((Environment.GetEnvironmentVariable("NUMSHARP_BENCHMARK_AFFINITY")
     ?? Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY")) is { Length: > 0 } hostAffinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(hostAffinity, 16);

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug-JITted NumSharp.Core — run with -c Release"); return; }
Console.WriteLine($"core={typeof(np).Assembly.Location}");
Console.WriteLine($"V256={Vector256.IsHardwareAccelerated} V512={Vector512.IsHardwareAccelerated} TC_CallCountingDelayMs={Environment.GetEnvironmentVariable("DOTNET_TC_CallCountingDelayMs") ?? "(default 100)"} affinity={Process.GetCurrentProcess().ProcessorAffinity}");

string sections = args.Length > 0 ? args[0].ToUpperInvariant() : "DGC";

// ------------------------------------------------------------------ helpers
double BestNs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 120) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(5, (int)Math.Ceiling(200.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e6 / it); }
    return best;
}
long AllocPerCall(Action body, int iters)
{
    for (int i = 0; i < 200; i++) body();
    long before = GC.GetAllocatedBytesForCurrentThread();
    for (int i = 0; i < iters; i++) body();
    return (GC.GetAllocatedBytesForCurrentThread() - before) / iters;
}
void Row(string id, Action body, int allocIters = 20_000)
{
    body();
    Console.WriteLine($"{id,-46}\t{BestNs(body),9:F0}\t{AllocPerCall(body, allocIters),7}");
}
(NDArray a, NDArray b, NDArray c) Data(long n)
{
    var idx = np.arange(n).astype(NPTypeCode.Double);
    var a = (idx % 977.0) / 977.0;
    var b = ((idx * 7.0) % 991.0) / 991.0 + 0.5;
    var c = ((idx * 13.0) % 983.0) / 983.0 * 0.5;
    return (a, b, c);
}

// =============================================================================
// D. fixed cost per call (n = 8)
// =============================================================================
if (sections.Contains('D'))
{
    Console.WriteLine("\n=== D. fixed cost (n=8): ns/call, managed B/call ===");
    var (a, b, c) = Data(8);
    var expr = (NDExpr)a * b + c;              // prebuilt tree
    var sumExpr = NDExpr.Sum((NDExpr)a * b);
    var whereExpr = NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b));
    var k = NDArray.Scalar(2.5);
    var paramExpr = (NDExpr)a * b + k;
    var o = np.empty_like(a);
    Console.WriteLine($"{"case",-46}\t{"ns",9}\t{"B/call",7}");
    Row("evaluate(a*b+c) REBUILT per call", () => np.evaluate((NDExpr)a * b + c));
    Row("evaluate(expr) prebuilt", () => np.evaluate(expr));
    Row("evaluate(expr, out=) prebuilt", () => np.evaluate(expr, @out: o));
    Row("evaluate(a*b+c, out=) REBUILT", () => np.evaluate((NDExpr)a * b + c, @out: o));
    Row("unfused a*b+c", () => { var _ = a * b + c; });
    Row("unfused multiply/add out= (2 passes)", () => { np.multiply(a, b, @out: o); np.add(o, c, @out: o); });
    Row("evaluate(Sum(a*b)) REBUILT", () => np.evaluate(NDExpr.Sum((NDExpr)a * b)));
    Row("evaluate(Sum(a*b)) prebuilt", () => np.evaluate(sumExpr));
    Row("unfused np.sum(a*b)", () => { var _ = np.sum(a * b); });
    Row("evaluate(Where(a>b,a,b)) REBUILT", () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b))));
    Row("evaluate(Where(a>b,a,b)) prebuilt", () => np.evaluate(whereExpr));
    Row("unfused np.where(a>b,a,b)", () => { var _ = np.where(a > b, a, b); });
    Row("evaluate(a*b+k) 0-d param REBUILT", () => np.evaluate((NDExpr)a * b + k));
    Row("evaluate(a*b+k) 0-d param prebuilt", () => np.evaluate(paramExpr));
    Row("unfused a*b+2.5", () => { var _ = a * b + 2.5; });
}

// =============================================================================
// G. the library consumers' trees (inline-rebuilt, as np.sinc / np.windows spell them)
// =============================================================================
if (sections.Contains('G'))
{
    Console.WriteLine("\n=== G. consumer trees: ns/call, managed B/call (tree rebuilt per call, as the library does) ===");
    Console.WriteLine($"{"case",-46}\t{"ns",9}\t{"B/call",7}");
    foreach (long n in new long[] { 1_000, 100_000 })
    {
        var (a, b, _) = Data(n);
        var x = a - 0.5;                                                     // sinc input crossing zero
        var m = np.arange(1 - n, n, 2).astype(NPTypeCode.Double);            // hanning's `n` vector (float64)
        double M = n;
        double eps = double.Epsilon;                                          // stand-in for finfo.eps
        int allocIters = n == 1_000 ? 20_000 : 500;
        // np.sinc: t = pi*x; y = where(t != 0, t, eps); sin(y)/y  (ONE fused pass, tree built inline)
        Row($"sinc-tree fused inline n={n}", () =>
        {
            NDExpr t = (NDExpr)x * Math.PI;
            NDExpr y = NDExpr.Where(NDExpr.NotEqual(t, 0.0), t, eps);
            np.evaluate(NDExpr.Sin(y) / y);
        }, allocIters);
        Row($"sinc unfused chain n={n}", () =>
        {
            var t = x * Math.PI;
            var y = np.where(t != 0.0, t, eps);
            var _ = np.sin(y) / y;
        }, allocIters);
        // np.hanning: 0.5 + 0.5*cos(pi*n/(M-1))  (tree built inline)
        Row($"hanning-tree fused inline n={n}", () => np.evaluate(0.5 + 0.5 * NDExpr.Cos(Math.PI * (NDExpr)m / (M - 1))), allocIters);
        Row($"hanning unfused chain n={n}", () => { var _ = 0.5 + 0.5 * np.cos(Math.PI * m / (M - 1)); }, allocIters);
        Row($"np.hanning({n}) (arange + fused tree)", () => np.hanning(n), allocIters);
        Row($"np.kaiser({n}, 5) (arange + Call + fused)", () => np.kaiser(n, 5.0), allocIters);
        Row($"np.bartlett({n}) (where-tree fused)", () => np.bartlett(n), allocIters);
    }
}

// =============================================================================
// C. the 1K rows, tree rebuilt per call: fused ns vs unfused ns (unfused/fused > 1 = fused wins)
// =============================================================================
if (sections.Contains('C'))
{
    Console.WriteLine("\n=== C. 1K rows, tree REBUILT per call: fused_ns unfused_ns unfused/fused ===");
    long n = 1_000;
    var (a, b, c) = Data(n);
    var af = a.astype(NPTypeCode.Single); var bf = b.astype(NPTypeCode.Single);
    var i4 = (np.arange(n) % 1000).astype(NPTypeCode.Int32);
    var i8 = np.arange(n) % 1000;
    var a2 = a.reshape(10, 100); var b2 = b.reshape(10, 100);
    var cases = new (string id, Action fused, Action unfused)[]
    {
        ("a*b+c f64",          () => np.evaluate((NDExpr)a * b + c),                                              () => { var _ = a * b + c; }),
        ("(a-b)/(a+b) f64",    () => np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b)),                    () => { var _ = (a - b) / (a + b); }),
        ("sqrt(a*a+b*b) f64",  () => np.evaluate(NDExpr.Sqrt((NDExpr)a * a + (NDExpr)b * b)),                    () => { var _ = np.sqrt(a * a + b * b); }),
        ("where(a>b,a,b) f64", () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), b), NDExpr.Arr(a), NDExpr.Arr(b))), () => { var _ = np.where(a > b, a, b); }),
        ("maximum(a,b) f64",   () => np.evaluate(NDExpr.Max(NDExpr.Arr(a), NDExpr.Arr(b))),                     () => { var _ = np.maximum(a, b); }),
        ("a>0.5 (bool out)",   () => np.evaluate(NDExpr.Greater(NDExpr.Arr(a), 0.5)),                            () => { var _ = a > 0.5; }),
        ("(a>0.2)&(b<0.8)",    () => np.evaluate(NDExpr.Greater(NDExpr.Arr(a), 0.2) & NDExpr.Less(NDExpr.Arr(b), 0.8)), () => { var _ = (a > 0.2) & (b < 0.8); }),
        ("leaky relu f64",     () => np.evaluate(NDExpr.Where(NDExpr.Greater(NDExpr.Arr(a), 0.5), NDExpr.Arr(a), (NDExpr)a * 0.01)), () => { var _ = np.where(a > 0.5, a, a * 0.01); }),
        ("exp(af)*bf f32",     () => np.evaluate(NDExpr.Exp(NDExpr.Arr(af)) * bf),                               () => { var _ = np.exp(af) * bf; }),
        ("i4*2+f8 mixed",      () => np.evaluate((NDExpr)i4 * 2 + c),                                             () => { var _ = i4 * 2 + c; }),
        ("i8*i8+1 int64",      () => np.evaluate((NDExpr)i8 * i8 + 1),                                            () => { var _ = i8 * i8 + 1; }),
        ("abs(a) f64",         () => np.evaluate(NDExpr.Abs(NDExpr.Arr(a))),                                     () => { var _ = np.abs(a); }),
        ("sum(a*b) f64",       () => np.evaluate(NDExpr.Sum((NDExpr)a * b)),                                     () => { var _ = np.sum(a * b); }),
        ("mean((a-b)^2) f64",  () => np.evaluate(NDExpr.Mean((NDExpr.Arr(a) - b) * (NDExpr.Arr(a) - b))),        () => { var _ = np.mean((a - b) * (a - b)); }),
        ("sum(a2*b2,ax1)",     () => np.evaluate(NDExpr.Sum((NDExpr)a2 * b2, 1)),                                () => { var _ = np.sum(a2 * b2, 1); }),
    };
    Console.WriteLine($"{"case",-22}\t{"fused_ns",9}\t{"unfused_ns",10}\t{"unf/fused",9}\t{"fusedB",7}");
    int wins = 0;
    foreach (var (id, fused, unfused) in cases)
    {
        fused(); unfused();
        double tf = BestNs(fused); double tu = BestNs(unfused);
        long bf_ = AllocPerCall(fused, 5_000);
        if (tu >= tf) wins++;
        Console.WriteLine($"{id,-22}\t{tf,9:F0}\t{tu,10:F0}\t{tu / tf,9:F2}\t{bf_,7}");
    }
    Console.WriteLine($"fused >= unfused on {wins}/{cases.Length} rows");
}

Console.WriteLine("\ndone.");
