#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaluate_bench.cs — fusion gate: fused np.evaluate vs unfused np.* chains.
// NumSharp side of the benchmark/fusion subsystem (driven by fusion_sheet.py,
// which rewrites the #:project path above to the running checkout). The absolute
// path lets it also run directly:  dotnet run -c Release - < benchmark/fusion/evaluate_bench.cs
// Companion: evaluate_bench.py (NumPy absolutes on the same box).
// =============================================================================
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends.Iteration;

#if DEBUG
Console.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

var asm = typeof(np).Assembly;
var dbg = (System.Diagnostics.DebuggableAttribute?)Attribute.GetCustomAttribute(asm, typeof(System.Diagnostics.DebuggableAttribute));
if (dbg is { IsJITOptimizerDisabled: true })
{
    Console.WriteLine("FATAL: NumSharp.Core built Debug (JIT optimizer disabled) — rerun with -c Release");
    return;
}

const int N = 4_000_000;
var a = np.arange(N).astype(np.float64) + 1.0;
var b = (np.arange(N).astype(np.float64) % 977.0) + 2.0;
var c = np.arange(N).astype(np.float64) * 0.5;
var af = a.astype(np.float32);
var bf = b.astype(np.float32);

double Best(Func<double> run, int rounds = 9)
{
    double best = double.MaxValue;
    for (int i = 0; i < rounds; i++)
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        best = Math.Min(best, run());
    }

    return best;
}

double Time(Action body)
{
    var sw = Stopwatch.StartNew();
    body();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds;
}

// correctness cross-checks once before timing
{
    var fused = np.evaluate((NpyExpr)a * b + c);
    var unfused = a * b + c;
    for (int i = 0; i < 5; i++)
    {
        int idx = i * (N / 5) + 13;
        double d = Math.Abs(fused.GetDouble(idx) - unfused.GetDouble(idx));
        if (d > 1e-9) { Console.WriteLine($"MISMATCH a*b+c at {idx}: {d}"); return; }
    }

    var fusedNd = np.evaluate((NpyExpr.Arr(a) - b) / (NpyExpr.Arr(a) + b));
    var unfusedNd = (a - b) / (a + b);
    for (int i = 0; i < 5; i++)
    {
        int idx = i * (N / 5) + 13;
        double d = Math.Abs(fusedNd.GetDouble(idx) - unfusedNd.GetDouble(idx));
        if (d > 1e-12) { Console.WriteLine($"MISMATCH (a-b)/(a+b) at {idx}: {d}"); return; }
    }

    double sFused = np.evaluate(NpyExpr.Sum((NpyExpr)a * b)).GetDouble(0);
    double sUnfused = ((NDArray)np.sum(a * b)).GetDouble(0);
    if (Math.Abs(sFused - sUnfused) / Math.Abs(sUnfused) > 1e-12)
    {
        Console.WriteLine($"MISMATCH sum(a*b): {sFused} vs {sUnfused}");
        return;
    }

    Console.WriteLine("correctness cross-checks ok");
}

// warmup (kernel compile + caches)
np.evaluate((NpyExpr)a * b + c);
np.evaluate((NpyExpr.Arr(a) - b) / (NpyExpr.Arr(a) + b));
np.evaluate(NpyExpr.Sum((NpyExpr)a * b));
np.evaluate(NpyExpr.Sum((NpyExpr)af * bf));
_ = a * b + c;
_ = (a - b) / (a + b);
_ = np.sum(a * b);

Console.WriteLine($"\n4M float64, best of 9:");

double tFusedMulAdd = Best(() => Time(() => np.evaluate((NpyExpr)a * b + c)));
double tUnfusedMulAdd = Best(() => Time(() => { var _ = a * b + c; }));
Console.WriteLine($"  a*b+c       fused {tFusedMulAdd,7:F2} ms   unfused {tUnfusedMulAdd,7:F2} ms   ({tUnfusedMulAdd / tFusedMulAdd:F2}x)");

double tFusedNormDiff = Best(() => Time(() => np.evaluate((NpyExpr.Arr(a) - b) / (NpyExpr.Arr(a) + b))));
double tUnfusedNormDiff = Best(() => Time(() => { var _ = (a - b) / (a + b); }));
Console.WriteLine($"  (a-b)/(a+b) fused {tFusedNormDiff,7:F2} ms   unfused {tUnfusedNormDiff,7:F2} ms   ({tUnfusedNormDiff / tFusedNormDiff:F2}x)");

double tFusedSum = Best(() => Time(() => np.evaluate(NpyExpr.Sum((NpyExpr)a * b))));
double tUnfusedSum = Best(() => Time(() => { var _ = np.sum(a * b); }));
Console.WriteLine($"  sum(a*b)    fused {tFusedSum,7:F2} ms   unfused {tUnfusedSum,7:F2} ms   ({tUnfusedSum / tFusedSum:F2}x)");

double tFusedSumF4 = Best(() => Time(() => np.evaluate(NpyExpr.Sum((NpyExpr)af * bf))));
double tUnfusedSumF4 = Best(() => Time(() => { var _ = np.sum(af * bf); }));
Console.WriteLine($"  sum(af*bf)  fused {tFusedSumF4,7:F2} ms   unfused {tUnfusedSumF4,7:F2} ms   ({tUnfusedSumF4 / tFusedSumF4:F2}x)  [f32]");

// out= variant removes the result allocation from the fused path
var outArr = np.empty_like(a);
np.evaluate((NpyExpr)a * b + c, @out: outArr);
double tFusedOut = Best(() => Time(() => np.evaluate((NpyExpr)a * b + c, @out: outArr)));
Console.WriteLine($"  a*b+c out=  fused {tFusedOut,7:F2} ms");

// mixed dtype: i4 inputs promoting to f8 inside the kernel (no buffering)
var ai = np.arange(N).astype(np.int32);
np.evaluate((NpyExpr)ai * 2 + c);
double tFusedMixed = Best(() => Time(() => np.evaluate((NpyExpr)ai * 2 + c)));
double tUnfusedMixed = Best(() => Time(() => { var _ = ai * 2 + c; }));
Console.WriteLine($"  i4*2+f8     fused {tFusedMixed,7:F2} ms   unfused {tUnfusedMixed,7:F2} ms   ({tUnfusedMixed / tFusedMixed:F2}x)");
