#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaluate_bench.cs — fusion gate: fused np.evaluate vs unfused np.* chains, plus
// an operand-layout sweep of a*b+c (C/F/T/strided/bcast) checking the fused
// single-pass advantage survives non-contiguous operands.
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
    var fused = np.evaluate((NDExpr)a * b + c);
    var unfused = a * b + c;
    for (int i = 0; i < 5; i++)
    {
        int idx = i * (N / 5) + 13;
        double d = Math.Abs(fused.GetDouble(idx) - unfused.GetDouble(idx));
        if (d > 1e-9) { Console.WriteLine($"MISMATCH a*b+c at {idx}: {d}"); return; }
    }

    var fusedNd = np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
    var unfusedNd = (a - b) / (a + b);
    for (int i = 0; i < 5; i++)
    {
        int idx = i * (N / 5) + 13;
        double d = Math.Abs(fusedNd.GetDouble(idx) - unfusedNd.GetDouble(idx));
        if (d > 1e-12) { Console.WriteLine($"MISMATCH (a-b)/(a+b) at {idx}: {d}"); return; }
    }

    double sFused = np.evaluate(NDExpr.Sum((NDExpr)a * b)).GetDouble(0);
    double sUnfused = ((NDArray)np.sum(a * b)).GetDouble(0);
    if (Math.Abs(sFused - sUnfused) / Math.Abs(sUnfused) > 1e-12)
    {
        Console.WriteLine($"MISMATCH sum(a*b): {sFused} vs {sUnfused}");
        return;
    }

    Console.WriteLine("correctness cross-checks ok");
}

// warmup (kernel compile + caches)
np.evaluate((NDExpr)a * b + c);
np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b));
np.evaluate(NDExpr.Sum((NDExpr)a * b));
np.evaluate(NDExpr.Sum((NDExpr)af * bf));
_ = a * b + c;
_ = (a - b) / (a + b);
_ = np.sum(a * b);

Console.WriteLine($"\n4M float64, best of 9:");

double tFusedMulAdd = Best(() => Time(() => np.evaluate((NDExpr)a * b + c)));
double tUnfusedMulAdd = Best(() => Time(() => { var _ = a * b + c; }));
Console.WriteLine($"  a*b+c       fused {tFusedMulAdd,7:F2} ms   unfused {tUnfusedMulAdd,7:F2} ms   ({tUnfusedMulAdd / tFusedMulAdd:F2}x)");

double tFusedNormDiff = Best(() => Time(() => np.evaluate((NDExpr.Arr(a) - b) / (NDExpr.Arr(a) + b))));
double tUnfusedNormDiff = Best(() => Time(() => { var _ = (a - b) / (a + b); }));
Console.WriteLine($"  (a-b)/(a+b) fused {tFusedNormDiff,7:F2} ms   unfused {tUnfusedNormDiff,7:F2} ms   ({tUnfusedNormDiff / tFusedNormDiff:F2}x)");

double tFusedSum = Best(() => Time(() => np.evaluate(NDExpr.Sum((NDExpr)a * b))));
double tUnfusedSum = Best(() => Time(() => { var _ = np.sum(a * b); }));
Console.WriteLine($"  sum(a*b)    fused {tFusedSum,7:F2} ms   unfused {tUnfusedSum,7:F2} ms   ({tUnfusedSum / tFusedSum:F2}x)");

double tFusedSumF4 = Best(() => Time(() => np.evaluate(NDExpr.Sum((NDExpr)af * bf))));
double tUnfusedSumF4 = Best(() => Time(() => { var _ = np.sum(af * bf); }));
Console.WriteLine($"  sum(af*bf)  fused {tFusedSumF4,7:F2} ms   unfused {tUnfusedSumF4,7:F2} ms   ({tUnfusedSumF4 / tFusedSumF4:F2}x)  [f32]");

// out= variant removes the result allocation from the fused path
var outArr = np.empty_like(a);
np.evaluate((NDExpr)a * b + c, @out: outArr);
double tFusedOut = Best(() => Time(() => np.evaluate((NDExpr)a * b + c, @out: outArr)));
// NOTE: this is ONE fused pass into out=; the NumPy twin's "a*b+c out=" is a
// deliberately-handicapped TWO-pass (multiply→out, add→out) since NumPy has no
// native fusion — the two absolute-ms columns are NOT apples-to-apples. The
// allocating "a*b+c" rows above are the fair fused-vs-unfused comparison.
Console.WriteLine($"  a*b+c out=  fused {tFusedOut,7:F2} ms   [1-pass fused-into-out]");

// mixed dtype: i4 inputs promoting to f8 inside the kernel (no buffering)
var ai = np.arange(N).astype(np.int32);
np.evaluate((NDExpr)ai * 2 + c);
double tFusedMixed = Best(() => Time(() => np.evaluate((NDExpr)ai * 2 + c)));
double tUnfusedMixed = Best(() => Time(() => { var _ = ai * 2 + c; }));
Console.WriteLine($"  i4*2+f8     fused {tFusedMixed,7:F2} ms   unfused {tUnfusedMixed,7:F2} ms   ({tUnfusedMixed / tFusedMixed:F2}x)");

// -------- operand-layout sweep ------------------------------------------------
// The fusion premise is "read each operand once in ONE NDIter pass". That is
// most stressed on NON-contiguous operands: does the fused advantage survive a
// strided / F / broadcast operand, or does the fused kernel fall back/buffer and
// collapse to the unfused chain? The 1-D cases above never answer this. Here the
// flagship a*b+c runs 2-D (2000x2000 = 4M) with all three operands in the SAME
// layout. The unfused÷fused ratio is self-normalizing per layout, so strided's
// (2M) / bcast's (stride-0) differing element counts don't distort the headline.
const int LR = 2000, LC = 2000;
var a2 = (np.arange(LR * LC).astype(np.float64) + 1.0).reshape(LR, LC);
var b2 = (np.arange(LR * LC).astype(np.float64) % 977.0 + 2.0).reshape(LR, LC);
var c2 = (np.arange(LR * LC).astype(np.float64) * 0.5).reshape(LR, LC);
NDArray Lay(NDArray x, string l) => l switch
{
    "C" => x, "F" => x.copy(order: 'F'), "T" => x.T,
    "strided" => x[":, ::2"],
    "bcast" => np.broadcast_to(x["0:1, :"], new Shape(LR, LC)),
    _ => throw new Exception(l),
};
Console.WriteLine($"\n  a*b+c across operand layouts (2-D {LR}x{LC}, all 3 operands same layout):");
foreach (var l in new[] { "C", "F", "T", "strided", "bcast" })
{
    NDArray al, bl, cl;
    try { al = Lay(a2, l); bl = Lay(b2, l); cl = Lay(c2, l); }
    catch (Exception e) { Console.WriteLine($"    [{l,-7}] build: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); continue; }
    try
    {
        np.evaluate((NDExpr)al * bl + cl);                                  // warm + throw gate
        double tf = Best(() => Time(() => np.evaluate((NDExpr)al * bl + cl)));
        double tu = Best(() => Time(() => { var _ = al * bl + cl; }));
        Console.WriteLine($"    [{l,-7}] fused {tf,7:F2} ms   unfused {tu,7:F2} ms   ({tu / tf:F2}x)");
    }
    catch (Exception e) { Console.WriteLine($"    [{l,-7}] {e.GetType().Name}: {e.Message.Split('\n')[0]}"); }
}
