#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;

// PHASE 5b PROOF — is a ONE-PASS var faster than the current TWO-PASS np.var,
// and is it numerically acceptable? Two axes of truth:
//   SPEED: one-pass reads data once (sum+sumsq together); two-pass reads twice.
//   ACCURACY: naive one-pass (E[x²]-E[x]²) catastrophically cancels when the
//             mean ≫ std; two-pass (subtract mean first) and Welford are stable.
// Measured vs np.var (NumSharp two-pass) and NumPy, on benign AND adversarial data.

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

unsafe
{
    foreach (var (mean, lbl) in new[] { (0.0, "benign(mean=0)"), (1e8, "adversarial(mean=1e8)") })
    {
        long n = 10_000_000;
        double* d = (double*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(n * 8));
        var rng = new Random(12345);
        for (long i = 0; i < n; i++) d[i] = mean + (rng.NextDouble() - 0.5); // std ≈ 0.2887 (uniform[-0.5,0.5])
        double trueVar = 1.0 / 12.0; // var of uniform[-0.5,0.5]

        // reference: high-precision two-pass in this process (Kahan-free, but mean-subtracted)
        double refMean = 0; for (long i = 0; i < n; i++) refMean += d[i]; refMean /= n;
        double refSS = 0; for (long i = 0; i < n; i++) { double e = d[i] - refMean; refSS += e * e; }
        double refVar = refSS / n;

        double v1 = Var.OnePassNaive(d, n);
        double v2 = Var.OnePassWelford(d, n);
        var arr = np.zeros(new Shape((int)n), NPTypeCode.Double);
        for (long i = 0; i < n; i++) arr.SetAtIndex(d[i], i); // mirror into NDArray for np.var
        double v3 = Convert.ToDouble(np.var(arr).GetAtIndex(0));

        int it = 8;
        double tNaive   = Bench(() => { var _ = Var.OnePassNaive(d, n); }, it);
        double tWelford = Bench(() => { var _ = Var.OnePassWelford(d, n); }, it);
        double tTwoPass = Bench(() => { using var _ = np.var(arr); }, it);

        Console.WriteLine($"=== {lbl}  (refVar={refVar:G6}) ===");
        Console.WriteLine($"  one-pass naive    val={v1:G6}   relerr={Math.Abs(v1 - refVar) / refVar:G3}   {tNaive:F3} ms");
        Console.WriteLine($"  one-pass Welford  val={v2:G6}   relerr={Math.Abs(v2 - refVar) / refVar:G3}   {tWelford:F3} ms");
        Console.WriteLine($"  np.var two-pass   val={v3:G6}   relerr={Math.Abs(v3 - refVar) / refVar:G3}   {tTwoPass:F3} ms");

        arr.Dispose();
        System.Runtime.InteropServices.NativeMemory.Free(d);
    }
}

static class Var
{
    // Naive one-pass: accumulate sum and sum-of-squares in one SIMD pass.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe double OnePassNaive(double* d, long n)
    {
        var vs = Vector256<double>.Zero; var vss = Vector256<double>.Zero;
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
            for (; i + 4 <= n; i += 4)
            { var v = Vector256.Load(d + i); vs = Vector256.Add(vs, v); vss = Vector256.Add(vss, Vector256.Multiply(v, v)); }
        double s = Vector256.Sum(vs), ss = Vector256.Sum(vss);
        for (; i < n; i++) { s += d[i]; ss += d[i] * d[i]; }
        double m = s / n;
        return ss / n - m * m;
    }

    // Welford one-pass: numerically stable streaming variance (scalar — no SIMD form).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe double OnePassWelford(double* d, long n)
    {
        double mean = 0, m2 = 0; long k = 0;
        for (long i = 0; i < n; i++) { k++; double delta = d[i] - mean; mean += delta / k; m2 += delta * (d[i] - mean); }
        return m2 / n;
    }
}
partial class Program { }
