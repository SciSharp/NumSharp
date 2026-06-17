#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Diagnostics;
using System.Numerics;
using NumSharp;

static double Time(Action a, int iters, int warmup = 3)
{
    for (int i = 0; i < warmup; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) a();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / iters;
}

unsafe void Run(int N)
{
    int rows = (int)Math.Sqrt(N);
    int cols = N / rows;
    long total = (long)rows * cols;

    // Build a C-contiguous complex array identical in layout to the benchmark.
    var a = np.arange(total).astype(NPTypeCode.Double).reshape(rows, cols).astype(NPTypeCode.Complex);
    // give it nonzero imaginary content
    var b = (a + new Complex(0, 1));
    a = b;

    int it = N >= 1_000_000 ? 20 : 200;

    // --- Live NumSharp paths ---
    double tSum0 = Time(() => { using var _ = np.sum(a, axis: 0); }, it);
    double tMean0 = Time(() => { using var _ = np.mean(a, axis: 0); }, it);

    // --- Clean hand-written pointer reduction (no boxing, no slices, no virtual calls) ---
    var outBuf = new Complex[cols];
    Complex* basePtr = (Complex*)a.Address + a.Shape.offset;
    double tClean0 = Time(() =>
    {
        for (int c = 0; c < cols; c++) outBuf[c] = Complex.Zero;
        Complex* p = basePtr;
        for (int r = 0; r < rows; r++)
        {
            Complex* row = p + (long)r * cols;
            for (int c = 0; c < cols; c++)
                outBuf[c] += row[c];
        }
    }, it);

    double tCleanMean0 = Time(() =>
    {
        for (int c = 0; c < cols; c++) outBuf[c] = Complex.Zero;
        Complex* p = basePtr;
        for (int r = 0; r < rows; r++)
        {
            Complex* row = p + (long)r * cols;
            for (int c = 0; c < cols; c++)
                outBuf[c] += row[c];
        }
        for (int c = 0; c < cols; c++) outBuf[c] /= rows;
    }, it);

    Console.WriteLine($"N={N,-10} shape=({rows},{cols})");
    Console.WriteLine($"  sum  axis=0 : np.sum  = {tSum0,8:F4} ms | clean ptr loop = {tClean0,8:F4} ms | overhead = {tSum0/tClean0,6:F1}x");
    Console.WriteLine($"  mean axis=0 : np.mean = {tMean0,8:F4} ms | clean ptr loop = {tCleanMean0,8:F4} ms | overhead = {tMean0/tCleanMean0,6:F1}x");

    // correctness sanity
    using var nsSum = np.sum(a, axis: 0);
    Console.WriteLine($"  parity[0]: np.sum={nsSum.GetComplex(0)}  clean={outBuf[0]*rows + Complex.Zero} (clean shown post-mean-div, ignore)");
}

Console.WriteLine("=== Complex128 axis=0 reduction: live path vs clean pointer loop ===");
Run(100_000);
Run(10_000_000);
