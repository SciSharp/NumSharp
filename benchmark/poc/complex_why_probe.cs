#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;

static double Time(Action a, int it)
{
    for (int i = 0; i < 3; i++) a();
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < it; i++) a();
    sw.Stop(); return sw.Elapsed.TotalMilliseconds / it;
}

// Scalar complex slab-fold (what "Monolithic" does today via System.Numerics.Complex +)
static unsafe void ScalarSlab(Complex* p, Complex* o, int rows, int cols)
{ for (int r = 0; r < rows; r++) { Complex* row = p + (long)r * cols; for (int c = 0; c < cols; c++) o[c] += row[c]; } }

// SIMD complex slab-fold: Complex[] == contiguous double pairs, fold 2*cols doubles with AVX.
static unsafe void SimdSlab(Complex* p, Complex* o, int rows, int cols)
{
    double* pd = (double*)p; double* od = (double*)o;
    long len = 2L * cols;
    for (int r = 0; r < rows; r++)
    {
        double* row = pd + (long)r * len;
        long i = 0;
        if (Avx.IsSupported)
            for (; i + 4 <= len; i += 4)
                Avx.Store(od + i, Avx.Add(Avx.LoadVector256(od + i), Avx.LoadVector256(row + i)));
        for (; i < len; i++) od[i] += row[i];
    }
}

static unsafe void Bandwidth()
{
    Console.WriteLine($"AVX={Avx.IsSupported} AVX2={Avx2.IsSupported} AVX512={Avx512F.IsSupported}");
    Console.WriteLine("=== axis=0 gap at 10M: scalar Complex.+ vs SIMD (double-pair) ===");
    foreach (int N in new[] { 1_000_000, 10_000_000 })
    {
        int rows = (int)Math.Sqrt(N), cols = N / rows;
        var a = (np.arange((long)rows * cols).astype(NPTypeCode.Double).reshape(rows, cols).astype(NPTypeCode.Complex)) + new Complex(1, 1);
        int it = 30;
        Complex* p = (Complex*)a.Address + a.Shape.offset;
        var outBuf = new Complex[cols];
        fixed (Complex* o = outBuf)
        {
            Complex* oo = o;
            double scal = Time(() => { for (int c = 0; c < cols; c++) oo[c] = Complex.Zero; ScalarSlab(p, oo, rows, cols); }, it);
            double simd = Time(() => { for (int c = 0; c < cols; c++) oo[c] = Complex.Zero; SimdSlab(p, oo, rows, cols); }, it);
            double gb = (double)N * 16 / 1e9;
            Console.WriteLine($"N={N,-9} ({rows}x{cols})  scalar {scal,7:F3} ms ({gb / (scal / 1000),5:F1} GB/s) | SIMD {simd,7:F3} ms ({gb / (simd / 1000),5:F1} GB/s)");
        }
    }
    Console.WriteLine($"(NumPy axis=0: 1M=0.333ms, 10M=7.588ms = {160.0 / 7.588:F1} GB/s)\n");
}

static unsafe void SmallN()
{
    Console.WriteLine("=== axis=0 gap at 100K: reduction work vs per-op NDArray allocation ===");
    int rows = 316, cols = 316;
    var a = (np.arange((long)rows * cols).astype(NPTypeCode.Double).reshape(rows, cols).astype(NPTypeCode.Complex)) + new Complex(1, 1);
    Complex* p = (Complex*)a.Address + a.Shape.offset;
    var outBuf = new Complex[cols];
    int it = 2000;

    double allocOnly = Time(() => { using var z = np.zeros(new Shape(cols), NPTypeCode.Complex); }, it);
    double computeOnly;
    fixed (Complex* o = outBuf)
    {
        Complex* oo = o;
        computeOnly = Time(() => { for (int c = 0; c < cols; c++) oo[c] = Complex.Zero; SimdSlab(p, oo, rows, cols); }, it);
    }
    double full = Time(() =>
    {
        using var ret = np.zeros(new Shape(cols), NPTypeCode.Complex);
        Complex* o = (Complex*)ret.Address + ret.Shape.offset;
        SimdSlab(p, o, rows, cols);
    }, it);
    Console.WriteLine($"  np.zeros((316,),Complex) alloc only : {allocOnly,7:F4} ms");
    Console.WriteLine($"  pure SIMD reduce (no alloc)        : {computeOnly,7:F4} ms");
    Console.WriteLine($"  full (alloc + SIMD reduce)         : {full,7:F4} ms   <- NumPy = 0.0238 ms");
    Console.WriteLine($"  => allocation = {allocOnly / full * 100:F0}% of the full op; pure compute = {computeOnly / full * 100:F0}%");
}

Bandwidth();
SmallN();
