#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Iteration;

// Micro: is the per-chunk SLAB body slow because it's invoked PER-CALL (delegate
// boundary blocks cross-row pipelining / register retention) or is the algorithm
// itself slow? Time 1000 rows three ways on identical memory:
//   A) one delegate call per row (the per-chunk model)
//   B) one INLINED call per row (same body, no delegate, no cross-row state)
//   C) monolithic: keep the whole slab loop in one function (what Direct does)

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
    foreach (var (rows, cols, lbl) in new[] { (1000, 1000, "1M"), (3162, 3162, "10M") })
    {
        long n = (long)rows * cols;
        double* inp = (double*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(n * 8));
        double* outp = (double*)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)(cols * 8));
        for (long i = 0; i < n; i++) inp[i] = 0.0009 * i + 0.7;
        int it = n <= 1_000_000 ? 50 : 20;

        NpyInnerLoopFunc kd = Slab.Kernel;
        long inS = 8, outS = 8;
        void** dp = stackalloc void*[2];
        long* st = stackalloc long[2]; st[0] = inS; st[1] = outS;

        // A) per-row delegate call
        double tA = Bench(() =>
        {
            for (long c = 0; c < cols; c++) outp[c] = 0;
            for (long r = 0; r < rows; r++)
            {
                dp[0] = inp + r * cols; dp[1] = outp;
                kd(dp, st, cols, null);
            }
        }, it);

        // B) per-row INLINED call (same body, no delegate)
        double tB = Bench(() =>
        {
            for (long c = 0; c < cols; c++) outp[c] = 0;
            for (long r = 0; r < rows; r++)
                Slab.Inline(inp + r * cols, outp, cols);
        }, it);

        // C) monolithic loop (Direct-style: rows loop INSIDE one function)
        double tC = Bench(() =>
        {
            for (long c = 0; c < cols; c++) outp[c] = 0;
            Slab.Monolithic(inp, outp, rows, cols);
        }, it);

        Console.WriteLine($"--- {lbl} SLAB add ({rows}x{cols}) ---");
        Console.WriteLine($"  A) per-row delegate   {tA:F4} ms");
        Console.WriteLine($"  B) per-row inlined    {tB:F4} ms");
        Console.WriteLine($"  C) monolithic         {tC:F4} ms");

        System.Runtime.InteropServices.NativeMemory.Free(inp);
        System.Runtime.InteropServices.NativeMemory.Free(outp);
    }
}

static class Slab
{
    public static readonly NpyInnerLoopFunc Kernel;
    static unsafe Slab() { Kernel = K; }
    static unsafe void K(void** dataptrs, long* strides, long count, void* aux)
        => Inline((double*)dataptrs[0], (double*)dataptrs[1], count);

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Inline(double* id, double* od, long count)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
            for (; i + 4 <= count; i += 4)
                Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
        for (; i < count; i++) od[i] += id[i];
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Monolithic(double* inp, double* od, long rows, long cols)
    {
        for (long r = 0; r < rows; r++)
        {
            double* id = inp + r * cols;
            long i = 0;
            if (Vector256.IsHardwareAccelerated)
                for (; i + 4 <= cols; i += 4)
                    Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
            for (; i < cols; i++) od[i] += id[i];
        }
    }
}
partial class Program { }
