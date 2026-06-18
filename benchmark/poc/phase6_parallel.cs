#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;

// Can NpyIter beat single-threaded Direct/NumPy by PARALLELIZING the reduce
// (the RANGE / PARALLEL_SAFE technique)? Prototype: split the reduce across
// threads, combine partials. double sum, axis0 (SLAB) + axis1 (PINNED), vs
// single-thread per-chunk and np.sum (Direct). 32 cores available.

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.Error.WriteLine($"[opt] core={optCore}  procs={Environment.ProcessorCount}");

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
    foreach (var (r, c, lbl) in new[] { (1000, 1000, "1M"), (3162, 3162, "10M"), (10000, 10000, "100M") })
    {
        long n = (long)r * c;
        var a = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7;
        nint pa = (nint)a.Address;
        int it = n <= 1_000_000 ? 50 : (n <= 10_000_000 ? 20 : 8);

        nint po1 = (nint)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)((long)r * 8));
        using (var d1 = np.sum(a, axis: 1))
        {
            K.ParAxis1(pa, po1, r, c);
            double mx = 0; for (int i = 0; i < r; i++) mx = Math.Max(mx, Math.Abs(((double*)po1)[i] - Convert.ToDouble(d1.GetAtIndex(i))));
            double t1T = Bench(() => K.SeqAxis1(pa, po1, r, c), it);
            double tPT = Bench(() => K.ParAxis1(pa, po1, r, c), it);
            double tDi = Bench(() => { using var x = np.sum(a, axis: 1); }, it);
            Console.WriteLine($"sum axis1 {lbl,4}: 1T {t1T,9:F4}  PAR {tPT,9:F4}  direct {tDi,9:F4}  | par/1T {t1T / tPT:F1}x  par/direct {tDi / tPT:F2}x  maxdiff {mx:G3}");
        }
        System.Runtime.InteropServices.NativeMemory.Free((void*)po1);

        nint po0 = (nint)System.Runtime.InteropServices.NativeMemory.Alloc((nuint)((long)c * 8));
        using (var d0 = np.sum(a, axis: 0))
        {
            K.ParAxis0(pa, po0, r, c);
            double mx = 0; for (int j = 0; j < c; j++) mx = Math.Max(mx, Math.Abs(((double*)po0)[j] - Convert.ToDouble(d0.GetAtIndex(j))));
            double t1T = Bench(() => K.SeqAxis0(pa, po0, r, c), it);
            double tPT = Bench(() => K.ParAxis0(pa, po0, r, c), it);
            double tDi = Bench(() => { using var x = np.sum(a, axis: 0); }, it);
            Console.WriteLine($"sum axis0 {lbl,4}: 1T {t1T,9:F4}  PAR {tPT,9:F4}  direct {tDi,9:F4}  | par/1T {t1T / tPT:F1}x  par/direct {tDi / tPT:F2}x  maxdiff {mx:G3}");
        }
        System.Runtime.InteropServices.NativeMemory.Free((void*)po0);
        a.Dispose();
    }
}

static unsafe class K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double HSum(double* d, long count)
    {
        long i = 0;
        Vector256<double> a0 = Vector256<double>.Zero, a1 = a0, a2 = a0, a3 = a0;
        if (Vector256.IsHardwareAccelerated && count >= 16)
        {
            long lim = count - count % 16;
            for (; i < lim; i += 16)
            { a0 = Vector256.Add(a0, Vector256.Load(d + i)); a1 = Vector256.Add(a1, Vector256.Load(d + i + 4));
              a2 = Vector256.Add(a2, Vector256.Load(d + i + 8)); a3 = Vector256.Add(a3, Vector256.Load(d + i + 12)); }
        }
        double s = Vector256.Sum(Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)));
        for (; i < count; i++) s += d[i];
        return s;
    }
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SlabAdd(double* dst, double* src, long count)
    {
        long i = 0;
        if (Vector256.IsHardwareAccelerated)
            for (; i + 4 <= count; i += 4) Vector256.Store(Vector256.Add(Vector256.Load(dst + i), Vector256.Load(src + i)), dst + i);
        for (; i < count; i++) dst[i] += src[i];
    }

    // axis1 (PINNED): out[i] = sum(row i) — embarrassingly parallel over rows.
    public static void SeqAxis1(nint a, nint o, int r, int c)
    { double* pa = (double*)a, po = (double*)o; for (int i = 0; i < r; i++) po[i] = HSum(pa + (long)i * c, c); }
    public static void ParAxis1(nint a, nint o, int r, int c)
        => Parallel.For(0, r, i => ((double*)o)[i] = HSum((double*)a + (long)i * c, c));

    // axis0 (SLAB): out[j] = sum(col j) — parallel over row-blocks into private partials, then combine.
    public static void SeqAxis0(nint a, nint o, int r, int c)
    { double* pa = (double*)a, po = (double*)o; for (int j = 0; j < c; j++) po[j] = 0; for (int i = 0; i < r; i++) SlabAdd(po, pa + (long)i * c, c); }
    public static void ParAxis0(nint a, nint o, int r, int c)
    {
        int nT = Math.Min(Environment.ProcessorCount, Math.Max(1, r / 64));
        var partials = new double[nT][];
        Parallel.For(0, nT, t =>
        {
            int r0 = (int)((long)t * r / nT), r1 = (int)((long)(t + 1) * r / nT);
            var buf = new double[c];
            fixed (double* pb = buf) { double* pa = (double*)a; for (int i = r0; i < r1; i++) SlabAdd(pb, pa + (long)i * c, c); }
            partials[t] = buf;
        });
        double* po = (double*)o;
        for (int j = 0; j < c; j++) po[j] = 0;
        for (int t = 0; t < nT; t++) { var pt = partials[t]; for (int j = 0; j < c; j++) po[j] += pt[j]; }
    }
}
partial class Program { }
