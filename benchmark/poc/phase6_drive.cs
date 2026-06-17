#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Iteration;

// Bisect the 10× ForEach-reduce regression. Same NpyIterRef.NewReduce iterator,
// driven 3 ways on the real kernel:
//   1) ForEach            (delegate + InvokeInner mask wrapper)
//   2) manual + delegate  (Iternext loop, kernel via delegate, no wrapper)
//   3) manual + direct     (Iternext loop, kernel as a DIRECT static call — inlinable)
// vs np.sum (Direct). Pinpoints whether the cost is ForEach's wrapper, the
// delegate invoke, or the Iternext/kernel interaction itself.

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
    NpyInnerLoopFunc kd = Slab.Kernel;
    foreach (var (r, c, lbl) in new[] { (1000, 1000, "1M"), (3162, 3162, "10M") })
    {
        long n = (long)r * c;
        var a = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7;
        int it = n <= 1_000_000 ? 50 : 20;
        var outDims = new int[] { c };

        double tForEach = Bench(() =>
        {
            using var o = np.zeros(new Shape(outDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, 0);
            iter.ForEach(kd);
        }, it);

        double tManualDel = Bench(() =>
        {
            using var o = np.zeros(new Shape(outDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, 0);
            void** dp = iter.GetDataPtrArray();
            long* bs = iter.GetInnerStrideArray();
            long inner = *iter.GetInnerLoopSizePtr();
            do { kd(dp, bs, inner, null); } while (iter.Iternext());
        }, it);

        double tManualDir = Bench(() =>
        {
            using var o = np.zeros(new Shape(outDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, 0);
            void** dp = iter.GetDataPtrArray();
            long* bs = iter.GetInnerStrideArray();
            long inner = *iter.GetInnerLoopSizePtr();
            do { Slab.DirectCall(dp, bs, inner, null); } while (iter.Iternext());
        }, it);

        double tDirect = Bench(() => { using var rr = np.sum(a, axis: 0); }, it);

        Console.WriteLine($"--- {lbl} double C axis0 ---");
        Console.WriteLine($"  1) ForEach           {tForEach:F4} ms");
        Console.WriteLine($"  2) manual+delegate   {tManualDel:F4} ms");
        Console.WriteLine($"  3) manual+direct     {tManualDir:F4} ms");
        Console.WriteLine($"  4) np.sum (Direct)   {tDirect:F4} ms");
    }
}

static class Slab
{
    public static readonly NpyInnerLoopFunc Kernel;
    static unsafe Slab() { Kernel = DirectCall; }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DirectCall(void** dataptrs, long* strides, long count, void* aux)
    {
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        if (outS == 0)
        {
            double* o = (double*)outp; double acc = *o; double* d = (double*)inp; long i = 0;
            if (inS == 8 && Vector256.IsHardwareAccelerated && count >= 4)
            {
                var v = Vector256<double>.Zero;
                for (; i + 4 <= count; i += 4) v = Vector256.Add(v, Vector256.Load(d + i));
                acc += Vector256.Sum(v);
            }
            for (; i < count; i++) acc += d[i];
            *o = acc;
        }
        else
        {
            double* id = (double*)inp; double* od = (double*)outp; long i = 0;
            if (inS == 8 && outS == 8 && Vector256.IsHardwareAccelerated)
                for (; i + 4 <= count; i += 4)
                    Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
            for (; i < count; i++) od[i] += id[i];
        }
    }
}
partial class Program { }
