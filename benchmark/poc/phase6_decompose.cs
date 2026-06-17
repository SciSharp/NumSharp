#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

// Decompose the per-chunk reduce cost: iterator construction vs ForEach drive
// (per-row dispatch). Proves WHY per-chunk regresses on cache-resident numeric.

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
    foreach (var (r, c, lbl) in new[] { (1000, 1000, "1M"), (3162, 3162, "10M") })
    {
        long n = (long)r * c;
        var a = np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7;
        int it = n <= 1_000_000 ? 50 : 20;
        int axis = 0; // C-contig axis0 → SLAB, outer walk = reduced axis (r rows)

        // 1) np.zeros(out) alloc only
        var outShapeDims = new int[] { c };
        double tZeros = Bench(() => { using var o = np.zeros(new Shape(outShapeDims), NPTypeCode.Double); }, it);

        // 2) construct + dispose iterator only (no drive)
        double tCtor = Bench(() =>
        {
            using var o = np.zeros(new Shape(outShapeDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, axis);
        }, it);

        // 3) full: zeros + construct + ForEach drive, counting invocations
        Counter.Calls = 0; Counter.Elems = 0;
        double tFull = Bench(() =>
        {
            using var o = np.zeros(new Shape(outShapeDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, axis);
            iter.ForEach(Counter.Kernel);
        }, it);

        // 3b) drive with an EMPTY kernel (counts only, no memory) → isolates pure
        //     ForEach machinery (iternext delegate + InvokeInner + stride access)
        //     from the compute-kernel delegate's own per-call dispatch.
        double tEmpty = Bench(() =>
        {
            using var o = np.zeros(new Shape(outShapeDims), NPTypeCode.Double);
            using var iter = NpyIterRef.NewReduce(a, o, axis);
            iter.ForEach(Counter.Empty);
        }, it);

        // direct baseline
        double tDirect = Bench(() => { using var rr = np.sum(a, axis: axis); }, it);

        Console.WriteLine($"--- {lbl} double C axis0 ---");
        Console.WriteLine($"  zeros(out)        {tZeros:F4} ms");
        Console.WriteLine($"  +construct iter   {tCtor:F4} ms   (ctor alone ≈ {tCtor - tZeros:F4})");
        Console.WriteLine($"  +ForEach (full)   {tFull:F4} ms   (drive alone ≈ {tFull - tCtor:F4})");
        Console.WriteLine($"  ForEach empty K   {tEmpty:F4} ms   (pure iter machinery; compute-delegate cost ≈ {tFull - tEmpty:F4})");
        Console.WriteLine($"  np.sum (direct)   {tDirect:F4} ms");
        Console.WriteLine($"  ForEach inner calls/run ≈ {Counter.Calls / (long)(it + 3):N0}, total elems/run ≈ {Counter.Elems / (long)(it + 3):N0}");
        Console.WriteLine($"  drive overhead/call ≈ {((tFull - tCtor) * 1e6) / Math.Max(1, Counter.Calls / (long)(it + 3)):F1} ns");
    }
}

static class Counter
{
    public static long Calls;
    public static long Elems;
    public static readonly NpyInnerLoopFunc Kernel;
    public static readonly NpyInnerLoopFunc Empty;
    static unsafe Counter() { Kernel = K; Empty = E; }   // method-group → delegate needs unsafe
    static unsafe void E(void** dataptrs, long* strides, long count, void* aux) { Calls++; Elems += count; }
    static unsafe void K(void** dataptrs, long* strides, long count, void* aux)
    {
        Calls++; Elems += count;
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        if (outS == 0)
        {
            double* o = (double*)outp; double acc = *o;
            double* d = (double*)inp; long i = 0;
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
