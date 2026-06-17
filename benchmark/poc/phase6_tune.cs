#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends.Iteration;

// Tune the PRODUCTION 8-way generic SIMD sum kernel to Direct parity on BOTH
// pinned (axis1) and slab (axis0) before porting to core. Generic over T so the
// same body covers double+float (and later int/long). Must be ≤ ~1.05× Direct.

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

static unsafe NDArray PerChunk(NDArray a, int axis, NPTypeCode tc, NpyInnerLoopFunc k)
{
    var od = new System.Collections.Generic.List<int>();
    for (int i = 0; i < a.ndim; i++) if (i != axis) od.Add((int)a.shape[i]);
    var o = np.zeros(new Shape(od.ToArray()), tc);
    using var iter = NpyIterRef.NewReduce(a, o, axis);
    iter.ForEach(k);
    return o;
}

unsafe
{
    var kerns = new (NPTypeCode tc, string name, NpyInnerLoopFunc k)[]
    {
        (NPTypeCode.Double, "double", Sum.D),
        (NPTypeCode.Single, "float",  Sum.F),
    };
    foreach (var (tc, name, k) in kerns)
        foreach (var (r, c, lbl) in new[] { (1000, 1000, "1M"), (3162, 3162, "10M") })
        {
            long n = (long)r * c;
            var aC = (np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7).astype(tc);
            int it = n <= 1_000_000 ? 40 : 15;
            for (int ax = 0; ax < 2; ax++)
            {
                int axc = ax;
                using (var pc = PerChunk(aC, axc, tc, k))
                using (var di = np.sum(aC, axis: axc))
                {
                    double mx = 0; for (long i = 0; i < pc.size; i++) mx = Math.Max(mx, Math.Abs(Convert.ToDouble(pc.GetAtIndex(i)) - Convert.ToDouble(di.GetAtIndex(i))));
                    double tPC = Bench(() => { using var rr = PerChunk(aC, axc, tc, k); }, it);
                    double tDi = Bench(() => { using var rr = np.sum(aC, axis: axc); }, it);
                    Console.WriteLine($"{name} {lbl} axis{ax}: perchunk {tPC:F4}  direct {tDi:F4}  ratio {tPC/tDi:F2}x  maxdiff {mx:G3}");
                }
            }
        }
}

static class Sum
{
    public static readonly NpyInnerLoopFunc D, F;
    static unsafe Sum() { D = K<double>; F = K<float>; }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    static unsafe void K<T>(void** dataptrs, long* strides, long count, void* aux) where T : unmanaged, INumber<T>
    {
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        int sz = sizeof(T);
        if (outS == 0)
        {
            T acc = *(T*)outp;
            if (inS == sz)
            {
                T* d = (T*)inp; long i = 0; int W = Vector256<T>.Count;
                if (Vector256.IsHardwareAccelerated && count >= W * 8)
                {
                    Vector256<T> a0 = Vector256<T>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
                    long lim = count - count % (W * 8);
                    for (; i < lim; i += W * 8)
                    {
                        a0 = Vector256.Add(a0, Vector256.Load(d + i));
                        a1 = Vector256.Add(a1, Vector256.Load(d + i + W));
                        a2 = Vector256.Add(a2, Vector256.Load(d + i + W * 2));
                        a3 = Vector256.Add(a3, Vector256.Load(d + i + W * 3));
                        a4 = Vector256.Add(a4, Vector256.Load(d + i + W * 4));
                        a5 = Vector256.Add(a5, Vector256.Load(d + i + W * 5));
                        a6 = Vector256.Add(a6, Vector256.Load(d + i + W * 6));
                        a7 = Vector256.Add(a7, Vector256.Load(d + i + W * 7));
                    }
                    a0 = Vector256.Add(Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3)),
                                       Vector256.Add(Vector256.Add(a4, a5), Vector256.Add(a6, a7)));
                    acc += Vector256.Sum(a0);
                }
                for (; i < count; i++) acc += d[i];
            }
            else for (long kk = 0; kk < count; kk++) acc += *(T*)(inp + kk * inS);
            *(T*)outp = acc;
        }
        else
        {
            if (inS == sz && outS == sz)
            {
                T* id = (T*)inp; T* od = (T*)outp; long i = 0; int W = Vector256<T>.Count;
                if (Vector256.IsHardwareAccelerated)
                {
                    long lim = count - count % (W * 4);
                    for (; i < lim; i += W * 4)
                    {
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i),         Vector256.Load(id + i)),         od + i);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + W),     Vector256.Load(id + i + W)),     od + i + W);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + W * 2), Vector256.Load(id + i + W * 2)), od + i + W * 2);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + W * 3), Vector256.Load(id + i + W * 3)), od + i + W * 3);
                    }
                    for (; i + W <= count; i += W)
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                }
                for (; i < count; i++) od[i] += id[i];
            }
            else for (long kk = 0; kk < count; kk++) *(T*)(outp + kk * outS) += *(T*)(inp + kk * inS);
        }
    }
}
partial class Program { }
