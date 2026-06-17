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
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

// =============================================================================
// PHASE 6 PROOF — can a per-chunk (NpyIter-driven) SIMD reduction kernel match
// the Direct whole-array SIMD path (np.sum) on the numeric dtypes?
//
// This hand-writes the Vector256 per-chunk sum kernel that Phase 6 would
// IL-emit, drives it through NpyIterRef.NewReduce + ForEach (the migration
// target), and times it head-to-head vs np.sum (Direct path) and NumPy.
// If per-chunk ≈ Direct, the migration is zero-regression → Phase 6 is valid.
// =============================================================================

bool optCore = !typeof(np).Assembly.GetCustomAttributes(typeof(System.Diagnostics.DebuggableAttribute), false)
    .Cast<System.Diagnostics.DebuggableAttribute>().Any(a => a.IsJITOptimizerDisabled);
Console.Error.WriteLine($"[opt] core={optCore}  (MUST be true)");

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

// Per-chunk reduce: build the 2-op REDUCE iterator, seed 0, drive the kernel.
static unsafe NDArray PerChunkSum(NDArray a, int axis, NPTypeCode tc, NpyInnerLoopFunc kernel)
{
    var dims = a.shape;
    var outDims = new System.Collections.Generic.List<int>();
    for (int i = 0; i < a.ndim; i++) if (i != axis) outDims.Add((int)dims[i]);
    var outp = np.zeros(new Shape(outDims.ToArray()), tc);   // sum identity = 0
    using var iter = NpyIterRef.NewReduce(a, outp, axis);
    iter.ForEach(kernel);
    return outp;
}

static bool Same(NDArray x, NDArray y, out string why)
{
    why = "";
    if (x.size != y.size) { why = $"size {x.size}!={y.size}"; return false; }
    for (long i = 0; i < x.size; i++)
    {
        double a = Convert.ToDouble(x.GetAtIndex(i)), b = Convert.ToDouble(y.GetAtIndex(i));
        // 5e-3 rel: per-chunk 4-accumulator order vs Direct differ at float32 precision;
        // loose enough for fp reordering, tight enough to catch an indexing/garbage bug.
        if (Math.Abs(a - b) > 5e-3 * (1 + Math.Abs(b)) + 1e-6) { why = $"[{i}] {a}!={b}"; return false; }
    }
    return true;
}

var sizes = new (int r, int c, string lbl)[] { (1000,1000,"1M"), (3162,3162,"10M") };
int corrFails = 0;

unsafe   // method-group → NpyInnerLoopFunc (void** params) conversions need unsafe context
{
foreach (var (kindTc, kindName, kernel) in new (NPTypeCode, string, NpyInnerLoopFunc)[]
{
    (NPTypeCode.Double, "double", K.DoubleSum),
    (NPTypeCode.Single, "float",  K.FloatSum),
    (NPTypeCode.Int64,  "int64",  K.LongSum),
})
{
    foreach (var (r, c, lbl) in sizes)
    {
        long n = (long)r * c;
        NDArray aC = (np.arange(n).astype(NPTypeCode.Double).reshape(r, c) * 0.0009 + 0.7).astype(kindTc);
        NDArray aT = aC.T;
        int it = n <= 1_000_000 ? 40 : 15;

        foreach (var (tag, a) in new[] { ("C", aC), ("T", aT) })
            for (int ax = 0; ax < 2; ax++)
            {
                // correctness: per-chunk vs np.sum (Direct)
                int axc = ax; NDArray av = a;
                using (var pc = PerChunkSum(av, axc, kindTc == NPTypeCode.Single ? NPTypeCode.Single : kindTc, kernel))
                using (var dir = np.sum(av, axis: axc))
                    if (!Same(pc, dir, out string why)) { Console.WriteLine($"CORR FAIL {kindName}/{tag} ax{ax} {lbl}: {why}"); corrFails++; }

                double tPC  = Bench(() => { using var rr = PerChunkSum(av, axc, kindTc == NPTypeCode.Single ? NPTypeCode.Single : kindTc, kernel); }, it);
                double tDir = Bench(() => { using var rr = np.sum(av, axis: axc); }, it);
                Console.WriteLine($"{kindName}|{tag}|axis{ax}|{lbl}|perchunk|{tPC:F4}");
                Console.WriteLine($"{kindName}|{tag}|axis{ax}|{lbl}|direct|{tDir:F4}");
            }
    }
}
}
Console.Error.WriteLine(corrFails == 0 ? "[correctness] ALL PASS" : $"[correctness] {corrFails} FAIL");

// ============================ per-chunk kernels ============================
static class K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void DoubleSum(void** dataptrs, long* strides, long count, void* aux)
    {
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        if (outS == 0) // PINNED: horizontal reduce contiguous stripe → one slot
        {
            double* o = (double*)outp; double acc = *o;
            if (inS == 8)
            {
                double* d = (double*)inp; long i = 0;
                if (Vector256.IsHardwareAccelerated && count >= 16)
                {
                    var a0 = Vector256<double>.Zero; var a1 = Vector256<double>.Zero;
                    var a2 = Vector256<double>.Zero; var a3 = Vector256<double>.Zero;
                    for (; i + 16 <= count; i += 16)
                    {
                        a0 = Vector256.Add(a0, Vector256.Load(d + i));
                        a1 = Vector256.Add(a1, Vector256.Load(d + i + 4));
                        a2 = Vector256.Add(a2, Vector256.Load(d + i + 8));
                        a3 = Vector256.Add(a3, Vector256.Load(d + i + 12));
                    }
                    a0 = Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3));
                    acc += Vector256.Sum(a0);
                }
                for (; i < count; i++) acc += d[i];
            }
            else for (long k = 0; k < count; k++) acc += *(double*)(inp + k * inS);
            *o = acc;
        }
        else // SLAB: out[c] += in[c]
        {
            if (inS == 8 && outS == 8)
            {
                double* id = (double*)inp; double* od = (double*)outp; long i = 0;
                if (Vector256.IsHardwareAccelerated)
                {
                    for (; i + 16 <= count; i += 16)
                    {
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i),      Vector256.Load(id + i)),      od + i);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + 4),  Vector256.Load(id + i + 4)),  od + i + 4);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + 8),  Vector256.Load(id + i + 8)),  od + i + 8);
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i + 12), Vector256.Load(id + i + 12)), od + i + 12);
                    }
                    for (; i + 4 <= count; i += 4)
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                }
                for (; i < count; i++) od[i] += id[i];
            }
            else for (long k = 0; k < count; k++) *(double*)(outp + k * outS) += *(double*)(inp + k * inS);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void FloatSum(void** dataptrs, long* strides, long count, void* aux)
    {
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        if (outS == 0)
        {
            float* o = (float*)outp; float acc = *o;
            if (inS == 4)
            {
                float* d = (float*)inp; long i = 0;
                if (Vector256.IsHardwareAccelerated && count >= 32)
                {
                    var a0 = Vector256<float>.Zero; var a1 = Vector256<float>.Zero;
                    var a2 = Vector256<float>.Zero; var a3 = Vector256<float>.Zero;
                    for (; i + 32 <= count; i += 32)
                    {
                        a0 = Vector256.Add(a0, Vector256.Load(d + i));
                        a1 = Vector256.Add(a1, Vector256.Load(d + i + 8));
                        a2 = Vector256.Add(a2, Vector256.Load(d + i + 16));
                        a3 = Vector256.Add(a3, Vector256.Load(d + i + 24));
                    }
                    a0 = Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3));
                    acc += Vector256.Sum(a0);
                }
                for (; i < count; i++) acc += d[i];
            }
            else for (long k = 0; k < count; k++) acc += *(float*)(inp + k * inS);
            *o = acc;
        }
        else
        {
            if (inS == 4 && outS == 4)
            {
                float* id = (float*)inp; float* od = (float*)outp; long i = 0;
                if (Vector256.IsHardwareAccelerated)
                    for (; i + 8 <= count; i += 8)
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                for (; i < count; i++) od[i] += id[i];
            }
            else for (long k = 0; k < count; k++) *(float*)(outp + k * outS) += *(float*)(inp + k * inS);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void LongSum(void** dataptrs, long* strides, long count, void* aux)
    {
        byte* inp = (byte*)dataptrs[0]; long inS = strides[0];
        byte* outp = (byte*)dataptrs[1]; long outS = strides[1];
        if (outS == 0)
        {
            long* o = (long*)outp; long acc = *o;
            if (inS == 8)
            {
                long* d = (long*)inp; long i = 0;
                if (Vector256.IsHardwareAccelerated && count >= 16)
                {
                    var a0 = Vector256<long>.Zero; var a1 = Vector256<long>.Zero;
                    var a2 = Vector256<long>.Zero; var a3 = Vector256<long>.Zero;
                    for (; i + 16 <= count; i += 16)
                    {
                        a0 = Vector256.Add(a0, Vector256.Load(d + i));
                        a1 = Vector256.Add(a1, Vector256.Load(d + i + 4));
                        a2 = Vector256.Add(a2, Vector256.Load(d + i + 8));
                        a3 = Vector256.Add(a3, Vector256.Load(d + i + 12));
                    }
                    a0 = Vector256.Add(Vector256.Add(a0, a1), Vector256.Add(a2, a3));
                    acc += Vector256.Sum(a0);
                }
                for (; i < count; i++) acc += d[i];
            }
            else for (long k = 0; k < count; k++) acc += *(long*)(inp + k * inS);
            *o = acc;
        }
        else
        {
            if (inS == 8 && outS == 8)
            {
                long* id = (long*)inp; long* od = (long*)outp; long i = 0;
                if (Vector256.IsHardwareAccelerated)
                    for (; i + 4 <= count; i += 4)
                        Vector256.Store(Vector256.Add(Vector256.Load(od + i), Vector256.Load(id + i)), od + i);
                for (; i < count; i++) od[i] += id[i];
            }
            else for (long k = 0; k < count; k++) *(long*)(outp + k * outS) += *(long*)(inp + k * inS);
        }
    }
}
partial class Program { }
