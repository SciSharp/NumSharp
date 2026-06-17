#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
using System;
using System.Numerics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

// Diagnose what the NpyIter 2-op REDUCE iterator feeds the kernel for axis=0 vs
// axis=1 on a C-contig array: inner-loop count + per-op byte strides + #calls.

int calls = 0; long firstCount = 0, firstInStride = 0, firstOutStride = 0, minCount = 0, maxCount = 0;

Console.WriteLine("=== What the kernel receives (C-contig 3162x3162 complex) ===");
Go(0);
Go(1);

unsafe void Diag(void** p, long* strides, long count, void* aux)
{
    if (calls == 0) { firstCount = count; firstInStride = strides[0]; firstOutStride = strides[1]; minCount = maxCount = count; }
    else { if (count < minCount) minCount = count; if (count > maxCount) maxCount = count; }
    calls++;
    byte* inp = (byte*)p[0]; long inS = strides[0];
    byte* outp = (byte*)p[1]; long outS = strides[1];
    if (outS == 0) { Complex acc = *(Complex*)outp; for (long i = 0; i < count; i++) acc += *(Complex*)(inp + i * inS); *(Complex*)outp = acc; }
    else { for (long i = 0; i < count; i++) { Complex* o = (Complex*)(outp + i * outS); *o += *(Complex*)(inp + i * inS); } }
}

// SIMD kernel: handles both slab (outS!=0) and pinned (outS==0) via double-pair AVX.
unsafe void SimdKernel(void** p, long* strides, long count, void* aux)
{
    double* inp = (double*)p[0]; long inS = strides[0];
    double* outp = (double*)p[1]; long outS = strides[1];
    if (outS == 0) // pinned reduce: 2-lane (re,im) accumulator
    {
        var acc = System.Runtime.Intrinsics.Vector256<double>.Zero;
        long i = 0; long n2 = count * 2; double* d = inp; // inS is contiguous (16) here
        for (; i + 4 <= n2; i += 4) acc = System.Runtime.Intrinsics.X86.Avx.Add(acc, System.Runtime.Intrinsics.X86.Avx.LoadVector256(d + i));
        double* tmp = stackalloc double[4]; System.Runtime.Intrinsics.X86.Avx.Store(tmp, acc);
        double re = tmp[0] + tmp[2], im = tmp[1] + tmp[3];
        for (; i < n2; i += 2) { re += d[i]; im += d[i + 1]; }
        double* o = (double*)p[1]; o[0] += re; o[1] += im;
    }
    else // slab: out[c]+=in[c] over 2*count doubles
    {
        long n2 = count * 2; long i = 0;
        for (; i + 4 <= n2; i += 4) System.Runtime.Intrinsics.X86.Avx.Store(outp + i, System.Runtime.Intrinsics.X86.Avx.Add(System.Runtime.Intrinsics.X86.Avx.LoadVector256(outp + i), System.Runtime.Intrinsics.X86.Avx.LoadVector256(inp + i)));
        for (; i < n2; i++) outp[i] += inp[i];
    }
}

unsafe double TimeForEach(NDArray a, NDArray ret, int axis, NpyInnerLoopFunc k)
{
    for (int w = 0; w < 3; w++) { using var it = Build(a, ret, axis); it.ForEach(k); }
    GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
    var sw = System.Diagnostics.Stopwatch.StartNew();
    for (int t = 0; t < 20; t++) { using var it = Build(a, ret, axis); it.ForEach(k); }
    sw.Stop(); return sw.Elapsed.TotalMilliseconds / 20;
}

static unsafe NpyIterRef Build(NDArray a, NDArray outArr, int axis)
{
    int ndim = a.ndim;
    int[] aAxes = new int[ndim]; int[] outAxes = new int[ndim]; int oc = 0;
    for (int i = 0; i < ndim; i++) { aAxes[i] = i; outAxes[i] = (i == axis) ? -1 : oc++; }
    return NpyIterRef.AdvancedNew(2, new[] { a, outArr },
        NpyIterGlobalFlags.REDUCE_OK | NpyIterGlobalFlags.EXTERNAL_LOOP,
        NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_NO_CASTING,
        new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE },
        null, ndim, new[] { aAxes, outAxes });
}

unsafe void Go(int axis)
{
    int R = 3162, C = 3162;
    var a = (np.arange((long)R * C).astype(NPTypeCode.Double).reshape(R, C).astype(NPTypeCode.Complex)) + new Complex(1, 1);
    var ret = np.zeros(new Shape(new long[] { axis == 0 ? C : R }), NPTypeCode.Complex);
    calls = 0;
    using (var iter = Build(a, ret, axis)) iter.ForEach(Diag);
    long eb = 16;
    Console.WriteLine($"axis={axis}: calls={calls,6}  firstCount={firstCount,6} (min={minCount} max={maxCount})  inStride={firstInStride / eb} elems  outStride={firstOutStride}");
    Console.WriteLine($"          => inner = {(firstInStride == eb ? "CONTIGUOUS input" : firstInStride == 0 ? "broadcast" : $"STRIDED input ({firstInStride / eb} elems = cache-hostile)")}, output {(firstOutStride == 0 ? "PINNED (reduce)" : "slab")}");

    double scalarMs = TimeForEach(a, ret, axis, Diag);
    double simdMs = TimeForEach(a, ret, axis, SimdKernel);
    double npy = axis == 0 ? 7.59 : 9.12;
    Console.WriteLine($"          scalar kernel: {scalarMs,6:F2} ms ({scalarMs / npy:F2}x NumPy) | SIMD kernel: {simdMs,6:F2} ms ({simdMs / npy:F2}x NumPy)  [NumPy {npy}]");
}
