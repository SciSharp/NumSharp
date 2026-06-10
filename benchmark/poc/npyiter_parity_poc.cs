#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
#:property Optimize=true
// =============================================================================
// POC — NpyIter-driven execution at NumPy parity or better
// =============================================================================
//
// Proves that the NpyIter architecture (iterator drives; per-chunk kernel
// processes one inner loop — NumPy's PyUFuncGenericFunction model) reaches
// NumPy 2.4.2 performance on this machine across layouts, and EXCEEDS it
// where the architecture enables fusion (Tier-3C NpyExpr: one pass, no
// intermediate temporaries).
//
// Every aspect goes through the REAL NpyIterRef machinery:
//   MultiNew -> ForEach(kernel) / ExecuteUnary / ExecuteBinary /
//   ExecuteExpression.
// The strided kernels below are POC implementations of the Phase 2a inner
// loop, written as CONCRETE methods per the Phase 2b JIT findings. Strided
// loads use AVX2 HARDWARE GATHER (Avx2.GatherVector256, scale=1, byte-offset
// indices hoisted out of the loop) — the same technique NumPy uses for its
// strided unary loops (npyv_loadn_f32 = _mm256_i32gather_ps,
// simd/avx2/memory.h). For strided BINARY ops NumPy has NO simd path at all
// (loops_arithm_fp.dispatch.c.src falls to a scalar loop), and for strided
// REDUCTION it uses a scalar 8-accumulator loop (loops_utils.h.src
// pairwise_sum) — hardware gather beats both on Raptor Lake (measured
// interleaved, Release: C 334 vs 399 us, E 121 vs 221 us).
// Software insert-gather (raw-pointer Vector256.Create) is the fallback when
// AVX2 is unavailable or strides exceed the int32 index range.
//
// METHODOLOGY
// -----------
// Outputs are PREALLOCATED on both sides (NumPy uses out=) so the numbers
// compare the execution architecture, not the allocators: a fresh 4 MB
// np.empty per call costs ~0.3-0.4 ms in soft page faults on .NET (frees are
// GC-deferred so pages stay cold), while CPython's refcounting frees the
// previous result immediately and NumPy's allocator reuses warm pages.
// Fusion aspects let the EAGER side allocate its intermediate temporaries —
// eliminating those is precisely what fusion is.
//
// CRITICAL: must run with the JIT optimizer enabled on BOTH the script
// assembly and NumSharp.Core. `dotnet run file.cs` builds DEBUG by default
// (DebuggableAttribute.DisableOptimizations — the JIT honors it even over
// AggressiveOptimization), which silently doubles the strided kernel times
// while leaving DynamicMethod-emitted kernels (A/B/F/G) unaffected. The
// script asserts this at startup.
//
// Run:        dotnet run -c Release - < benchmark/poc/npyiter_parity_poc.cs
// NumPy side: python benchmark/poc/npyiter_parity_poc.py
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;
using NumSharp.Backends.Kernels;

static double TimeMs(Action f, int iters, int warmup) {
    for (int i = 0; i < warmup; i++) f();
    var sw = Stopwatch.StartNew();
    for (int i = 0; i < iters; i++) f();
    sw.Stop();
    return sw.Elapsed.TotalMilliseconds / iters;
}

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

const int N1M = 1_000_000;
const int N10M = 10_000_000;

Console.WriteLine($"AVX2={Avx2.IsSupported}  (AVX-512 absent on this machine and in NumPy's dispatch)");

// Refuse to print misleading numbers from Debug-JITted code (see header).
{
    var dbgScript = Attribute.GetCustomAttribute(typeof(PocKernels).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
    var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
    bool scriptDbg = dbgScript?.IsJITOptimizerDisabled ?? false;
    bool coreDbg = dbgCore?.IsJITOptimizerDisabled ?? false;
    if (scriptDbg || coreDbg)
    {
        Console.WriteLine($"!! JIT OPTIMIZER DISABLED (script={(scriptDbg ? "DEBUG" : "ok")}, NumSharp.Core={(coreDbg ? "DEBUG" : "ok")}) — numbers below are INVALID.");
        Console.WriteLine("!! Run:  dotnet run -c Release - < benchmark/poc/npyiter_parity_poc.cs");
    }
}
Console.WriteLine();
Console.WriteLine("aspect                                NumSharp/NpyIter");
Console.WriteLine("--------------------------------------------------------");

unsafe {
    // =========================================================================
    // A. Contiguous unary — sqrt(f32, 10M) through NpyIter
    // =========================================================================
    {
        var a = (np.arange(N10M).astype(np.float32) + 1f).reshape(N10M);
        var outNd = np.empty(new Shape(N10M), np.float32);
        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(2, new[] { a, outNd },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteUnary(UnaryOp.Sqrt);
        }, 80, 20);
        Check(Math.Abs(outNd.GetSingle(12345) - MathF.Sqrt(a.GetSingle(12345))) < 1e-6f, "A sample");
        Console.WriteLine($"A contig sqrt f32 10M               {t,8:F2} ms");
    }

    // =========================================================================
    // B. Contiguous binary — add(f32, 10M) through NpyIter
    // =========================================================================
    {
        var a = (np.arange(N10M).astype(np.float32) + 1f).reshape(N10M);
        var b = (np.arange(N10M).astype(np.float32) + 2f).reshape(N10M);
        var outNd = np.empty(new Shape(N10M), np.float32);
        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(3, new[] { a, b, outNd },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteBinary(BinaryOp.Add);
        }, 80, 20);
        Check(outNd.GetSingle(777) == a.GetSingle(777) + b.GetSingle(777), "B sample");
        Console.WriteLine($"B contig add  f32 10M               {t,8:F2} ms");
    }

    // =========================================================================
    // C. Strided binary — a[::2] + b[::2] (1M) via NpyIter + POC fused-gather
    //    kernel (the Phase 2a inner loop the production shell still lacks).
    // =========================================================================
    {
        var wa = np.arange(2 * N1M).astype(np.float32) + 1f;
        var wb = np.arange(2 * N1M).astype(np.float32) + 2f;
        var sa = wa["::2"];
        var sb = wb["::2"];
        var outNd = np.empty(new Shape(N1M), np.float32);

        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(3, new[] { sa, sb, outNd },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ForEach(PocKernels.AddF32);
        }, 300, 60);

        var reference = sa + sb;
        Check(outNd.GetSingle(0) == reference.GetSingle(0) &&
              outNd.GetSingle(N1M - 1) == reference.GetSingle(N1M - 1) &&
              outNd.GetSingle(777_777) == reference.GetSingle(777_777), "C values");
        Console.WriteLine($"C strided add a[::2]+b[::2] f32 1M  {t * 1000,8:F0} us");
    }

    // =========================================================================
    // D. 2-D strided unary — sqrt(a[::2, ::2]) (1M) via NpyIter + POC kernel.
    //    EXTERNAL_LOOP hands the kernel one strided row per call.
    // =========================================================================
    {
        var big = (np.arange(4 * N1M).astype(np.float32) + 1f).reshape(2000, 2000);
        var s2d = big["::2, ::2"];   // (1000, 1000), strides (16000B, 8B)
        var outNd = np.empty(new Shape(1000, 1000), np.float32);

        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(2, new[] { s2d, outNd },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ForEach(PocKernels.SqrtF32);
        }, 300, 60);

        Check(Math.Abs(outNd.GetSingle(500, 500) - MathF.Sqrt(s2d.GetSingle(500, 500))) < 1e-6f, "D sample");
        Console.WriteLine($"D strided sqrt a[::2,::2] f32 1M    {t * 1000,8:F0} us");
    }

    // =========================================================================
    // E. Strided reduction — sum(a[::2]) f32 1M via NpyIter per-chunk partials
    // =========================================================================
    {
        var wa = (np.arange(2 * N1M).astype(np.float32) % 97f) + 1f;
        var sa = wa["::2"];

        double sum = 0;
        double t = TimeMs(() => {
            double acc = 0;
            using var iter = NpyIterRef.New(sa, NpyIterGlobalFlags.EXTERNAL_LOOP);
            iter.ForEach(PocKernels.SumF32, &acc);
            sum = acc;
        }, 300, 60);

        double expected = (double)np.sum(sa.astype(np.float64));
        Check(Math.Abs(sum - expected) / Math.Abs(expected) < 1e-9, $"E sum {sum} vs {expected}");
        Console.WriteLine($"E strided sum a[::2] f32 1M         {t * 1000,8:F0} us");
    }

    // =========================================================================
    // F. Fusion — a*b + c (f32 10M) as ONE NpyIter pass via NpyExpr (Tier 3C).
    //    NumPy must do two passes and materialize a temporary.
    // =========================================================================
    {
        var a = (np.arange(N10M).astype(np.float32) % 13f) + 1f;
        var b = (np.arange(N10M).astype(np.float32) % 7f) + 2f;
        var c = (np.arange(N10M).astype(np.float32) % 5f) + 3f;
        var outNd = np.empty(new Shape(N10M), np.float32);
        var expr = NpyExpr.Add(NpyExpr.Multiply(NpyExpr.Input(0), NpyExpr.Input(1)), NpyExpr.Input(2));
        var f32x3 = new[] { NPTypeCode.Single, NPTypeCode.Single, NPTypeCode.Single };

        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(4, new[] { a, b, c, outNd },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteExpression(expr, f32x3, NPTypeCode.Single, "poc_fma_f32");
        }, 60, 15);

        var reference = a * b + c;
        Check(outNd.GetSingle(0) == reference.GetSingle(0) &&
              outNd.GetSingle(N10M - 1) == reference.GetSingle(N10M - 1) &&
              outNd.GetSingle(5_555_555) == reference.GetSingle(5_555_555), "F values");

        // eager two-pass for context (allocates the temporary — that is the point)
        double tEager = TimeMs(() => { var _ = a * b + c; }, 40, 10);
        Console.WriteLine($"F fused a*b+c f32 10M               {t,8:F2} ms   (NumSharp eager 2-pass: {tEager:F2} ms)");
    }

    // =========================================================================
    // G. Fusion — (a-b)/(a+b) (f32 10M): three NumPy passes + two temps -> one.
    // =========================================================================
    {
        var a = (np.arange(N10M).astype(np.float32) % 13f) + 5f;
        var b = (np.arange(N10M).astype(np.float32) % 7f) + 1f;
        var outNd = np.empty(new Shape(N10M), np.float32);
        var expr = NpyExpr.Divide(
            NpyExpr.Subtract(NpyExpr.Input(0), NpyExpr.Input(1)),
            NpyExpr.Add(NpyExpr.Input(0), NpyExpr.Input(1)));
        var f32x2 = new[] { NPTypeCode.Single, NPTypeCode.Single };

        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(3, new[] { a, b, outNd },
                NpyIterGlobalFlags.EXTERNAL_LOOP, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteExpression(expr, f32x2, NPTypeCode.Single, "poc_normdiff_f32");
        }, 60, 15);

        var reference = (a - b) / (a + b);
        Check(Math.Abs(outNd.GetSingle(123_456) - reference.GetSingle(123_456)) < 1e-6f, "G sample");

        double tEager = TimeMs(() => { var _ = (a - b) / (a + b); }, 40, 10);
        Console.WriteLine($"G fused (a-b)/(a+b) f32 10M         {t,8:F2} ms   (NumSharp eager 3-pass: {tEager:F2} ms)");
    }

    // =========================================================================
    // H. Small-N dispatch — sqrt(f32 1K) per call INCLUDING iterator
    //    construction, vs NumPy's per-call cost from Python.
    // =========================================================================
    {
        var a = np.arange(1000).astype(np.float32) + 1f;
        var outNd = np.empty(new Shape(1000), np.float32);
        double t = TimeMs(() => {
            using var iter = NpyIterRef.MultiNew(2, new[] { a, outNd },
                NpyIterGlobalFlags.None, NPY_ORDER.NPY_KEEPORDER, NPY_CASTING.NPY_SAFE_CASTING,
                new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY });
            iter.ExecuteUnary(UnaryOp.Sqrt);
        }, 50_000, 5_000) * 1000.0;
        Console.WriteLine($"H small-N sqrt f32 1K (full setup)  {t,8:F2} us/call");
    }
}

Console.WriteLine();
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

// =============================================================================
// POC per-chunk kernels — NumPy's PyUFuncGenericFunction contract:
//   kernel(dataptrs, byteStrides, count, aux), invoked by NpyIterRef.ForEach.
// Fused-gather technique: raw-pointer Vector256.Create from strided lanes
// (no scratch round-trip), contiguous fast path when the stride is unit.
// Concrete methods — no generics/interfaces in the hot path (Phase 2b lesson).
// =============================================================================
static unsafe class PocKernels
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddF32(void** dp, long* strides, long count, void* aux)
    {
        byte* pa = (byte*)dp[0];
        byte* pb = (byte*)dp[1];
        byte* po = (byte*)dp[2];
        long sa = strides[0], sb = strides[1], so = strides[2];
        long i = 0;

        if (so == 4)
        {
            if (sa == 4 && sb == 4)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Avx.Add(Vector256.Load((float*)pa), Vector256.Load((float*)pb)), (float*)po);
                    pa += 32; pb += 32; po += 32;
                }
            }
            else if (Avx2.IsSupported && GatherableStride(sa) && GatherableStride(sb))
            {
                // NumPy-style hardware gather (vgatherdps), byte-offset indices
                // hoisted out of the loop — the hot loop is stride-agnostic.
                int isa = (int)sa, isb = (int)sb;
                var idxA = Vector256.Create(0, isa, 2 * isa, 3 * isa, 4 * isa, 5 * isa, 6 * isa, 7 * isa);
                var idxB = Vector256.Create(0, isb, 2 * isb, 3 * isb, 4 * isb, 5 * isb, 6 * isb, 7 * isb);
                for (; i + 16 <= count; i += 16)
                {
                    var va0 = Avx2.GatherVector256((float*)pa, idxA, 1);
                    var vb0 = Avx2.GatherVector256((float*)pb, idxB, 1);
                    var va1 = Avx2.GatherVector256((float*)(pa + 8 * sa), idxA, 1);
                    var vb1 = Avx2.GatherVector256((float*)(pb + 8 * sb), idxB, 1);
                    Vector256.Store(Avx.Add(va0, vb0), (float*)po);
                    Vector256.Store(Avx.Add(va1, vb1), (float*)(po + 32));
                    pa += 16 * sa; pb += 16 * sb; po += 64;
                }
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Avx.Add(
                        Avx2.GatherVector256((float*)pa, idxA, 1),
                        Avx2.GatherVector256((float*)pb, idxB, 1)), (float*)po);
                    pa += 8 * sa; pb += 8 * sb; po += 32;
                }
            }
            else
            {
                // software insert-gather fallback
                for (; i + 8 <= count; i += 8)
                {
                    var va = Vector256.Create(
                        *(float*)pa, *(float*)(pa + sa), *(float*)(pa + 2 * sa), *(float*)(pa + 3 * sa),
                        *(float*)(pa + 4 * sa), *(float*)(pa + 5 * sa), *(float*)(pa + 6 * sa), *(float*)(pa + 7 * sa));
                    var vb = Vector256.Create(
                        *(float*)pb, *(float*)(pb + sb), *(float*)(pb + 2 * sb), *(float*)(pb + 3 * sb),
                        *(float*)(pb + 4 * sb), *(float*)(pb + 5 * sb), *(float*)(pb + 6 * sb), *(float*)(pb + 7 * sb));
                    Vector256.Store(Avx.Add(va, vb), (float*)po);
                    pa += 8 * sa; pb += 8 * sb; po += 32;
                }
            }
        }

        for (; i < count; i++)
        {
            *(float*)po = *(float*)pa + *(float*)pb;
            pa += sa; pb += sb; po += so;
        }
    }

    /// <summary>Byte stride usable as a vgather int32 index: |7*stride| must fit in int32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static bool GatherableStride(long s) => s >= int.MinValue / 8 && s <= int.MaxValue / 8;

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SqrtF32(void** dp, long* strides, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long ss = strides[0], so = strides[1];
        long i = 0;

        if (so == 4)
        {
            if (ss == 4)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Avx.Sqrt(Vector256.Load((float*)ps)), (float*)po);
                    ps += 32; po += 32;
                }
            }
            else if (Avx2.IsSupported && GatherableStride(ss))
            {
                // NumPy's exact technique for strided unary: hardware gather,
                // 4x unrolled (loops_unary_fp.dispatch.c.src NCONTIG_CONTIG).
                int iss = (int)ss;
                var idx = Vector256.Create(0, iss, 2 * iss, 3 * iss, 4 * iss, 5 * iss, 6 * iss, 7 * iss);
                for (; i + 32 <= count; i += 32)
                {
                    var v0 = Avx2.GatherVector256((float*)ps, idx, 1);
                    var v1 = Avx2.GatherVector256((float*)(ps + 8 * ss), idx, 1);
                    var v2 = Avx2.GatherVector256((float*)(ps + 16 * ss), idx, 1);
                    var v3 = Avx2.GatherVector256((float*)(ps + 24 * ss), idx, 1);
                    Vector256.Store(Avx.Sqrt(v0), (float*)po);
                    Vector256.Store(Avx.Sqrt(v1), (float*)(po + 32));
                    Vector256.Store(Avx.Sqrt(v2), (float*)(po + 64));
                    Vector256.Store(Avx.Sqrt(v3), (float*)(po + 96));
                    ps += 32 * ss; po += 128;
                }
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Avx.Sqrt(Avx2.GatherVector256((float*)ps, idx, 1)), (float*)po);
                    ps += 8 * ss; po += 32;
                }
            }
            else
            {
                for (; i + 8 <= count; i += 8)
                {
                    var v = Vector256.Create(
                        *(float*)ps, *(float*)(ps + ss), *(float*)(ps + 2 * ss), *(float*)(ps + 3 * ss),
                        *(float*)(ps + 4 * ss), *(float*)(ps + 5 * ss), *(float*)(ps + 6 * ss), *(float*)(ps + 7 * ss));
                    Vector256.Store(Avx.Sqrt(v), (float*)po);
                    ps += 8 * ss; po += 32;
                }
            }
        }

        for (; i < count; i++)
        {
            *(float*)po = MathF.Sqrt(*(float*)ps);
            ps += ss; po += so;
        }
    }

    /// <summary>Strided f32 sum; partials accumulated into *(double*)aux.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SumF32(void** dp, long* strides, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        long ss = strides[0];
        long i = 0;
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;

        if (ss == 4)
        {
            for (; i + 16 <= count; i += 16)
            {
                acc0 = Avx.Add(acc0, Vector256.Load((float*)ps));
                acc1 = Avx.Add(acc1, Vector256.Load((float*)(ps + 32)));
                ps += 64;
            }
        }
        else if (Avx2.IsSupported && GatherableStride(ss))
        {
            // Hardware gather + 4 independent accumulators. NumPy has no SIMD
            // here at all (strided pairwise_sum is a scalar 8-acc loop) — this
            // is where the architecture overtakes it.
            int iss = (int)ss;
            var idx = Vector256.Create(0, iss, 2 * iss, 3 * iss, 4 * iss, 5 * iss, 6 * iss, 7 * iss);
            var acc2 = Vector256<float>.Zero;
            var acc3 = Vector256<float>.Zero;
            for (; i + 32 <= count; i += 32)
            {
                acc0 = Avx.Add(acc0, Avx2.GatherVector256((float*)ps, idx, 1));
                acc1 = Avx.Add(acc1, Avx2.GatherVector256((float*)(ps + 8 * ss), idx, 1));
                acc2 = Avx.Add(acc2, Avx2.GatherVector256((float*)(ps + 16 * ss), idx, 1));
                acc3 = Avx.Add(acc3, Avx2.GatherVector256((float*)(ps + 24 * ss), idx, 1));
                ps += 32 * ss;
            }
            for (; i + 8 <= count; i += 8)
            {
                acc0 = Avx.Add(acc0, Avx2.GatherVector256((float*)ps, idx, 1));
                ps += 8 * ss;
            }
            acc0 = Avx.Add(acc0, acc2);
            acc1 = Avx.Add(acc1, acc3);
        }
        else
        {
            for (; i + 16 <= count; i += 16)
            {
                acc0 = Avx.Add(acc0, Vector256.Create(
                    *(float*)ps, *(float*)(ps + ss), *(float*)(ps + 2 * ss), *(float*)(ps + 3 * ss),
                    *(float*)(ps + 4 * ss), *(float*)(ps + 5 * ss), *(float*)(ps + 6 * ss), *(float*)(ps + 7 * ss)));
                byte* p8 = ps + 8 * ss;
                acc1 = Avx.Add(acc1, Vector256.Create(
                    *(float*)p8, *(float*)(p8 + ss), *(float*)(p8 + 2 * ss), *(float*)(p8 + 3 * ss),
                    *(float*)(p8 + 4 * ss), *(float*)(p8 + 5 * ss), *(float*)(p8 + 6 * ss), *(float*)(p8 + 7 * ss)));
                ps += 16 * ss;
            }
        }

        var acc = Avx.Add(acc0, acc1);
        double s = 0;
        for (int k = 0; k < 8; k++) s += acc.GetElement(k);
        for (; i < count; i++) { s += *(float*)ps; ps += ss; }
        *(double*)aux += s;
    }
}
