#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_core_bench.cs — the ITERATOR itself, measured: construction across
// flag configurations, traversal/orchestration overhead across chunk profiles,
// buffering/cast windows, index tracking, per-element protocol, small-N
// pipeline scaling. Companion: npyiter_core_bench.py (NumPy side, same ids).
//
// This is NOT the kernel-parity POC (npyiter_parity_poc.cs measures end-to-end
// op throughput). Here every kernel is deliberately TRIVIAL (memcpy / scalar /
// matched-to-NumPy loop families) so the measured time is dominated by what the
// ITERATOR does: broadcast resolution, order/coalescing, per-chunk dispatch,
// iternext, window fills, casts, index tracking.
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_core_bench.cs
// (file-based apps build Debug by default; the guard below refuses to mislead)
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgScript = Attribute.GetCustomAttribute(typeof(Kern).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if ((dbgScript?.IsJITOptimizerDisabled ?? false) || (dbgCore?.IsJITOptimizerDisabled ?? false))
{
    Console.WriteLine("FATAL: Debug-JITted assemblies (script or NumSharp.Core) — numbers would be INVALID.");
    Console.WriteLine("Run:   dotnet run -c Release - < benchmark/poc/npyiter_core_bench.cs");
    return;
}

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

// best-of-rounds of (loop of iters)/iters, in ms per call
double BestMs(Action body, int iters, int warm, int rounds = 5)
{
    for (int i = 0; i < warm; i++) body();
    double best = double.MaxValue;
    for (int r = 0; r < rounds; r++)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) body();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
    }
    return best;
}

void Row(string id, string label, double ms, string note = "")
{
    string val = ms >= 1.0 ? $"{ms,10:F3} ms" : ms >= 0.001 ? $"{ms * 1000,10:F2} us" : $"{ms * 1e6,10:F1} ns";
    Console.WriteLine($"{id,-6} {label,-52} {val}  {note}");
}

// static op-flag arrays (call-invariant; production hoists these the same way)
var RO1 = new[] { NpyIterPerOpFlags.READONLY };
var RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
var RO_RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
var ELW3 = new[]
{
    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
    NpyIterPerOpFlags.READONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
    NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.OVERLAP_ASSUME_ELEMENTWISE_PER_OP,
};
const NPY_ORDER K = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;

Console.WriteLine($"NumSharp NpyIter core bench — {Environment.ProcessorCount} cores, V256={Vector256.IsHardwareAccelerated}");
Console.WriteLine($"{"id",-6} {"aspect",-52} {"per call",13}");
Console.WriteLine(new string('-', 90));

unsafe
{
    // =========================================================================
    // C — CONSTRUCTION (construct + dispose only, no execution)
    // =========================================================================
    {
        var a = np.arange(1000).astype(np.float64);
        var b = np.arange(1000).astype(np.float64) + 1.0;
        var o = np.empty(new Shape(1000), np.float64);
        var a32 = np.arange(1000).astype(np.float32);
        var o64 = np.empty(new Shape(1000), np.float64);
        var g32 = np.arange(1024).astype(np.float64).reshape(32, 32);
        var row32 = np.arange(32).astype(np.float64);
        var og32 = np.empty(new Shape(32, 32), np.float64);
        var a4d = np.arange(1024).astype(np.float64).reshape(8, 8, 4, 4);
        var o4d = np.empty(new Shape(8, 8, 4, 4), np.float64);
        var back2d = np.arange(64 * 8).astype(np.float64).reshape(64, 8);
        var sview = back2d[":, :4"];
        var sdst = np.empty(new Shape(64, 4), np.float64);

        var ops2 = new[] { a, o };
        var ops3 = new[] { a, b, o };
        var opsCast = new[] { a32, o64 };
        var opsBc = new[] { g32, row32, og32 };
        var ops4d = new[] { a4d, o4d };
        var opsSv = new[] { sview, sdst };
        var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };

        var ops8 = new NDArray[8];
        for (int i = 0; i < 7; i++) ops8[i] = a;
        ops8[7] = o;
        var ro8 = new NpyIterPerOpFlags[8];
        for (int i = 0; i < 7; i++) ro8[i] = NpyIterPerOpFlags.READONLY;
        ro8[7] = NpyIterPerOpFlags.WRITEONLY;

        Row("C1", "ctor 1-op contig 1K f64, no flags",
            BestMs(() => { using var it = NpyIterRef.New(a); }, 400_000, 50_000));
        Row("C2", "ctor 2-op [a,out] contig 1K",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, ops2, NpyIterGlobalFlags.None, K, SAFE, RO_WO); }, 400_000, 50_000));
        Row("C3", "ctor 3-op [a,b,out] contig 1K",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, ops3, NpyIterGlobalFlags.None, K, SAFE, RO_RO_WO); }, 400_000, 50_000));
        Row("C4", "ctor 3-op EXTERNAL_LOOP",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, ops3, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO); }, 400_000, 50_000));
        Row("C5", "ctor 3-op broadcast (32,32)+(32,)->(32,32)",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, opsBc, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO); }, 200_000, 25_000));
        Row("C6", "ctor 2-op BUFFERED cast f32->f64 eager (EXL|GROW)",
            BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(2, opsCast,
                    NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER,
                    K, SAFE, RO_WO, f64x2);
            }, 100_000, 12_000));
        Row("C7", "ctor C6 + DELAY_BUFALLOC (defer alloc+fill)",
            BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(2, opsCast,
                    NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER | NpyIterGlobalFlags.DELAY_BUFALLOC,
                    K, SAFE, RO_WO, f64x2);
            }, 200_000, 25_000));
        Row("C8", "ctor 1-op MULTI_INDEX (32,32)",
            BestMs(() => { using var it = NpyIterRef.New(g32, NpyIterGlobalFlags.MULTI_INDEX); }, 400_000, 50_000));
        Row("C9", "ctor 1-op C_INDEX (32,32)",
            BestMs(() => { using var it = NpyIterRef.New(g32, NpyIterGlobalFlags.C_INDEX); }, 400_000, 50_000));
        Row("C10", "ctor 3-op ufunc config (EXL|BUF|GROW|DELAY|CIO|ZS)",
            BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(3, ops3,
                    NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.GROWINNER |
                    NpyIterGlobalFlags.DELAY_BUFALLOC | NpyIterGlobalFlags.COPY_IF_OVERLAP | NpyIterGlobalFlags.ZEROSIZE_OK,
                    K, SAFE, ELW3);
            }, 200_000, 25_000));
        Row("C11", "ctor 8-op contig 1K",
            BestMs(() => { using var it = NpyIterRef.MultiNew(8, ops8, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, ro8); }, 200_000, 25_000));
        Row("C12", "ctor 2-op 4-D contig (8,8,4,4) [coalesce 4 axes]",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, ops4d, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO); }, 200_000, 25_000));
        Row("C13", "ctor 2-op strided 2-D view (64,4) of (64,8)",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, opsSv, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO); }, 200_000, 25_000));

        // anchor: what production np.add(out=) pays end-to-end at 1K
        np.add(a, b, o);
        Row("H0", "anchor np.add(a,b,out=o) 1K f64 e2e",
            BestMs(() => np.add(a, b, o), 100_000, 12_000));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // T — TRAVERSAL / ORCHESTRATION (trivial kernels; iterator cost dominates)
    // =========================================================================

    // T1: contiguous copy 10M f64 — coalesces to ONE chunk (memcpy anchor)
    {
        const int N = 10_000_000;
        var src = np.arange(N).astype(np.float64);
        var dst = np.empty(new Shape(N), np.float64);
        var ops = new[] { src, dst };
        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
            it.ForEach(Kern.CopyF64);
        }, 25, 8, 7);
        Check(dst.GetDouble(N - 3) == src.GetDouble(N - 3), "T1 copy");
        Row("T1", "copy contig f64 10M (1 chunk)", t, $"{80.0 / t:F0}+{80.0 / t:F0} GB/s rw");
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T2: strided-row copy, total fixed 2M f64, inner width sweep
    //     rows = 2M/w chunks; measures iternext + per-chunk dispatch scaling
    {
        const int TOTAL = 2_097_152;
        foreach (int w in new[] { 4, 16, 64, 256, 1024 })
        {
            int rows = TOTAL / w;
            var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
            var sv = back[$":, :{w}"];
            var dst = np.empty(new Shape(rows, w), np.float64);
            var ops = new[] { sv, dst };
            double t = BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
                it.ForEach(Kern.CopyF64);
            }, w <= 16 ? 12 : 25, 5, 7);
            Check(dst.GetDouble(rows - 1, w - 1) == sv.GetDouble(rows - 1, w - 1), $"T2 w={w}");
            Row($"T2.{w}", $"copy strided rows f64 2M total, inner w={w} ({rows} chunks)", t,
                $"{t * 1e6 / rows:F0} ns/chunk");

            if (w == 4)
            {
                double tg = BestMs(() =>
                {
                    using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
                    it.ExecuteGeneric(new CopyLoop());
                }, 12, 5, 7);
                Row("T2g", "  same w=4 via ExecuteGeneric (struct, no delegate)", tg, $"{tg * 1e6 / rows:F0} ns/chunk");

                double tc = BestMs(() => NpyIter.Copy(dst, sv), 12, 5, 7);
                Row("T2c", "  same w=4 via NpyIter.Copy (production route)", tc, $"{tc * 1e6 / rows:F0} ns/chunk");
            }
        }
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T3: transposed copy (1448^2 f64 ~= 2M) — order resolution stress
    {
        const int n = 1448;
        var A = np.arange(n * n).astype(np.float64).reshape(n, n);
        var At = A.T;
        var D = np.empty(new Shape(n, n), np.float64);
        var ops = new[] { At, D };
        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
            it.ForEach(Kern.CopyF64);
        }, 12, 5, 7);
        Check(D.GetDouble(5, 7) == A.GetDouble(7, 5), "T3 transpose copy");
        Row("T3", "copy transposed (1448,1448) f64 -> contig", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T4: broadcast add f32 (2000,2000) — row then col operand
    {
        var a = (np.arange(4_000_000).astype(np.float32) % 977f).reshape(2000, 2000);
        var row = np.arange(2000).astype(np.float32);
        var col = np.arange(2000).astype(np.float32).reshape(2000, 1);
        var o = np.empty(new Shape(2000, 2000), np.float32);

        var opsR = new[] { a, row, o };
        double tr = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(3, opsR, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
            it.ForEach(Kern.AddF32);
        }, 25, 8, 7);
        Check(o.GetSingle(3, 17) == a.GetSingle(3, 17) + row.GetSingle(17), "T4r row bcast");
        Row("T4r", "add row-bcast (2000,2000)+(2000,) f32", tr);

        var opsC = new[] { a, col, o };
        double tc = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(3, opsC, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
            it.ForEach(Kern.AddF32);
        }, 25, 8, 7);
        Check(o.GetSingle(3, 17) == a.GetSingle(3, 17) + col.GetSingle(3, 0), "T4c col bcast");
        Row("T4c", "add col-bcast (2000,2000)+(2000,1) f32", tc);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T5: buffered cast copy f32 -> f64, 4M — the window machinery itself
    {
        const int M = 4_194_304;
        var s32 = np.arange(M).astype(np.float32);
        var d64 = np.empty(new Shape(M), np.float64);
        var ops = new[] { s32, d64 };
        var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };

        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, ops,
                NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER,
                K, SAFE, RO_WO, f64x2);
            it.ForEach(Kern.CopyF64);
        }, 25, 8, 7);
        Check(d64.GetDouble(123_456) == (double)s32.GetSingle(123_456), "T5 cast copy");
        Row("T5", "buffered cast copy f32->f64 4M (GROWINNER)", t);

        double tn = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, ops,
                NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP,
                K, SAFE, RO_WO, f64x2);
            it.ForEach(Kern.CopyF64);
        }, 25, 8, 7);
        Row("T5n", "  same, no GROWINNER (8192-elem windows)", tn);

        double tb = BestMs(() =>
        {
            using var it = NpyIterRef.AdvancedNew(2, ops,
                NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP,
                K, SAFE, RO_WO, f64x2, -1, null, null, 65536);
            it.ForEach(Kern.CopyF64);
        }, 25, 8, 7);
        Row("T5b", "  same, no GROW, bufferSize=65536", tb);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T6: buffered mixed binary f32+f64 -> f64, 4M (ufunc-style buffering)
    {
        const int M = 4_194_304;
        var a32 = (np.arange(M).astype(np.float32) % 977f) + 1f;
        var b64 = (np.arange(M).astype(np.float64) % 31.0) + 2.0;
        var o64 = np.empty(new Shape(M), np.float64);
        var ops = new[] { a32, b64, o64 };
        var f64x3 = new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double };

        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(3, ops,
                NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER,
                K, SAFE, RO_RO_WO, f64x3);
            it.ForEach(Kern.AddF64);
        }, 25, 8, 7);
        Check(Math.Abs(o64.GetDouble(777) - ((double)a32.GetSingle(777) + b64.GetDouble(777))) < 1e-12, "T6 mixed add");
        Row("T6", "buffered mixed add f32+f64->f64 4M", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T7: per-element protocol (no EXTERNAL_LOOP) on (1000,1000) f64
    //     base coalesces to 1-D; C_INDEX / MULTI_INDEX forbid coalescing
    {
        var g = np.arange(1_000_000).astype(np.float64).reshape(1000, 1000);
        const long N = 1_000_000;

        double tBase = BestMs(() =>
        {
            using var it = NpyIterRef.New(g);
            var next = it.GetIterNext();
            void** dp = it.GetDataPtrArray();
            double s = 0;
            do { s += *(double*)dp[0]; } while (next(ref *it.RawState));
            if (s < 0) Console.WriteLine("never");
        }, 20, 6, 7);
        Row("T7a", "per-element walk+read (1000,1000) f64 [coalesced]", tBase, $"{tBase * 1e6 / N:F1} ns/elem");

        double tC = BestMs(() =>
        {
            using var it = NpyIterRef.New(g, NpyIterGlobalFlags.C_INDEX);
            var next = it.GetIterNext();
            void** dp = it.GetDataPtrArray();
            double s = 0;
            do { s += *(double*)dp[0]; } while (next(ref *it.RawState));
            if (s < 0) Console.WriteLine("never");
        }, 12, 4, 7);
        Row("T7b", "  + C_INDEX (flat-index tracking, 2-D walk)", tC, $"{tC * 1e6 / N:F1} ns/elem");

        double tM = BestMs(() =>
        {
            using var it = NpyIterRef.New(g, NpyIterGlobalFlags.MULTI_INDEX);
            var next = it.GetIterNext();
            void** dp = it.GetDataPtrArray();
            double s = 0;
            do { s += *(double*)dp[0]; } while (next(ref *it.RawState));
            if (s < 0) Console.WriteLine("never");
        }, 12, 4, 7);
        Row("T7c", "  + MULTI_INDEX (2-D walk, no coalesce)", tM, $"{tM * 1e6 / N:F1} ns/elem");
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // T8: full reduce sum via ExecuteReducing — contig 10M, strided ::2 of 2M
    {
        var r10 = (np.arange(10_000_000).astype(np.float64) % 97.0) + 1.0;
        double sum = 0;
        double t = BestMs(() =>
        {
            using var it = NpyIterRef.New(r10, NpyIterGlobalFlags.EXTERNAL_LOOP);
            sum = it.ExecuteReducing<SumLoop, double>(default, 0.0);
        }, 25, 8, 7);
        double expect = (double)np.sum(r10);
        Check(Math.Abs(sum - expect) / expect < 1e-9, $"T8 sum {sum} vs {expect}");
        Row("T8", "reduce sum f64 10M contig via ExecuteReducing", t);

        var back = (np.arange(2_000_000).astype(np.float64) % 53.0) + 1.0;
        var sv = back["::2"];
        double t2 = BestMs(() =>
        {
            using var it = NpyIterRef.New(sv, NpyIterGlobalFlags.EXTERNAL_LOOP);
            sum = it.ExecuteReducing<SumLoop, double>(default, 0.0);
        }, 100, 25, 7);
        expect = (double)np.sum(sv);
        Check(Math.Abs(sum - expect) / expect < 1e-9, $"T8s sum {sum} vs {expect}");
        Row("T8s", "reduce sum f64 1M strided a[::2]", t2);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // H — SMALL-N PIPELINE SCALING: ctor + ForEach(add f64) + dispose per call
    // =========================================================================
    {
        foreach (int n in new[] { 8, 64, 512, 4096, 32_768, 262_144, 2_097_152 })
        {
            var a = np.arange(n).astype(np.float64);
            var b = np.arange(n).astype(np.float64) + 1.0;
            var o = np.empty(new Shape(n), np.float64);
            var ops = new[] { a, b, o };
            int iters = n <= 4096 ? 100_000 : n <= 262_144 ? 2_000 : 100;
            int warm = iters / 8;
            double t = BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(3, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
                it.ForEach(Kern.AddF64);
            }, iters, warm);
            Check(o.GetDouble(n - 1) == a.GetDouble(n - 1) + b.GetDouble(n - 1), $"H{n}");
            Row($"H{n}", $"iter pipeline add f64 N={n} (ctor+run+dispose)", t);
        }

        // HR: iterator REUSE — Reset + ForEach only (the floor without ctor).
        // NumPy cannot do this from Python; C-level consumers can (ResetBasePointers).
        {
            const int n = 512;
            var a = np.arange(n).astype(np.float64);
            var b = np.arange(n).astype(np.float64) + 1.0;
            var o = np.empty(new Shape(n), np.float64);
            var ops = new[] { a, b, o };
            var it = NpyIterRef.MultiNew(3, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
            for (int i = 0; i < 50_000; i++) { it.Reset(); it.ForEach(Kern.AddF64); }
            double best = double.MaxValue;
            for (int r = 0; r < 5; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 400_000; i++) { it.Reset(); it.ForEach(Kern.AddF64); }
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / 400_000);
            }
            it.Dispose();
            Check(o.GetDouble(n - 1) == a.GetDouble(n - 1) + b.GetDouble(n - 1), "HR512");
            Row("HR512", "REUSED iterator (Reset+ForEach only) N=512", best);
        }
    }
}

Console.WriteLine(new string('-', 90));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

// =============================================================================
// Trivial kernels — matched to NumPy's loop families so the KERNEL side is
// equal-cost and the iterator differences dominate:
//   copy:   contig -> memcpy (NumPy contig transfer); strided -> scalar loop
//           (NumPy _aligned_strided_to_strided_size8 is scalar)
//   add:    contig -> V256 loop (NumPy AVX loop); stride-0 operand -> bcast
//           register (NumPy scalar-arg specialization); strided -> scalar
//   sum:    contig -> 4-acc V256; strided -> scalar 4-acc (NumPy pairwise
//           strided is scalar)
// =============================================================================
static unsafe class Kern
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long ss = st[0], so = st[1];
        if (ss == 8 && so == 8)
        {
            Buffer.MemoryCopy(ps, po, count * 8, count * 8);
            return;
        }
        for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddF64(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0];
        byte* pb = (byte*)dp[1];
        byte* po = (byte*)dp[2];
        long sa = st[0], sb = st[1], so = st[2];
        long i = 0;
        if (so == 8)
        {
            if (sa == 8 && sb == 8)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256.Load((double*)pa) + Vector256.Load((double*)pb), (double*)po);
                    Vector256.Store(Vector256.Load((double*)(pa + 32)) + Vector256.Load((double*)(pb + 32)), (double*)(po + 32));
                    pa += 64; pb += 64; po += 64;
                }
            }
            else if (sa == 8 && sb == 0)
            {
                var vb = Vector256.Create(*(double*)pb);
                for (; i + 4 <= count; i += 4)
                {
                    Vector256.Store(Vector256.Load((double*)pa) + vb, (double*)po);
                    pa += 32; po += 32;
                }
            }
            else if (sa == 0 && sb == 8)
            {
                var va = Vector256.Create(*(double*)pa);
                for (; i + 4 <= count; i += 4)
                {
                    Vector256.Store(va + Vector256.Load((double*)pb), (double*)po);
                    pb += 32; po += 32;
                }
            }
        }
        for (; i < count; i++)
        {
            *(double*)po = *(double*)pa + *(double*)pb;
            pa += sa; pb += sb; po += so;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddF32(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0];
        byte* pb = (byte*)dp[1];
        byte* po = (byte*)dp[2];
        long sa = st[0], sb = st[1], so = st[2];
        long i = 0;
        if (so == 4)
        {
            if (sa == 4 && sb == 4)
            {
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256.Load((float*)pa) + Vector256.Load((float*)pb), (float*)po);
                    pa += 32; pb += 32; po += 32;
                }
            }
            else if (sa == 4 && sb == 0)
            {
                var vb = Vector256.Create(*(float*)pb);
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256.Load((float*)pa) + vb, (float*)po);
                    pa += 32; po += 32;
                }
            }
            else if (sa == 0 && sb == 4)
            {
                var va = Vector256.Create(*(float*)pa);
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(va + Vector256.Load((float*)pb), (float*)po);
                    pb += 32; po += 32;
                }
            }
        }
        for (; i < count; i++)
        {
            *(float*)po = *(float*)pa + *(float*)pb;
            pa += sa; pb += sb; po += so;
        }
    }
}

readonly struct CopyLoop : INpyInnerLoop
{
    public unsafe void Execute(void** dataptrs, long* strides, long count)
        => Kern.CopyF64(dataptrs, strides, count, null);
}

readonly struct SumLoop : INpyReducingInnerLoop<double>
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public unsafe bool Execute(void** dp, long* st, long count, ref double acc)
    {
        byte* ps = (byte*)dp[0];
        long ss = st[0];
        long i = 0;
        double s = 0;
        if (ss == 8)
        {
            var a0 = Vector256<double>.Zero;
            var a1 = Vector256<double>.Zero;
            var a2 = Vector256<double>.Zero;
            var a3 = Vector256<double>.Zero;
            for (; i + 16 <= count; i += 16)
            {
                a0 += Vector256.Load((double*)ps);
                a1 += Vector256.Load((double*)(ps + 32));
                a2 += Vector256.Load((double*)(ps + 64));
                a3 += Vector256.Load((double*)(ps + 96));
                ps += 128;
            }
            s = Vector256.Sum((a0 + a1) + (a2 + a3));
        }
        else
        {
            double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (; i + 4 <= count; i += 4)
            {
                s0 += *(double*)ps;
                s1 += *(double*)(ps + ss);
                s2 += *(double*)(ps + 2 * ss);
                s3 += *(double*)(ps + 3 * ss);
                ps += 4 * ss;
            }
            s = (s0 + s1) + (s2 + s3);
        }
        for (; i < count; i++) { s += *(double*)ps; ps += ss; }
        acc += s;
        return true;
    }
}
