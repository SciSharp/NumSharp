#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_frontier2_bench.cs — frontier round 2: overlap/alias taxes,
// comparison->bool, early-exit reductions, broadcast-view reduce
// (materialization probe), mixed-dtype & scalar small-N, empty arrays, 8-D
// construction, and the parallel banded-iteration dividend (Wave-6.2 preview).
// Companion: npyiter_frontier2_bench.py (same ids).
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_frontier2_bench.cs
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgScript = Attribute.GetCustomAttribute(typeof(F2Kern).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if ((dbgScript?.IsJITOptimizerDisabled ?? false) || (dbgCore?.IsJITOptimizerDisabled ?? false))
{
    Console.WriteLine("FATAL: Debug-JITted assemblies — rerun: dotnet run -c Release - < benchmark/poc/npyiter_frontier2_bench.cs");
    return;
}

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

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
    Console.WriteLine($"{id,-6} {label,-56} {val}  {note}");
}

var RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
const NPY_ORDER K = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;

Console.WriteLine($"NumSharp NpyIter frontier-2 bench — {Environment.ProcessorCount} threads, V256={Vector256.IsHardwareAccelerated}");
Console.WriteLine($"{"id",-6} {"aspect",-56} {"per call",13}");
Console.WriteLine(new string('-', 95));

unsafe
{
    // =========================================================================
    // C14 — 8-D construction (high-ndim ctor scaling)
    // =========================================================================
    {
        var a8 = np.arange(65536).astype(np.float64).reshape(4, 4, 4, 4, 4, 4, 4, 4);
        var o8 = np.empty(new Shape(4, 4, 4, 4, 4, 4, 4, 4), np.float64);
        var ops = new[] { a8, o8 };
        Row("C14", "ctor 2-op 8-D contig (4^8) EXLOOP",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO); }, 200_000, 25_000));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // V — overlap / aliasing taxes (COPY_IF_OVERLAP machinery per call)
    // =========================================================================
    {
        const int M = 4_194_304;
        var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(M).astype(np.float64) % 31.0) + 2.0;

        // V1: exact alias — out IS an input (the OVERLAP_ASSUME_ELEMENTWISE
        //     short-circuit class: no copy needed, in-place is well-defined)
        np.add(a, b, a);
        double t = BestMs(() => np.add(a, b, a), 25, 8, 7);
        Row("V1", "in-place np.add(a, b, out=a) f64 4M (exact alias)", t);

        // V2: shifted overlap — write-ahead direction, forces temp + write-back
        var x = (np.arange(M).astype(np.float64) % 53.0) + 1.0;
        var xs = x[":-1"];
        var xd = x["1:"];
        np.add(xs, xs, xd);   // correctness of this pattern pinned by Wave-1.1 tests
        t = BestMs(() => np.add(xs, xs, xd), 12, 4, 7);
        Row("V2", "np.add(x[:-1], x[:-1], out=x[1:]) 4M (forced copy)", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // D — comparison -> bool output (1-byte writes)
    // =========================================================================
    {
        const int M = 4_194_304;
        var a = (np.arange(M).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(M).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(M), np.bool_);
        np.less(a, b, o);
        double t = BestMs(() => np.less(a, b, o), 25, 8, 7);
        Check(o.GetBoolean(777) == (a.GetDouble(777) < b.GetDouble(777)), "D1");
        Row("D1", "np.less(a, b, out=bool) f64 4M", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // E — boolean reduce: full scan vs early exit
    // =========================================================================
    {
        const int M = 10_000_000;
        var idx = np.arange(M);
        NDArray allFalse = idx == -1;          // no true anywhere
        NDArray earlyHit = idx == 1000;        // true at 1000

        bool r = (bool)np.any(allFalse);
        Check(!r, "E1 result");
        Row("E1", "np.any(bool 10M, ALL-FALSE: full scan)",
            BestMs(() => { var _ = np.any(allFalse); }, 50, 12, 7));
        r = (bool)np.any(earlyHit);
        Check(r, "E2 result");
        Row("E2", "np.any(bool 10M, TRUE at idx 1000: early exit)",
            BestMs(() => { var _ = np.any(earlyHit); }, 200, 50, 7));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // F — reduce over a BROADCAST view (materialization probe)
    // =========================================================================
    {
        var a8k = (np.arange(8192).astype(np.float64) % 97.0) + 1.0;
        var bc = np.broadcast_to(a8k, new Shape(1024, 8192));   // 8M logical / 64KB physical
        double expect = 1024.0 * (double)np.sum(a8k);
        double got = (double)np.sum(bc);
        Check(Math.Abs(got - expect) / expect < 1e-9, $"F1 sum {got} vs {expect}");
        Row("F1", "np.sum over broadcast_to(8K -> (1024,8192))",
            BestMs(() => { var _ = np.sum(bc); }, 25, 8, 7));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // M/O — small-N frontier: mixed dtype, array+scalar, empty
    // =========================================================================
    {
        var ai = np.arange(1000).astype(np.int32);
        var bf = np.arange(1000).astype(np.float64) + 0.5;
        var of = np.empty(new Shape(1000), np.float64);
        np.add(ai, bf, of);
        Check(of.GetDouble(777) == 777 + 777.5, "M1");
        Row("M1", "np.add(i32 1K, f64 1K, out=f64) mixed small-N",
            BestMs(() => np.add(ai, bf, of), 100_000, 12_000));

        var a1k = np.arange(1000).astype(np.float64);
        var s = NDArray.Scalar(5.0, NPTypeCode.Double);
        var o1k = np.empty(new Shape(1000), np.float64);
        np.add(a1k, s, o1k);
        Check(o1k.GetDouble(777) == 782.0, "O3");
        Row("O3", "np.add(a 1K, scalar, out=) array+scalar small-N",
            BestMs(() => np.add(a1k, s, o1k), 100_000, 12_000));

        var e1 = np.empty(new Shape(0), np.float64);
        var e2 = np.empty(new Shape(0), np.float64);
        var eo = np.empty(new Shape(0), np.float64);
        np.add(e1, e2, eo);
        Row("O4", "np.add on EMPTY (0,) arrays, out=",
            BestMs(() => np.add(e1, e2, eo), 200_000, 25_000));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // PAR — the parallel dividend (Wave-6.2 preview): f64 sin is SCALAR on
    //       both sides at AVX2 (NumPy's f64 SIMD sin needs AVX512), so this is
    //       the compute-bound class where banded parallel iteration shines.
    // =========================================================================
    {
        const int M = 4_194_304;
        const int BANDS = 8;
        var src = (np.arange(M).astype(np.float64) % 6.283185) - 3.1415926;
        var dst = np.empty(new Shape(M), np.float64);

        Row("PAR0", "production np.sin(x, out=) f64 4M (single-thread)",
            BestMs(() => np.sin(src, dst), 5, 2, 5));

        var opsAll = new[] { src, dst };
        double t1 = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, opsAll, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
            it.ForEach(F2Kern.SinF64);
        }, 5, 2, 5);
        Check(Math.Abs(dst.GetDouble(777) - Math.Sin(src.GetDouble(777))) < 1e-15, "PAR1");
        Row("PAR1", "iterator sin f64 4M, ONE iterator (single-thread)", t1);

        // 8 disjoint row bands, one iterator per band, Parallel.For — exactly
        // what Wave 6.2 automates via PARALLEL_SAFE + RANGED + Copy().
        var src2d = src.reshape(BANDS, M / BANDS);
        var dst2d = dst.reshape(BANDS, M / BANDS);
        var srcRows = new NDArray[BANDS];
        var dstRows = new NDArray[BANDS];
        for (int i = 0; i < BANDS; i++) { srcRows[i] = src2d[i]; dstRows[i] = dst2d[i]; }

        double t8 = BestMs(() =>
        {
            System.Threading.Tasks.Parallel.For(0, BANDS, i =>
            {
                var ops = new[] { srcRows[i], dstRows[i] };
                using var it = NpyIterRef.MultiNew(2, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
                it.ForEach(F2Kern.SinF64);
            });
        }, 10, 3, 5);
        Check(Math.Abs(dst.GetDouble(M - 777) - Math.Sin(src.GetDouble(M - 777))) < 1e-15, "PAR8");
        Row("PAR8", "iterator sin f64 4M, 8 BANDED iterators (Parallel.For)", t8,
            $"{t1 / t8:F1}x over single");
    }
}

Console.WriteLine(new string('-', 95));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

static unsafe class F2Kern
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SinF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long ss = st[0], so = st[1];
        for (long i = 0; i < count; i++)
        {
            *(double*)po = Math.Sin(*(double*)ps);
            ps += ss; po += so;
        }
    }
}
