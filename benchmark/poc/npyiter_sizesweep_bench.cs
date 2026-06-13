#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_sizesweep_bench.cs — the SAME six NpyIter operation families measured
// across four element-count tiers: scalar (1), 1K, 100K, 1M. Companion:
// npyiter_sizesweep_bench.py (NumPy side, identical ids).
//
// Mirrors the official benchmark report's per-size-tier layout, but for the
// iterator core. Each kernel is matched to NumPy's loop family so the measured
// time is dominated by what the ITERATOR pays (construction, broadcast/order
// resolution, per-chunk dispatch, iternext, reduction seed) at each size:
//   add   — contiguous binary, V256        vs np.add(a, b, out=o)
//   sqrt  — contiguous unary, V256 Sqrt     vs np.sqrt(a, out=o)
//   sum   — full reduction, 4-acc V256      vs np.sum(a)
//   copy  — contiguous copy, memcpy chunk   vs np.positive(a, out=o)  [real ufunc nditer]
//   sadd  — strided binary a[::2]+b[::2]     vs np.add(a2[::2], b2[::2], out=o)
//   bcast — stride-0 binary a + b1           vs np.add(a, b1, out=o)
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_sizesweep_bench.cs
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
    Console.WriteLine("Run:   dotnet run -c Release - < benchmark/poc/npyiter_sizesweep_bench.cs");
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

void Row(string id, string label, double ms)
{
    string val = ms >= 1.0 ? $"{ms,10:F3} ms" : ms >= 0.001 ? $"{ms * 1000,10:F2} us" : $"{ms * 1e6,10:F1} ns";
    Console.WriteLine($"{id,-12} {label,-46} {val}");
}

// iters/warm tuned per size so each (aspect,size) measures for a comparable wall time
(int iters, int warm) Pick(int n) =>
    n <= 1 ? (400_000, 40_000) :
    n <= 1_000 ? (200_000, 25_000) :
    n <= 100_000 ? (4_000, 600) :
    (400, 80);

var RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
var RO_RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
const NPY_ORDER K = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;
const NpyIterGlobalFlags EXL = NpyIterGlobalFlags.EXTERNAL_LOOP;

Console.WriteLine($"NumSharp NpyIter size-sweep — {Environment.ProcessorCount} cores, V256={Vector256.IsHardwareAccelerated}");
Console.WriteLine($"{"id",-12} {"aspect",-46} {"per call",13}");
Console.WriteLine(new string('-', 86));

var SIZES = new (string tag, int n)[] { ("1", 1), ("1K", 1_000), ("100K", 100_000), ("1M", 1_000_000) };

unsafe
{
    foreach (var (tag, n) in SIZES)
    {
        var (iters, warm) = Pick(n);
        int rounds = n >= 100_000 ? 7 : 5;

        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(n), np.float64);
        var b1 = NDArray.Scalar(3.0, NPTypeCode.Double).reshape(1);   // shape (1) -> broadcasts stride-0
        var a2 = (np.arange(2 * n).astype(np.float64) % 53.0) + 1.0;
        var b2 = (np.arange(2 * n).astype(np.float64) % 17.0) + 1.0;
        var sa = a2["::2"];
        var sb = b2["::2"];
        var so = np.empty(new Shape(n), np.float64);

        var add3 = new[] { a, b, o };
        var copy2 = new[] { a, o };
        var bc3 = new[] { a, b1, o };
        var sadd3 = new[] { sa, sb, so };

        // add — contiguous binary V256
        np.add(a, b, o);
        Check(o.GetDouble(n - 1) == a.GetDouble(n - 1) + b.GetDouble(n - 1), $"add@{tag}");
        Row($"add@{tag}", $"binary add contig f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, add3, EXL, K, SAFE, RO_RO_WO); it.ForEach(Kern.AddF64); }, iters, warm, rounds));

        // sqrt — contiguous unary V256 Sqrt
        Row($"sqrt@{tag}", $"unary sqrt contig f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, copy2, EXL, K, SAFE, RO_WO); it.ForEach(Kern.SqrtF64); }, iters, warm, rounds));
        Check(Math.Abs(o.GetDouble(n - 1) - Math.Sqrt(a.GetDouble(n - 1))) < 1e-9, $"sqrt@{tag}");

        // sum — full reduction
        double sum = 0;
        Row($"sum@{tag}", $"reduce sum contig f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.New(a, EXL); sum = it.ExecuteReducing<SumLoop, double>(default, 0.0); }, iters, warm, rounds));
        Check(Math.Abs(sum - (double)np.sum(a)) / Math.Max(1.0, Math.Abs((double)np.sum(a))) < 1e-9, $"sum@{tag}");

        // copy — contiguous copy (real ufunc nditer on NumPy side = np.positive)
        Row($"copy@{tag}", $"unary copy contig f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.MultiNew(2, copy2, EXL, K, SAFE, RO_WO); it.ForEach(Kern.CopyF64); }, iters, warm, rounds));
        Check(o.GetDouble(n - 1) == a.GetDouble(n - 1), $"copy@{tag}");

        // sadd — strided binary a[::2] + b[::2]
        np.add(sa, sb, so);
        Check(so.GetDouble(n - 1) == sa.GetDouble(n - 1) + sb.GetDouble(n - 1), $"sadd@{tag}");
        Row($"sadd@{tag}", $"strided add a[::2]+b[::2] f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, sadd3, EXL, K, SAFE, RO_RO_WO); it.ForEach(Kern.AddF64); }, iters, warm, rounds));

        // bcast — stride-0 binary a + b1
        Row($"bcast@{tag}", $"broadcast add a+b1(1) f64 N={n}",
            BestMs(() => { using var it = NpyIterRef.MultiNew(3, bc3, EXL, K, SAFE, RO_RO_WO); it.ForEach(Kern.AddF64); }, iters, warm, rounds));
        Check(o.GetDouble(n - 1) == a.GetDouble(n - 1) + 3.0, $"bcast@{tag}");

        GC.Collect(); GC.WaitForPendingFinalizers();
    }
}

Console.WriteLine(new string('-', 86));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

static unsafe class Kern
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long ss = st[0], so = st[1];
        if (ss == 8 && so == 8) { Buffer.MemoryCopy(ps, po, count * 8, count * 8); return; }
        for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SqrtF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long ss = st[0], so = st[1];
        long i = 0;
        if (ss == 8 && so == 8)
        {
            for (; i + 4 <= count; i += 4)
            {
                Vector256.Store(Vector256.Sqrt(Vector256.Load((double*)ps)), (double*)po);
                ps += 32; po += 32;
            }
        }
        for (; i < count; i++) { *(double*)po = Math.Sqrt(*(double*)ps); ps += ss; po += so; }
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
