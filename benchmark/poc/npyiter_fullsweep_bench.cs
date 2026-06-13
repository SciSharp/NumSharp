#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_fullsweep_bench.cs — EVERY distinct NpyIter operation family from
// rounds 1-3, now swept across the four size tiers scalar(1)/1K/100K/1M.
// Companion: npyiter_fullsweep_bench.py (identical ids).
//
// The earlier rounds used SIZE as the id axis (H8/H64/H512/H4096/H32K/H256K/H2M
// are one op at seven sizes; T2.4..T2.1024 / S.4..S.1024 are strided copy at five
// widths; O1..O4/M1 are small-N add variants). Collapsing those size-variants,
// the distinct families are the 33 below + 3 dividends. Each runs at all 4 tiers.
//
//   ELEMENTWISE (raw NpyIter + matched kernel — isolates iterator cost):
//     add sqrt copy sadd bcast frev castbuf mixbuf
//   REDUCTIONS/SCANS (production np.*):
//     psum sumax0 sumax1 sumdt amin cumsum anyff anyeh
//   SELECTION/INDEXING (production np.*):
//     where bread bassign cnz argw gather scatter
//   COPY/CAST/LAYOUT (production np.*):
//     flatten astype ravelT inplace lessbool
//   INDEX MATH (production np.*):
//     unravel ravelmi
//   KERNEL-BOUND DTYPES (production np.add):
//     cplx f16 i8
//   DIVIDENDS (raw NpyIter — NumPy has no equivalent, reported separately):
//     fuse7 reuse par8
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_fullsweep_bench.cs
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Threading.Tasks;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgScript = Attribute.GetCustomAttribute(typeof(K).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if ((dbgScript?.IsJITOptimizerDisabled ?? false) || (dbgCore?.IsJITOptimizerDisabled ?? false))
{
    Console.WriteLine("FATAL: Debug-JITted assemblies — numbers would be INVALID.");
    Console.WriteLine("Run:   dotnet run -c Release - < benchmark/poc/npyiter_fullsweep_bench.cs");
    return;
}

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

double BestMs(Action body, int iters, int warm, int rounds)
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

void Row(string id, double ms)
{
    string val = ms >= 1.0 ? $"{ms,10:F3} ms" : ms >= 0.001 ? $"{ms * 1000,10:F2} us" : $"{ms * 1e6,10:F1} ns";
    Console.WriteLine($"{id,-14} {val}");
}

(int iters, int warm, int rounds) Pick(int n) =>
    n <= 1 ? (200_000, 20_000, 5) :
    n <= 1_000 ? (80_000, 10_000, 5) :
    n <= 100_000 ? (2_500, 400, 4) :
    (120, 30, 3);

(int R, int C) Grid(int n) => n == 1 ? (1, 1) : n == 1_000 ? (25, 40) : n == 100_000 ? (250, 400) : (1_000, 1_000);

var RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
var RO_RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
const NPY_ORDER KO = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;
const NpyIterGlobalFlags EXL = NpyIterGlobalFlags.EXTERNAL_LOOP;
const NpyIterGlobalFlags BUFEXL = NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER;
var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };
var f64x3 = new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double };

Console.WriteLine($"NumSharp NpyIter FULL sweep — {Environment.ProcessorCount} cores, V256={Vector256.IsHardwareAccelerated}");
Console.WriteLine($"{"id",-14} {"per call",10}");
Console.WriteLine(new string('-', 28));

var SIZES = new (string tag, int n)[] { ("1", 1), ("1K", 1_000), ("100K", 100_000), ("1M", 1_000_000) };

unsafe
{
    foreach (var (tag, n) in SIZES)
    {
        var (iters, warm, rounds) = Pick(n);
        var (R, C) = Grid(n);

        // ---- fixtures -------------------------------------------------------
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(n), np.float64);
        var b1 = NDArray.Scalar(3.0, NPTypeCode.Double).reshape(1);
        var a2 = (np.arange(2 * n).astype(np.float64) % 53.0) + 1.0;
        var b2 = (np.arange(2 * n).astype(np.float64) % 17.0) + 1.0;
        var sa = a2["::2"]; var sb = b2["::2"]; var so = np.empty(new Shape(n), np.float64);
        var a32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
        var o64 = np.empty(new Shape(n), np.float64);
        var rev = a["::-1"]; var dstRev = np.empty(new Shape(n), np.float64);
        var af32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
        var A = ((np.arange(n).astype(np.float64) % 97.0) + 1.0).reshape(R, C);
        var At = A.T;
        NDArray mask = (np.arange(n) % 2) == 0; var maskB = mask.MakeGeneric<bool>();
        var aMaskDst = a.copy(); var five = NDArray.Scalar(5.0, NPTypeCode.Double);
        NDArray cond = (np.arange(n) % 2) == 0;
        NDArray allFalse = np.arange(n) == -1;
        NDArray earlyHit = np.arange(n) == Math.Min(1000, n - 1);
        var idx = ((np.arange(n).astype(np.int64) * 2654435761L) % n).astype(np.int32);
        var idxVals = np.arange(n).astype(np.float64);
        var aScatter = a.copy();
        var flat = ((np.arange(n).astype(np.int64) * 2654435761L) % ((long)R * C)).astype(np.int64);
        var dims = new[] { R, C };
        var ac = np.arange(n).astype(np.complex128);
        var bc = (np.arange(n).astype(np.float64) % 7.0 + 1.0).astype(np.complex128);
        var oc = np.empty(new Shape(n), np.complex128);
        var ah = (np.arange(n) % 1000).astype(np.float16);
        var bh = (np.arange(n) % 31).astype(np.float16);
        var oh = np.empty(new Shape(n), np.float16);
        var ai8 = (np.arange(n) % 100).astype(np.int8);
        var bi8 = (np.arange(n) % 27).astype(np.int8);
        var oi8 = np.empty(new Shape(n), np.int8);

        var add3 = new[] { a, b, o };
        var copy2 = new[] { a, o };
        var bc3 = new[] { a, b1, o };
        var sadd3 = new[] { sa, sb, so };
        var cast2 = new[] { a32, o64 };
        var mix3 = new[] { af32, b, o64 };
        var rev2 = new[] { rev, dstRev };

        // ---- ELEMENTWISE (raw iterator + matched kernel) --------------------
        np.add(a, b, o); Check(o.GetDouble(n - 1) == a.GetDouble(n - 1) + b.GetDouble(n - 1), $"add@{tag}");
        Row($"add@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(3, add3, EXL, KO, SAFE, RO_RO_WO); it.ForEach(K.AddF64); }, iters, warm, rounds));
        Row($"sqrt@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(2, copy2, EXL, KO, SAFE, RO_WO); it.ForEach(K.SqrtF64); }, iters, warm, rounds));
        Row($"copy@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(2, copy2, EXL, KO, SAFE, RO_WO); it.ForEach(K.CopyF64); }, iters, warm, rounds));
        np.add(sa, sb, so); Check(so.GetDouble(n - 1) == sa.GetDouble(n - 1) + sb.GetDouble(n - 1), $"sadd@{tag}");
        Row($"sadd@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(3, sadd3, EXL, KO, SAFE, RO_RO_WO); it.ForEach(K.AddF64); }, iters, warm, rounds));
        Row($"bcast@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(3, bc3, EXL, KO, SAFE, RO_RO_WO); it.ForEach(K.AddF64); }, iters, warm, rounds));
        { using var itw = NpyIterRef.MultiNew(2, rev2, EXL, KO, SAFE, RO_WO); itw.ForEach(K.CopyF64); }
        Check(dstRev.GetDouble(0) == a.GetDouble(n - 1), $"frev@{tag}");
        Row($"frev@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(2, rev2, EXL, KO, SAFE, RO_WO); it.ForEach(K.CopyF64); }, iters, warm, rounds));
        Row($"castbuf@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(2, cast2, BUFEXL, KO, SAFE, RO_WO, f64x2); it.ForEach(K.CopyF64); }, iters, warm, rounds));
        Row($"mixbuf@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(3, mix3, BUFEXL, KO, SAFE, RO_RO_WO, f64x3); it.ForEach(K.AddF64); }, iters, warm, rounds));

        // ---- REDUCTIONS / SCANS (production) --------------------------------
        Row($"psum@{tag}", BestMs(() => { var _ = np.sum(a); }, iters, warm, rounds));
        Row($"sumax0@{tag}", BestMs(() => { var _ = np.sum(A, 0); }, iters, warm, rounds));
        Row($"sumax1@{tag}", BestMs(() => { var _ = np.sum(A, 1); }, iters, warm, rounds));
        Row($"sumdt@{tag}", BestMs(() => { var _ = np.sum(af32, NPTypeCode.Double); }, iters, warm, rounds));
        Row($"amin@{tag}", BestMs(() => { var _ = np.amin(A, 1); }, iters, warm, rounds));
        Row($"cumsum@{tag}", BestMs(() => { var _ = np.cumsum(a); }, iters, warm, rounds));
        Check(!(bool)np.any(allFalse), $"anyff@{tag}");
        Row($"anyff@{tag}", BestMs(() => { var _ = np.any(allFalse); }, iters, warm, rounds));
        Check((bool)np.any(earlyHit), $"anyeh@{tag}");
        Row($"anyeh@{tag}", BestMs(() => { var _ = np.any(earlyHit); }, iters, warm, rounds));

        // ---- SELECTION / INDEXING (production) ------------------------------
        Row($"where@{tag}", BestMs(() => { var _ = np.where(cond, a, b); }, iters, warm, rounds));
        Row($"bread@{tag}", BestMs(() => { var _ = a[maskB]; }, iters, warm, rounds));
        Row($"bassign@{tag}", BestMs(() => aMaskDst[maskB] = five, iters, warm, rounds));
        Row($"cnz@{tag}", BestMs(() => { var _ = np.count_nonzero(a); }, iters, warm, rounds));
        Row($"argw@{tag}", BestMs(() => { var _ = np.argwhere(mask); }, iters, warm, rounds));
        Row($"gather@{tag}", BestMs(() => { var _ = a[idx]; }, iters, warm, rounds));
        Row($"scatter@{tag}", BestMs(() => aScatter[idx] = idxVals, iters, warm, rounds));

        // ---- COPY / CAST / LAYOUT (production) ------------------------------
        Row($"flatten@{tag}", BestMs(() => { var _ = A.flatten(); }, iters, warm, rounds));
        Row($"astype@{tag}", BestMs(() => { var _ = A.astype(np.float32); }, iters, warm, rounds));
        Row($"ravelT@{tag}", BestMs(() => { var _ = np.ravel(At); }, iters, warm, rounds));
        var ipa = a.copy(); np.add(ipa, b, ipa);
        Row($"inplace@{tag}", BestMs(() => np.add(ipa, b, ipa), iters, warm, rounds));
        var ob = np.empty(new Shape(n), np.bool_); np.less(a, b, ob);
        Row($"lessbool@{tag}", BestMs(() => np.less(a, b, ob), iters, warm, rounds));

        // ---- INDEX MATH (production) ----------------------------------------
        var coords = np.unravel_index(flat, dims);
        Row($"unravel@{tag}", BestMs(() => { var _ = np.unravel_index(flat, dims); }, iters, warm, rounds));
        var ci = coords[0]; var cj = coords[1]; var packed = new NDArray[] { ci, cj };
        Row($"ravelmi@{tag}", BestMs(() => { var _ = np.ravel_multi_index(packed, dims); }, iters, warm, rounds));

        // ---- KERNEL-BOUND DTYPES (production np.add) ------------------------
        np.add(ac, bc, oc); Row($"cplx@{tag}", BestMs(() => np.add(ac, bc, oc), iters, warm, rounds));
        np.add(ah, bh, oh); Row($"f16@{tag}", BestMs(() => np.add(ah, bh, oh), iters, warm, rounds));
        np.add(ai8, bi8, oi8); Row($"i8@{tag}", BestMs(() => np.add(ai8, bi8, oi8), iters, warm, rounds));

        // ---- DIVIDENDS (raw NpyIter; NumPy has no equivalent) ---------------
        // fuse7: one-pass sum of 7 arrays in a single 8-op iterator
        var ins = new NDArray[8];
        for (int i = 0; i < 7; i++) ins[i] = (np.arange(n).astype(np.float64) % (7.0 + i)) + 1.0;
        ins[7] = np.empty(new Shape(n), np.float64);
        var flags8 = new NpyIterPerOpFlags[8];
        for (int i = 0; i < 7; i++) flags8[i] = NpyIterPerOpFlags.READONLY;
        flags8[7] = NpyIterPerOpFlags.WRITEONLY;
        Row($"fuse7@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(8, ins, EXL, KO, SAFE, flags8); it.ForEach(K.Sum7F64); }, iters, warm, rounds));

        // reuse: Reset + ForEach only (no ctor) — the floor NumPy can't reach from Python
        {
            var ru = NpyIterRef.MultiNew(3, add3, EXL, KO, SAFE, RO_RO_WO);
            for (int i = 0; i < warm; i++) { ru.Reset(); ru.ForEach(K.AddF64); }
            double best = double.MaxValue;
            for (int r = 0; r < rounds; r++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iters; i++) { ru.Reset(); ru.ForEach(K.AddF64); }
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds / iters);
            }
            ru.Dispose();
            Row($"reuse@{tag}", best);
        }

        // par8: 8 banded iterators, sin f64, Parallel.For (needs n>=8)
        if (n >= 8)
        {
            var src = (np.arange(n).astype(np.float64) % 6.283185) - 3.1415926;
            var dst = np.empty(new Shape(n), np.float64);
            var src2d = src.reshape(8, n / 8); var dst2d = dst.reshape(8, n / 8);
            var srcRows = new NDArray[8]; var dstRows = new NDArray[8];
            for (int i = 0; i < 8; i++) { srcRows[i] = src2d[i]; dstRows[i] = dst2d[i]; }
            Row($"par8@{tag}", BestMs(() =>
            {
                Parallel.For(0, 8, i =>
                {
                    var ops = new[] { srcRows[i], dstRows[i] };
                    using var it = NpyIterRef.MultiNew(2, ops, EXL, KO, SAFE, RO_WO);
                    it.ForEach(K.SinF64);
                });
            }, Math.Max(10, iters / 20), 4, rounds));
        }

        GC.Collect(); GC.WaitForPendingFinalizers();
    }
}

Console.WriteLine(new string('-', 28));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

static unsafe class K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void CopyF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1];
        if (ss == 8 && so == 8) { Buffer.MemoryCopy(ps, po, count * 8, count * 8); return; }
        for (long i = 0; i < count; i++) { *(double*)po = *(double*)ps; ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SqrtF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1]; long i = 0;
        if (ss == 8 && so == 8)
            for (; i + 4 <= count; i += 4) { Vector256.Store(Vector256.Sqrt(Vector256.Load((double*)ps)), (double*)po); ps += 32; po += 32; }
        for (; i < count; i++) { *(double*)po = Math.Sqrt(*(double*)ps); ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddF64(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0]; byte* pb = (byte*)dp[1]; byte* po = (byte*)dp[2];
        long sa = st[0], sb = st[1], so = st[2]; long i = 0;
        if (so == 8)
        {
            if (sa == 8 && sb == 8)
                for (; i + 8 <= count; i += 8)
                {
                    Vector256.Store(Vector256.Load((double*)pa) + Vector256.Load((double*)pb), (double*)po);
                    Vector256.Store(Vector256.Load((double*)(pa + 32)) + Vector256.Load((double*)(pb + 32)), (double*)(po + 32));
                    pa += 64; pb += 64; po += 64;
                }
            else if (sa == 8 && sb == 0)
            {
                var vb = Vector256.Create(*(double*)pb);
                for (; i + 4 <= count; i += 4) { Vector256.Store(Vector256.Load((double*)pa) + vb, (double*)po); pa += 32; po += 32; }
            }
        }
        for (; i < count; i++) { *(double*)po = *(double*)pa + *(double*)pb; pa += sa; pb += sb; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SinF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0]; byte* po = (byte*)dp[1]; long ss = st[0], so = st[1];
        for (long i = 0; i < count; i++) { *(double*)po = Math.Sin(*(double*)ps); ps += ss; po += so; }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Sum7F64(void** dp, long* st, long count, void* aux)
    {
        bool contig = true;
        for (int op = 0; op < 8; op++) contig &= st[op] == 8;
        long i = 0;
        if (contig)
        {
            double* p0 = (double*)dp[0]; double* p1 = (double*)dp[1]; double* p2 = (double*)dp[2];
            double* p3 = (double*)dp[3]; double* p4 = (double*)dp[4]; double* p5 = (double*)dp[5];
            double* p6 = (double*)dp[6]; double* po = (double*)dp[7];
            for (; i + 4 <= count; i += 4)
            {
                var v = Vector256.Load(p0 + i) + Vector256.Load(p1 + i) + Vector256.Load(p2 + i)
                      + Vector256.Load(p3 + i) + Vector256.Load(p4 + i) + Vector256.Load(p5 + i) + Vector256.Load(p6 + i);
                Vector256.Store(v, po + i);
            }
            for (; i < count; i++) po[i] = p0[i] + p1[i] + p2[i] + p3[i] + p4[i] + p5[i] + p6[i];
            return;
        }
        for (; i < count; i++)
        {
            double s = 0;
            for (int op = 0; op < 7; op++) s += *(double*)((byte*)dp[op] + i * st[op]);
            *(double*)((byte*)dp[7] + i * st[7]) = s;
        }
    }
}
