#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_bench.cs — THE canonical NumSharp NpyIter benchmark (NumSharp side).
// Companion: npyiter_bench.py (identical ids). Orchestrated by npyiter_sheet.py
// into a single results sheet.
//
// Covers, in one place, every NpyIter aspect probed across POC rounds 1-3:
//   operations x size : 33 families x {scalar,1K,100K,1M}  (the dashboard)
//   construction      : 9 iterator flag configs vs np.nditer construction
//   chunkwidth        : per-chunk dispatch overhead across inner widths
//   pathology         : the known regression canaries (bcast-reduce 54x, etc.)
//   dividends         : NumSharp-only wins (fusion / reuse / parallel banding)
//
// The benchmark is SECTION-ADDRESSABLE via the NPYITER_SECTION env var so the
// orchestrator can run each category in its own short-lived process (crash
// isolation — the full mixed run intermittently AVs under GC pressure). With
// NPYITER_SECTION unset or "all", it runs everything in one process.
//
// Run ONLY with:  dotnet run -c Release - < benchmark/npyiter/npyiter_bench.cs
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
    Console.WriteLine("Run:   dotnet run -c Release - < benchmark/npyiter/npyiter_bench.cs");
    return;
}

string section = (Environment.GetEnvironmentVariable("NPYITER_SECTION") ?? "all").Trim().ToLowerInvariant();
bool Want(string s) => section == "all" || section == s;

int fails = 0;
void Check(bool ok, string what) { if (!ok) { fails++; Console.Error.WriteLine($"  CORRECTNESS FAIL: {what}"); } }

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

// Rows are machine-readable: "id<TAB>milliseconds". The orchestrator parses these.
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");

(int iters, int warm, int rounds) Pick(int n) =>
    n <= 1 ? (200_000, 20_000, 5) :
    n <= 1_000 ? (80_000, 10_000, 5) :
    n <= 100_000 ? (2_500, 400, 4) :
    (120, 30, 3);

(int R, int C) Grid(int n) => n == 1 ? (1, 1) : n == 1_000 ? (25, 40) : n == 100_000 ? (250, 400) : (1_000, 1_000);

var RO1 = new[] { NpyIterPerOpFlags.READONLY };
var RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
var RO_RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
const NPY_ORDER KO = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;
const NpyIterGlobalFlags EXL = NpyIterGlobalFlags.EXTERNAL_LOOP;
const NpyIterGlobalFlags BUFEXL = NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER;
var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };
var f64x3 = new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double };

Console.Error.WriteLine($"[npyiter_bench] section={section}  cores={Environment.ProcessorCount} V256={Vector256.IsHardwareAccelerated}");

var SIZES = new (string tag, int n)[] { ("1", 1), ("1K", 1_000), ("100K", 100_000), ("1M", 1_000_000) };
bool wantOps = Want("elementwise") || Want("reductions") || Want("selection") || Want("copycast") || Want("indexmath") || Want("dtypes") || Want("dividends");

unsafe
{
    // =====================================================================
    // OPERATIONS x SIZE  (sections: elementwise/reductions/selection/
    //                     copycast/indexmath/dtypes/dividends)
    // =====================================================================
    if (wantOps)
    {
        foreach (var (tag, n) in SIZES)
        {
            var (iters, warm, rounds) = Pick(n);
            var (R, C) = Grid(n);

            var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
            var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
            var o = np.empty(new Shape(n), np.float64);

            if (Want("elementwise"))
            {
                var b1 = NDArray.Scalar(3.0, NPTypeCode.Double).reshape(1);
                var a2 = (np.arange(2 * n).astype(np.float64) % 53.0) + 1.0;
                var b2 = (np.arange(2 * n).astype(np.float64) % 17.0) + 1.0;
                var sa = a2["::2"]; var sb = b2["::2"]; var so = np.empty(new Shape(n), np.float64);
                var a32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
                var o64 = np.empty(new Shape(n), np.float64);
                var rev = a["::-1"]; var dstRev = np.empty(new Shape(n), np.float64);
                var af32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
                var add3 = new[] { a, b, o }; var copy2 = new[] { a, o };
                var bc3 = new[] { a, b1, o }; var sadd3 = new[] { sa, sb, so };
                var cast2 = new[] { a32, o64 }; var mix3 = new[] { af32, b, o64 }; var rev2 = new[] { rev, dstRev };

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
            }

            if (Want("reductions"))
            {
                var af32 = (np.arange(n).astype(np.float32) % 977f) + 1f;
                var A = ((np.arange(n).astype(np.float64) % 97.0) + 1.0).reshape(R, C);
                NDArray allFalse = np.arange(n) == -1;
                NDArray earlyHit = np.arange(n) == Math.Min(1000, n - 1);
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
            }

            if (Want("selection"))
            {
                NDArray mask = (np.arange(n) % 2) == 0; var maskB = mask.MakeGeneric<bool>();
                var aMaskDst = a.copy(); var five = NDArray.Scalar(5.0, NPTypeCode.Double);
                NDArray cond = (np.arange(n) % 2) == 0;
                var idx = ((np.arange(n).astype(np.int64) * 2654435761L) % n).astype(np.int32);
                var idxVals = np.arange(n).astype(np.float64);
                var aScatter = a.copy();
                Row($"where@{tag}", BestMs(() => { var _ = np.where(cond, a, b); }, iters, warm, rounds));
                Row($"bread@{tag}", BestMs(() => { var _ = a[maskB]; }, iters, warm, rounds));
                Row($"bassign@{tag}", BestMs(() => aMaskDst[maskB] = five, iters, warm, rounds));
                Row($"cnz@{tag}", BestMs(() => { var _ = np.count_nonzero(a); }, iters, warm, rounds));
                Row($"argw@{tag}", BestMs(() => { var _ = np.argwhere(mask); }, iters, warm, rounds));
                Row($"gather@{tag}", BestMs(() => { var _ = a[idx]; }, iters, warm, rounds));
                Row($"scatter@{tag}", BestMs(() => aScatter[idx] = idxVals, iters, warm, rounds));
            }

            if (Want("copycast"))
            {
                var A = ((np.arange(n).astype(np.float64) % 97.0) + 1.0).reshape(R, C);
                var At = A.T;
                Row($"flatten@{tag}", BestMs(() => { var _ = A.flatten(); }, iters, warm, rounds));
                Row($"astype@{tag}", BestMs(() => { var _ = A.astype(np.float32); }, iters, warm, rounds));
                Row($"ravelT@{tag}", BestMs(() => { var _ = np.ravel(At); }, iters, warm, rounds));
                var ipa = a.copy(); np.add(ipa, b, ipa);
                Row($"inplace@{tag}", BestMs(() => np.add(ipa, b, ipa), iters, warm, rounds));
                var ob = np.empty(new Shape(n), np.bool_); np.less(a, b, ob);
                Row($"lessbool@{tag}", BestMs(() => np.less(a, b, ob), iters, warm, rounds));
            }

            if (Want("indexmath"))
            {
                var flat = ((np.arange(n).astype(np.int64) * 2654435761L) % ((long)R * C)).astype(np.int64);
                var dims = new[] { R, C };
                var coords = np.unravel_index(flat, dims);
                Row($"unravel@{tag}", BestMs(() => { var _ = np.unravel_index(flat, dims); }, iters, warm, rounds));
                var ci = coords[0]; var cj = coords[1]; var packed = new NDArray[] { ci, cj };
                Row($"ravelmi@{tag}", BestMs(() => { var _ = np.ravel_multi_index(packed, dims); }, iters, warm, rounds));
            }

            if (Want("dtypes"))
            {
                var ac = np.arange(n).astype(np.complex128);
                var bc = (np.arange(n).astype(np.float64) % 7.0 + 1.0).astype(np.complex128);
                var oc = np.empty(new Shape(n), np.complex128);
                var ah = (np.arange(n) % 1000).astype(np.float16);
                var bh = (np.arange(n) % 31).astype(np.float16);
                var oh = np.empty(new Shape(n), np.float16);
                var ai8 = (np.arange(n) % 100).astype(np.int8);
                var bi8 = (np.arange(n) % 27).astype(np.int8);
                var oi8 = np.empty(new Shape(n), np.int8);
                np.add(ac, bc, oc); Row($"cplx@{tag}", BestMs(() => np.add(ac, bc, oc), iters, warm, rounds));
                np.add(ah, bh, oh); Row($"f16@{tag}", BestMs(() => np.add(ah, bh, oh), iters, warm, rounds));
                np.add(ai8, bi8, oi8); Check(oi8.GetSByte(5) == (sbyte)((5 % 100) + (5 % 27)), $"i8@{tag}");
                Row($"i8@{tag}", BestMs(() => np.add(ai8, bi8, oi8), iters, warm, rounds));
            }

            if (Want("dividends"))
            {
                var ins = new NDArray[8];
                for (int i = 0; i < 7; i++) ins[i] = (np.arange(n).astype(np.float64) % (7.0 + i)) + 1.0;
                ins[7] = np.empty(new Shape(n), np.float64);
                var flags8 = new NpyIterPerOpFlags[8];
                for (int i = 0; i < 7; i++) flags8[i] = NpyIterPerOpFlags.READONLY;
                flags8[7] = NpyIterPerOpFlags.WRITEONLY;
                Row($"fuse7@{tag}", BestMs(() => { using var it = NpyIterRef.MultiNew(8, ins, EXL, KO, SAFE, flags8); it.ForEach(K.Sum7F64); }, iters, warm, rounds));

                var add3 = new[] { a, b, o };
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
            }

            GC.Collect(); GC.WaitForPendingFinalizers();
        }
    }

    // =====================================================================
    // CONSTRUCTION — iterator build+dispose across flag configs vs np.nditer.
    // Size-invariant (ctor cost is setup); measured at 1K.
    // =====================================================================
    if (Want("construction"))
    {
        var a = np.arange(1000).astype(np.float64);
        var b = np.arange(1000).astype(np.float64) + 1.0;
        var o = np.empty(new Shape(1000), np.float64);
        var a32 = np.arange(1000).astype(np.float32);
        var o64 = np.empty(new Shape(1000), np.float64);
        var g32 = np.arange(1024).astype(np.float64).reshape(32, 32);
        var a4d = np.arange(1024).astype(np.float64).reshape(8, 8, 4, 4);
        var o4d = np.empty(new Shape(8, 8, 4, 4), np.float64);
        var a8d = np.arange(65536).astype(np.float64).reshape(4, 4, 4, 4, 4, 4, 4, 4);
        var o8d = np.empty(new Shape(4, 4, 4, 4, 4, 4, 4, 4), np.float64);
        var back2d = np.arange(64 * 8).astype(np.float64).reshape(64, 8);
        var sview = back2d[":, :4"];
        var sdst = np.empty(new Shape(64, 4), np.float64);

        var ops2 = new[] { a, o }; var ops3 = new[] { a, b, o };
        var opsCast = new[] { a32, o64 }; var ops4d = new[] { a4d, o4d };
        var ops8d = new[] { a8d, o8d }; var opsSv = new[] { sview, sdst };
        var ops8 = new NDArray[8]; for (int i = 0; i < 7; i++) ops8[i] = a; ops8[7] = o;
        var ro8 = new NpyIterPerOpFlags[8]; for (int i = 0; i < 7; i++) ro8[i] = NpyIterPerOpFlags.READONLY; ro8[7] = NpyIterPerOpFlags.WRITEONLY;
        var ufuncFlags = EXL | NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.GROWINNER | NpyIterGlobalFlags.DELAY_BUFALLOC | NpyIterGlobalFlags.COPY_IF_OVERLAP | NpyIterGlobalFlags.ZEROSIZE_OK;

        Row("ctor.1op", BestMs(() => { using var it = NpyIterRef.New(a); }, 400_000, 50_000, 5));
        Row("ctor.3op_exl", BestMs(() => { using var it = NpyIterRef.MultiNew(3, ops3, EXL, KO, SAFE, RO_RO_WO); }, 400_000, 50_000, 5));
        Row("ctor.ufunc", BestMs(() => { using var it = NpyIterRef.MultiNew(3, ops3, ufuncFlags, KO, SAFE, RO_RO_WO); }, 200_000, 25_000, 5));
        Row("ctor.bufcast", BestMs(() => { using var it = NpyIterRef.MultiNew(2, opsCast, BUFEXL, KO, SAFE, RO_WO, f64x2); }, 100_000, 12_000, 5));
        Row("ctor.multiindex", BestMs(() => { using var it = NpyIterRef.New(g32, NpyIterGlobalFlags.MULTI_INDEX); }, 400_000, 50_000, 5));
        Row("ctor.8op", BestMs(() => { using var it = NpyIterRef.MultiNew(8, ops8, EXL, KO, SAFE, ro8); }, 200_000, 25_000, 5));
        Row("ctor.4d", BestMs(() => { using var it = NpyIterRef.MultiNew(2, ops4d, EXL, KO, SAFE, RO_WO); }, 200_000, 25_000, 5));
        Row("ctor.8d", BestMs(() => { using var it = NpyIterRef.MultiNew(2, ops8d, EXL, KO, SAFE, RO_WO); }, 200_000, 25_000, 5));
        Row("ctor.strided2d", BestMs(() => { using var it = NpyIterRef.MultiNew(2, opsSv, EXL, KO, SAFE, RO_WO); }, 200_000, 25_000, 5));
        GC.Collect(); GC.WaitForPendingFinalizers();
    }

    // =====================================================================
    // CHUNKWIDTH — per-chunk dispatch overhead. Total fixed 2M f64, strided
    // rows of inner width w => 2M/w chunks. Honest comparator = real strided
    // ufunc copy (np.positive), not np.copyto's raw walker.
    // =====================================================================
    if (Want("chunkwidth"))
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
                using var it = NpyIterRef.MultiNew(2, ops, EXL, KO, SAFE, RO_WO);
                it.ForEach(K.CopyF64);
            }, w <= 16 ? 12 : 25, 5, 7);
            Check(dst.GetDouble(rows - 1, w - 1) == sv.GetDouble(rows - 1, w - 1), $"cw{w}");
            Row($"cw.{w}", t);
        }
        GC.Collect(); GC.WaitForPendingFinalizers();
    }

    // =====================================================================
    // PATHOLOGY — the regression canaries (known losses / taxes worth tracking)
    // =====================================================================
    if (Want("pathology"))
    {
        // bcast-reduce: sum over a broadcast view (the 54x general-path loss)
        {
            var a8k = (np.arange(8192).astype(np.float64) % 97.0) + 1.0;
            var bc = np.broadcast_to(a8k, new Shape(1024, 8192));
            double expect = 1024.0 * (double)np.sum(a8k);
            Check(Math.Abs((double)np.sum(bc) - expect) / expect < 1e-9, "path.bcast_reduce");
            Row("path.bcast_reduce", BestMs(() => { var _ = np.sum(bc); }, 25, 8, 7));
        }
        // ALLOCATE out: NumSharp zeros (np.zeros) vs NumPy empty
        {
            const int M = 4_194_304;
            var a = np.arange(M).astype(np.float64);
            var b = np.arange(M).astype(np.float64) + 1.0;
            var f64x3b = new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double };
            var allocFlags = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.ALLOCATE };
            NDArray sink = null;
            Row("path.allocate", BestMs(() =>
            {
                var ops = new NDArray[] { a, b, null };
                using var it = NpyIterRef.MultiNew(3, ops, EXL, KO, SAFE, allocFlags, f64x3b);
                it.ForEach(K.AddF64);
                sink = ops[2];
            }, 10, 4, 7));
            Check(sink.GetDouble(777) == a.GetDouble(777) + b.GetDouble(777), "path.allocate");
        }
        // overlap forced-copy: shifted alias forces COPY_IF_OVERLAP temp+writeback
        {
            const int M = 4_194_304;
            var x = (np.arange(M).astype(np.float64) % 53.0) + 1.0;
            var xs = x[":-1"]; var xd = x["1:"];
            np.add(xs, xs, xd);
            Row("path.overlap_copy", BestMs(() => np.add(xs, xs, xd), 12, 4, 7));
        }
        // F-order out: iterator add C+C -> F-order out (order resolution)
        {
            const int nn = 1448;
            var aC = ((np.arange(nn * nn).astype(np.float64) % 97.0) + 1.0).reshape(nn, nn);
            var bC = ((np.arange(nn * nn).astype(np.float64) % 31.0) + 2.0).reshape(nn, nn);
            var oF = np.empty(new Shape(nn, nn), np.float64).T;
            var opsX = new[] { aC, bC, oF };
            { using var it = NpyIterRef.MultiNew(3, opsX, EXL, KO, SAFE, RO_RO_WO); it.ForEach(K.AddF64); }
            Check(Math.Abs(oF.GetDouble(5, 7) - (aC.GetDouble(5, 7) + bC.GetDouble(5, 7))) < 1e-9, "path.forder_out");
            Row("path.forder_out", BestMs(() =>
            {
                using var it = NpyIterRef.MultiNew(3, opsX, EXL, KO, SAFE, RO_RO_WO);
                it.ForEach(K.AddF64);
            }, 12, 4, 7));
        }
        // 0-d scalar ufunc (production)
        {
            var s1 = NDArray.Scalar(2.5, NPTypeCode.Double);
            var s2 = NDArray.Scalar(1.5, NPTypeCode.Double);
            var s3 = NDArray.Scalar(0.0, NPTypeCode.Double);
            np.add(s1, s2, s3);
            Check(s3.GetDouble(0) == 4.0, "path.zerodim");
            Row("path.zerodim", BestMs(() => np.add(s1, s2, s3), 200_000, 25_000, 5));
        }
        GC.Collect(); GC.WaitForPendingFinalizers();
    }
}

Console.Error.WriteLine(fails == 0 ? "[ok] all correctness checks pass" : $"[WARN] {fails} correctness failures");
Console.Error.WriteLine($"[section-done] {section}");

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
