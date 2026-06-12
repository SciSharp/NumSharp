#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// npyiter_frontier_bench.cs — adversarial follow-up to npyiter_core_bench:
// probe the suspected NOT-winning frontier of NpyIter, plus balancing
// win-candidates. Companion: npyiter_frontier_bench.py (same ids).
//
//   R  axis reductions THROUGH the iterator (op_axes + REDUCE_OK) — the Wave-5
//      gap territory (production axis sums bypass NpyIter today)
//   A  ALLOCATE outputs — NumSharp zeros them (np.zeros), NumPy allocates EMPTY
//   W  where= masked execution at degenerate mask-run lengths (all-true /
//      alternating run=1 / blocky run=64)
//   B  buffered cast with a STRIDED source vs NumPy's one-pass cast transfer
//   X  forced-layout output (C+C -> F-order out) and reversed-source copy
//   O  0-d scalar ufunc calls (production)
//   P  production np.copyto at the tiny-chunk frontier
//   Y  the architecture dividend: 7-input single-pass sum vs chained 2-op passes
//   Z  kernel-bound dtype frontier (complex128 / float16 / int8) — context rows,
//      these measure KERNELS riding the iterator, not the iterator itself
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/npyiter_frontier_bench.cs
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgScript = Attribute.GetCustomAttribute(typeof(FKern).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if ((dbgScript?.IsJITOptimizerDisabled ?? false) || (dbgCore?.IsJITOptimizerDisabled ?? false))
{
    Console.WriteLine("FATAL: Debug-JITted assemblies — rerun: dotnet run -c Release - < benchmark/poc/npyiter_frontier_bench.cs");
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
var RO_RW = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READWRITE };
var RO_RO_WO = new[] { NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.WRITEONLY };
const NPY_ORDER K = NPY_ORDER.NPY_KEEPORDER;
const NPY_CASTING SAFE = NPY_CASTING.NPY_SAFE_CASTING;

Console.WriteLine($"NumSharp NpyIter frontier bench — V256={Vector256.IsHardwareAccelerated}");
Console.WriteLine($"{"id",-6} {"aspect",-56} {"per call",13}");
Console.WriteLine(new string('-', 95));

unsafe
{
    // =========================================================================
    // R — axis reductions through the iterator (op_axes + REDUCE_OK)
    // =========================================================================
    {
        var A = ((np.arange(4_000_000).astype(np.float64) % 97.0) + 1.0).reshape(2000, 2000);
        var outR = np.zeros(new Shape(2000), np.float64);
        var zer = np.zeros(new Shape(2000), np.float64);

        // production context (bypasses NpyIter today — Direct axis kernels)
        Row("R0a", "production np.sum(A, axis=0) f64 (2000,2000)",
            BestMs(() => np.sum(A, 0), 50, 12, 7));
        Row("R0b", "production np.sum(A, axis=1) f64 (2000,2000)",
            BestMs(() => np.sum(A, 1), 50, 12, 7));

        // R1: axis-0 sum THROUGH the iterator: iter shape (2000,2000),
        //     out mapped [-1, 0] => outer axis reduces, inner accumulates rows
        var opsR = new[] { A, outR };
        var axesR1 = new[] { new[] { 0, 1 }, new[] { -1, 0 } };
        double t = BestMs(() =>
        {
            NpyIter.Copy(outR, zer);   // reset accumulator
            using var it = NpyIterRef.AdvancedNew(2, opsR,
                NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.REDUCE_OK,
                K, SAFE, RO_RW, null, 2, axesR1);
            it.ForEach(FKern.AddAccumF64);
        }, 50, 12, 7);
        var expect0 = np.sum(A, 0);
        Check(Math.Abs(outR.GetDouble(777) - expect0.GetDouble(777)) < 1e-6, "R1 axis-0 sum");
        Row("R1", "iterator axis-0 sum via op_axes+REDUCE (2000 chunks)", t);

        // R2: axis-1 sum: out mapped [0, -1] => inner axis reduces into a slot
        var axesR2 = new[] { new[] { 0, 1 }, new[] { 0, -1 } };
        t = BestMs(() =>
        {
            NpyIter.Copy(outR, zer);
            using var it = NpyIterRef.AdvancedNew(2, opsR,
                NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.REDUCE_OK,
                K, SAFE, RO_RW, null, 2, axesR2);
            it.ForEach(FKern.SumIntoF64);
        }, 50, 12, 7);
        var expect1 = np.sum(A, 1);
        Check(Math.Abs(outR.GetDouble(777) - expect1.GetDouble(777)) < 1e-6, "R2 axis-1 sum");
        Row("R2", "iterator axis-1 sum via op_axes+REDUCE (2000 chunks)", t);

        // R3: same axis-0 reduction but BUFFERED — the legacy reduce double-loop.
        // SKIPPED: reproducibly CRASHES with AccessViolationException (uncatchable).
        // Root cause: ForEach on a BUFFERED+REDUCE iterator takes GetIterNext(),
        // which has no BUFFER+REDUCE branch and falls through to ExternalLoopNext
        // — the state then walks BUFFER pointers with SOURCE-array strides while
        // GetInnerLoopSizePtr hands the kernel BufIterEnd as the count. The only
        // safe driver for this config is BufferedReduce<TKernel>/Iternext().
        Console.WriteLine("R3     iterator axis-0 sum BUFFERED — SKIPPED: AccessViolation (ForEach x BUFFERED+REDUCE driver bug, see comment)");
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // A — ALLOCATE output: NumSharp zeros it, NumPy allocates EMPTY
    // =========================================================================
    {
        const int M = 4_194_304;
        var a = np.arange(M).astype(np.float64);
        var b = np.arange(M).astype(np.float64) + 1.0;
        var o = np.empty(new Shape(M), np.float64);
        var opsOut = new[] { a, b, o };
        var f64x3 = new[] { NPTypeCode.Double, NPTypeCode.Double, NPTypeCode.Double };
        var allocFlags = new[]
        {
            NpyIterPerOpFlags.READONLY, NpyIterPerOpFlags.READONLY,
            NpyIterPerOpFlags.WRITEONLY | NpyIterPerOpFlags.ALLOCATE,
        };

        double tOut = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(3, opsOut, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
            it.ForEach(FKern.AddF64);
        }, 25, 8, 7);
        Row("A3", "anchor: iterator add f64 4M, out= provided", tOut);

        NDArray sink = null;
        double tAlloc = BestMs(() =>
        {
            var ops = new NDArray[] { a, b, null };
            using var it = NpyIterRef.MultiNew(3, ops, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, allocFlags, f64x3);
            it.ForEach(FKern.AddF64);
            sink = ops[2];
        }, 10, 4, 7);
        Check(sink.GetDouble(777) == a.GetDouble(777) + b.GetDouble(777), "A1 ALLOCATE add");
        Row("A1", "iterator add f64 4M, out=null + ALLOCATE (np.zeros!)", tAlloc,
            $"+{(tAlloc - tOut):F2} ms vs out=");

        double tProd = BestMs(() => { var _ = np.add(a, b); }, 10, 4, 7);
        Row("A2", "production np.add(a,b) allocating, f64 4M", tProd);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // W — where= masked execution (production np.add rides ARRAYMASK)
    // =========================================================================
    {
        const int M = 4_194_304;
        var a = (np.arange(M).astype(np.float32) % 977f) + 1f;
        var b = (np.arange(M).astype(np.float32) % 31f) + 2f;
        var o = np.zeros(new Shape(M), np.float32) - 1f;
        var idx = np.arange(M);
        NDArray mAll = idx >= 0;                 // all true
        NDArray mAlt = (idx % 2) == 0;           // run length 1 (worst case)
        NDArray mBlk = (idx % 128) < 64;         // run length 64

        np.add(a, b, o, mAll);
        double t = BestMs(() => np.add(a, b, o, mAll), 25, 8, 7);
        Check(o.GetSingle(777) == a.GetSingle(777) + b.GetSingle(777), "W1");
        Row("W1", "np.add(out=, where=ALL-TRUE) f32 4M", t);

        NpyIter.Copy(o, np.zeros(new Shape(M), np.float32) - 1f);
        t = BestMs(() => np.add(a, b, o, mAlt), 5, 2, 5);
        Check(o.GetSingle(776) == a.GetSingle(776) + b.GetSingle(776) && o.GetSingle(777) == -1f, "W2 mask semantics");
        Row("W2", "np.add(out=, where=ALTERNATING, run=1) f32 4M", t);

        t = BestMs(() => np.add(a, b, o, mBlk), 25, 8, 7);
        Check(o.GetSingle(63) == a.GetSingle(63) + b.GetSingle(63), "W3");
        Row("W3", "np.add(out=, where=BLOCKY, run=64) f32 4M", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // B — buffered cast with STRIDED source
    // =========================================================================
    {
        const int M = 4_194_304;          // backing; view = 2M
        var back = np.arange(M).astype(np.float32);
        var sv = back["::2"];
        var d64 = np.empty(new Shape(M / 2), np.float64);
        var ops = new[] { sv, d64 };
        var f64x2 = new[] { NPTypeCode.Double, NPTypeCode.Double };

        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, ops,
                NpyIterGlobalFlags.BUFFERED | NpyIterGlobalFlags.EXTERNAL_LOOP | NpyIterGlobalFlags.GROWINNER,
                K, SAFE, RO_WO, f64x2);
            it.ForEach(FKern.CopyF64);
        }, 25, 8, 7);
        Check(d64.GetDouble(777) == (double)sv.GetSingle(777), "B1 strided cast");
        Row("B1", "buffered cast copy f32[::2]->f64 2M (windowed)", t);

        double tp = BestMs(() => np.copyto(d64, sv), 25, 8, 7);
        Row("B1p", "  production np.copyto same (strided cast)", tp);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // X — layout frontier: F-order out, reversed source
    // =========================================================================
    {
        const int n = 1448;
        var aC = ((np.arange(n * n).astype(np.float64) % 97.0) + 1.0).reshape(n, n);
        var bC = ((np.arange(n * n).astype(np.float64) % 31.0) + 2.0).reshape(n, n);
        var oF = np.empty(new Shape(n, n), np.float64).T;   // logical (n,n), F-order memory
        var opsX = new[] { aC, bC, oF };

        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(3, opsX, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_RO_WO);
            it.ForEach(FKern.AddF64);
        }, 12, 4, 7);
        Check(Math.Abs(oF.GetDouble(5, 7) - (aC.GetDouble(5, 7) + bC.GetDouble(5, 7))) < 1e-12, "X1 F-out add");
        Row("X1", "iterator add C+C -> F-ORDER out (1448,1448) f64", t);

        double tp = BestMs(() => np.add(aC, bC, oF), 12, 4, 7);
        Check(Math.Abs(oF.GetDouble(7, 5) - (aC.GetDouble(7, 5) + bC.GetDouble(7, 5))) < 1e-12, "X1p F-out add");
        Row("X1p", "  production np.add(out=F-order) same", tp);

        const int M = 4_194_304;
        var src = np.arange(M).astype(np.float64);
        var rev = src["::-1"];
        var dst = np.empty(new Shape(M), np.float64);
        var opsRev = new[] { rev, dst };
        t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(2, opsRev, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, RO_WO);
            it.ForEach(FKern.CopyF64);
        }, 25, 8, 7);
        Check(dst.GetDouble(0) == src.GetDouble(M - 1) && dst.GetDouble(M - 1) == src.GetDouble(0), "X2 reversed copy");
        Row("X2", "iterator copy REVERSED a[::-1] f64 4M -> contig", t);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // O — 0-d scalar ufunc calls (production)
    // =========================================================================
    {
        var s1 = NDArray.Scalar(2.5, NPTypeCode.Double);
        var s2 = NDArray.Scalar(1.5, NPTypeCode.Double);
        var s3 = NDArray.Scalar(0.0, NPTypeCode.Double);
        np.add(s1, s2, s3);
        Check(s3.GetDouble(0) == 4.0, "O1 0-d add");
        Row("O1", "production np.add(0-d, 0-d, out=0-d)",
            BestMs(() => np.add(s1, s2, s3), 200_000, 25_000));
        Row("O2", "production np.add(0-d, 0-d) allocating",
            BestMs(() => { var _ = np.add(s1, s2); }, 100_000, 12_000));
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // P — production np.copyto at the tiny-chunk frontier (w=4)
    // =========================================================================
    {
        const int TOTAL = 2_097_152;
        const int w = 4;
        int rows = TOTAL / w;
        var back = np.arange(rows * 2 * w).astype(np.float64).reshape(rows, 2 * w);
        var sv = back[$":, :{w}"];
        var dst = np.empty(new Shape(rows, w), np.float64);
        double t = BestMs(() => np.copyto(dst, sv), 12, 5, 7);
        Check(dst.GetDouble(rows - 1, w - 1) == sv.GetDouble(rows - 1, w - 1), "P4 copyto");
        Row("P4", "production np.copyto strided rows w=4 (524288 chunks)", t, $"{t * 1e6 / rows:F0} ns/chunk");
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // Y — architecture dividend: 7-input single-pass sum (one iterator)
    //     vs the best NumPy can do from Python: 6 chained 2-op passes (out=)
    // =========================================================================
    {
        const int M = 4_194_304;
        var ins = new NDArray[8];
        for (int i = 0; i < 7; i++) ins[i] = (np.arange(M).astype(np.float64) % (7.0 + i)) + 1.0;
        ins[7] = np.empty(new Shape(M), np.float64);
        var flags8 = new NpyIterPerOpFlags[8];
        for (int i = 0; i < 7; i++) flags8[i] = NpyIterPerOpFlags.READONLY;
        flags8[7] = NpyIterPerOpFlags.WRITEONLY;

        double t = BestMs(() =>
        {
            using var it = NpyIterRef.MultiNew(8, ins, NpyIterGlobalFlags.EXTERNAL_LOOP, K, SAFE, flags8);
            it.ForEach(FKern.Sum7F64);
        }, 25, 8, 7);
        double expect = 0;
        for (int i = 0; i < 7; i++) expect += ins[i].GetDouble(777);
        Check(Math.Abs(ins[7].GetDouble(777) - expect) < 1e-9, "Y1 7-input sum");
        Row("Y1", "ONE-PASS sum of 7 arrays f64 4M (8-op iterator)", t);

        var acc = np.empty(new Shape(M), np.float64);
        double tc = BestMs(() =>
        {
            np.add(ins[0], ins[1], acc);
            for (int i = 2; i < 7; i++) np.add(acc, ins[i], acc);
        }, 25, 8, 7);
        Check(Math.Abs(acc.GetDouble(777) - expect) < 1e-9, "Y2 chained sum");
        Row("Y2", "  chained 6x np.add(out=) same data (NumSharp)", tc);
    }
    GC.Collect(); GC.WaitForPendingFinalizers();

    // =========================================================================
    // Z — kernel-bound dtype frontier (context: kernels riding the iterator)
    // =========================================================================
    {
        const int M = 4_194_304;
        try
        {
            var ac = np.arange(M).astype(np.complex128);
            var bc = (np.arange(M).astype(np.float64) % 7.0 + 1.0).astype(np.complex128);
            var oc = np.empty(new Shape(M), np.complex128);
            np.add(ac, bc, oc);
            Row("Z1", "production np.add complex128 4M (out=)",
                BestMs(() => np.add(ac, bc, oc), 12, 4, 7));
            np.multiply(ac, bc, oc);
            Row("Z2", "production np.multiply complex128 4M (out=)",
                BestMs(() => np.multiply(ac, bc, oc), 12, 4, 7));
        }
        catch (Exception ex) { Console.WriteLine($"Z1/Z2  complex128 — THROWS: {ex.Message.Split('\n')[0]}"); }

        try
        {
            var ah = (np.arange(M) % 1000).astype(np.float16);
            var bh = (np.arange(M) % 31).astype(np.float16);
            var oh = np.empty(new Shape(M), np.float16);
            np.add(ah, bh, oh);
            Row("Z3", "production np.add float16 4M (out=)",
                BestMs(() => np.add(ah, bh, oh), 12, 4, 7));
        }
        catch (Exception ex) { Console.WriteLine($"Z3     float16 — THROWS: {ex.Message.Split('\n')[0]}"); }

        try
        {
            var ai = (np.arange(M) % 100).astype(np.int8);
            var bi = (np.arange(M) % 27).astype(np.int8);
            var oi = np.empty(new Shape(M), np.int8);
            np.add(ai, bi, oi);
            Row("Z4", "production np.add int8 4M (out=)",
                BestMs(() => np.add(ai, bi, oi), 50, 12, 7));
        }
        catch (Exception ex) { Console.WriteLine($"Z4     int8 — THROWS: {ex.Message.Split('\n')[0]}"); }
    }
}

Console.WriteLine(new string('-', 95));
Console.WriteLine(fails == 0 ? "ALL CORRECTNESS CHECKS PASS" : $"{fails} CORRECTNESS FAILURES");
Console.Error.WriteLine("[done]");

// =============================================================================
// Kernels (trivial, NumPy-loop-family-matched)
// =============================================================================
static unsafe class FKern
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
    public static void AddF64(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0];
        byte* pb = (byte*)dp[1];
        byte* po = (byte*)dp[2];
        long sa = st[0], sb = st[1], so = st[2];
        long i = 0;
        if (so == 8 && sa == 8 && sb == 8)
        {
            for (; i + 8 <= count; i += 8)
            {
                Vector256.Store(Vector256.Load((double*)pa) + Vector256.Load((double*)pb), (double*)po);
                Vector256.Store(Vector256.Load((double*)(pa + 32)) + Vector256.Load((double*)(pb + 32)), (double*)(po + 32));
                pa += 64; pb += 64; po += 64;
            }
        }
        for (; i < count; i++) { *(double*)po = *(double*)pa + *(double*)pb; pa += sa; pb += sb; po += so; }
    }

    /// <summary>out[i] += a[i] — axis-0 reduction inner loop (both contig).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void AddAccumF64(void** dp, long* st, long count, void* aux)
    {
        byte* pa = (byte*)dp[0];
        byte* po = (byte*)dp[1];
        long sa = st[0], so = st[1];
        long i = 0;
        if (sa == 8 && so == 8)
        {
            for (; i + 4 <= count; i += 4)
            {
                Vector256.Store(Vector256.Load((double*)po) + Vector256.Load((double*)pa), (double*)po);
                pa += 32; po += 32;
            }
        }
        for (; i < count; i++) { *(double*)po += *(double*)pa; pa += sa; po += so; }
    }

    /// <summary>*out += sum(a[0..count)) — axis-1 reduction inner loop (out stride 0).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SumIntoF64(void** dp, long* st, long count, void* aux)
    {
        byte* ps = (byte*)dp[0];
        long ss = st[0];
        long i = 0;
        double s = 0;
        if (ss == 8)
        {
            var a0 = Vector256<double>.Zero;
            var a1 = Vector256<double>.Zero;
            for (; i + 8 <= count; i += 8)
            {
                a0 += Vector256.Load((double*)ps);
                a1 += Vector256.Load((double*)(ps + 32));
                ps += 64;
            }
            s = Vector256.Sum(a0 + a1);
        }
        for (; i < count; i++) { s += *(double*)ps; ps += ss; }
        *(double*)dp[1] += s;
    }

    /// <summary>out = in0+in1+...+in6 (8-op single pass; contig fast path).</summary>
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
                      + Vector256.Load(p3 + i) + Vector256.Load(p4 + i) + Vector256.Load(p5 + i)
                      + Vector256.Load(p6 + i);
                Vector256.Store(v, po + i);
            }
            for (; i < count; i++)
                po[i] = p0[i] + p1[i] + p2[i] + p3[i] + p4[i] + p5[i] + p6[i];
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
