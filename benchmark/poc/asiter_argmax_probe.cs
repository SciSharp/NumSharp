#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
//
// asiter_argmax_probe.cs — argmin/argmax need C-ORDER first-occurrence index.
// KEEPORDER (default) would re-traverse and return the wrong index on non-contig.
// Probe: ExecuteReducing with NPY_CORDER + a running-index kernel.
//   - Correctness: index must match the AsIterator baseline (contig AND strided).
//   - Speed: vs AsIterator baseline.
//
// Run:  dotnet run -c Release - < benchmark/poc/asiter_argmax_probe.cs
//
using System;
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends;
using NumSharp.Backends.Iteration;

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false) { Console.WriteLine("FATAL: Debug build. Use dotnet run -c Release."); return; }

const int ROUNDS = 7;
var rng = new Random(777);

NDArray HalfArr(int n)
{
    var a = new double[n];
    for (int i = 0; i < n; i++) a[i] = rng.NextDouble();
    // plant a unique max at a known-ish spot to exercise tiebreak/index
    a[n / 3] = 5.0;
    return np.array(a).astype(NPTypeCode.Half);
}
NDArray Strided(NDArray flat)
{
    int n = (int)flat.size; int r = (int)Math.Sqrt(n);
    while (r > 1 && n % r != 0) r--;
    return flat.reshape(r, n / r).T;
}

double Bench(Func<long> op, long n, out long res)
{
    int reps = (int)Math.Max(1, 8_000_000L / n);
    for (int i = 0; i < 2; i++) res = op(); res = op();
    double best = double.MaxValue;
    for (int round = 0; round < ROUNDS; round++)
    {
        var sw = Stopwatch.StartNew();
        long last = 0; for (int i = 0; i < reps; i++) last = op();
        sw.Stop();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds / reps);
        res = last;
    }
    return best;
}

Console.WriteLine($"{"op",-12} {"layout",-9} {"N",9} | A(ms) AsIter | B(ms) Exec-CORDER  speedup | idxA  idxB  match");
Console.WriteLine(new string('-', 95));

foreach (int n in new[] { 4096, 4_000_000 })
{
    var hC = HalfArr(n); var hS = Strided(HalfArr(n));
    foreach (var (lay, arr) in new[] { ("contig", hC), ("strided", hS) })
    {
        double ra = Bench(() => ArgMax_AsIter(arr), n, out long ia);
        double rb = Bench(() => ArgMax_Exec(arr), n, out long ib);
        Console.WriteLine($"{"Half.argmax",-12} {lay,-9} {n,9} | {ra,11:F4} | {rb,11:F4} {ra/rb,7:F2}x | {ia,5} {ib,5}  {(ia==ib ? "ok" : "MISMATCH!")}");
    }
}

// baseline: AsIterator, logical (C) order, first occurrence
static long ArgMax_AsIter(NDArray arr)
{
    var it = arr.AsIterator<Half>(); long bi = 0, idx = 0;
    double best = (double)it.MoveNext(); if (double.IsNaN(best)) return 0; idx = 1;
    while (it.HasNext()) { double v = (double)it.MoveNext(); if (double.IsNaN(v)) return idx; if (v > best) { best = v; bi = idx; } idx++; }
    return bi;
}
// candidate: ExecuteReducing, FORCED C-ORDER so chunks arrive in logical order;
// kernel keeps a running flat index.
static unsafe long ArgMax_Exec(NDArray arr)
{
    using var it = NpyIterRef.New(arr, NpyIterGlobalFlags.EXTERNAL_LOOP,
        NPY_ORDER.NPY_CORDER, NPY_CASTING.NPY_SAFE_CASTING);
    var a = it.ExecuteReducing<HalfArgMaxK, ArgAcc>(default, new ArgAcc { Best = double.NegativeInfinity, BestIdx = -1, Cur = 0 });
    return a.SawNaNIdx >= 0 ? a.SawNaNIdx : a.BestIdx;
}

public struct ArgAcc { public double Best; public long BestIdx; public long Cur; public long SawNaNIdx; }

public readonly struct HalfArgMaxK : INpyReducingInnerLoop<ArgAcc>
{
    public unsafe bool Execute(void** dp, long* st, long count, ref ArgAcc a)
    {
        byte* p = (byte*)dp[0]; long s = st[0];
        double best = a.Best; long bi = a.BestIdx; long cur = a.Cur;
        if (bi < 0) { } // init sentinel
        for (long i = 0; i < count; i++)
        {
            double v = (double)*(Half*)(p + i * s);
            if (double.IsNaN(v)) { a.SawNaNIdx = cur + i; return false; }
            if (bi < 0 || v > best) { best = v; bi = cur + i; }
        }
        a.Best = best; a.BestIdx = bi; a.Cur = cur + count; a.SawNaNIdx = -1;
        return true;
    }
}
