#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_remainder_probe.cs — attribution of evaltext_bench's "filter" cells (docs/plans/evaluate-text.md, §5 / §6).
//
// The text filter (names == "paris") & (age > 30) is slower fused than unfused at 100K+ although its text stage is the
// same np.equal call on both routes. This probe times the NUMERIC remainder alone — m & (age > 30), m a contiguous bool
// (the layout of a stage result), age int64 contiguous or [::2] of a 2n array — fused against unfused. It uses only the
// public API, so it compiles against any NumSharp.Core with np.evaluate: run it on a commit before the text work (a
// detached worktree, with #:project pointed there) and on the current one to show the gap is the numeric kernel's own.
// Measured 2026-11-05: 0.64x contiguous / 0.34x strided at 100K, 0.91x / 0.52x at 1M on 78b28c2b, the same on E1.
//
// Run (Release only): dotnet run -c Release benchmark/fusion/probes/evaltext_remainder_probe.cs
// Pinned to 0x14 (two P-cores) unless NS_PROBE_AFFINITY (hex) says otherwise; best of 7 rounds after a paused double
// warm-up, two interleaved rounds per route, the routes checked equal first.
// =============================================================================
using System.Diagnostics;
using NumSharp;
using NumSharp.Backends.Iteration;

// The pin is set before any array exists, so the tier-1 compiler threads inherit it with the rest of the process.
string affinity = Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } mask ? mask : "14";
Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

#if DEBUG
Console.WriteLine("FATAL: Debug build — rerun with -c Release");
return;
#pragma warning disable CS0162
#endif

foreach (string layout in new[] { "contig", "strided" })
foreach (int n in new[] { 100_000, 1_000_000 })
{
    int total = layout == "contig" ? n : 2 * n;
    NDArray m = (np.arange(n) % 8) == 0;
    NDArray baseAge = np.arange(total).astype(NPTypeCode.Int64) % 97;
    NDArray age = layout == "contig" ? baseAge : baseAge["::2"];

    Func<NDArray> fused = () => np.evaluate((NDExpr)m & ((NDExpr)age > 30));
    Func<NDArray> unfused = () => { using var b = age > 30; return m & b; };
    using (var r1 = fused())
    using (var r2 = unfused())
        if (!np.array_equal(r1, r2))
            throw new Exception($"{layout} {n}: fused != unfused");

    // Best per-call time of f in microseconds: a warm-up burst queues the hot methods for tier-1, the pause lets the
    // background compiler install them, a second burst runs the tier-1 code once; then the minimum over 7 rounds.
    double Time(Func<NDArray> f)
    {
        for (int w = 0; w < 20; w++) f().Dispose();
        Thread.Sleep(100);
        for (int w = 0; w < 20; w++) f().Dispose();
        int reps = n <= 100_000 ? 200 : 25;
        double best = double.MaxValue;
        for (int r = 0; r < 7; r++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < reps; i++) f().Dispose();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / reps);
        }
        return best;
    }

    double tf = double.MaxValue, tu = double.MaxValue;
    for (int round = 0; round < 2; round++)
    {
        tf = Math.Min(tf, Time(fused));
        tu = Math.Min(tu, Time(unfused));
    }
    Console.WriteLine($"numeric-only m & (age > 30), {layout}, n={n}: fused {tf:F1} us, unfused {tu:F1} us, unfused/fused {tu / tf:F2}x");
}
