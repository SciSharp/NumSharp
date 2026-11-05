#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// evaltext_where_mask_probe.cs — attribution of evaltext_e2_bench's "find" cells (docs/plans/evaluate-text.md, §5 / §6).
//
// The text tree Where(Find(names, "o") >= 0, price, 0) is slower fused than unfused on contiguous data at 100K+, although
// its Find stage costs the same on both routes (np.evaluate(NDExpr.Find(names, o)) ≈ np.strings.find). This probe times
// the NUMERIC remainder Where(f >= 0, price, 0) alone — f int64, price float64 — fused against unfused, over a mask that
// repeats every 7 elements and over a random one (find's results are random in the benchmark's data): the fused kernel's
// time depends on the mask's predictability (a per-element branch), the unfused np.where's does not (a vector select).
// Public API only, so it runs against any commit with np.evaluate. Measured 2026-11-05: every-7th mask fused 67 µs /
// unfused 134 µs at 100K (2.0x), 1.08 / 1.90 ms at 1M; random mask fused 248 µs / unfused 115 µs at 100K (0.46x), 2.55 /
// 2.20 ms at 1M — over find's own result (the benchmark's city names) the fused remainder took 5.0 ms at 1M.
//
// Run (Release only): dotnet run -c Release --artifacts-path <its own folder> benchmark/fusion/probes/evaltext_where_mask_probe.cs
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

foreach (int n in new[] { 100_000, 1_000_000 })
foreach (string pattern in new[] { "every-7th", "random" })
{
    // -1 .. 5: the int64 range a find result takes; "random" draws it from a fixed-seed generator.
    var rnd = new Random(42);
    var values = new long[n];
    for (int i = 0; i < n; i++)
        values[i] = pattern == "random" ? rnd.Next(-1, 6) : i % 7 - 1;
    var f = np.array(values);
    var price = np.arange(n).astype(NPTypeCode.Double) * 0.5;

    Func<NDArray> fused = () => np.evaluate(NDExpr.Where((NDExpr)f >= 0, price, 0));
    Func<NDArray> unfused = () => { using var c = f >= 0; return np.where(c, price, 0); };
    using (var a = fused())
    using (var b = unfused())
        if (!np.array_equal(a, b))
            throw new Exception($"{pattern} {n}: fused != unfused");

    // Best per-call time of g in microseconds: a warm-up burst queues the hot methods for tier-1, the pause lets the
    // background compiler install them, a second burst runs the tier-1 code once; then the minimum over 7 rounds.
    double Time(Func<NDArray> g)
    {
        for (int w = 0; w < 20; w++) g().Dispose();
        Thread.Sleep(100);
        for (int w = 0; w < 20; w++) g().Dispose();
        int reps = n <= 100_000 ? 200 : 25;
        double best = double.MaxValue;
        for (int r = 0; r < 7; r++)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < reps; i++) g().Dispose();
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
    Console.WriteLine($"Where(f >= 0, price, 0), {pattern} mask, n={n}: fused {tf:F1} us, unfused {tu:F1} us, unfused/fused {tu / tf:F2}x");
}
