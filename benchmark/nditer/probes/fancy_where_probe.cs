#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// fancy_where_probe.cs — the Tier 1 levers of docs/NDITER_PERF_DISCOVERY.md §7, measured:
//   (A) the fancy-index operator (a[idx], a[idx] = v, m[ridx], m[ridx] = row) against np.take/np.put,
//       int32 AND int64 indices, 1K / 100K / 10M
//   (B) the where= masked driver across mask run lengths (1, 8, 64, 1024, half, all-true, all-false)
//   (C) where= over narrow strided rows (w = 3 / 4 / 16; byte mask, row mask, column mask)
//   (D) the int32-index-in-place kernel variants against the int64-only alternative (widen the index
//       array to an int64 temp, run the int64 kernel) and against a native int64 index — C# only, no
//       NumPy twin; the measurement that justifies keeping the idx32 variants (NumSharp indexes with
//       int64 everywhere; a narrower width is used only where it removes a copy)
//
//   NS_PROBE_AFFINITY=0x4 DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/nditer/probes/fancy_where_probe.cs [ABCD]
//   NS_PROBE_AFFINITY=0x4 python benchmark/nditer/probes/numpy_twins.py fancy_where [ABC]   # NumPy side, same keys
//   python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv
//
// Keyed TSV on stdout (key 	 microseconds), best-of-rounds. Pin BOTH sides to one P-core and never
// run them at once (docs/NDITER_PERF_DISCOVERY.md §3, trap 11). Masks are built with np.floor_divide:
// `/` on an integer NDArray is NumPy true division, so `(ar / 64 % 2) == 0` is a SPARSE mask (one
// true element every 128), not 64-element runs — the trap the first where= numbers fell into.
// =============================================================================
using System.Diagnostics;
using NumSharp;

if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core"); return; }

var sections = args.Length == 0 ? "ABC" : string.Concat(args);

static double BestUs(Action body)
{
    var warm = Stopwatch.StartNew(); body(); while (warm.ElapsedMilliseconds < 80) body();
    var pilot = Stopwatch.StartNew(); int pc = 0; while (pilot.ElapsedMilliseconds < 20) { body(); pc++; }
    double perCall = pilot.Elapsed.TotalMilliseconds / pc;
    int it = Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(5, (int)Math.Ceiling(250.0 / Math.Max(it * perCall, 1e-9)));
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) body(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1e3 / it); }
    return best;
}
static void Row(string key, double us) => Console.WriteLine($"{key}\t{us:G17}");

if (sections.Contains('A'))
{
    foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
    {
        string tag = n switch { 1_000 => "1K", 100_000 => "100K", _ => "10M" };
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var ai = np.arange(n).astype(np.int32);
        var idx32 = ((np.arange(n).astype(np.int64) * 2654435761L) % n).astype(np.int32);
        var idx64 = idx32.astype(np.int64);
        var vals = np.arange(n).astype(np.float64);
        var valsI = np.arange(n).astype(np.int32);
        var dst = a.copy(); var dstI = ai.copy();
        Row($"{tag}|f64|get|a[idx32]", BestUs(() => { var r = a[idx32]; r.Dispose(); }));
        Row($"{tag}|f64|get|a[idx64]", BestUs(() => { var r = a[idx64]; r.Dispose(); }));
        Row($"{tag}|f64|get|take(a,idx32)", BestUs(() => { var r = np.take(a, idx32); r.Dispose(); }));
        Row($"{tag}|f64|get|take(a,idx64)", BestUs(() => { var r = np.take(a, idx64); r.Dispose(); }));
        Row($"{tag}|i32|get|a[idx32]", BestUs(() => { var r = ai[idx32]; r.Dispose(); }));
        Row($"{tag}|i32|get|take(a,idx64)", BestUs(() => { var r = np.take(ai, idx64); r.Dispose(); }));
        Row($"{tag}|f64|set|a[idx32]=v", BestUs(() => dst[idx32] = vals));
        Row($"{tag}|f64|set|a[idx64]=v", BestUs(() => dst[idx64] = vals));
        Row($"{tag}|f64|set|put(a,idx64,v)", BestUs(() => np.put(dst, idx64, vals)));
        Row($"{tag}|f64|set|a[idx64]=scalar", BestUs(() => dst[idx64] = 3.0));
        Row($"{tag}|i32|set|a[idx32]=v", BestUs(() => dstI[idx32] = valsI));
        Row($"{tag}|i32|set|put(a,idx64,v)", BestUs(() => np.put(dstI, idx64, valsI)));
        // 2-D row gather / scatter: (n/8, 8) f64, row indices
        if (n >= 8)
        {
            int rows = n / 8;
            var m = a.reshape(rows, 8);
            var ridx = ((np.arange(rows).astype(np.int64) * 2654435761L) % rows).astype(np.int64);
            var rvals = np.arange(rows * 8).astype(np.float64).reshape(rows, 8);
            var md = m.copy();
            Row($"{tag}|f64|get2d|m[ridx]", BestUs(() => { var r = m[ridx]; r.Dispose(); }));
            Row($"{tag}|f64|get2d|take(m,ridx,0)", BestUs(() => { var r = np.take(m, ridx, axis: 0); r.Dispose(); }));
            Row($"{tag}|f64|set2d|m[ridx]=v", BestUs(() => md[ridx] = rvals));
            Row($"{tag}|f64|set2d|m[ridx]=row", BestUs(() => md[ridx] = rvals["0"]));
        }
        a.Dispose(); ai.Dispose(); idx32.Dispose(); idx64.Dispose(); vals.Dispose(); valsI.Dispose(); dst.Dispose(); dstI.Dispose();
    }
}

if (sections.Contains('B'))
{
    foreach (int n in new[] { 100_000, 1_000_000 })
    {
        string tag = n == 100_000 ? "100K" : "1M";
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var o = np.empty(new Shape(n), np.float64);
        var ar = np.arange(n);
        foreach (int run in new[] { 1, 8, 64, 1024 })
        {
            NDArray mask = (np.floor_divide(ar, run) % 2) == 0;
            Row($"{tag}|add|where=run{run}", BestUs(() => np.add(a, b, o, where: mask)));
            if (run == 64) Row($"{tag}|sqrt|where=run{run}", BestUs(() => np.sqrt(a, o, where: mask)));
            mask.Dispose();
        }
        NDArray half = ar < n / 2;
        NDArray allT = ar >= 0;
        NDArray allF = ar < 0;
        Row($"{tag}|add|where=half", BestUs(() => np.add(a, b, o, where: half)));
        Row($"{tag}|add|where=allTrue", BestUs(() => np.add(a, b, o, where: allT)));
        Row($"{tag}|add|where=allFalse", BestUs(() => np.add(a, b, o, where: allF)));
        Row($"{tag}|add|unmasked", BestUs(() => np.add(a, b, o)));
        half.Dispose(); allT.Dispose(); allF.Dispose(); a.Dispose(); b.Dispose(); o.Dispose(); ar.Dispose();
    }
}

if (sections.Contains('C'))
{
    // where= on narrow strided rows: (rows, w) column views, mask contiguous (rows, w)
    foreach (int w in new[] { 3, 4, 16 })
    {
        const int TOTAL = 1_000_000; int rows = TOTAL / w;
        var back = (np.arange(rows * 2 * w).astype(np.float64) % 97.0 + 1.0).reshape(rows, 2 * w);
        var back2 = (np.arange(rows * 2 * w).astype(np.float64) % 31.0 + 2.0).reshape(rows, 2 * w);
        var sv = back[$":, :{w}"]; var sv2 = back2[$":, :{w}"];
        var o = np.empty(new Shape(rows, w), np.float64);
        var ar = np.arange(rows * w).reshape(rows, w);
        NDArray blocks = (np.floor_divide(ar, 64) % 2) == 0;
        NDArray rowmask = ((np.arange(rows) % 2) == 0)[":, np.newaxis"];   // broadcast column mask (rows,1)
        NDArray colmask = ((np.arange(w) % 2) == 0)["np.newaxis, :"];      // broadcast row mask (1,w)
        Row($"1M|w{w}|add|where=blocks64", BestUs(() => np.add(sv, sv2, o, where: blocks)));
        Row($"1M|w{w}|add|where=rowmask", BestUs(() => np.add(sv, sv2, o, where: rowmask)));
        Row($"1M|w{w}|add|where=colmask", BestUs(() => np.add(sv, sv2, o, where: colmask)));
        Row($"1M|w{w}|sqrt|where=blocks64", BestUs(() => np.sqrt(sv, o, where: blocks)));
        Row($"1M|w{w}|add|unmasked", BestUs(() => np.add(sv, sv2, o)));
        Row($"1M|w{w}|sqrt|unmasked", BestUs(() => np.sqrt(sv, o)));
        blocks.Dispose(); rowmask.Dispose(); colmask.Dispose(); back.Dispose(); back2.Dispose(); o.Dispose(); ar.Dispose();
    }
}

if (sections.Contains('D'))
{
    // int32 index read in place (the idx32 kernels) vs the int64-only design (astype(int64) temp + the
    // int64 kernel, temp disposed) vs a native int64 index. Keys: <tag>|<dtype>|<op>|idx32-inplace /
    // idx32-widen / idx64. The variants earn their place only while idx32-widen / idx32-inplace > 1.
    foreach (int n in new[] { 1_000, 100_000, 10_000_000 })
    {
        string tag = n switch { 1_000 => "1K", 100_000 => "100K", _ => "10M" };
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var idx32 = ((np.arange(n).astype(np.int64) * 2654435761L) % n).astype(np.int32);
        var idx64 = idx32.astype(np.int64);
        var vals = np.arange(n).astype(np.float64);
        var dst = a.copy();
        Row($"{tag}|f64|get|idx32-inplace", BestUs(() => { var r = a[idx32]; r.Dispose(); }));
        Row($"{tag}|f64|get|idx32-widen", BestUs(() => { var t = idx32.astype(np.int64); var r = a[t]; r.Dispose(); t.Dispose(); }));
        Row($"{tag}|f64|get|idx64", BestUs(() => { var r = a[idx64]; r.Dispose(); }));
        Row($"{tag}|f64|take|idx32-inplace", BestUs(() => { var r = np.take(a, idx32); r.Dispose(); }));
        Row($"{tag}|f64|take|idx32-widen", BestUs(() => { var t = idx32.astype(np.int64); var r = np.take(a, t); r.Dispose(); t.Dispose(); }));
        Row($"{tag}|f64|take|idx64", BestUs(() => { var r = np.take(a, idx64); r.Dispose(); }));
        Row($"{tag}|f64|set|idx32-inplace", BestUs(() => dst[idx32] = vals));
        Row($"{tag}|f64|set|idx32-widen", BestUs(() => { var t = idx32.astype(np.int64); dst[t] = vals; t.Dispose(); }));
        Row($"{tag}|f64|set|idx64", BestUs(() => dst[idx64] = vals));
        Row($"{tag}|f64|put|idx32-inplace", BestUs(() => np.put(dst, idx32, vals)));
        Row($"{tag}|f64|put|idx32-widen", BestUs(() => { var t = idx32.astype(np.int64); np.put(dst, t, vals); t.Dispose(); }));
        Row($"{tag}|f64|put|idx64", BestUs(() => np.put(dst, idx64, vals)));
        a.Dispose(); idx32.Dispose(); idx64.Dispose(); vals.Dispose(); dst.Dispose();
    }
}
