#:project ../../../src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// neighbours_probe.cs — the neighbours of the Tier 1 levers and the Tier 2 claims, measured
// (docs/NDITER_PERF_CONTINUATION.md):
//   (E) fancy-index shapes still on the delegate route: two index arrays, slice + index array,
//       an F-order source, a strided index view, their setters; boolean masks / compress /
//       take(axis=1) / put-into-a-view for contrast
//   (F) where= neighbours: a cast out (buffered), comparison where=, copyto(where=), np.where,
//       where= without out, and the cast-out masked narrow-row case
//   (G) the Tier 2 claims: buffered-cast unary routes, widening flat reductions, narrow inner /
//       few-row leading axis reductions, tiny-op fixed cost
//
//   NS_PROBE_AFFINITY=0x4 DOTNET_TC_CallCountingDelayMs=0 dotnet run -c Release benchmark/nditer/probes/neighbours_probe.cs [EFG]
//   NS_PROBE_AFFINITY=0x4 python benchmark/nditer/probes/numpy_twins.py neighbours [EFG]   # NumPy side, same keys
//   python benchmark/nditer/probes/numpy_twins.py join before.tsv after.tsv numpy.tsv
//
// Keyed TSV on stdout (key 	 microseconds), best-of-rounds. Pin BOTH sides to one P-core and never
// run them at once (docs/NDITER_PERF_DISCOVERY.md §3, traps 11–13).
// =============================================================================
using System.Diagnostics;
using NumSharp;

if (Environment.GetEnvironmentVariable("NS_PROBE_AFFINITY") is { Length: > 0 } affinity)
    Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)Convert.ToInt64(affinity, 16);

var sections = args.Length == 0 ? "EFG" : string.Concat(args);

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

if (sections.Contains('E'))
{
    foreach (int n in new[] { 100_000, 1_000_000 })
    {
        string tag = n == 100_000 ? "100K" : "1M";
        int side = (int)Math.Sqrt(n);                       // 316 / 1000
        var m = (np.arange(side * side).astype(np.float64) % 97.0 + 1.0).reshape(side, side);
        var mT = m.T;                                       // non-contiguous (F-order) source
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var ri = ((np.arange(n).astype(np.int64) * 2654435761L) % side);          // n row indices
        var ci = ((np.arange(n).astype(np.int64) * 40503L) % side);               // n col indices
        var rk = ((np.arange(side).astype(np.int64) * 2654435761L) % side);       // side row indices
        var idx = ((np.arange(n).astype(np.int64) * 2654435761L) % n);
        var idxView = np.concatenate(new[] { idx, idx })["::2"];                  // genuine stride-2 view over the index values
        var mask = (np.arange(n) % 2) == 0;                                       // 50 % true
        var maskFew = (np.arange(n) % 100) == 0;                                  // 1 % true
        var vals = np.arange(n).astype(np.float64);
        var dst = a.copy();
        var dstView = np.arange(2 * n).astype(np.float64)["::2"];                 // non-contiguous put target
        var colVals = np.arange(side * side).astype(np.float64).reshape(side, side);

        Row($"{tag}|fancy|m[ri,ci] (2 index arrays)", BestUs(() => { var r = m[ri, ci]; r.Dispose(); }));
        Row($"{tag}|fancy|m[:, rk] (column gather)", BestUs(() => { var r = m[Slice.All, rk]; r.Dispose(); }));
        Row($"{tag}|fancy|m[rk, :2] (rows + slice)", BestUs(() => { var r = m[rk, new Slice(0, 2)]; r.Dispose(); }));
        Row($"{tag}|fancy|mT[rk] (F-order source rows)", BestUs(() => { var r = mT[rk]; r.Dispose(); }));
        Row($"{tag}|fancy|a[idx[::2]] (strided index view)", BestUs(() => { var r = a[idxView]; r.Dispose(); }));
        Row($"{tag}|fancy|m[ri,ci]=v (2-array set)", BestUs(() => m[ri, ci] = vals));
        Row($"{tag}|fancy|m[:, rk]=v (column set)", BestUs(() => m[Slice.All, rk] = colVals));
        Row($"{tag}|mask|a[mask] (50%)", BestUs(() => { var r = a[mask]; r.Dispose(); }));
        Row($"{tag}|mask|a[mask] (1%)", BestUs(() => { var r = a[maskFew]; r.Dispose(); }));
        Row($"{tag}|mask|a[mask]=scalar (50%)", BestUs(() => dst[mask] = 3.0));
        Row($"{tag}|mask|a[mask]=v (50%)", BestUs(() => dst[mask] = vals[mask]));
        Row($"{tag}|mask|compress", BestUs(() => { var r = np.compress(mask, a); r.Dispose(); }));
        Row($"{tag}|mask|count_nonzero", BestUs(() => np.count_nonzero(mask)));
        Row($"{tag}|take|axis1 m.take(rk, axis=1)", BestUs(() => { var r = np.take(m, rk, axis: 1); r.Dispose(); }));
        Row($"{tag}|put|put(view[::2], idx, v)", BestUs(() => np.put(dstView, idx, vals)));
        Row($"{tag}|put|view[::2][idx]=v", BestUs(() => dstView[idx] = vals));
    }
}

if (sections.Contains('F'))
{
    foreach (int n in new[] { 100_000, 1_000_000 })
    {
        string tag = n == 100_000 ? "100K" : "1M";
        var a = (np.arange(n).astype(np.float64) % 97.0) + 1.0;
        var b = (np.arange(n).astype(np.float64) % 31.0) + 2.0;
        var a32 = a.astype(np.float32); var b32 = b.astype(np.float32);
        var o = np.empty(new Shape(n), np.float64);
        var ob = np.empty(new Shape(n), np.@bool);
        var ar = np.arange(n);
        NDArray blocks = (np.floor_divide(ar, 64) % 2) == 0;
        Row($"{tag}|where|add(f32,f32,out=f64,where) cast-out", BestUs(() => np.add(a32, b32, o, where: blocks)));
        Row($"{tag}|where|add(f32,f32,out=f64) cast-out unmasked", BestUs(() => np.add(a32, b32, o)));
        Row($"{tag}|where|less(a,b,out,where)", BestUs(() => np.less(a, b, ob, where: blocks)));
        Row($"{tag}|where|less(a,b,out)", BestUs(() => np.less(a, b, ob)));
        Row($"{tag}|where|copyto(o,a,where)", BestUs(() => np.copyto(o, a, where: blocks)));
        Row($"{tag}|where|np.where(mask,a,b)", BestUs(() => { var r = np.where(blocks, a, b); r.Dispose(); }));
        Row($"{tag}|where|add(a,b,where) no out (alloc)", BestUs(() => { var r = np.add(a, b, where: blocks); r.Dispose(); }));
        // masked narrow rows with a CAST out (buffered → excluded from the block kernel)
        int w = 4, rows = n / w;
        var sv = a.reshape(rows, w); var sv2 = b.reshape(rows, w);
        var o32 = np.empty(new Shape(rows, w), np.float32);
        var m2 = blocks.reshape(rows, w);
        var back = (np.arange(rows * 2 * w).astype(np.float64) % 97.0 + 1.0).reshape(rows, 2 * w);
        var svs = back[":, :4"];
        var o2 = np.empty(new Shape(rows, w), np.float64);
        Row($"{tag}|where|rows w4 add(view,view,out=f32,where) cast-out", BestUs(() => np.add(svs, sv2, o32, where: m2)));
        Row($"{tag}|where|rows w4 add(view,view,out,where)", BestUs(() => np.add(svs, sv2, o2, where: m2)));
    }
}

if (sections.Contains('G'))
{
    foreach (int n in new[] { 100_000, 1_000_000 })
    {
        string tag = n == 100_000 ? "100K" : "1M";
        var i32 = (np.arange(n).astype(np.int32) % 97) + 1;
        var i64 = i32.astype(np.int64);
        var f32 = i32.astype(np.float32);
        var f64 = i32.astype(np.float64);
        var o64 = np.empty(new Shape(n), np.float64);
        Row($"{tag}|cast|sqrt(i32)", BestUs(() => { var r = np.sqrt(i32); r.Dispose(); }));
        Row($"{tag}|cast|sqrt(i32,out=f64)", BestUs(() => np.sqrt(i32, o64)));
        Row($"{tag}|cast|sqrt(i64)", BestUs(() => { var r = np.sqrt(i64); r.Dispose(); }));
        Row($"{tag}|cast|sqrt(f32)", BestUs(() => { var r = np.sqrt(f32); r.Dispose(); }));
        Row($"{tag}|cast|sqrt(f64)", BestUs(() => { var r = np.sqrt(f64); r.Dispose(); }));
        Row($"{tag}|cast|exp(i32)", BestUs(() => { var r = np.exp(i32); r.Dispose(); }));
        Row($"{tag}|cast|negative(i32,out=f64)", BestUs(() => np.negative(i32, o64)));
        Row($"{tag}|cast|add(i32,f64)", BestUs(() => { var r = np.add(i32, f64); r.Dispose(); }));
        Row($"{tag}|cast|astype(i32->f64)", BestUs(() => { var r = i32.astype(np.float64); r.Dispose(); }));
        Row($"{tag}|reduce|sum(f32,dtype=f64)", BestUs(() => np.sum(f32, dtype: np.float64)));
        Row($"{tag}|reduce|sum(f32)", BestUs(() => np.sum(f32)));
        Row($"{tag}|reduce|sum(i32) (->i64)", BestUs(() => np.sum(i32)));
        Row($"{tag}|reduce|mean(f32)", BestUs(() => np.mean(f32)));
        Row($"{tag}|reduce|sum(i32,dtype=f64)", BestUs(() => np.sum(i32, dtype: np.float64)));
        foreach (int w in new[] { 3, 4, 8, 16, 64 })
        {
            int rows = n / w;
            var x = f64[$":{rows * w}"].reshape(rows, w);
            Row($"{tag}|axis|sum(x,axis=1) w{w}", BestUs(() => { var r = np.sum(x, axis: 1); r.Dispose(); }));
            Row($"{tag}|axis|max(x,axis=1) w{w}", BestUs(() => { var r = np.max(x, axis: 1); r.Dispose(); }));
            Row($"{tag}|axis|mean(x,axis=1) w{w}", BestUs(() => { var r = np.mean(x, axis: 1); r.Dispose(); }));
            var xt = f64[$":{rows * w}"].reshape(w, rows);
            Row($"{tag}|axis|sum(xt,axis=0) ({w},N)", BestUs(() => { var r = np.sum(xt, axis: 0); r.Dispose(); }));
        }
    }
    // tiny ops: the iterator fixed cost
    foreach (int n in new[] { 1, 16, 100, 1000 })
    {
        var a = np.arange(n).astype(np.float64) + 1.0; var b = a.copy(); var o = a.copy();
        var i32 = np.arange(n).astype(np.int32);
        Row($"tiny{n}|add(out)", BestUs(() => np.add(a, b, o)));
        Row($"tiny{n}|add (alloc)", BestUs(() => { var r = np.add(a, b); r.Dispose(); }));
        Row($"tiny{n}|sqrt(out)", BestUs(() => np.sqrt(a, o)));
        Row($"tiny{n}|sqrt(i32) cast", BestUs(() => { var r = np.sqrt(i32); r.Dispose(); }));
        Row($"tiny{n}|sum", BestUs(() => np.sum(a)));
        Row($"tiny{n}|less(out)", BestUs(() => { var r = np.less(a, b); r.Dispose(); }));
    }
}
