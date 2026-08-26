#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// elementwise_layout_bench.cs — NumSharp side of the elementwise op x LAYOUT x
// dtype matrix. Companion: elementwise_layout_bench.py (identical keys).
// Probes whether binary / unary / comparison / copy kernels have the same
// "SIMD only on C-contiguous, scalar otherwise" cliff the reductions had.
// Run ONLY with:  dotnet run -c Release - < benchmark/layout/elementwise_layout_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core — use dotnet run -c Release - < file"); return; }

double BestMs(Action body, int iters, int warm, int rounds)
{
    // Warmup; its timed tail (first call excluded — JIT/pool-cold) doubles as a per-call pilot.
    body();
    var pilot = Stopwatch.StartNew();
    for (int i = 1; i < Math.Max(2, warm); i++) body();
    pilot.Stop();
    double perCall = pilot.Elapsed.TotalMilliseconds / Math.Max(1, Math.Max(2, warm) - 1);
    // Sample rule: >=50 timed rounds (>=20 when one round exceeds 10 ms); rounds auto-size to ~1 ms.
    // Caller's (iters, rounds) are floors/hints only — iters is superseded by the pilot batch.
    int it = perCall > 10.0 ? 1 : Math.Clamp((int)Math.Round(1.0 / Math.Max(perCall, 1e-6)), 1, 1_000_000);
    int rds = Math.Max(rounds, perCall > 10.0 ? 20 : 50);
    double best = double.MaxValue;
    for (int r = 0; r < rds; r++)
    {
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < it; i++) body();
        best = Math.Min(best, sw.Elapsed.TotalMilliseconds / it);
    }
    return best;
}
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");
(int it, int wm, int rd) Pick(int n) => n <= 100_000 ? (200, 30, 3) : (30, 6, 3);

var SIZES = new (string tag, int R, int C)[] { ("100K", 316, 316), ("1M", 1000, 1000) };
var DTYPES = new (string name, NPTypeCode tc)[]
{
    ("f64", NPTypeCode.Double), ("f32", NPTypeCode.Single), ("c128", NPTypeCode.Complex),
    ("f16", NPTypeCode.Half), ("i32", NPTypeCode.Int32), ("i64", NPTypeCode.Int64),
};
string[] OPS = { "add", "mul", "neg", "abs", "sqrt", "less", "copy" };
string[] LAYOUTS = { "C", "F", "T", "strided", "sliced", "negrow", "negcol", "bcast" };

NDArray Layout(NDArray b, string l) => l switch
{
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "strided" => b[":, ::2"], "sliced" => b["1:" + (b.shape[0]-1) + ", 1:" + (b.shape[1]-1)],
    "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape((int)b.shape[0], (int)b.shape[1])),
    _ => throw new Exception(l),
};
NDArray Op(string op, NDArray v) => op switch
{
    "add" => v + v, "mul" => v * v, "neg" => -v, "abs" => np.abs(v),
    "sqrt" => np.sqrt(v), "less" => np.less(v, v), "copy" => v.copy(),
    _ => throw new Exception(op),
};

Console.Error.WriteLine($"[elementwise_layout_bench] cores={Environment.ProcessorCount}");
foreach (var (tag, R, C) in SIZES)
{
    var (it, wm, rd) = Pick(R * C);
    foreach (var (dn, tc) in DTYPES)
    {
        NDArray baseArr;
        try { baseArr = ((np.arange(R * C) % 17) + 1).astype(tc).reshape(R, C); }
        catch (Exception e) { Console.Error.WriteLine($"build {tag}/{dn}: {e.GetType().Name}"); continue; }
        foreach (var lay in LAYOUTS)
        {
            NDArray v;
            try { v = Layout(baseArr, lay); } catch (Exception e) { Console.Error.WriteLine($"layout {tag}/{dn}/{lay}: {e.GetType().Name}"); continue; }
            foreach (var op in OPS)
            {
                string id = $"{tag}|{dn}|{lay}|{op}";
                try { var _ = Op(op, v); Row(id, BestMs(() => { var r = Op(op, v); }, it, wm, rd)); }
                catch (Exception e) { Console.Error.WriteLine($"{id}: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); }
            }
        }
    }
}
Console.Error.WriteLine("[elementwise_layout_bench] done");
