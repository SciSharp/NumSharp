#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// reduce_layout_bench.cs — NumSharp side of the reduction x LAYOUT x dtype x op
// parity matrix. Companion: reduce_layout_bench.py (identical keys). Merge with
// reduce_layout_merge.py.
//
// COVERAGE GAP THIS FILLS: prior reduction benches only measured contiguous (and
// complex C/T). Reductions over OFFSET / NEGATIVE-STRIDE / SLICED views were
// broken (NpyIter op_axes ignored Shape.offset) until the fix, so were never
// benchmarked. This sweeps every layout x the NpyIter-routed dtypes x ops.
//
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/reduce_layout_bench.cs
// =============================================================================
using System.Diagnostics;
using System.Runtime.CompilerServices;
using NumSharp;

var dbgCore = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbgCore?.IsJITOptimizerDisabled ?? false)
{
    Console.Error.WriteLine("FATAL: Debug-JITted NumSharp.Core — numbers INVALID. Use: dotnet run -c Release - < file");
    return;
}

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
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");

(int iters, int warm, int rounds) Pick(int n) =>
    n <= 100_000 ? (200, 30, 3) : (15, 4, 2);   // 100K, 1M

var SIZES = new (string tag, int R, int C)[] { ("100K", 316, 316), ("1M", 1000, 1000) };
var DTYPES = new (string name, NPTypeCode tc)[]
{
    ("f64", NPTypeCode.Double), ("f32", NPTypeCode.Single), ("c128", NPTypeCode.Complex),
    ("dec", NPTypeCode.Decimal), ("f16", NPTypeCode.Half), ("i32", NPTypeCode.Int32), ("i64", NPTypeCode.Int64),
};
string[] OPS = { "sum", "min", "max", "prod" };
string[] LAYOUTS = { "C", "F", "T", "strided", "negstride", "sliced" };

NDArray Layout(NDArray baseArr, string layout) => layout switch
{
    "C" => baseArr,
    "F" => baseArr.copy(order: 'F'),
    "T" => baseArr.T,
    "strided" => baseArr[":, ::2"],
    "negstride" => baseArr["::-1, :"],
    "sliced" => baseArr["1:" + (baseArr.shape[0] - 1) + ", 1:" + (baseArr.shape[1] - 1)],
    _ => throw new Exception(layout),
};
NDArray Reduce(string op, NDArray a, int axis) => op switch
{
    "sum" => np.sum(a, axis), "min" => np.amin(a, axis), "max" => np.amax(a, axis), "prod" => np.prod(a, axis),
    _ => throw new Exception(op),
};

Console.Error.WriteLine($"[reduce_layout_bench] cores={Environment.ProcessorCount}");

foreach (var (tag, R, C) in SIZES)
{
    var (iters, warm, rounds) = Pick(R * C);
    foreach (var (dname, tc) in DTYPES)
    {
        // build base once per (size,dtype); modest magnitudes keep prod finite-ish (timing is the point)
        NDArray baseArr;
        try { baseArr = ((np.arange(R * C) % 17) + 1).astype(tc).reshape(R, C); }
        catch (Exception e) { Console.Error.WriteLine($"build {tag}/{dname}: {e.GetType().Name}"); continue; }

        foreach (var layout in LAYOUTS)
        {
            NDArray v;
            try { v = Layout(baseArr, layout); }
            catch (Exception e) { Console.Error.WriteLine($"layout {tag}/{dname}/{layout}: {e.GetType().Name}"); continue; }

            foreach (var op in OPS)
            {
                for (int axis = 0; axis <= 1; axis++)
                {
                    string id = $"{tag}|{dname}|{layout}|{op}|ax{axis}";
                    try
                    {
                        var _ = Reduce(op, v, axis);             // warm correctness/JIT
                        Row(id, BestMs(() => { var r = Reduce(op, v, axis); }, iters, warm, rounds));
                    }
                    catch (Exception e) { Console.Error.WriteLine($"{id}: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); }
                }
            }
        }
    }
}
Console.Error.WriteLine("[reduce_layout_bench] done");
