#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// operand_bench.cs — NumSharp side of the benchmark/operand subsystem: the layout
// classes the per-operand layout grid (benchmark/layout) can't express. Companion:
// operand_bench.py (identical keys). Driven + rendered by operand_sheet.py.
//   1-D contiguous/strided/reversed · scalar operand (lhs/rhs) · mixed operand
//   layouts (C+F, C+T) · binary broadcast (row + col) · column-broadcast unary.
// Key: {case}|{dt}\t{ms}.  Run: dotnet run -c Release - < benchmark/operand/operand_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core — use dotnet run -c Release - < file"); return; }

double Best(Action f, int it, int wm, int rd)
{
    for (int i = 0; i < wm; i++) f();
    double b = double.MaxValue;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");

const int N1 = 1_000_000, R = 1000, C = 1000;   // 1-D 1M ; 2-D 1M
const int it = 30, wm = 6, rd = 3;
var DTYPES = new (string name, NPTypeCode tc)[]
{
    ("f64", NPTypeCode.Double), ("f32", NPTypeCode.Single), ("f16", NPTypeCode.Half),
    ("i32", NPTypeCode.Int32), ("i64", NPTypeCode.Int64), ("c128", NPTypeCode.Complex),
};

Console.Error.WriteLine($"[operand_bench] cores={Environment.ProcessorCount}");
foreach (var (dn, tc) in DTYPES)
{
    NDArray a1, a1s, a1r, a2, a2F, a2T, row, col, colb, sc;
    try
    {
        a1 = ((np.arange(N1) % 17) + 1).astype(tc);
        a1s = a1["::2"]; a1r = a1["::-1"];
        a2 = ((np.arange(R * C) % 17) + 1).astype(tc).reshape(R, C);
        a2F = a2.copy(order: 'F');
        a2T = a2.T;                                   // (C,R) F-contig view; square so same shape
        row = a2["0:1, :"];                           // (1,C) → binary row-broadcast
        col = a2[":, 0:1"];                           // (R,1) → binary col-broadcast
        colb = np.broadcast_to(col, new Shape(R, C)); // (R,C) inner stride-0 → unary on col-broadcast
        sc = NDArray.Scalar(2, tc);                   // 0-d scalar operand
    }
    catch (Exception e) { Console.Error.WriteLine($"build {dn}: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); continue; }

    void Case(string key, Action body)
    {
        try { body(); Row($"{key}|{dn}", Best(body, it, wm, rd)); }
        catch (Exception e) { Console.Error.WriteLine($"{key}|{dn}: {e.GetType().Name}: {e.Message.Split('\n')[0]}"); }
    }

    Case("1d_C",           () => { var _ = a1 + a1; });
    Case("1d_strided",     () => { var _ = a1s + a1s; });
    Case("1d_rev",         () => { var _ = a1r + a1r; });
    Case("scalar_rhs",     () => { var _ = a2 + sc; });
    Case("scalar_lhs",     () => { var _ = sc + a2; });
    Case("mix_C_F",        () => { var _ = a2 + a2F; });
    Case("mix_C_T",        () => { var _ = a2 + a2T; });
    Case("bcast_row",      () => { var _ = a2 + row; });
    Case("bcast_col",      () => { var _ = a2 + col; });
    Case("colbcast_unary", () => { var _ = np.positive(colb); });
}
Console.Error.WriteLine("[operand_bench] done");
