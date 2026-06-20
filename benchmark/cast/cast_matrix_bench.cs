#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// cast_matrix_bench.cs — NumSharp side. Phase 0 of CAST_BEAT_NUMPY_PLAN.md.
// Full-matrix cast discovery: for every src dtype × layout × dst dtype at 1M,
// times v.astype(dst, copy:true) — the astype → DefaultEngine.Cast → NpyIter.Copy
// path. astype(copy:true, order:'K') is NumPy-faithful (forces a real cast/copy
// even on the src==dst diagonal, so the same-type-copy path is regression-checked
// in the same sweep).
//   Output key:  1M|{src}|{layout}|{dst}\t{ms}
//   Companion:   cast_matrix_bench.py (NumPy astype baseline, identical keys).
//   Merge:       cast_matrix_merge.py  →  cast_matrix.md
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/cast_matrix_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core — rerun with -c Release"); return; }

double Best(Action f, int it, int wm, int rd)
{
    for (int i = 0; i < wm; i++) f();
    double b = 1e9;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");

const int R = 1000, C = 1000;       // 1M elements
const int it = 20, wm = 5, rd = 3;  // discovery sweep: best-of-3, plenty to rank cells

var DTYPES = new (string name, NPTypeCode tc)[]
{
    ("bool", NPTypeCode.Boolean), ("u8", NPTypeCode.Byte), ("i8", NPTypeCode.SByte),
    ("i16", NPTypeCode.Int16), ("u16", NPTypeCode.UInt16), ("i32", NPTypeCode.Int32),
    ("u32", NPTypeCode.UInt32), ("i64", NPTypeCode.Int64), ("u64", NPTypeCode.UInt64),
    ("char", NPTypeCode.Char), ("f16", NPTypeCode.Half), ("f32", NPTypeCode.Single),
    ("f64", NPTypeCode.Double), ("dec", NPTypeCode.Decimal), ("c128", NPTypeCode.Complex),
};
// 8 layouts: C, F-contig, transpose, offset-slice (stride-1 inner), neg-outer
// (stride-1 inner), neg-inner (reversed inner), strided-inner [:, ::2] (no
// stride-1 axis), broadcast (stride-0). DST of a cast is always fresh-contig.
string[] LAYOUTS = { "C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast" };

NDArray Layout(NDArray b, string l) => l switch
{
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "sliced" => b["1:" + (b.shape[0] - 1) + ", 1:" + (b.shape[1] - 1)],
    "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"],
    "strided" => b[":, ::2"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape((int)b.shape[0], (int)b.shape[1])),
    _ => throw new Exception(l),
};

Console.Error.WriteLine($"[cast_matrix_bench] cores={Environment.ProcessorCount}");
int emitted = 0;
foreach (var (sn, stc) in DTYPES)
{
    NDArray baseArr;
    try { baseArr = ((np.arange(R * C) % 17) + 1).astype(stc).reshape(R, C); }
    catch (Exception e) { Console.Error.WriteLine($"build {sn}: {e.GetType().Name}: {e.Message}"); continue; }
    foreach (var lay in LAYOUTS)
    {
        NDArray v;
        try { v = Layout(baseArr, lay); }
        catch (Exception e) { Console.Error.WriteLine($"layout {sn}/{lay}: {e.GetType().Name}"); continue; }
        foreach (var (dn, dtc) in DTYPES)
        {
            try
            {
                var probe = v.astype(dtc, copy: true);          // correctness/throw gate + warm path
                if (probe.size != v.size) { Console.Error.WriteLine($"SIZE {sn}/{lay}/{dn}: {probe.size}!={v.size}"); continue; }
                Row($"1M|{sn}|{lay}|{dn}", Best(() => { var r = v.astype(dtc, copy: true); }, it, wm, rd));
                emitted++;
            }
            catch (Exception e) { Console.Error.WriteLine($"cast {sn}/{lay}/{dn}: {e.GetType().Name}: {e.Message}"); }
        }
    }
}
Console.Error.WriteLine($"[cast_matrix_bench] done — {emitted} rows");
