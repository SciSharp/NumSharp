#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// =============================================================================
// copy_path_bench.cs — NumSharp side. For each dtype × layout × size measures:
//   pos  = np.positive(v)  → the NpyIter ufunc path (identity = copy via iterator)
//   copy = v.copy()        → the current Storage.Clone path (legacy)
// Companion: copy_path_bench.py (NumPy np.positive baseline, identical keys).
// Answers "how does the NpyIter copy path compare to NumPy across all
// dtypes/layout variations" — and how much the routing would change vs today.
// Run ONLY with:  dotnet run -c Release - < benchmark/poc/copy_path_bench.cs
// =============================================================================
using System.Diagnostics;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(DebuggableAttribute)) as DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core"); return; }

double Best(Action f, int it, int wm, int rd)
{
    for (int i = 0; i < wm; i++) f();
    double b = 1e9;
    for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); }
    return b;
}
void Row(string id, double ms) => Console.WriteLine($"{id}\t{ms:G17}");
(int it, int wm, int rd) Pick(int n) => n <= 100_000 ? (200, 30, 3) : (40, 8, 3);

var SIZES = new (string tag, int R, int C)[] { ("100K", 316, 316), ("1M", 1000, 1000) };
var DTYPES = new (string name, NPTypeCode tc)[]
{
    ("bool", NPTypeCode.Boolean), ("u8", NPTypeCode.Byte), ("i8", NPTypeCode.SByte),
    ("i16", NPTypeCode.Int16), ("u16", NPTypeCode.UInt16), ("i32", NPTypeCode.Int32),
    ("u32", NPTypeCode.UInt32), ("i64", NPTypeCode.Int64), ("u64", NPTypeCode.UInt64),
    ("char", NPTypeCode.Char), ("f16", NPTypeCode.Half), ("f32", NPTypeCode.Single),
    ("f64", NPTypeCode.Double), ("dec", NPTypeCode.Decimal), ("c128", NPTypeCode.Complex),
};
string[] LAYOUTS = { "C", "F", "T", "strided", "sliced", "negstride", "bcast" };

NDArray Layout(NDArray b, string l) => l switch
{
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "strided" => b[":, ::2"], "sliced" => b["1:" + (b.shape[0]-1) + ", 1:" + (b.shape[1]-1)],
    "negstride" => b["::-1, :"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape((int)b.shape[0], (int)b.shape[1])),
    _ => throw new Exception(l),
};

Console.Error.WriteLine($"[copy_path_bench] cores={Environment.ProcessorCount}");
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
            try { var _ = np.positive(v); Row($"{tag}|{dn}|{lay}|pos", Best(() => { var r = np.positive(v); }, it, wm, rd)); }
            catch (Exception e) { Console.Error.WriteLine($"pos {tag}/{dn}/{lay}: {e.GetType().Name}"); }
            try { var _ = v.copy(); Row($"{tag}|{dn}|{lay}|copy", Best(() => { var r = v.copy(); }, it, wm, rd)); }
            catch (Exception e) { Console.Error.WriteLine($"copy {tag}/{dn}/{lay}: {e.GetType().Name}"); }
        }
    }
}
Console.Error.WriteLine("[copy_path_bench] done");
