#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Best-of-7 NumSharp timing for i64/u64 -> f16 across 8 layouts at 1M. Writes ns_ms to a tsv;
// the python twin measures NumPy and prints NPY/NS ratios.
//   Run: dotnet run -c Release - < benchmark/poc/i64u64_half_bench.cs
using System;
using System.Diagnostics;
using System.IO;
using NumSharp;

const int R = 1000, C = 1000, it = 25, wm = 8, rd = 7;
const string DIR = @"K:\source\NumSharp\benchmark\poc\_xref";

double Best(Action f) { for (int i = 0; i < wm; i++) f(); double b = 1e9; for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); } return b; }
NDArray Lay(NDArray b, string l) => l switch
{
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "sliced" => b[$"1:{R - 1}, 1:{C - 1}"], "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"],
    "strided" => b[":, ::2"], "bcast" => np.broadcast_to(b["0:1, :"], new Shape(R, C)),
    _ => throw new Exception(l)
};
string[] lays = { "C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast" };
var sb = new System.Text.StringBuilder();
foreach (var (tag, tc) in new[] { ("i64", NPTypeCode.Int64), ("u64", NPTypeCode.UInt64) })
{
    var ba = ((np.arange(R * C) % 17) + 1).astype(tc).reshape(R, C);
    foreach (var l in lays)
    {
        var v = Lay(ba, l);
        var _ = v.astype(NPTypeCode.Half, copy: true);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        double ns = Best(() => { var r = v.astype(NPTypeCode.Half, copy: true); });
        sb.AppendLine($"{tag}|{l}\t{ns:F5}");
        Console.WriteLine($"{tag}|{l,-8}\tns={ns:F5}ms");
    }
}
File.WriteAllText(Path.Combine(DIR, "i64u64_half_ns.tsv"), sb.ToString());
Console.WriteLine("wrote ns tsv");
