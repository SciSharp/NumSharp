#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Best-of-7 NumSharp timing for the bucket-C cells: same-type 1-byte copies + i64->narrow,
// across 8 layouts at 1M. Run: dotnet run -c Release - < benchmark/poc/subword_arb_bench.cs
using System;
using System.Diagnostics;
using System.IO;
using NumSharp;

const int R = 1000, C = 1000, it = 30, wm = 8, rd = 7;
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
var TC = new System.Collections.Generic.Dictionary<string, NPTypeCode> {
    {"bool",NPTypeCode.Boolean},{"u8",NPTypeCode.Byte},{"i8",NPTypeCode.SByte},
    {"i16",NPTypeCode.Int16},{"u16",NPTypeCode.UInt16},{"i64",NPTypeCode.Int64}};
var pairs = new[] { ("i8","i8"),("u8","u8"),("bool","bool"),("i64","i8"),("i64","u8"),("i64","i16"),("i64","u16") };
var sb = new System.Text.StringBuilder();
foreach (var (s, d) in pairs)
{
    var ba = ((np.arange(R * C) % 17) + 1).astype(TC[s]).reshape(R, C);
    foreach (var l in lays)
    {
        var v = Lay(ba, l);
        var _ = v.astype(TC[d], copy: true);
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        double ns = Best(() => { var r = v.astype(TC[d], copy: true); });
        sb.AppendLine($"{s}|{l}|{d}\t{ns:F5}");
    }
    Console.WriteLine($"{s}->{d} done");
}
File.WriteAllText(Path.Combine(DIR, "subword_arb_ns.tsv"), sb.ToString());
Console.WriteLine("wrote ns tsv");
