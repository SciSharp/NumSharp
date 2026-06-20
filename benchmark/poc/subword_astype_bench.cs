#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Integrated astype best-of-7 for the same-type sub-word strided/negcol cells the
// SubwordCopy kernel now handles (includes the real output allocation). Reports ms +
// ratio vs the NumPy best-of-3 baseline from cast_results.tsv (>1.0 = NumSharp faster).
//   Run: dotnet run -c Release - < benchmark/poc/subword_astype_bench.cs
using System;
using System.Diagnostics;
using NumSharp;

var dbg = Attribute.GetCustomAttribute(typeof(np).Assembly, typeof(System.Diagnostics.DebuggableAttribute)) as System.Diagnostics.DebuggableAttribute;
if (dbg?.IsJITOptimizerDisabled ?? false) { Console.Error.WriteLine("FATAL: Debug core — rerun -c Release"); return; }

const int R = 1000, C = 1000, it = 30, wm = 8, rd = 7;
double Best(Action f) { for (int i = 0; i < wm; i++) f(); double b = 1e9; for (int r = 0; r < rd; r++) { var sw = Stopwatch.StartNew(); for (int i = 0; i < it; i++) f(); b = Math.Min(b, sw.Elapsed.TotalMilliseconds / it); } return b; }

var DT = new (string n, NPTypeCode tc)[] {
    ("bool", NPTypeCode.Boolean), ("u8", NPTypeCode.Byte), ("i8", NPTypeCode.SByte),
    ("i16", NPTypeCode.Int16), ("u16", NPTypeCode.UInt16), ("char", NPTypeCode.Char), ("f16", NPTypeCode.Half),
};
// NumPy baselines (ms) per cell key (best-of-3 sweep)
var NP = new System.Collections.Generic.Dictionary<string, double> {
    {"bool|strided",0.0936},{"u8|strided",0.0948},{"i8|strided",0.0927},{"i16|strided",0.1071},
    {"u16|strided",0.1068},{"char|strided",0.1081},{"f16|strided",0.1100},
    {"bool|negcol",0.3495},{"u8|negcol",0.2379},{"i8|negcol",0.2212},{"i16|negcol",0.4177},
    {"u16|negcol",0.4251},{"char|negcol",0.4206},{"f16|negcol",0.4126},
    {"i8|C",0.0528},{"i8|F",0.0546},{"i8|T",0.0542},
};

NDArray Layout(NDArray b, string l) => l switch {
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "negcol" => b[":, ::-1"], "strided" => b[":, ::2"], _ => throw new Exception(l),
};

Console.WriteLine($"{"cell",-16}{"ns_ms",10}{"np_ms",10}{"ratio",9}  (>1 = NS faster)");
foreach (var lay in new[] { "strided", "negcol", "C", "F", "T" })
{
    foreach (var (n, tc) in DT)
    {
        var key = $"{n}|{lay}";
        if (!NP.ContainsKey(key)) continue;
        var baseArr = ((np.arange(R * C) % 251) + 1).astype(tc).reshape(R, C);
        var v = Layout(baseArr, lay);
        var _ = v.astype(tc, copy: true);
        double ns = Best(() => { var r = v.astype(tc, copy: true); });
        double npv = NP[key]; double ratio = npv / ns;
        string icon = ratio >= 1.0 ? "OK " : ratio >= 0.9 ? "~  " : "LAG";
        Console.WriteLine($"{key,-16}{ns,10:F4}{npv,10:F4}{ratio,9:F3}  {icon}");
    }
}
