#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Integrated correctness: v.astype(sameType, copy:true) must reproduce the view's
// logical values EXACTLY for every sub-word dtype across all 8 cast-matrix layouts.
// The view indexer (v[i,j]) is independent code from the copy kernel, so a wrong-element
// deinterleave/reverse bug fails here.
//   Run: dotnet run -c Release - < benchmark/poc/subword_verify.cs
using System;
using NumSharp;

const int R = 130, C = 130;   // even width so [:, ::2] is well-defined; small for exhaustive compare
var DT = new (string n, NPTypeCode tc)[] {
    ("bool", NPTypeCode.Boolean), ("u8", NPTypeCode.Byte), ("i8", NPTypeCode.SByte),
    ("i16", NPTypeCode.Int16), ("u16", NPTypeCode.UInt16), ("char", NPTypeCode.Char), ("f16", NPTypeCode.Half),
};
string[] LAY = { "C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast" };

NDArray Layout(NDArray b, string l) => l switch {
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "sliced" => b["1:" + (b.shape[0] - 1) + ", 1:" + (b.shape[1] - 1)],
    "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"], "strided" => b[":, ::2"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape((int)b.shape[0], (int)b.shape[1])),
    _ => throw new Exception(l),
};

int fails = 0, checks = 0;
foreach (var (n, tc) in DT)
{
    var baseArr = ((np.arange(R * C) % 251) + 1).astype(tc).reshape(R, C);
    foreach (var lay in LAY)
    {
        NDArray v = Layout(baseArr, lay);
        NDArray r = v.astype(tc, copy: true);
        // compare r (fresh contig) against v element-wise via independent indexer
        long bad = 0; long total = v.size;
        var vf = v.flat; var rf = r.flat;
        if (r.size != v.size) { Console.WriteLine($"  SIZE {n}/{lay}: {r.size}!={v.size}"); fails++; continue; }
        for (long i = 0; i < total; i++)
        {
            var a = vf.GetValue(i); var b = rf.GetValue(i);
            if (!a.Equals(b)) { bad++; if (bad <= 2) Console.WriteLine($"  DIFF {n}/{lay} @ {i}: view={a} copy={b}"); }
        }
        checks++;
        if (bad != 0) { Console.WriteLine($"  FAIL {n}/{lay}: {bad}/{total} differ"); fails++; }
    }
}
Console.WriteLine($"\n{checks - fails}/{checks} layout×dtype checks OK" + (fails == 0 ? "  ALL GOOD" : $"  {fails} FAILED"));
