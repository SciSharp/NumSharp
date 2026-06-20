#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Verify NumSharp i64/u64 -> f16 astype is BIT-EXACT with NumPy across all 8 layouts.
//   Run: dotnet run -c Release - < benchmark/poc/i64u64_half_layout_verify.cs
using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NumSharp;

const int R = 130, C = 130;
const string DIR = @"K:\source\NumSharp\benchmark\poc\_xref";
var refLines = File.ReadAllLines(Path.Combine(DIR, "i64u64_half_layout.tsv"))
    .Where(l => l.Length > 0)
    .ToDictionary(l => l.Split('\t')[0], l => l.Split('\t')[1]);

NDArray Lay(NDArray b, string l) => l switch
{
    "C" => b,
    "F" => b.copy(order: 'F'),
    "T" => b.T,
    "sliced" => b[$"1:{R - 1}, 1:{C - 1}"],
    "negrow" => b["::-1, :"],
    "negcol" => b[":, ::-1"],
    "strided" => b[":, ::2"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape(R, C)),
    _ => throw new Exception(l)
};

string Sha(NDArray a)
{
    var c = a.astype(NPTypeCode.Half, copy: true);
    // contiguous C-order bytes of the logical result
    var flat = c.flat;
    int n = (int)c.size;
    byte[] bytes = new byte[n * 2];
    for (int i = 0; i < n; i++)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)flat.GetAtIndex(i));
        bytes[i * 2] = (byte)(bits & 0xFF);
        bytes[i * 2 + 1] = (byte)(bits >> 8);
    }
    return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}

string[] lays = { "C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast" };
int pass = 0, fail = 0;

// i64 and u64 base arrays mirroring the python ref
var baseI = (np.arange(R * C) * 991 - 200000).astype(NPTypeCode.Int64).reshape(R, C);
var baseU = (np.arange(R * C) * 991).astype(NPTypeCode.UInt64).reshape(R, C);

foreach (var (tag, ba) in new[] { ("i64", baseI), ("u64", baseU) })
{
    foreach (var l in lays)
    {
        string key = $"{tag}|{l}";
        string got = Sha(Lay(ba, l));
        bool ok = refLines.TryGetValue(key, out var want) && want == got;
        if (ok) pass++; else { fail++; Console.WriteLine($"  MISMATCH {key}: got {got[..12]} want {(want ?? "??")[..12]}"); }
    }
}
Console.WriteLine($"i64/u64 -> f16 layouts: {pass} pass, {fail} fail  {(fail == 0 ? "ALL BIT-EXACT OK" : "FAILED")}");
