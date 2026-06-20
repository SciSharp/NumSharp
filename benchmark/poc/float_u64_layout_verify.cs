#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Verify {f32,f64,c128} -> u64 astype is BIT-EXACT with NumPy across 8 layouts after the bucket-B
// negcol-reverse / strided-deinterleave kernels. Run: dotnet run -c Release - < this
using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using NumSharp;

const int R = 130, C = 130;
const string DIR = @"K:\source\NumSharp\benchmark\poc\_xref";
var refLines = File.ReadAllLines(Path.Combine(DIR, "float_u64_layout.tsv"))
    .Where(l => l.Length > 0).ToDictionary(l => l.Split('\t')[0], l => l.Split('\t')[1]);

NDArray Lay(NDArray b, string l) => l switch
{
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "sliced" => b[$"1:{R - 1}, 1:{C - 1}"], "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"],
    "strided" => b[":, ::2"], "bcast" => np.broadcast_to(b["0:1, :"], new Shape(R, C)),
    _ => throw new Exception(l)
};
string Sha(NDArray a)
{
    var c = a.astype(NPTypeCode.UInt64, copy: true);
    var flat = c.flat; int n = (int)c.size; byte[] bytes = new byte[n * 8];
    for (int i = 0; i < n; i++) BitConverter.GetBytes(Convert.ToUInt64(flat.GetAtIndex(i))).CopyTo(bytes, i * 8);
    return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
}

// rebuild the same base as the python ref
var flatArr = new double[R * C];
for (int i = 0; i < flatArr.Length; i++) flatArr[i] = i * 7.3 - 3000.0;
double[] specials = { double.PositiveInfinity, double.NegativeInfinity, double.NaN, 1e20, -1e20, 2e19, 9.3e18, -5.0, 0.0 };
for (int k = 0; k < specials.Length; k++) flatArr[(k * 911 + 13) % flatArr.Length] = specials[k];
var base64 = np.array(flatArr).reshape(R, C);
var base32 = base64.astype(NPTypeCode.Single);
var basec = base64.astype(NPTypeCode.Complex);

string[] lays = { "C", "F", "T", "sliced", "negrow", "negcol", "strided", "bcast" };
int pass = 0, fail = 0;
foreach (var (tag, ba) in new[] { ("f32", base32), ("f64", base64), ("c128", basec) })
    foreach (var l in lays)
    {
        string key = $"{tag}|{l}", got = Sha(Lay(ba, l));
        bool ok = refLines.TryGetValue(key, out var want) && want == got;
        if (ok) pass++; else { fail++; Console.WriteLine($"  MISMATCH {key}: got {got[..12]} want {(want ?? "??")[..12]}"); }
    }
Console.WriteLine($"{{f32,f64,c128}} -> u64 layouts: {pass} pass, {fail} fail  {(fail == 0 ? "ALL BIT-EXACT OK" : "FAILED")}");
