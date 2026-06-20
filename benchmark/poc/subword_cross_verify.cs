#:project K:/source/NumSharp/src/NumSharp.Core/NumSharp.Core.csproj
#:property AssemblyName=NumSharp.DotNetRunScript
#:property PublishAot=false
#:property AllowUnsafeBlocks=true
// Prove SubwordCopy's same-size cross-type byte copies are BIT-EXACT with NumPy across
// all 8 layouts: rebuild each (src,layout,dst), astype(dst), sha256 the contiguous result
// bytes, compare to subword_cross_npref.py's hashes.
//   First run: python benchmark/poc/subword_cross_npref.py
//   Then:      dotnet run -c Release - < benchmark/poc/subword_cross_verify.cs
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using NumSharp;

const int R = 130, C = 130;
var TC = new Dictionary<string, NPTypeCode> {
    {"bool",NPTypeCode.Boolean},{"u8",NPTypeCode.Byte},{"i8",NPTypeCode.SByte},
    {"i16",NPTypeCode.Int16},{"u16",NPTypeCode.UInt16},{"char",NPTypeCode.Char},{"f16",NPTypeCode.Half},
};
var refs = new Dictionary<string, string>();
foreach (var ln in System.IO.File.ReadAllLines(@"K:\source\NumSharp\benchmark\poc\_xref\np_hashes.tsv"))
{ var p = ln.Split('\t'); if (p.Length >= 2) refs[p[0]] = p[1]; }

NDArray Layout(NDArray b, string l) => l switch {
    "C" => b, "F" => b.copy(order: 'F'), "T" => b.T,
    "sliced" => b["1:" + (b.shape[0] - 1) + ", 1:" + (b.shape[1] - 1)],
    "negrow" => b["::-1, :"], "negcol" => b[":, ::-1"], "strided" => b[":, ::2"],
    "bcast" => np.broadcast_to(b["0:1, :"], new Shape((int)b.shape[0], (int)b.shape[1])),
    _ => throw new Exception(l),
};

unsafe string Hash(NDArray a)
{
    var c = a.Shape.IsContiguous ? a : a.copy();        // C-order contiguous bytes
    int esz = c.dtypesize; long bytes = (long)c.size * esz;
    var buf = new byte[bytes];
    System.Runtime.InteropServices.Marshal.Copy((IntPtr)((byte*)c.Storage.Address + (long)c.Shape.offset * esz), buf, 0, (int)bytes);
    return Convert.ToHexString(SHA256.HashData(buf)).ToLowerInvariant();
}

int ok = 0, fail = 0;
foreach (var key in refs.Keys)
{
    var p = key.Split('|'); var sn = p[0]; var lay = p[1]; var dn = p[2];
    var baseArr = ((np.arange(R * C) % 251) + 1).astype(TC[sn]).reshape(R, C);
    var v = Layout(baseArr, lay);
    var r = v.astype(TC[dn], copy: true);
    var h = Hash(r);
    if (h == refs[key]) ok++;
    else { fail++; Console.WriteLine($"  MISMATCH {key}: ns={h[..12]} np={refs[key][..12]}"); }
}
Console.WriteLine($"\n{ok}/{ok + fail} cross-cast hashes match NumPy" + (fail == 0 ? "  BIT-EXACT" : $"  {fail} FAILED"));
