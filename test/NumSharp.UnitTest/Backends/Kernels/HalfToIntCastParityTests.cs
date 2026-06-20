using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD float16 → {bool, i8, u8, i16, u16, char, i32} cast — EXHAUSTIVELY over all 65536
/// half bit patterns. f16 was a lagging source in Phase 0 (geomean 0.69×) because Half is excluded
/// from IsFloatCast, so every f16→X fell to the IL scalar (.NET 10 has no vectorized VCVTPH2PS).
///
/// The widen (<c>DirectILKernelGenerator.Cast.Half.cs</c>) is the branchless Giesen
/// "half_to_float_fast5" (shift + one float-multiply that rebiases normals AND subnormals, + one
/// compare for Inf/NaN). It is bit-exact with <c>(float)Half</c> for every non-NaN half, and any
/// NaN/Inf maps to INT_MIN under cvtt — so f16→int is bit-exact with <c>Converts.To{X}(Half)</c>
/// for ALL 65536 inputs (this test proves exactly that). f16→bool needs no float at all: a half is
/// truthy iff <c>(bits &amp; 0x7FFF) != 0</c> (±0→False, Inf/NaN/subnormal→True).
/// </summary>
[TestClass]
public class HalfToIntCastParityTests
{
    private static readonly Half[] AllHalves = BuildAll();
    private static Half[] BuildAll()
    {
        var a = new Half[65536];
        for (int i = 0; i < 65536; i++) a[i] = BitConverter.UInt16BitsToHalf((ushort)i);
        return a;
    }

    private static void Exhaustive(NPTypeCode dst, Func<Half, long> reference, string label)
    {
        var casted = np.array(AllHalves).astype(dst);
        Assert.AreEqual(65536, (int)casted.size, $"{label}: size");
        for (int i = 0; i < 65536; i++)
        {
            long got = Convert.ToInt64(casted.GetAtIndex(i));
            long want = reference(AllHalves[i]);
            if (got != want)
                Assert.Fail($"{label}: half 0x{i:X4} ({(float)AllHalves[i]}) expected {want}, got {got}");
        }
    }

    [TestMethod] public void HalfToBool_AllHalves() => Exhaustive(NPTypeCode.Boolean, h => Converts.ToBoolean(h) ? 1 : 0, "f16->bool");
    [TestMethod] public void HalfToSByte_AllHalves() => Exhaustive(NPTypeCode.SByte, h => Converts.ToSByte(h), "f16->i8");
    [TestMethod] public void HalfToByte_AllHalves() => Exhaustive(NPTypeCode.Byte, h => Converts.ToByte(h), "f16->u8");
    [TestMethod] public void HalfToInt16_AllHalves() => Exhaustive(NPTypeCode.Int16, h => Converts.ToInt16(h), "f16->i16");
    [TestMethod] public void HalfToUInt16_AllHalves() => Exhaustive(NPTypeCode.UInt16, h => Converts.ToUInt16(h), "f16->u16");
    [TestMethod] public void HalfToChar_AllHalves() => Exhaustive(NPTypeCode.Char, h => (ushort)Converts.ToChar(h), "f16->char");
    [TestMethod] public void HalfToInt32_AllHalves() => Exhaustive(NPTypeCode.Int32, h => Converts.ToInt32(h), "f16->i32");

    // Strided/non-contiguous halves must match the contiguous (NumPy-pinned) result exactly —
    // guards the widen front-end through StridedNarrowDriver (srcSize=2) on every layout.
    [TestMethod]
    public void AllLayouts_BitExact_VsContiguous()
    {
        var grid = np.array(AllHalves).reshape(256, 256);
        var dsts = new[] { NPTypeCode.Boolean, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.Char, NPTypeCode.Int32 };
        var layouts = new (string, Func<NDArray, NDArray>)[]
        {
            ("F",       g => g.copy(order: 'F')),
            ("T",       g => g.T),
            ("sliced",  g => g["1:255, 1:255"]),
            ("negrow",  g => g["::-1, :"]),
            ("negcol",  g => g[":, ::-1"]),
            ("strided", g => g[":, ::2"]),
        };
        foreach (var (lname, lf) in layouts)
        foreach (var dst in dsts)
        {
            var view = lf(grid);
            var viaView = view.astype(dst);
            var viaContig = view.copy().astype(dst);
            for (long i = 0; i < viaView.size; i++)
                Assert.AreEqual(Convert.ToInt64(viaContig.GetAtIndex((int)i)), Convert.ToInt64(viaView.GetAtIndex((int)i)),
                    $"f16|{lname}|{dst}: element {i} (strided != contiguous)");
        }
    }
}
