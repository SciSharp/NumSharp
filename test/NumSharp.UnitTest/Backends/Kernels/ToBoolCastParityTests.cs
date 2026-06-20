using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD {int,float} → Boolean cast to NumPy 2.4.2 (== <c>x != 0</c>).
///
/// Phase-0 found <c>*→bool</c> was the worst dst column (geomean 0.61×) — ResolveStrategy excludes
/// Boolean from IsIntegerCast, so every cast-to-bool fell to the per-element IL scalar while NumPy
/// vectorizes the compare. The fix (<c>DirectILKernelGenerator.Cast.ToBool.cs</c>) is
/// <c>~Vector.Equals(v, 0) &amp; 1 → truncating Vector.Narrow → byte</c>.
///
/// The float semantics are the subtle part and are pinned here: floats compare in IEEE float/double,
/// so <c>NaN → True</c> (NaN != 0) and <c>-0.0 → False</c> (-0.0 == 0.0) — exactly NumPy. A naive
/// integer bit-test on the float would get -0.0 wrong (its bits are 0x8000_0000 ≠ 0).
/// </summary>
[TestClass]
public class ToBoolCastParityTests
{
    private static bool B(NDArray a, int i) => (bool)a.GetAtIndex(i);

    [TestMethod]
    public void FloatEdges_MatchNumPy()
    {
        // NumPy: np.array([...], dtype).astype(bool) — NaN/±inf/subnormal are truthy, ±0 falsy.
        float[] fe = { 0f, -0f, 1f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity,
                       float.Epsilon, 1e-30f, 123.5f, -0.0001f };
        bool[] want = { false, false, true, true, true, true, true, true, true, true, true };

        var f32 = np.array(fe).astype(NPTypeCode.Boolean);
        var f64 = np.array(fe.Select(x => (double)x).ToArray()).astype(NPTypeCode.Boolean);
        for (int i = 0; i < fe.Length; i++)
        {
            Assert.AreEqual(want[i], B(f32, i), $"f32->bool: {fe[i]} expected {want[i]}");
            Assert.AreEqual(want[i], B(f64, i), $"f64->bool: {fe[i]} expected {want[i]}");
        }
    }

    [TestMethod]
    public void IntEdges_MatchNumPy()
    {
        // 0 -> False, any nonzero (incl. MinValue / -1) -> True.
        long[] vals = { 0, 1, -1, 2, long.MinValue, long.MaxValue, 256, -256 };
        bool[] want = { false, true, true, true, true, true, true, true };

        var i64 = np.array(vals).astype(NPTypeCode.Boolean);
        var i32 = np.array(vals.Select(v => (int)v).ToArray()).astype(NPTypeCode.Boolean);
        var i16 = np.array(vals.Select(v => (short)v).ToArray()).astype(NPTypeCode.Boolean);
        for (int i = 0; i < vals.Length; i++)
        {
            Assert.AreEqual(want[i], B(i64, i), $"i64->bool: {vals[i]}");
            // i32/i16 truncate the value but 0 stays 0 and the chosen nonzeros stay nonzero.
            Assert.AreEqual((int)vals[i] != 0, B(i32, i), $"i32->bool: {(int)vals[i]}");
            Assert.AreEqual((short)vals[i] != 0, B(i16, i), $"i16->bool: {(short)vals[i]}");
        }
    }

    // The SIMD bulk (32-wide), the narrow chain, the strided staging, and the scalar tail must all
    // agree across every layout — guards the i16/u16/char strided-staging width bug (a 2-byte source
    // must not be copied through the 8-byte staging path).
    [TestMethod]
    public void AllSrc_AllLayouts_BitExact_VsContiguous()
    {
        int R = 48, C = 48;
        var seed = new int[R * C];
        for (int i = 0; i < R * C; i++) seed[i] = (i % 5 == 0) ? 0 : (i - (R * C / 2));   // ~20% zeros
        var baseI = np.array(seed).reshape(R, C);

        var srcs = new[]
        {
            NPTypeCode.Byte, NPTypeCode.SByte, NPTypeCode.Int16, NPTypeCode.UInt16, NPTypeCode.Char,
            NPTypeCode.Int32, NPTypeCode.UInt32, NPTypeCode.Int64, NPTypeCode.UInt64,
            NPTypeCode.Single, NPTypeCode.Double
        };
        var layouts = new (string, Func<NDArray, NDArray>)[]
        {
            ("F",       g => g.copy(order: 'F')),
            ("T",       g => g.T),
            ("sliced",  g => g["1:" + (R - 1) + ", 1:" + (C - 1)]),
            ("negrow",  g => g["::-1, :"]),
            ("negcol",  g => g[":, ::-1"]),
            ("strided", g => g[":, ::2"]),
        };

        foreach (var stc in srcs)
        {
            var grid = baseI.astype(stc);
            foreach (var (lname, lf) in layouts)
            {
                var view = lf(grid);
                var viaView = view.astype(NPTypeCode.Boolean);
                var viaContig = view.copy().astype(NPTypeCode.Boolean);
                Assert.AreEqual(viaContig.size, viaView.size, $"{stc}|{lname}: size");
                for (long i = 0; i < viaView.size; i++)
                    Assert.AreEqual(B(viaContig, (int)i), B(viaView, (int)i),
                        $"{stc}|{lname}: element {i} (strided != contiguous)");
            }
        }
    }
}
