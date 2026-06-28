using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;
using Half = System.Half;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD Half → int cast (<c>TryGetHalfToXKernel</c> / <c>TryGetHalfToXStridedKernel</c>):
/// Giesen branchless half→float bit-fiddle widen (no F16C in this .NET) + cvttps2dq + truncating
/// <c>Vector.Narrow</c>. Must be BIT-EXACT with <see cref="Converts"/>.To{X}(Half), hence NumPy 2.4.2
/// (NaN/inf → INT_MIN → low-bits wrap for narrow targets).
/// </summary>
[TestClass]
public class HalfToIntCastParityTests
{
    private static NDArray HalfArray(params float[] vs)
    {
        var h = new Half[vs.Length];
        for (int i = 0; i < vs.Length; i++) h[i] = (Half)vs[i];
        return np.array(h);
    }

    // NumPy 2.4.2: list(np.array([...],np.float16).astype(dt))
    [TestMethod]
    public void Half_To_Int32_MatchesNumPy()
    {
        var src = HalfArray(1.5f, 300f, 128.5f, 255.5f, -300f, 65504f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0.0006f);
        var expect = new int[] { 1, 300, 128, 255, -300, 65504, int.MinValue, int.MinValue, int.MinValue, 0 };
        var got = src.astype(NPTypeCode.Int32);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetInt32(i), $"f16->i32[{i}]");
    }

    [TestMethod]
    public void Half_To_SByte_Wrap_MatchesNumPy()
    {
        var src = HalfArray(1.5f, 300f, 128.5f, 255.5f, -300f, 65504f, float.NaN, float.PositiveInfinity);
        var expect = new sbyte[] { 1, 44, -128, -1, -44, -32, 0, 0 };
        var got = src.astype(NPTypeCode.SByte);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetSByte(i), $"f16->i8[{i}]");
    }

    [TestMethod]
    public void Half_To_Int16_Wrap_MatchesNumPy()
    {
        var src = HalfArray(1.5f, 300f, 128.5f, 255.5f, -300f, 65504f, float.NaN);
        var expect = new short[] { 1, 300, 128, 255, -300, -32, 0 };
        var got = src.astype(NPTypeCode.Int16);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetInt16(i), $"f16->i16[{i}]");
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Byte)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.UInt16)]
    [DataRow(NPTypeCode.Char)]
    public void Half_To_Int_Contig_EqualsConverts(NPTypeCode dst)
    {
        const int N = 50_003;
        var rnd = new Random(37);
        float[] sp = { 0f, -0f, 1.5f, 300f, 128.5f, -300f, 65504f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 6e-5f };
        var data = new Half[N];
        for (int i = 0; i < N; i++) data[i] = rnd.Next(100) < 15 ? (Half)sp[rnd.Next(sp.Length)] : (Half)(float)(rnd.NextDouble() * 600 - 300);
        var got = np.array(data).astype(dst);
        for (int i = 0; i < N; i++)
            Assert.AreEqual(Convert(data[i], dst), System.Convert.ToInt64(got.GetAtIndex(i)), $"f16->{dst}[{i}]");
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Int16)]
    public void Half_To_Int_Strided_EqualsConverts(NPTypeCode dst)
    {
        const int Rows = 200, Cols = 301;
        var rnd = new Random(41);
        var data = new Half[Rows * Cols];
        for (int i = 0; i < data.Length; i++) data[i] = (Half)(float)(rnd.NextDouble() * 600 - 300);
        var mat = np.array(data).reshape(Rows, Cols);

        var got = mat[":, ::2"].astype(dst).flatten();
        int idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c += 2)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(got.GetAtIndex(idx++)), $"f16[:, ::2]->{dst} r{r}c{c}");

        var rev = mat[":, ::-1"].astype(dst).flatten();
        idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = Cols - 1; c >= 0; c--)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(rev.GetAtIndex(idx++)), $"f16[:, ::-1]->{dst} r{r}c{c}");
    }

    private static long Convert(Half v, NPTypeCode dst) => dst switch
    {
        NPTypeCode.Int32 => Converts.ToInt32(v),
        NPTypeCode.SByte => Converts.ToSByte(v),
        NPTypeCode.Byte => Converts.ToByte(v),
        NPTypeCode.Int16 => Converts.ToInt16(v),
        NPTypeCode.UInt16 => Converts.ToUInt16(v),
        NPTypeCode.Char => Converts.ToChar(v),
        _ => throw new ArgumentOutOfRangeException(nameof(dst)),
    };
}
