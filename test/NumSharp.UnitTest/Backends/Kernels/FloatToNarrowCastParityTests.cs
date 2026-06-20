using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD float|double → narrow-int cast (<c>TryGetFloatToNarrowIntKernel</c> /
/// <c>TryGetFloatToNarrowIntStridedKernel</c>): cvtt + truncating <c>Vector.Narrow</c>.
///
/// Phase-0's worst cells (f32→i8 was 0.09× NumPy). The kernel must be BIT-EXACT with the
/// <see cref="Converts"/> reference (hence NumPy 2.4.2), including the WRAP (not saturate)
/// semantics: NumPy float→narrow-int truncates toward zero to int32 (INT_MIN sentinel on
/// NaN/overflow) then takes the low bytes — e.g. f32→i8 of 128.5 → -128 (NOT +127),
/// 300 → 44, NaN → 0.
/// </summary>
[TestClass]
public class FloatToNarrowCastParityTests
{
    // NumPy 2.4.2 oracle: list(np.array([...],np.float32).astype(dt))
    [TestMethod]
    public void Single_To_SByte_Wrap_MatchesNumPy()
    {
        var src = np.array(new float[] { 300f, 128.5f, 255.5f, 256.7f, -129.5f, -300f, 1e9f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 127.4f, -128.6f });
        var expect = new sbyte[] { 44, -128, -1, 0, 127, -44, 0, 0, 0, 0, 127, -128 };
        var got = src.astype(NPTypeCode.SByte);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetSByte(i), $"f32->i8[{i}] ({src.GetSingle(i)})");
    }

    [TestMethod]
    public void Single_To_Byte_Wrap_MatchesNumPy()
    {
        var src = np.array(new float[] { 300f, 128.5f, 255.5f, 256.7f, -129.5f, -300f, float.NaN, -1.5f });
        var expect = new byte[] { 44, 128, 255, 0, 127, 212, 0, 255 };
        var got = src.astype(NPTypeCode.Byte);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetByte(i), $"f32->u8[{i}]");
    }

    [TestMethod]
    public void Single_To_Int16_Wrap_MatchesNumPy()
    {
        var src = np.array(new float[] { 32768.9f, 65535.9f, -300f, 256.7f, float.NaN, 1e9f });
        var expect = new short[] { -32768, -1, -300, 256, 0, -13824 };
        var got = src.astype(NPTypeCode.Int16);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetInt16(i), $"f32->i16[{i}]");
    }

    [TestMethod]
    public void Double_To_Int16_Wrap_MatchesNumPy()
    {
        // 3e9 doesn't fit i32 -> cvttpd2dq INT_MIN -> low16 = 0; 100000 -> 100000 mod 65536 = 34464
        var src = np.array(new double[] { 3e9, 100000.0, -100000.0, double.NaN, double.PositiveInfinity, 256.7 });
        var expect = new short[] { 0, unchecked((short)34464), unchecked((short)(-34464)), 0, 0, 256 };
        var got = src.astype(NPTypeCode.Int16);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetInt16(i), $"f64->i16[{i}]");
    }

    // Random + edge values: SIMD kernel == scalar Converts reference, all narrow targets,
    // contiguous AND strided, odd length to exercise the scalar tail.
    [DataTestMethod]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Byte)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.UInt16)]
    [DataRow(NPTypeCode.Char)]
    public void Single_To_Narrow_Contig_EqualsConverts(NPTypeCode dst)
    {
        const int N = 50_003;
        var rnd = new Random(17);
        float[] sp = { 0f, -0f, 300f, 128.5f, -300f, 1e9f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 65535.9f, 32768.9f };
        var data = new float[N];
        for (int i = 0; i < N; i++) data[i] = rnd.Next(100) < 15 ? sp[rnd.Next(sp.Length)] : (float)(rnd.NextDouble() * 140000 - 70000);
        var got = np.array(data).astype(dst);
        for (int i = 0; i < N; i++)
            Assert.AreEqual(Convert(data[i], dst), System.Convert.ToInt64(got.GetAtIndex(i)), $"f32->{dst}[{i}] ({data[i]})");
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.Char)]
    public void Single_To_Narrow_Strided_EqualsConverts(NPTypeCode dst)
    {
        const int Rows = 200, Cols = 301;
        var rnd = new Random(23);
        var data = new float[Rows * Cols];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rnd.NextDouble() * 140000 - 70000);
        var mat = np.array(data).reshape(Rows, Cols);

        // inner-strided [:, ::2]
        var view = mat[":, ::2"];
        var got = view.astype(dst).flatten();
        int idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c += 2)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(got.GetAtIndex(idx++)), $"f32[:, ::2]->{dst} r{r}c{c}");

        // reversed [:, ::-1]
        var rev = mat[":, ::-1"].astype(dst).flatten();
        idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = Cols - 1; c >= 0; c--)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(rev.GetAtIndex(idx++)), $"f32[:, ::-1]->{dst} r{r}c{c}");
    }

    private static long Convert(float v, NPTypeCode dst) => dst switch
    {
        NPTypeCode.SByte => Converts.ToSByte(v),
        NPTypeCode.Byte => Converts.ToByte(v),
        NPTypeCode.Int16 => Converts.ToInt16(v),
        NPTypeCode.UInt16 => Converts.ToUInt16(v),
        NPTypeCode.Char => Converts.ToChar(v),
        _ => throw new ArgumentOutOfRangeException(nameof(dst)),
    };
}
