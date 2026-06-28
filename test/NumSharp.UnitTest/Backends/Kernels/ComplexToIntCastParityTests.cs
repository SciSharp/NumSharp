using System;
using System.Numerics;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD Complex → int cast (<c>TryGetComplexToIntKernel</c> /
/// <c>TryGetComplexToIntStridedKernel</c>): real-part deinterleave (vunpcklpd + vpermq)
/// + cvttpd2dq + truncating <c>Vector.Narrow</c>.
///
/// NumPy drops the imaginary part (ComplexWarning), takes the real (a double), then does
/// float→int (cvtt, INT_MIN sentinel, low-bits wrap for narrow). Must be BIT-EXACT with
/// <see cref="Converts"/>.To{X}(Complex), hence NumPy 2.4.2.
/// </summary>
[TestClass]
public class ComplexToIntCastParityTests
{
    [TestMethod]
    public void Complex_To_Int32_DropsImag_MatchesNumPy()
    {
        var src = np.array(new Complex[] { new(3, 4), new(3e9, 1), new(128.5, -9), new(-300, 0), new(double.NaN, 2), new(double.PositiveInfinity, -1), new(2147483653.0, 0) });
        var expect = new int[] { 3, -2147483648, 128, -300, -2147483648, -2147483648, -2147483648 };
        var got = src.astype(NPTypeCode.Int32);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetInt32(i), $"c128->i32[{i}]");
    }

    [TestMethod]
    public void Complex_To_SByte_Wrap_MatchesNumPy()
    {
        var src = np.array(new Complex[] { new(3, 4), new(3e9, 1), new(128.5, -9), new(-300, 0), new(double.NaN, 2), new(double.PositiveInfinity, -1), new(2147483653.0, 0) });
        var expect = new sbyte[] { 3, 0, -128, -44, 0, 0, 0 };
        var got = src.astype(NPTypeCode.SByte);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetSByte(i), $"c128->i8[{i}]");
    }

    [TestMethod]
    public void Complex_To_Int16_Wrap_MatchesNumPy()
    {
        var src = np.array(new Complex[] { new(3, 4), new(3e9, 1), new(128.5, -9), new(-300, 0), new(double.NaN, 2), new(2147483653.0, 0) });
        var expect = new short[] { 3, 0, 128, -300, 0, 0 };
        var got = src.astype(NPTypeCode.Int16);
        for (int i = 0; i < expect.Length; i++)
            Assert.AreEqual(expect[i], got.GetInt16(i), $"c128->i16[{i}]");
    }

    // Random + edge: SIMD kernel == scalar Converts reference, all int targets,
    // contiguous AND strided, odd length to exercise the scalar tail.
    [DataTestMethod]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Byte)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.UInt16)]
    [DataRow(NPTypeCode.Char)]
    public void Complex_To_Int_Contig_EqualsConverts(NPTypeCode dst)
    {
        const int N = 50_003;
        var rnd = new Random(29);
        (double, double)[] sp = { (3, 4), (3e9, 1), (128.5, -9), (-300, 0), (double.NaN, 2), (double.PositiveInfinity, -1), (2147483653.0, 0), (-128.6, 7), (65535.9, 1) };
        var data = new Complex[N];
        for (int i = 0; i < N; i++)
        {
            if (rnd.Next(100) < 15) { var t = sp[rnd.Next(sp.Length)]; data[i] = new Complex(t.Item1, t.Item2); }
            else data[i] = new Complex(rnd.NextDouble() * 140000 - 70000, rnd.NextDouble() * 10);
        }
        var got = np.array(data).astype(dst);
        for (int i = 0; i < N; i++)
            Assert.AreEqual(Convert(data[i], dst), System.Convert.ToInt64(got.GetAtIndex(i)), $"c128->{dst}[{i}]");
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Int16)]
    public void Complex_To_Int_Strided_EqualsConverts(NPTypeCode dst)
    {
        const int Rows = 200, Cols = 301;
        var rnd = new Random(31);
        var data = new Complex[Rows * Cols];
        for (int i = 0; i < data.Length; i++) data[i] = new Complex(rnd.NextDouble() * 140000 - 70000, rnd.NextDouble() * 10);
        var mat = np.array(data).reshape(Rows, Cols);

        var view = mat[":, ::2"]; // inner-strided
        var got = view.astype(dst).flatten();
        int idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c += 2)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(got.GetAtIndex(idx++)), $"c128[:, ::2]->{dst} r{r}c{c}");

        var sliced = mat["1:150, 1:280"].astype(dst).flatten(); // offset, contiguous inner
        idx = 0;
        for (int r = 1; r < 150; r++)
            for (int c = 1; c < 280; c++)
                Assert.AreEqual(Convert(data[r * Cols + c], dst), System.Convert.ToInt64(sliced.GetAtIndex(idx++)), $"c128[1:150,1:280]->{dst} r{r}c{c}");
    }

    private static long Convert(Complex v, NPTypeCode dst) => dst switch
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
