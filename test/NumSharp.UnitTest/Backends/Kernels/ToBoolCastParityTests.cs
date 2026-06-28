using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp;
using NumSharp.Utilities;
using Half = System.Half;

namespace NumSharp.UnitTest.Backends.Kernels;

/// <summary>
/// Pins the SIMD {int,float,half,char} → bool cast (<c>TryGetToBoolKernel</c> /
/// <c>TryGetToBoolStridedKernel</c>): vectorized <c>v != 0</c> compare + truncating narrow to
/// 0/1 bytes. Float uses OrderedEqualNonSignaling so <c>-0.0 → False</c> and <c>NaN → True</c>
/// (bit-exact NumPy <c>!= 0</c>); half uses <c>(bits &amp; 0x7fff) != 0</c>.
/// </summary>
[TestClass]
public class ToBoolCastParityTests
{
    [TestMethod]
    public void Single_To_Bool_IEEE_MatchesNumPy()
    {
        // NumPy: np.array([0,-0,1.5,nan,inf,-inf,1e-40],f32).astype(bool)
        var src = np.array(new float[] { 0f, -0f, 1.5f, float.NaN, float.PositiveInfinity, float.NegativeInfinity, 1e-40f });
        var expect = new[] { false, false, true, true, true, true, true };
        var got = src.astype(NPTypeCode.Boolean);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetBoolean(i), $"f32->bool[{i}]");
    }

    [TestMethod]
    public void Double_To_Bool_IEEE_MatchesNumPy()
    {
        var src = np.array(new double[] { 0.0, -0.0, 1.5, double.NaN, double.PositiveInfinity, 5e-324 });
        var expect = new[] { false, false, true, true, true, true };
        var got = src.astype(NPTypeCode.Boolean);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetBoolean(i), $"f64->bool[{i}]");
    }

    [TestMethod]
    public void Half_To_Bool_IEEE_MatchesNumPy()
    {
        var src = np.array(new Half[] { (Half)0f, (Half)(-0f), (Half)1.5f, Half.NaN, Half.PositiveInfinity, (Half)6e-8f });
        var expect = new[] { false, false, true, true, true, true };
        var got = src.astype(NPTypeCode.Boolean);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetBoolean(i), $"f16->bool[{i}]");
    }

    [TestMethod]
    public void Int32_To_Bool_MatchesNumPy()
    {
        var src = np.array(new int[] { 0, 1, -1, 256, int.MinValue, 0 });
        var expect = new[] { false, true, true, true, true, false };
        var got = src.astype(NPTypeCode.Boolean);
        for (int i = 0; i < expect.Length; i++) Assert.AreEqual(expect[i], got.GetBoolean(i), $"i32->bool[{i}]");
    }

    // Random: SIMD kernel == scalar reference, all source dtypes, contiguous AND strided,
    // odd length to exercise the scalar tail.
    [DataTestMethod]
    [DataRow(NPTypeCode.Single)]
    [DataRow(NPTypeCode.Double)]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.UInt32)]
    [DataRow(NPTypeCode.Int64)]
    [DataRow(NPTypeCode.UInt64)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.UInt16)]
    [DataRow(NPTypeCode.SByte)]
    [DataRow(NPTypeCode.Byte)]
    [DataRow(NPTypeCode.Char)]
    [DataRow(NPTypeCode.Half)]
    public void X_To_Bool_Contig_EqualsReference(NPTypeCode src)
    {
        const int N = 50_003;
        // ~30% zeros to exercise both branches, plus float specials.
        var baseArr = ((np.arange(N) % 7) - 3).astype(src); // values -3..3 incl 0
        var got = baseArr.astype(NPTypeCode.Boolean);
        for (int i = 0; i < N; i++)
            Assert.AreEqual(ToBool(baseArr, i), got.GetBoolean(i), $"{src}->bool[{i}]");
    }

    [DataTestMethod]
    [DataRow(NPTypeCode.Single)]
    [DataRow(NPTypeCode.Double)]
    [DataRow(NPTypeCode.Int32)]
    [DataRow(NPTypeCode.Int64)]
    [DataRow(NPTypeCode.Int16)]
    [DataRow(NPTypeCode.Half)]
    public void X_To_Bool_Strided_EqualsReference(NPTypeCode src)
    {
        const int Rows = 200, Cols = 301;
        var baseArr = ((np.arange(Rows * Cols) % 5) - 2).astype(src).reshape(Rows, Cols);
        var flat = baseArr.flatten();

        var got = baseArr[":, ::2"].astype(NPTypeCode.Boolean).flatten();
        int idx = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c += 2)
                Assert.AreEqual(ToBool(flat, r * Cols + c), got.GetBoolean(idx++), $"{src}[:, ::2]->bool r{r}c{c}");
    }

    private static bool ToBool(NDArray a, long i) => a.GetTypeCode switch
    {
        NPTypeCode.Single => Converts.ToBoolean(a.GetSingle(i)),
        NPTypeCode.Double => Converts.ToBoolean(a.GetDouble(i)),
        NPTypeCode.Half => Converts.ToBoolean((Half)a.GetValue<Half>(i)),
        NPTypeCode.Char => a.GetChar(i) != 0,
        _ => System.Convert.ToDecimal(a.GetAtIndex(i)) != 0m, // ToDecimal holds ulong.MaxValue (ToInt64 overflows)
    };
}
