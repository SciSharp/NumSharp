using System;
using System.Numerics;
using AwesomeAssertions;
using NumSharp;
using NumSharp.Backends;

namespace NumSharp.UnitTest.Statistics;

/// <summary>
/// Battle tests for np.ptp, comparing against NumPy 2.4.2 reference values.
/// Covers axis variants, keepdims, dtype preservation (uint8/int8 wraparound),
/// tuple-axis, out= parameter, NaN propagation, and the empty-axis error.
/// </summary>
[TestClass]
public class np_ptp_BattleTests
{
    private static double At(NDArray nd, int i) => Convert.ToDouble(nd.GetAtIndex(i));

    // ── 1D / 2D basics ───────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_1D_Long()
    {
        // NumPy: np.ptp([3,1,4,1,5,9,2]) -> 8
        var a = np.array(new long[] { 3, 1, 4, 1, 5, 9, 2 });
        var p = np.ptp(a);
        At(p, 0).Should().Be(8.0);
    }

    [TestMethod]
    public void Ptp_2D_Axis0()
    {
        // NumPy: np.ptp([[10,7,4],[3,2,1]], axis=0) -> [7,5,3]
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var p = np.ptp(a, axis: 0);
        p.shape.Should().Equal(new long[] { 3 });
        At(p, 0).Should().Be(7.0);
        At(p, 1).Should().Be(5.0);
        At(p, 2).Should().Be(3.0);
    }

    [TestMethod]
    public void Ptp_2D_Axis1()
    {
        // NumPy: np.ptp([[10,7,4],[3,2,1]], axis=1) -> [6,2]
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var p = np.ptp(a, axis: 1);
        p.shape.Should().Equal(new long[] { 2 });
        At(p, 0).Should().Be(6.0);
        At(p, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void Ptp_2D_Keepdims_PreservesShape()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var p = np.ptp(a, axis: 1, keepdims: true);
        p.shape.Should().Equal(new long[] { 2, 1 });
        At(p, 0).Should().Be(6.0);
        At(p, 1).Should().Be(2.0);
    }

    [TestMethod]
    public void Ptp_AxisNone_3D()
    {
        // NumPy: np.ptp(np.arange(24).reshape(2,3,4)) -> 23
        var a = np.arange(24).reshape(2, 3, 4);
        var p = np.ptp(a);
        At(p, 0).Should().Be(23.0);
    }

    [TestMethod]
    public void Ptp_AxisNone_Keepdims_AllOnes()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var p = np.ptp(a, axis: (int?)null, keepdims: true);
        p.shape.Should().Equal(new long[] { 1, 1, 1 });
        At(p, 0).Should().Be(23.0);
    }

    [TestMethod]
    public void Ptp_NegativeAxis_MatchesPositive()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var p1 = np.ptp(a, axis: -1);
        var p2 = np.ptp(a, axis: 2);
        p1.shape.Should().Equal(p2.shape);
        for (int i = 0; i < (int)p1.size; i++)
            At(p1, i).Should().Be(At(p2, i));
    }

    // ── tuple axis ───────────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_3D_TupleAxis_01()
    {
        // NumPy: np.ptp(np.arange(24).reshape(2,3,4), axis=(0,1)) -> [20,20,20,20]
        var a = np.arange(24).reshape(2, 3, 4);
        var p = np.ptp(a, axis: new[] { 0, 1 });
        p.shape.Should().Equal(new long[] { 4 });
        for (int i = 0; i < 4; i++) At(p, i).Should().Be(20.0);
    }

    [TestMethod]
    public void Ptp_3D_TupleAxis_Keepdims()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        var p = np.ptp(a, axis: new[] { 0, 1 }, keepdims: true);
        p.shape.Should().Equal(new long[] { 1, 1, 4 });
    }

    [TestMethod]
    public void Ptp_DuplicateAxis_Throws()
    {
        var a = np.arange(24).reshape(2, 3, 4);
        Action act = () => np.ptp(a, axis: new[] { 0, 0 });
        act.Should().Throw<ArgumentException>();
    }

    // ── dtype preservation (NumPy semantics, including wraparound) ───────

    [TestMethod]
    public void Ptp_UInt8_WrapsTo255_PreservesDtype()
    {
        // NumPy: np.ptp(np.array([0,255], dtype=np.uint8)) -> 255 (dtype preserved)
        var a = np.array(new byte[] { 0, 255 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Byte);
        p.GetAtIndex<byte>(0).Should().Be(255);
    }

    [TestMethod]
    public void Ptp_Int8_WrapsToNeg1_PreservesDtype()
    {
        // NumPy: np.ptp(np.array([-128,127], dtype=np.int8)) -> -1 (int8 wraparound)
        var a = np.array(new sbyte[] { -128, 127 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.SByte);
        p.GetAtIndex<sbyte>(0).Should().Be(-1);
    }

    [TestMethod]
    public void Ptp_Float32_PreservesDtype()
    {
        var a = np.array(new float[] { 1f, 2f, 5f, 3f });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Single);
        p.GetAtIndex<float>(0).Should().Be(4.0f);
    }

    [TestMethod]
    public void Ptp_Float64_PreservesDtype()
    {
        var a = np.array(new double[] { 1, 2, 5, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Double);
        At(p, 0).Should().Be(4.0);
    }

    // ── NaN ─────────────────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_WithNaN_PropagatesNaN()
    {
        var a = np.array(new double[] { 1, double.NaN, 3 });
        var p = np.ptp(a);
        double.IsNaN(At(p, 0)).Should().BeTrue();
    }

    // ── error paths ──────────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_Null_Throws()
    {
        Action act = () => np.ptp((NDArray)null);
        act.Should().Throw<ArgumentNullException>();
    }

    [TestMethod]
    public void Ptp_EmptyArray_Throws()
    {
        // NumPy: ValueError: zero-size array to reduction operation maximum which has no identity
        var a = np.array(new double[0]);
        Action act = () => np.ptp(a);
        act.Should().Throw<ArgumentException>();
    }

    // ── out= parameter ───────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_OutParameter_WritesAndReturnsOut()
    {
        var a = np.array(new long[,] { { 10, 7, 4 }, { 3, 2, 1 } });
        var outNd = np.zeros(new Shape(2), dtype: typeof(long));
        var r = np.ptp(a, axis: 1, @out: outNd);
        ReferenceEquals(r.Storage, outNd.Storage).Should().BeTrue();
        At(outNd, 0).Should().Be(6.0);
        At(outNd, 1).Should().Be(2.0);
    }

    // ── NumPy docstring example (int8 per-row wraparound) ─────────────────

    [TestMethod]
    public void Ptp_Int8_DocExample_PerRowWraparound()
    {
        // NumPy 2.4.2 docs: np.ptp(int8[[1,127],[0,127],[-1,127],[-2,127]], axis=1)
        //                    -> array([126, 127, -128, -127], dtype=int8)
        var y = np.array(new sbyte[,] { { 1, 127 }, { 0, 127 }, { -1, 127 }, { -2, 127 } });
        var p = np.ptp(y, axis: 1);
        p.typecode.Should().Be(NPTypeCode.SByte);
        p.shape.Should().Equal(new long[] { 4 });
        p.GetAtIndex<sbyte>(0).Should().Be(126);
        p.GetAtIndex<sbyte>(1).Should().Be(127);
        p.GetAtIndex<sbyte>(2).Should().Be(-128);
        p.GetAtIndex<sbyte>(3).Should().Be(-127);
    }

    // ── layout variations ────────────────────────────────────────────────

    [TestMethod]
    public void Ptp_ZeroD_Scalar_ReturnsZero()
    {
        var s = NDArray.Scalar(5.0);
        var p = np.ptp(s);
        At(p, 0).Should().Be(0.0);
    }

    [TestMethod]
    public void Ptp_Transposed_MatchesAxisSwap()
    {
        // NumPy: np.ptp(a.T, axis=0) == np.ptp(a, axis=1) -> [4,4,4,4]
        var a = np.arange(20).reshape(4, 5);
        var p = np.ptp(a.T, axis: 0);
        p.shape.Should().Equal(new long[] { 4 });
        for (int i = 0; i < 4; i++) At(p, i).Should().Be(4.0);
    }

    [TestMethod]
    public void Ptp_NegativeStrideView()
    {
        // NumPy: np.ptp(np.arange(20).reshape(4,5)[::-1]) -> 19
        var a = np.arange(20).reshape(4, 5);
        var rev = a["::-1"];
        var p = np.ptp(rev);
        At(p, 0).Should().Be(19.0);
    }

    [TestMethod]
    public void Ptp_StridedView()
    {
        // NumPy: np.ptp(np.arange(20).reshape(4,5)[:, ::2], axis=1) -> [4,4,4,4]
        var a = np.arange(20).reshape(4, 5);
        var s = a[":, ::2"];
        var p = np.ptp(s, axis: 1);
        p.shape.Should().Equal(new long[] { 4 });
        for (int i = 0; i < 4; i++) At(p, i).Should().Be(4.0);
    }

    [TestMethod]
    public void Ptp_HigherRank_5D()
    {
        var hi = np.arange(2 * 3 * 2 * 3 * 2).reshape(2, 3, 2, 3, 2);
        At(np.ptp(hi), 0).Should().Be(71.0);
        var ax2 = np.ptp(hi, axis: 2);
        ax2.shape.Should().Equal(new long[] { 2, 3, 3, 2 });
        for (int i = 0; i < (int)ax2.size; i++) At(ax2, i).Should().Be(6.0);
    }

    // ── full dtype sweep — values + dtype preservation ───────────────────

    [TestMethod]
    public void Ptp_Int16_PreservesDtype()
    {
        var a = np.array(new short[] { 0, 1, 2, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Int16);
        p.GetAtIndex<short>(0).Should().Be(3);
    }

    [TestMethod]
    public void Ptp_UInt16_PreservesDtype()
    {
        var a = np.array(new ushort[] { 0, 1, 2, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.UInt16);
        p.GetAtIndex<ushort>(0).Should().Be(3);
    }

    [TestMethod]
    public void Ptp_Int32_PreservesDtype()
    {
        var a = np.array(new int[] { 0, 1, 2, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Int32);
        p.GetAtIndex<int>(0).Should().Be(3);
    }

    [TestMethod]
    public void Ptp_UInt32_PreservesDtype()
    {
        var a = np.array(new uint[] { 0, 1, 2, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.UInt32);
        p.GetAtIndex<uint>(0).Should().Be(3u);
    }

    [TestMethod]
    public void Ptp_UInt64_PreservesDtype()
    {
        var a = np.array(new ulong[] { 0, 1, 2, 3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.UInt64);
        p.GetAtIndex<ulong>(0).Should().Be(3ul);
    }

    [TestMethod]
    public void Ptp_Half_PreservesDtype()
    {
        var a = np.array(new Half[] { (Half)0, (Half)1, (Half)2, (Half)3 });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Half);
        ((float)p.GetAtIndex<Half>(0)).Should().BeApproximately(3.0f, 0.01f);
    }

    [TestMethod]
    public void Ptp_Decimal_PreservesDtype()
    {
        var a = np.array(new decimal[] { 0m, 1m, 2m, 3m });
        var p = np.ptp(a);
        p.typecode.Should().Be(NPTypeCode.Decimal);
        p.GetAtIndex<decimal>(0).Should().Be(3m);
    }

    [TestMethod]
    public void Ptp_Complex_LexicographicMax()
    {
        // NumPy uses lexicographic ordering for complex (real wins, then imag).
        // For [1+2j, 3+0j, 2+5j]: max=3+0j, min=1+2j -> ptp=2-2j.
        var c = np.array(new Complex[] { new(1, 2), new(3, 0), new(2, 5) });
        var p = np.ptp(c);
        p.typecode.Should().Be(NPTypeCode.Complex);
        var v = p.GetAtIndex<Complex>(0);
        v.Real.Should().Be(2.0);
        v.Imaginary.Should().Be(-2.0);
    }

    [TestMethod]
    public void Ptp_Bool_Throws()
    {
        // NumPy raises TypeError on bool subtract; NumSharp raises on the
        // amax/amin step (Boolean is not supported there). Either way: rejected.
        var a = np.array(new bool[] { false, true, false, true });
        Action act = () => np.ptp(a);
        act.Should().Throw<Exception>();
    }
}
