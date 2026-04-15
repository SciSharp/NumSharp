using System;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest.Creation;

/// <summary>
/// Battle tests for np.arange - verifies NumSharp matches NumPy behavior exactly.
/// All expected values captured from NumPy 2.x.
/// </summary>
[TestClass]
public class ArangeBattleTests
{
    #region Basic Integer Ranges

    [TestMethod]
    public async Task Arange_StopOnly_Zero_ReturnsEmptyInt64()
    {
        // np.arange(0): array([], dtype=int64)
        var result = np.arange(0);

        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(long));
        result.shape.Should().BeEquivalentTo(new long[] { 0 });
    }

    [TestMethod]
    public async Task Arange_StopOnly_One_ReturnsSingleElementInt64()
    {
        // np.arange(1): array([0]) dtype=int64
        var result = np.arange(1);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L);
    }

    [TestMethod]
    public async Task Arange_StopOnly_Five_ReturnsZeroToFourInt64()
    {
        // np.arange(5): array([0, 1, 2, 3, 4]) dtype=int64
        var result = np.arange(5);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L);
    }

    [TestMethod]
    public async Task Arange_StopOnly_Ten_ReturnsZeroToNineInt64()
    {
        // np.arange(10): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) dtype=int64
        var result = np.arange(10);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L);
    }

    #endregion

    #region Start and Stop

    [TestMethod]
    public async Task Arange_StartStop_ZeroToFive()
    {
        // np.arange(0, 5): array([0, 1, 2, 3, 4]) dtype=int64
        var result = np.arange(0, 5);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L);
    }

    [TestMethod]
    public async Task Arange_StartStop_OneToFive()
    {
        // np.arange(1, 5): array([1, 2, 3, 4]) dtype=int64
        var result = np.arange(1, 5);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(1L, 2L, 3L, 4L);
    }

    [TestMethod]
    public async Task Arange_StartStop_FiveToTen()
    {
        // np.arange(5, 10): array([5, 6, 7, 8, 9]) dtype=int64
        var result = np.arange(5, 10);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(5L, 6L, 7L, 8L, 9L);
    }

    [TestMethod]
    public async Task Arange_StartStop_NegativeFiveToFive()
    {
        // np.arange(-5, 5): array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]) dtype=int64
        var result = np.arange(-5, 5);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(-5L, -4L, -3L, -2L, -1L, 0L, 1L, 2L, 3L, 4L);
    }

    [TestMethod]
    public async Task Arange_StartStop_NegativeTenToNegativeFive()
    {
        // np.arange(-10, -5): array([-10, -9, -8, -7, -6]) dtype=int64
        var result = np.arange(-10, -5);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(-10L, -9L, -8L, -7L, -6L);
    }

    #endregion

    #region With Step

    [TestMethod]
    public async Task Arange_WithStep_ZeroToTenByTwo()
    {
        // np.arange(0, 10, 2): array([0, 2, 4, 6, 8]) dtype=int64
        var result = np.arange(0, 10, 2);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 2L, 4L, 6L, 8L);
    }

    [TestMethod]
    public async Task Arange_WithStep_ZeroToTenByThree()
    {
        // np.arange(0, 10, 3): array([0, 3, 6, 9]) dtype=int64
        var result = np.arange(0, 10, 3);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 3L, 6L, 9L);
    }

    [TestMethod]
    public async Task Arange_WithStep_OneToTenByTwo()
    {
        // np.arange(1, 10, 2): array([1, 3, 5, 7, 9]) dtype=int64
        var result = np.arange(1, 10, 2);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(1L, 3L, 5L, 7L, 9L);
    }

    #endregion

    #region Negative Step

    [TestMethod]
    public async Task Arange_NegativeStep_TenToZeroByMinusOne()
    {
        // np.arange(10, 0, -1): array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1]) dtype=int64
        var result = np.arange(10, 0, -1);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(10L, 9L, 8L, 7L, 6L, 5L, 4L, 3L, 2L, 1L);
    }

    [TestMethod]
    public async Task Arange_NegativeStep_TenToZeroByMinusTwo()
    {
        // np.arange(10, 0, -2): array([10, 8, 6, 4, 2]) dtype=int64
        var result = np.arange(10, 0, -2);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(10L, 8L, 6L, 4L, 2L);
    }

    [TestMethod]
    public async Task Arange_NegativeStep_FiveToNegativeFiveByMinusOne()
    {
        // np.arange(5, -5, -1): array([5, 4, 3, 2, 1, 0, -1, -2, -3, -4]) dtype=int64
        var result = np.arange(5, -5, -1);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(5L, 4L, 3L, 2L, 1L, 0L, -1L, -2L, -3L, -4L);
    }

    [TestMethod]
    public async Task Arange_NegativeStep_FiveToNegativeFiveByMinusTwo()
    {
        // np.arange(5, -5, -2): array([5, 3, 1, -1, -3]) dtype=int64
        var result = np.arange(5, -5, -2);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(5L, 3L, 1L, -1L, -3L);
    }

    [TestMethod]
    public async Task Arange_NegativeStep_NegativeOneToNegativeTenByMinusOne()
    {
        // np.arange(-1, -10, -1): array([-1, -2, -3, -4, -5, -6, -7, -8, -9]) dtype=int64
        var result = np.arange(-1, -10, -1);

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(-1L, -2L, -3L, -4L, -5L, -6L, -7L, -8L, -9L);
    }

    #endregion

    #region Empty Arrays

    [TestMethod]
    public async Task Arange_Empty_StartGreaterThanStop()
    {
        // np.arange(5, 0): array([], dtype=int64)
        var result = np.arange(5, 0);

        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(long));
    }

    [TestMethod]
    public async Task Arange_Empty_StartEqualsStop()
    {
        // np.arange(5, 5): array([], dtype=int64)
        var result = np.arange(5, 5);

        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(long));
    }

    [TestMethod]
    public async Task Arange_Empty_ZeroToNegative()
    {
        // np.arange(0, -5): array([], dtype=int64)
        var result = np.arange(0, -5);

        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(long));
    }

    [TestMethod]
    public async Task Arange_Empty_WrongStepDirection()
    {
        // np.arange(0, 5, -1): array([], dtype=int64)
        var result = np.arange(0, 5, -1);

        result.size.Should().Be(0);
        result.dtype.Should().Be(typeof(long));
    }

    #endregion

    #region Float Ranges

    [TestMethod]
    public async Task Arange_Float_ZeroToFive()
    {
        // np.arange(0.0, 5.0): array([0., 1., 2., 3., 4.]) dtype=float64
        var result = np.arange(0.0, 5.0);

        result.dtype.Should().Be(typeof(double));
        result.Should().BeOfValues(0.0, 1.0, 2.0, 3.0, 4.0);
    }

    [TestMethod]
    public async Task Arange_Float_ZeroToOneByPointOne()
    {
        // np.arange(0.0, 1.0, 0.1): array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) dtype=float64
        var result = np.arange(0.0, 1.0, 0.1);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(10);
        // Check first and last due to floating point precision
        ((double)result[0]).Should().Be(0.0);
        ((double)result[9]).Should().BeApproximately(0.9, 1e-10);
    }

    [TestMethod]
    public async Task Arange_Float_ZeroToOneByPointTwo()
    {
        // np.arange(0.0, 1.0, 0.2): array([0., 0.2, 0.4, 0.6, 0.8]) dtype=float64
        var result = np.arange(0.0, 1.0, 0.2);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(5);
    }

    [TestMethod]
    public async Task Arange_Float_ZeroToOneByPointThree()
    {
        // np.arange(0.0, 1.0, 0.3): array([0., 0.3, 0.6, 0.9]) dtype=float64
        var result = np.arange(0.0, 1.0, 0.3);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(4);
    }

    [TestMethod]
    public async Task Arange_Float_HalfToFivePointFive()
    {
        // np.arange(0.5, 5.5): array([0.5, 1.5, 2.5, 3.5, 4.5]) dtype=float64
        var result = np.arange(0.5, 5.5);

        result.dtype.Should().Be(typeof(double));
        result.Should().BeOfValues(0.5, 1.5, 2.5, 3.5, 4.5);
    }

    [TestMethod]
    public async Task Arange_Float_OnePointFiveToFivePointFiveByHalf()
    {
        // np.arange(1.5, 5.5, 0.5): array([1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]) dtype=float64
        var result = np.arange(1.5, 5.5, 0.5);

        result.dtype.Should().Be(typeof(double));
        result.Should().BeOfValues(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0);
    }

    #endregion

    #region Float Negative Step

    [TestMethod]
    public async Task Arange_FloatNegativeStep_FiveToZeroByMinusOne()
    {
        // np.arange(5.0, 0.0, -1.0): array([5., 4., 3., 2., 1.]) dtype=float64
        var result = np.arange(5.0, 0.0, -1.0);

        result.dtype.Should().Be(typeof(double));
        result.Should().BeOfValues(5.0, 4.0, 3.0, 2.0, 1.0);
    }

    [TestMethod]
    public async Task Arange_FloatNegativeStep_OneToZeroByMinusPointTwo()
    {
        // np.arange(1.0, 0.0, -0.2): array([1., 0.8, 0.6, 0.4, 0.2]) dtype=float64
        var result = np.arange(1.0, 0.0, -0.2);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(5);
        ((double)result[0]).Should().Be(1.0);
        ((double)result[4]).Should().BeApproximately(0.2, 1e-10);
    }

    #endregion

    #region With dtype Parameter

    [TestMethod]
    public async Task Arange_Dtype_Int32()
    {
        // np.arange(5, dtype=np.int32): array([0, 1, 2, 3, 4], dtype=int32)
        var result = np.arange(5, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    [TestMethod]
    public async Task Arange_Dtype_Int64()
    {
        // np.arange(5, dtype=np.int64): array([0, 1, 2, 3, 4]) dtype=int64
        var result = np.arange(5, typeof(long));

        result.dtype.Should().Be(typeof(long));
        result.Should().BeOfValues(0L, 1L, 2L, 3L, 4L);
    }

    [TestMethod]
    public async Task Arange_Dtype_Float32()
    {
        // np.arange(5, dtype=np.float32): array([0., 1., 2., 3., 4.], dtype=float32)
        var result = np.arange(5, typeof(float));

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(0f, 1f, 2f, 3f, 4f);
    }

    [TestMethod]
    public async Task Arange_Dtype_Float64()
    {
        // np.arange(5, dtype=np.float64): array([0., 1., 2., 3., 4.]) dtype=float64
        var result = np.arange(5, typeof(double));

        result.dtype.Should().Be(typeof(double));
        result.Should().BeOfValues(0.0, 1.0, 2.0, 3.0, 4.0);
    }

    [TestMethod]
    public async Task Arange_Dtype_FloatInputToInt32()
    {
        // np.arange(5.0, dtype=np.int32): array([0, 1, 2, 3, 4], dtype=int32)
        var result = np.arange(5.0, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    [TestMethod]
    public async Task Arange_Dtype_WithStepToFloat32()
    {
        // np.arange(0, 10, 2, dtype=np.float32): array([0., 2., 4., 6., 8.], dtype=float32)
        var result = np.arange(0, 10, 2, typeof(float));

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(0f, 2f, 4f, 6f, 8f);
    }

    [TestMethod]
    public async Task Arange_Dtype_FloatRangeToInt32()
    {
        // np.arange(0.0, 5.0, dtype=np.int32): array([0, 1, 2, 3, 4], dtype=int32)
        var result = np.arange(0.0, 5.0, 1.0, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    #endregion

    #region Various Integer dtypes

    [TestMethod]
    public async Task Arange_Dtype_UInt8()
    {
        // np.arange(10, dtype=np.uint8): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8)
        var result = np.arange(10, typeof(byte));

        result.dtype.Should().Be(typeof(byte));
        result.Should().BeOfValues((byte)0, (byte)1, (byte)2, (byte)3, (byte)4, (byte)5, (byte)6, (byte)7, (byte)8, (byte)9);
    }

    [TestMethod]
    public async Task Arange_Dtype_Int16()
    {
        // np.arange(10, dtype=np.int16): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int16)
        var result = np.arange(10, typeof(short));

        result.dtype.Should().Be(typeof(short));
        result.Should().BeOfValues((short)0, (short)1, (short)2, (short)3, (short)4, (short)5, (short)6, (short)7, (short)8, (short)9);
    }

    [TestMethod]
    public async Task Arange_Dtype_UInt16()
    {
        // np.arange(10, dtype=np.uint16): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint16)
        var result = np.arange(10, typeof(ushort));

        result.dtype.Should().Be(typeof(ushort));
        result.Should().BeOfValues((ushort)0, (ushort)1, (ushort)2, (ushort)3, (ushort)4, (ushort)5, (ushort)6, (ushort)7, (ushort)8, (ushort)9);
    }

    [TestMethod]
    public async Task Arange_Dtype_UInt32()
    {
        // np.arange(10, dtype=np.uint32): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint32)
        var result = np.arange(10, typeof(uint));

        result.dtype.Should().Be(typeof(uint));
        result.Should().BeOfValues(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u);
    }

    [TestMethod]
    public async Task Arange_Dtype_UInt64()
    {
        // np.arange(10, dtype=np.uint64): array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint64)
        var result = np.arange(10, typeof(ulong));

        result.dtype.Should().Be(typeof(ulong));
        result.Should().BeOfValues(0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, 9ul);
    }

    #endregion

    #region Large Ranges

    [TestMethod]
    public async Task Arange_Large_Thousand()
    {
        // np.arange(1000).shape: (1000,)
        var result = np.arange(1000);

        result.shape.Should().BeEquivalentTo(new long[] { 1000 });
        result.size.Should().Be(1000);
        ((long)result[0]).Should().Be(0L);
        ((long)result[999]).Should().Be(999L);
    }

    [TestMethod]
    public async Task Arange_Large_WithStep()
    {
        // np.arange(0, 1000000, 1000).shape: (1000,)
        // np.arange(0, 1000000, 1000)[-5:]: array([995000, 996000, 997000, 998000, 999000])
        var result = np.arange(0, 1000000, 1000);

        result.shape.Should().BeEquivalentTo(new long[] { 1000 });
        ((long)result[995]).Should().Be(995000L);
        ((long)result[996]).Should().Be(996000L);
        ((long)result[997]).Should().Be(997000L);
        ((long)result[998]).Should().Be(998000L);
        ((long)result[999]).Should().Be(999000L);
    }

    #endregion

    #region Floating Point Edge Cases

    [TestMethod]
    public async Task Arange_FloatEdge_PointOneToPointFour()
    {
        // np.arange(0.1, 0.4, 0.1): array([0.1, 0.2, 0.3, 0.4]) len=4
        // Note: NumPy includes 0.4 due to floating point rounding
        var result = np.arange(0.1, 0.4, 0.1);

        result.dtype.Should().Be(typeof(double));
        // NumPy returns 4 elements due to floating point math
        result.size.Should().Be(4);
    }

    [TestMethod]
    public async Task Arange_FloatEdge_ZeroToPointSix()
    {
        // np.arange(0.0, 0.6, 0.1): array([0., 0.1, 0.2, 0.3, 0.4, 0.5]) len=6
        var result = np.arange(0.0, 0.6, 0.1);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(6);
    }

    [TestMethod]
    public async Task Arange_FloatEdge_ZeroToPointSeven()
    {
        // np.arange(0.0, 0.7, 0.1): array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]) len=7
        var result = np.arange(0.0, 0.7, 0.1);

        result.dtype.Should().Be(typeof(double));
        result.size.Should().Be(7);
    }

    #endregion

    #region NPTypeCode Overloads

    [TestMethod]
    public async Task Arange_NPTypeCode_Int32()
    {
        var result = np.arange(5, NPTypeCode.Int32);

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    [TestMethod]
    public async Task Arange_NPTypeCode_WithStartStop()
    {
        var result = np.arange(2, 8, NPTypeCode.Int16);

        result.dtype.Should().Be(typeof(short));
        result.Should().BeOfValues((short)2, (short)3, (short)4, (short)5, (short)6, (short)7);
    }

    [TestMethod]
    public async Task Arange_NPTypeCode_WithStep()
    {
        var result = np.arange(0, 10, 2, NPTypeCode.Single);

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(0f, 2f, 4f, 6f, 8f);
    }

    #endregion

    #region Single-Precision Float (float32)

    [TestMethod]
    public async Task Arange_Float32_Basic()
    {
        // Explicit float overload
        var result = np.arange(0f, 5f);

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(0f, 1f, 2f, 3f, 4f);
    }

    [TestMethod]
    public async Task Arange_Float32_WithStep()
    {
        var result = np.arange(0f, 2f, 0.5f);

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(0f, 0.5f, 1f, 1.5f);
    }

    [TestMethod]
    public async Task Arange_Float32_NegativeStep()
    {
        var result = np.arange(5f, 0f, -1f);

        result.dtype.Should().Be(typeof(float));
        result.Should().BeOfValues(5f, 4f, 3f, 2f, 1f);
    }

    #endregion

    #region Error Cases

    [TestMethod]
    public async Task Arange_ZeroStep_ThrowsArgumentException()
    {
        await Assert.ThrowsAsync<ArgumentException>(async () => np.arange(0, 10, 0));
    }

    [TestMethod]
    public async Task Arange_ZeroStepFloat_ThrowsArgumentException()
    {
        await Assert.ThrowsAsync<ArgumentException>(async () => np.arange(0.0, 10.0, 0.0));
    }

    #endregion

    #region Fractional Step with Integer dtype (NumPy uses integer arithmetic)

    /// <summary>
    /// NumPy calculates delta in target dtype: delta = int(start+step) - int(start).
    /// For arange(0, 5, 0.5, int32): delta = int(0.5) - int(0) = 0, so all values are 0.
    /// </summary>
    [TestMethod]
    public async Task Arange_FractionalStep_IntDtype_ZeroDelta()
    {
        // np.arange(0, 5, 0.5, dtype=np.int32) → [0,0,0,0,0,0,0,0,0,0]
        // NumPy: int(0)=0, int(0+0.5)=0, delta=0, all values=0
        var result = np.arange(0, 5, 0.5, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.size.Should().Be(10);
        // NumPy returns all zeros because delta=0
        result.Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    }

    /// <summary>
    /// For arange(5, 0, -0.5, int32): delta = int(4.5) - int(5) = 4 - 5 = -1.
    /// Values decrement by 1, not by 0.5.
    /// </summary>
    [TestMethod]
    public async Task Arange_FractionalNegativeStep_IntDtype_IntegerDelta()
    {
        // np.arange(5, 0, -0.5, dtype=np.int32) → [5,4,3,2,1,0,-1,-2,-3,-4]
        // NumPy: int(5)=5, int(5-0.5)=4, delta=-1
        var result = np.arange(5, 0, -0.5, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.size.Should().Be(10);
        // NumPy decrements by 1 (the integer delta), not by 0.5
        result.Should().BeOfValues(5, 4, 3, 2, 1, 0, -1, -2, -3, -4);
    }

    /// <summary>
    /// For arange(0, 5, 0.7, int32): delta = int(0.7) - int(0) = 0.
    /// </summary>
    [TestMethod]
    public async Task Arange_FractionalStep_0_7_IntDtype()
    {
        // np.arange(0, 5, 0.7, dtype=np.int32) → [0,0,0,0,0,0,0,0]
        var result = np.arange(0, 5, 0.7, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.size.Should().Be(8);
        result.Should().BeOfValues(0, 0, 0, 0, 0, 0, 0, 0);
    }

    /// <summary>
    /// For arange(5, 0, -0.7, int32): delta = int(4.3) - int(5) = 4 - 5 = -1.
    /// </summary>
    [TestMethod]
    public async Task Arange_FractionalNegativeStep_0_7_IntDtype()
    {
        // np.arange(5, 0, -0.7, dtype=np.int32) → [5,4,3,2,1,0,-1,-2]
        var result = np.arange(5, 0, -0.7, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.size.Should().Be(8);
        result.Should().BeOfValues(5, 4, 3, 2, 1, 0, -1, -2);
    }

    /// <summary>
    /// Float start with integer step and int dtype works correctly.
    /// </summary>
    [TestMethod]
    public async Task Arange_FloatStart_IntStep_IntDtype()
    {
        // np.arange(0.5, 5.5, 1, dtype=np.int32) → [0,1,2,3,4]
        // NumPy: int(0.5)=0, int(1.5)=1, delta=1
        var result = np.arange(0.5, 5.5, 1, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    /// <summary>
    /// Another float start case.
    /// </summary>
    [TestMethod]
    public async Task Arange_FloatStart_0_9_IntDtype()
    {
        // np.arange(0.9, 5.9, 1, dtype=np.int32) → [0,1,2,3,4]
        var result = np.arange(0.9, 5.9, 1, typeof(int));

        result.dtype.Should().Be(typeof(int));
        result.Should().BeOfValues(0, 1, 2, 3, 4);
    }

    #endregion
}
