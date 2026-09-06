using System;
using System.Numerics;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Tests for the NaN-aware cumulative scans np.nancumsum / np.nancumprod.
    /// Every expected value is verified against NumPy 2.4.2:
    ///   nancumsum treats NaN as 0 (leading NaNs -> 0, all-NaN/empty slice -> 0),
    ///   nancumprod treats NaN as 1 (leading NaNs -> 1, all-NaN/empty slice -> 1).
    /// Complex NaN (real OR imag) collapses the WHOLE element to the identity (0+0j / 1+0j).
    /// </summary>
    [TestClass]
    public class NanCumulativeTests : TestClass
    {
        #region np.nancumsum

        [TestMethod]
        public void NanCumSum_Basic1D_IgnoresNaN()
        {
            // np.nancumsum([1., nan, 3.]) == [1., 1., 4.]
            var r = np.nancumsum(np.array(new[] { 1.0, double.NaN, 3.0 }));
            r.Should().BeOfType(NPTypeCode.Double);
            r.Should().Be(np.array(new[] { 1.0, 1.0, 4.0 }));
        }

        [TestMethod]
        public void NanCumSum_LeadingNaN_ReplacedByZero()
        {
            // np.nancumsum([nan, nan, 2., 3.]) == [0., 0., 2., 5.]
            var r = np.nancumsum(np.array(new[] { double.NaN, double.NaN, 2.0, 3.0 }));
            r.Should().Be(np.array(new[] { 0.0, 0.0, 2.0, 5.0 }));
        }

        [TestMethod]
        public void NanCumSum_AllNaN_ReturnsZeros()
        {
            // np.nancumsum([nan, nan]) == [0., 0.]
            var r = np.nancumsum(np.array(new[] { double.NaN, double.NaN }));
            r.Should().Be(np.array(new[] { 0.0, 0.0 }));
        }

        [TestMethod]
        public void NanCumSum_NoNaN_SameAsCumSum()
        {
            // np.nancumsum([1., 2., 3.]) == np.cumsum == [1., 3., 6.]
            var r = np.nancumsum(np.array(new[] { 1.0, 2.0, 3.0 }));
            r.Should().Be(np.array(new[] { 1.0, 3.0, 6.0 }));
        }

        [TestMethod]
        public void NanCumSum_Float32_PreservesDtype()
        {
            // np.nancumsum(float32) stays float32
            var r = np.nancumsum(np.array(new[] { 1.0f, float.NaN, 3.0f }));
            r.Should().BeOfType(NPTypeCode.Single);
            r.Should().Be(np.array(new[] { 1.0f, 1.0f, 4.0f }));
        }

        [TestMethod]
        public void NanCumSum_Float16_PreservesDtype()
        {
            // np.nancumsum(float16) stays float16 -> [1., 1., 3.]
            var r = np.nancumsum(np.array(new[] { (Half)1.0, Half.NaN, (Half)2.0 }));
            r.Should().BeOfType(NPTypeCode.Half);
            r.Should().Be(np.array(new[] { (Half)1.0, (Half)1.0, (Half)3.0 }));
        }

        [TestMethod]
        public void NanCumSum_Complex_ReplacesWholeElement()
        {
            // input [1+2j, nan+5j, 3+0j, nan+nanj]; NaN (real OR imag) -> 0+0j
            // np.nancumsum -> [1+2j, 1+2j, 4+2j, 4+2j]
            var c = np.array(new Complex[] { new(1, 2), new(double.NaN, 5), new(3, 0), new(double.NaN, double.NaN) });
            var r = np.nancumsum(c);
            r.Should().BeOfType(NPTypeCode.Complex);
            r.Should().Be(np.array(new Complex[] { new(1, 2), new(1, 2), new(4, 2), new(4, 2) }));
        }

        [TestMethod]
        public void NanCumSum_2D_AxisNone_RavelsCOrder()
        {
            // a = [[1,2],[3,nan]]; np.nancumsum(a) == [1., 3., 6., 6.]
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumsum(a);
            r.Should().Be(np.array(new[] { 1.0, 3.0, 6.0, 6.0 }));
        }

        [TestMethod]
        public void NanCumSum_2D_Axis0()
        {
            // np.nancumsum(a, axis=0) == [[1,2],[4,2]]
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumsum(a, axis: 0);
            r.Should().Be(np.array(new[,] { { 1.0, 2.0 }, { 4.0, 2.0 } }));
        }

        [TestMethod]
        public void NanCumSum_2D_Axis1()
        {
            // np.nancumsum(a, axis=1) == [[1,3],[3,3]]
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumsum(a, axis: 1);
            r.Should().Be(np.array(new[,] { { 1.0, 3.0 }, { 3.0, 3.0 } }));
        }

        [TestMethod]
        public void NanCumSum_2D_NegativeAxis()
        {
            // axis=-1 == axis=1
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumsum(a, axis: -1);
            r.Should().Be(np.array(new[,] { { 1.0, 3.0 }, { 3.0, 3.0 } }));
        }

        [TestMethod]
        public void NanCumSum_Dtype_IntOnFloatNaN_CastsThenAccumulates()
        {
            // np.nancumsum([1.5, nan, 3.5], dtype=int64) == [1, 1, 4]
            // (NaN->0, then each element cast to int64 BEFORE accumulation: 1.5->1, 0->0, 3.5->3)
            var r = np.nancumsum(np.array(new[] { 1.5, double.NaN, 3.5 }), typeCode: NPTypeCode.Int64);
            r.Should().BeOfType(NPTypeCode.Int64);
            r.Should().Be(np.array(new[] { 1L, 1L, 4L }));
        }

        [TestMethod]
        public void NanCumSum_Dtype_Float32()
        {
            // np.nancumsum([1.5, nan, 3.5], dtype=float32) == [1.5, 1.5, 5.0]
            var r = np.nancumsum(np.array(new[] { 1.5, double.NaN, 3.5 }), typeCode: NPTypeCode.Single);
            r.Should().BeOfType(NPTypeCode.Single);
            r.Should().Be(np.array(new[] { 1.5f, 1.5f, 5.0f }));
        }

        [TestMethod]
        public void NanCumSum_Int_Passthrough_WidensToInt64()
        {
            // integers can't be NaN -> identical to np.cumsum, int32 -> int64 (NEP50)
            var r = np.nancumsum(np.array(new[] { 1, 2, 3 }));
            r.Should().BeOfType(NPTypeCode.Int64);
            r.Should().Be(np.array(new[] { 1L, 3L, 6L }));
        }

        [TestMethod]
        public void NanCumSum_Bool_Passthrough_WidensToInt64()
        {
            // np.nancumsum([True, False, True]) == [1, 1, 2] int64
            var r = np.nancumsum(np.array(new[] { true, false, true }));
            r.Should().BeOfType(NPTypeCode.Int64);
            r.Should().Be(np.array(new[] { 1L, 1L, 2L }));
        }

        [TestMethod]
        public void NanCumSum_Scalar_Returns1D()
        {
            // np.nancumsum(5.0) == array([5.]) shape (1,)
            var r = np.nancumsum(NDArray.Scalar(5.0));
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(1, r.shape[0]);
            r.Should().Be(np.array(new[] { 5.0 }));
        }

        [TestMethod]
        public void NanCumSum_Empty_ReturnsEmpty()
        {
            var r = np.nancumsum(np.array(Array.Empty<double>()));
            Assert.AreEqual(0, r.size);
        }

        [TestMethod]
        public void NanCumSum_Inf_Preserved()
        {
            // inf is NOT NaN -> kept: np.nancumsum([1, inf, 2]) == [1, inf, inf]
            var r = np.nancumsum(np.array(new[] { 1.0, double.PositiveInfinity, 2.0 }));
            r.Should().Be(np.array(new[] { 1.0, double.PositiveInfinity, double.PositiveInfinity }));
        }

        [TestMethod]
        public void NanCumSum_Out_ReturnsSameInstance()
        {
            // out= is filled and returned; result accumulates in float then cast into out
            var o = np.zeros(3, NPTypeCode.Double);
            var r = np.nancumsum(np.array(new[] { 1.0, double.NaN, 3.0 }), @out: o);
            Assert.IsTrue(ReferenceEquals(r, o));
            o.Should().Be(np.array(new[] { 1.0, 1.0, 4.0 }));
        }

        [TestMethod]
        public void NanCumSum_Out_IntCast_AccumulatesInFloatThenCasts()
        {
            // out=int64 (no dtype): accumulate in float64 [1.5,1.5,5.0] then cast unsafe -> [1,1,5]
            var oi = np.zeros(3, NPTypeCode.Int64);
            np.nancumsum(np.array(new[] { 1.5, double.NaN, 3.5 }), @out: oi);
            oi.Should().Be(np.array(new[] { 1L, 1L, 5L }));
        }

        [TestMethod]
        public void NanCumSum_Strided_Every2nd()
        {
            // base[::2] of [1,nan,2,nan,3,4] -> [1,2,3]; nancumsum -> [1,3,6]
            var basea = np.array(new[] { 1.0, double.NaN, 2.0, double.NaN, 3.0, 4.0 });
            var r = np.nancumsum(basea["::2"]);
            r.Should().Be(np.array(new[] { 1.0, 3.0, 6.0 }));
        }

        [TestMethod]
        public void NanCumSum_NegativeStride_Reversed()
        {
            // [1,nan,2,nan,3,4][::-1] -> [4,3,nan,2,nan,1]; nancumsum -> [4,7,7,9,9,10]
            var basea = np.array(new[] { 1.0, double.NaN, 2.0, double.NaN, 3.0, 4.0 });
            var r = np.nancumsum(basea["::-1"]);
            r.Should().Be(np.array(new[] { 4.0, 7.0, 7.0, 9.0, 9.0, 10.0 }));
        }

        [TestMethod]
        public void NanCumSum_Transposed_Axis0()
        {
            // m = [[1,nan,3],[4,5,nan]]; m.T axis0 nancumsum
            var m = np.array(new[,] { { 1.0, double.NaN, 3.0 }, { 4.0, 5.0, double.NaN } });
            var r = np.nancumsum(m.T, axis: 0);
            r.Should().Be(np.array(new[,] { { 1.0, 4.0 }, { 1.0, 9.0 }, { 4.0, 9.0 } }));
        }

        [TestMethod]
        public void NanCumSum_Broadcast_Axis1()
        {
            // broadcast [1,nan,3] to (2,3); nancumsum axis1 -> [[1,1,4],[1,1,4]]
            var b = np.broadcast_to(np.array(new[] { 1.0, double.NaN, 3.0 }), new Shape(2, 3));
            var r = np.nancumsum(b, axis: 1);
            r.Should().Be(np.array(new[,] { { 1.0, 1.0, 4.0 }, { 1.0, 1.0, 4.0 } }));
        }

        [TestMethod]
        public void NanCumSum_AxisOutOfBounds_Throws()
        {
            Assert.ThrowsException<ArgumentOutOfRangeException>(
                () => np.nancumsum(np.array(new[] { 1.0, 2.0 }), axis: 5));
        }

        #endregion

        #region np.nancumprod

        [TestMethod]
        public void NanCumProd_Basic1D_IgnoresNaN()
        {
            // np.nancumprod([1., nan, 3.]) == [1., 1., 3.]
            var r = np.nancumprod(np.array(new[] { 1.0, double.NaN, 3.0 }));
            r.Should().BeOfType(NPTypeCode.Double);
            r.Should().Be(np.array(new[] { 1.0, 1.0, 3.0 }));
        }

        [TestMethod]
        public void NanCumProd_LeadingNaN_ReplacedByOne()
        {
            // np.nancumprod([nan, nan, 2., 3.]) == [1., 1., 2., 6.]
            var r = np.nancumprod(np.array(new[] { double.NaN, double.NaN, 2.0, 3.0 }));
            r.Should().Be(np.array(new[] { 1.0, 1.0, 2.0, 6.0 }));
        }

        [TestMethod]
        public void NanCumProd_AllNaN_ReturnsOnes()
        {
            // np.nancumprod([nan, nan]) == [1., 1.]
            var r = np.nancumprod(np.array(new[] { double.NaN, double.NaN }));
            r.Should().Be(np.array(new[] { 1.0, 1.0 }));
        }

        [TestMethod]
        public void NanCumProd_NoNaN_SameAsCumProd()
        {
            var r = np.nancumprod(np.array(new[] { 1.0, 2.0, 3.0, 4.0 }));
            r.Should().Be(np.array(new[] { 1.0, 2.0, 6.0, 24.0 }));
        }

        [TestMethod]
        public void NanCumProd_2D_Axis0()
        {
            // a = [[1,2],[3,nan]]; np.nancumprod(a, axis=0) == [[1,2],[3,2]]
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumprod(a, axis: 0);
            r.Should().Be(np.array(new[,] { { 1.0, 2.0 }, { 3.0, 2.0 } }));
        }

        [TestMethod]
        public void NanCumProd_2D_Axis1()
        {
            // np.nancumprod(a, axis=1) == [[1,2],[3,3]]
            var a = np.array(new[,] { { 1.0, 2.0 }, { 3.0, double.NaN } });
            var r = np.nancumprod(a, axis: 1);
            r.Should().Be(np.array(new[,] { { 1.0, 2.0 }, { 3.0, 3.0 } }));
        }

        [TestMethod]
        public void NanCumProd_Complex_ReplacesWholeElement()
        {
            // input [1+2j, nan+5j, 3+0j, nan+nanj]; NaN -> 1+0j
            // np.nancumprod -> [1+2j, 1+2j, 3+6j, 3+6j]
            var c = np.array(new Complex[] { new(1, 2), new(double.NaN, 5), new(3, 0), new(double.NaN, double.NaN) });
            var r = np.nancumprod(c);
            r.Should().BeOfType(NPTypeCode.Complex);
            r.Should().Be(np.array(new Complex[] { new(1, 2), new(1, 2), new(3, 6), new(3, 6) }));
        }

        [TestMethod]
        public void NanCumProd_Float16_PreservesDtype()
        {
            // np.nancumprod([1., nan, 2.] float16) == [1., 1., 2.]
            var r = np.nancumprod(np.array(new[] { (Half)1.0, Half.NaN, (Half)2.0 }));
            r.Should().BeOfType(NPTypeCode.Half);
            r.Should().Be(np.array(new[] { (Half)1.0, (Half)1.0, (Half)2.0 }));
        }

        [TestMethod]
        public void NanCumProd_Int_Passthrough_WidensToInt64()
        {
            // np.nancumprod([1,2,3] int32) == [1,2,6] int64
            var r = np.nancumprod(np.array(new[] { 1, 2, 3 }));
            r.Should().BeOfType(NPTypeCode.Int64);
            r.Should().Be(np.array(new[] { 1L, 2L, 6L }));
        }

        [TestMethod]
        public void NanCumProd_ZeroThenNaN_Propagates()
        {
            // np.nancumprod([0., nan, 5.]) == [0., 0., 0.] (NaN->1, but 0*1*5 chain stays 0)
            var r = np.nancumprod(np.array(new[] { 0.0, double.NaN, 5.0 }));
            r.Should().Be(np.array(new[] { 0.0, 0.0, 0.0 }));
        }

        [TestMethod]
        public void NanCumProd_Scalar_Returns1D()
        {
            // np.nancumprod(nan) == array([1.]) shape (1,)
            var r = np.nancumprod(NDArray.Scalar(double.NaN));
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(1, r.shape[0]);
            r.Should().Be(np.array(new[] { 1.0 }));
        }

        [TestMethod]
        public void NanCumProd_Dtype_Float32()
        {
            var r = np.nancumprod(np.array(new[] { 2.0, double.NaN, 4.0 }), typeCode: NPTypeCode.Single);
            r.Should().BeOfType(NPTypeCode.Single);
            r.Should().Be(np.array(new[] { 2.0f, 2.0f, 8.0f }));
        }

        #endregion

        #region np.cumprod scalar regression (root-cause fix shared by nancumprod)

        [TestMethod]
        public void CumProd_Scalar_Int32_Returns1D_Int64()
        {
            // np.cumprod(np.int32(5)) == array([5]) int64 shape (1,) — was 0-d int32 (NEP50-skip bug)
            var r = np.cumprod(NDArray.Scalar(5));
            r.Should().BeOfType(NPTypeCode.Int64);
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(1, r.shape[0]);
            r.Should().Be(np.array(new[] { 5L }));
        }

        [TestMethod]
        public void CumProd_Scalar_Float64_Returns1D()
        {
            // np.cumprod(np.float64(5.0)) == array([5.]) shape (1,)
            var r = np.cumprod(NDArray.Scalar(5.0));
            r.Should().BeOfType(NPTypeCode.Double);
            Assert.AreEqual(1, r.ndim);
            Assert.AreEqual(1, r.shape[0]);
        }

        [TestMethod]
        public void CumProd_SingleElementInt_WidensToInt64()
        {
            // np.cumprod(np.array([7], dtype=int32)) == array([7]) int64 (1,)
            var r = np.cumprod(np.array(new[] { 7 }));
            r.Should().BeOfType(NPTypeCode.Int64);
            r.Should().Be(np.array(new[] { 7L }));
        }

        #endregion
    }
}
