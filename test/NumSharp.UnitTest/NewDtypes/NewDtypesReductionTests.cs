using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Reduction operation tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesReductionTests
    {
        #region SByte (int8) Reductions

        [TestMethod]
        public void SByte_Sum()
        {
            // NumPy: np.sum(np.array([-128, -1, 0, 1, 127], dtype=np.int8)) = -1 (dtype: int64)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.sum(a);

            result.typecode.Should().Be(NPTypeCode.Int64);
            result.GetAtIndex<long>(0).Should().Be(-1L);
        }

        [TestMethod]
        public void SByte_Prod()
        {
            // NumPy: np.prod(np.array([-128, -1, 0, 1, 127], dtype=np.int8)) = 0 (dtype: int64)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.prod(a);

            result.typecode.Should().Be(NPTypeCode.Int64);
            result.GetAtIndex<long>(0).Should().Be(0L);
        }

        [TestMethod]
        public void SByte_Mean()
        {
            // NumPy: np.mean(np.array([-128, -1, 0, 1, 127], dtype=np.int8)) = -0.2 (dtype: float64)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.mean(a);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(-0.2, 0.0001);
        }

        [TestMethod]
        public void SByte_Min()
        {
            // NumPy: np.min(np.array([-128, -1, 0, 1, 127], dtype=np.int8)) = -128 (dtype: int8)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.min(a);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)-128);
        }

        [TestMethod]
        public void SByte_Max()
        {
            // NumPy: np.max(np.array([-128, -1, 0, 1, 127], dtype=np.int8)) = 127 (dtype: int8)
            var a = np.array(new sbyte[] { -128, -1, 0, 1, 127 });
            var result = np.max(a);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)127);
        }

        [TestMethod]
        public void SByte_Std()
        {
            // NumPy: np.std(np.array([1, 2, 3, 4, 5], dtype=np.int8)) = 1.4142135623730951 (dtype: float64)
            var a = np.array(new sbyte[] { 1, 2, 3, 4, 5 });
            var result = np.std(a);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(1.4142135623730951, 0.0001);
        }

        [TestMethod]
        public void SByte_Var()
        {
            // NumPy: np.var(np.array([1, 2, 3, 4, 5], dtype=np.int8)) = 2.0 (dtype: float64)
            var a = np.array(new sbyte[] { 1, 2, 3, 4, 5 });
            var result = np.var(a);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(2.0, 0.0001);
        }

        [TestMethod]
        public void SByte_Sum_Axis()
        {
            // NumPy: np.sum(np.array([[-1, 2], [3, -4]], dtype=np.int8), axis=0) = [2, -2] (dtype: int64)
            // NumPy: np.sum(..., axis=1) = [1, -1] (dtype: int64)
            var c = np.array(new sbyte[,] { { -1, 2 }, { 3, -4 } });

            var axis0 = np.sum(c, axis: 0);
            axis0.typecode.Should().Be(NPTypeCode.Int64);
            axis0.GetAtIndex<long>(0).Should().Be(2L);
            axis0.GetAtIndex<long>(1).Should().Be(-2L);

            var axis1 = np.sum(c, axis: 1);
            axis1.typecode.Should().Be(NPTypeCode.Int64);
            axis1.GetAtIndex<long>(0).Should().Be(1L);
            axis1.GetAtIndex<long>(1).Should().Be(-1L);
        }

        [TestMethod]
        public void SByte_ArgMax()
        {
            // NumPy: np.argmax(np.array([-5, 10, 3, -2, 8], dtype=np.int8)) = 1
            var a = np.array(new sbyte[] { -5, 10, 3, -2, 8 });
            var result = np.argmax(a);
            result.Should().Be(1);
        }

        [TestMethod]
        public void SByte_ArgMin()
        {
            // NumPy: np.argmin(np.array([-5, 10, 3, -2, 8], dtype=np.int8)) = 0
            var a = np.array(new sbyte[] { -5, 10, 3, -2, 8 });
            var result = np.argmin(a);
            result.Should().Be(0);
        }

        #endregion

        #region Half (float16) Reductions

        [TestMethod]
        public void Half_Sum()
        {
            // NumPy: np.sum(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)) = 15.0 (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = np.sum(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)15.0);
        }

        [TestMethod]
        public void Half_Sum_WithNaN()
        {
            // NumPy: np.sum(np.array([0.0, 1.5, -2.5, nan, inf], dtype=np.float16)) = nan (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.5, (Half)(-2.5), Half.NaN, Half.PositiveInfinity });
            var result = np.sum(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            Half.IsNaN(result.GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // NaN-aware reductions not supported for Half yet
        public void Half_NanSum()
        {
            // NumPy: np.nansum(np.array([0.0, 1.5, -2.5, nan, inf], dtype=np.float16)) = inf (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.5, (Half)(-2.5), Half.NaN, Half.PositiveInfinity });
            var result = np.nansum(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            Half.IsPositiveInfinity(result.GetAtIndex<Half>(0)).Should().BeTrue();
        }

        [TestMethod]
        [OpenBugs] // Mean division not supported for Half yet
        public void Half_Mean()
        {
            // NumPy: np.mean(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)) = 3.0 (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = np.mean(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)3.0);
        }

        [TestMethod]
        [OpenBugs] // NaN-aware reductions not supported for Half yet
        public void Half_NanMin()
        {
            // NumPy: np.nanmin(np.array([0.0, 1.5, -2.5, nan, inf], dtype=np.float16)) = -2.5 (dtype: float16)
            var h = np.array(new Half[] { (Half)0.0, (Half)1.5, (Half)(-2.5), Half.NaN, Half.PositiveInfinity });
            var result = np.nanmin(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)(-2.5));
        }

        [TestMethod]
        [OpenBugs] // Std not supported for Half yet
        public void Half_Std()
        {
            // NumPy: np.std(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16)) = 1.4140625 (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = np.std(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            // float16 has limited precision
            ((double)result.GetAtIndex<Half>(0)).Should().BeApproximately(1.414, 0.01);
        }

        [TestMethod]
        public void Half_ArgMax()
        {
            // NumPy: np.argmax(np.array([1.5, 0.5, 2.5, 1.0], dtype=np.float16)) = 2
            var h = np.array(new Half[] { (Half)1.5, (Half)0.5, (Half)2.5, (Half)1.0 });
            var result = np.argmax(h);
            result.Should().Be(2);
        }

        [TestMethod]
        public void Half_ArgMin()
        {
            // NumPy: np.argmin(np.array([1.5, 0.5, 2.5, 1.0], dtype=np.float16)) = 1
            var h = np.array(new Half[] { (Half)1.5, (Half)0.5, (Half)2.5, (Half)1.0 });
            var result = np.argmin(h);
            result.Should().Be(1);
        }

        #endregion

        #region Complex (complex128) Reductions

        [TestMethod]
        public void Complex_Sum()
        {
            // NumPy: np.sum(np.array([1+2j, 3+4j, 0+0j, -1-1j])) = (3+5j) (dtype: complex128)
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = np.sum(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(3, 5));
        }

        [TestMethod]
        [OpenBugs] // Mean division not supported for Complex yet
        public void Complex_Mean()
        {
            // NumPy: np.mean(np.array([1+2j, 3+4j, 0+0j, -1-1j])) = (0.75+1.25j) (dtype: complex128)
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0), new(-1, -1) });
            var result = np.mean(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(0.75, 1.25));
        }

        [TestMethod]
        [OpenBugs] // Std not supported for Complex yet
        public void Complex_Std()
        {
            // NumPy: np.std(np.array([1+0j, 2+0j, 3+0j, 4+0j, 5+0j])) = 1.4142135623730951 (dtype: float64)
            var z = np.array(new Complex[] { new(1, 0), new(2, 0), new(3, 0), new(4, 0), new(5, 0) });
            var result = np.std(z);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(1.4142135623730951, 0.0001);
        }

        [TestMethod]
        [OpenBugs] // Axis reductions not supported for Complex yet
        public void Complex_Sum_Axis()
        {
            // NumPy: np.sum(np.array([[1+2j, 3+4j], [5+6j, 7+8j]]), axis=0) = [6+8j, 10+12j]
            // NumPy: np.sum(..., axis=1) = [4+6j, 12+14j]
            var zc = np.array(new Complex[,] { { new(1, 2), new(3, 4) }, { new(5, 6), new(7, 8) } });

            var axis0 = np.sum(zc, axis: 0);
            axis0.typecode.Should().Be(NPTypeCode.Complex);
            axis0.GetAtIndex<Complex>(0).Should().Be(new Complex(6, 8));
            axis0.GetAtIndex<Complex>(1).Should().Be(new Complex(10, 12));

            var axis1 = np.sum(zc, axis: 1);
            axis1.typecode.Should().Be(NPTypeCode.Complex);
            axis1.GetAtIndex<Complex>(0).Should().Be(new Complex(4, 6));
            axis1.GetAtIndex<Complex>(1).Should().Be(new Complex(12, 14));
        }

        [TestMethod]
        [OpenBugs] // ArgMax not supported for Complex yet
        public void Complex_ArgMax_ByMagnitude()
        {
            // NumPy: np.argmax(np.array([1+2j, 3+4j, 0+0j])) = 1 (by magnitude: [2.236, 5.0, 0.0])
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0) });
            var result = np.argmax(z);
            result.Should().Be(1);
        }

        [TestMethod]
        [OpenBugs] // ArgMin not supported for Complex yet
        public void Complex_ArgMin_ByMagnitude()
        {
            // NumPy: np.argmin(np.array([1+2j, 3+4j, 0+0j])) = 2 (by magnitude: [2.236, 5.0, 0.0])
            var z = np.array(new Complex[] { new(1, 2), new(3, 4), new(0, 0) });
            var result = np.argmin(z);
            result.Should().Be(2);
        }

        #endregion
    }
}
