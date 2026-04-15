using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Edge case tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesEdgeCaseTests
    {
        #region Half Special Values

        [TestMethod]
        [OpenBugs] // isinf/isnan/isfinite not supported for Half yet
        public void Half_Infinity_Operations()
        {
            var h = np.array(new Half[] { Half.PositiveInfinity, Half.NegativeInfinity, Half.NaN, (Half)0.0 });

            // np.isinf
            var isinf = np.isinf(h);
            isinf.GetAtIndex<bool>(0).Should().BeTrue();
            isinf.GetAtIndex<bool>(1).Should().BeTrue();
            isinf.GetAtIndex<bool>(2).Should().BeFalse();
            isinf.GetAtIndex<bool>(3).Should().BeFalse();

            // np.isnan
            var isnan = np.isnan(h);
            isnan.GetAtIndex<bool>(0).Should().BeFalse();
            isnan.GetAtIndex<bool>(1).Should().BeFalse();
            isnan.GetAtIndex<bool>(2).Should().BeTrue();
            isnan.GetAtIndex<bool>(3).Should().BeFalse();

            // np.isfinite
            var isfinite = np.isfinite(h);
            isfinite.GetAtIndex<bool>(0).Should().BeFalse();
            isfinite.GetAtIndex<bool>(1).Should().BeFalse();
            isfinite.GetAtIndex<bool>(2).Should().BeFalse();
            isfinite.GetAtIndex<bool>(3).Should().BeTrue();
        }

        [TestMethod]
        public void Half_NaN_Comparisons()
        {
            // NumPy: NaN == NaN is False, NaN < x is False
            var h1 = np.array(new Half[] { (Half)1.0, (Half)2.0, Half.NaN });
            var h2 = np.array(new Half[] { (Half)1.0, (Half)3.0, Half.NaN });

            var eq = h1 == h2;
            eq.GetAtIndex<bool>(0).Should().BeTrue();
            eq.GetAtIndex<bool>(1).Should().BeFalse();
            eq.GetAtIndex<bool>(2).Should().BeFalse(); // NaN == NaN is False

            var lt = h1 < h2;
            lt.GetAtIndex<bool>(0).Should().BeFalse();
            lt.GetAtIndex<bool>(1).Should().BeTrue();
            lt.GetAtIndex<bool>(2).Should().BeFalse(); // NaN < NaN is False
        }

        #endregion

        #region Complex Special Values

        [TestMethod]
        [OpenBugs] // isinf/isnan not supported for Complex yet
        public void Complex_Infinity_Operations()
        {
            var z = np.array(new Complex[] {
                new(0, 0),
                new(1, 0),
                new(0, 1),
                new(double.PositiveInfinity, 0),
                new(double.NaN, 0)
            });

            // np.abs - should handle special values
            var absZ = np.abs(z);
            absZ.GetAtIndex<double>(0).Should().Be(0.0);
            absZ.GetAtIndex<double>(1).Should().Be(1.0);
            absZ.GetAtIndex<double>(2).Should().Be(1.0);
            double.IsPositiveInfinity(absZ.GetAtIndex<double>(3)).Should().BeTrue();
            double.IsNaN(absZ.GetAtIndex<double>(4)).Should().BeTrue();

            // np.isinf
            var isinf = np.isinf(z);
            isinf.GetAtIndex<bool>(0).Should().BeFalse();
            isinf.GetAtIndex<bool>(1).Should().BeFalse();
            isinf.GetAtIndex<bool>(2).Should().BeFalse();
            isinf.GetAtIndex<bool>(3).Should().BeTrue();
            isinf.GetAtIndex<bool>(4).Should().BeFalse();

            // np.isnan
            var isnan = np.isnan(z);
            isnan.GetAtIndex<bool>(0).Should().BeFalse();
            isnan.GetAtIndex<bool>(1).Should().BeFalse();
            isnan.GetAtIndex<bool>(2).Should().BeFalse();
            isnan.GetAtIndex<bool>(3).Should().BeFalse();
            isnan.GetAtIndex<bool>(4).Should().BeTrue();
        }

        #endregion

        #region All/Any

        [TestMethod]
        public void SByte_All_Any()
        {
            // NumPy: np.all([0, 1, 2], dtype=int8) = False
            // NumPy: np.any([0, 1, 2], dtype=int8) = True
            var a = np.array(new sbyte[] { 0, 1, 2 });
            np.all(a).Should().BeFalse();
            np.any(a).Should().BeTrue();

            // All non-zero
            var a2 = np.array(new sbyte[] { 1, 2, 3 });
            np.all(a2).Should().BeTrue();
        }

        [TestMethod]
        public void Half_All_Any()
        {
            // NumPy: np.all([0.0, 1.0, nan], dtype=float16) = False (0.0 is falsy)
            // NumPy: np.any([0.0, 1.0, nan], dtype=float16) = True
            var h = np.array(new Half[] { (Half)0.0, (Half)1.0, Half.NaN });
            np.all(h).Should().BeFalse();
            np.any(h).Should().BeTrue();
        }

        [TestMethod]
        public void Complex_All_Any()
        {
            // NumPy: np.all([0+0j, 1+0j, 0+1j]) = False (0+0j is falsy)
            // NumPy: np.any([0+0j, 1+0j, 0+1j]) = True
            var z = np.array(new Complex[] { new(0, 0), new(1, 0), new(0, 1) });
            np.all(z).Should().BeFalse();
            np.any(z).Should().BeTrue();
        }

        #endregion

        #region Count Nonzero

        [TestMethod]
        public void SByte_CountNonzero()
        {
            // NumPy: np.count_nonzero([0, 1, 0, 2, 0], dtype=int8) = 2
            var a = np.array(new sbyte[] { 0, 1, 0, 2, 0 });
            var result = np.count_nonzero(a);
            result.Should().Be(2);
        }

        [TestMethod]
        public void Half_CountNonzero()
        {
            // NumPy: np.count_nonzero([0.0, 1.0, 0.0, nan], dtype=float16) = 2
            // Note: NaN is considered nonzero
            var h = np.array(new Half[] { (Half)0.0, (Half)1.0, (Half)0.0, Half.NaN });
            var result = np.count_nonzero(h);
            result.Should().Be(2);
        }

        [TestMethod]
        public void Complex_CountNonzero()
        {
            // NumPy: np.count_nonzero([0+0j, 1+0j, 0+1j, 0+0j]) = 2
            var z = np.array(new Complex[] { new(0, 0), new(1, 0), new(0, 1), new(0, 0) });
            var result = np.count_nonzero(z);
            result.Should().Be(2);
        }

        #endregion

        #region Broadcasting

        [TestMethod]
        public void SByte_Broadcasting()
        {
            // NumPy: int8 [[1], [2], [3]] + [10, 20, 30] = [[11, 21, 31], [12, 22, 32], [13, 23, 33]]
            var a = np.array(new sbyte[,] { { 1 }, { 2 }, { 3 } });
            var b = np.array(new sbyte[] { 10, 20, 30 });
            var result = a + b;

            result.shape.Should().BeEquivalentTo(new[] { 3, 3 });
            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)11);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)21);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)31);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)12);
            result.GetAtIndex<sbyte>(8).Should().Be((sbyte)33);
        }

        [TestMethod]
        public void Half_Broadcasting()
        {
            // NumPy: float16 [[1.0], [2.0]] + [0.5, 1.5] = [[1.5, 2.5], [2.5, 3.5]]
            var h1 = np.array(new Half[,] { { (Half)1.0 }, { (Half)2.0 } });
            var h2 = np.array(new Half[] { (Half)0.5, (Half)1.5 });
            var result = h1 + h2;

            result.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.5);
            result.GetAtIndex<Half>(1).Should().Be((Half)2.5);
            result.GetAtIndex<Half>(2).Should().Be((Half)2.5);
            result.GetAtIndex<Half>(3).Should().Be((Half)3.5);
        }

        #endregion

        #region Slicing

        [TestMethod]
        public void SByte_Slicing()
        {
            // NumPy: slicing preserves dtype
            var a = np.array(new sbyte[] { 1, 2, 3, 4, 5 });

            var slice1 = a["1:4"];
            slice1.typecode.Should().Be(NPTypeCode.SByte);
            slice1.GetAtIndex<sbyte>(0).Should().Be((sbyte)2);
            slice1.GetAtIndex<sbyte>(1).Should().Be((sbyte)3);
            slice1.GetAtIndex<sbyte>(2).Should().Be((sbyte)4);

            var slice2 = a["::2"];
            slice2.typecode.Should().Be(NPTypeCode.SByte);
            slice2.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            slice2.GetAtIndex<sbyte>(1).Should().Be((sbyte)3);
            slice2.GetAtIndex<sbyte>(2).Should().Be((sbyte)5);
        }

        [TestMethod]
        public void Half_Slicing()
        {
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });

            var slice = h["1:4"];
            slice.typecode.Should().Be(NPTypeCode.Half);
            slice.GetAtIndex<Half>(0).Should().Be((Half)2.0);
            slice.GetAtIndex<Half>(1).Should().Be((Half)3.0);
            slice.GetAtIndex<Half>(2).Should().Be((Half)4.0);
        }

        [TestMethod]
        public void Complex_Slicing()
        {
            var z = np.array(new Complex[] { new(1, 1), new(2, 2), new(3, 3), new(4, 4) });

            var slice = z["1:3"];
            slice.typecode.Should().Be(NPTypeCode.Complex);
            slice.GetAtIndex<Complex>(0).Should().Be(new Complex(2, 2));
            slice.GetAtIndex<Complex>(1).Should().Be(new Complex(3, 3));
        }

        #endregion

        #region Dot/MatMul

        [TestMethod]
        [OpenBugs] // Dot not supported for SByte yet
        public void SByte_Dot()
        {
            // NumPy: np.dot([1, 2, 3], [4, 5, 6], dtype=int8) = 32 (dtype: int8)
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = np.array(new sbyte[] { 4, 5, 6 });
            var result = np.dot(a, b);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)32);
        }

        [TestMethod]
        public void Half_Dot()
        {
            // NumPy: np.dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], dtype=float16) = 32.0 (dtype: float16)
            var h1 = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0 });
            var h2 = np.array(new Half[] { (Half)4.0, (Half)5.0, (Half)6.0 });
            var result = np.dot(h1, h2);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)32.0);
        }

        [TestMethod]
        [OpenBugs] // Dot not supported for Complex (multiply not working)
        public void Complex_Dot()
        {
            // NumPy: np.dot([1+1j, 2+2j], [1-1j, 2-2j]) = (10+0j)
            var z1 = np.array(new Complex[] { new(1, 1), new(2, 2) });
            var z2 = np.array(new Complex[] { new(1, -1), new(2, -2) });
            var result = np.dot(z1, z2);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(10, 0));
        }

        [TestMethod]
        public void SByte_MatMul_2x2()
        {
            // NumPy: np.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]], dtype=int8) = [[19, 22], [43, 50]]
            var a = np.array(new sbyte[,] { { 1, 2 }, { 3, 4 } });
            var b = np.array(new sbyte[,] { { 5, 6 }, { 7, 8 } });
            var result = np.matmul(a, b);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.shape.Should().BeEquivalentTo(new[] { 2, 2 });
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)19);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)22);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)43);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)50);
        }

        #endregion
    }
}
