using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Comparison and conversion tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesComparisonTests
    {
        #region SByte Comparisons

        [TestMethod]
        public void SByte_Equal()
        {
            // NumPy: np.array([1, 2, 3], dtype=np.int8) == np.array([2, 2, 2], dtype=np.int8)
            // Result: [False, True, False]
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = np.array(new sbyte[] { 2, 2, 2 });
            var result = a == b;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeTrue();
            result.GetAtIndex<bool>(2).Should().BeFalse();
        }

        [TestMethod]
        public void SByte_LessThan()
        {
            // NumPy: np.array([1, 2, 3], dtype=np.int8) < np.array([2, 2, 2], dtype=np.int8)
            // Result: [True, False, False]
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = np.array(new sbyte[] { 2, 2, 2 });
            var result = a < b;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeTrue();
            result.GetAtIndex<bool>(1).Should().BeFalse();
            result.GetAtIndex<bool>(2).Should().BeFalse();
        }

        [TestMethod]
        public void SByte_GreaterThan()
        {
            // NumPy: np.array([1, 2, 3], dtype=np.int8) > np.array([2, 2, 2], dtype=np.int8)
            // Result: [False, False, True]
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var b = np.array(new sbyte[] { 2, 2, 2 });
            var result = a > b;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeFalse();
            result.GetAtIndex<bool>(2).Should().BeTrue();
        }

        #endregion

        #region Half Comparisons

        [TestMethod]
        public void Half_Equal()
        {
            var h1 = np.array(new Half[] { (Half)1.0, (Half)2.0, Half.NaN });
            var h2 = np.array(new Half[] { (Half)1.0, (Half)3.0, Half.NaN });
            var result = h1 == h2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeTrue();
            result.GetAtIndex<bool>(1).Should().BeFalse();
            result.GetAtIndex<bool>(2).Should().BeFalse(); // NaN == NaN is False
        }

        [TestMethod]
        public void Half_LessThan_WithNaN()
        {
            var h1 = np.array(new Half[] { (Half)1.0, (Half)2.0, Half.NaN });
            var h2 = np.array(new Half[] { (Half)1.0, (Half)3.0, Half.NaN });
            var result = h1 < h2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeTrue();
            result.GetAtIndex<bool>(2).Should().BeFalse(); // NaN < NaN is False
        }

        #endregion

        #region Complex Comparisons

        [TestMethod]
        public void Complex_Equal()
        {
            // NumPy: complex == complex uses exact equality
            var z1 = np.array(new Complex[] { new(1, 2), new(3, 4) });
            var z2 = np.array(new Complex[] { new(1, 2), new(2, 3) });
            var result = z1 == z2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeTrue();
            result.GetAtIndex<bool>(1).Should().BeFalse();
        }

        [TestMethod]
        public void Complex_LessThan_Lexicographic()
        {
            // NumPy 2.x: complex < uses lexicographic ordering (first by real, then imaginary)
            // c1: [1+2j, 3+4j, 1+5j, 2+0j]
            // c2: [1+3j, 2+4j, 1+5j, 1+0j]
            // Result: [True, False, False, False]
            // (1,2) < (1,3): same real, 2<3 => True
            // (3,4) < (2,4): 3>2 => False
            // (1,5) < (1,5): equal => False
            // (2,0) < (1,0): 2>1 => False
            var c1 = np.array(new Complex[] { new(1, 2), new(3, 4), new(1, 5), new(2, 0) });
            var c2 = np.array(new Complex[] { new(1, 3), new(2, 4), new(1, 5), new(1, 0) });
            var result = c1 < c2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeTrue();
            result.GetAtIndex<bool>(1).Should().BeFalse();
            result.GetAtIndex<bool>(2).Should().BeFalse();
            result.GetAtIndex<bool>(3).Should().BeFalse();
        }

        [TestMethod]
        public void Complex_GreaterThan_Lexicographic()
        {
            // NumPy 2.x: complex > uses lexicographic ordering
            // c1: [1+2j, 3+4j, 1+5j, 2+0j]
            // c2: [1+3j, 2+4j, 1+5j, 1+0j]
            // Result: [False, True, False, True]
            var c1 = np.array(new Complex[] { new(1, 2), new(3, 4), new(1, 5), new(2, 0) });
            var c2 = np.array(new Complex[] { new(1, 3), new(2, 4), new(1, 5), new(1, 0) });
            var result = c1 > c2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeTrue();
            result.GetAtIndex<bool>(2).Should().BeFalse();
            result.GetAtIndex<bool>(3).Should().BeTrue();
        }

        [TestMethod]
        public void Complex_LessEqual_Lexicographic()
        {
            // NumPy 2.x: complex <= uses lexicographic ordering
            // c1: [1+2j, 3+4j, 1+5j, 2+0j]
            // c2: [1+3j, 2+4j, 1+5j, 1+0j]
            // Result: [True, False, True, False]
            var c1 = np.array(new Complex[] { new(1, 2), new(3, 4), new(1, 5), new(2, 0) });
            var c2 = np.array(new Complex[] { new(1, 3), new(2, 4), new(1, 5), new(1, 0) });
            var result = c1 <= c2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeTrue();
            result.GetAtIndex<bool>(1).Should().BeFalse();
            result.GetAtIndex<bool>(2).Should().BeTrue();
            result.GetAtIndex<bool>(3).Should().BeFalse();
        }

        [TestMethod]
        public void Complex_GreaterEqual_Lexicographic()
        {
            // NumPy 2.x: complex >= uses lexicographic ordering
            // c1: [1+2j, 3+4j, 1+5j, 2+0j]
            // c2: [1+3j, 2+4j, 1+5j, 1+0j]
            // Result: [False, True, True, True]
            var c1 = np.array(new Complex[] { new(1, 2), new(3, 4), new(1, 5), new(2, 0) });
            var c2 = np.array(new Complex[] { new(1, 3), new(2, 4), new(1, 5), new(1, 0) });
            var result = c1 >= c2;

            result.typecode.Should().Be(NPTypeCode.Boolean);
            result.GetAtIndex<bool>(0).Should().BeFalse();
            result.GetAtIndex<bool>(1).Should().BeTrue();
            result.GetAtIndex<bool>(2).Should().BeTrue();
            result.GetAtIndex<bool>(3).Should().BeTrue();
        }

        #endregion

        #region astype Conversions

        [TestMethod]
        public void SByte_AsType_ToHalf()
        {
            // NumPy: np.array([1, 2, 3], dtype=np.int8).astype(np.float16)
            // Result: [1.0, 2.0, 3.0]
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var result = a.astype(NPTypeCode.Half);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)3.0);
        }

        [TestMethod]
        public void SByte_AsType_ToComplex()
        {
            // NumPy: np.array([1, 2, 3], dtype=np.int8).astype(np.complex128)
            // Result: [1+0j, 2+0j, 3+0j]
            var a = np.array(new sbyte[] { 1, 2, 3 });
            var result = a.astype(NPTypeCode.Complex);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(2, 0));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(3, 0));
        }

        [TestMethod]
        public void SByte_AsType_ToInt32()
        {
            var a = np.array(new sbyte[] { -128, 0, 127 });
            var result = a.astype(NPTypeCode.Int32);

            result.typecode.Should().Be(NPTypeCode.Int32);
            result.GetAtIndex<int>(0).Should().Be(-128);
            result.GetAtIndex<int>(1).Should().Be(0);
            result.GetAtIndex<int>(2).Should().Be(127);
        }

        [TestMethod]
        public void Half_AsType_ToSByte()
        {
            // NumPy: np.array([1.5, 2.5, 3.5], dtype=np.float16).astype(np.int8)
            // Result: [1, 2, 3] (truncates)
            var h = np.array(new Half[] { (Half)1.5, (Half)2.5, (Half)3.5 });
            var result = h.astype(NPTypeCode.SByte);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)2);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)3);
        }

        [TestMethod]
        public void Half_AsType_ToDouble()
        {
            var h = np.array(new Half[] { (Half)1.5, (Half)2.5, (Half)3.5 });
            var result = h.astype(NPTypeCode.Double);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().BeApproximately(1.5, 0.001);
            result.GetAtIndex<double>(1).Should().BeApproximately(2.5, 0.001);
            result.GetAtIndex<double>(2).Should().BeApproximately(3.5, 0.001);
        }

        [TestMethod]
        [OpenBugs] // Half to Complex conversion not supported yet
        public void Half_AsType_ToComplex()
        {
            // NumPy: np.array([1.5, 2.5, 3.5], dtype=np.float16).astype(np.complex128)
            // Result: [1.5+0j, 2.5+0j, 3.5+0j]
            var h = np.array(new Half[] { (Half)1.5, (Half)2.5, (Half)3.5 });
            var result = h.astype(NPTypeCode.Complex);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Real.Should().BeApproximately(1.5, 0.001);
            result.GetAtIndex<Complex>(1).Real.Should().BeApproximately(2.5, 0.001);
            result.GetAtIndex<Complex>(2).Real.Should().BeApproximately(3.5, 0.001);
            result.GetAtIndex<Complex>(0).Imaginary.Should().Be(0);
        }

        [TestMethod]
        public void Complex_AsType_ToDouble_DiscardsImaginary()
        {
            // NumPy: np.array([1+2j, 3+4j]).astype(np.float64)
            // Result: [1.0, 3.0] with ComplexWarning (discards imaginary)
            var z = np.array(new Complex[] { new(1, 2), new(3, 4) });
            var result = z.astype(NPTypeCode.Double);

            result.typecode.Should().Be(NPTypeCode.Double);
            result.GetAtIndex<double>(0).Should().Be(1.0);
            result.GetAtIndex<double>(1).Should().Be(3.0);
        }

        #endregion

        #region Power Operations

        [TestMethod]
        public void SByte_Power()
        {
            // NumPy: np.power([1, 2, 3, 4], 2, dtype=int8)
            // Result: [1, 4, 9, 16] (dtype: int8)
            var a = np.array(new sbyte[] { 1, 2, 3, 4 });
            var result = np.power(a, 2);

            result.typecode.Should().Be(NPTypeCode.SByte);
            result.GetAtIndex<sbyte>(0).Should().Be((sbyte)1);
            result.GetAtIndex<sbyte>(1).Should().Be((sbyte)4);
            result.GetAtIndex<sbyte>(2).Should().Be((sbyte)9);
            result.GetAtIndex<sbyte>(3).Should().Be((sbyte)16);
        }

        [TestMethod]
        public void Half_Power()
        {
            // NumPy: np.power([1, 2, 3, 4], 2, dtype=float16)
            // Result: [1.0, 4.0, 9.0, 16.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0 });
            var result = np.power(h, 2);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)4.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)9.0);
            result.GetAtIndex<Half>(3).Should().Be((Half)16.0);
        }

        [TestMethod]
        public void Complex_Power()
        {
            // NumPy: np.power([1+0j, 0+1j, 1+1j], 2)
            // Result: [1+0j, -1+0j, 0+2j]
            var z = np.array(new Complex[] { new(1, 0), new(0, 1), new(1, 1) });
            var result = np.power(z, 2);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 0));
            result.GetAtIndex<Complex>(1).Real.Should().BeApproximately(-1, 0.0001);
            result.GetAtIndex<Complex>(1).Imaginary.Should().BeApproximately(0, 0.0001);
            result.GetAtIndex<Complex>(2).Real.Should().BeApproximately(0, 0.0001);
            result.GetAtIndex<Complex>(2).Imaginary.Should().BeApproximately(2, 0.0001);
        }

        #endregion
    }
}
