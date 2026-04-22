using System;
using System.Numerics;
using AwesomeAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.Backends;

namespace NumSharp.UnitTest.NewDtypes
{
    /// <summary>
    /// Cumulative operation tests for SByte (int8), Half (float16), Complex (complex128)
    /// All expected values verified against NumPy 2.x
    /// </summary>
    [TestClass]
    public class NewDtypesCumulativeTests
    {
        #region SByte (int8) Cumulative

        [TestMethod]
        public void SByte_CumSum()
        {
            // NumPy: np.cumsum(np.array([1, 2, 3, 4, 5], dtype=np.int8))
            // Result: [1, 3, 6, 10, 15] (dtype: int64)
            var a = np.array(new sbyte[] { 1, 2, 3, 4, 5 });
            var result = np.cumsum(a);

            result.typecode.Should().Be(NPTypeCode.Int64);
            result.GetAtIndex<long>(0).Should().Be(1L);
            result.GetAtIndex<long>(1).Should().Be(3L);
            result.GetAtIndex<long>(2).Should().Be(6L);
            result.GetAtIndex<long>(3).Should().Be(10L);
            result.GetAtIndex<long>(4).Should().Be(15L);
        }

        [TestMethod]
        public void SByte_CumProd()
        {
            // NumPy: np.cumprod(np.array([1, 2, 3, 4, 5], dtype=np.int8))
            // Result: [1, 2, 6, 24, 120] (dtype: int64)
            var a = np.array(new sbyte[] { 1, 2, 3, 4, 5 });
            var result = np.cumprod(a);

            result.typecode.Should().Be(NPTypeCode.Int64);
            result.GetAtIndex<long>(0).Should().Be(1L);
            result.GetAtIndex<long>(1).Should().Be(2L);
            result.GetAtIndex<long>(2).Should().Be(6L);
            result.GetAtIndex<long>(3).Should().Be(24L);
            result.GetAtIndex<long>(4).Should().Be(120L);
        }

        #endregion

        #region Half (float16) Cumulative

        [TestMethod]
        public void Half_CumSum()
        {
            // NumPy: np.cumsum(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16))
            // Result: [1.0, 3.0, 6.0, 10.0, 15.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = np.cumsum(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)3.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)6.0);
            result.GetAtIndex<Half>(3).Should().Be((Half)10.0);
            result.GetAtIndex<Half>(4).Should().Be((Half)15.0);
        }

        [TestMethod]
        public void Half_CumProd()
        {
            // NumPy: np.cumprod(np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16))
            // Result: [1.0, 2.0, 6.0, 24.0, 120.0] (dtype: float16)
            var h = np.array(new Half[] { (Half)1.0, (Half)2.0, (Half)3.0, (Half)4.0, (Half)5.0 });
            var result = np.cumprod(h);

            result.typecode.Should().Be(NPTypeCode.Half);
            result.GetAtIndex<Half>(0).Should().Be((Half)1.0);
            result.GetAtIndex<Half>(1).Should().Be((Half)2.0);
            result.GetAtIndex<Half>(2).Should().Be((Half)6.0);
            result.GetAtIndex<Half>(3).Should().Be((Half)24.0);
            result.GetAtIndex<Half>(4).Should().Be((Half)120.0);
        }

        #endregion

        #region Complex (complex128) Cumulative

        [TestMethod]
        public void Complex_CumSum()
        {
            // NumPy: np.cumsum(np.array([1+1j, 2+2j, 3+3j]))
            // Result: [1+1j, 3+3j, 6+6j] (dtype: complex128)
            var z = np.array(new Complex[] { new(1, 1), new(2, 2), new(3, 3) });
            var result = np.cumsum(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 1));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(3, 3));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(6, 6));
        }

        [TestMethod]
        public void Complex_CumProd()
        {
            // NumPy: np.cumprod(np.array([1+1j, 2+2j, 3+3j]))
            // Result: [1+1j, 0+4j, -12+12j] (dtype: complex128)
            var z = np.array(new Complex[] { new(1, 1), new(2, 2), new(3, 3) });
            var result = np.cumprod(z);

            result.typecode.Should().Be(NPTypeCode.Complex);
            result.GetAtIndex<Complex>(0).Should().Be(new Complex(1, 1));
            result.GetAtIndex<Complex>(1).Should().Be(new Complex(0, 4));
            result.GetAtIndex<Complex>(2).Should().Be(new Complex(-12, 12));
        }

        #endregion
    }
}
