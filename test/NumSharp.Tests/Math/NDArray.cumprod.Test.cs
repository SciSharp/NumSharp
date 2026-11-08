using NumSharp.Tests.Utilities;

namespace NumSharp.Tests.Math
{
    /// <summary>
    /// Tests for cumulative product (cumprod) functionality.
    /// These test the DirectILKernelGenerator.Scan.cs CumProd implementation.
    /// </summary>
    [TestClass]
    public class NDArrayCumprodTest : TestClass
    {
        /// <summary>
        ///     Along a length-1 axis multiply.accumulate still has its NEP 50 accumulator (parity QA-09): bool and the
        ///     signed integers narrower than int64 give int64, the unsigned ones (char, the uint16 masquerade, with them)
        ///     uint64, and floats keep their dtype — through np.cumprod, ndarray.cumprod and np.nancumprod alike — and the
        ///     result is a fresh copy of the values. NumPy 2.4.2:
        ///     <c>np.cumprod(np.arange(6).astype(d).reshape(2, 1, 3), axis=1)</c> (and its two twins).
        /// </summary>
        [TestMethod]
        public void CumprodLengthOneAxis_PromotesLikeMultiplyAccumulate()
        {
            var cases = new (NPTypeCode Input, NPTypeCode Expected)[]
            {
                (NPTypeCode.Boolean, NPTypeCode.Int64), (NPTypeCode.SByte, NPTypeCode.Int64), (NPTypeCode.Byte, NPTypeCode.UInt64),
                (NPTypeCode.Int16, NPTypeCode.Int64), (NPTypeCode.UInt16, NPTypeCode.UInt64), (NPTypeCode.Char, NPTypeCode.UInt64),
                (NPTypeCode.Int32, NPTypeCode.Int64), (NPTypeCode.UInt32, NPTypeCode.UInt64), (NPTypeCode.Int64, NPTypeCode.Int64),
                (NPTypeCode.UInt64, NPTypeCode.UInt64), (NPTypeCode.Half, NPTypeCode.Half), (NPTypeCode.Single, NPTypeCode.Single),
            };
            foreach (var (input, expected) in cases)
            {
                using var a = np.arange(6).astype(input).reshape(2, 1, 3);
                using var viaNp = np.cumprod(a, axis: 1);
                using var viaMember = a.cumprod(axis: 1);
                using var viaNan = np.nancumprod(a, axis: 1);
                foreach (var r in new[] { viaNp, viaMember, viaNan })
                {
                    r.typecode.Should().Be(expected, $"multiply.accumulate's accumulator for {input}");
                    r.shape.Should().Equal(new long[] { 2, 1, 3 });
                    np.shares_memory(r, a).Should().BeFalse("the accumulation is a fresh array, not the input");
                    // bool is [0, 1, 1, 1, 1, 1] (True is 1), every other dtype 0 … 5.
                    using var asDouble = r.astype(np.float64);
                    asDouble.ToArray<double>().Should().Equal(input == NPTypeCode.Boolean
                        ? new[] { 0.0, 1, 1, 1, 1, 1 }
                        : new[] { 0.0, 1, 2, 3, 4, 5 });
                }
            }
        }

        /// <summary>
        /// Basic cumprod test with double array.
        /// NumPy: np.cumprod([1, 2, 3, 4, 5]) = [1, 2, 6, 24, 120]
        /// </summary>
        [TestMethod]
        public void CumprodBasicDouble()
        {
            NDArray arr = new double[] { 1, 2, 3, 4, 5 };
            NDArray expected = new double[] { 1, 2, 6, 24, 120 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with zeros - product becomes 0 and stays 0.
        /// NumPy: np.cumprod([1, 2, 0, 4, 5]) = [1, 2, 0, 0, 0]
        /// </summary>
        [TestMethod]
        public void CumprodWithZero()
        {
            NDArray arr = new double[] { 1, 2, 0, 4, 5 };
            NDArray expected = new double[] { 1, 2, 0, 0, 0 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with negative numbers.
        /// NumPy: np.cumprod([1, -2, 3, -4]) = [1, -2, -6, 24]
        /// </summary>
        [TestMethod]
        public void CumprodWithNegatives()
        {
            NDArray arr = new double[] { 1, -2, 3, -4 };
            NDArray expected = new double[] { 1, -2, -6, 24 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod 2D without axis (flattens first).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]]) = [1, 2, 6, 24, 120, 720]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dFlattened()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[] { 1, 2, 6, 24, 120, 720 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod along axis 0 (rows).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]], axis=0) = [[1, 2, 3], [4, 10, 18]]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dAxis0()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[,] { { 1, 2, 3 }, { 4, 10, 18 } };

            var result = np.cumprod(arr, axis: 0);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod along axis 1 (columns).
        /// NumPy: np.cumprod([[1, 2, 3], [4, 5, 6]], axis=1) = [[1, 2, 6], [4, 20, 120]]
        /// For int32 input, NumPy 2.x returns int64.
        /// </summary>
        [TestMethod]
        public void Cumprod2dAxis1()
        {
            NDArray arr = new int[,] { { 1, 2, 3 }, { 4, 5, 6 } };
            // NumPy-aligned: cumprod of int32 returns int64
            NDArray expected = new long[,] { { 1, 2, 6 }, { 4, 20, 120 } };

            var result = np.cumprod(arr, axis: 1);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with single element.
        /// NumPy: np.cumprod([42]) = [42]
        /// </summary>
        [TestMethod]
        public void CumprodSingleElement()
        {
            NDArray arr = new double[] { 42 };
            NDArray expected = new double[] { 42 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with all ones.
        /// NumPy: np.cumprod([1, 1, 1, 1]) = [1, 1, 1, 1]
        /// </summary>
        [TestMethod]
        public void CumprodAllOnes()
        {
            NDArray arr = new double[] { 1, 1, 1, 1 };
            NDArray expected = new double[] { 1, 1, 1, 1 };

            var result = np.cumprod(arr);
            result.Should().Be(expected);
        }

        /// <summary>
        /// Cumprod with float32 type.
        /// </summary>
        [TestMethod]
        public void CumprodFloat32()
        {
            NDArray arr = np.array(new float[] { 1f, 2f, 3f, 4f });
            NDArray expected = np.array(new float[] { 1f, 2f, 6f, 24f });

            var result = np.cumprod(arr);
            result.typecode.Should().Be(NPTypeCode.Single);
            result.Should().Be(expected);
        }

        /// <summary>
        ///     Parity QA-09 (the dtype fixed 2026-11-05, the layout with the ufunc result-layout port, 2026-11-08):
        ///     <c>np.cumprod(int32 array of shape (2, 1, 3), axis=1)</c> is int64 in NumPy (multiply.accumulate's NEP 50
        ///     accumulator: signed integers narrower than int64 → int64, unsigned → uint64), whatever the axis length,
        ///     through <c>np.cumprod</c>, <c>ndarray.cumprod</c> and <c>np.nancumprod</c>. The length-1-axis shortcut used to
        ///     return an un-promoted C copy, then astype's K copy; it now casts into the accumulate's own layout
        ///     (<c>LengthOneAxisScan</c>: NumPy runs the same accumulate, no shortcut), as np.cumsum's twin does.
        /// </summary>
        [TestMethod]
        public void Cumprod_LengthOneAxis_PromotesNarrowIntegers()
        {
            using var i32 = np.arange(6).astype(np.int32).reshape(2, 1, 3);
            using var u8 = np.arange(6).astype(np.uint8).reshape(2, 1, 3);
            using var f = np.asfortranarray(np.ones(new Shape(1, 3, 2)));

            using var signedResult = np.cumprod(i32, axis: 1);
            using var unsignedResult = np.nancumprod(u8, axis: 1);
            using var memberResult = i32.cumprod(1);
            using var fResult = np.cumprod(f, axis: 0);

            signedResult.dtype.Should().Be(np.int64);
            unsignedResult.dtype.Should().Be(np.uint64);
            memberResult.dtype.Should().Be(np.int64);
            signedResult.ToArray<long>().Should().Equal(0L, 1L, 2L, 3L, 4L, 5L);
            // np.cumprod(np.asfortranarray(np.ones((1, 3, 2))), axis=0).strides == (48, 8, 24) in NumPy 2.4.2: the
            // accumulate's iterator orders the axes by the operand's strides, the extent-1 axis outermost.
            fResult.strides.Should().Equal(48L, 8L, 24L);
        }
    }
}
