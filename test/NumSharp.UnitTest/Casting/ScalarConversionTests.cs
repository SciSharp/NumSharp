using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Tests for scalar NDArray to primitive type conversions.
    ///     These verify that implicit/explicit casts perform proper type conversion
    ///     rather than raw byte reinterpretation.
    /// </summary>
    public class ScalarConversionTests
    {
        /// <summary>
        ///     BUG 72 FIX: (double) cast on int64 scalar NDArray should convert, not reinterpret bytes.
        ///
        ///     NumPy:    float(np.sum(np.arange(10))) = 45.0
        ///     Fixed:    (double)(NDArray)sum = 45.0 (numeric conversion)
        ///     Was:      returns ~6.95e-310 (int64 bytes read as double)
        /// </summary>
        [Test]
        public void DoubleCast_Int64Scalar_ConvertsCorrectly()
        {
            // np.sum returns a scalar NDArray (ndim=0)
            var sum = np.sum(np.arange(10));
            sum.dtype.Should().Be(typeof(long), "NumPy 2.x: int32 sum accumulates to int64");
            sum.ndim.Should().Be(0, "np.sum without axis returns a scalar (ndim=0)");

            // Same-dtype cast works
            long longVal = (long)(NDArray)sum;
            longVal.Should().Be(45, "(long) cast on int64 scalar works");

            // Cross-dtype cast should convert, not reinterpret bytes
            double dblVal = (double)(NDArray)sum;
            dblVal.Should().Be(45.0,
                "Cross-dtype cast should perform numeric conversion. " +
                "Bug 72 caused raw int64 bytes to be read as double, returning garbage.");
        }

        /// <summary>
        ///     BUG 72 FIX: (double) cast on int32 scalar NDArray should convert correctly.
        /// </summary>
        [Test]
        public void DoubleCast_Int32Scalar_ConvertsCorrectly()
        {
            var scalar = NDArray.Scalar(42);
            scalar.dtype.Should().Be(typeof(int));
            scalar.ndim.Should().Be(0);

            double result = (double)(NDArray)scalar;
            result.Should().Be(42.0, "int32 scalar cast to double should convert correctly");
        }

        /// <summary>
        ///     Test various cross-dtype scalar conversions.
        /// </summary>
        /// <remarks>
        ///     Note: NumSharp uses IConvertible.ToXxx() which rounds to nearest,
        ///     while NumPy truncates for float->int. This is a known behavioral difference.
        ///     The main fix (Bug 72) is that cross-dtype conversion now works correctly
        ///     instead of reinterpreting raw bytes as the wrong type.
        /// </remarks>
        [Test]
        public void CrossDtypeConversions_WorkCorrectly()
        {
            // int64 -> double
            var int64Scalar = NDArray.Scalar(123L);
            ((double)(NDArray)int64Scalar).Should().Be(123.0);

            // double -> int (IConvertible rounds to nearest, NumPy truncates)
            var doubleScalar = NDArray.Scalar(3.7);
            ((int)(NDArray)doubleScalar).Should().Be(4, "IConvertible rounds to nearest (NumPy truncates - known difference)");

            // float -> long (IConvertible rounds to nearest)
            var floatScalar = NDArray.Scalar(999.5f);
            ((long)(NDArray)floatScalar).Should().Be(1000L, "IConvertible rounds to nearest");

            // byte -> double
            var byteScalar = NDArray.Scalar((byte)255);
            ((double)(NDArray)byteScalar).Should().Be(255.0);
        }
    }
}
