using AwesomeAssertions;
using NumSharp;

namespace NumSharp.UnitTest
{
    /// <summary>
    ///     Tests for np.sign dtype preservation.
    /// </summary>
    public class SignDtypeTests
    {
        /// <summary>
        ///     BUG 64 FIX: np.sign preserves input dtype for integer types.
        ///
        ///     NumPy:    np.sign(int32[]).dtype = int32
        ///     Fixed:    Returns int32
        ///     Was:      Returns Double
        /// </summary>
        [Test]
        public void Sign_Int32_PreservesDtype()
        {
            var a = np.array(new int[] { -3, 0, 5 });
            var result = np.sign(a);

            result.dtype.Should().Be(typeof(int), "np.sign preserves input dtype");
            result.GetInt32(0).Should().Be(-1, "sign(-3) = -1");
            result.GetInt32(1).Should().Be(0, "sign(0) = 0");
            result.GetInt32(2).Should().Be(1, "sign(5) = 1");
        }

        /// <summary>
        ///     Test np.sign preserves dtype for other integer types.
        /// </summary>
        [Test]
        public void Sign_Int64_PreservesDtype()
        {
            var a = np.array(new long[] { -100L, 0L, 100L });
            var result = np.sign(a);

            result.dtype.Should().Be(typeof(long), "np.sign preserves int64 dtype");
            result.GetInt64(0).Should().Be(-1L);
            result.GetInt64(1).Should().Be(0L);
            result.GetInt64(2).Should().Be(1L);
        }

        /// <summary>
        ///     Test np.sign preserves dtype for float types.
        /// </summary>
        [Test]
        public void Sign_Double_PreservesDtype()
        {
            var a = np.array(new double[] { -2.5, 0.0, 3.7 });
            var result = np.sign(a);

            result.dtype.Should().Be(typeof(double), "np.sign preserves double dtype");
            result.GetDouble(0).Should().Be(-1.0);
            result.GetDouble(1).Should().Be(0.0);
            result.GetDouble(2).Should().Be(1.0);
        }

        /// <summary>
        ///     Test np.sign with byte type.
        /// </summary>
        [Test]
        public void Sign_Byte_PreservesDtype()
        {
            // Byte is unsigned, so sign is always 0 or 1
            var a = np.array(new byte[] { 0, 1, 255 });
            var result = np.sign(a);

            result.dtype.Should().Be(typeof(byte), "np.sign preserves byte dtype");
            result.GetByte(0).Should().Be(0, "sign(0) = 0");
            result.GetByte(1).Should().Be(1, "sign(1) = 1");
            result.GetByte(2).Should().Be(1, "sign(255) = 1");
        }
    }
}
