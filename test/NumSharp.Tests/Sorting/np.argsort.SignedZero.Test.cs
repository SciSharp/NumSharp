using AwesomeAssertions;

namespace NumSharp.Tests.Sorting
{
    /// <summary>
    ///     <c>-0.0</c> and <c>+0.0</c> are EQUAL keys to NumPy's sort (<c>npy::float_tag::less</c>), so its stable argsort
    ///     keeps them in input order. NumSharp's float32 / float64 argsort keyed the radix by the raw bits and put every
    ///     <c>-0.0</c> first (<c>np.argsort([0.0, -0.0])</c> was [1, 0]); np.lexsort's tie order and the numpy.ma set
    ///     operations inherited it. Expected values are NumPy 2.4.2's (<c>kind='stable'</c>; NumPy's default kind agrees on
    ///     these small arrays).
    /// </summary>
    [TestClass]
    public class np_argsort_signed_zero_tests
    {
        /// <summary>Both float widths: the zeros of either sign keep their input order, NaN still last.</summary>
        [TestMethod]
        public void Argsort_SignedZeros_KeepInputOrder()
        {
            np.argsort(np.array(new[] { 0.0, -0.0, 0.0, -0.0 }), -1).ToArray<long>().Should().Equal(0L, 1L, 2L, 3L);
            np.argsort(np.array(new[] { 0.0f, -0.0f }), -1).ToArray<long>().Should().Equal(0L, 1L);
            np.argsort(np.array(new[] { 1.0, -0.0, double.NaN, 0.0, -1.0 }), -1).ToArray<long>()
              .Should().Equal(4L, 1L, 3L, 0L, 2L);
        }

        /// <summary>np.lexsort's primary key ties on <c>±0</c>, so the secondary key decides: [1, 0] and [0, 1].</summary>
        [TestMethod]
        public void Lexsort_SignedZeroPrimaryKey_IsATie()
        {
            np.lexsort(new[] { np.array(new long[] { 1, 0 }), np.array(new[] { 0.0, -0.0 }) }).ToArray<long>()
              .Should().Equal(1L, 0L);
            np.lexsort(new[] { np.array(new long[] { 0, 1 }), np.array(new[] { -0.0, 0.0 }) }).ToArray<long>()
              .Should().Equal(0L, 1L);
        }
    }
}
