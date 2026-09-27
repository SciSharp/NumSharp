using System;

namespace NumSharp.Tests.RandomSampling
{
    /// <summary>
    ///     Byte-exact parity tests for <see cref="NumPyRandom.random_integers"/> against NumPy 2.4.2's
    ///     <c>np.random.random_integers</c> (MT19937, seed 42). It is the CLOSED-interval
    ///     <c>randint(low, high+1, dtype='l')</c>; the C long is int64 in NumSharp's LP64 model (Linux/macOS NumPy;
    ///     Windows NumPy's long is int32, with the same values wherever both accept the bounds).
    /// </summary>
    [TestClass]
    public class NpRandomRandomIntegersTest
    {
        [TestMethod]
        public void RandomIntegers_LowHighSize_ByteExact()
        {
            np.random.seed(42);
            var a = np.random.random_integers(1, 6, new Shape(3, 2));
            a.dtype.Should().Be(typeof(long)); // dtype='l': the LP64 C long
            a.shape.Should().Equal(3, 2);
            for (int i = 0; i < 6; i++)
                Convert.ToInt64(a.GetAtIndex(i)).Should().Be(new long[] { 4, 5, 3, 5, 5, 2 }[i]);
        }

        [TestMethod]
        public void RandomIntegers_Scalar_ByteExact()
        {
            np.random.seed(42);
            var a = np.random.random_integers(5); // high=None -> [1, 5]
            Convert.ToInt64(a.GetAtIndex(0)).Should().Be(4);
        }

        [TestMethod]
        public void RandomIntegers_ClosedInterval_IsInclusive()
        {
            np.random.seed(1);
            var a = np.random.random_integers(1, 1, new Shape(8));
            for (int i = 0; i < 8; i++)
                Convert.ToInt64(a.GetAtIndex(i)).Should().Be(1); // [1,1] closed -> all ones
        }

        [TestMethod]
        public void RandomIntegers_HighNone_DrawsFromOneToLow()
        {
            np.random.seed(123);
            var a = np.random.random_integers(6, size: new Shape(500));
            for (long i = 0; i < a.size; i++)
            {
                long v = Convert.ToInt64(a.GetAtIndex(i));
                v.Should().BeInRange(1, 6);
            }
        }

        [TestMethod]
        public void RandomIntegers_EqualsRandintPlusOne()
        {
            np.random.seed(7);
            var ri = np.random.random_integers(2, 9, new Shape(20));
            np.random.seed(7);
            var rint = np.random.randint(2, 10, new Shape(20)); // [2, 10) == [2, 9], the same default long dtype
            for (long i = 0; i < 20; i++)
                Convert.ToInt64(ri.GetAtIndex(i)).Should().Be(Convert.ToInt64(rint.GetAtIndex(i)));
        }
    }
}
