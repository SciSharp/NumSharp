using System;

namespace NumSharp.Tests.Sorting_Searching_Counting
{
    /// <summary>
    /// np.digitize — validated against NumPy 2.4.2. digitize is searchsorted + a monotonicity
    /// check; values, shapes, dtype (int64/intp) and error taxonomy all mirror NumPy.
    /// </summary>
    [TestClass]
    public class NpDigitizeTests
    {
        private static void AssertInt64(NDArray result, long[] expectedFlat, int[] expectedShape = null)
        {
            Assert.AreEqual(NPTypeCode.Int64, result.typecode, "digitize returns int64 (intp)");
            if (expectedShape != null)
            {
                Assert.AreEqual(expectedShape.Length, result.ndim, "ndim");
                for (int i = 0; i < expectedShape.Length; i++)
                    Assert.AreEqual(expectedShape[i], result.shape[i], $"shape[{i}]");
            }
            var flat = result.flatten();
            Assert.AreEqual(expectedFlat.Length, (int)flat.size, "size");
            for (int i = 0; i < expectedFlat.Length; i++)
                Assert.AreEqual(expectedFlat[i], flat.GetInt64((long)i), $"element {i}");
        }

        // ---- increasing bins ----

        [TestMethod]
        public void Increasing_RightFalse()
        {
            // np.digitize([0.2,6.4,3.0,1.6], [0,1,2.5,4,10]) = [1,4,3,2]
            var x = np.array(new[] { 0.2, 6.4, 3.0, 1.6 });
            var bins = np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 });
            AssertInt64(np.digitize(x, bins), new long[] { 1, 4, 3, 2 });
        }

        [TestMethod]
        public void Increasing_RightTrue()
        {
            // np.digitize([1.2,10,12.4,15.5,20], [0,5,10,15,20], right=True) = [1,2,3,4,4]
            var x = np.array(new[] { 1.2, 10.0, 12.4, 15.5, 20.0 });
            var bins = np.array(new[] { 0.0, 5.0, 10.0, 15.0, 20.0 });
            AssertInt64(np.digitize(x, bins, right: true), new long[] { 1, 2, 3, 4, 4 });
        }

        // ---- decreasing bins ----

        [TestMethod]
        public void Decreasing_RightFalse()
        {
            // np.digitize([1.2,10,12.4,15.5,20], [20,15,10,5,0]) = [4,2,2,1,0]
            var x = np.array(new[] { 1.2, 10.0, 12.4, 15.5, 20.0 });
            var bins = np.array(new[] { 20.0, 15.0, 10.0, 5.0, 0.0 });
            AssertInt64(np.digitize(x, bins), new long[] { 4, 2, 2, 1, 0 });
        }

        [TestMethod]
        public void Decreasing_RightTrue()
        {
            // np.digitize([1.2,10,12.4,15.5,20], [20,15,10,5,0], right=True) = [4,3,2,1,1]
            var x = np.array(new[] { 1.2, 10.0, 12.4, 15.5, 20.0 });
            var bins = np.array(new[] { 20.0, 15.0, 10.0, 5.0, 0.0 });
            AssertInt64(np.digitize(x, bins, right: true), new long[] { 4, 3, 2, 1, 1 });
        }

        // ---- dtype promotion (result_type(bins, x)); must NOT truncate x into bins' dtype ----

        [TestMethod]
        public void IntBins_FloatX_Promotes()
        {
            // np.digitize([1.2,10,12.4,15.5,20], [0,5,10,15,20]) = [1,3,3,4,5]
            var x = np.array(new[] { 1.2, 10.0, 12.4, 15.5, 20.0 });
            var bins = np.array(new[] { 0, 5, 10, 15, 20 });
            AssertInt64(np.digitize(x, bins), new long[] { 1, 3, 3, 4, 5 });
        }

        [TestMethod]
        public void NegativeFloatX_IntBins_NoTruncation()
        {
            // np.digitize([-0.5, 0.5], [-1,0,1]) = [1,2]  — truncating -0.5 toward 0 would give the wrong bin.
            var x = np.array(new[] { -0.5, 0.5 });
            var bins = np.array(new[] { -1, 0, 1 });
            AssertInt64(np.digitize(x, bins), new long[] { 1, 2 });
        }

        [TestMethod]
        public void Int64X_BeyondInt32Bins_Promotes()
        {
            // np.digitize([3e9:int64], [0,1e9,2e9]:int32) = [3]  — casting x into int32 bins would overflow.
            var x = np.array(new long[] { 3_000_000_000L });
            var bins = np.array(new[] { 0, 1_000_000_000, 2_000_000_000 });
            AssertInt64(np.digitize(x, bins), new long[] { 3 });
        }

        [TestMethod]
        public void Float32X_Float64Bins()
        {
            var x = np.array(new float[] { 1.5f });
            var bins = np.array(new[] { 0.0, 1.0, 2.0 });
            AssertInt64(np.digitize(x, bins), new long[] { 2 });
        }

        // ---- shapes ----

        [TestMethod]
        public void TwoDimensionalX_PreservesShape()
        {
            // np.digitize([[0.2,6.4],[3.0,1.6]], [0,1,2.5,4,10]) = [[1,4],[3,2]]
            var x = np.array(new[] { 0.2, 6.4, 3.0, 1.6 }).reshape(2, 2);
            var bins = np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 });
            AssertInt64(np.digitize(x, bins), new long[] { 1, 4, 3, 2 }, new[] { 2, 2 });
        }

        [TestMethod]
        public void ScalarX_ReturnsZeroDim()
        {
            // np.digitize(np.array(2.5), bins) = np.int64(3), shape ()
            var bins = np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 });
            var r = np.digitize(NDArray.Scalar(2.5), bins);
            Assert.AreEqual(NPTypeCode.Int64, r.typecode);
            Assert.AreEqual(0, r.ndim);
            Assert.AreEqual(3L, r.GetInt64(new long[0]));
        }

        // ---- edge bins ----

        [TestMethod]
        public void EmptyBins_AllZero()
        {
            // np.digitize([1,2], []) = [0,0]  (empty bins is treated as monotonic increasing)
            AssertInt64(np.digitize(np.array(new[] { 1.0, 2.0 }), np.array(new double[] { })), new long[] { 0, 0 });
        }

        [TestMethod]
        public void SingleBin()
        {
            AssertInt64(np.digitize(np.array(new[] { 1.0, 2.0, 3.0 }), np.array(new[] { 2.0 })), new long[] { 0, 1, 1 });
        }

        [TestMethod]
        public void AllEqualBins_TreatedIncreasing()
        {
            // np.digitize([1,2,3], [2,2,2]) = [0,3,3]
            AssertInt64(np.digitize(np.array(new[] { 1.0, 2.0, 3.0 }), np.array(new[] { 2.0, 2.0, 2.0 })), new long[] { 0, 3, 3 });
        }

        [TestMethod]
        public void DuplicateIncreasingBins()
        {
            AssertInt64(np.digitize(np.array(new[] { 1.0, 2.0, 2.5, 3.0 }), np.array(new[] { 1.0, 2.0, 2.0, 3.0 })), new long[] { 1, 3, 3, 4 });
        }

        [TestMethod]
        public void OutOfRange()
        {
            // below -> 0, above -> len(bins)
            AssertInt64(np.digitize(np.array(new[] { -1.0, 100.0 }), np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 })), new long[] { 0, 5 });
        }

        // ---- x dtypes ----

        [TestMethod]
        public void IntX()
        {
            AssertInt64(np.digitize(np.array(new[] { 1, 2, 3, 6 }), np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 })), new long[] { 2, 2, 3, 4 });
        }

        [TestMethod]
        public void BoolX()
        {
            // True->1.0, False->0.0 promoted to float64
            AssertInt64(np.digitize(np.array(new[] { true, false }), np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 })), new long[] { 2, 1 });
        }

        // ---- NaN handling (depends on searchsorted's NaN-as-largest total order) ----

        [TestMethod]
        public void NaNInX_IncreasingBins_GoesToEnd()
        {
            // np.digitize([nan,1.0], [0,1,2.5,4,10]) = [5,2]  (NaN sorts last -> len(bins))
            AssertInt64(np.digitize(np.array(new[] { double.NaN, 1.0 }), np.array(new[] { 0.0, 1.0, 2.5, 4.0, 10.0 })), new long[] { 5, 2 });
        }

        [TestMethod]
        public void NaNInBins_MonotonicityQuirk()
        {
            // check_array_monotonic uses strict </> after the first differing pair, so a NaN pair never
            // registers a violation: [1,2,nan,3] is reported increasing (matching NumPy).
            // np.digitize([1,2], [1,2,nan,3]) = [1,2]
            AssertInt64(np.digitize(np.array(new[] { 1.0, 2.0 }), np.array(new[] { 1.0, 2.0, double.NaN, 3.0 })), new long[] { 1, 2 });
        }

        // ---- errors (verbatim NumPy messages) ----

        [TestMethod]
        public void ComplexX_Throws()
        {
            var ex = Assert.ThrowsException<IncorrectTypeException>(() =>
                np.digitize(np.array(new[] { new System.Numerics.Complex(1, 2) }), np.array(new[] { 0.0, 1.0 })));
            StringAssert.Contains(ex.Message, "x may not be complex");
        }

        [TestMethod]
        public void TwoDimBins_Throws()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.digitize(np.array(new[] { 0.2, 6.4 }), np.array(new[] { 0.0, 1.0, 2.0, 3.0 }).reshape(2, 2)));
            StringAssert.Contains(ex.Message, "object too deep for desired array");
        }

        [TestMethod]
        public void ZeroDimBins_Throws()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.digitize(np.array(new[] { 1.0, 2.0 }), NDArray.Scalar(5.0)));
            StringAssert.Contains(ex.Message, "object of too small depth for desired array");
        }

        [TestMethod]
        public void NonMonotonicBins_Throws()
        {
            var ex = Assert.ThrowsException<ArgumentException>(() =>
                np.digitize(np.array(new[] { 1.0 }), np.array(new[] { 0.0, 2.0, 1.0, 3.0 })));
            StringAssert.Contains(ex.Message, "bins must be monotonically increasing or decreasing");
        }
    }
}
