using System;

namespace NumSharp.Tests.Statistics
{
    /// <summary>
    ///     Parity QA-55: the cast rule of an out of another kind follows NumPy's statements. np.median writes through
    ///     <c>np.mean(part[indexer], axis, out=out)</c>, whose add.reduce and <c>true_divide(..., casting='unsafe')</c> store
    ///     into an out of ANY kind (truncated into an integer out, nonzero → True into a bool out). np.ptp is
    ///     <c>subtract(maximum.reduce(out=out), minimum.reduce(), out)</c>: an integer out of a float input is subtract's
    ///     same_kind refusal (a TypeError) raised after the maxima were stored, while integer inputs wrap in their own
    ///     dtype and widen exactly. Every expected value is NumPy 2.4.2's (scratchpad probe p174_numpy.py).
    /// </summary>
    [TestClass]
    public class MedianPtpIntegerOutsTests
    {
        private const string SubtractRefusal =
            "Cannot cast ufunc 'subtract' output from dtype('float64') to dtype('int64') with casting rule 'same_kind'";

        /// <summary>An out of <paramref name="shape"/> and <paramref name="dtype"/> filled with -1.</summary>
        /// <param name="dtype">The out's dtype.</param>
        /// <param name="shape">The out's shape (empty for a 0-d out).</param>
        /// <returns>The out.</returns>
        private static NDArray Full(DType dtype, params long[] shape) => np.full(new Shape(shape, copy: false), -1, dtype);

        /// <summary>The out's elements in C order, as doubles.</summary>
        /// <param name="o">The out.</param>
        /// <returns>The values.</returns>
        private static double[] Vals(NDArray o)
        {
            using var widened = o.astype(np.float64);
            using var flat = widened.flatten();
            return flat.ToArray<double>();
        }

        /// <summary>
        ///     np.median into integer and bool outs: the mean of the middle elements, cast unsafely (1.5 → 1, a nonzero
        ///     median → True), along an axis and flat into a 0-d out.
        /// </summary>
        /// <exception cref="AssertFailedException">An out differs from NumPy's.</exception>
        [TestMethod]
        public void Median_IntoAnIntegerOrBoolOut_CastsUnsafely()
        {
            using var a = np.arange(12.0).reshape(3, 4);

            using (var o = Full(np.int64, 3))
            {
                ReferenceEquals(np.median(a, 1, o), o).Should().BeTrue("NumPy returns the out");
                Vals(o).Should().Equal(1.0, 5, 9);
            }
            using (var o = Full(np.int32, 4))
            {
                np.median(a, 0, o);
                Vals(o).Should().Equal(4.0, 5, 6, 7);
            }
            using (var shifted = a - 5.5)
            using (var o = np.zeros(new Shape(3), np.@bool))
            {
                np.median(shifted, 1, o);
                o.ToArray<bool>().Should().Equal(true, false, true);
            }
            using (var o = Full(np.int64))
            {
                np.median(a, (int?)null, o);
                Vals(o).Should().Equal(5.0);
            }
            using (var quarter = a + 0.75)
            using (var o = Full(np.int64, 3))
            {
                np.median(quarter, 1, o);
                Vals(o).Should().Equal(2.0, 6, 10);
            }
        }

        /// <summary>
        ///     np.ptp of a float input into an integer out: subtract's same_kind TypeError, with the maxima already stored
        ///     in the out (maximum.reduce ran first), for one axis and for an axis tuple; a float out of another width
        ///     takes the ranges.
        /// </summary>
        /// <exception cref="AssertFailedException">The refusal or an out differs from NumPy's.</exception>
        [TestMethod]
        public void Ptp_FloatInputIntoAnIntegerOut_IsSubtractsTypeError_AfterTheMaxima()
        {
            using var a = np.arange(12.0).reshape(3, 4);
            using (var o = Full(np.int64, 4))
            {
                new Action(() => np.ptp(a, 0, o)).Should().Throw<TypeError>().WithMessage(SubtractRefusal);
                Vals(o).Should().Equal(8.0, 9, 10, 11);
            }
            using (var o = Full(np.float32, 4))
            {
                np.ptp(a, 0, o);
                Vals(o).Should().Equal(8.0, 8, 8, 8);
            }

            using var a3 = np.arange(24.0).reshape(2, 3, 4);
            using (var o = Full(np.int64, 4))
            {
                new Action(() => np.ptp(a3, new[] { 0, 1 }, o)).Should().Throw<TypeError>().WithMessage(SubtractRefusal);
                Vals(o).Should().Equal(20.0, 21, 22, 23);
            }
            using (var o = Full(np.float64, 4))
            {
                np.ptp(a3, new[] { 0, 1 }, o);
                Vals(o).Should().Equal(20.0, 20, 20, 20);
            }
        }

        /// <summary>
        ///     np.ptp of integer and bool inputs: the extremes widen exactly into a wider out (int8 127 − (−128) = 255),
        ///     wrap into an out of the input's own width (−1), and a bool input subtracts as integers.
        /// </summary>
        /// <exception cref="AssertFailedException">An out differs from NumPy's.</exception>
        [TestMethod]
        public void Ptp_IntegerInputs_WidenOrWrapLikeNumPy()
        {
            using var i1 = np.array(new sbyte[,] { { 127, -128 }, { -128, 127 } });
            using (var o = Full(np.int64, 2))
            {
                np.ptp(i1, 0, o);
                Vals(o).Should().Equal(255.0, 255);
            }
            using (var o = Full(np.int8, 2))
            {
                np.ptp(i1, 0, o);
                Vals(o).Should().Equal(-1.0, -1);
            }
            using (var flat = np.array(new sbyte[] { 127, -128 }))
            using (var o = Full(np.int64))
            {
                np.ptp(flat, (int?)null, o);
                Vals(o).Should().Equal(255.0);
            }
            using (var b = np.array(new[,] { { true, false }, { false, false } }))
            using (var o = Full(np.int64, 2))
            {
                np.ptp(b, 0, o);
                Vals(o).Should().Equal(1.0, 0);
            }
        }
    }
}
