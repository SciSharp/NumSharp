using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the median along the specified axis, ignoring NaNs. Like <c>np.median</c>,
        ///     an even-sized (NaN-free) slice reduces to the <b>mean</b> of the two central values —
        ///     NumPy's mean-of-middle rule, NOT <c>np.nanquantile(a, 0.5)</c> (they disagree by up
        ///     to 1 ULP), so <c>medianMean: true</c> is REQUIRED. A slice that is entirely NaN (or
        ///     empty) yields NaN, matching NumPy's "All-NaN slice" behaviour.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html</remarks>
        public static NDArray nanmedian(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        public static NDArray nanmedian(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }
    }
}
