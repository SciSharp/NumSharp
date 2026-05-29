using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the median along the specified axis.
        ///     For an even-sized slice the median is the mean of the two central values;
        ///     for an odd-sized slice it is the single central value. Equivalent to
        ///     <c>np.quantile(a, 0.5)</c> for our purposes, which matches NumPy's contract.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.median.html</remarks>
        public static NDArray median(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's np.median returns nan for an empty slice (np.quantile/percentile raise);
            // emptyReturnsNaN routes the empty-axis case to a nan fill instead.
            return QuantileEngine.Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true);
        }

        public static NDArray median(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true);
        }
    }
}
