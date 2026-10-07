using NumSharp.Statistics;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Compute the median along the specified axis.
        ///     For an even-sized slice the median is the <b>mean</b> of the two central values,
        ///     <c>(a+b)/2</c>; for an odd-sized slice it is the single central value. This is NOT
        ///     the same code path as <c>np.quantile(a, 0.5)</c> — NumPy computes <c>np.median</c>
        ///     via <c>mean(part[middle])</c>, which disagrees with the quantile <c>_lerp</c> at
        ///     q=0.5 by up to 1 ULP on ~10 % of inputs — so <c>medianMean: true</c> is REQUIRED
        ///     for bit-exact parity.
        /// </summary>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.median.html</remarks>
        public static NDArray median(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, overwrite_input, keepdims), static s => median(s.a, s.axis, s.@out, s.overwrite_input, s.keepdims), axis, keepdims, null);
            // Text: NumPy's _ureduce normalizes the axis (a 0-d input refuses every axis), then the reduction refuses
            // with its verbatim text; see NDStringUfuncs.ReduceError.
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's np.median returns nan for an empty slice (np.quantile/percentile raise);
            // emptyReturnsNaN routes the empty-axis case to a nan fill instead. medianMean selects
            // the mean-of-middle reduction (NumPy's np.median), not the quantile(0.5) lerp.
            return QuantileEngine.Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        public static NDArray median(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            // Text: NumPy's _ureduce normalizes the tuple (every axis, then repeats), then the reduction refuses with its
            // verbatim text; see NDStringUfuncs.ReduceError.
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis);
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }
    }
}
