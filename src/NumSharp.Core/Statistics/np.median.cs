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
            // Text (input or out=): NumPy's _ureduce normalizes the axis (a 0-d input refuses every axis), then the
            // closing mean(part, out=out) resolves add on (out, part) and refuses with its verbatim text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis, @out);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's np.median returns nan for an empty slice (np.quantile/percentile raise);
            // emptyReturnsNaN routes the empty-axis case to a nan fill instead. medianMean selects
            // the mean-of-middle reduction (NumPy's np.median), not the quantile(0.5) lerp.
            return QuantileEngine.Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     The median over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into one, an
        ///     even-sized slice gives the mean of its two central values (NumPy's <c>mean(part[middle])</c>, not the
        ///     quantile lerp), an empty slice gives NaN, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused or deferred, see the
        ///     exceptions).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The median over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/> that NumPy's closing mean refuses (its
        ///     verbatim text).</exception>
        /// <exception cref="StringComputationNotSupportedException">Text data that NumPy partitions.</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (<c>np.median(m, axis=())</c> reduces axis 1).</remarks>
        public static NDArray median(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text (input or out=): NumPy's _ureduce normalizes the tuple (every axis, then repeats), then the closing
            // mean(part, out=out) refuses on (out, part) with its verbatim text; see NDStringUfuncs.ReduceError /
            // ReduceOutError.
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis, @out);
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }
    }
}
