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
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, overwrite_input, keepdims), static s => nanmedian(s.a, s.axis, s.@out, s.overwrite_input, s.keepdims), axis, keepdims, null);
            // Text: NumPy sends EMPTY data to np.nanmean(a, axis, out=out) first (whose add.reduce resolves on (out, a));
            // otherwise its _ureduce normalizes the axis (a 0-d input refuses every axis) and the NaN test refuses text;
            // see NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is not null && a.size == 0)
                Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmean", a, axis, @out);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmedian", a, axis);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            return QuantileEngine.Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     The median over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the reduced axes are merged
        ///     into one, an even-sized (NaN-free) slice gives the mean of its two central values, a slice that holds only
        ///     NaNs (or nothing) gives NaN, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused or deferred, see the
        ///     exceptions).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The NaN-ignoring median over the reduced axes; <paramref name="out"/> itself when one is
        ///     given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data that NumPy's NaN test (or, for empty text data, its <c>nanmean</c>)
        ///     refuses (its verbatim text).</exception>
        /// <exception cref="StringComputationNotSupportedException">Text data that NumPy partitions.</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule (NumPy's nanmedian of a matrix calls <c>matrix.sum</c> with <c>keepdims</c> and raises
        ///     TypeError for every axis).</remarks>
        public static NDArray nanmedian(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text: NumPy sends EMPTY data to np.nanmean(a, axis, out=out) first (whose add.reduce resolves on (out, a));
            // otherwise its _ureduce normalizes the tuple (every axis, then repeats) and the NaN test refuses text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            if (a is not null && a.size == 0)
                Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmean", a, axis, @out);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmedian", a, axis);
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }
    }
}
