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
        /// <param name="a">The data (bool computes as its 0 / 1 values; text: see the exceptions).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The NaN-ignoring median; <paramref name="out"/> itself when one is given. Over a NEP 55
        ///     <c>StringDType</c> array a median exists only for an all-NaN reduction: its NA object (a read-only 0-d
        ///     <c>StringDType</c> array, NumPy's Python object), or <c>np.float64(nan)</c> where NumPy's mean of NaN-like NAs
        ///     gives it.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data that NumPy's NaN test (or, for empty data, its <c>nanmean</c>)
        ///     refuses (its verbatim text). A <c>StringDType</c> array: the median's mean refuses the concatenated text;
        ///     along an axis shorter than 600 numpy.ma's median refuses first (<c>Unsuitable type StringDType() for
        ///     calculating minimum.</c>); <paramref name="keepdims"/> over NumPy's Python-object result (<c>'float' object is
        ///     not subscriptable</c>).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a sorted line of two or more elements
        ///     (<c>Cannot compare null that is not a nan-like value</c>), an empty input's mean (no identity), or the store into
        ///     <paramref name="out"/> (NumPy's setitem texts, e.g. <c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> reduction along a long axis whose row result NumPy
        ///     stores in an OBJECT array (a <c>None</c> NA object picked by <c>np.apply_along_axis</c>): NumSharp has no object
        ///     dtype.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.nanmedian.html</remarks>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>).</exception>
        public static NDArray nanmedian(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, overwrite_input, keepdims), static s => nanmedian(s.a, s.axis, s.@out, s.overwrite_input, s.keepdims), axis, keepdims, null);
            // NumPy's np.nanmedian body (the text routes, then the NaN-ignoring median) is QuantileEngine.NanMedian's: the
            // facade holds only the class rule above.
            return QuantileEngine.NanMedian(a, axis, @out, overwrite_input, keepdims);
        }

        /// <summary>
        ///     The median over SEVERAL axes at once, ignoring NaNs (NumPy's tuple <c>axis</c>): the reduced axes are merged
        ///     into one, an even-sized (NaN-free) slice gives the mean of its two central values, a slice that holds only
        ///     NaNs (or nothing) gives NaN, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it, see the
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
        /// <exception cref="ValueError">An axis repeats; or a <c>StringDType</c> array's errors (see the one-axis overload: a
        ///     None-like NA in a sorted line, an empty input's mean, the store into <paramref name="out"/>).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data that NumPy's NaN test (or, for empty text data, its <c>nanmean</c>)
        ///     refuses (its verbatim text); a <c>StringDType</c> array's median, numpy.ma or keepdims refusals (see the
        ///     one-axis overload).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> row result NumPy stores in an OBJECT array (see the
        ///     one-axis overload).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule (NumPy's nanmedian of a matrix calls <c>matrix.sum</c> with <c>keepdims</c> and raises
        ///     TypeError for every axis).</remarks>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index: <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (empty data
        ///     answers <c>np.nanmean</c> first), raised before the text refusals (<c>isnan</c>).</exception>
        public static NDArray nanmedian(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.nanmedian body is QuantileEngine.NanMedian's.
            return QuantileEngine.NanMedian(a, axis, @out, overwrite_input, keepdims);
        }
    }
}
