using System;
using NumSharp.Backends;
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
        /// <param name="a">The data (bool computes as its 0 / 1 values; text: see the exceptions).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The median (NaN for an empty slice); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/>: NumPy's closing <c>mean</c> refuses (its
        ///     verbatim text). A NEP 55 <c>StringDType</c> array is sorted first (NumPy's partition of a <c>StringDType</c>
        ///     line), then its mean's divide refuses the concatenated text.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a line of two or more elements (the sort's
        ///     <c>Cannot compare null that is not a nan-like value</c>), an empty reduced axis (the mean's add.reduce has no
        ///     identity) or a read-only <paramref name="out"/>.</exception>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition (<c>chararray.partition() got an unexpected keyword
        ///     argument 'axis'</c>), after the axes and keepdims' out are checked. With <c>overwrite_input</c> and no axis
        ///     NumPy partitions the text itself instead: its broadcast error for an even length other than 2
        ///     (<see cref="IncorrectShapeException"/>), else the closing mean's refusal of text.</exception>
        /// <remarks>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.median.html
        ///     <para>A <c>StringDType</c> median exists only when the middle elements are NaN-like NAs: NumPy's
        ///     <c>np.float64(nan)</c>, a read-only 0-d float64 here.</para>
        /// </remarks>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before the text refusals.</exception>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        public static NDArray median(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, overwrite_input, keepdims), static s => median(s.a, s.axis, s.@out, s.overwrite_input, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: _median partitions the chararray, and chararray.partition is the STRING
            // partition (NDCharArrayBoundary): NumPy's TypeError, or the string partition's own answer (MedianOfChararray)
            if (NDCharArrayBoundary.Sees(a))
                throw DefaultEngine.MedianOfChararray(a, axis.HasValue ? new[] { axis.Value } : null, @out, overwrite_input, keepdims);
            // NumPy's np.median body (the text routes, then the numeric median) is TensorEngine.Median's: the facade
            // holds only the class rules above.
            // NumPy's _ureduce turns an int axis into a 1-tuple: the engine member takes the axes as an array.
            return a.TensorEngine.Median(a, axis.HasValue ? new[] { axis.Value } : null, @out, overwrite_input,
                keepdims);
        }

        /// <summary>
        ///     The median over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the reduced axes are merged into one, an
        ///     even-sized slice gives the mean of its two central values (NumPy's <c>mean(part[middle])</c>, not the
        ///     quantile lerp), an empty slice gives NaN, and an EMPTY axis list reduces nothing.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it, see the
        ///     exceptions).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened
        ///     data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy
        ///     anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The median over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats; or, for a <c>StringDType</c> array, a None-like NA in a sorted line of
        ///     two or more elements (<c>Cannot compare null that is not a nan-like value</c>), an empty reduced axis (the
        ///     mean's add.reduce has no identity) or a read-only <paramref name="out"/>. A <c>StringDType</c> median exists
        ///     only over NaN-like NAs (<c>np.float64(nan)</c>).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/> that NumPy's closing mean refuses (its
        ///     verbatim text); a <c>StringDType</c> array after its sort (the mean's divide refuses the concatenated text).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (<c>np.median(m, axis=())</c> reduces axis 1).</remarks>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition
        ///     (<c>chararray.partition() got an unexpected keyword argument 'axis'</c>), after the axes and keepdims' out are
        ///     checked. With <c>overwrite_input</c> and no axis NumPy partitions the text itself instead: its broadcast error
        ///     for an even length other than 2 (<see cref="IncorrectShapeException"/>), else the closing mean's refusal of
        ///     text.</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index:
        ///     <c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c> (too few dimensions, or a reduced axis of length 0),
        ///     raised for every dtype before the text refusals.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype:
        ///     <c>_ureduce</c>'s merge cannot resolve its -1 (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>), raised before the text refusals.</exception>
        public static NDArray median(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            if (a is null) throw new ArgumentNullException(nameof(a));
            // a numpy.char.chararray operand (decided before the matrix region below, which would hide it): the STRING
            // partition's answer (MedianOfChararray, NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                throw DefaultEngine.MedianOfChararray(a, axis, @out, overwrite_input, keepdims);
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.median body (the text routes, then the numeric median) is TensorEngine.Median's: the facade
            // holds only the class rules above.
            return a.TensorEngine.Median(a, axis, @out, overwrite_input, keepdims);
        }
    }
}
