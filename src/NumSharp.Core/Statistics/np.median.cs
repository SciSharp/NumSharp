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
        public static NDArray median(NDArray a,
            int? axis = null, NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            // a numpy.matrix operand: a row, a column or the scalar, as NumPy's matrix method returns
            if (NDMatrixBoundary.Sees(a))
                return NDMatrixBoundary.Reduced((a, axis, @out, overwrite_input, keepdims), static s => median(s.a, s.axis, s.@out, s.overwrite_input, s.keepdims), axis, keepdims, null);
            // a numpy.char.chararray operand: _median partitions the chararray, and chararray.partition is the STRING
            // partition (NDCharArrayBoundary): NumPy's TypeError, or the string partition's own answer (MedianOfChararray)
            if (NDCharArrayBoundary.Sees(a))
                throw MedianOfChararray(a, axis.HasValue ? new[] { axis.Value } : null, @out, overwrite_input, keepdims);
            // NumPy's np.median body (the text routes, then the numeric median) is QuantileEngine.Median's: the facade
            // holds only the class rules above.
            return QuantileEngine.Median(a, axis, @out, overwrite_input, keepdims);
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
            // a numpy.char.chararray operand (decided before the matrix region below, which would hide it): the STRING
            // partition's answer (MedianOfChararray, NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                throw MedianOfChararray(a, axis, @out, overwrite_input, keepdims);
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // NumPy's np.median body (the text routes, then the numeric median) is QuantileEngine.Median's: the facade
            // holds only the class rules above.
            return QuantileEngine.Median(a, axis, @out, overwrite_input, keepdims);
        }

        /// <summary>
        ///     NumPy's <c>np.median</c> of a <c>numpy.char.chararray</c>, statement by statement up to its failure:
        ///     <c>_ureduce</c> normalizes the axes, indexes <c>out</c> for keepdims and merges several reduced axes; then
        ///     <c>_median</c> partitions the chararray — and <c>chararray.partition</c> is the STRING partition
        ///     (<c>partition(sep)</c>). Every form passes <c>axis=</c> to it (<c>np.partition</c>, or <c>a.partition(kth,
        ///     axis=axis)</c>) and gets the TypeError, except <c>overwrite_input=True</c> without an axis: NumPy's
        ///     <c>part.partition(kth)</c> then really partitions the strings around the kth list as text, which fails to
        ///     broadcast for an even length other than 2 and otherwise computes and leaves the data as it was, so the closing
        ///     <c>mean(part[indexer])</c> refuses the text.
        /// </summary>
        /// <param name="a">The chararray.</param>
        /// <param name="axes">The reduced axes, or null for every axis.</param>
        /// <param name="out">The <c>out=</c> argument, or null.</param>
        /// <param name="overwrite_input">NumPy's <c>overwrite_input</c>.</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>.</param>
        /// <returns>The exception NumPy raises (never null): the caller throws it.</returns>
        /// <exception cref="AxisError">An axis is out of bounds (thrown here, NumPy's first check).</exception>
        /// <exception cref="ValueError">An axis repeats (thrown here).</exception>
        /// <exception cref="IndexError">keepdims with an <paramref name="out"/> NumPy cannot index (thrown here).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0 (the merge
        ///     reshape, thrown here).</exception>
        private static Exception MedianOfChararray(NDArray a, int[] axes, NDArray @out, bool overwrite_input, bool keepdims)
        {
            int nd = a.ndim;
            // The plain view the remaining statements read: no subclass rule may run on them.
            using var region = NDCharArrayBoundary.Inside();
            // _ureduce's prologue, the quantile family's one copy (QuantileEngine.UReduce): `axis = normalize_axis_tuple(axis,
            // nd)` (every axis checked, then the repeats), keepdims' `out[(Ellipsis,) + index_out]` (index 0 on every reduced
            // axis, every axis without one), then the merge of several reduced axes (`reshape(kept + (-1,))`, whose -1 NumPy
            // cannot resolve beside a kept axis of length 0). Only its errors matter here: the arrays it makes are reclaimed.
            QuantileEngine.UReduceChecks(a, axes, @out, keepdims);
            // An axis argument (the empty tuple included) reaches `np.partition(a, kth, axis=...)` or `a.partition(kth,
            // axis=axis)`, the string partition's keyword refusal; only overwrite_input without one partitions the text.
            if (!overwrite_input || axes is not null)
                return NDCharArrayBoundary.PartitionKeywordError();

            // `part = a.ravel(); part.partition(kth)`: the string partition of the flattened text around the kth indices
            // written as text — [n // 2 - 1, n // 2] for an even length, [(n - 1) // 2] for an odd one. np.strings.partition
            // broadcasts the text against that separator list with its find ufunc's four operands (a, sep, start, end).
            long n = a.size;
            long k = n % 2 == 0 ? 2 : 1;
            if (n != k && n != 1 && k != 1)
                return new IncorrectShapeException($"operands could not be broadcast together with shapes ({n},) ({k},) () () ");
            // The string partition computed and left `part` as it was: `mean(part[indexer], axis=0, out=out)` refuses the text
            // as the median of plain text does — except a U chararray into a StringDType out, whose mean concatenates the
            // elements at the middle POSITIONS (nothing was moved) into the out before its division refuses.
            if (QuantileEngine.MedianSumsIntoText(a, @out))
            {
                using var part = np.ravel(a);
                long half = n / 2;
                using var middle = n % 2 == 1 ? part[new Slice(half, half + 1)] : part[new Slice(half - 1, half + 1)];
                // keepdims without an axis: `out[(Ellipsis,) + (0,) * nd]` (as indexed above).
                NDArray target = null;
                if (keepdims)
                {
                    var index = new Slice[1 + nd];
                    index[0] = Slice.Ellipsis;
                    for (int d = 0; d < nd; d++)
                        index[1 + d] = Slice.Index(0);
                    target = @out[index];
                }
                using var ownedTarget = target;
                NDArray into = target ?? @out;
                if (!into.Shape.IsWriteable)
                    throw new ValueError("output array is read-only");
#pragma warning disable NDW012 // not a leak: SumNumPyOfCast returns its out (into), which the caller's out owns
                Backends.Strings.NDVStringReduction.SumNumPyOfCast(middle, 0, keepdims: false, into);
#pragma warning restore NDW012
                DefaultEngine.DivideByCountLikeNumPy(into, middle.size);
            }
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, (int?)null, @out);
            throw new InvalidOperationException("np.median over text must refuse in the closing mean.");
        }
    }
}
