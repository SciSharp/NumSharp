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
            // Text (input or out=): NumPy's _ureduce normalizes the axis (a 0-d input refuses every axis), then the
            // closing mean(part, out=out) resolves add on (out, part) and refuses with its verbatim text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            // StringDType: NumPy sorts the text (its NA rules), slices the middle and calls mean, whose concatenating sum
            // meets the divide (only NaN-like NAs interpolate into nan).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                return Backends.Strings.NDVStringStatistics.Median(a, axis.HasValue ? new[] { axis.Value } : null, @out, keepdims);
            // A number the closing mean sums in an implicit loop, or U text into a StringDType out: NumPy computes that
            // sum INTO the text out, then its division refuses (ThrowMedianIntoTextOut, which always throws).
            if (MedianSumsIntoText(a, @out))
                ThrowMedianIntoTextOut(a, axis.HasValue ? new[] { axis.Value } : null, @out, keepdims);
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
        /// <exception cref="IncorrectShapeException">A <c>StringDType</c> array with several reduced axes beside a kept axis of
        ///     length 0 (NumPy's <c>_ureduce</c> merge cannot resolve its -1).</exception>
        /// <remarks>A <c>numpy.matrix</c> operand is computed as a plain array and the result is plain: this overload has
        ///     no matrix rule, because NumPy's tuple-axis answer for a matrix is an artifact of matrix semantics inside its
        ///     own Python code (<c>np.median(m, axis=())</c> reduces axis 1).</remarks>
        /// <exception cref="TypeError">A <c>numpy.char.chararray</c> operand: NumPy's
        ///     <c>a.partition(kth, axis=...)</c> is the STRING partition
        ///     (<c>chararray.partition() got an unexpected keyword argument 'axis'</c>), after the axes and keepdims' out are
        ///     checked. With <c>overwrite_input</c> and no axis NumPy partitions the text itself instead: its broadcast error
        ///     for an even length other than 2 (<see cref="IncorrectShapeException"/>), else the closing mean's refusal of
        ///     text.</exception>
        public static NDArray median(NDArray a, int[] axis,
            NDArray @out = null, bool overwrite_input = false, bool keepdims = false)
        {
            // a numpy.char.chararray operand (decided before the matrix region below, which would hide it): the STRING
            // partition's answer (MedianOfChararray, NDCharArrayBoundary)
            if (NDCharArrayBoundary.Sees(a))
                throw MedianOfChararray(a, axis, @out, overwrite_input, keepdims);
            using var plain = NDMatrixBoundary.PlainFor(a, @out);   // no matrix rule here: a numpy.matrix operand is computed as a plain array (NDMatrixBoundary)
            // Text (input or out=): NumPy's _ureduce normalizes the tuple (every axis, then repeats), then the closing
            // mean(part, out=out) refuses on (out, part) with its verbatim text; see NDStringUfuncs.ReduceError /
            // ReduceOutError.
            // StringDType: NumPy sorts the text (its NA rules), slices the middle and calls mean (see the one-axis overload).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                return Backends.Strings.NDVStringStatistics.Median(a, axis, @out, keepdims);
            // The closing mean's sum computed into a text out before its division refuses (see the one-axis overload).
            if (MedianSumsIntoText(a, @out))
                ThrowMedianIntoTextOut(a, axis, @out, keepdims);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis, @out);
            return QuantileEngine.Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     Whether NumPy's <c>np.median</c> of <paramref name="a"/> into <paramref name="out"/> computes the closing
        ///     mean's SUM into a text out before its division refuses: a number <c>_mean</c> sums in an implicit loop (bool
        ///     and the integers in float64, float16 in float32 — the sum is then cast into the text), or <c>U</c> text into a
        ///     <c>StringDType</c> out (the <c>StringDType</c> add promoter's concatenation).
        /// </summary>
        /// <param name="a">The data (a <c>StringDType</c> input has its own route and is never asked).</param>
        /// <param name="out">The <c>out=</c> argument, or null.</param>
        /// <returns>True when the call takes <see cref="ThrowMedianIntoTextOut"/>.</returns>
        private static bool MedianSumsIntoText(NDArray a, NDArray @out)
        {
            if (@out is null || !Backends.Strings.NDStringUfuncs.IsText(@out) || Backends.Strings.NDStringUfuncs.IsVString(a))
                return false;
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                return a.dtype.Meta == DTypeRegistry.Str && Backends.Strings.NDStringUfuncs.IsVString(@out);
            return MedianMeanLoop(a.typecode) is not null;
        }

        /// <summary>
        ///     The loop NumPy's <c>_mean</c> sums a dtype in when no <c>dtype=</c> is given: float64 for bool and the integers
        ///     (NumSharp's char, the uint16 twin, included), float32 for float16; null for every other dtype, which sums in
        ///     its own dtype.
        /// </summary>
        /// <param name="code">The input's type code.</param>
        /// <returns>The implicit loop, or null.</returns>
        private static DType MedianMeanLoop(NPTypeCode code) => code switch
        {
            NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64 or NPTypeCode.Char => np.float64,
            NPTypeCode.Half => np.float32,
            _ => null,
        };

        /// <summary>
        ///     NumPy's <c>np.median</c> into a TEXT <c>out=</c> whose closing mean computes before it refuses (see
        ///     <see cref="MedianSumsIntoText"/>), statement by statement: <c>_ureduce</c> normalizes the axes, indexes a
        ///     keepdims out and merges several reduced axes; <c>_median</c> partitions the data and slices the middle
        ///     elements; <c>mean(middle, axis, out=out)</c> parses the out (its read-only check), sums the middle INTO it — in
        ///     the implicit loop, the sum cast into the text, or as the <c>StringDType</c> concatenation — and then
        ///     <c>true_divide(out, rcount, out=out)</c> has no text loop. So the out holds the sum when the refusal is thrown
        ///     (<c>np.median(np.array([4, 1, 3, 2]), out=np.zeros((), 'U6'))</c> leaves <c>'5.0'</c>; with <c>['d', 'a', 'c',
        ///     'b']</c> and a <c>StringDType</c> out it leaves <c>'bc'</c>, probed).
        /// </summary>
        /// <param name="a">The data: a bool, integer, char or float16 array, or a <c>U</c> array with a <c>StringDType</c>
        ///     out.</param>
        /// <param name="axes">The reduced axes as the caller passed them, or null for every axis.</param>
        /// <param name="out">The text out.</param>
        /// <param name="keepdims">NumPy's <c>keepdims</c>: the out is indexed at 0 on every reduced axis first.</param>
        /// <exception cref="AxisError">An axis is out of bounds (<c>_ureduce</c>'s first check).</exception>
        /// <exception cref="ValueError">An axis repeats, a read-only out, an out of the wrong shape (the reduction
        ///     iterator's remapped-shapes text), or an empty <c>U</c> reduction (the concatenation has no identity).</exception>
        /// <exception cref="IndexError">keepdims with an out NumPy cannot index.</exception>
        /// <exception cref="TypeError">The division's refusal of the text out (NumPy's <c>ufunc 'divide' not supported for
        ///     the input types ...</c> for <c>S</c> / <c>U</c>, its no-loop text for <c>StringDType</c>): this method never
        ///     returns normally.</exception>
        /// <exception cref="InvalidOperationException">The division did not refuse (a wiring error: it always does for
        ///     text).</exception>
        [NDScoped] // the keepdims view, the merged and partitioned data, the middle slice and the numeric sum are this call's own
        private static void ThrowMedianIntoTextOut(NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            int nd = a.ndim;
            // _ureduce: `axis = _nx.normalize_axis_tuple(axis, nd)` — every axis checked, then the repeats.
            Backends.Strings.NDStringUfuncs.ValidateReduceAxes("median", nd, axes);
            int[] reduced = axes is null ? null : QuantileEngine.NormalizeAxes(axes, nd);
            NDArray target = @out;
            if (keepdims)
            {
                // `out[(Ellipsis,) + index_out]`: index 0 on every reduced axis (every axis without an axis argument).
                var index = new Slice[1 + nd];
                index[0] = Slice.Ellipsis;
                for (int d = 0; d < nd; d++)
                    index[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.Index(0) : Slice.All;
                target = @out[index];
            }

            NDArray data;
            int axis;
            if (reduced is null)
            {
                // `partition(a, kth, axis=None)` ravels the data; the mean then reduces axis 0 of the partition.
                data = np.ravel(a);
                axis = 0;
            }
            else if (reduced.Length == 1)
            {
                data = a;
                axis = reduced[0];
            }
            else
            {
                // Several (or no) reduced axes: the kept ones moved to the front, the rest merged into one trailing axis
                // (`a.reshape(a.shape[:nkeep] + (-1,))`); the order inside the merged axis cannot change the middle values.
                var keep = new System.Collections.Generic.List<int>();
                for (int d = 0; d < nd; d++)
                    if (Array.IndexOf(reduced, d) < 0)
                        keep.Add(d);
                var destinations = new int[keep.Count];
                for (int i = 0; i < destinations.Length; i++)
                    destinations[i] = i;
                NDArray moved = np.moveaxis(a, keep.ToArray(), destinations);
                var merged = new long[keep.Count + 1];
                for (int d = 0; d < keep.Count; d++)
                    merged[d] = moved.Shape.dimensions[d];
                merged[keep.Count] = -1;
                data = moved.reshape(Shape.Adopt(merged));
                axis = keep.Count;
            }

            // _median: the partition indexes (the two central positions for an even length, one for an odd), with -1 added
            // for an inexact dtype (float16); np.partition places the middle values (a text partition is the sorted copy).
            long sz = data.Shape.dimensions[axis];
            var kth = new System.Collections.Generic.List<int>(3);
            if (sz % 2 == 0)
            {
                kth.Add((int)(sz / 2 - 1));
                kth.Add((int)(sz / 2));
            }
            else
                kth.Add((int)((sz - 1) / 2));
            if (a.typecode == NPTypeCode.Half)
                kth.Add(-1);
            NDArray part = np.partition(data, kth.ToArray(), axis);
            long half = part.Shape.dimensions[axis] / 2;
            var indexer = new Slice[part.ndim];
            for (int d = 0; d < part.ndim; d++)
                indexer[d] = Slice.All;
            indexer[axis] = part.Shape.dimensions[axis] % 2 == 1 ? new Slice(half, half + 1) : new Slice(half - 1, half + 1);
            NDArray middle = part[indexer];

            // mean(middle, axis=axis, out=target): the count, then umr_sum's out= (read-only check), its iterator's shape
            // checks and the sum written into the out, then the in-place division, which refuses the text out.
            long rcount = middle.Shape.dimensions[axis];
            if (!target.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            DType loop = MedianMeanLoop(a.typecode);
            if (loop is null)
                Backends.Strings.NDVStringReduction.SumNumPyOfCast(middle, axis, keepdims: false, target);
            else
            {
                ReductionArguments.ValidateOperands(middle, ReductionArguments.AxisFlags(middle.ndim, axis), false, target, null, "add");
                NDArray total = middle.TensorEngine.ReduceAdd(middle, axis, keepdims: false, loop);
                copyto(target, total, casting: "unsafe");
            }
            DefaultEngine.DivideByCountLikeNumPy(target, rcount);
            throw new InvalidOperationException("np.median's division by the count must refuse a text out.");
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
            // _ureduce: `axis = _nx.normalize_axis_tuple(axis, nd)` — every axis checked, then the repeats.
            Backends.Strings.NDStringUfuncs.ValidateReduceAxes("median", a.ndim, axes);
            int nd = a.ndim;
            int[] reduced = axes is null ? null : QuantileEngine.NormalizeAxes(axes, nd);
            // The plain view the remaining statements read: no subclass rule may run on them.
            using var region = NDCharArrayBoundary.Inside();
            if (keepdims && @out is not null)
            {
                // `out[(Ellipsis,) + index_out]`: index 0 on every reduced axis (every axis without one), before the call.
                var index = new Slice[1 + nd];
                index[0] = Slice.Ellipsis;
                for (int d = 0; d < nd; d++)
                    index[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.Index(0) : Slice.All;
                using var target = @out[index];
            }
            if (reduced is not null && reduced.Length > 1)
            {
                // The merge of several reduced axes: the kept ones moved to the front, then `reshape(kept + (-1,))`, whose -1
                // NumPy cannot resolve beside a kept axis of length 0.
                var keep = new System.Collections.Generic.List<int>();
                for (int d = 0; d < nd; d++)
                    if (Array.IndexOf(reduced, d) < 0)
                        keep.Add(d);
                var destinations = new int[keep.Count];
                for (int i = 0; i < destinations.Length; i++)
                    destinations[i] = i;
                using var moved = np.moveaxis(a, keep.ToArray(), destinations);
                var merged = new long[keep.Count + 1];
                for (int d = 0; d < keep.Count; d++)
                    merged[d] = moved.Shape.dimensions[d];
                merged[keep.Count] = -1;
                using var reshaped = moved.reshape(Shape.Adopt(merged));
            }
            if (!overwrite_input || reduced is not null)
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
            if (MedianSumsIntoText(a, @out))
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
                Backends.Strings.NDVStringReduction.SumNumPyOfCast(middle, 0, keepdims: false, into);
                DefaultEngine.DivideByCountLikeNumPy(into, middle.size);
            }
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, (int?)null, @out);
            throw new InvalidOperationException("np.median over text must refuse in the closing mean.");
        }
    }
}
