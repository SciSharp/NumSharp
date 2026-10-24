using System;
using NumSharp.Backends;

namespace NumSharp.Statistics
{
    internal static partial class QuantileEngine
    {
        /// <summary>
        ///     NumPy's <c>np.median</c> body over one axis (or the flattened data): the text routes in NumPy's order, then the
        ///     numeric median through <see cref="Compute"/>. The facade applies the class rules (<c>numpy.matrix</c>,
        ///     <c>numpy.char.chararray</c>) first and calls this for everything else.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it, a
        ///     <c>StringDType</c> array is sorted first).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The median (NaN for an empty slice); <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/>: NumPy's closing <c>mean</c> refuses (its
        ///     verbatim text). A <c>StringDType</c> array is sorted first, then its mean's divide refuses the concatenated
        ///     text.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a line of two or more elements, an empty
        ///     reduced axis (the mean's add.reduce has no identity) or a read-only <paramref name="out"/>.</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index
        ///     (<c>_ureduce</c>'s <c>out[(Ellipsis,) + index_out]</c>: too few dimensions, or a reduced axis of length 0),
        ///     for every dtype and before the text refusals.</exception>
        /// <remarks>
        ///     Moved from <c>np.median</c> (<c>Statistics/np.median.cs</c>) by the np-layer dtype plan's P0 pilot, statement for
        ///     statement and in the same order, so the facade holds no dtype test. Since the plan's P1a step 2 every route
        ///     runs NumPy's one <c>_ureduce</c> prologue (<see cref="UReduce"/>) first; until then each route had its own copy,
        ///     and the copies disagreed (the plan's Appendix B1).
        /// </remarks>
        internal static NDArray Median(NDArray a, int? axis, NDArray @out, bool overwrite_input, bool keepdims)
        {
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
            // Fixed-width text data or a text out: _ureduce's prologue runs before _median's closing mean refuses, so the axis
            // and keepdims' out index (an out NumPy cannot index is its IndexError) come first. The numeric route runs the
            // same prologue inside Compute.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || (@out is not null && Backends.Strings.NDStringUfuncs.IsText(@out)))
                UReduceChecks(a, axis, @out, keepdims);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis, @out);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // NumPy's np.median returns nan for an empty slice (np.quantile/percentile raise);
            // emptyReturnsNaN routes the empty-axis case to a nan fill instead. medianMean selects
            // the mean-of-middle reduction (NumPy's np.median), not the quantile(0.5) lerp.
            return Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     NumPy's <c>np.median</c> body over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the text routes in NumPy's
        ///     order, then the numeric median through <see cref="Compute"/>, the reduced axes merged into one. The facade
        ///     applies the <c>numpy.char.chararray</c> rule first and computes a <c>numpy.matrix</c> as a plain array.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened data, an
        ///     EMPTY list reduces nothing.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The median over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats; or, for a <c>StringDType</c> array, a None-like NA in a sorted line
        ///     of two or more elements, an empty reduced axis or a read-only <paramref name="out"/>.</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data or a text <paramref name="out"/> that NumPy's closing mean refuses (its
        ///     verbatim text); a <c>StringDType</c> array after its sort.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0, any dtype (NumPy's
        ///     <c>_ureduce</c> merge cannot resolve its -1: <c>cannot reshape array of size 0 into shape (0,newaxis)</c>);
        ///     before the plan's P1a step 2 only a <c>StringDType</c> array raised it (plan Appendix B1).</exception>
        /// <exception cref="IndexError"><paramref name="keepdims"/> with an <paramref name="out"/> NumPy cannot index, for
        ///     every dtype and before the text refusals.</exception>
        /// <remarks>Moved from <c>np.median</c> by the np-layer dtype plan's P0 pilot, statement for statement.</remarks>
        internal static NDArray Median(NDArray a, int[] axis, NDArray @out, bool overwrite_input, bool keepdims)
        {
            // Text (input or out=): NumPy's _ureduce normalizes the tuple (every axis, then repeats), then the closing
            // mean(part, out=out) refuses on (out, part) with its verbatim text; see NDStringUfuncs.ReduceError /
            // ReduceOutError.
            // StringDType: NumPy sorts the text (its NA rules), slices the middle and calls mean (see the one-axis overload).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                return Backends.Strings.NDVStringStatistics.Median(a, axis, @out, keepdims);
            // The closing mean's sum computed into a text out before its division refuses (see the one-axis overload).
            if (MedianSumsIntoText(a, @out))
                ThrowMedianIntoTextOut(a, axis, @out, keepdims);
            // Fixed-width text data or a text out: _ureduce's prologue first — the axes, keepdims' out index and the merge of
            // several axes (a kept axis of length 0 beside them is NumPy's reshape error: plan Appendix B1) — then the
            // closing mean's refusal.
            if (Backends.Strings.NDStringUfuncs.IsString(a) || (@out is not null && Backends.Strings.NDStringUfuncs.IsText(@out)))
                UReduceChecks(a, axis, @out, keepdims);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("median", a, axis, @out);
            return Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     NumPy's <c>np.nanmedian</c> body over one axis (or the flattened data): the text routes in NumPy's order, then
        ///     the NaN-ignoring median through <see cref="Compute"/>. Like <c>np.median</c>, an even-sized (NaN-free) slice
        ///     reduces to the MEAN of its two central values (NumPy's mean-of-middle rule, not <c>np.nanquantile(a, 0.5)</c>,
        ///     which differs by up to 1 ULP); a slice that is entirely NaN, or empty, gives NaN. The facade applies the
        ///     <c>numpy.matrix</c> rule first and calls this for everything else.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it, a
        ///     <c>StringDType</c> array drops its NaN-like NAs and takes the median of the rest).</param>
        /// <param name="axis">The axis to reduce (negative counts from the end), or null for the flattened data.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axis is kept with length one.</param>
        /// <returns>The NaN-ignoring median; <paramref name="out"/> itself when one is given. Over a <c>StringDType</c>
        ///     array a median exists only for an all-NaN reduction (its NA object, or <c>np.float64(nan)</c>).</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Fixed-width text data that NumPy's NaN test (or, for empty data, its <c>nanmean</c>)
        ///     refuses (its verbatim text); a <c>StringDType</c> array's median, numpy.ma or keepdims refusals.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> array: a None-like NA in a sorted line of two or more elements,
        ///     an empty input's mean (no identity), or the store into <paramref name="out"/>.</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> reduction along a long axis whose row result NumPy
        ///     stores in an OBJECT array: NumSharp has no object dtype.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index (<c>_ureduce</c>'s prologue), before the text refusals.</exception>
        /// <remarks>
        ///     Moved from <c>np.nanmedian</c> (<c>Statistics/np.nanmedian.cs</c>) by the np-layer dtype plan's P1a (step 1),
        ///     statement for statement and in the same order, so the facade holds no dtype test (the move <see cref="Median"/>
        ///     made in the plan's P0 pilot).
        /// </remarks>
        internal static NDArray NanMedian(NDArray a, int? axis, NDArray @out, bool overwrite_input, bool keepdims)
        {
            // Text: NumPy sends EMPTY data to np.nanmean(a, axis, out=out) first (whose add.reduce resolves on (out, a));
            // otherwise its _ureduce normalizes the axis (a 0-d input refuses every axis) and the NaN test refuses text;
            // see NDStringUfuncs.ReduceError / ReduceOutError.
            // StringDType: np.nanmean for an empty input, else the NaN-like NAs removed and the median of the rest (numpy.ma's
            // median along an axis shorter than 600, np.apply_along_axis along a longer one).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                return Backends.Strings.NDVStringStatistics.NanMedian(a, axis.HasValue ? new[] { axis.Value } : null, @out, keepdims);
            if (a is not null && a.size == 0)
                Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmean", a, axis, @out);
            // Fixed-width text data (not empty: that answered np.nanmean above): _ureduce's prologue — the axes, keepdims'
            // out index — runs before the NaN test refuses the text.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                UReduceChecks(a, axis, @out, keepdims);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmedian", a, axis);
            int[] axisArr = axis.HasValue ? new[] { axis.Value } : null;
            // medianMean selects NumPy's mean-of-middle reduction; emptyReturnsNaN and ignoreNaN are the nan forms' rules.
            return Compute(a, new[] { 0.5 }, axisArr, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
                allowBooleanContinuous: true, medianMean: true);
        }

        /// <summary>
        ///     NumPy's <c>np.nanmedian</c> body over SEVERAL axes at once (NumPy's tuple <c>axis</c>): the text routes in
        ///     NumPy's order, then the NaN-ignoring median through <see cref="Compute"/>, the reduced axes merged into one.
        ///     The facade computes a <c>numpy.matrix</c> as a plain array.
        /// </summary>
        /// <param name="a">The data (bool computes as its 0 / 1 values; text is refused as NumPy refuses it).</param>
        /// <param name="axis">The axes to reduce (negative values count from the end); null reduces the flattened data, an
        ///     EMPTY list reduces nothing.</param>
        /// <param name="out">An optional output array of the result's shape, written and returned itself.</param>
        /// <param name="overwrite_input">Whether the data may be partitioned in place (NumSharp stages a copy anyway).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <returns>The NaN-ignoring median over the reduced axes; <paramref name="out"/> itself when one is given.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="a"/> is null.</exception>
        /// <exception cref="ValueError">An axis repeats; or a <c>StringDType</c> array's errors (see the one-axis
        ///     overload).</exception>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ArgumentException"><paramref name="out"/> does not have the result's shape.</exception>
        /// <exception cref="TypeError">Text data that NumPy's NaN test (or, for empty text data, its <c>nanmean</c>)
        ///     refuses; a <c>StringDType</c> array's refusals (see the one-axis overload).</exception>
        /// <exception cref="NotSupportedException">A <c>StringDType</c> row result NumPy stores in an OBJECT array.</exception>
        /// <exception cref="IndexError">Data that is not empty, under <paramref name="keepdims"/>, with an
        ///     <paramref name="out"/> NumPy cannot index (<c>_ureduce</c>'s prologue), before the text refusals. (Data that is
        ///     not empty never fails the prologue's merge; empty data answers <c>np.nanmean</c> before it.)</exception>
        /// <remarks>Moved from <c>np.nanmedian</c> by the np-layer dtype plan's P1a (step 1), statement for statement.</remarks>
        internal static NDArray NanMedian(NDArray a, int[] axis, NDArray @out, bool overwrite_input, bool keepdims)
        {
            // Text: NumPy sends EMPTY data to np.nanmean(a, axis, out=out) first (whose add.reduce resolves on (out, a));
            // otherwise its _ureduce normalizes the tuple (every axis, then repeats) and the NaN test refuses text; see
            // NDStringUfuncs.ReduceError / ReduceOutError.
            // StringDType: see the one-axis overload.
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
                return Backends.Strings.NDVStringStatistics.NanMedian(a, axis, @out, keepdims);
            if (a is not null && a.size == 0)
                Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmean", a, axis, @out);
            // Fixed-width text data (not empty: that answered np.nanmean above): _ureduce's prologue — the axes, keepdims'
            // out index — runs before the NaN test refuses the text.
            if (Backends.Strings.NDStringUfuncs.IsString(a))
                UReduceChecks(a, axis, @out, keepdims);
            Backends.Strings.NDStringUfuncs.ThrowIfReduce("nanmedian", a, axis);
            return Compute(a, new[] { 0.5 }, axis, @out, overwrite_input,
                QuantileMethod.Linear, keepdims, qIsScalar: true, emptyReturnsNaN: true, ignoreNaN: true,
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
        /// <remarks>Also read by <c>np.median</c>'s <c>numpy.char.chararray</c> rule (<c>MedianOfChararray</c>), whose closing
        ///     mean meets the same out.</remarks>
        internal static bool MedianSumsIntoText(NDArray a, NDArray @out)
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
            // _ureduce, the family's one prologue (UReduce): `axis = _nx.normalize_axis_tuple(axis, nd)` (every axis checked,
            // then the repeats), keepdims' `out[(Ellipsis,) + index_out]` (index 0 on every reduced axis, every axis without an
            // axis argument), and the merge of several (or no) reduced axes into one trailing axis behind the kept ones
            // (`a.reshape(a.shape[:nkeep] + (-1,))`; the order inside the merged axis cannot change the middle values).
            UReduced prologue = UReduce(a, axes, @out, keepdims);
            NDArray target = prologue.Target;

            NDArray data;
            int axis;
            if (prologue.Reduced is null)
            {
                // `partition(a, kth, axis=None)` ravels the data; the mean then reduces axis 0 of the partition.
                data = np.ravel(a);
                axis = 0;
            }
            else
            {
                data = prologue.Data;
                axis = prologue.Axis.Value;
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
                np.copyto(target, total, casting: "unsafe");
            }
            DefaultEngine.DivideByCountLikeNumPy(target, rcount);
            throw new InvalidOperationException("np.median's division by the count must refuse a text out.");
        }
    }
}
