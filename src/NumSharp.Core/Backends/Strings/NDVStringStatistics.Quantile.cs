using System;
using System.Collections.Generic;
using System.Numerics;
using NumSharp.Backends.Sorting;
using NumSharp.Statistics;

// =============================================================================
// NDVStringStatistics.Quantile.cs — the order statistics over NEP 55 StringDType text: np.median, the interpolating
// methods of np.quantile / np.percentile, np.nanmedian, np.nanquantile / np.nanpercentile, numpy.ma's median, and a
// StringDType q
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_function_base_impl.py, _nanfunctions_impl.py, numpy/ma/extras.py; probed against 2.4.2)
//   * Every one of them is Python code over ufuncs. StringDType has a sort (its descriptor's compare) but no selection
//     loop, so each `partition` is a full sort of the line: a None-like NA in a line of two or more elements raises
//     "Cannot compare null that is not a nan-like value" there, before anything else reads the data.
//   * `_ureduce` normalizes the axes, indexes `out` with `(Ellipsis,) + index_out` under keepdims, merges several reduced
//     axes into one last axis, runs the reduction, and returns `out` itself or `r[(Ellipsis,) + index_r]` under keepdims —
//     which fails when the reduction returned a Python object (a str: "string indices must be integers, not 'tuple'"; the
//     NaN NA object: "'float' object is not subscriptable") and works on a NumPy scalar (np.float64).
//   * `_median` partitions, slices the one or two middle elements and calls `mean`: StringDType's add CONCATENATES, so the
//     mean always ends in a refusal except over NaN-like NAs (`np.float64(nan)`).
//   * `_quantile`'s interpolating methods read the two neighbours and call `_lerp`, whose first statement is the Python
//     subtraction `b - a`: of two Python objects for a scalar result ("unsupported operand type(s) for -: 'str' and
//     'str'"; two NaN NA objects subtract into NaN and the result is `np.float64(nan)`), of two StringDType arrays
//     otherwise (subtract has no StringDType loop).
//   * The nan forms remove the NaN-like NAs (`np.isnan` is true only for them) and run the plain statistic on what is left;
//     along an axis they go through `np.apply_along_axis`, whose buffer takes the FIRST row's result dtype — a `<U` of the
//     first string's length (later strings truncated, a later NaN written as 'nan' truncated), or float64 when the first
//     row was all NaN (a later string then fails to parse). nanmedian along an axis shorter than 600 goes through
//     numpy.ma's median instead, whose masked sort asks StringDType for an extremum fill and refuses.
//   * numpy.ma's median of a StringDType masked array without a mask sorts the data and ends in the masked mean's
//     `true_divide` refusal (its sum is a masked ndarray, never a Python object).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The statements above in NumPy's order, over NumSharp's StringDType sort (VStringSort), reductions
//   (NDVStringReduction), ufunc refusals and store rules (PyArray_Pack: NDArray.PackForStore / PackForVStringStore).
//   NumSharp has no Python objects: a reduction's "Python object" is a read-only 0-d StringDType array (its element is the
//   str or the NA object), a NumPy scalar a read-only 0-d numeric array.
// =============================================================================

namespace NumSharp.Backends.Strings
{
    internal static partial class NDVStringStatistics
    {
        /// <summary>
        ///     The axis length from which <c>np.nanmedian</c> along an axis leaves numpy.ma's median for
        ///     <c>np.apply_along_axis</c> (NumPy's <c>if a.shape[axis] &lt; 600: return _nanmedian_small(...)</c>).
        /// </summary>
        private const long NanMedianSmallLimit = 600;

        // =============================================================================================================
        // mean with out= (the closing statement of the median family)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_mean(a, axis, out=out, keepdims=keepdims)</c> over a <c>StringDType</c> array: without
        ///     <paramref name="out"/> it is <see cref="Mean"/>; with one, <c>umr_sum(arr, axis, out=out)</c> — its
        ///     read-only check, its loop resolution on (out, a), its reorderable / shape / identity checks, the sum written
        ///     INTO <paramref name="out"/> — and then <c>true_divide(ret, rcount, out=ret)</c> on the out, which has no loop
        ///     for text (a <c>StringDType</c> out: <c>(StringDType, Int64DType)</c>; a <c>U</c> / <c>S</c> out: the legacy
        ///     refusal).
        /// </summary>
        /// <param name="op">The NumPy function whose loop checks run (<c>"mean"</c>, <c>"nanmean"</c>).</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The reduced axes, normalized by the caller (null for every axis, empty for none).</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>Only without <paramref name="out"/>, the one mean that exists (see <see cref="Mean"/>).</returns>
        /// <exception cref="ValueError">A read-only out, two or more reduced axes, an out of the wrong rank or length, an
        ///     empty reduced axis, or the add loop's NA refusal.</exception>
        /// <exception cref="TypeError">The loop's refusal (an out of another class, incompatible NA objects) or the divide's.</exception>
        /// <remarks>
        ///     The sum reaches the out before the divide refuses, as in NumPy, so a caller that catches the error finds the
        ///     concatenated text in its out.
        /// </remarks>
        internal static NDArray MeanNumPy(string op, NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            if (@out is null)
                return Mean(op, a, axes, keepdims, null);
            // umr_sum is a direct ufunc.reduce: out= is parsed (the read-only check) before anything else.
            if (!@out.Shape.IsWriteable)
                throw new ValueError("output array is read-only");
            long rcount = ReducedCount(a, axes);
            NDArray ret;
            if (axes is { Length: > 1 })
            {
                // Two or more axes: the loop resolves first, then the StringDType add is not reorderable.
                if (NDStringUfuncs.VStringReduceLoopAnswer(op, a, null, @out) is { } refusal)
                    throw refusal;
                throw NDStringUfuncs.VStringNotReorderable("add", a, axes);
            }
            if (axes is { Length: 0 })
            {
                // add.reduce over NO axis copies the elements into the out (after the loop and the out's shape check).
                if (NDStringUfuncs.VStringReduceLoopAnswer(op, a, null, @out) is { } refusal)
                    throw refusal;
                ReductionArguments.ValidateOperands(a, new bool[a.ndim], keepdims, @out, null, "add");
                np.copyto(@out, a, "unsafe");
                ret = @out;
            }
            else
            {
                ret = NDVStringReduction.SumNumPy(op, a, axes is null ? null : axes[0], keepdims, @out);
            }
            using var count = NDArray.Scalar(rcount);
            // `um.true_divide(ret, rcount, out=ret, casting='unsafe')`: no loop divides text.
            using var quotient = ret.TensorEngine.Divide(ret, count, null, ret);
            throw new InvalidOperationException("np.true_divide of a text sum returned.");
        }

        // =============================================================================================================
        // np.median
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.median</c> over a <c>StringDType</c> array: <c>_ureduce</c> (the axes, keepdims' out index, the
        ///     merge of several axes) around <c>_median</c> — the partition (a sort), the one or two middle elements, and
        ///     <c>mean(part[indexer], axis=axis, out=out)</c>.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The <c>axis</c> argument as given (null for every axis; may be empty).</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Keep the reduced axes with length one.</param>
        /// <returns>The one median that exists: <c>np.float64(nan)</c> when the middle elements are NaN-like NAs (a
        ///     read-only 0-d float64, or under <paramref name="keepdims"/> a writeable view of NumPy's scalar indexed with new
        ///     axes); every other input raises.</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="ValueError">A repeated axis; a None-like NA in a sorted line of two or more elements; an empty
        ///     reduced axis (the mean's add.reduce has no identity); a read-only out; or the mean's other checks.</exception>
        /// <exception cref="TypeError">The mean's divide: NumPy's legacy refusal for a text sum that is a Python str,
        ///     <c>(StringDType, Int64DType)</c> for an ndarray sum, Python's <c>None / int</c> text; or the loop's refusal of
        ///     an out of another class.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0 (<c>_ureduce</c>'s
        ///     merge cannot resolve its -1).</exception>
        /// <exception cref="IndexError">keepdims with an out of too few dimensions (NumPy's <c>out[(Ellipsis,) +
        ///     index_out]</c>).</exception>
        /// <remarks>Scoped: the merged and sorted copies and the middle slice are reclaimed; only the result leaves.</remarks>
        [NDScoped]
        internal static NDArray Median(NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            int nd = a.ndim;
            // _ureduce's prologue, the quantile family's one copy (DefaultEngine.UReduce): the axes (every axis checked, then
            // the repeats), keepdims' view of out, then several reduced axes merged into one last axis behind the kept ones.
            UReduced prologue = DefaultEngine.UReduce(a, axes, @out, keepdims);
            int[] reduced = prologue.Reduced;
            NDArray target = prologue.Target;
            NDArray work = prologue.Data;
            int? axis = prologue.Axis;

            // _median: `part = partition(a, kth, axis=axis)` — StringDType has no selection loop, so NumPy sorts each line
            // (a flattened copy for axis=None); its kth values always lie inside the line (or the array is empty).
            NDArray part = AxisSort.SortVString(work, axis, false);
            int ax = axis ?? 0;
            NDArray middle = part[MiddleIndexer(part, ax)];
            // `rout = mean(part[indexer], axis=axis, out=out)`: the concatenating sum, then the divide.
            NDArray r = MeanNumPy("mean", middle, new[] { ax }, target, keepdims: false);
            if (@out is not null)
                return @out;
            return keepdims ? KeepdimsResult(r, nd, reduced) : r;
        }

        /// <summary>
        ///     <c>_median</c>'s indexer: the one middle element of an odd line (<c>slice(index, index + 1)</c>) or the two of an
        ///     even one (<c>slice(index - 1, index + 1)</c>), <c>index = part.shape[axis] // 2</c>, every other axis whole.
        /// </summary>
        /// <param name="part">The sorted array.</param>
        /// <param name="axis">The reduced axis (non-negative).</param>
        /// <returns>The slices (an empty line gives <c>slice(-1, 1)</c>, an empty selection).</returns>
        private static Slice[] MiddleIndexer(NDArray part, int axis)
        {
            long length = part.Shape.dimensions[axis];
            long index = length / 2;
            var indexer = new Slice[part.ndim];
            for (int d = 0; d < indexer.Length; d++)
                indexer[d] = Slice.All;
            indexer[axis] = length % 2 == 1 ? new Slice(index, index + 1) : new Slice(index - 1, index + 1);
            return indexer;
        }

        // =============================================================================================================
        // numpy.ma.median of a StringDType masked array without a mask
        // =============================================================================================================

        /// <summary>
        ///     numpy.ma's <c>median</c> of a <c>StringDType</c> masked array whose mask is <c>nomask</c>: <c>_ureduce</c>'s
        ///     axis normalization, <c>_median</c>'s <c>sort(a, axis)</c> (numpy.ma's sort, which without a mask sorts the data;
        ///     axis None flattens), then the masked mean of the middle — whose sum is a masked NDARRAY, so its
        ///     <c>true_divide</c> refuses whatever the text (NaN-like NAs included); an empty reduced axis meets the mean's
        ///     add.reduce first.
        /// </summary>
        /// <param name="data">The data (a <c>StringDType</c> array).</param>
        /// <param name="axis">The <c>axis</c> argument, or null.</param>
        /// <returns>The exception NumPy raises (never null: this median always refuses).</returns>
        /// <exception cref="AxisError">An axis out of bounds (thrown, as NumPy raises it first).</exception>
        /// <exception cref="ValueError">A None-like NA in a sorted line of two or more elements (thrown by the sort).</exception>
        /// <remarks>
        ///     A plain ndarray handed to numpy.ma's median takes <c>np.median</c> instead (NumPy's <c>if not hasattr(a,
        ///     'mask')</c>), whose 0-d text sum is a Python str and gets the legacy divide refusal: only a masked array reaches
        ///     here.
        /// </remarks>
        internal static Exception MaskedMedian(NDArray data, int? axis)
        {
            int nd = data.ndim;
            int? ax = null;
            if (axis is { } given)
            {
                NDStringUfuncs.ValidateReduceAxes("median", nd, new[] { given });
                ax = given < 0 ? given + nd : given;
            }
            using var sorted = AxisSort.SortVString(data, ax, false);
            // `if asorted.shape[axis] == 0: return np.ma.mean(asorted[indexer], axis=axis)`: an empty reduced axis has no
            // identity for the masked mean's add.reduce, whatever the other axes hold.
            if (sorted.Shape.dimensions[ax ?? 0] == 0)
                return new ValueError("zero-size array to reduction operation add which has no identity");
            // `mid.mean()` / `np.ma.mean(low_high, axis=axis)`: the sum of a masked array is an ndarray.
            return DivideRefusal(null);
        }

        // =============================================================================================================
        // np.nanmedian
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>np.nanmedian</c> over a <c>StringDType</c> array: an empty input is <c>np.nanmean(a, axis, out=out,
        ///     keepdims=keepdims)</c> (StringDType's mean); otherwise <c>_ureduce</c> around <c>_nanmedian</c> — the NaN-like
        ///     NAs removed and np.median of the rest for a flat or 1-D reduction; along an axis shorter than 600, numpy.ma's
        ///     median of the NaN-masked array (its sort asks for an extremum fill StringDType does not have); along a longer
        ///     one, <c>np.apply_along_axis</c> of the 1-D median.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axes">The <c>axis</c> argument as given (null for every axis).</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Keep the reduced axes with length one.</param>
        /// <returns>The NA object of an all-NaN reduction (a read-only 0-d <c>StringDType</c> array holding it, NumPy's
        ///     Python object), the float64 array <c>np.apply_along_axis</c> builds when every row is all NaN, or
        ///     <paramref name="out"/> itself.</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="ValueError">A repeated axis; a None-like NA in a line of two or more elements; an empty input's
        ///     mean (no identity); the assignment into <paramref name="out"/> (<c>cannot convert float NaN to integer</c>).</exception>
        /// <exception cref="TypeError">The median's divide refusal; numpy.ma's <c>Unsuitable type ... for calculating
        ///     minimum.</c>; keepdims over a Python object (<c>'float' object is not subscriptable</c>).</exception>
        /// <exception cref="NotSupportedException">A row result NumPy stores in an OBJECT array (a <c>None</c> NA object
        ///     picked by <c>np.apply_along_axis</c>): NumSharp has no object dtype.</exception>
        /// <remarks>Scoped: the merged copy, the masks, the compacted lines and the row results are reclaimed.</remarks>
        [NDScoped]
        internal static NDArray NanMedian(NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            int nd = a.ndim;
            // `if a.size == 0: return np.nanmean(a, axis, out=out, keepdims=keepdims)`: StringDType has no NaN mask, so
            // nanmean is np.mean — whose axis normalization comes first (`_count_reduce_items`).
            if (a.size == 0)
                return EmptyNanMean(a, axes, @out, keepdims);
            // _ureduce's prologue, the quantile family's one copy (DefaultEngine.UReduce): the axes, keepdims' view of out,
            // the merge of several reduced axes (data that is not empty always merges).
            UReduced prologue = DefaultEngine.UReduce(a, axes, @out, keepdims);
            int[] reduced = prologue.Reduced;
            NDArray target = prologue.Target;
            NDArray work = prologue.Data;
            int? axis = prologue.Axis;

            NDArray r;
            if (axis is null || work.ndim == 1)
            {
                // `part = a.ravel()`; `_nanmedian1d(part)`, assigned into out when one is given.
                r = NanMedian1D(work.ndim == 1 ? work : np.ravel(work));
                if (target is not null)
                {
                    PyAssign(target, r);
                    return @out;
                }
            }
            else if (work.Shape.dimensions[axis.Value] < NanMedianSmallLimit)
            {
                // _nanmedian_small: `np.ma.median(np.ma.masked_array(a, np.isnan(a)), axis=axis)`. np.isnan computes (true
                // for the NaN-like NAs), the mask is an array (never nomask), and the masked sort's argsort asks
                // minimum_fill_value for the fill StringDType does not have.
                using var mask = work.TensorEngine.IsNan(work);
                throw new TypeError($"Unsuitable type {work.dtype} for calculating minimum.");
            }
            else
            {
                r = np.apply_along_axis(row => AsAnyArray(NanMedian1D(row)), axis.Value, work);
                if (target is not null)
                {
                    np.copyto(target, r, "unsafe");
                    return @out;
                }
            }
            if (@out is not null)
                return @out;
            return keepdims ? KeepdimsResult(r, nd, reduced) : r;
        }

        /// <summary>
        ///     NumPy's <c>_nanmedian1d</c>: the NaN-like NAs removed (<c>_remove_nan_1d</c>); an all-NaN line returns its
        ///     last element (<c>arr1d[-1]</c>, the NA object), anything else <c>np.median</c> of what is left.
        /// </summary>
        /// <param name="line">A 1-D <c>StringDType</c> line.</param>
        /// <returns>The NA object as a read-only 0-d <c>StringDType</c> array, or the median (see <see cref="Median"/>).</returns>
        /// <exception cref="Exception">The median's errors.</exception>
        private static NDArray NanMedian1D(NDArray line)
        {
            using var nan = line.TensorEngine.IsNan(line);
            long nans = nan.TensorEngine.CountNonZero(nan);
            if (nans == line.size)
                return ScalarElement(line, line.size - 1);
            // _remove_nan_1d's order (see RemoveNaN1D), as np.median then partitions it.
            using var kept = nans == 0 ? null : RemoveNaN1D(line, nan, nans);
            return Median(kept ?? line, null, null, false);
        }

        /// <summary>
        ///     <c>np.nanmean(a, axis, out=out, keepdims=keepdims)</c> of an EMPTY <c>StringDType</c> array, the first statement
        ///     of the nan forms: StringDType has no NaN mask (<c>_replace_nan</c> returns <c>mask=None</c>), so it is
        ///     <c>np.mean</c>: the axes through <c>_count_reduce_items</c>' <c>normalize_axis_index</c>, then
        ///     <see cref="MeanNumPy"/> — an empty reduced axis has no identity, an empty result divides an empty ndarray.
        /// </summary>
        /// <param name="a">The empty array.</param>
        /// <param name="axes">The <c>axis</c> argument as given, or null.</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <returns>Never returns for an empty array (every form refuses); typed for the callers' symmetry.</returns>
        /// <exception cref="AxisError">An axis out of bounds.</exception>
        /// <exception cref="ValueError">An empty reduced axis, two or more axes (not reorderable), or the out's checks.</exception>
        /// <exception cref="TypeError">The divide's refusal, or the loop's refusal of an out of another class.</exception>
        /// <remarks>
        ///     Internal, not private: a fixed-width text q (<see cref="NDStringUfuncs.QuantileTextQAnswer"/>) reaches the same
        ///     first statement of nanquantile over empty StringDType data.
        /// </remarks>
        internal static NDArray EmptyNanMean(NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            int nd = a.ndim;
            int[] normalized = null;
            if (axes is not null)
            {
                normalized = new int[axes.Length];
                for (int i = 0; i < axes.Length; i++)
                {
                    int ax = axes[i] < 0 ? axes[i] + nd : axes[i];
                    if (ax < 0 || ax >= nd)
                        throw new AxisError(axes[i], nd);
                    normalized[i] = ax;
                }
            }
            return MeanNumPy("mean", a, normalized, @out, keepdims);
        }

        // =============================================================================================================
        // np.nanquantile / np.nanpercentile
        // =============================================================================================================

        /// <summary>
        ///     NumPy's <c>_nanquantile_unchecked</c> over a <c>StringDType</c> array once q passed its checks: an empty input
        ///     is <c>np.nanmean</c>; otherwise <c>_ureduce</c> around <c>_nanquantile_ureduce_func</c> — the flat or 1-D
        ///     <c>_nanquantile_1d</c> (NaN-like NAs removed, np.quantile of the rest, an all-NaN line a q-shaped array of NAs),
        ///     assigned into <paramref name="out"/>; along an axis <c>np.apply_along_axis</c> of it (the buffer takes the first
        ///     row's dtype), q's axes moved first.
        /// </summary>
        /// <param name="op"><c>nanquantile</c> or <c>nanpercentile</c>.</param>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="q">The quantiles in [0, 1] (a percentile already divided by 100), in C order.</param>
        /// <param name="qShape">q's shape: empty for a scalar q.</param>
        /// <param name="qIsInteger">True for an integer or bool q ('linear' then takes the element).</param>
        /// <param name="axes">The <c>axis</c> argument as given (null for every axis).</param>
        /// <param name="out">The out array, or null.</param>
        /// <param name="method">The method name (checked per line, only where a line has a non-NaN element).</param>
        /// <param name="keepdims">Keep the reduced axes with length one.</param>
        /// <param name="qType">q's precision (<see cref="QuantileIndexes.PrecisionOf"/>): each line's virtual indexes are
        ///     computed in it.</param>
        /// <returns>The quantiles: a read-only 0-d <c>StringDType</c> array for NumPy's Python object (a str or the NA
        ///     object), a <c>StringDType</c> array for an array q, the <c>&lt;U</c> / float64 array
        ///     <c>np.apply_along_axis</c> builds along an axis, or <paramref name="out"/> itself.</returns>
        /// <exception cref="Exception">The errors of <see cref="QuantileText.Compute"/> (per line), of the assignment into
        ///     <paramref name="out"/> (NumPy's setitem texts: <c>could not convert string to float: 'b'</c>), of
        ///     <c>np.apply_along_axis</c>'s buffer (<c>could not convert string to float: np.str_('b')</c>), or keepdims over a
        ///     Python object.</exception>
        /// <exception cref="NotSupportedException">A row result NumPy stores in an OBJECT array (a <c>None</c> NA object).</exception>
        /// <remarks>Scoped: the merged copy, the masks, the compacted lines and the row results are reclaimed.</remarks>
        [NDScoped]
        internal static NDArray NanQuantile(string op, NDArray a, double[] q, long[] qShape, bool qIsInteger, int[] axes,
            NDArray @out, string method, bool keepdims, NPTypeCode qType = NPTypeCode.Double)
        {
            int nd = a.ndim;
            if (a.size == 0)
                return EmptyNanMean(a, axes, @out, keepdims);
            // _ureduce's prologue, the quantile family's one copy (DefaultEngine.UReduce): the axes, keepdims' view of out,
            // the merge of several reduced axes (data that is not empty always merges).
            UReduced prologue = DefaultEngine.UReduce(a, axes, @out, keepdims);
            int[] reduced = prologue.Reduced;
            NDArray target = prologue.Target;
            NDArray work = prologue.Data;
            int? axis = prologue.Axis;
            string quantileOp = op == "nanpercentile" ? "percentile" : "quantile";

            NDArray r;
            if (axis is null || work.ndim == 1)
            {
                r = NanQuantile1D(quantileOp, work.ndim == 1 ? work : np.ravel(work), q, qShape, qIsInteger, method, qType);
                // `if out is not None: out[...] = result`.
                if (target is not null)
                    PyAssign(target, r);
            }
            else
            {
                int ax = axis.Value;
                r = np.apply_along_axis(row => AsAnyArray(NanQuantile1D(quantileOp, row, q, qShape, qIsInteger, method, qType)), ax, work);
                // `from_ax = [axis + i for i in range(q.ndim)]; np.moveaxis(result, from_ax, list(range(q.ndim)))` with the
                // axis _ureduce handed the function (prologue.NumPyAxis): the reduced axis, or the UNnormalized -1 for merged
                // axes, which for a q of two or more dimensions moves q's last axis and the first kept axes to the front
                // (NumPy 2.4.2's answer, reproduced; DefaultEngine.AlongAxisLayout's remarks).
                if (qShape.Length != 0)
                {
                    int numpyAxis = prologue.NumPyAxis.Value;
                    var from = new int[qShape.Length];
                    var to = new int[qShape.Length];
                    for (int i = 0; i < from.Length; i++)
                    {
                        from[i] = numpyAxis + i;
                        to[i] = i;
                    }
                    r = np.moveaxis(r, from, to);
                }
                if (target is not null)
                    np.copyto(target, r, "unsafe");
            }
            if (@out is not null)
                return @out;
            return keepdims ? KeepdimsResult(r, nd, reduced) : r;
        }

        /// <summary>
        ///     NumPy's <c>_nanquantile_1d</c>: the NaN-like NAs removed; an all-NaN line is <c>np.full(q.shape, np.nan,
        ///     dtype=arr1d.dtype)[()]</c> (the NA, or q-shaped NAs) — no method check, no q rank check; otherwise
        ///     <c>_quantile_unchecked</c> of what is left (<see cref="QuantileText.Compute"/>).
        /// </summary>
        /// <param name="op"><c>quantile</c> or <c>percentile</c>, for the texts.</param>
        /// <param name="line">A 1-D <c>StringDType</c> line.</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="qShape">q's shape.</param>
        /// <param name="qIsInteger">True for an integer or bool q.</param>
        /// <param name="method">The method name.</param>
        /// <param name="qType">q's precision.</param>
        /// <returns>A read-only 0-d <c>StringDType</c> array (NumPy's Python object), a <c>StringDType</c> array, or the
        ///     interpolation's <c>np.float64</c>.</returns>
        /// <exception cref="Exception">The errors of <see cref="QuantileText.Compute"/>.</exception>
        private static NDArray NanQuantile1D(string op, NDArray line, double[] q, long[] qShape, bool qIsInteger, string method,
            NPTypeCode qType)
        {
            using var nan = line.TensorEngine.IsNan(line);
            long nans = nan.TensorEngine.CountNonZero(nan);
            if (nans == line.size)
            {
                // Only a NaN-like NA is a NaN, so the line's descriptor has one: np.full stores its Python float nan as the NA.
                var full = new NDArray(line.dtype, qShape.Length == 0 ? Shape.Scalar : Shape.Adopt((long[])qShape.Clone()), true);
                using (var nanLiteral = NDArray.PythonLiteral(PositiveNaN))
                using (var na = NDArray.PackForVStringStore(nanLiteral, line.dtype))
                    np.copyto(full, na, "unsafe");
                // `[()]` of a 0-d array is its Python object (the NA); of an array it is a VIEW of the whole array (NumPy's
                // result does not own its data). The caller's scope reclaims `full`; the view keeps its storage.
                return qShape.Length == 0 ? full.MarkReductionScalar() : full.view();
            }
            // NumPy's _remove_nan_1d order, not a mask selection: the partition that picks the element sorts what is left,
            // and texts equal under NpyString_cmp (the same bytes up to a shared NUL) keep the order they arrive in.
            using var kept = nans == 0 ? null : RemoveNaN1D(line, nan, nans);
            return QuantileText.Compute(op, kept ?? line, q, qShape, qIsInteger, null, null, method, false, qType: qType);
        }

        /// <summary>
        ///     NumPy's <c>_remove_nan_1d</c> over a <c>StringDType</c> line holding NaN-like NAs (not all of them): a copy whose
        ///     first NaN slots are filled, in order, with the non-NaN elements of the last <c>nans</c> slots, cut to its first
        ///     <c>n - nans</c> elements — NOT the NaN-free elements in input order. The order matters wherever ties do: the
        ///     partition that picks a quantile's element sorts this array, and texts equal under <c>NpyString_cmp</c> keep the
        ///     order they arrive in (probed: a <c>np.nanpercentile(..., method='nearest')</c> over tied texts picks by it).
        /// </summary>
        /// <param name="line">The 1-D line.</param>
        /// <param name="nan">Its NaN mask (<c>np.isnan(line)</c>).</param>
        /// <param name="nans">The number of NaNs (1 to <c>line.size - 1</c>).</param>
        /// <returns>A new 1-D array of the line's descriptor (a view of the filled copy, as NumPy's <c>arr1d[:-s.size]</c>).</returns>
        /// <remarks>Scoped by the caller's boundary: the copy, the tail views and the positions are reclaimed there; the
        ///     returned view keeps the copy's storage.</remarks>
        private static NDArray RemoveNaN1D(NDArray line, NDArray nan, long nans)
        {
            long n = line.size;
            var arr = line.copy();
            // enonan = arr1d[-s.size:][~c[-s.size:]]; arr1d[s[:enonan.size]] = enonan; return arr1d[:-s.size]
            using var tailMask = nan[$"{n - nans}:"];
            using var tailKeep = tailMask.TensorEngine.LogicalNot(tailMask);
            using var tail = arr[$"{n - nans}:"];
            using var enonan = tail[tailKeep];
            if (enonan.size > 0)
            {
                using var positions = nan.TensorEngine.FlatNonZero(nan);
                using var head = positions[$":{enonan.size}"];
                arr[head] = enonan;
            }
            return arr[$":{n - nans}"];
        }

        // =============================================================================================================
        // the interpolating quantile methods (called by QuantileText.Compute once the line is moved first)
        // =============================================================================================================

        /// <summary>
        ///     NumPy's interpolating branch of <c>_quantile</c> over a <c>StringDType</c> array whose reduced axis is first:
        ///     the partition (a sort of every line), the neighbours, and <c>_lerp</c> — whose <c>b - a</c> is Python's
        ///     subtraction of two objects for a scalar result and <c>subtract</c>'s refusal for an array one.
        /// </summary>
        /// <param name="arr">The data with the reduced axis first (NumPy's <c>arr</c>; its first axis is not empty: the
        ///     caller reported the empty neighbour lookup).</param>
        /// <param name="q">The quantiles.</param>
        /// <param name="qShape">q's shape.</param>
        /// <param name="method">An interpolating method.</param>
        /// <param name="target">The out (keepdims' view of it), or null.</param>
        /// <param name="out">The caller's out, or null (returned itself).</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="nd">The input's rank.</param>
        /// <param name="reduced">The normalized reduced axes, or null for every axis.</param>
        /// <param name="qType">q's precision (<see cref="QuantileIndexes.PrecisionOf"/>): a float16 / float32 q array computes
        ///     the neighbours, the weight and the lerp in its own dtype, and the result has that dtype.</param>
        /// <returns>The interpolation of two numeric NA objects: <c>np.float64</c> (a read-only 0-d float64, or q's float16 /
        ///     float32 for such a q array; complex128 for a complex NA object), its keepdims view, or <paramref name="out"/>
        ///     itself.</returns>
        /// <exception cref="ValueError">A None-like NA in a sorted line of two or more elements.</exception>
        /// <exception cref="TypeError">The subtraction: Python's <c>unsupported operand type(s) for -: 'str' and 'str'</c> (the
        ///     types of the next and the previous neighbour), or <c>subtract</c>'s no-loop text for an array result; the out's
        ///     casts (<c>Cannot cast ufunc 'add' output from dtype('float64') to dtype('int64') with casting rule
        ///     'same_kind'</c>; a text out passes the add and meets <c>subtract</c>'s refusal of its class).</exception>
        internal static NDArray QuantileInterpolate(NDArray arr, double[] q, long[] qShape, QuantileMethod method,
            NDArray target, NDArray @out, bool keepdims, int nd, int[] reduced, NPTypeCode qType = NPTypeCode.Double)
        {
            long n = arr.Shape.dimensions[0];
            // `arr.partition(np.unique(np.concatenate(([0, -1], previous_indexes.ravel(), next_indexes.ravel()))), axis=0)`:
            // a sort of every line (StringDType has no selection loop); a line of one element compares nothing.
            using var sorted = n > 1 ? arr.TensorEngine.Sort(arr, 0) : null;
            NDArray line = sorted ?? arr;
            // `previous = arr[previous_indexes]`, `next = arr[next_indexes]` are StringDType ARRAYS unless the result is a
            // scalar (a scalar q over a 1-D line): `b - a` is then np.subtract, which has no StringDType loop.
            if (qShape.Length != 0 || arr.ndim != 1)
                throw NDStringUfuncs.NoLoop("subtract", arr.dtype, arr.dtype);
            // _get_indexes (floor / +1, both -1 at or past the last index, both 0 before the first) and _get_gamma (the
            // method's fix_gamma), in q's precision (QuantileIndexes); the partition already refused an index out of range.
            long[] plan = QuantileText.Plan(method, n, q, qType);
            long previous = plan[0], next = plan[1];
            NeighbourWeight(method, n, q[0], qType, out double gamma, out double oneMinusGamma);
            object a = line.Storage.GetAtIndex(previous < 0 ? previous + n : previous);
            object b = line.Storage.GetAtIndex(next < 0 ? next + n : next);
            return PyLerp(a, b, gamma, oneMinusGamma, qType, target, @out, keepdims, nd, reduced);
        }

        /// <summary>
        ///     NumPy's <c>_get_gamma</c> of one quantile in q's precision: the weight t (the method's fix_gamma applied) and
        ///     <c>_lerp</c>'s <c>1 - t</c>, both values of q's dtype (see <see cref="QuantileIndexes.Neighbours{TQ}"/>).
        /// </summary>
        /// <param name="method">The interpolating method.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The quantile.</param>
        /// <param name="qType">q's precision.</param>
        /// <param name="t">The weight.</param>
        /// <param name="oneMinusT"><c>1 - t</c> in q's dtype.</param>
        private static unsafe void NeighbourWeight(QuantileMethod method, long n, double q, NPTypeCode qType, out double t,
            out double oneMinusT)
        {
            long* raw = stackalloc long[2];
            double* weight = stackalloc double[2];
            // StringDType is not an inexact dtype (a NaN virtual index is not mapped to the last element).
            Kernels.DirectILKernelGenerator.QuantilePlan(qType, method, n, &q, 1, false, raw, weight);
            t = weight[0];
            oneMinusT = weight[1];
        }

        /// <summary>
        ///     NumPy's <c>_lerp(a, b, t, out)</c> for two Python objects (a scalar result): <c>diff_b_a = b - a</c> (Python's
        ///     subtraction), <c>add(a, diff_b_a * t, out=out)</c> — the float64 (complex128) loop, its output cast into the out
        ///     under 'same_kind' — then <c>subtract(b, diff_b_a * (1 - t), out=..., where=t &gt;= 0.5, dtype=type(out.dtype))</c>,
        ///     and <c>lerp[()]</c>.
        /// </summary>
        /// <param name="a">The previous neighbour (a <see cref="string"/>, null, or the NA object).</param>
        /// <param name="b">The next neighbour.</param>
        /// <param name="t">The weight (gamma), a value of q's precision.</param>
        /// <param name="oneMinusT"><c>1 - t</c> in q's precision.</param>
        /// <param name="precision">q's precision: under a float16 / float32 q array t is an array of that dtype and the two
        ///     Python numbers are weak, so every operation rounds to it and the result has that dtype; double otherwise.</param>
        /// <param name="target">The out (keepdims' view of it), or null.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="keepdims">Keep the reduced axes.</param>
        /// <param name="nd">The input's rank.</param>
        /// <param name="reduced">The normalized reduced axes, or null.</param>
        /// <returns>The result (see <see cref="QuantileInterpolate"/>).</returns>
        /// <exception cref="TypeError">See <see cref="QuantileInterpolate"/>.</exception>
        /// <remarks>A complex NA object under a float16 / float32 q is complex64 in NumPy (a Python complex meeting the q's
        ///     dtype); NumSharp has one complex dtype, so it stays complex128.</remarks>
        private static NDArray PyLerp(object a, object b, double t, double oneMinusT, NPTypeCode precision, NDArray target,
            NDArray @out, bool keepdims, int nd, int[] reduced)
        {
            // `b - a`: Python's binary subtraction, which names the left operand (b) first.
            if (!TryPyNumber(a, out double ra, out Complex ca, out bool aComplex)
                || !TryPyNumber(b, out double rb, out Complex cb, out bool bComplex))
                throw new TypeError($"unsupported operand type(s) for -: '{PyTypeName(b)}' and '{PyTypeName(a)}'");

            bool complex = aComplex || bComplex;
            double real = 0;
            Complex value = default;
            if (complex)
            {
                Complex za = aComplex ? ca : new Complex(ra, 0), zb = bComplex ? cb : new Complex(rb, 0);
                Complex diff = zb - za;
                value = t >= 0.5 ? zb - diff * (1 - t) : za + diff * t;
            }
            else
            {
                // The same operation order as NumPy's loops: a + diff * t, replaced where t >= 0.5 by b - diff * (1 - t). The
                // Python subtraction `b - a` is a double; each later operation meets t (q's dtype) and rounds to it.
                double diff = rb - ra;
                double tm(double x) => QuantileIndexes.Round(precision, x);
                real = t >= 0.5 ? tm(tm(rb) - tm(tm(diff) * oneMinusT)) : tm(tm(ra) + tm(tm(diff) * t));
            }

            NPTypeCode realType = precision is NPTypeCode.Half or NPTypeCode.Single ? precision : NPTypeCode.Double;
            if (target is null)
            {
                var r = (complex ? NDArray.Scalar(value) : RealScalar(real, realType)).MarkReductionScalar();
                return keepdims ? KeepdimsResult(r, nd, reduced) : r;
            }

            DType loop = complex ? np.complex128 : DType.From(realType);
            using var lerp = complex ? NDArray.Scalar(value) : RealScalar(real, realType);
            if (target.typecode == NPTypeCode.String)
            {
                // A text out: the add's output cast lets the value through (written as its str), and the subtract's dtype=
                // names the out's class, for which no loop takes two Python numbers.
                np.copyto(target, lerp, "unsafe");
                string operand = PyClassOf(b).ToString();
                throw NDStringUfuncs.VStringNoLoop("subtract", new[] { operand, DTypeRegistry.PyFloat.ToString() },
                    target.dtype.Meta.ToString());
            }
            if (!np.can_cast(loop, target.dtype, "same_kind"))
                throw new TypeError($"Cannot cast ufunc 'add' output from {loop.ToString(true)} to {target.dtype.ToString(true)} " +
                                    "with casting rule 'same_kind'");
            np.copyto(target, lerp, "unsafe");
            return @out;
        }

        /// <summary>The lerp's real result as a 0-d array of its dtype (float64, or q's float16 / float32).</summary>
        /// <param name="real">The value (already rounded to <paramref name="realType"/>).</param>
        /// <param name="realType">Double, Half or Single.</param>
        /// <returns>A new 0-d array.</returns>
        private static NDArray RealScalar(double real, NPTypeCode realType)
        {
            if (realType == NPTypeCode.Double)
                return NDArray.Scalar(real);
            // The value is exact in the narrower dtype, so the cast does not round it again.
            using var wide = NDArray.Scalar(real);
            return wide.astype(realType);
        }

        /// <summary>The NEP 50 class of a Python number (the abstract DType a weak Python scalar resolves through).</summary>
        /// <param name="value">The number (an int / bool, a float, or a complex).</param>
        /// <returns><c>_PyLongDType</c>, <c>_PyFloatDType</c> or <c>_PyComplexDType</c>.</returns>
        private static DTypeMeta PyClassOf(object value) => value switch
        {
            Complex => DTypeRegistry.PyComplex,
            double or float => DTypeRegistry.PyFloat,
            _ => DTypeRegistry.PyLong,
        };

        // =============================================================================================================
        // a StringDType q
        // =============================================================================================================

        /// <summary>
        ///     NumPy's answer for the percentile / quantile family when q is a <c>StringDType</c> array, in its statement
        ///     order: percentile divides q by 100 first (no loop takes StringDType and a Python int); quantile validates q
        ///     with <c>_quantile_is_valid</c> — a 1-D q of fewer than 10 elements compared element by element with
        ///     <c>0.0 &lt;=</c> (Python's refusal for a str, False for a NaN NA), any other q through <c>q.min() &gt;= 0 and
        ///     q.max() &lt;= 1</c> (StringDType's reductions: two axes are not reorderable, an empty q has no identity) — and a
        ///     q that passes (an empty one, or numeric NA objects in range) reaches the data and then the virtual index, whose
        ///     arithmetic over a StringDType array refuses per method.
        /// </summary>
        /// <param name="op"><c>quantile</c>, <c>nanquantile</c>, <c>percentile</c> or <c>nanpercentile</c>.</param>
        /// <param name="a">The data (any dtype).</param>
        /// <param name="q">The <c>StringDType</c> q.</param>
        /// <param name="method">The method name.</param>
        /// <param name="axes">The <c>axis</c> argument, or null.</param>
        /// <param name="out">The caller's out, or null: a q that passed reaches <c>_ureduce</c>, whose keepdims index of it
        ///     comes before q's rank, the method and the virtual index.</param>
        /// <param name="keepdims">Whether the reduced axes are kept (keepdims' out index).</param>
        /// <returns>The exception to throw, or null when NumPy returns <c>np.nanmean(a, axis)</c> (nanquantile of EMPTY
        ///     numeric data with a q that passed) — the caller computes that.</returns>
        /// <exception cref="Exception">The reductions' and the data's own errors, thrown where NumPy raises them (the axis,
        ///     keepdims' out index, the merge of several axes, the add.reduce of empty text data, a None-like NA in q's
        ///     minimum).</exception>
        /// <remarks>
        ///     One residual, shared with a fixed-width text q: a nanquantile whose every reduced line of the data is all NaN
        ///     returns NumPy's empty result there, where this reports the virtual index's refusal.
        /// </remarks>
        internal static Exception QuantileVStringQ(string op, NDArray a, NDArray q, string method, int[] axes,
            NDArray @out, bool keepdims)
        {
            if (op is "percentile" or "nanpercentile")
                // `q = np.true_divide(q, 100)`.
                return NDStringUfuncs.VStringNoLoop("divide", new[] { q.dtype.Meta.ToString(), DTypeRegistry.PyLong.ToString() });

            if (q.ndim == 1 && q.size < 10)
            {
                // `for i in range(q.size): if not (0.0 <= q[i] <= 1.0): return False` — a chained comparison: a NaN fails
                // the first test (False), a str or None cannot be compared with a float.
                for (long i = 0; i < q.size; i++)
                {
                    object element = q.GetData(new[] { i }).Storage.GetAtIndex(0);
                    if (!TryPyNumber(element, out double v, out _, out bool isComplex) || isComplex)
                        return new TypeError($"'<=' not supported between instances of 'float' and '{PyTypeName(element)}'");
                    if (!(0.0 <= v && v <= 1.0))
                        return new ValueError("Quantiles must be in the range [0, 1]");
                }
            }
            else
            {
                // `if not (q.min() >= 0 and q.max() <= 1): return False` — StringDType's minimum / maximum reductions.
                using (var min = q.TensorEngine.ReduceAMin(q, null))
                {
                    object m = min.Storage.GetAtIndex(0);
                    if (!TryPyNumber(m, out double v, out _, out bool isComplex) || isComplex)
                        return new TypeError($"'>=' not supported between instances of '{PyTypeName(m)}' and 'int'");
                    if (!(v >= 0))
                        return new ValueError("Quantiles must be in the range [0, 1]");
                }
                using (var max = q.TensorEngine.ReduceAMax(q, null))
                {
                    object m = max.Storage.GetAtIndex(0);
                    if (!TryPyNumber(m, out double v, out _, out bool isComplex) || isComplex)
                        return new TypeError($"'<=' not supported between instances of '{PyTypeName(m)}' and 'int'");
                    if (!(v <= 1))
                        return new ValueError("Quantiles must be in the range [0, 1]");
                }
            }

            // q passed: NumPy reaches the data.
            if (op == "nanquantile")
            {
                if (a.size == 0)
                {
                    if (!NDStringUfuncs.IsText(a))
                        return null;
                    // np.nanmean of empty StringDType text refuses in every form (EmptyNanMean never returns for it); the
                    // disposal only keeps an impossible result from waiting for the finalizer.
                    if (NDStringUfuncs.IsVString(a))
                        EmptyNanMean(a, axes, null, false)?.Dispose();
                    // Fixed-width text: np.nanmean's add.reduce refusal.
                    return NDStringUfuncs.AddNotCompatible(a.dtype, "reduce");
                }
                // _ureduce's prologue (the family's one copy): the axes, keepdims' out index (non-empty data cannot fail the
                // merge).
                DefaultEngine.UReduceChecks(a, axes, @out, keepdims);
                // Fixed-width text has no isnan loop; StringDType text does (its NaN-like NAs), and goes on to the virtual
                // index of what is left. (IsVString first: IsString refuses a StringDType operand by design.)
                if (!NDStringUfuncs.IsVString(a) && NDStringUfuncs.IsString(a))
                    return NDStringUfuncs.NotSupportedForInputTypes("isnan");
            }
            else
            {
                // _ureduce's prologue: the axes, keepdims' out index, the merge of several axes (a kept axis of length 0
                // beside them is NumPy's reshape error) — all before q's rank, the method and the virtual index.
                DefaultEngine.UReduceChecks(a, axes, @out, keepdims);
            }
            // _quantile_ureduce_func: `if q.ndim > 2: raise ValueError("q must be a scalar or 1d")`.
            if (q.ndim > 2)
                return new ValueError("q must be a scalar or 1d");
            QuantileMethod parsed = DefaultEngine.ParseMethod(method);
            long n = ValuesCount(a, axes);
            try
            {
                VirtualIndexOverText(parsed, n, q);
            }
            catch (Exception e)
            {
                return e;
            }
            return new InvalidOperationException("The virtual index of a StringDType q returned.");
        }

        /// <summary>The length of the reduced line NumPy's <c>_quantile</c> reads (<c>values_count</c>).</summary>
        /// <param name="a">The data.</param>
        /// <param name="axes">The axes (validated), or null.</param>
        /// <returns>The product of the reduced dimensions (every dimension for null).</returns>
        private static long ValuesCount(NDArray a, int[] axes)
        {
            if (axes is null)
                return a.size;
            long n = 1;
            foreach (int ax in axes)
                n *= a.Shape.dimensions[ax < 0 ? ax + a.ndim : ax];
            return n;
        }

        /// <summary>
        ///     The method's <c>get_virtual_index(n, quantiles)</c> over a <c>StringDType</c> q, statement for statement with
        ///     the engine's ufunc members (the loops of NumPy's ufuncs, so their StringDType refusals and NA rules answer): <c>np.floor((n - 1) * q)</c> (linear's
        ///     <c>_get_indexes</c>, lower, midpoint), <c>np.ceil</c> (higher), <c>np.around</c> (nearest: <c>rint</c>),
        ///     <c>(n * q) - 1</c> (the inverted-CDF family), <c>n * q + (alpha + q * (1 - alpha - beta)) - 1</c> (the
        ///     continuous methods).
        /// </summary>
        /// <param name="method">The method.</param>
        /// <param name="n">The line's length.</param>
        /// <param name="q">The <c>StringDType</c> q.</param>
        /// <exception cref="Exception">The first refusal (thrown): every path ends in one.</exception>
        private static void VirtualIndexOverText(QuantileMethod method, long n, NDArray q)
        {
            switch (method)
            {
                case QuantileMethod.Linear:
                case QuantileMethod.Lower:
                case QuantileMethod.Midpoint:
                case QuantileMethod.Higher:
                case QuantileMethod.Nearest:
                {
                    using var nm1 = NDArray.PythonLiteral(n - 1);
                    using var product = nm1.TensorEngine.Multiply(nm1, q);
                    using var rounded = method == QuantileMethod.Higher ? product.TensorEngine.Ceil(product)
                        : method == QuantileMethod.Nearest ? product.TensorEngine.Rint(product)
                        : product.TensorEngine.Floor(product);
                    return;
                }
                case QuantileMethod.InvertedCdf:
                case QuantileMethod.AveragedInvertedCdf:
                case QuantileMethod.ClosestObservation:
                {
                    using var count = NDArray.PythonLiteral(n);
                    using var product = count.TensorEngine.Multiply(count, q);
                    using var one = NDArray.PythonLiteral(1L);
                    using var shifted = product.TensorEngine.Subtract(product, one);
                    return;
                }
                default:
                {
                    // _compute_virtual_index: `n * quantiles` first, then `(alpha + quantiles * (1 - alpha - beta))` — the
                    // weight a Python int for interpolated_inverted_cdf (0) and weibull (1), a float otherwise.
                    (object alpha, object weight) = method switch
                    {
                        QuantileMethod.InterpolatedInvertedCdf => ((object)0L, (object)0L),
                        QuantileMethod.Weibull => (0L, 1L),
                        QuantileMethod.Hazen => (0.5, 0.0),
                        QuantileMethod.MedianUnbiased => (1 / 3.0, 1 - 1 / 3.0 - 1 / 3.0),
                        _ => (3 / 8.0, 1 - 3 / 8.0 - 3 / 8.0),
                    };
                    using var count = NDArray.PythonLiteral(n);
                    using var product = count.TensorEngine.Multiply(count, q);
                    using var w = weight is long lw ? NDArray.PythonLiteral(lw) : NDArray.PythonLiteral((double)weight);
                    using var scaled = q.TensorEngine.Multiply(q, w);
                    using var al = alpha is long la ? NDArray.PythonLiteral(la) : NDArray.PythonLiteral((double)alpha);
                    using var shifted = al.TensorEngine.Add(al, scaled);
                    return;
                }
            }
        }

        // =============================================================================================================
        // _ureduce pieces, Python objects and stores
        // =============================================================================================================

        /// <summary>
        ///     <c>_ureduce</c>'s keepdims rule for the result: <c>r[(Ellipsis,) + index_r]</c>, a new axis at every reduced
        ///     position. A NumPy scalar (<c>np.float64</c>) indexes into a writeable view of a fresh 0-d array; a Python object
        ///     cannot be indexed with a tuple (a str: <c>string indices must be integers, not 'tuple'</c>; the NaN NA object:
        ///     <c>'float' object is not subscriptable</c>).
        /// </summary>
        /// <param name="r">The reduction's result.</param>
        /// <param name="nd">The input's rank.</param>
        /// <param name="reduced">The normalized reduced axes, or null.</param>
        /// <returns>The view.</returns>
        /// <exception cref="TypeError">A Python object result (a 0-d <c>StringDType</c> array).</exception>
        private static NDArray KeepdimsResult(NDArray r, int nd, int[] reduced)
        {
            if (r.ndim == 0)
            {
                if (r.typecode == NPTypeCode.String)
                    throw ScalarSubscriptError(r.Storage.GetAtIndex(0));
                // A NumPy scalar's subscript goes through a fresh, writeable 0-d array.
                r = r.copy();
            }
            var expand = new Slice[1 + nd];
            expand[0] = Slice.Ellipsis;
            for (int d = 0; d < nd; d++)
                expand[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.NewAxis : Slice.All;
            return r[expand];
        }

        /// <summary>
        ///     Python's refusal to index an element object with a tuple, as <c>_ureduce</c>'s <c>r[(Ellipsis,) + index_r]</c>
        ///     meets it: a <c>str</c> (<c>string indices must be integers, not 'tuple'</c>), anything else
        ///     <c>'&lt;type&gt;' object is not subscriptable</c> (<c>float</c>, <c>NoneType</c>, ...).
        /// </summary>
        /// <param name="element">The element (a <see cref="string"/>, null, or an NA object).</param>
        /// <returns>The exception (not thrown).</returns>
        internal static TypeError ScalarSubscriptError(object element)
            => element is string or char
                ? new TypeError("string indices must be integers, not 'tuple'")
                : new TypeError($"'{PyTypeName(element)}' object is not subscriptable");

        /// <summary>
        ///     A fresh read-only 0-d <c>StringDType</c> array holding one element of a line: NumPy's <c>arr1d[-1]</c> is the
        ///     element's Python object, never a view of the input.
        /// </summary>
        /// <param name="line">A 1-D <c>StringDType</c> array.</param>
        /// <param name="index">The element's index.</param>
        /// <returns>The 0-d array (owned by the caller).</returns>
        private static NDArray ScalarElement(NDArray line, long index)
        {
            using var view = line.GetData(new[] { index });
            return view.copy().MarkReductionScalar();
        }

        /// <summary>
        ///     <c>np.asanyarray(result)</c> as <c>np.apply_along_axis</c> applies it to each row's result: a Python str becomes
        ///     a 0-d <c>&lt;U</c> array of its length (<c>&lt;U1</c> for the empty string), a float (the NaN NA object) a 0-d
        ///     float64, an int / bool / complex its NumPy array; any array passes as it is.
        /// </summary>
        /// <param name="result">The row's result: a 0-d <c>StringDType</c> array is NumPy's Python object.</param>
        /// <returns>The array the buffer receives.</returns>
        /// <exception cref="NotSupportedException">An object NumPy would wrap in an OBJECT array (<c>None</c>, an integer
        ///     past int64, another object): NumSharp has no object dtype.</exception>
        private static NDArray AsAnyArray(NDArray result)
        {
            if (result.ndim != 0 || result.typecode != NPTypeCode.String)
                return result;
            object element = result.Storage.GetAtIndex(0);
            switch (element)
            {
                case string s:
                    // An unsized U discovers its width from the text (np.asanyarray('') is <U1).
                    return Backends.UnmanagedStorage.FromAnyScalar(s, null);
                case double d:
                    return NDArray.Scalar(PyFloatValue(d));
                case float f:
                    return NDArray.Scalar(PyFloatValue(f));
                case bool b:
                    return NDArray.Scalar(b);
                case long l:
                    return NDArray.Scalar(l);
                case int i:
                    return NDArray.Scalar((long)i);
                case Complex c:
                    return NDArray.Scalar(c);
                case BigInteger big when big >= long.MinValue && big <= long.MaxValue:
                    return NDArray.Scalar((long)big);
            }
            throw new NotSupportedException(
                $"np.apply_along_axis over a row whose result is {NDPyObject.Repr(element)} builds an object array in NumPy, a " +
                "dtype NumSharp does not have.");
        }

        /// <summary>
        ///     NumPy's <c>out[...] = value</c>: a Python object (a read-only 0-d <c>StringDType</c> array's element) stored by
        ///     the destination's setitem — <c>PyArray_Pack</c>: a str parsed by <c>int()</c> / <c>float()</c> (its errors
        ///     quoting the str), the NaN NA object stored as NaN, refused by an integer (<c>cannot convert float NaN to
        ///     integer</c>), stored as the NA or as <c>'nan'</c> by a <c>StringDType</c>, as <c>str()</c> truncated by
        ///     <c>S</c> / <c>U</c> — and any array assigned with NumPy's unsafe cast and broadcast.
        /// </summary>
        /// <param name="target">The destination.</param>
        /// <param name="value">The value.</param>
        /// <exception cref="Exception">The store's errors (NumPy's setitem texts, or the cast's for an array).</exception>
        internal static void PyAssign(NDArray target, NDArray value)
        {
            if (value.ndim != 0 || value.typecode != NPTypeCode.String)
            {
                np.copyto(target, value, "unsafe");
                return;
            }
            object element = value.Storage.GetAtIndex(0);
            using var literal = element switch
            {
                null => null,
                string s => NDArray.PythonStrLiteral(s),
                double d => NDArray.PythonLiteral(PyFloatValue(d)),
                float f => NDArray.PythonLiteral(PyFloatValue(f)),
                bool b => NDArray.PythonBoolLiteral(b),
                long l => NDArray.PythonLiteral(l),
                int i => NDArray.PythonLiteral((long)i),
                Complex c => NDArray.PythonLiteral(c),
                _ => throw new NotSupportedException($"Storing {NDPyObject.Repr(element)} has no NumSharp form."),
            };
            NDArray packed = NDStringUfuncs.IsVString(target)
                ? NDArray.PackForVStringStore(literal, target.dtype)
                : NDArray.PackForStore(literal, target.dtype, true);
            try
            {
                np.copyto(target, packed, "unsafe");
            }
            finally
            {
                if (!ReferenceEquals(packed, literal))
                    packed?.Dispose();
            }
        }

        /// <summary>NumPy's <c>np.nan</c>: the POSITIVE quiet NaN (.NET's <see cref="double.NaN"/> has its sign bit set).</summary>
        private static readonly double PositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);
    }
}
