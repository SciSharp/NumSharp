using System;
using System.Collections.Generic;
using NumSharp.Backends;
using NumSharp.Backends.Strings;

// =============================================================================
// QuantileWeighted.cs — the `weights=` keyword of np.quantile / np.percentile / np.nanquantile / np.nanpercentile
// =============================================================================
//
// WHAT NUMPY DOES (numpy/lib/_function_base_impl.py and numpy/lib/_nanfunctions_impl.py, 2.4.2)
//   * The public functions check complex data, then q (np.asanyarray and the [0, 1] range; percentile divides by 100
//     first), then the weights, in this order:
//       - `method != "inverted_cdf"`: ValueError "Only method 'inverted_cdf' supports weights. Got: {method}." (the name
//         is not looked up, so an unknown name gets the same text);
//       - the axis, normalized with argname "axis" (AxisError "axis: axis 5 is out of bounds for array of dimension 2",
//         ValueError "repeated axis in `axis` argument"), so the texts differ from the unweighted call's;
//       - _weights_are_valid: TypeError "Axis must be specified when shapes of a and weights differ." without an axis,
//         ValueError "Shape of weights must be consistent with shape of a along specified axis." when the weights have
//         neither a's shape nor the reduced axes' shape; weights of the reduced axes' shape are transposed into the axes'
//         order and reshaped with length 1 at every other axis;
//       - `np.any(weights < 0)`: ValueError "Weights must be non-negative." (text weights raise the comparison's own
//         UFuncTypeError).
//   * The nan forms answer empty data with np.nanmean(a, axis, out=out, keepdims=keepdims), after those checks.
//   * _ureduce: with keepdims and out, the reduction writes into out[(Ellipsis,) + index_out] (every reduced axis taken
//     at 0, every axis for axis=None); one reduced axis is the reduction's axis; several (or an empty tuple) are merged
//     into one LAST axis behind the kept ones (np.moveaxis of the kept axes to the front, then a reshape) for the data
//     AND the weights; it returns out itself, or r[(Ellipsis,) + index_r] (a new axis at every reduced position) under
//     keepdims.
//   * _quantile_ureduce_func: a q of rank above 2 raises ValueError "q must be a scalar or 1d"; axis=None flattens the
//     data and the weights.
//   * The weighted branch of _quantile: the axis moved first; `np.argsort(arr, axis=0)` (the default kind); the data
//     gathered in that order (take_along_axis), the weights too (take_along_axis when they have the data's shape, else
//     reshaped to 1-D and fancy-indexed with the argsort); `np.isnan(arr[-1, ...])` for float data (an empty axis raises
//     IndexError "index -1 is out of bounds for axis 0 with size 0" there, or at cdf[-1, ...] for other data);
//     cdf = cumsum(weights, dtype=float64) / its last row; a NaN in the last row raises ValueError "Weights included NaN,
//     inf or were all zero."; cdf is cast to q's dtype when q is a float array (so a float32 q searches float32 cdf
//     values); every zero of cdf becomes -1 when the first row holds one (a leading zero weight never answers q = 0);
//     the result is empty_like(data, shape=q.shape + kept dims), or out (ValueError "Wrong shape of argument 'out',
//     shape=(2,) is required; got shape=(3,)."); per kept position, searchsorted(cdf, q, side="left"), clipped to n - 1,
//     takes the data there into result[(..., position)]; slices whose sorted last element is NaN answer that NaN (a
//     NumPy scalar for a 0-d result without out, else np.copyto with the mask). The result has the data's dtype.
//   * _nanquantile_ureduce_func: axis=None or 1-D data → _nanquantile_1d of the flattened data and weights, then
//     `out[...] = result`; otherwise the data and the weights move their axis LAST, the result is out (NO shape check) or
//     empty_like(data, shape=q.shape + kept dims), and per kept position ii
//     `result[(...,) + ii] = _nanquantile_1d(a[ii], q, weights=weights[ii])`. weights[ii] indexes the VALIDATED weights,
//     whose kept axes have length 1 when the caller gave weights of the reduced axes only, so NumPy 2.4.2 raises its own
//     IndexError there ("index 1 is out of bounds for axis 0 with size 1") — NumPy's answer, reproduced.
//   * _nanquantile_1d: _remove_nan_1d of the data and the same positions of the weights (np.isnan: S / U text refuses
//     with isnan's TypeError); an all-NaN line answers np.full(q.shape, nan, dtype=data.dtype)[()] (no q check); else
//     _quantile_unchecked of what is left (the plain path above, flattened).
//
// HOW THIS FILE GETS THE SAME ANSWERS
//   The statements above, over NumSharp's np.* functions, for every dtype at once (numbers, S / U, StringDType): the
//   weighted branch reads values only through argsort / take / searchsorted / the NaN test, which every dtype has.
//   Arrangement differs in one place: the reduced axis is moved LAST instead of first, so after the gathers every slice is
//   a contiguous row. The answers do not depend on where the axis sits: each line is sorted, gathered, summed and searched
//   on its own, in the same element order.
//   Three routes give NumPy's answers with less work, each replayed over the whole corpus with the other route forced:
//     * one line (1-D data, axis=None, every line of the nan forms) never gathers the whole sorted data (NumPy reads only
//       its last element and the picks, which are the data at the argsort's entries), and when the search compares in
//       float64 it divides only the cumulative sums its bisection probes instead of the whole cdf (SearchLazily; float16 /
//       float32 q, whose cdf NumPy casts first, run NumPy's statements, SearchEagerly);
//     * two or more dimensions run NumPy's statements, and the per-row searchsorted is one vectorized count of the cdf
//       entries below each q over all rows (a cdf row is non-decreasing), or NumPy's loop row by row once rows are long
//       (CountingMaxWork).
//
// KNOWN DIVERGENCE (host-dependent in NumPy itself)
//   The cumulative sum adds the weights of TIED values in the order NumPy's default argsort leaves them. For numbers that
//   order is an artifact of the sort NumPy dispatches (introsort for 8- and 16-bit types; x86-simd-sort for 32- and
//   64-bit types on AVX2 / AVX-512 hosts, introsort elsewhere), so it differs between machines; NumSharp's argsort is
//   stable (ties in input order). When the weights of a tied run add up differently in another order (0.1 + 0.2 + 0.3 is
//   not 0.3 + 0.2 + 0.1), a q that falls between the two roundings of a cdf step picks the neighbouring element. Text is
//   exact: NumPy sorts it with its generic introsort, which NumSharp replays.
// =============================================================================

namespace NumSharp.Statistics
{
    /// <summary>
    ///     NumPy's weighted quantile (the <c>weights=</c> keyword, method <c>inverted_cdf</c> only) for
    ///     <c>np.quantile</c>, <c>np.percentile</c>, <c>np.nanquantile</c> and <c>np.nanpercentile</c> over every dtype:
    ///     the weights checks, <c>_ureduce</c>, the reduction functions and the weighted branch of <c>_quantile</c>,
    ///     statement for statement (see the file header).
    /// </summary>
    internal static class QuantileWeighted
    {
        /// <summary>NumPy's <c>np.nan</c>: the POSITIVE quiet NaN (.NET's <see cref="double.NaN"/> has its sign bit set).</summary>
        private static readonly double PositiveNaN = BitConverter.Int64BitsToDouble(0x7FF8000000000000L);

        /// <summary>
        ///     Test hook: when set (on the current thread), every search of two or more rows takes NumPy's row-by-row route
        ///     (<see cref="SearchRowByRow"/>), which otherwise only work past <see cref="CountingMaxWork"/> reaches, so a
        ///     replay of the whole corpus proves it gives the counting route's answers. Thread-static so a parallel test never
        ///     changes another's route.
        /// </summary>
        [ThreadStatic] internal static bool ForceRowByRowTestHook;

        /// <summary>Test hook: incremented (on the current thread) each time <see cref="SearchRowByRow"/> runs, so a test
        ///     can prove the forced route ran.</summary>
        [ThreadStatic] internal static int RowByRowRuns;

        /// <summary>
        ///     Test hook: when set (on the current thread), one line searches with NumPy's statements
        ///     (<see cref="SearchEagerly"/>) for every q dtype, where a float64 search otherwise divides only the entries it
        ///     probes (<see cref="SearchLazily"/>), so a replay proves the two give the same indices. Thread-static.
        /// </summary>
        [ThreadStatic] internal static bool ForceEagerSearchTestHook;

        /// <summary>Test hook: incremented (on the current thread) each time <see cref="SearchEagerly"/> runs.</summary>
        [ThreadStatic] internal static int EagerSearchRuns;

        /// <summary>
        ///     <c>np.quantile</c> / <c>np.percentile</c> with weights: NumPy's weights checks (after the caller's q checks),
        ///     then <c>_ureduce</c> with <c>_quantile_ureduce_func</c> and the weighted branch of <c>_quantile</c>.
        /// </summary>
        /// <param name="a">The data (complex already refused by the caller; text included).</param>
        /// <param name="q">The quantiles as NumPy's <c>q</c> array after its conversion: a C# number is a 0-d float64 (or
        ///     int64) array, a list a float64 array, an array its own dtype; a percentile already divided by 100 in q's
        ///     dtype (<c>np.true_divide(q, 100)</c>). Its dtype decides the cdf's dtype (a float q) and its shape the
        ///     result's leading axes. Already range-checked by the caller.</param>
        /// <param name="axis">The reduced axes as the caller gave them, or null for every axis.</param>
        /// <param name="out">An optional output array (NumPy's rules: see the file header).</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="method">The method name; only <c>inverted_cdf</c> supports weights.</param>
        /// <param name="weights">The weights (not null): <paramref name="a"/>'s shape, or the reduced axes' shape.</param>
        /// <returns>The quantiles in the data's dtype; <paramref name="out"/> itself when one is given; a 0-d ARRAY (NumPy's
        ///     weighted branch does not unpack it), except for a NaN slice of a 0-d result, which is NumPy's scalar.</returns>
        /// <exception cref="ValueError">Another method; a repeated axis (<c>repeated axis in `axis` argument</c>); weights of a
        ///     wrong shape; a negative weight; a q of rank above 2; weights that sum to NaN, inf or zero; an out of the wrong
        ///     shape.</exception>
        /// <exception cref="AxisError">An axis out of range (<c>axis: axis 5 is out of bounds for array of dimension
        ///     2</c>).</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis; text weights (the comparison's
        ///     refusal; chararray weights: <c>comparison of non-string arrays</c>).</exception>
        /// <exception cref="IndexError">An empty reduced axis; an out of too few dimensions under keepdims.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0 (NumPy's reshape
        ///     ValueError text).</exception>
        /// <exception cref="Exception">The errors of the sort (a None-like NA in a <c>StringDType</c> line of two or more
        ///     elements) and of the store into <paramref name="out"/> (NumPy's setitem texts).</exception>
        [NDScoped]
        internal static NDArray Compute(NDArray a, NDArray q, int[] axis, NDArray @out, bool keepdims, string method,
            NDArray weights)
        {
            // Plain-array semantics for everything this computation calls: the public functions decided the subclass rules
            // already (a chararray `a` came through NDCharArrayBoundary.Kept), and the statements below are NumPy's own,
            // which meet the operands as plain arrays (the weights' comparison, the one exception, is in Prologue).
            using var plain = NDMatrixBoundary.PlainFor(a, weights, @out);
            NDArray wgt = Prologue(a, axis, method, weights, out int[] reduced);
            return UReduce(a, q, reduced, @out, keepdims, wgt, nan: false);
        }

        /// <summary>
        ///     <c>np.nanquantile</c> / <c>np.nanpercentile</c> with weights: NumPy's weights checks, the empty-data answer of
        ///     <c>_nanquantile_unchecked</c>, then <c>_ureduce</c> with <c>_nanquantile_ureduce_func</c>.
        /// </summary>
        /// <typeparam name="TState">The arguments of the caller's unweighted call, packed by value (a tuple).</typeparam>
        /// <param name="a">The data.</param>
        /// <param name="q">NumPy's q array (see <see cref="Compute(NDArray, NDArray, int[], NDArray, bool, string, NDArray)"/>).</param>
        /// <param name="axis">The reduced axes as the caller gave them, or null.</param>
        /// <param name="out">An optional output array.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="method">The method name; only <c>inverted_cdf</c> supports weights.</param>
        /// <param name="weights">The weights (not null).</param>
        /// <param name="state">The arguments <paramref name="emptyNanAnswer"/> needs.</param>
        /// <param name="emptyNanAnswer">The caller's unweighted call (a <c>static</c> lambda): for empty data it answers
        ///     <c>np.nanmean(a, axis, out, keepdims)</c> before it reads q or the method, which is NumPy's answer here
        ///     too, after the weights checks.</param>
        /// <returns>The quantiles of each line's non-NaN values (see the plain overload); for empty data, np.nanmean's
        ///     answer.</returns>
        /// <exception cref="TypeError">The plain overload's, and <c>S</c> / <c>U</c> data (<c>isnan</c>'s refusal).</exception>
        /// <exception cref="IndexError">The plain overload's, and weights of the reduced axes only over data with a kept
        ///     axis longer than 1 (NumPy 2.4.2's own <c>weights[ii]</c> error), or an out too small for the slice
        ///     positions (NumPy stores per position without a shape check).</exception>
        /// <exception cref="Exception">The plain overload's other errors and the store's.</exception>
        [NDScoped]
        internal static NDArray Compute<TState>(NDArray a, NDArray q, int[] axis, NDArray @out, bool keepdims,
            string method, NDArray weights, TState state, Func<TState, NDArray> emptyNanAnswer)
        {
            // Plain-array semantics (see the plain overload). A chararray `a` reaches the nan forms with its class, and
            // NumPy's answer for it is the plain text's: isnan refuses text, and empty text gets np.nanmean's refusal (the
            // unweighted call, which computes plain inside this region too).
            using var plain = NDMatrixBoundary.PlainFor(a, weights, @out);
            NDArray wgt = Prologue(a, axis, method, weights, out int[] reduced);
            // _nanquantile_unchecked: `if a.size == 0: return np.nanmean(a, axis, out=out, keepdims=keepdims)` — the
            // caller's unweighted route gives exactly that answer before it reads q or the method.
            if (a.size == 0)
                return emptyNanAnswer(state);
            return UReduce(a, q, reduced, @out, keepdims, wgt, nan: true);
        }

        /// <summary>
        ///     The weights block of NumPy's quantile functions, in its order: the method, the axis (normalized with argname
        ///     <c>axis</c>), <c>_weights_are_valid</c>, and <c>np.any(weights &lt; 0)</c>.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="axis">The reduced axes as the caller gave them, or null.</param>
        /// <param name="method">The method name.</param>
        /// <param name="weights">The weights (not null).</param>
        /// <param name="reduced">Receives the normalized reduced axes (the caller's order), or null.</param>
        /// <returns>The validated weights (the caller's array, or a reshaped view of it).</returns>
        /// <exception cref="ValueError">Another method, a repeated axis, weights of a wrong shape, a negative weight.</exception>
        /// <exception cref="AxisError">An axis out of range.</exception>
        /// <exception cref="TypeError">Weights of another shape without an axis; text weights (the comparison's
        ///     refusal; a chararray's own comparison refuses with <c>comparison of non-string arrays</c>).</exception>
        private static NDArray Prologue(NDArray a, int[] axis, string method, NDArray weights, out int[] reduced)
        {
            // `if method != "inverted_cdf"` — compared, not looked up: an unknown name gets this text too. Python's f-string
            // prints None for a missing name.
            if (method != "inverted_cdf")
                throw new ValueError($"Only method 'inverted_cdf' supports weights. Got: {method ?? "None"}.");
            // `axis = normalize_axis_tuple(axis, a.ndim, argname="axis")`: the argument name prefixes NumPy's texts here,
            // unlike the unweighted call, where only _ureduce normalizes the axis.
            reduced = axis is null ? null : DefaultEngine.normalize_axis_tuple(axis, a.ndim, "axis");
            NDArray wgt = WeightsAreValid(weights, a, reduced);
            // `if np.any(weights < 0)` compares the weights as the caller gave them (`np.asanyarray` keeps a subclass): a
            // chararray's own `<` is compare_chararrays, which refuses a number whatever the text holds (probed 2.4.2).
            if (NDCharArrayBoundary.IsCharArray(weights))
                throw new TypeError("comparison of non-string arrays");
            // Otherwise the comparison with a Python int, so text weights raise the comparison's refusal naming
            // _PyLongDType.
            using (var zero = NDArray.PythonLiteral(0L))
            using (var negative = np.less(wgt, zero))
            {
                if (np.any(negative))
                    throw new ValueError("Weights must be non-negative.");
            }
            return wgt;
        }

        /// <summary>
        ///     NumPy's <c>_weights_are_valid</c>: weights of the data's shape pass as they are; otherwise the reduced axes must
        ///     be given and the weights must have their shape, and they are transposed into the axes' order and reshaped with
        ///     length 1 at every other axis (so they broadcast against the data).
        /// </summary>
        /// <param name="weights">The caller's weights (never modified).</param>
        /// <param name="a">The data.</param>
        /// <param name="axis">The normalized reduced axes, in the caller's order, or null.</param>
        /// <returns>The weights, or a reshaped view (or copy) of them.</returns>
        /// <exception cref="TypeError">The shapes differ and no axis was given.</exception>
        /// <exception cref="ValueError">The weights have neither the data's shape nor the reduced axes' shape.</exception>
        private static NDArray WeightsAreValid(NDArray weights, NDArray a, int[] axis)
        {
            if (SameDims(a.Shape.dimensions, weights.Shape.dimensions))
                return weights;
            if (axis is null)
                throw new TypeError("Axis must be specified when shapes of a and weights differ.");
            // `if wgt.shape != tuple(a.shape[ax] for ax in axis)`
            long[] expected = new long[axis.Length];
            for (int i = 0; i < axis.Length; i++)
                expected[i] = a.Shape.dimensions[axis[i]];
            if (!SameDims(expected, weights.Shape.dimensions))
                throw new ValueError("Shape of weights must be consistent with shape of a along specified axis.");
            // `wgt = wgt.transpose(np.argsort(axis))`: the weights' axes reordered into the data's axis order (the axes are
            // distinct after normalization, so the argsort is a plain ordering).
            var order = new int[axis.Length];
            for (int i = 0; i < order.Length; i++)
                order[i] = i;
            Array.Sort((int[])axis.Clone(), order);
            NDArray wgt = np.transpose(weights, order);
            // `wgt.reshape(tuple((s if ax in axis else 1) for ax, s in enumerate(a.shape)))`
            int nd = a.ndim;
            var shape = new long[nd];
            for (int d = 0; d < nd; d++)
                shape[d] = Array.IndexOf(axis, d) >= 0 ? a.Shape.dimensions[d] : 1L;
            return wgt.reshape(Shape.Adopt(shape));
        }

        /// <summary>
        ///     NumPy's <c>_ureduce</c> for the weighted call: the keepdims view of <paramref name="out"/>, the merge of several
        ///     reduced axes (data and weights alike), the reduction function, and the keepdims result.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="reduced">The normalized reduced axes, or null for every axis.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="keepdims">Whether the reduced axes are kept with length one.</param>
        /// <param name="wgt">The validated weights.</param>
        /// <param name="nan">True for the nan forms.</param>
        /// <returns><paramref name="out"/> itself, or the result (with a new axis at every reduced position under
        ///     keepdims).</returns>
        /// <exception cref="Exception">The reduction's errors (see <see cref="Compute"/>); the keepdims index of an out of too
        ///     few dimensions; the subscript of a Python object result under keepdims (<c>'float' object is not
        ///     subscriptable</c> for the NaN NA object of an all-NaN <c>StringDType</c> slice).</exception>
        /// <remarks>
        ///     The prologue is the quantile family's one copy (<see cref="QuantileEngine.UReduce"/>, since the np-layer dtype
        ///     plan's P1a step 2); this class had its own until then, with the same order and texts. The merged axis reaches
        ///     the reduction as a non-negative axis (NumPy passes <c>-1</c>): the functions below normalize either form.
        /// </remarks>
        [NDScoped]
        private static NDArray UReduce(NDArray a, NDArray q, int[] reduced, NDArray @out, bool keepdims, NDArray wgt,
            bool nan)
        {
            int nd = a.ndim;
            // `kwargs['out'] = out[(Ellipsis,) + index_out]` under keepdims (every reduced axis taken at 0, every axis without
            // an axis argument), then for several axes (or an empty tuple) the kept axes first and the reduced ones merged
            // into one last axis, the data and the weights alike (`reshape_arr`); the reduction runs along it. The axes were
            // normalized by Prologue already (with argname "axis"), so normalizing them again here cannot raise.
            UReduced prologue = QuantileEngine.UReduce(a, reduced, @out, keepdims, wgt);
            (NDArray r, bool pyObject) = nan
                ? NanUreduceFunc(prologue.Data, q, prologue.Weights, prologue.Axis, prologue.Target)
                : (UreduceFunc(prologue.Data, q, prologue.Weights, prologue.Axis, prologue.Target), false);
            if (@out is not null)
                return @out;
            return keepdims ? KeepdimsResult(r, pyObject, nd, reduced) : r;
        }

        /// <summary>
        ///     NumPy's <c>_quantile_ureduce_func</c> for the weighted call: q's rank check, then the flattened data and weights
        ///     (axis=None) or the data as it is, into the weighted branch of <c>_quantile</c>.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="wgt">The weights (validated, merged like the data).</param>
        /// <param name="axis">The reduction's axis (non-negative: the merged last axis arrives as its index; a negative value
        ///     would count from the end), or null for the flattened data.</param>
        /// <param name="out">The out (or its keepdims view), or null.</param>
        /// <returns>The result (see <see cref="WeightedQuantile"/>).</returns>
        /// <exception cref="ValueError">A q of rank above 2.</exception>
        /// <exception cref="Exception">The weighted branch's errors.</exception>
        [NDScoped]
        private static NDArray UreduceFunc(NDArray a, NDArray q, NDArray wgt, int? axis, NDArray @out)
        {
            if (q.ndim > 2)
                throw new ValueError("q must be a scalar or 1d");
            // `arr = a.flatten(); wgt = weights.flatten()` for axis=None (the copy NumPy makes is unobservable here: the
            // weighted branch only reads the data, so a raveled view serves), else the data itself.
            if (axis is null)
                return WeightedQuantile(np.ravel(a), q, 0, @out, np.ravel(wgt));
            return WeightedQuantile(a, q, axis.Value, @out, wgt);
        }

        /// <summary>
        ///     The weighted branch of NumPy's <c>_quantile</c> (see the file header), with the axis moved last. One line
        ///     (1-D data) continues in <see cref="FlatWeightedQuantile"/>; two or more dimensions follow NumPy's statements
        ///     here, every kept position a row.
        /// </summary>
        /// <param name="arr">The data (at least one dimension).</param>
        /// <param name="q">NumPy's q array (rank 0 to 2).</param>
        /// <param name="axis">The reduced axis (non-negative, or negative counting from the end).</param>
        /// <param name="out">The out (or its keepdims view), or null.</param>
        /// <param name="weights">The weights: the data's shape, or length 1 at every axis but <paramref name="axis"/>.</param>
        /// <returns>The result in the data's dtype (a fresh array, or <paramref name="out"/>); NumPy's scalar (a read-only
        ///     0-d array) for the NaN slice of a 0-d result without out.</returns>
        /// <exception cref="IndexError">An empty axis (<c>index -1 is out of bounds for axis 0 with size 0</c>).</exception>
        /// <exception cref="ValueError">Weights that sum to NaN, inf or zero in some slice; an out of the wrong
        ///     shape.</exception>
        /// <exception cref="Exception">The sort's errors and the store's (NumPy's setitem texts for an out of another
        ///     dtype; np.copyto's same_kind refusal for a NaN slice into an integer out).</exception>
        [NDScoped]
        private static NDArray WeightedQuantile(NDArray arr, NDArray q, int axis, NDArray @out, NDArray weights)
        {
            int nd = arr.ndim;
            int ax = axis < 0 ? axis + nd : axis;
            long n = arr.Shape.dimensions[ax];
            // The reduced axis LAST (NumPy moves it first; see the file header): after the gathers every slice is a row.
            if (ax != nd - 1)
            {
                arr = np.moveaxis(arr, ax, -1);
                weights = np.moveaxis(weights, ax, -1);
            }
            bool supportsNans = QuantileEngine.IsFloatTc(arr.typecode);
            // `index_array = np.argsort(arr, axis=0)`: the default kind (see the file header for the order of ties).
            NDArray order = np.argsort(arr, -1);
            // One line (1-D data, the flattened data of axis=None, every line of the nan forms) takes the route that reads
            // the sorted data only where NumPy's statements read it.
            if (nd == 1)
                return FlatWeightedQuantile(arr, q, @out, weights, order, n, supportsNans);
            arr = np.take_along_axis(arr, order, -1);
            // Weights of the data's shape are gathered the same way; weights with length 1 at the kept axes are one line,
            // `weights.reshape(-1)[index_array, ...]`, gathered by the whole argsort.
            weights = SameDims(weights.Shape.dimensions, arr.Shape.dimensions)
                ? np.take_along_axis(weights, order, -1)
                : np.ravel(weights)[order];
            // `arr[-1, ...]` (float data) or `cdf[-1, ...]`: both meet an empty axis with the same IndexError.
            if (n == 0)
                throw new IndexError("index -1 is out of bounds for axis 0 with size 0");
            NDArray last = arr[Slice.Ellipsis, Slice.Index(-1)];
            NDArray slicesHavingNans = supportsNans ? np.isnan(last) : null;

            // `cdf = weights.cumsum(axis=0, dtype=np.float64); cdf /= cdf[-1, ...]`: each line divided by its own total
            // (the total read before the division, as NumPy's overlap copy reads it).
            NDArray cdf = np.cumsum(weights, -1, np.float64);
            NDArray total = cdf[Slice.Ellipsis, Slice.Index(-1), Slice.NewAxis].copy();
            np.true_divide(cdf, total, @out: cdf);
            using (var totals = cdf[Slice.Ellipsis, Slice.Index(-1)])
            using (var totalNaN = np.isnan(totals))
            {
                if (np.any(totalNaN))
                    throw new ValueError("Weights included NaN, inf or were all zero.");
            }
            // `if quantiles.dtype.kind == "f": cdf = cdf.astype(quantiles.dtype)` — a float32 q searches float32 values.
            if (QuantileEngine.IsFloatTc(q.typecode) && q.typecode != NPTypeCode.Double)
                cdf = cdf.astype(q.dtype);
            // `if np.any(cdf[0, ...] == 0): cdf[cdf == 0] = -1`: leading zero weights never answer q = 0.
            using (var first = cdf[Slice.Ellipsis, Slice.Index(0)])
            using (var zero = NDArray.PythonLiteral(0L))
            using (var firstZero = np.equal(first, zero))
            {
                if (np.any(firstZero))
                {
                    using var zeros = np.equal(cdf, zero);
                    using var minusOne = NDArray.PythonLiteral(-1L);
                    np.putmask(cdf, zeros, minusOne);
                }
            }

            // `r_shape = q.shape + arr.shape[1:]` (the kept axes are the leading ones here).
            int kept = nd - 1;
            var rShape = new long[q.ndim + kept];
            for (int i = 0; i < q.ndim; i++)
                rShape[i] = q.Shape.dimensions[i];
            for (int i = 0; i < kept; i++)
                rShape[q.ndim + i] = arr.Shape.dimensions[i];
            if (@out is not null && !SameDims(@out.Shape.dimensions, rShape))
                throw new ValueError($"Wrong shape of argument 'out', shape={PyTuple(rShape)} is required; " +
                                     $"got shape={PyTuple(@out.Shape.dimensions)}.");

            using var clip = NDArray.PythonLiteral(n - 1);
            // Two or more dimensions here (one line takes FlatWeightedQuantile): the kept positions are the rows.
            long rows = 1;
            for (int i = 0; i < kept; i++)
                rows *= arr.Shape.dimensions[i];
            NDArray result;
            using (var arrRows = arr.reshape(Shape.Adopt(new[] { rows, n })))
            using (var cdfRows = cdf.reshape(Shape.Adopt(new[] { rows, n })))
            {
                result = q.size * n <= CountingMaxWork && !ForceRowByRowTestHook
                    ? SearchByCounting(arr, arrRows, cdfRows, q, rows, rShape, clip, @out)
                    : SearchRowByRow(arr, arrRows, cdfRows, q, rows, kept, rShape, clip, @out);
            }

            // `if np.any(slices_having_nans)`: a NaN slice answers its sorted last element (the NaN).
            if (slicesHavingNans is not null && np.any(slicesHavingNans))
            {
                if (result.ndim == 0 && @out is null)
                    // `result = arr[-1]`: NumPy's scalar (read-only here), not the 0-d array.
                    return last.copy().MarkReductionScalar();
                np.copyto(result, last, where: slicesHavingNans);
            }
            return result;
        }

        /// <summary>
        ///     The weighted branch of <c>_quantile</c> for one line, with NumPy's answers but without two of its passes over
        ///     the line, because its statements read their results only in a few places:
        ///     <list type="bullet">
        ///       <item>NumPy gathers the whole sorted data (<c>arr = np.take_along_axis(arr, index_array, axis=0)</c>) and then
        ///             reads only its last element (the NaN test and fill) and the picks of <c>take(arr, indices)</c>. Sorted
        ///             position <c>i</c> holds <c>arr[order[i]]</c>, so those elements are gathered through the argsort
        ///             instead.</item>
        ///       <item>NumPy divides every cumulative sum by the total (<c>cdf /= cdf[-1]</c>) and binary-searches the result.
        ///             When the search compares in float64, the division moves into the search
        ///             (<see cref="SearchLazily"/>), which divides only the sums it probes; a float16 / float32 q (NumPy casts
        ///             the cdf to q's dtype) and NumSharp's other q dtypes run NumPy's statements
        ///             (<see cref="SearchEagerly"/>).</item>
        ///     </list>
        ///     At 1M float64 values the two passes took about 4 ms of the call (2.7 ms the gather, 1.1 ms the division); the
        ///     argsort, which every route needs, takes 39 ms (measured).
        /// </summary>
        /// <param name="arr">The data (1-D, any layout).</param>
        /// <param name="q">NumPy's q array (rank 0 to 2).</param>
        /// <param name="out">The out (or its keepdims view), or null.</param>
        /// <param name="weights">The validated weights (1-D, the data's length).</param>
        /// <param name="order">The data's argsort (1-D int64).</param>
        /// <param name="n">The data's length.</param>
        /// <param name="supportsNans">Whether the data is a float dtype (a NaN sorts last and answers its slice).</param>
        /// <returns>See <see cref="WeightedQuantile"/>.</returns>
        /// <exception cref="IndexError">Empty data (<c>index -1 is out of bounds for axis 0 with size 0</c>).</exception>
        /// <exception cref="ValueError">Weights that sum to NaN, inf or zero; an out of the wrong shape.</exception>
        /// <exception cref="Exception">The store's errors (NumPy's setitem texts for an out of another dtype; np.copyto's
        ///     same_kind refusal for a NaN slice into an integer out).</exception>
        [NDScoped]
        private static NDArray FlatWeightedQuantile(NDArray arr, NDArray q, NDArray @out, NDArray weights, NDArray order,
            long n, bool supportsNans)
        {
            // `weights = np.take_along_axis(weights, index_array, axis=0)` (one line of the data's length: a plain take).
            NDArray sortedWeights = np.take(weights, order);
            // `arr[-1, ...]` (float data) or `cdf[-1, ...]`: both meet an empty line with the same IndexError.
            if (n == 0)
                throw new IndexError("index -1 is out of bounds for axis 0 with size 0");
            // `slices_having_nans = np.isnan(arr[-1, ...])`: the sorted last element is the data at the argsort's last
            // entry (a NaN sorts last, so the line holds one exactly when that element is one).
            NDArray last = null, slicesHavingNans = null;
            if (supportsNans)
            {
                using var lastEntry = order[Slice.Index(n - 1)];
                last = np.take(arr, lastEntry);
                slicesHavingNans = np.isnan(last);
            }

            // `cdf = weights.cumsum(axis=0, dtype=np.float64)`; `cdf /= cdf[-1, ...]` makes the last entry total / total,
            // which `np.isnan(cdf[-1]).any()` finds NaN exactly when the total is zero, infinite or NaN (weights are not
            // negative here, and a finite positive total divided by itself is exactly 1).
            NDArray cumulative = np.cumsum(sortedWeights, -1, np.float64);
            double total = cumulative.GetDouble(n - 1);
            if (double.IsNaN(total / total))
                throw new ValueError("Weights included NaN, inf or were all zero.");

            // `r_shape = q.shape` for one line.
            if (@out is not null && !SameDims(@out.Shape.dimensions, q.Shape.dimensions))
                throw new ValueError($"Wrong shape of argument 'out', shape={PyTuple(q.Shape.dimensions)} is required; " +
                                     $"got shape={PyTuple(@out.Shape.dimensions)}.");

            // `indices = minimum(searchsorted(cdf, quantiles, side="left"), values_count - 1)`
            NDArray indices = SearchesInFloat64(q.typecode) && !ForceEagerSearchTestHook
                ? SearchLazily(cumulative, total, q, n)
                : SearchEagerly(cumulative, q, n);
            // `take(arr, indices, axis=0)` of the sorted data: sorted position i holds arr[order[i]]. Without out, the taken
            // array IS the result NumPy builds by `empty_like(arr, shape=q.shape)` and `result[...] = taken` (same dtype,
            // values and fresh C layout).
            NDArray picks = np.take(order, indices);
            NDArray taken = np.take(arr, picks, 0);
            NDArray result;
            if (@out is null)
                result = taken;
            else
            {
                @out[Slice.Ellipsis] = taken;
                taken.Dispose();
                result = @out;
            }

            // `if np.any(slices_having_nans)`: a NaN line answers its sorted last element (the NaN).
            if (slicesHavingNans is not null && np.any(slicesHavingNans))
            {
                if (result.ndim == 0 && @out is null)
                    // `result = arr[-1]`: NumPy's scalar (read-only here), not the 0-d array; `last` is a fresh 0-d take.
                    return last.MarkReductionScalar();
                np.copyto(result, last, where: slicesHavingNans);
            }
            return result;
        }

        /// <summary>
        ///     Whether NumPy's <c>searchsorted(cdf, q)</c> compares in float64 for a q of this dtype: the cdf stays float64
        ///     (it is cast to q's dtype only for a float q) and its common dtype with a bool, an integer or a float64 q is
        ///     float64. A float16 / float32 q and NumSharp's dtypes without a NumPy counterpart (decimal, char, complex) are
        ///     not included: <see cref="SearchEagerly"/> runs NumPy's statements for them.
        /// </summary>
        /// <param name="tc">q's dtype.</param>
        /// <returns>True when <see cref="SearchLazily"/> gives NumPy's indices.</returns>
        private static bool SearchesInFloat64(NPTypeCode tc) => tc switch
        {
            NPTypeCode.Boolean or NPTypeCode.SByte or NPTypeCode.Byte or NPTypeCode.Int16 or NPTypeCode.UInt16
                or NPTypeCode.Int32 or NPTypeCode.UInt32 or NPTypeCode.Int64 or NPTypeCode.UInt64
                or NPTypeCode.Double => true,
            _ => false,
        };

        /// <summary>
        ///     NumPy's search of a float64 cdf with the normalization done at the probes. The cdf entry a
        ///     <c>searchsorted(cdf, q, side="left")</c> compares is <c>cumulative[i] / total</c>, the IEEE division NumPy's
        ///     in-place <c>cdf /= cdf[-1]</c> runs on every entry; dividing only the entries a binary search probes counts the
        ///     same entries below each q value (the entries are non-decreasing: the sums are, and dividing by a positive total
        ///     keeps their order). NumPy's <c>cdf[cdf == 0] = -1</c> (when the first entry is zero) is applied to each probed
        ///     entry, which keeps the order too, since the zeros are the leading entries.
        /// </summary>
        /// <param name="cumulative">The cumulative sums of the sorted weights (1-D float64, fresh and contiguous).</param>
        /// <param name="total">The last cumulative sum (finite and positive: the caller checked it).</param>
        /// <param name="q">NumPy's q array (a dtype <see cref="SearchesInFloat64"/> accepts; already range-checked).</param>
        /// <param name="n">The line's length (at least 1).</param>
        /// <returns>The int64 indices, q's shape, already clipped to <c>n - 1</c> (NumPy's <c>minimum</c>).</returns>
        /// <remarks>A scalar loop over q's values, each a binary search of about log2(n) probes, as NumPy's own
        ///     <c>binsearch</c>; no pass over the line. Scoped: the float64 copy of q is reclaimed (q itself, when it is
        ///     already a C-contiguous float64 array, is the caller's and is left alone).</remarks>
        [NDScoped]
        private static unsafe NDArray SearchLazily(NDArray cumulative, double total, NDArray q, long n)
        {
            // q's values in C order as float64: exact for every dtype this route takes (bool; the integers, 0 or 1 after the
            // range check; float64), and the dtype NumPy's searchsorted compares the float64 cdf with them in.
            NDArray values = np.asarray(q, np.float64, 'C');
            NDArray indices = new NDArray(NPTypeCode.Int64,
                q.ndim == 0 ? Shape.Scalar : Shape.Adopt((long[])q.Shape.dimensions.Clone()), false);
            double* sums = (double*)cumulative.Address + cumulative.Shape.offset;
            double* v = (double*)values.Address + values.Shape.offset;
            long* found = (long*)indices.Address;
            // `if np.any(cdf[0, ...] == 0)`: the first entry decides whether zero entries count as -1 (a -0.0 sum too).
            bool zerosAreMinusOne = sums[0] / total == 0;
            long m = q.size;
            for (long j = 0; j < m; j++)
            {
                double target = v[j];
                // `side="left"`: the number of entries below the target, by bisection over the non-decreasing entries.
                long lo = 0, hi = n;
                while (lo < hi)
                {
                    long mid = lo + ((hi - lo) >> 1);
                    double entry = sums[mid] / total;
                    if (zerosAreMinusOne && entry == 0)
                        entry = -1;
                    if (entry < target)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                // `indices = minimum(indices, values_count - 1)`: q = 1 can find len(arr).
                found[j] = lo < n - 1 ? lo : n - 1;
            }
            return indices;
        }

        /// <summary>
        ///     NumPy's statements for the search when it does not compare in float64: <c>cdf /= cdf[-1]</c>, the cast to a
        ///     float16 / float32 q's dtype (<c>cdf.astype(quantiles.dtype)</c>, so the search compares the rounded entries),
        ///     <c>cdf[cdf == 0] = -1</c> when the first entry is zero, <c>searchsorted(cdf, q, side="left")</c> and the clip
        ///     to <c>n - 1</c>.
        /// </summary>
        /// <param name="cumulative">The cumulative sums of the sorted weights (1-D float64, fresh; divided in place).</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="n">The line's length (at least 1).</param>
        /// <returns>The int64 indices, q's shape.</returns>
        /// <exception cref="Exception">np.searchsorted's errors for a q dtype it cannot compare with the cdf.</exception>
        [NDScoped]
        private static NDArray SearchEagerly(NDArray cumulative, NDArray q, long n)
        {
            EagerSearchRuns++;
            // `cdf /= cdf[-1, ...]`: the total read before the division, as NumPy's overlap copy reads it.
            NDArray total = cumulative[Slice.Index(n - 1)].copy();
            np.true_divide(cumulative, total, @out: cumulative);
            NDArray cdf = cumulative;
            // `if quantiles.dtype.kind == "f": cdf = cdf.astype(quantiles.dtype)` — a float32 q searches float32 values.
            if (QuantileEngine.IsFloatTc(q.typecode) && q.typecode != NPTypeCode.Double)
                cdf = cdf.astype(q.dtype);
            // `if np.any(cdf[0, ...] == 0): cdf[cdf == 0] = -1`: leading zero weights never answer q = 0.
            NDArray zero = NDArray.PythonLiteral(0L);
            if (np.equal(cdf[Slice.Index(0)], zero).GetBoolean())
                np.putmask(cdf, np.equal(cdf, zero), NDArray.PythonLiteral(-1L));
            NDArray found = np.searchsorted(cdf, q, "left");
            return np.minimum(found, NDArray.PythonLiteral(n - 1));
        }

        /// <summary>
        ///     The work (q values times line length) below which the per-row search runs as one vectorized count over all rows
        ///     (<see cref="SearchByCounting"/>) instead of NumPy's loop of one <c>searchsorted</c> per row
        ///     (<see cref="SearchRowByRow"/>). A row's loop iteration costs several microseconds of calls (a view, the search,
        ///     the clip, the take, the store), the count about a third of a nanosecond per compared element, so the count wins
        ///     until a row is long enough for its sort to dwarf the loop anyway (measured: 10,000 rows of 10 elements ran
        ///     0.64x NumPy row by row).
        /// </summary>
        private const long CountingMaxWork = 32_768;

        /// <summary>
        ///     NumPy's per-row search, row by row: <c>searchsorted(cdf_row, q, side="left")</c>, clipped to <c>n - 1</c>, taken
        ///     from the sorted row and stored into <c>result[(..., position)]</c> in C order of the kept positions (NumPy's
        ///     <c>np.ndindex</c> order, which decides which store fails first).
        /// </summary>
        /// <param name="arr">The sorted data, the reduced axis last (its kept dimensions give the positions).</param>
        /// <param name="arrRows">The sorted data as <c>(rows, n)</c>.</param>
        /// <param name="cdfRows">The normalized cumulative weights as <c>(rows, n)</c>, in q's float dtype.</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="rows">The number of rows (the product of the kept dimensions).</param>
        /// <param name="kept">The number of kept dimensions.</param>
        /// <param name="rShape">The result's shape: q's dimensions, then the kept ones.</param>
        /// <param name="clip">A Python int <c>n - 1</c>.</param>
        /// <param name="out">The out (already shape-checked), or null for a fresh result of the data's dtype.</param>
        /// <returns>The result.</returns>
        /// <exception cref="Exception">The store's errors (NumPy's setitem texts for an out of another dtype).</exception>
        private static NDArray SearchRowByRow(NDArray arr, NDArray arrRows, NDArray cdfRows, NDArray q, long rows, int kept,
            long[] rShape, NDArray clip, NDArray @out)
        {
            RowByRowRuns++;
            NDArray result = @out ?? np.empty_like(arr, shape: Shape.Adopt(rShape));
            var position = new long[kept];
            var target = new Slice[1 + kept];
            target[0] = Slice.Ellipsis;
            for (long k = 0; k < rows; k++)
            {
                using var cdfRow = cdfRows[Slice.Index(k)];
                using var found = np.searchsorted(cdfRow, q, "left");
                // `indices = minimum(indices, values_count - 1)`: q = 1 can find len(arr).
                using var indices = np.minimum(found, clip);
                using var arrRow = arrRows[Slice.Index(k)];
                using var taken = np.take(arrRow, indices, 0);
                // `result[(...,) + kk] = ...`: the store with NumPy's setitem (unsafe cast) into out.
                for (int i = 0; i < kept; i++)
                    target[1 + i] = Slice.Index(position[i]);
                result[target] = taken;
                Advance(position, arr.Shape.dimensions);
            }
            return result;
        }

        /// <summary>
        ///     The per-row search of every row at once, with NumPy's results: a cdf row is non-decreasing (cumulative weights
        ///     over their total, the leading zeros made -1, never NaN once the totals passed their check), so
        ///     <c>searchsorted(row, v, side="left")</c> is the number of row entries below <c>v</c> — one vectorized
        ///     <c>less</c> and <c>count_nonzero</c> per q value over all rows, compared in the same common dtype as the
        ///     search. The picks are gathered with <c>take_along_axis</c>; an out of another dtype is stored row by row in
        ///     NumPy's order (the parse of a text into a number fails at NumPy's first element).
        /// </summary>
        /// <param name="arr">The sorted data, the reduced axis last.</param>
        /// <param name="arrRows">The sorted data as <c>(rows, n)</c>.</param>
        /// <param name="cdfRows">The normalized cumulative weights as <c>(rows, n)</c>, in q's float dtype.</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="rows">The number of rows.</param>
        /// <param name="rShape">The result's shape: q's dimensions, then the kept ones.</param>
        /// <param name="clip">A Python int <c>n - 1</c>.</param>
        /// <param name="out">The out (already shape-checked), or null for a fresh result of the data's dtype.</param>
        /// <returns>The result.</returns>
        /// <exception cref="Exception">The store's errors.</exception>
        [NDScoped]
        private static NDArray SearchByCounting(NDArray arr, NDArray arrRows, NDArray cdfRows, NDArray q, long rows,
            long[] rShape, NDArray clip, NDArray @out)
        {
            long qCount = q.size;
            NDArray qFlat = np.ravel(q);
            // counts[j, k] = searchsorted(cdf[k], q[j], 'left'): the entries of row k below q[j].
            NDArray counts = np.empty(new Shape(qCount, rows), np.int64);
            for (long j = 0; j < qCount; j++)
            {
                using var qj = qFlat[Slice.Index(j)];
                using var below = np.less(cdfRows, qj);
                using var count = np.count_nonzero(below, 1);
                using var dst = counts[Slice.Index(j)];
                np.copyto(dst, count);
            }
            np.minimum(counts, clip, @out: counts);
            // picked[k, j] = sorted row k at counts[j, k].
            NDArray picked = np.take_along_axis(arrRows, counts.T, 1);
            if (@out is not null && @out.typecode != arr.typecode)
            {
                // A cast store can fail per element (a text that is no number): NumPy stores position by position, each
                // position's q values in order.
                int kept = arr.ndim - 1;
                var position = new long[kept];
                var target = new Slice[1 + kept];
                target[0] = Slice.Ellipsis;
                var qShape = Shape.Adopt((long[])q.Shape.dimensions.Clone());
                for (long k = 0; k < rows; k++)
                {
                    using var row = picked[Slice.Index(k)];
                    using var shaped = row.reshape(qShape);
                    for (int i = 0; i < kept; i++)
                        target[1 + i] = Slice.Index(position[i]);
                    @out[target] = shaped;
                    Advance(position, arr.Shape.dimensions);
                }
                return @out;
            }
            // q's values first, then the positions: the transposed picks in C order are the result's elements.
            NDArray result = @out ?? np.empty_like(arr, shape: Shape.Adopt(rShape));
            NDArray byQ = picked.T.reshape(Shape.Adopt(rShape));
            np.copyto(result, byQ, "unsafe");
            return result;
        }

        /// <summary>Advances a C-order position over the leading <c>position.Length</c> dimensions of <paramref name="dims"/>.</summary>
        /// <param name="position">The position (modified in place; wraps to zeros after the last one).</param>
        /// <param name="dims">The dimensions (at least <c>position.Length</c> of them).</param>
        private static void Advance(long[] position, long[] dims)
        {
            for (int i = position.Length - 1; i >= 0; i--)
            {
                if (++position[i] < dims[i])
                    return;
                position[i] = 0;
            }
        }

        /// <summary>
        ///     NumPy's <c>_nanquantile_ureduce_func</c> for the weighted call (see the file header): the flat route stores into
        ///     out with <c>out[...] = result</c>; the per-slice route writes into out (no shape check) or a fresh result.
        /// </summary>
        /// <param name="a">The data (not empty).</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="wgt">The weights (validated, merged like the data).</param>
        /// <param name="axis">The reduction's axis (non-negative: the merged last axis arrives as its index), or null.</param>
        /// <param name="out">The out (or its keepdims view), or null.</param>
        /// <returns>The result and whether it is a Python object (the NaN NA object of an all-NaN <c>StringDType</c> line with
        ///     a scalar q, which NumPy returns as itself, not as an array).</returns>
        /// <exception cref="IndexError">Weights with length 1 at a kept axis longer than 1 (NumPy 2.4.2's own error), or an
        ///     out too small for the slice positions.</exception>
        /// <exception cref="Exception">The per-slice errors (see <see cref="NanQuantile1D"/>) and the store's.</exception>
        private static (NDArray Value, bool PyObject) NanUreduceFunc(NDArray a, NDArray q, NDArray wgt, int? axis, NDArray @out)
        {
            if (axis is null || a.ndim == 1)
            {
                (NDArray r, bool pyObject) = NanQuantile1D(np.ravel(a), q, np.ravel(wgt));
                // `if out is not None: out[...] = result` — a broadcasting store (an array value), or the setitem of a
                // Python object.
                if (@out is not null)
                    Store(@out, r, pyObject);
                return (r, pyObject);
            }
            return (NanRows(a, q, wgt, axis.Value, @out), false);
        }

        /// <summary>
        ///     The per-slice route of <c>_nanquantile_ureduce_func</c>: data and weights move their axis last, and every kept
        ///     position <c>ii</c> runs <c>result[(...,) + ii] = _nanquantile_1d(a[ii], q, weights=weights[ii])</c>.
        /// </summary>
        /// <param name="a">The data (two or more dimensions).</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="wgt">The weights.</param>
        /// <param name="axis">The reduction's axis.</param>
        /// <param name="out">The out (no shape check: NumPy stores per position, broadcasting), or null.</param>
        /// <returns>The result (<paramref name="out"/>, or a fresh array of the data's dtype).</returns>
        /// <exception cref="IndexError">NumPy's <c>weights[ii]</c> over weights with length 1 at a kept axis, or the store's
        ///     index into a too small out.</exception>
        /// <exception cref="Exception">The per-slice errors and the store's.</exception>
        [NDScoped]
        private static NDArray NanRows(NDArray a, NDArray q, NDArray wgt, int axis, NDArray @out)
        {
            a = np.moveaxis(a, axis, -1);
            wgt = np.moveaxis(wgt, axis, -1);
            int kept = a.ndim - 1;
            NDArray result = @out;
            if (result is null)
            {
                // `np.empty_like(a, shape=q.shape + a.shape[:-1])`: the data's dtype (weights limit the method to
                // inverted_cdf, which only takes elements).
                var shape = new long[q.ndim + kept];
                for (int i = 0; i < q.ndim; i++)
                    shape[i] = q.Shape.dimensions[i];
                for (int i = 0; i < kept; i++)
                    shape[q.ndim + i] = a.Shape.dimensions[i];
                result = np.empty_like(a, shape: Shape.Adopt(shape));
            }
            long rows = 1;
            for (int i = 0; i < kept; i++)
                rows *= a.Shape.dimensions[i];
            var position = new long[kept];
            var index = new Slice[kept];
            var target = new Slice[1 + kept];
            target[0] = Slice.Ellipsis;
            for (long k = 0; k < rows; k++)
            {
                for (int i = 0; i < kept; i++)
                {
                    index[i] = Slice.Index(position[i]);
                    target[1 + i] = index[i];
                }
                // Python evaluates `a[ii]`, then `weights[ii]` (NumPy's IndexError for length-1 kept axes), then the slice,
                // then the store.
                using var line = a[index];
                using var weightLine = wgt[index];
                (NDArray r, bool pyObject) = NanQuantile1D(line, q, weightLine);
                try
                {
                    using var view = result[target];
                    Store(view, r, pyObject);
                }
                finally
                {
                    r.Dispose();
                }
                for (int i = kept - 1; i >= 0; i--)
                {
                    if (++position[i] < a.Shape.dimensions[i])
                        break;
                    position[i] = 0;
                }
            }
            return result;
        }

        /// <summary>
        ///     NumPy's <c>_nanquantile_1d</c> with weights: the NaN elements and their weights removed in
        ///     <c>_remove_nan_1d</c>'s order; an all-NaN line answers <c>np.full(q.shape, nan, dtype=data.dtype)[()]</c>;
        ///     otherwise <c>_quantile_unchecked</c> of what is left.
        /// </summary>
        /// <param name="line">A 1-D line of data.</param>
        /// <param name="q">NumPy's q array.</param>
        /// <param name="weightLine">The line's weights (same length).</param>
        /// <returns>The line's result and whether it is a Python object (the NaN NA object of an all-NaN
        ///     <c>StringDType</c> line with a scalar q).</returns>
        /// <exception cref="TypeError">The line is <c>S</c> / <c>U</c> text (<c>isnan</c> has no text loop).</exception>
        /// <exception cref="Exception">The plain weighted path's errors for what is left.</exception>
        private static (NDArray Value, bool PyObject) NanQuantile1D(NDArray line, NDArray q, NDArray weightLine)
        {
            using var nan = np.isnan(line);
            long nans = np.count_nonzero(nan);
            if (nans == line.size)
                return (AllNaN(line.dtype, q), q.ndim == 0 && NDStringUfuncs.IsVString(line));
            if (nans == 0)
                return (UreduceFunc(line, q, weightLine, null, null), false);
            (NDArray kept, NDArray keptWeights) = RemoveNaN1D(line, weightLine, nan, nans);
            try
            {
                return (UreduceFunc(kept, q, keptWeights, null, null), false);
            }
            finally
            {
                kept.Dispose();
                keptWeights.Dispose();
            }
        }

        /// <summary>
        ///     NumPy's <c>_remove_nan_1d</c> with a second array: copies whose first NaN slots are filled, in order, with the
        ///     non-NaN elements of the last <c>nans</c> slots, cut to their first <c>n - nans</c> elements — the same moves
        ///     for the data and the weights, so every element keeps its weight.
        /// </summary>
        /// <param name="line">The 1-D data line (holding some, not all, NaNs).</param>
        /// <param name="weightLine">Its weights.</param>
        /// <param name="nan">The line's NaN mask.</param>
        /// <param name="nans">The number of NaNs.</param>
        /// <returns>The kept data and weights (views of fresh copies, owned by the caller).</returns>
        /// <remarks>The order matters: the sort keeps tied elements in arrival order where NumPy's sort does, and the
        ///     cumulative sum adds the weights in sorted order. Not scoped (it returns two arrays): every temporary is
        ///     disposed here, and each copy's NDArray is released once its view holds the buffer.</remarks>
        private static (NDArray Data, NDArray Weights) RemoveNaN1D(NDArray line, NDArray weightLine, NDArray nan, long nans)
        {
            long n = line.size;
            using var tailMask = nan[$"{n - nans}:"];
            using var tailKeep = np.logical_not(tailMask);
            using var positions = np.flatnonzero(nan);
            NDArray data = Fill(line.copy(), tailKeep, positions, n, nans);
            NDArray weights = Fill(weightLine.copy(), tailKeep, positions, n, nans);
            return (data, weights);

            // `enonan = arr1d[-s.size:][~c[-s.size:]]; arr1d[s[:enonan.size]] = enonan; return arr1d[:-s.size]` over a copy
            // this helper owns: the returned view keeps the copy's buffer, so the copy's own NDArray is released.
            static NDArray Fill(NDArray copy, NDArray keep, NDArray positions, long n, long nans)
            {
                try
                {
                    using (var tail = copy[$"{n - nans}:"])
                    using (var enonan = tail[keep])
                    {
                        if (enonan.size > 0)
                        {
                            using var head = positions[$":{enonan.size}"];
                            copy[head] = enonan;
                        }
                    }
                    return copy[$":{n - nans}"];
                }
                finally
                {
                    copy.Dispose();
                }
            }
        }

        /// <summary>
        ///     <c>np.full(q.shape, np.nan, dtype=dtype)[()]</c>: the answer of an all-NaN line — NaN in the data's float dtype,
        ///     or the NaN-like NA of a <c>StringDType</c> (np.full stores its Python float nan as the NA).
        /// </summary>
        /// <param name="dtype">The data's dtype (float, or a <c>StringDType</c> with a NaN-like NA).</param>
        /// <param name="q">NumPy's q array (its shape is the answer's).</param>
        /// <returns>A scalar for a 0-d q (NumPy's <c>[()]</c> of a 0-d array is its element: a NumPy scalar, or the NA
        ///     object); otherwise a view of the full array (<c>[()]</c> of an array is a view of all of it).</returns>
        private static NDArray AllNaN(DType dtype, NDArray q)
        {
            var shape = q.ndim == 0 ? Shape.Scalar : Shape.Adopt((long[])q.Shape.dimensions.Clone());
            var full = new NDArray(dtype, shape, true);
            using (var nanLiteral = NDArray.PythonLiteral(PositiveNaN))
            {
                if (full.typecode == NPTypeCode.String)
                {
                    using var na = NDArray.PackForVStringStore(nanLiteral, dtype);
                    np.copyto(full, na, "unsafe");
                }
                else
                    np.copyto(full, nanLiteral, "unsafe");
            }
            if (q.ndim == 0)
                return full.MarkReductionScalar();
            NDArray view = full.view();
            full.Dispose();
            return view;
        }

        /// <summary>
        ///     NumPy's <c>target[...] = value</c> as the nan forms store a line's answer: an array value by the unsafe cast and
        ///     broadcast, a Python object (the NaN NA object) or a NumPy scalar (a 0-d q's answer) by the destination's setitem
        ///     (<see cref="QuantileEngine.StoreNumPyScalar"/>: an integer destination refuses NaN and infinity).
        /// </summary>
        /// <param name="target">The destination (out, its view, or the result's slice).</param>
        /// <param name="value">The value.</param>
        /// <param name="pyObject">Whether <paramref name="value"/> stands for a Python object.</param>
        /// <exception cref="Exception">The store's errors (NumPy's setitem and cast texts).</exception>
        private static void Store(NDArray target, NDArray value, bool pyObject)
        {
            if (pyObject)
                NDVStringStatistics.PyAssign(target, value);
            else if (value.ndim == 0)
                // A 0-d q's answer is a NumPy SCALAR: stored by the target's setitem (an integer target refuses NaN).
                QuantileEngine.StoreNumPyScalar(target, value);
            else
                np.copyto(target, value, "unsafe");
        }

        /// <summary>
        ///     <c>r[(Ellipsis,) + index_r]</c>: a new axis at every reduced position. A NumPy scalar indexes into a fresh,
        ///     writeable array; a Python object (the NaN NA object) cannot be indexed with a tuple.
        /// </summary>
        /// <param name="r">The result.</param>
        /// <param name="pyObject">Whether <paramref name="r"/> stands for a Python object.</param>
        /// <param name="nd">The data's rank.</param>
        /// <param name="reduced">The normalized reduced axes, or null.</param>
        /// <returns>The view.</returns>
        /// <exception cref="TypeError">A Python object result (<c>'float' object is not subscriptable</c>).</exception>
        private static NDArray KeepdimsResult(NDArray r, bool pyObject, int nd, int[] reduced)
        {
            if (pyObject)
                throw NDVStringStatistics.ScalarSubscriptError(r.Storage.GetAtIndex(0));
            // A NumPy scalar (read-only here) subscripts through a fresh writeable array; a 0-d array through its own view.
            if (!r.Shape.IsWriteable)
                r = r.copy();
            var expand = new Slice[1 + nd];
            expand[0] = Slice.Ellipsis;
            for (int d = 0; d < nd; d++)
                expand[1 + d] = reduced is null || Array.IndexOf(reduced, d) >= 0 ? Slice.NewAxis : Slice.All;
            return r[expand];
        }

        /// <summary>Whether two dimension lists are equal (NumPy's tuple comparison of shapes).</summary>
        /// <param name="x">The first dimensions.</param>
        /// <param name="y">The second dimensions.</param>
        /// <returns>True when they have the same length and values.</returns>
        private static bool SameDims(long[] x, long[] y)
        {
            if (x.Length != y.Length)
                return false;
            for (int i = 0; i < x.Length; i++)
                if (x[i] != y[i])
                    return false;
            return true;
        }

        /// <summary>Python's <c>repr</c> of a shape tuple: <c>()</c>, <c>(2,)</c>, <c>(2, 3)</c>.</summary>
        /// <param name="dims">The dimensions.</param>
        /// <returns>The text.</returns>
        private static string PyTuple(long[] dims)
            => dims.Length == 1 ? $"({dims[0]},)" : "(" + string.Join(", ", dims) + ")";
    }
}
