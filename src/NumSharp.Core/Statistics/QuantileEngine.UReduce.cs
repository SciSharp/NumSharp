using System;

namespace NumSharp.Statistics
{
    /// <summary>
    ///     What NumPy's <c>_ureduce</c> hands its reduction function (<c>_median</c>, <c>_quantile_ureduce_func</c>, the nan
    ///     forms' functions): the data with several reduced axes merged into one, the one axis the function reduces, the
    ///     normalized axes, keepdims' view of <c>out</c>, and the weights merged like the data.
    /// </summary>
    /// <remarks>
    ///     <see cref="Data"/>, <see cref="Target"/> and <see cref="Weights"/> are NEW arrays when the prologue made them (a
    ///     merge, keepdims' out index) and the caller's own arrays otherwise. The callers are <c>[NDScoped]</c> (their scope
    ///     reclaims what the prologue made) or run the prologue inside an <see cref="NDScope"/> of their own; a caller of
    ///     neither kind must dispose exactly the arrays that differ from its arguments.
    /// </remarks>
    internal readonly struct UReduced
    {
        /// <summary>Records one prologue's outcome.</summary>
        /// <param name="data">The data the reduction function receives.</param>
        /// <param name="axis">The axis it reduces, or null for every axis.</param>
        /// <param name="reduced">The normalized reduced axes in the caller's order, or null.</param>
        /// <param name="target">The out the reduction writes into, or null.</param>
        /// <param name="weights">The weights merged like the data, or null.</param>
        internal UReduced(NDArray data, int? axis, int[] reduced, NDArray target, NDArray weights)
        {
            Data = data;
            Axis = axis;
            Reduced = reduced;
            Target = target;
            Weights = weights;
        }

        /// <summary>
        ///     The data: the caller's array for <c>axis=None</c> or one reduced axis; for several (or an EMPTY tuple of) reduced
        ///     axes, <c>np.moveaxis(a, keep, range(nkeep)).reshape(a.shape[:nkeep] + (-1,))</c> — the kept axes first, the
        ///     reduced ones merged into one last axis, its elements in the order of their original axes.
        /// </summary>
        internal NDArray Data { get; }

        /// <summary>
        ///     The axis of <see cref="Data"/> the function reduces: null for <c>axis=None</c> (every axis: the function
        ///     flattens), the one reduced axis, or the merged last axis (<c>Data.ndim - 1</c>, NumPy's <c>-1</c>).
        /// </summary>
        internal int? Axis { get; }

        /// <summary>
        ///     The axis exactly as NumPy's <c>_ureduce</c> hands it to the reduction function (<c>kwargs['axis']</c>): null
        ///     for <c>axis=None</c>, the one reduced axis normalized, or <c>-1</c> for merged axes (the empty tuple
        ///     included). Use <see cref="Axis"/> to index; use this where NumPy's reduction function does arithmetic on its
        ///     axis argument: <c>_nanquantile_ureduce_func</c>'s <c>from_ax = [axis + i for i in range(q.ndim)]</c> runs on
        ///     the UNnormalized <c>-1</c>, which scrambles q's axes for a q of two or more dimensions.
        /// </summary>
        internal int? NumPyAxis => Reduced is null ? null : Reduced.Length == 1 ? Axis : -1;

        /// <summary>The reduced axes normalized (<c>normalize_axis_tuple</c>, the caller's order), or null for
        ///     <c>axis=None</c>.</summary>
        internal int[] Reduced { get; }

        /// <summary>
        ///     The out the reduction writes into: under keepdims, <c>out[(Ellipsis,) + index_out]</c> (every reduced axis taken
        ///     at 0, every axis for <c>axis=None</c>); otherwise the caller's out (or null).
        /// </summary>
        internal NDArray Target { get; }

        /// <summary>The weights merged like the data (NumPy's <c>reshape_arr</c> of <c>kwargs["weights"]</c>), the caller's
        ///     weights without a merge, or null.</summary>
        internal NDArray Weights { get; }
    }

    internal static partial class QuantileEngine
    {
        /// <summary>
        ///     NumPy's <c>_ureduce</c> up to its call of the reduction function — the ONE prologue of <c>np.median</c>,
        ///     <c>np.quantile</c>, <c>np.percentile</c> and their nan forms, in NumPy's order: the axes normalized
        ///     (<c>normalize_axis_tuple</c>: every axis range-checked, then the repeats), keepdims' view of <c>out</c>, then the
        ///     merge of several reduced axes into one. Everything the family checks after it (q's rank, the method, the data's
        ///     dtype) is the reduction function's business, so a route that runs this first reports NumPy's errors in NumPy's
        ///     order.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="axes">The reduced axes as the caller gave them (negative values count from the end), or null for
        ///     every axis. An EMPTY array reduces nothing: it merges into a trailing axis of length 1.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="keepdims">Whether the reduced axes are kept: then the reduction writes into keepdims' view of
        ///     <paramref name="out"/>.</param>
        /// <param name="weights">The weights (the weighted quantile), merged like the data; or null.</param>
        /// <returns>The prologue's outcome (see <see cref="UReduced"/> for which arrays are new).</returns>
        /// <exception cref="AxisError">An axis is out of range for <paramref name="a"/>.</exception>
        /// <exception cref="ValueError">An axis repeats (<c>repeated axis</c>).</exception>
        /// <exception cref="IndexError">Under keepdims, an out of fewer dimensions than the data (<c>too many indices for
        ///     array: array is 1-dimensional, but 2 were indexed</c>) or with a reduced axis of length 0 (<c>index 0 is out of
        ///     bounds for axis 0 with size 0</c>).</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0: the merge's
        ///     <c>-1</c> cannot be resolved (NumPy's ValueError <c>cannot reshape array of size 0 into shape
        ///     (0,newaxis)</c>).</exception>
        /// <remarks>
        ///     The np-layer dtype plan's P1a (step 2) made this the family's one copy: the numeric route
        ///     (<see cref="Compute"/>) merged the axes with an explicit length after q's rank and the method were checked, so a
        ///     kept axis of length 0 gave an empty result (plan Appendix B1), and the fixed-width text routes of np.median /
        ///     np.nanmedian / np.nanquantile refused the text before keepdims' out was indexed.
        /// </remarks>
        internal static UReduced UReduce(NDArray a, int[] axes, NDArray @out, bool keepdims, NDArray weights = null)
        {
            int nd = a.ndim;
            if (axes is null)
            {
                // `elif keepdims and out is not None: kwargs['out'] = out[(Ellipsis,) + (0,) * nd]`.
                NDArray whole = keepdims && @out is not null ? KeepdimsOutTarget(@out, nd, null) : @out;
                return new UReduced(a, null, null, whole, weights);
            }
            int[] reduced = NormalizeAxes(axes, nd);
            // `if keepdims and out is not None: kwargs['out'] = out[(Ellipsis,) + index_out]` — before the merge, so an out
            // of too few dimensions is NumPy's IndexError even where the merge would refuse.
            NDArray target = keepdims && @out is not null ? KeepdimsOutTarget(@out, nd, reduced) : @out;
            if (reduced.Length == 1)
                return new UReduced(a, reduced[0], reduced, target, weights);
            // Several axes (or none): the kept ones first, the reduced ones merged into one last axis (`reshape_arr`), the
            // weights alike. NumPy's reshape cannot resolve the -1 beside a kept axis of length 0, so that data raises here.
            int[] keep = KeptAxes(nd, reduced);
            NDArray data = MergeReducedAxes(a, keep);
            NDArray mergedWeights = weights is null ? null : MergeReducedAxes(weights, keep);
            return new UReduced(data, keep.Length, reduced, target, mergedWeights);
        }

        /// <summary>
        ///     <see cref="UReduce"/> for a route that only needs its ERRORS: a text route that NumPy runs through the whole
        ///     prologue before its own refusal (np.median's closing mean, the nan forms' <c>isnan</c>). The arrays the prologue
        ///     makes are reclaimed at once.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="axes">The reduced axes as the caller gave them, or null for every axis.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="keepdims">Whether the reduced axes are kept.</param>
        /// <exception cref="AxisError">An axis is out of range.</exception>
        /// <exception cref="ValueError">An axis repeats.</exception>
        /// <exception cref="IndexError">keepdims' index of an out of too few dimensions, or of a reduced axis of length 0.</exception>
        /// <exception cref="IncorrectShapeException">Several reduced axes beside a kept axis of length 0.</exception>
        internal static void UReduceChecks(NDArray a, int[] axes, NDArray @out, bool keepdims)
        {
            // The out view and the merged data are needed only for their errors: the scope returns their buffers' references.
            using var scope = NDScope.Open();
            UReduce(a, axes, @out, keepdims);
        }

        /// <summary>
        ///     <see cref="UReduceChecks(NDArray, int[], NDArray, bool)"/> for a one-axis (or <c>axis=None</c>) call: no merge,
        ///     so only the axis and keepdims' out index can raise.
        /// </summary>
        /// <param name="a">The data.</param>
        /// <param name="axis">The axis, or null for every axis.</param>
        /// <param name="out">The caller's out, or null.</param>
        /// <param name="keepdims">Whether the reduced axis is kept.</param>
        /// <exception cref="AxisError">The axis is out of range.</exception>
        /// <exception cref="IndexError">keepdims' index of an out of too few dimensions, or of a reduced axis of length 0.</exception>
        internal static void UReduceChecks(NDArray a, int? axis, NDArray @out, bool keepdims)
            => UReduceChecks(a, axis.HasValue ? new[] { axis.Value } : null, @out, keepdims);

        /// <summary>
        ///     The kept axes of <c>_ureduce</c>'s merge: <c>sorted(set(range(nd)) - set(axis))</c>, increasing.
        /// </summary>
        /// <param name="nd">The data's rank.</param>
        /// <param name="reduced">The normalized reduced axes.</param>
        /// <returns>Every axis that is not reduced, in increasing order.</returns>
        private static int[] KeptAxes(int nd, int[] reduced)
        {
            int count = 0;
            for (int d = 0; d < nd; d++)
                if (Array.IndexOf(reduced, d) < 0)
                    count++;
            var keep = new int[count];
            for (int d = 0, k = 0; d < nd; d++)
                if (Array.IndexOf(reduced, d) < 0)
                    keep[k++] = d;
            return keep;
        }

        /// <summary>
        ///     NumPy's <c>reshape_arr</c>: the kept axes moved to the front (<c>np.moveaxis(x, keep, range(nkeep))</c>), the
        ///     others merged into one last axis (<c>reshape(x.shape[:nkeep] + (-1,))</c>), in their original order.
        /// </summary>
        /// <param name="x">The data or the weights (its own dimensions are merged: weights of length 1 at a kept axis stay
        ///     length 1 there).</param>
        /// <param name="keep">The kept axes, increasing.</param>
        /// <returns>The merged array (a view when the layout allows, else a copy).</returns>
        /// <exception cref="IncorrectShapeException">A kept axis of length 0 beside merged ones: the -1 cannot be resolved
        ///     (<c>cannot reshape array of size 0 into shape (0,newaxis)</c>).</exception>
        private static NDArray MergeReducedAxes(NDArray x, int[] keep)
        {
            var destinations = new int[keep.Length];
            for (int i = 0; i < destinations.Length; i++)
                destinations[i] = i;
            // No kept axis: the moveaxis is the identity, and the whole array becomes the one merged axis.
            NDArray moved = keep.Length == 0 ? x : np.moveaxis(x, keep, destinations);
            var merged = new long[keep.Length + 1];
            for (int d = 0; d < keep.Length; d++)
                merged[d] = moved.Shape.dimensions[d];
            merged[keep.Length] = -1;
            return moved.reshape(Shape.Adopt(merged));
        }
    }
}
