using System;
using NumSharp.Backends;

// =============================================================================
// Default.Unique.cs - DefaultEngine's np.unique and the Array-API unique_* family, after their facades' class boundaries
// (numpy/lib/_arraysetops_impl.py: unique, _unique1d, unique_all, unique_counts, unique_inverse, unique_values)
// =============================================================================
//
// DefaultEngine's overrides of TensorEngine.Unique, UniqueValues, UniqueCounts, UniqueInverse and UniqueAll
// (TensorEngine.ArraySetOps.cs). NumPy writes unique in Python, with the C helper _unique_hash in unique.cpp; the
// np-layer dtype plan's P2 moved the statements from NDArray.unique and the np.unique / np.unique_values facades into the
// internal operation class UniqueEngine, and its phase PE (E1) made that class these DefaultEngine partials. The facades
// and the public instance members NDArray.unique(...) keep their signatures, documentation and class boundaries
// (numpy.char.chararray, numpy.matrix) and call the engine members. Nothing here opens a class boundary: every caller
// has already decided the operand is a plain array (a chararray operand computes through NDCharArrayBoundary, a
// numpy.matrix operand inside NDMatrixBoundary's plain region).
//
// Two helpers were renamed in the merge, because DefaultEngine shares one namespace of member names: Compute is
// UniqueCompute (QuantileEngine had a Compute too, QuantileCompute since it joined DefaultEngine), and Splitmix is
// UniqueSplitmix (np.isin's helper of the same name joined DefaultEngine first). NumPy's statements call np.moveaxis,
// np.lexsort, np.take, np.any, np.cumsum, np.nonzero, np.diff, ...; here they are the operands' engine members and the
// array object's.
//
// The routes, in NumPy's order:
//   * text first: the axis range check, StringDType's refusal of the structured axis path (ThrowIfTextUnique), then
//     StringDType's flat path (Default.Unique.VString.cs) or fixed-width text's statements (Default.Unique.Text.cs);
//   * `if axis is None or ar.ndim == 1`: the flat path (Default.Unique.Kwargs.cs, the sort + mask pipeline, with the hash
//     routes of Default.Unique.Hash.cs and the radix sorts of Default.Unique.Radix.cs), its inverse reshaped to the input's
//     shape;
//   * otherwise the row path of an axis (uniqueAxisKwargs).
// =============================================================================

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.Unique"/>: <c>np.unique</c>'s statements after its class boundary,
        ///     the checks NumPy makes before it reads text, then the computation inside a boundary scope that keeps only
        ///     the returned outputs.
        /// </summary>
        /// <param name="ar">The plain operand (only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices (along the axis, if given).</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="axis">The axis, or null to flatten the operand first.</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c> (the flat numeric path only).</param>
        /// <param name="sorted">NumPy's <c>sorted</c>: accepted, the values always come back sorted (see
        ///     <see cref="UniqueCompute"/>).</param>
        /// <returns>[values, index?, inverse?, counts?], NumPy's order; each array yielded from the scope.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand with an axis over two or more dimensions.</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <remarks>Also the body of both <c>NDArray.unique</c> members since the plan's phase PE (they were the unscoped
        ///     <c>Bare</c> and <c>UniqueCompute</c> calls; the checks and the outputs are the same, the internals are now
        ///     reclaimed at this method's return).</remarks>
        public override NDArray[] Unique(NDArray ar, bool return_index = false, bool return_inverse = false,
            bool return_counts = false, int? axis = null, bool equal_nan = true, bool sorted = true)
        {
            // NumPy's one check before it reads text, the axis range, then StringDType's refusal of the structured axis path
            // (an axis over two or more dimensions). Text of every class then computes in UniqueCompute.
            ThrowIfTextUnique(ar, axis);
            // Boundary scope: the sort/argsort/mask internals are reclaimed at exit; the present outputs are yielded (the
            // facade's UniqueResult carries only those arrays).
            using var scope = NDScope.Open();
            return scope.Returns(UniqueCompute(ar, return_index, return_inverse, return_counts, axis, equal_nan, sorted));
        }

        /// <summary>
        ///     NumPy's <c>np.unique</c> computation (the body of <see cref="Unique"/> and of the text routes of the
        ///     <c>unique_*</c> family): text first (StringDType's flat path, or fixed-width text's statements), then
        ///     <c>if axis is None or ar.ndim == 1</c> the flat path with its inverse reshaped to the input's shape,
        ///     otherwise the row path of the axis. Not scoped: its callers own the scope.
        /// </summary>
        /// <param name="a">The plain operand (only read).</param>
        /// <param name="return_index">Also return the first-occurrence indices (along the axis, if given).</param>
        /// <param name="return_inverse">Also return the reconstruction indices.</param>
        /// <param name="return_counts">Also return the counts.</param>
        /// <param name="axis">The axis, or null to flatten the operand first. A 1-D operand's axis is normalized and
        ///     checked, then the flat path runs (NumPy's rule).</param>
        /// <param name="equal_nan">NumPy's <c>equal_nan</c>: the flat numeric path collapses NaNs when true; the row path
        ///     and fixed-width text ignore it; StringDType's values-only route uses it.</param>
        /// <param name="sorted">NumPy's <c>sorted</c> (NumPy 2.3). The values always come back sorted: NumPy's
        ///     <c>sorted=False</c> returns its hash set's iteration order for integer, complex and text values, which is
        ///     platform-specific (identical as a set). Over StringDType it decides one thing NumPy's does: without the
        ///     sort, a None-like NA is never compared (and never refused).</param>
        /// <returns>[values, index?, inverse?, counts?]: the values in the operand's dtype, the index outputs int64. The
        ///     flat path's inverse has the input's shape (a view: <c>()</c> for a 0-d input, <c>(0, 3)</c> for an empty
        ///     <c>(0, 3)</c> one); the row path's inverse is 1-D, as NumPy returns it with an axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> operand of two or more dimensions with an axis (NumPy's
        ///     structured path refuses it).</exception>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose NA object is neither NaN-like nor a string,
        ///     where NumPy's sort compares an NA (<see cref="UniqueVString"/>).</exception>
        /// <exception cref="NotSupportedException">A dtype no route serves (none of the 15 numeric dtypes or the three
        ///     text dtypes reaches it).</exception>
        internal NDArray[] UniqueCompute(
            NDArray a,
            bool return_index,
            bool return_inverse = false,
            bool return_counts = false,
            int? axis = null,
            bool equal_nan = true,
            bool sorted = true)
        {
            // sorted does not change the values: NumSharp cannot reproduce NumPy's non-portable sorted=False hash
            // order (integer/complex/text, values-only), so it always returns the deterministic sorted
            // result — identical AS A SET, and consistent with the unique_values family. Over StringDType it decides one
            // thing NumPy's does: without the sort, a None-like NA is never compared (and never refused).

            // Text: NumPy's axis check first. StringDType is tested first (IsString refuses it: its fixed-width readers would
            // misread the 16-byte cells) and computes NumPy's flat path (Default.Unique.VString.cs); fixed-width text
            // computes NumPy's statements (Default.Unique.Text.cs).
            ThrowIfTextUnique(a, axis);
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                var vstringOutputs = UniqueVString(a, return_index, return_inverse, return_counts, equal_nan, sorted);
                // The flat path reshapes the inverse to the input's shape, as the numeric path.
                if (return_inverse)
                    ReshapeFlatInverseToInput(a, vstringOutputs, return_index ? 2 : 1);
                return vstringOutputs;
            }
            if (Backends.Strings.NDStringUfuncs.IsString(a))
            {
                var textOutputs = UniqueText(a, return_index, return_inverse, return_counts, axis);
                // The flat path (axis=None or a 1-D input) reshapes the inverse to the input's shape, as the numeric path.
                if (return_inverse && (axis is null || a.ndim == 1))
                    ReshapeFlatInverseToInput(a, textOutputs, return_index ? 2 : 1);
                return textOutputs;
            }

            // NumPy: `if axis is None or ar.ndim == 1` takes the flat path — a given axis normalized first — so a 1-D input
            // with axis 0 is the flat unique (its inverse the reshaped view below), not the slab path.
            if (axis is { } given)
            {
                int resolved = given < 0 ? given + a.ndim : given;
                if (resolved < 0 || resolved >= a.ndim)
                    throw new AxisError(given, a.ndim);
                if (a.ndim != 1)
                    return uniqueAxisKwargs(a, resolved, return_index, return_inverse, return_counts, equal_nan);
            }

            var outputs = uniqueFlatKwargs(a, return_index, return_inverse, return_counts, equal_nan);
            if (return_inverse)
                ReshapeFlatInverseToInput(a, outputs, return_index ? 2 : 1);
            return outputs;
        }

        /// <summary>
        ///     The checks of <c>np.unique</c> over text that come before its computation: an explicit axis must be in range
        ///     (NumPy's <c>np.moveaxis</c> re-raised as <c>AxisError(axis, ndim)</c>, or <c>normalize_axis_index</c> for a 1-D
        ///     input: the same text), and a <c>StringDType</c> array refuses the structured path of an axis over two or more
        ///     dimensions (<see cref="ThrowIfVStringUnique"/>). Does nothing for a numeric array.
        /// </summary>
        /// <param name="a">The array <c>unique</c> was called on.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <exception cref="AxisError">A string array with an out-of-range <paramref name="axis"/> (<c>axis 9 is out of
        ///     bounds for array of dimension 1</c>; a 0-d array refuses every axis).</exception>
        /// <exception cref="TypeError">A <c>StringDType</c> array of two or more dimensions with an axis (see
        ///     <see cref="ThrowIfVStringUnique"/>).</exception>
        internal static void ThrowIfTextUnique(NDArray a, int? axis)
        {
            // StringDType first: IsString refuses it loudly (its callers are fixed-width routes).
            if (Backends.Strings.NDStringUfuncs.IsVString(a))
            {
                ThrowIfVStringUnique(a, axis);
                return;
            }
            if (!Backends.Strings.NDStringUfuncs.IsString(a))
                return;
            if (axis is { } ax && (ax < -a.ndim || ax >= a.ndim))
                throw new AxisError(ax, a.ndim);
        }

        /// <summary>
        ///     The guard of <c>np.unique</c> (and the Array-API family) over NEP 55 <c>StringDType</c> text, in NumPy 2.4.2's
        ///     order: the flat path (<c>axis=None</c>, or a 1-D input whose axis passed <c>normalize_axis_index</c>) computes
        ///     (<see cref="UniqueVString"/>); an axis over two or more dimensions takes the structured path, which cannot hold
        ///     <c>StringDType</c> fields.
        /// </summary>
        /// <param name="a">The <c>StringDType</c> array.</param>
        /// <param name="axis">The axis, or null for the flattened array.</param>
        /// <exception cref="AxisError">The axis is out of bounds (<c>np.moveaxis</c>'s check re-raised as
        ///     <c>AxisError(axis, ndim)</c>, or <c>normalize_axis_index</c> for a 1-D input: the same text).</exception>
        /// <exception cref="TypeError">An axis over two or more dimensions: <c>The axis argument to unique is not supported
        ///     for dtype StringDType()</c> (the structured view of the rows refuses <c>StringDType</c> fields), or — for rows of
        ///     zero width, which skip that view — <c>Cannot change data-type for array of references.</c> (the field-less
        ///     result's view back as <c>StringDType</c>).</exception>
        private static void ThrowIfVStringUnique(NDArray a, int? axis)
        {
            if (axis is { } ax)
            {
                if (ax < -a.ndim || ax >= a.ndim)
                    throw new AxisError(ax, a.ndim);
                if (a.ndim != 1)
                {
                    // `ar = moveaxis(ar, axis, 0).reshape(n, prod(rest))`: the row width is every other axis's extent.
                    int norm = ax < 0 ? ax + a.ndim : ax;
                    long width = 1;
                    for (int d = 0; d < a.ndim; d++)
                        if (d != norm)
                            width *= a.Shape.dimensions[d];
                    if (width > 0)
                        throw new TypeError($"The axis argument to unique is not supported for dtype {a.dtype}");
                    throw new TypeError("Cannot change data-type for array of references.");
                }
            }
        }

        /// <summary>
        ///     NumPy's <c>inv_idx.reshape(inverse_shape)</c> for the flat path (<c>axis=None</c>, or a 1-D input): the inverse
        ///     is always a VIEW with the input's shape (<c>flags.owndata</c> False, as NumPy's reshape returns) — <c>()</c> for a
        ///     0-d input (the flat path makes <c>(1,)</c>), <c>(0, 3)</c> for an empty <c>(0, 3)</c> one (it makes <c>(0,)</c>).
        /// </summary>
        /// <param name="a">The operand whose shape the inverse takes.</param>
        /// <param name="outputs">The flat path's outputs; the inverse at <paramref name="at"/> is replaced by its reshaped view.</param>
        /// <param name="at">The inverse's position (1 without <c>return_index</c>, 2 with it).</param>
        private static void ReshapeFlatInverseToInput(NDArray a, NDArray[] outputs, int at)
        {
            var inverse = outputs[at];
            outputs[at] = inverse.reshape(Shape.Adopt((long[])a.Shape.dimensions.Clone()));
            // The view holds its own reference to the inverse's buffer; the flat wrapper is released.
            inverse.Dispose();
        }

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.UniqueValues"/>: <c>np.unique_values</c>' statements after its class
        ///     boundary, <c>unique(x, equal_nan=False, sorted=False)</c> over the flattened operand. Text computes through
        ///     <see cref="UniqueCompute"/>; numbers take the hash and sort routes of <see cref="uniqueValuesFast"/>.
        /// </summary>
        /// <param name="x">The plain operand (only read).</param>
        /// <returns>The unique values, sorted (NumPy returns its hash set's order for integers, complex and text: the
        ///     documented divergence).</returns>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE): the hash / sort internals are reclaimed at
        ///     this method's return instead of the facade's, the values are yielded.</remarks>
        [NDScoped]
        public override NDArray UniqueValues(NDArray x)
        {
            // NumPy hashes text (sorts it with an index output); text of every class computes in UniqueCompute (this guard refuses
            // only StringDType's structured axis path, which no axis-less call reaches).
            ThrowIfTextUnique(x, null);
            // StringDType first: IsString refuses it loudly (its callers are fixed-width routes).
            if (Backends.Strings.NDStringUfuncs.IsVString(x) || Backends.Strings.NDStringUfuncs.IsString(x))
                return UniqueCompute(x, return_index: false, return_inverse: false, return_counts: false, axis: null, equal_nan: false,
                               sorted: false)[0];
            return uniqueValuesFast(x);
        }

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.UniqueCounts"/>: <c>np.unique_counts</c>' statements after its class
        ///     boundary, <c>unique(x, return_counts=True, equal_nan=False)</c>. Text computes through
        ///     <see cref="UniqueCompute"/>; numbers take <see cref="uniqueCountsFast"/>.
        /// </summary>
        /// <param name="x">The plain operand (only read).</param>
        /// <returns>[values, counts].</returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE): the internals are reclaimed at this method's
        ///     return, both outputs are yielded.</remarks>
        [NDScoped]
        public override NDArray[] UniqueCounts(NDArray x)
        {
            // NumPy sorts text: text of every class computes in UniqueCompute (this guard refuses only StringDType's structured
            // axis path, which no axis-less call reaches).
            ThrowIfTextUnique(x, null);
            return Backends.Strings.NDStringUfuncs.IsVString(x) || Backends.Strings.NDStringUfuncs.IsString(x)
                ? UniqueCompute(x, return_index: false, return_inverse: false, return_counts: true, axis: null, equal_nan: false)
                : uniqueCountsFast(x);
        }

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.UniqueInverse"/>: <c>np.unique_inverse</c>'s statements after its
        ///     class boundary, <c>unique(x, return_inverse=True, equal_nan=False)</c>, the inverse reshaped to the operand's
        ///     shape.
        /// </summary>
        /// <param name="x">The plain operand (only read).</param>
        /// <returns>[values, inverse_indices]; the inverse has <paramref name="x"/>'s shape.</returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE). The flat inverse that the reshape views is
        ///     not returned, so the scope disposes it at this method's return (np.unique_inverse's scope did, before);
        ///     the view holds its own reference and keeps the buffer alive.</remarks>
        [NDScoped]
        public override NDArray[] UniqueInverse(NDArray x)
        {
            // NumPy sorts text: text of every class computes in UniqueCompute below (this guard refuses only StringDType's
            // structured axis path, which no axis-less call reaches).
            ThrowIfTextUnique(x, null);
            var r = UniqueCompute(x, return_index: false, return_inverse: true, return_counts: false,
                            axis: null, equal_nan: false);
            return new[] { r[0], ReshapeInverseToInput(r[1], x) };
        }

        /// <summary>
        ///     DefaultEngine's <see cref="TensorEngine.UniqueAll"/>: <c>np.unique_all</c>'s statements after its class
        ///     boundary, <c>unique(x, return_index=True, return_inverse=True, return_counts=True, equal_nan=False)</c>, the
        ///     inverse reshaped to the operand's shape.
        /// </summary>
        /// <param name="x">The plain operand (only read).</param>
        /// <returns>[values, indices, inverse_indices, counts]; the inverse has <paramref name="x"/>'s shape.</returns>
        /// <exception cref="ValueError">A <c>StringDType</c> operand whose sort compares a None-like NA.</exception>
        /// <exception cref="OutOfMemoryException">A corrupt <c>StringDType</c> cell (NumPy's <c>MemoryError</c> text).</exception>
        /// <remarks>Owns its scope (a public entry since the plan's phase PE). The flat inverse that the reshape views is
        ///     not returned, so the scope disposes it at this method's return (np.unique_all's scope did, before); the
        ///     view holds its own reference and keeps the buffer alive.</remarks>
        [NDScoped]
        public override NDArray[] UniqueAll(NDArray x)
        {
            // NumPy sorts text: text of every class computes in UniqueCompute below (this guard refuses only StringDType's
            // structured axis path, which no axis-less call reaches).
            ThrowIfTextUnique(x, null);
            var r = UniqueCompute(x, return_index: true, return_inverse: true, return_counts: true,
                            axis: null, equal_nan: false);
            return new[] { r[0], r[1], ReshapeInverseToInput(r[2], x), r[3] };
        }

        /// <summary>
        ///     Reshapes the flat inverse-index array to the ORIGINAL input shape, matching NumPy's
        ///     <c>inv_idx.reshape(ar.shape)</c> so the input can be reconstructed with
        ///     <c>np.take(values, inverse_indices)</c>. The underlying kwargs path already yields the
        ///     input shape for ndim ≥ 1; a 0-d input needs its <c>()</c> scalar shape restored (the flat
        ///     path returns it as <c>(1,)</c>). Reshape is a zero-copy view over the freshly-allocated
        ///     inverse array.
        /// </summary>
        /// <param name="inverse">The inverse the computation returned.</param>
        /// <param name="x">The operand whose shape the inverse takes.</param>
        /// <returns>A view of <paramref name="inverse"/> with <paramref name="x"/>'s shape.</returns>
        private static NDArray ReshapeInverseToInput(NDArray inverse, NDArray x)
            => inverse.reshape(Shape.Adopt(x.Shape.dimensions));
    }
}
