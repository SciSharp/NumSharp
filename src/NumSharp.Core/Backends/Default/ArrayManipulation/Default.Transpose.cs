using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using NumSharp;
using NumSharp.Backends.Unmanaged;

namespace NumSharp.Backends
{
    public partial class DefaultEngine
    {
        // =============================================================================
        // The axis math is Shape's (View/Shape.Axes.cs): check_and_adjust_axis, normalize_axis_tuple, the permutations
        // of the axis views and the transposed view's layout. The public statics below shipped in 0.70.0 and forward to
        // it, so a caller of the released API keeps working; the library calls Shape's members. The four overrides are
        // the array-object members TensorEngine shipped (they stay on the engine): Shape computes their permutation,
        // the storage aliases the view.
        // =============================================================================

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, nd.ndim)</c>: <paramref name="axis"/> wrapped into
        ///     <c>[0, nd.ndim)</c>, or NumPy's <see cref="AxisError"/> naming the axis as given. Forwards to
        ///     <see cref="Shape.check_and_adjust_axis(int, int, string)"/>.
        /// </summary>
        /// <param name="nd">The array whose rank the axis addresses.</param>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndim, ndim)</c>.</exception>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(NDArray nd, int axis)
        {
            return Shape.check_and_adjust_axis(nd.ndim, axis, null);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim)</c> (C: <c>check_and_adjust_axis</c>): <paramref name="axis"/>
        ///     wrapped into <c>[0, ndims)</c>, or NumPy's <see cref="AxisError"/> naming the axis as given. Forwards to
        ///     <see cref="Shape.check_and_adjust_axis(int, int, string)"/>.
        /// </summary>
        /// <param name="ndims">The rank the axis addresses.</param>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndims, ndims)</c>: "axis {axis} is out
        ///     of bounds for array of dimension {ndims}".</exception>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(int ndims, int axis)
        {
            return Shape.check_and_adjust_axis(ndims, axis, null);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim, msg_prefix)</c> (C: <c>check_and_adjust_axis_msg</c>):
        ///     <paramref name="axis"/> wrapped into <c>[0, ndims)</c>, or NumPy's <see cref="AxisError"/> with the argument
        ///     name in front of its text. Forwards to <see cref="Shape.check_and_adjust_axis(int, int, string)"/>, the ONE
        ///     axis check every axis-taking function routes through.
        /// </summary>
        /// <param name="ndims">The rank the axis addresses. A 0-d operand is rank 0, so no axis fits it — the functions
        ///     that treat a 0-d operand as 1-D (NumPy's <c>PyArray_CheckAxis</c>: argmax, cumsum, take, compress, …)
        ///     pass 1 here themselves.</param>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <param name="msgPrefix">The argument name NumPy prints before the text (<c>"source"</c>, <c>"axis1"</c>,
        ///     <c>"axisa"</c>), or <c>null</c> for none.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndims, ndims)</c>.</exception>
        /// <remarks>
        ///     The range is tested on the value as given, BEFORE any wrap. A wrap-then-test spelling
        ///     (<c>while (axis &lt; 0) axis += ndim</c>) wraps <c>-ndim - 1</c> to <c>-1</c> and then to <c>ndim - 1</c>, so
        ///     an out-of-range axis silently reduces the last axis, and on a 0-d operand (<c>ndim == 0</c>) it never
        ///     terminates.
        /// </remarks>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(int ndims, int axis, string msgPrefix)
        {
            return Shape.check_and_adjust_axis(ndims, axis, msgPrefix);
        }

        /// <summary>
        ///     NumPy's <c>numpy.lib.array_utils.normalize_axis_tuple</c>: every axis of <paramref name="axis"/> wrapped
        ///     into <c>[0, ndim)</c> (an AxisError naming the argument when one is out of range), then — unless
        ///     <paramref name="allow_duplicate"/> — NumPy's "repeated axis" ValueError when two name the same axis.
        ///     Forwards to <see cref="Shape.normalize_axis_tuple(int[], int, object, bool)"/>.
        /// </summary>
        /// <param name="axis">The un-normalized axes. Never modified: the result is a new array.</param>
        /// <param name="ndim">The rank the axes address.</param>
        /// <param name="argname">The argument name: the AxisError's prefix, and the repeat text's
        ///     <c>"repeated axis in `{argname}` argument"</c>. <c>null</c> (or an empty name, for the repeat text) gives
        ///     the plain texts.</param>
        /// <param name="allow_duplicate">When false (NumPy's default) a repeated axis raises; when true the repeats are
        ///     kept in order.</param>
        /// <returns>The normalized axes, one per input entry, in input order.</returns>
        /// <exception cref="AxisError">An axis is out of range (checked for every axis before any repeat).</exception>
        /// <exception cref="ValueError">Two axes are the same after normalization and duplicates are not allowed.</exception>
        public static int[] normalize_axis_tuple(int[] axis, int ndim, object argname = null, bool allow_duplicate = false)
        {
            return Shape.normalize_axis_tuple(axis, ndim, argname, allow_duplicate);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_tuple</c> for one integer axis: the one-element tuple of the normalized axis.
        ///     Forwards to <see cref="Shape.normalize_axis_tuple(int, int, object, bool)"/>.
        /// </summary>
        /// <param name="axis">The un-normalized axis.</param>
        /// <param name="ndim">The rank the axis addresses.</param>
        /// <param name="argname">The argument name for the AxisError's prefix, or <c>null</c>.</param>
        /// <param name="allow_duplicate">Unused for one axis (a single axis cannot repeat); kept for NumPy's signature.</param>
        /// <returns>The normalized axis as a one-element array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        public static int[] normalize_axis_tuple(int axis, int ndim, object argname = null, bool allow_duplicate = false)
        {
            return Shape.normalize_axis_tuple(axis, ndim, argname, allow_duplicate);
        }

        /// <summary>
        ///     NumPy's <c>np.moveaxis</c>: the axes of <paramref name="source"/> moved to the positions of
        ///     <paramref name="destination"/>, the other axes keeping their order — a view through
        ///     <see cref="Transpose"/>.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="source">The axes to move (negative values count from the end).</param>
        /// <param name="destination">Their new positions.</param>
        /// <returns>A view with the permuted axes.</returns>
        /// <exception cref="AxisError">An axis is out of range: "source: axis …" or "destination: axis …".</exception>
        /// <exception cref="ValueError">An argument repeats an axis ("repeated axis in `source` argument"), or the two
        ///     have different lengths ("`source` and `destination` arguments must have the same number of elements").</exception>
        public override NDArray MoveAxis(NDArray nd, int[] source, int[] destination)
        {
            // np.moveaxis' statements in NumPy's order (source normalized and repeat-checked first, then destination, then
            // the lengths compared) give the permutation; the view is this engine's Transpose of it.
            return Transpose(nd, Shape.MoveAxisPermutation(nd.ndim, source, destination));
        }

        /// <summary>
        ///     NumPy's <c>PyArray_SwapAxes</c>: a view with <paramref name="axis1"/> and <paramref name="axis2"/>
        ///     exchanged.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="axis1">The first axis (negative values count from the end).</param>
        /// <param name="axis2">The second axis.</param>
        /// <returns>A view with the two axes swapped.</returns>
        /// <exception cref="AxisError">An axis is out of range: "axis1: axis …" or "axis2: axis …" (axis1 checked first).</exception>
        public override NDArray SwapAxes(NDArray nd, int axis1, int axis2)
        {
            return Transpose(nd, Shape.SwapAxesPermutation(nd.ndim, axis1, axis2));
        }

        /// <summary>
        ///     NumPy's <c>np.rollaxis</c>: <paramref name="axis"/> moved to lie before position <paramref name="start"/>,
        ///     the other axes keeping their order.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="axis">The axis to roll (negative values count from the end).</param>
        /// <param name="start">The position the axis is rolled to: <c>[-ndim, ndim]</c>, negative values counting from
        ///     the end.</param>
        /// <returns>A view of <paramref name="nd"/> (a new view object even when no axis moves, NumPy's <c>a[...]</c>).</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range, or <paramref name="start"/> is outside
        ///     its range: "'start' arg requires {-n} &lt;= start &lt; {n + 1}, but {start} was passed in" (the start after
        ///     its negative wrap, as NumPy prints it).</exception>
        public override NDArray RollAxis(NDArray nd, int axis, int start = 0)
        {
            var permutation = Shape.RollAxisPermutation(nd.ndim, axis, start);

            // `return a[...]`: a NEW view of the array, never the array object itself (NumPy's identity rule — the caller
            // may reshape or resize the result without touching the input). Null is the "no axis moves" answer.
            if (permutation == null)
                return new NDArray(nd.Storage.Alias()) { TensorEngine = nd.TensorEngine };

            return Transpose(nd, permutation);
        }

        /// <summary>
        ///     NumPy's <c>PyArray_Transpose</c>: a VIEW of <paramref name="nd"/> whose axis <c>i</c> is the source axis
        ///     <c>premute[i]</c>, or with the axes reversed when <paramref name="premute"/> is null.
        /// </summary>
        /// <param name="nd">The array.</param>
        /// <param name="premute">The axes in their new order (NumPy's <c>axes</c>), or null for <c>axes=None</c>. An
        ///     empty array is NumPy's <c>axes=()</c>, which only a 0-d array matches.</param>
        /// <returns>A new view sharing <paramref name="nd"/>'s buffer, with permuted dimensions and strides; it keeps the
        ///     source's writeability (a read-only broadcast or memmap stays read-only).</returns>
        /// <exception cref="ValueError">The permutation has another length than the array's rank ("axes don't match
        ///     array") or names an axis twice ("repeated axis in transpose").</exception>
        /// <exception cref="AxisError">An entry of the permutation is out of range.</exception>
        public override NDArray Transpose(NDArray nd, int[] premute = null)
        {
            // PyArray_Transpose's permutation: axes=None reverses, an explicit permutation is checked entry by entry
            // (Shape.TransposePermutation).
            var permutation = Shape.TransposePermutation(nd.ndim, premute);

            // An EMPTY array takes the same view path as any other: NumPy's PyArray_Transpose always returns a view of
            // its input with permuted dimensions AND strides (owndata False, the source's writeability kept). A former
            // shortcut here allocated a fresh array instead, so a transposed empty array owned its data, lost the
            // permuted strides, and was writeable even when its source was read-only.

            // NumPy-aligned: Transpose returns a VIEW by permuting strides.
            // For contiguous arrays, this is a simple stride permutation.
            // For non-contiguous arrays (sliced, already transposed), we need to
            // permute the CURRENT strides, which already encode the view's layout.
            //
            // No data copy is needed - transpose is always O(1).
            // The transposed shape shares memory with the original.
            var newShape = Shape.TransposedShape(nd.Shape, permutation);

            // Return an alias (view) with the permuted shape. Alias() inherits writeability from the
            // source, so transposing a read-only array (broadcast / 'r' memmap) stays read-only.
            return new NDArray(nd.Storage.Alias(newShape)) { TensorEngine = nd.TensorEngine };
        }
    }
}
