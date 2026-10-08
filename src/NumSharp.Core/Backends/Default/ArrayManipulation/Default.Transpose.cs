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
        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, nd.ndim)</c>: <paramref name="axis"/> wrapped into
        ///     <c>[0, nd.ndim)</c>, or NumPy's <see cref="AxisError"/> naming the axis as given.
        /// </summary>
        /// <param name="nd">The array whose rank the axis addresses.</param>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndim, ndim)</c>.</exception>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(NDArray nd, int axis)
        {
            return check_and_adjust_axis(nd.ndim, axis, null);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim)</c> (C: <c>check_and_adjust_axis</c>): <paramref name="axis"/>
        ///     wrapped into <c>[0, ndims)</c>, or NumPy's <see cref="AxisError"/> naming the axis as given.
        /// </summary>
        /// <param name="ndims">The rank the axis addresses.</param>
        /// <param name="axis">The axis; negative values count from the end.</param>
        /// <returns>The non-negative axis.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside <c>[-ndims, ndims)</c>: "axis {axis} is out
        ///     of bounds for array of dimension {ndims}".</exception>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(int ndims, int axis)
        {
            return check_and_adjust_axis(ndims, axis, null);
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_index(axis, ndim, msg_prefix)</c> (C: <c>check_and_adjust_axis_msg</c>), the ONE
        ///     axis check every axis-taking function routes through: <paramref name="axis"/> wrapped into
        ///     <c>[0, ndims)</c>, or NumPy's <see cref="AxisError"/> with the argument name in front of its text.
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
        ///     (<c>while (axis &lt; 0) axis += ndim</c>) is the bug this replaces in its callers: it wraps
        ///     <c>-ndim - 1</c> to <c>-1</c> and then to <c>ndim - 1</c>, so an out-of-range axis silently reduced the
        ///     last axis, and on a 0-d operand (<c>ndim == 0</c>) it never terminates.
        /// </remarks>
        [MethodImpl(Inline)]
        public static int check_and_adjust_axis(int ndims, int axis, string msgPrefix)
        {
            if (axis < -ndims || axis >= ndims)
                throw new AxisError(axis, ndims, msgPrefix);
            return axis < 0 ? axis + ndims : axis;
        }

        /// <summary>
        ///     NumPy's <c>numpy.lib.array_utils.normalize_axis_tuple</c>: every axis of <paramref name="axis"/> wrapped
        ///     into <c>[0, ndim)</c> (an AxisError naming the argument when one is out of range), then — unless
        ///     <paramref name="allow_duplicate"/> — NumPy's "repeated axis" ValueError when two name the same axis.
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
        /// <remarks>
        ///     Two former behaviours are gone: the caller's array was rewritten in place (so a caller that passed its own
        ///     axes array saw it normalized), and a repeat was silently REMOVED (<c>Distinct()</c>), which made
        ///     <c>np.moveaxis(a, [0, 0], [0, 1])</c> report a length mismatch instead of NumPy's repeat.
        /// </remarks>
        public static int[] normalize_axis_tuple(int[] axis, int ndim, object argname = null, bool allow_duplicate = false)
        {
            string name = argname?.ToString();
            var normalized = new int[axis.Length];
            // normalize_axis_index(ax, ndim, argname) for each entry, in order: the FIRST out-of-range axis is reported,
            // and every range check runs before the repeat check (NumPy's tuple comprehension, then len(set(axis))).
            for (int i = 0; i < axis.Length; i++)
                normalized[i] = check_and_adjust_axis(ndim, axis[i], name);

            if (!allow_duplicate)
            {
                for (int i = 1; i < normalized.Length; i++)
                    for (int j = 0; j < i; j++)
                        if (normalized[i] == normalized[j])
                            // `if argname:` — Python's truthiness, so an empty name gives the plain text.
                            throw new ValueError(string.IsNullOrEmpty(name) ? "repeated axis" : $"repeated axis in `{name}` argument");
            }

            return normalized;
        }

        /// <summary>
        ///     NumPy's <c>normalize_axis_tuple</c> for one integer axis: the one-element tuple of the normalized axis.
        /// </summary>
        /// <param name="axis">The un-normalized axis.</param>
        /// <param name="ndim">The rank the axis addresses.</param>
        /// <param name="argname">The argument name for the AxisError's prefix, or <c>null</c>.</param>
        /// <param name="allow_duplicate">Unused for one axis (a single axis cannot repeat); kept for NumPy's signature.</param>
        /// <returns>The normalized axis as a one-element array.</returns>
        /// <exception cref="AxisError"><paramref name="axis"/> is out of range.</exception>
        public static int[] normalize_axis_tuple(int axis, int ndim, object argname = null, bool allow_duplicate = false)
        {
            return normalize_axis_tuple(new int[] {axis}, ndim, argname, allow_duplicate);
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
            // NumPy's statements in NumPy's order: source is normalized (and repeat-checked) first, then destination,
            // and only then the lengths are compared.
            source = normalize_axis_tuple(source, nd.ndim, "source");
            destination = normalize_axis_tuple(destination, nd.ndim, "destination");
            if (source.Length != destination.Length)
                throw new ValueError("`source` and `destination` arguments must have the same number of elements");

            var order = Enumerable.Range(0, nd.ndim).Except(source).ToList();

            // for dest, src in sorted(zip(destination, source)): order.insert(dest, src)
            foreach (var (dest, src) in destination.Zip(source, (i, i1) => (dest: i, src: i1)).OrderBy(tuple => tuple.dest).ThenBy(tuple => tuple.src))
                order.Insert(dest, src);

            return Transpose(nd, order.ToArray());
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
            var ndims = nd.ndim;
            var dims = new long[ndims];
            for (int i = 0; i < ndims; i++)
                dims[i] = i;

            // check_and_adjust_axis_msg(&a1, n, axis1) then (&a2, n, axis2): NumPy names the argument in the text.
            axis1 = check_and_adjust_axis(ndims, axis1, "axis1");
            axis2 = check_and_adjust_axis(ndims, axis2, "axis2");

            dims[axis1] = axis2;
            dims[axis2] = axis1;

            // Convert long[] to int[] for Transpose (axes are limited to int range)
            var intDims = new int[ndims];
            for (int i = 0; i < ndims; i++)
                intDims[i] = (int)dims[i];
            return Transpose(nd, intDims);
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
            int n = nd.ndim;
            axis = check_and_adjust_axis(n, axis);
            if (start < 0)
                start += n;

            // NumPy's own message-only AxisError, reporting `start` AFTER the negative wrap.
            if (!(0 <= start && start < n + 1))
                throw new AxisError($"'start' arg requires {-n} <= start < {n + 1}, but {start} was passed in");

            if (axis < start)
                start -= 1;

            // `return a[...]`: a NEW view of the array, never the array object itself (NumPy's identity rule — the caller
            // may reshape or resize the result without touching the input).
            if (axis == start)
                return new NDArray(nd.Storage.Alias()) { TensorEngine = nd.TensorEngine };

            var premutes = new List<int>(n);
            for (int i = 0; i < n; i++)
                premutes.Add(i);

            premutes.RemoveAt(axis);
            premutes.Insert(start, axis);
            return Transpose(nd, premutes.ToArray());
        }

        public override NDArray Transpose(NDArray nd, int[] premute = null)
        {
            int i, n;
            var permutation = new int[nd.ndim];
            var reverse_permutation = new int[nd.ndim];
            // NumPy distinguishes axes=None (reverse all axes) from an explicit length-0
            // permutation axes=() / []: the empty permutation only matches a 0-d array,
            // and raises "axes don't match array" for any ndim >= 1. So only a null
            // `premute` means "reverse"; an empty array falls through to the length check.
            if (premute == null)
            {
                n = nd.ndim;
                for (i = 0; i < n; i++)
                {
                    permutation[i] = n - 1 - i;
                }
            }
            else
            {
                n = premute.Length;
                int[] axes = premute;
                // PyArray_Transpose: a permutation of another length is NumPy's ValueError before any axis is read.
                if (n != nd.ndim)
                    throw new ValueError("axes don't match array");

                for (i = 0; i < n; i++)
                    reverse_permutation[i] = -1;

                for (i = 0; i < n; i++)
                {
                    // check_and_adjust_axis (AxisError, no prefix), then the repeat test — per entry, in order.
                    int axis = check_and_adjust_axis(nd, axes[i]);
                    if (reverse_permutation[axis] != -1)
                        throw new ValueError("repeated axis in transpose");

                    reverse_permutation[axis] = i;
                    permutation[i] = axis;
                }
            }

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
            var shape = nd.Shape;
            var srcDims = shape.dimensions;
            var srcStrides = shape.strides;

            // Permute dimensions and strides
            var permutedDims = new long[n];
            var permutedStrides = new long[n];
            for (i = 0; i < n; i++)
            {
                permutedDims[i] = srcDims[permutation[i]];
                permutedStrides[i] = srcStrides[permutation[i]];
            }

            // Create the transposed shape via constructor (immutable)
            // IsContiguous is computed from strides and will be false for transposed arrays
            long bufSize = shape.bufferSize > 0 ? shape.bufferSize : shape.size;
            var newShape = new Shape(permutedDims, permutedStrides, shape.offset, bufSize);

            // Return an alias (view) with the permuted shape. Alias() inherits writeability from the
            // source, so transposing a read-only array (broadcast / 'r' memmap) stays read-only.
            return new NDArray(nd.Storage.Alias(newShape)) { TensorEngine = nd.TensorEngine };
        }
    }
}
