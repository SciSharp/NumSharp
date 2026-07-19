using System;
using System.Collections.Generic;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     The reversing slice (::-1) placed on every flipped axis of the cold-path indexer.
        /// </summary>
        private static readonly Slice ReversedSlice = new Slice(null, null, -1);

        /// <summary>fliplr's axis tuple — m[:, ::-1] == np.flip(m, axis: 1).</summary>
        private static readonly int[] FlipAxis1 = {1};

        /// <summary>flipud's axis tuple — m[::-1, ...] == np.flip(m, axis: 0).</summary>
        private static readonly int[] FlipAxis0 = {0};

        /// <summary>
        ///     Reverse the order of elements in an array along the given axis.
        ///     The shape of the array is preserved, but the elements are reordered.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">
        ///     Axis along which to flip over. The default, null, will flip over all of the axes of the
        ///     input array. If axis is negative it counts from the last to the first axis.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the entries of axis reversed. Since a view is returned,
        ///     this operation is done in constant time.
        /// </returns>
        /// <exception cref="AxisError">When <paramref name="axis"/> is out of bounds for the array's dimensions.</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flip.html</remarks>
        public static NDArray flip(NDArray m, int? axis = null)
        {
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (axis == null)
                return FlipAxes(m, null);

            // normalize_axis_index: AxisError reports the ORIGINAL axis value.
            int ndim = m.ndim;
            int ax = axis.Value;
            int adjusted = ax >= 0 ? ax : ndim + ax;
            if (adjusted < 0 || adjusted >= ndim)
                throw new AxisError(ax, ndim);

            return FlipAxes(m, new[] {adjusted});
        }

        /// <summary>
        ///     Reverse the order of elements in an array along the given axes.
        ///     The shape of the array is preserved, but the elements are reordered.
        /// </summary>
        /// <param name="m">Input array.</param>
        /// <param name="axis">
        ///     Axes along which to flip over — NumPy's tuple-of-ints form; flipping is performed on all of
        ///     the specified axes. Negative axes count from the last to the first axis. An empty array
        ///     flips no axis (returns an unreversed view of the whole array); null flips all axes.
        /// </param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the entries of the given axes reversed. Since a view is
        ///     returned, this operation is done in constant time.
        /// </returns>
        /// <exception cref="AxisError">When any axis is out of bounds for the array's dimensions.</exception>
        /// <exception cref="ValueError">When an axis is repeated ("repeated axis").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flip.html</remarks>
        public static NDArray flip(NDArray m, int[] axis)
        {
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (axis == null)
                return FlipAxes(m, null);

            int ndim = m.ndim;

            // normalize_axis_tuple: normalize the whole tuple first (AxisError reports the ORIGINAL
            // axis value), only then reject duplicates — NumPy raises AxisError for (0, 0, 5).
            var normalized = new int[axis.Length];
            for (int i = 0; i < axis.Length; i++)
            {
                int ax = axis[i];
                int adjusted = ax >= 0 ? ax : ndim + ax;
                if (adjusted < 0 || adjusted >= ndim)
                    throw new AxisError(ax, ndim);
                normalized[i] = adjusted;
            }

            var seen = new HashSet<int>();
            for (int i = 0; i < normalized.Length; i++)
                if (!seen.Add(normalized[i]))
                    throw new ValueError("repeated axis");

            return FlipAxes(m, normalized);
        }

        /// <summary>
        ///     Reverse the order of elements along axis 1 (left/right).
        ///     For a 2-D array, this flips the entries in each row in the left/right direction.
        ///     Columns are preserved, but appear in a different order than before.
        /// </summary>
        /// <param name="m">Input array, must be at least 2-D.</param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the columns reversed — equivalent to m[:, ::-1]
        ///     or np.flip(m, axis: 1). Since a view is returned, this operation is done in constant time.
        /// </returns>
        /// <exception cref="ValueError">When <paramref name="m"/> is less than 2-d ("Input must be &gt;= 2-d.").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html</remarks>
        public static NDArray fliplr(NDArray m)
        {
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (m.ndim < 2)
                throw new ValueError("Input must be >= 2-d.");

            return FlipAxes(m, FlipAxis1);
        }

        /// <summary>
        ///     Reverse the order of elements along axis 0 (up/down).
        ///     For a 2-D array, this flips the entries in each column in the up/down direction.
        ///     Rows are preserved, but appear in a different order than before.
        /// </summary>
        /// <param name="m">Input array, must be at least 1-D.</param>
        /// <returns>
        ///     A view of <paramref name="m"/> with the rows reversed — equivalent to m[::-1, ...]
        ///     or np.flip(m, axis: 0). Since a view is returned, this operation is done in constant time.
        /// </returns>
        /// <exception cref="ValueError">When <paramref name="m"/> is less than 1-d ("Input must be &gt;= 1-d.").</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.flipud.html</remarks>
        public static NDArray flipud(NDArray m)
        {
            if (m is null)
                throw new ArgumentNullException(nameof(m));

            if (m.ndim < 1)
                throw new ValueError("Input must be >= 1-d.");

            return FlipAxes(m, FlipAxis0);
        }

        /// <summary>
        ///     Builds the flipped view: reversing an axis is stride negation plus moving the base
        ///     offset to the last element along that axis — O(ndim), no data movement, exactly the
        ///     view NumPy's m[..., ::-1, ...] basic indexing produces.
        /// </summary>
        /// <param name="m">Input array (any layout: contiguous, sliced, transposed, broadcast).</param>
        /// <param name="axes">Normalized (non-negative, unique) axes to reverse; null reverses all axes.</param>
        private static NDArray FlipAxes(NDArray m, int[] axes)
        {
            var shape = m.Shape;
            int ndim = m.ndim;

            // Cold path — 0-d (the empty indexer m[()] yields the scalar back) and size-0 arrays
            // (nothing to reverse) go through the general slice machinery.
            if (ndim == 0 || shape.size == 0)
            {
                var indexer = new Slice[ndim];
                for (int i = 0; i < ndim; i++)
                    indexer[i] = Slice.All;
                if (axes == null)
                    for (int i = 0; i < ndim; i++)
                        indexer[i] = ReversedSlice;
                else
                    for (int i = 0; i < axes.Length; i++)
                        indexer[axes[i]] = ReversedSlice;
                return m[indexer];
            }

            var srcDims = shape.dimensions;
            var srcStrides = shape.strides;

            var dims = new long[ndim];
            var strides = new long[ndim];
            for (int i = 0; i < ndim; i++)
            {
                dims[i] = srcDims[i];
                strides[i] = srcStrides[i];
            }

            // size > 0 guarantees every dim >= 1, so (dim - 1) never underflows the offset.
            long offset = shape.offset;
            if (axes == null)
            {
                for (int i = 0; i < ndim; i++)
                {
                    offset += srcStrides[i] * (srcDims[i] - 1);
                    strides[i] = -srcStrides[i];
                }
            }
            else
            {
                for (int i = 0; i < axes.Length; i++)
                {
                    int ax = axes[i];
                    offset += srcStrides[ax] * (srcDims[ax] - 1);
                    strides[ax] = -srcStrides[ax];
                }
            }

            long bufferSize = shape.bufferSize > 0 ? shape.bufferSize : shape.size;
            var flipped = new Shape(dims, strides, offset, bufferSize);

            // Alias() inherits writeability from the source, so flipping a read-only array
            // (broadcast / 'r' memmap) stays read-only — same pattern as Transpose.
            return new NDArray(m.Storage.Alias(flipped)) {TensorEngine = m.TensorEngine};
        }
    }
}
