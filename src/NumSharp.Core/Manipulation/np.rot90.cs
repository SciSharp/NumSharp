using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Rotate an array by 90 degrees in the plane specified by axes.
        ///
        ///     Rotation direction is from the first towards the second axis. This means that
        ///     for a 2-D array with the default <paramref name="k"/> and <paramref name="axes"/>
        ///     the rotation will be counterclockwise.
        /// </summary>
        /// <param name="m">Array of two or more dimensions.</param>
        /// <param name="k">Number of times the array is rotated by 90 degrees.</param>
        /// <param name="axes">
        ///     The array is rotated in the plane defined by the axes. Axes must be different.
        ///     Defaults to <c>(0, 1)</c>.
        /// </param>
        /// <returns>A rotated view of <paramref name="m"/>.</returns>
        /// <remarks>
        ///     Port of NumPy's <c>numpy.rot90</c> (<c>numpy/lib/_function_base_impl.py</c>): a pure
        ///     composition of axis flips and a transpose, so the result is always a view that shares
        ///     memory with <paramref name="m"/> (read-only when the source is, e.g. a broadcast view).
        ///     <para><c>rot90(m, k=1, axes=(1, 0))</c> is the reverse of <c>rot90(m, k=1, axes=(0, 1))</c>.</para>
        ///     https://numpy.org/doc/stable/reference/generated/numpy.rot90.html
        /// </remarks>
        public static NDArray rot90(NDArray m, int k = 1, int[] axes = null)
        {
            axes ??= new[] {0, 1};

            if (axes.Length != 2)
                throw new ArgumentException("len(axes) must be 2.");

            int ndim = m.ndim;

            // NumPy checks "same axis" before the range check, and the |a0-a1|==ndim clause
            // catches a pair that names one physical axis through mixed signs (e.g. (0, -2) on
            // ndim=2, or (0, 3) on ndim=3) — which therefore reports "Axes must be different"
            // rather than "out of range".
            if (axes[0] == axes[1] || Math.Abs(axes[0] - axes[1]) == ndim)
                throw new ArgumentException("Axes must be different.");

            if (axes[0] >= ndim || axes[0] < -ndim
                || axes[1] >= ndim || axes[1] < -ndim)
                throw new ArgumentException(
                    $"Axes=({axes[0]}, {axes[1]}) out of range for array of ndim={ndim}.");

            // Python-style modulo: always in [0, 3], even for negative k.
            k = ((k % 4) + 4) % 4;

            if (k == 0)
                // NumPy returns m[:] — a full view that shares memory with m. For an empty array
                // return a fresh same-shape array instead (consistent with how the k=1/2/3 paths and
                // DefaultEngine.Transpose treat size==0): aliasing a zero-length offset view yields a
                // shape that downstream buffer ops can't materialize, and there is no data to share.
                return m.Shape.size == 0
                    ? new NDArray(m.dtype, m.Shape.dimensions)
                    : new NDArray(m.Storage.Alias(m.Shape)) {TensorEngine = m.TensorEngine};

            // Normalize the (validated) axes to [0, ndim) for the view ops below, matching
            // Python's negative indexing of both flip(m, axis) and axes_list[axis].
            int ax0 = axes[0] < 0 ? axes[0] + ndim : axes[0];
            int ax1 = axes[1] < 0 ? axes[1] + ndim : axes[1];

            if (k == 2)
                return FlipAxisView(FlipAxisView(m, ax0), ax1);

            // axes_list = identity permutation with ax0 and ax1 interchanged.
            var axesList = new int[ndim];
            for (int i = 0; i < ndim; i++)
                axesList[i] = i;
            (axesList[ax0], axesList[ax1]) = (axesList[ax1], axesList[ax0]);

            if (k == 1)
                return transpose(FlipAxisView(m, ax1), axesList);

            // k == 3
            return FlipAxisView(transpose(m, axesList), ax1);
        }

        /// <summary>
        ///     Reverses <paramref name="m"/> along a single <paramref name="axis"/> and returns a
        ///     view — the NumPy <c>flip(m, axis)</c> primitive for one axis. The axis' stride is
        ///     negated and the offset advanced to that axis' last element; no data is copied.
        /// </summary>
        private static NDArray FlipAxisView(NDArray m, int axis)
        {
            var shape = m.Shape;

            // Empty arrays cannot be aliased through Alias (no backing buffer); build a fresh
            // empty array with the same shape — flipping is a no-op on zero elements anyway.
            if (shape.size == 0)
                return new NDArray(m.dtype, shape.dimensions);

            var dims = (long[])shape.dimensions.Clone();
            var strides = (long[])shape.strides.Clone();
            long offset = shape.offset;

            // Advance to the last element along `axis`, then reverse its stride. A stride of 0
            // (broadcast axis) negates to 0 and the offset is unchanged — reversing a stretched
            // axis is a no-op, exactly as in NumPy.
            offset += strides[axis] * (dims[axis] - 1);
            strides[axis] = -strides[axis];

            long bufSize = shape.bufferSize > 0 ? shape.bufferSize : shape.size;
            var flipped = new Shape(dims, strides, offset, bufSize);

            // Alias inherits read-only from the source storage (broadcast / 'r' memmap).
            return new NDArray(m.Storage.Alias(flipped)) {TensorEngine = m.TensorEngine};
        }
    }
}
