using System;

namespace NumSharp
{
    public static partial class np
    {
        /// <summary>
        ///     Join a sequence of arrays along a new axis.
        ///     The axis parameter specifies the index of the new axis in the dimensions of the result.
        ///     For example, if axis=0 it will be the first dimension and if axis=-1 it will be the last dimension.
        /// </summary>
        /// <param name="arrays">Each array must have the same shape.</param>
        /// <param name="axis">The axis in the result array along which the input arrays are stacked.</param>
        /// <returns>The stacked array has one more dimension than the input arrays: stacking <c>n</c> 0-d arrays gives
        ///     shape <c>(n,)</c>.</returns>
        /// <exception cref="ArgumentNullException"><paramref name="arrays"/> is null.</exception>
        /// <exception cref="ValueError"><paramref name="arrays"/> is empty (<c>need at least one array to stack</c>), or the
        ///     arrays' shapes differ (<c>all input arrays must have the same shape</c>, checked before the axis).</exception>
        /// <exception cref="AxisError"><paramref name="axis"/> is outside the RESULT's rank (the operands' rank plus one;
        ///     <c>axis 3 is out of bounds for array of dimension 3</c>, the axis as given).</exception>
        /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.stack.html
        /// <para>NumPy's statements: <c>if len({arr.shape for arr in arrays}) != 1: raise ValueError(...)</c>, then each
        /// array gets a new axis at <paramref name="axis"/> (normalized against <c>ndim + 1</c>) and the expanded arrays are
        /// concatenated along it. A 0-d operand is expanded as it is — to <c>(1,)</c> — so scalars stack to <c>(n,)</c>
        /// (running them through <c>atleast_1d</c> first, as this method once did, gave <c>(n, 1)</c>). The caller's array
        /// of operands is not modified.</para></remarks>
        /// <exception cref="ValueError">A <c>numpy.matrix</c> operand whose result cannot be a matrix: more than two axes longer than 1 (<c>shape too large to be a matrix.</c>).</exception>
        public static NDArray stack(NDArray[] arrays, int axis = 0)
        {
            // a numpy.matrix operand comes back as a matrix, as from NumPy (NDMatrixBoundary)
            if (NDMatrixBoundary.Sees(arrays))
                return NDMatrixBoundary.Wrapped((arrays, axis), static s => stack(s.arrays, s.axis));
            if (arrays == null)
                throw new ArgumentNullException(nameof(arrays));

            // `if not arrays: raise ValueError('need at least one array to stack')`.
            if (arrays.Length == 0)
                throw new ValueError("need at least one array to stack");

            // `shapes = {arr.shape for arr in arrays}; if len(shapes) != 1: raise` — every shape, before the axis.
            var first = arrays[0];
            for (int i = 1; i < arrays.Length; i++)
                if (!arrays[i].Shape.dimensions.AsSpan().SequenceEqual(first.Shape.dimensions))
                    throw new ValueError("all input arrays must have the same shape");

            // `axis = normalize_axis_index(axis, result_ndim)` — NumPy's AxisError against the RESULT's rank, raised once
            // here (the expansion below then sees an axis in range).
            axis = Backends.DefaultEngine.check_and_adjust_axis(first.ndim + 1, axis);

            // `expanded_arrays = [arr[sl] for arr in arrays]` — the new axis at `axis` of the (ndim + 1)-D result.
            var expanded = new NDArray[arrays.Length];
            for (int i = 0; i < arrays.Length; i++)
                expanded[i] = np.expand_dims(arrays[i], axis);

            return np.concatenate(expanded, axis);
        }
    }
}
