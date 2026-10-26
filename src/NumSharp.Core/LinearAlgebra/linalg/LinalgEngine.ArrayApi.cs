using System;

namespace NumSharp
{
    internal static partial class LinalgEngine
    {
        /// <summary>
        ///     The body of <c>np.linalg.outer</c>: NumPy's rank check, then <c>np.outer</c> (which would flatten anything).
        /// </summary>
        /// <param name="x1">The first vector.</param>
        /// <param name="x2">The second vector.</param>
        /// <returns><c>np.outer(x1, x2)</c>, a fresh <c>(len(x1), len(x2))</c> array.</returns>
        /// <exception cref="ValueError">Either operand is not one-dimensional: <c>Input arrays must be one-dimensional,
        ///     but they are x1.ndim={n} and x2.ndim={m}.</c></exception>
        internal static NDArray Outer(NDArray x1, NDArray x2)
        {
            if (x1.ndim != 1 || x2.ndim != 1)
                throw new ValueError(
                    $"Input arrays must be one-dimensional, but they are x1.ndim={x1.ndim} and x2.ndim={x2.ndim}.");

            return np.outer(x1, x2);
        }

        /// <summary>
        ///     The body of <c>np.linalg.cross</c> past the class boundary, NumPy's two statements: the length check on the
        ///     component axis, then <c>_core_cross(x1, x2, axis=axis)</c>, which is <see cref="np.cross"/>.
        /// </summary>
        /// <param name="x1">The first operand (plain: the facade applied the class boundary).</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="axis">The axis holding the three components, in both operands and in the result.</param>
        /// <returns>np.cross's result: <c>promote_types(x1.dtype, x2.dtype)</c>, computed into a C-ordered array with the
        ///     component axis last and returned as <c>moveaxis(cp, -1, axis)</c>, a view of it.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range for either operand ("tuple index out of
        ///     range": NumPy indexes <c>x1.shape[axis]</c> before anything else).</exception>
        /// <exception cref="ValueError">The component axis of either operand does not have length 3, or the other axes
        ///     do not broadcast (np.cross's refusal).</exception>
        /// <exception cref="TypeError">Either operand holds text: the multiply's no-loop refusal, named with the promoted
        ///     descriptor on both sides.</exception>
        /// <remarks>
        ///     NumPy's statements are <c>if x1.shape[axis] != 3 or x2.shape[axis] != 3: raise ValueError(...)</c> and
        ///     <c>return _core_cross(x1, x2, axis=axis)</c>. np.cross promotes the two dtypes STRONGLY (an int8 matrix
        ///     against an int32 row is int32, a bool one int32 too) and fills a fresh array whose component axis it moves
        ///     to <paramref name="axis"/> last, so the result is a view in NumPy's layout (C order for the last axis, F
        ///     order for the first axis of a matrix).
        /// </remarks>
        [NDScoped] // reclaims np.cross's temporaries (its own scope) - the call is the body's one computation
        internal static NDArray Cross(NDArray x1, NDArray x2, int axis)
        {
            // `if x1.shape[axis] != 3 or x2.shape[axis] != 3` is NumPy's first statement: Python tuple indexing, so an
            // axis out of range for either operand is an IndexError (x1's first), not an AxisError.
            int a1 = ShapeIndex(axis, x1.ndim);
            int a2 = ShapeIndex(axis, x2.ndim);

            if (x1.Shape.dimensions[a1] != 3 || x2.Shape.dimensions[a2] != 3)
                throw new ValueError(
                    "Both input arrays must be (arrays of) 3-dimensional vectors, but they are " +
                    $"{x1.Shape.dimensions[a1]} and {x2.Shape.dimensions[a2]} dimensional instead.");

            // `return _core_cross(x1, x2, axis=axis)`: np.cross with axisa = axisb = axisc = axis (its own text refusal,
            // its broadcast and its axisc check included).
            return np.cross(x1, x2, axis: axis);
        }
    }
}
