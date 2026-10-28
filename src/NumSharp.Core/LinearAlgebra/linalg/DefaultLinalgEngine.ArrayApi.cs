using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's two statements: the rank check (<c>Input arrays must be one-dimensional, but they are x1.ndim={n} and
        ///     x2.ndim={m}.</c>), then <c>_core_outer(x1, x2, out=None)</c>, which is <see cref="np.outer"/> (on its own it would
        ///     flatten anything).
        /// </remarks>
        public override NDArray Outer(NDArray x1, NDArray x2)
        {
            if (x1.ndim != 1 || x2.ndim != 1)
                throw new ValueError(
                    $"Input arrays must be one-dimensional, but they are x1.ndim={x1.ndim} and x2.ndim={x2.ndim}.");

            return np.outer(x1, x2);
        }

        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's two statements: <c>if x1.shape[axis] != 3 or x2.shape[axis] != 3: raise ValueError(...)</c> (Python's
        ///     tuple indexing, so an axis out of range is an IndexError) and <c>return _core_cross(x1, x2, axis=axis)</c>,
        ///     which is <see cref="np.cross"/>. np.cross promotes the two dtypes STRONGLY (an int8 matrix against an int32 row
        ///     is int32, a bool one int32 too) and fills a fresh array whose component axis it moves to the requested axis
        ///     last, so the result is a view in NumPy's layout (C order for the last axis, F order for the first axis of a
        ///     matrix).
        /// </remarks>
        [NDScoped] // reclaims np.cross's temporaries (its own scope) - the call is the body's one computation
        public override NDArray Cross(NDArray x1, NDArray x2, int axis = -1)
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

        // The four Array-API forms below are NumPy's one-statement wrappers of the main namespace's functions
        // (`return _core_matmul(x1, x2)`, ...): each override is that call and nothing else, so it opens no scope (there
        // is no temporary to reclaim; the result is the callee's own). The np.linalg facades apply the callee's class
        // boundary before they call these, so a numpy.matrix operand arrives inside the facade's region.

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_matmul(x1, x2)</c>: <see cref="np.matmul"/> with every keyword
        ///     at its default.</remarks>
        public override NDArray Matmul(NDArray x1, NDArray x2) => np.matmul(x1, x2);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_tensordot(x1, x2, axes=axes)</c> with an integer
        ///     <paramref name="axes"/>: <see cref="np.tensordot(NDArray, NDArray, int)"/>.</remarks>
        public override NDArray Tensordot(NDArray x1, NDArray x2, int axes = 2) => np.tensordot(x1, x2, axes);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_tensordot(x1, x2, axes=axes)</c> with the pair of axis lists:
        ///     <see cref="np.tensordot(NDArray, NDArray, int[], int[])"/>.</remarks>
        public override NDArray Tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB)
            => np.tensordot(x1, x2, axesA, axesB);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)</c>:
        ///     <see cref="np.trace"/> over the last two axes.</remarks>
        public override NDArray Trace(NDArray x, int offset = 0, DType dtype = null) => np.trace(x, offset, -2, -1, dtype);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_vecdot(x1, x2, axis=axis)</c>: <see cref="np.vecdot"/> with
        ///     <paramref name="axis"/> and every other keyword at its default.</remarks>
        public override NDArray Vecdot(NDArray x1, NDArray x2, int axis = -1) => np.vecdot(x1, x2, axis: axis);
    }
}
