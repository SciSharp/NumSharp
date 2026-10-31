using System;

namespace NumSharp.Backends
{
    public partial class DefaultLinalgEngine
    {
        /// <inheritdoc/>
        /// <remarks>
        ///     NumPy's two statements: the rank check (<c>Input arrays must be one-dimensional, but they are x1.ndim={n} and
        ///     x2.ndim={m}.</c>), then <c>_core_outer(x1, x2, out=None)</c>, which is <see cref="np.outer"/> (on its own it would
        ///     flatten anything). np.outer is still a composition in its facade, with no engine member to call, so this is the
        ///     engine's one call up into np.* left in this file (pinned in ratchet R2's allowlist): it converts when the
        ///     np-layer dtype plan's phase P8 makes outer a <see cref="TensorEngine"/> member.
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
        ///     which is <see cref="np.cross"/>, computed by the member its facade dispatches to,
        ///     <see cref="TensorEngine.Cross(NDArray, NDArray, int, int, int, int?)"/> of <paramref name="x1"/>'s engine.
        ///     The core cross promotes the two dtypes STRONGLY (an int8 matrix against an int32 row is int32, a bool one int32
        ///     too) and fills a fresh array whose component axis it moves to the requested axis last, so the result is a view
        ///     in NumPy's layout (C order for the last axis, F order for the first axis of a matrix).
        /// </remarks>
        [NDScoped] // reclaims the core cross's temporaries (its own scope) - the call is the body's one computation
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

            // `return _core_cross(x1, x2, axis=axis)`: np.cross's engine member with the call np.cross makes for it -
            // axisa / axisb / axisc at their -1 defaults and axis set, which the member's `if axis is not None` then
            // spreads over the three (its own text refusal, its broadcast and its axisc check included).
            return x1.TensorEngine.Cross(x1, x2, -1, -1, -1, axis);
        }

        // The four Array-API forms below are NumPy's one-statement wrappers of the main namespace's functions
        // (`return _core_matmul(x1, x2)`, ...): each override is that call and nothing else, so it opens no scope (there
        // is no temporary to reclaim; the result is the callee's own). The call is the member the main namespace's facade
        // dispatches to (the operand engine's TensorEngine member), not the facade: the np.linalg facades apply the callee's
        // class boundary before they call these, so a numpy.matrix operand arrives inside the facade's region and the
        // facade's rule would only be skipped there anyway. tensordot is the exception: np.tensordot's statements are
        // still its facade's own (no engine member until the np-layer dtype plan's phase P8), so its two forms keep the
        // facade call, pinned in ratchet R2's allowlist.

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_matmul(x1, x2)</c>: the member <see cref="np.matmul"/> dispatches
        ///     to, <see cref="TensorEngine.Matmul(NDArray, NDArray, NDArray, int[][], int?, bool?, DType, string, char)"/>
        ///     of <paramref name="x1"/>'s engine, with every keyword at np.matmul's default (no out, axes, axis, keepdims or
        ///     dtype; <c>casting="same_kind"</c>, <c>order='K'</c>; <see cref="CoreMatmul"/>). A null
        ///     <paramref name="x1"/> fails at the dispatch with a <see cref="NullReferenceException"/>, as it did through
        ///     np.matmul.</remarks>
        public override NDArray Matmul(NDArray x1, NDArray x2) => CoreMatmul(x1, x2);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_tensordot(x1, x2, axes=axes)</c> with an integer
        ///     <paramref name="axes"/>: <see cref="np.tensordot(NDArray, NDArray, int)"/>, whose statements are its facade's
        ///     own (no engine member yet), so the call stays a facade call (pinned in ratchet R2's allowlist until phase P8
        ///     makes tensordot a <see cref="TensorEngine"/> member).</remarks>
        public override NDArray Tensordot(NDArray x1, NDArray x2, int axes = 2) => np.tensordot(x1, x2, axes);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_tensordot(x1, x2, axes=axes)</c> with the pair of axis lists:
        ///     <see cref="np.tensordot(NDArray, NDArray, int[], int[])"/>, a facade call for the reason the integer form
        ///     gives.</remarks>
        public override NDArray Tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB)
            => np.tensordot(x1, x2, axesA, axesB);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_trace(x, offset, axis1=-2, axis2=-1, dtype=dtype)</c>: the member
        ///     <see cref="np.trace"/> dispatches to, <see cref="TensorEngine.Trace(NDArray, int, int, int, DType, NDArray)"/>,
        ///     over the last two axes. A null <paramref name="x"/> is dispatched to the default engine (np.EngineOf's rule)
        ///     because the member itself refuses it with np.trace's <see cref="ArgumentNullException"/>, which np.linalg.trace
        ///     reported through np.trace before, rather than failing at the dispatch.</remarks>
        public override NDArray Trace(NDArray x, int offset = 0, DType dtype = null)
            => (x?.TensorEngine ?? BackendFactory.GetEngine()).Trace(x, offset, -2, -1, dtype, null);

        /// <inheritdoc/>
        /// <remarks>NumPy's one statement <c>return _core_vecdot(x1, x2, axis=axis)</c>: the member
        ///     <see cref="np.vecdot"/> dispatches to,
        ///     <see cref="TensorEngine.Vecdot(NDArray, NDArray, NDArray, int[][], int?, bool, DType)"/> of
        ///     <paramref name="x1"/>'s engine, with <paramref name="axis"/> and every other keyword at np.vecdot's default
        ///     (no out, axes or dtype; keepdims false).</remarks>
        public override NDArray Vecdot(NDArray x1, NDArray x2, int axis = -1)
            => x1.TensorEngine.Vecdot(x1, x2, null, null, axis, false, null);
    }
}
