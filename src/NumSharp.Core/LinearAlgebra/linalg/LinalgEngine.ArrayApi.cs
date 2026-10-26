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
        ///     The body of <c>np.linalg.cross</c> past the class boundary: NumPy's length check on the component axis,
        ///     then the cross product of the 3-vectors, the component axis at <paramref name="axis"/> of the result.
        /// </summary>
        /// <param name="x1">The first operand (plain: the facade applied the class boundary).</param>
        /// <param name="x2">The second operand.</param>
        /// <param name="axis">The axis holding the three components, in both operands and in the result.</param>
        /// <returns>The cross product, the component axis at <paramref name="axis"/> of the broadcast result.</returns>
        /// <exception cref="IndexError"><paramref name="axis"/> is out of range for either operand ("tuple index out of
        ///     range": NumPy indexes <c>x1.shape[axis]</c> before anything else).</exception>
        /// <exception cref="ValueError">The component axis of either operand does not have length 3.</exception>
        /// <exception cref="TypeError">Either operand holds text: the multiply's no-loop refusal, named with the promoted
        ///     descriptor on both sides.</exception>
        [NDScoped] // reclaims the moveaxis views, the six component products and the pre-moveaxis stack
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

            // NumPy then calls np.cross, which casts both operands to their promoted dtype and multiplies components
            // into the result: with text the multiply has no loop, named with the promoted descriptor on both sides
            // (<U2 beside float64 promotes to <U32).
            if (Backends.Strings.NDStringUfuncs.IsText(x1) || Backends.Strings.NDStringUfuncs.IsText(x2))
            {
                var promoted = DTypePromotion.PromoteTypes(x1.dtype, x2.dtype);
                throw Backends.Strings.NDStringUfuncs.NoLoop("multiply", promoted, promoted);
            }

            // Work with the component axis FIRST, so each of the three components is a plain
            // leading-axis index that drops the axis; taking them from the last axis instead
            // needs an ellipsis, and that leaves a length-1 axis behind on a bare 3-vector.
            var a = np.moveaxis(x1, a1, 0);
            var b = np.moveaxis(x2, a2, 0);

            var components = new[]
            {
                np.subtract(np.multiply(Component(a, 1), Component(b, 2)),
                    np.multiply(Component(a, 2), Component(b, 1))),
                np.subtract(np.multiply(Component(a, 2), Component(b, 0)),
                    np.multiply(Component(a, 0), Component(b, 2))),
                np.subtract(np.multiply(Component(a, 0), Component(b, 1)),
                    np.multiply(Component(a, 1), Component(b, 0)))
            };

            // expand_dims + concatenate rather than np.stack: stack runs its operands through
            // atleast_1d first, so three 0-D components — which is what a bare 3-vector's
            // components are — come out as (3,1) instead of (3,).
            for (int i = 0; i < components.Length; i++)
                components[i] = np.expand_dims(components[i], 0);

            var stacked = np.concatenate(components, 0);
            return np.moveaxis(stacked, 0, NormalizeAxis(axis, stacked.ndim));
        }

        /// <summary>One component of a vector array whose component axis is FIRST.</summary>
        /// <param name="v">The vectors, component axis first.</param>
        /// <param name="index">The component (0, 1 or 2).</param>
        /// <returns>A view of that component across every vector (the component axis dropped).</returns>
        private static NDArray Component(NDArray v, int index) => v[Slice.Index(index)];
    }
}
