using System;

namespace NumSharp
{
    public static partial class np
    {
        public static partial class linalg
        {
            /// <summary>
            ///     Matrix product — the Array-API spelling of <see cref="np.matmul"/>.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.matmul.html
            ///     <para>
            ///     This one really is the same operation as its main-namespace twin, unlike the rest
            ///     of the forms in this file.
            ///     </para>
            /// </remarks>
            public static NDArray matmul(NDArray x1, NDArray x2) => np.matmul(x1, x2);

            /// <summary>
            ///     Outer product of two VECTORS.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.outer.html
            ///     <para>
            ///     Not a synonym for <see cref="np.outer"/>: that one flattens whatever it is given,
            ///     while this rejects anything but 1-D. <c>np.outer(ones((2,3)), ones(3))</c> is a
            ///     <c>(6,3)</c> array; <c>np.linalg.outer</c> of the same pair raises.
            ///     </para>
            /// </remarks>
            public static NDArray outer(NDArray x1, NDArray x2)
            {
                if (x1.ndim != 1 || x2.ndim != 1)
                    throw new ValueError(
                        $"Input arrays must be one-dimensional, but they are x1.ndim={x1.ndim} and x2.ndim={x2.ndim}.");

                return np.outer(x1, x2);
            }

            /// <summary>
            ///     Tensor contraction — the Array-API spelling of <see cref="np.tensordot(NDArray, NDArray, int)"/>.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html</remarks>
            public static NDArray tensordot(NDArray x1, NDArray x2, int axes = 2)
                => np.tensordot(x1, x2, axes);

            /// <inheritdoc cref="tensordot(NDArray, NDArray, int)"/>
            public static NDArray tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB)
                => np.tensordot(x1, x2, axesA, axesB);

            /// <summary>
            ///     Sum along the diagonals of the LAST TWO axes.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.trace.html
            ///     <para>
            ///     <b>The axes differ from <see cref="np.trace"/>'s</b>, which defaults to the FIRST
            ///     two — so on a <c>(2,3,3)</c> stack this returns shape <c>(2,)</c> (one trace per
            ///     matrix) where <c>np.trace</c> returns <c>(3,)</c>. Neither is wrong; they are
            ///     different functions.
            ///     </para>
            /// </remarks>
            public static NDArray trace(NDArray x, int offset = 0, Type dtype = null)
                => np.trace(x, offset, -2, -1, dtype);

            /// <summary>
            ///     Diagonals of the LAST TWO axes.
            /// </summary>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.diagonal.html
            ///     <para>
            ///     As with <see cref="trace"/>, the axes differ from <see cref="np.diagonal"/>'s
            ///     first-two default: a <c>(2,3,3)</c> stack gives <c>(2,3)</c> here and
            ///     <c>(3,2)</c> there.
            ///     </para>
            /// </remarks>
            public static NDArray diagonal(NDArray x, int offset = 0)
                => np.diagonal(x, offset, -2, -1);

            /// <summary>
            ///     Transposes the last two axes — the Array-API spelling of
            ///     <see cref="np.matrix_transpose"/>. An O(1) view.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.matrix_transpose.html</remarks>
            public static NDArray matrix_transpose(NDArray x) => np.matrix_transpose(x);

            /// <summary>
            ///     Vector dot product over <paramref name="axis"/>, conjugating the first operand.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.vecdot.html</remarks>
            public static NDArray vecdot(NDArray x1, NDArray x2, int axis = -1)
                => np.vecdot(x1, x2, axis: axis);

            /// <summary>
            ///     Cross product of 3-element vectors.
            /// </summary>
            /// <param name="axis">The axis holding the three components. Defaults to the last.</param>
            /// <remarks>
            ///     https://numpy.org/doc/stable/reference/generated/numpy.linalg.cross.html
            ///     <para>
            ///     <b>Stricter than <see cref="np.cross"/>:</b> the Array-API form accepts 3-vectors
            ///     ONLY, over a single <paramref name="axis"/>, where the main-namespace
            ///     <see cref="np.cross"/> also takes the 2-vector form (deprecated in NumPy 2.0) and a
            ///     separate <c>axisa</c>/<c>axisb</c>/<c>axisc</c> per operand.
            ///     </para>
            /// </remarks>
            public static NDArray cross(NDArray x1, NDArray x2, int axis = -1)
            {
                int a1 = NormalizeAxis(axis, x1.ndim);
                int a2 = NormalizeAxis(axis, x2.ndim);

                if (x1.shape[a1] != 3 || x2.shape[a2] != 3)
                    throw new ValueError(
                        "Both input arrays must be (arrays of) 3-dimensional vectors, but they are " +
                        $"{x1.shape[a1]} and {x2.shape[a2]} dimensional instead.");

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
            private static NDArray Component(NDArray v, int index) => v[Slice.Index(index)];
        }
    }
}
