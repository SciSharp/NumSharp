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
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray matmul(NDArray x1, NDArray x2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Matmul(x1, x2);
            }

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
            /// <param name="x1">The first vector.</param>
            /// <param name="x2">The second vector.</param>
            /// <returns>The <c>(len(x1), len(x2))</c> outer product, a fresh array.</returns>
            /// <exception cref="ValueError">Either operand is not one-dimensional.</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray outer(NDArray x1, NDArray x2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Outer(x1, x2);
            }

            /// <summary>
            ///     Tensor contraction — the Array-API spelling of <see cref="np.tensordot(NDArray, NDArray, int)"/>.
            /// </summary>
            /// <remarks>https://numpy.org/doc/stable/reference/generated/numpy.linalg.tensordot.html</remarks>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray tensordot(NDArray x1, NDArray x2, int axes = 2)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Tensordot(x1, x2, axes);
            }

            /// <inheritdoc cref="tensordot(NDArray, NDArray, int)"/>
            public static NDArray tensordot(NDArray x1, NDArray x2, int[] axesA, int[] axesB)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Tensordot(x1, x2, axesA, axesB);
            }

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
            /// <exception cref="ArgumentNullException"><paramref name="x"/> is null.</exception>
            public static NDArray trace(NDArray x, int offset = 0, DType dtype = null)
            {
                if (x is null) throw new ArgumentNullException(nameof(x));
                return x.TensorEngine.Linalg.Trace(x, offset, dtype);
            }

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
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            public static NDArray vecdot(NDArray x1, NDArray x2, int axis = -1)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Vecdot(x1, x2, axis);
            }

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
            ///     separate <c>axisa</c>/<c>axisb</c>/<c>axisc</c> per operand. Past the length check it IS
            ///     <see cref="np.cross"/> (NumPy's <c>_core_cross(x1, x2, axis=axis)</c>): the dtypes promote
            ///     strongly (<c>promote_types</c>), and the result is a view of a fresh C-ordered array whose
            ///     component axis is moved to <paramref name="axis"/>.
            ///     </para>
            /// </remarks>
            /// <param name="x1">The first operand.</param>
            /// <param name="x2">The second operand.</param>
            /// <returns>The cross product, the component axis at <paramref name="axis"/> of the broadcast result.
            ///     A <c>numpy.char.chararray</c> operand gives a plain array: NumPy converts it with <c>asarray</c> (or
            ///     ends with <c>np.concatenate</c>), which drops the class.</returns>
            /// <exception cref="IndexError"><paramref name="axis"/> is out of range for either operand ("tuple index out of
            ///     range": NumPy indexes <c>x1.shape[axis]</c> before anything else).</exception>
            /// <exception cref="ValueError">The component axis of either operand does not have length 3, or the other axes
            ///     do not broadcast.</exception>
            /// <exception cref="TypeError">Either operand holds text (the multiply has no string loop).</exception>
            /// <exception cref="ArgumentNullException"><paramref name="x1"/> is null.</exception>
            [NDScoped] // what the call leaves (the Linalg member and the woven subclass path scope their own)
            public static NDArray cross(NDArray x1, NDArray x2, int axis = -1)
            {
                if (x1 is null) throw new ArgumentNullException(nameof(x1));
                return x1.TensorEngine.Linalg.Cross(x1, x2, axis);
            }
        }
    }
}
